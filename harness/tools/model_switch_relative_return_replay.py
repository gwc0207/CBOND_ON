"""Research-only no-threshold relative-utility selector replay.

This tool is deliberately isolated from the live dispatcher.  It freezes the
current live50 inputs under ``research_scratch``, predicts two coherent Helmert
relative-return contrasts using only observations strictly before each score
day, and directly chooses the candidate with the highest predicted utility.

There is no BaseGap/Robust/Fusion gate, no confidence threshold, no LCB, and
no Champion preference.  BaseGap is rebuilt only as an aligned comparator.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json5
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge

from cbond_on.infra.live.model_switch import decide_scoreopt_t1430_dispersion
from harness.model_switch_temporal_arbitration import (
    contrasts_to_utilities,
    pairwise_from_utilities,
    utilities_to_contrasts,
)


LIVE_CONFIG_PATH = REPO_ROOT / "cbond_on" / "config" / "live" / "live_config.json5"
STRATEGY_CONFIG_PATH = REPO_ROOT / "cbond_on" / "config" / "strategies" / "strategy01" / "strategy01_config.json5"
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_relative_utility_20260811")

MODELS = ("Regsim", "Ensemble", "HL20")
MODEL_IDS = {
    "Regsim": "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_50_20260805",
    "Ensemble": "ensemble_rankavg_baseline_hl20_labeltop20_50_20260805",
    "HL20": "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625_50_20260805",
}
RETURN_FILENAMES = {
    "Regsim": "Challenger_Regsim.csv",
    "Ensemble": "Challenger_Ensemble.csv",
    "HL20": "Champion_HL20.csv",
}
PAIRWISE = (("Regsim", "Ensemble"), ("Regsim", "HL20"), ("Ensemble", "HL20"))
SCORE_ROOTS = {
    model: Path(r"D:\cbond_on\results\scores\live") / model_id for model, model_id in MODEL_IDS.items()
}

DISP7_FEATURES = (
    "afternoon1300_1430_std",
    "afternoon1300_1430_iqr",
    "afternoon1300_1430_tail_spread",
    "last30_1330_1430_std",
    "last30_1330_1430_iqr",
    "last30_1330_1430_tail_spread",
    "dispersion_accel",
)
SCORE_GEOMETRY_FEATURES = (
    *(f"score_spearman_{left}_{right}" for left, right in PAIRWISE),
    *(f"score_top20_jaccard_{left}_{right}" for left, right in PAIRWISE),
    *(f"score_rank_absdiff_{left}_{right}" for left, right in PAIRWISE),
    "score_top20_union_intersection",
    *(f"score_top20_mean_z_{model}" for model in MODELS),
    *(f"score_top20_gap_z_{model}" for model in MODELS),
    *(f"score_prev_top20_jaccard_{model}" for model in MODELS),
)
LAG_RELATIVE_FEATURES = (
    *(f"lag_relative_ewm5_{model}" for model in MODELS),
    *(f"lag_relative_ewm20_{model}" for model in MODELS),
)

MAX_HISTORY = 360
MIN_PERIODS = 120
RIDGE_ALPHA = 100.0
PAIRWISE_LOGIT_C = 0.1
TOP_K = 20
VARIANTS = {
    "state_only_disp7": ("coherent_ridge", DISP7_FEATURES),
    "state_score_geometry_lags": ("coherent_ridge", (*DISP7_FEATURES, *SCORE_GEOMETRY_FEATURES, *LAG_RELATIVE_FEATURES)),
    "pairwise_ranker_state_score_geometry_lags": ("pairwise_logit", (*DISP7_FEATURES, *SCORE_GEOMETRY_FEATURES, *LAG_RELATIVE_FEATURES)),
}
PLAN = {
    "selector": "coherent_two_contrast_ridge_direct_argmax",
    "no_threshold": True,
    "no_basegap_gate": True,
    "no_robust_gate": True,
    "no_champion_preference": True,
    "tie_rule": "candidate order Regsim, Ensemble, HL20 only on exact numerical equality",
    "max_history_days": MAX_HISTORY,
    "min_training_days": MIN_PERIODS,
    "ridge_alpha": RIDGE_ALPHA,
    "pairwise_logit_c": PAIRWISE_LOGIT_C,
    "variants": {name: {"method": method, "feature_cols": list(features)} for name, (method, features) in VARIANTS.items()},
    "contract": "full-liquidation standalone shadow day_return selector counterfactuals only",
}


@dataclass(frozen=True)
class Snapshot:
    run_root: Path
    input_root: Path
    frozen_switch_config: dict[str, Any]
    return_paths: Mapping[str, Path]
    state_path: Path
    score_root: Path
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class RelativeForecast:
    utilities: np.ndarray
    contrasts: np.ndarray
    pairwise: np.ndarray
    history_days: int
    training_end: date | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _assert_output_root(path: Path) -> Path:
    root = DEFAULT_OUTPUT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"research output must be under {root}: {resolved}") from exc
    return resolved


def _windows_path(value: object) -> Path:
    if isinstance(value, Mapping):
        raw = value.get("windows")
    else:
        raw = value
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"expected a Windows path, got {value!r}")
    return Path(raw)


def _load_json5(path: Path) -> dict[str, Any]:
    value = json5.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping config in {path}")
    return value


def _read_return_panel(paths: Mapping[str, Path]) -> pd.DataFrame:
    panel: pd.DataFrame | None = None
    for model in MODELS:
        header = pd.read_csv(paths[model], nrows=0).columns.tolist()
        metadata_cols = [column for column in ("score_day", "signal_day", "buy_day", "sell_day") if column in header]
        frame = pd.read_csv(paths[model], usecols=["trade_date", "day_return", *metadata_cols])
        metadata_complete = frame[metadata_cols].notna().all(axis=1) if metadata_cols else False
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[model] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame[f"execution_metadata_complete_{model}"] = metadata_complete
        frame = frame[["score_day", model, f"execution_metadata_complete_{model}"]].dropna(subset=["score_day", model]).drop_duplicates("score_day", keep="last")
        if frame.empty:
            raise ValueError(f"empty valid return history for {model}: {paths[model]}")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if panel is None or panel.empty:
        raise ValueError("no aligned three-model return history")
    metadata_flags = [f"execution_metadata_complete_{model}" for model in MODELS]
    if not panel[metadata_flags].nunique(axis=1).eq(1).all():
        raise ValueError("candidate return histories disagree on execution metadata completeness")
    panel["execution_metadata_complete"] = panel[metadata_flags].all(axis=1)
    return panel.sort_values("score_day").reset_index(drop=True)


def _copy_hashed(source: Path, destination: Path, *, kind: str, entries: list[dict[str, object]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"missing required {kind}: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    copy_hash = sha256(destination)
    if source_hash != copy_hash:
        raise RuntimeError(f"copy hash mismatch for {kind}: {source}")
    entries.append(
        {
            "kind": kind,
            "source": str(source),
            "snapshot": str(destination),
            "sha256": source_hash,
            "bytes": int(source.stat().st_size),
        }
    )


def _score_path(root: Path, score_day: object) -> Path:
    timestamp = pd.Timestamp(score_day)
    return root / timestamp.strftime("%Y-%m") / f"{timestamp:%Y-%m-%d}.csv"


def _validate_live_contract(config: Mapping[str, Any], strategy: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path], Path]:
    switch = config.get("model_switch")
    if not isinstance(switch, Mapping):
        raise ValueError("live config has no model_switch mapping")
    if switch.get("mode") != "scoreopt_t1430_dispersion":
        raise ValueError(f"expected direct BaseGap live mode, got {switch.get('mode')!r}")
    if float(switch.get("margin", float("nan"))) != 0.0:
        raise ValueError("this research contract requires current direct BaseGap margin=0.0")
    champion = switch.get("champion")
    challengers = switch.get("challengers")
    if not isinstance(champion, Mapping) or not isinstance(challengers, list) or len(challengers) != 2:
        raise ValueError("expected one champion plus two challengers")
    configured = {
        "Regsim": champion,
        "Ensemble": challengers[0],
        "HL20": challengers[1],
    }
    for model, candidate in configured.items():
        if str(candidate.get("model_id")) != MODEL_IDS[model]:
            raise ValueError(f"unexpected current {model} model id: {candidate.get('model_id')!r}")
    return_paths = {model: _windows_path(candidate.get("return_path")) for model, candidate in configured.items()}
    state_path = _windows_path(switch.get("state_feature_path"))
    turnover = float(strategy.get("turnover_ratio", float("nan")))
    if not math.isclose(turnover, 1.0, abs_tol=0.0):
        raise ValueError(f"selector shadow-return contract needs turnover_ratio=1.0, got {turnover}")
    return dict(switch), return_paths, state_path


def _frozen_switch_config(switch: Mapping[str, Any], *, return_paths: Mapping[str, Path], state_path: Path) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(switch))
    frozen["state_feature_path"] = str(state_path)
    champion = dict(frozen["champion"])
    champion["return_path"] = str(return_paths["Regsim"])
    frozen["champion"] = champion
    challengers = [dict(item) for item in frozen["challengers"]]
    challengers[0]["return_path"] = str(return_paths["Ensemble"])
    challengers[1]["return_path"] = str(return_paths["HL20"])
    frozen["challengers"] = challengers
    return frozen


def capture_input_snapshot(run_root: Path) -> Snapshot:
    """Copy and hash all inputs needed for the frozen research replay."""

    run_root = _assert_output_root(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    input_root = run_root / "input_snapshot"
    entries: list[dict[str, object]] = []
    live_config = _load_json5(LIVE_CONFIG_PATH)
    strategy_config = _load_json5(STRATEGY_CONFIG_PATH)
    switch, source_returns, source_state = _validate_live_contract(live_config, strategy_config)
    source_panel = _read_return_panel(source_returns)

    _copy_hashed(LIVE_CONFIG_PATH, input_root / "configs" / "live_config.json5", kind="live_config", entries=entries)
    _copy_hashed(STRATEGY_CONFIG_PATH, input_root / "configs" / "strategy01_config.json5", kind="strategy_config", entries=entries)
    frozen_returns: dict[str, Path] = {}
    for model in MODELS:
        destination = input_root / "returns" / RETURN_FILENAMES[model]
        _copy_hashed(source_returns[model], destination, kind=f"return_history:{model}", entries=entries)
        frozen_returns[model] = destination
    frozen_state = input_root / "state" / "t1430_market_state_features_path_full.csv"
    _copy_hashed(source_state, frozen_state, kind="t1430_state", entries=entries)

    score_root = input_root / "scores"
    for score_day in source_panel["score_day"].tolist():
        for model in MODELS:
            source = _score_path(SCORE_ROOTS[model], score_day)
            destination = _score_path(score_root / model, score_day)
            _copy_hashed(source, destination, kind=f"score:{model}", entries=entries)

    frozen_switch = _frozen_switch_config(switch, return_paths=frozen_returns, state_path=frozen_state)
    frozen_config_path = input_root / "configs" / "frozen_model_switch_config.json"
    frozen_config_path.write_text(json.dumps(frozen_switch, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    entries.append(
        {
            "kind": "derived_frozen_model_switch_config",
            "source": "derived from input_snapshot/configs/live_config.json5",
            "snapshot": str(frozen_config_path),
            "sha256": sha256(frozen_config_path),
            "bytes": int(frozen_config_path.stat().st_size),
        }
    )
    source_code = [
        REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py",
        REPO_ROOT / "harness" / "model_switch_temporal_arbitration.py",
        Path(__file__).resolve(),
    ]
    manifest: dict[str, Any] = {
        "run_class": "research_only_no_threshold_relative_utility",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "frozen_at_utc": datetime.now(timezone.utc),
        "files": entries,
        "source_code": [{"path": str(path), "sha256": sha256(path)} for path in source_code],
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
        "models": MODEL_IDS,
        "date_coverage": {
            "return_days": int(len(source_panel)),
            "start": str(source_panel["score_day"].min()),
            "end": str(source_panel["score_day"].max()),
        },
        "comparison_contract": {
            "strategy": "strategy01_topk_turnover",
            "turnover_ratio": float(strategy_config["turnover_ratio"]),
            "label": "each candidate's aligned standalone full-cycle day_return",
            "cost_contract": "existing standalone shadow returns; no extra cross-model switching cost",
            "state_provenance_caveat": "legacy T1430 state is not strict-1429 certified",
        },
        "plan": PLAN,
    }
    (input_root / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return Snapshot(
        run_root=run_root,
        input_root=input_root,
        frozen_switch_config=frozen_switch,
        return_paths=frozen_returns,
        state_path=frozen_state,
        score_root=score_root,
        manifest=manifest,
    )


def verify_snapshot(snapshot: Snapshot) -> None:
    """Fail if any captured input no longer matches its recorded hash."""

    for item in snapshot.manifest["files"]:
        path = Path(str(item["snapshot"]))
        if not path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {path}")
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"snapshot hash mismatch: {path}")


def _load_state(state_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(state_path, usecols=["trade_date", *DISP7_FEATURES])
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame = frame[["score_day", *DISP7_FEATURES]].dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    for column in DISP7_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("score_day").reset_index(drop=True)


def _load_score_series(score_root: Path, model: str, score_day: object) -> pd.Series:
    path = _score_path(score_root / model, score_day)
    if not path.is_file():
        raise FileNotFoundError(f"snapshot score missing: {path}")
    frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    reported_days = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    expected = pd.Timestamp(score_day).date()
    if reported_days.isna().any() or not reported_days.eq(expected).all():
        raise ValueError(f"score-day mismatch in {path}: expected {expected}")
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["code", "score"]).loc[lambda value: value["code"].ne("")]
    series = frame.groupby("code", sort=True)["score"].max().astype(float)
    if len(series) < TOP_K + 1:
        raise ValueError(f"insufficient valid snapshot score rows for {model} on {expected}: {len(series)}")
    return series


def _top_codes(series: pd.Series) -> set[str]:
    return set(series.sort_values(ascending=False, kind="stable").head(TOP_K).index.astype(str))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def _top20_standardized_mean_and_gap(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    std = std if math.isfinite(std) and std > 0.0 else 1.0
    ordered = series.sort_values(ascending=False, kind="stable")
    top_mean = float(ordered.head(TOP_K).mean())
    gap = float(ordered.iloc[TOP_K - 1] - ordered.iloc[TOP_K])
    return (top_mean - mean) / std, gap / std


def build_score_disagreement_features(score_root: Path, score_days: Iterable[object]) -> pd.DataFrame:
    """Point-in-time score geometry using only current and prior aligned score files."""

    dates = [pd.Timestamp(day).date() for day in score_days]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate score day requested for score geometry")
    records: list[dict[str, object]] = []
    previous: dict[str, set[str]] | None = None
    for score_day in dates:
        score_by_model = {model: _load_score_series(score_root, model, score_day) for model in MODELS}
        top_by_model = {model: _top_codes(series) for model, series in score_by_model.items()}
        record: dict[str, object] = {"score_day": score_day}
        pair_top_sets: list[set[str]] = []
        for left, right in PAIRWISE:
            paired = pd.concat([score_by_model[left], score_by_model[right]], axis=1, join="inner").dropna()
            if len(paired) < TOP_K:
                raise ValueError(f"insufficient common score universe for {left}/{right} on {score_day}")
            left_score = paired.iloc[:, 0]
            right_score = paired.iloc[:, 1]
            corr = float(stats.spearmanr(left_score, right_score).statistic)
            record[f"score_spearman_{left}_{right}"] = corr if math.isfinite(corr) else 0.0
            rank_left = left_score.rank(method="average", pct=True)
            rank_right = right_score.rank(method="average", pct=True)
            record[f"score_rank_absdiff_{left}_{right}"] = float(np.abs(rank_left - rank_right).mean())
            left_top, right_top = top_by_model[left], top_by_model[right]
            record[f"score_top20_jaccard_{left}_{right}"] = _jaccard(left_top, right_top)
            pair_top_sets.extend([left_top, right_top])
        union = set().union(*top_by_model.values())
        intersection = set.intersection(*top_by_model.values())
        record["score_top20_union_intersection"] = float(len(intersection) / len(union)) if union else float("nan")
        for model in MODELS:
            top_mean_z, top_gap_z = _top20_standardized_mean_and_gap(score_by_model[model])
            record[f"score_top20_mean_z_{model}"] = top_mean_z
            record[f"score_top20_gap_z_{model}"] = top_gap_z
            record[f"score_prev_top20_jaccard_{model}"] = np.nan if previous is None else _jaccard(top_by_model[model], previous[model])
        records.append(record)
        previous = top_by_model
    frame = pd.DataFrame(records)
    if len(frame) > 1 and frame.loc[1:, SCORE_GEOMETRY_FEATURES].isna().any().any():
        bad = frame.loc[1:, frame.loc[1:, SCORE_GEOMETRY_FEATURES].isna().any(axis=1), "score_day"].tolist()
        raise ValueError(f"unexpected score geometry NA after first aligned day: {bad[:5]}")
    return frame


def _add_lag_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()
    returns = result[list(MODELS)].to_numpy(dtype=float)
    relative = returns - returns.mean(axis=1, keepdims=True)
    relative_frame = pd.DataFrame(relative, columns=MODELS)
    for halflife in (5, 20):
        value = relative_frame.ewm(halflife=float(halflife), adjust=False, min_periods=1).mean().shift(1)
        for model in MODELS:
            result[f"lag_relative_ewm{halflife}_{model}"] = value[model].to_numpy(dtype=float)
    return result


def relative_utilities_from_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("returns must be a finite vector with three candidates")
    return values - float(values.mean())


def select_highest_utility(utilities: Sequence[float] | np.ndarray) -> int:
    """Pure argmax; MODELS order settles only exactly equal floating values."""

    values = np.asarray(utilities, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("utilities must be a finite vector with three candidates")
    maximum = float(np.max(values))
    return int(next(index for index, value in enumerate(values) if float(value) == maximum))


def predict_coherent_ridge(
    history_x: np.ndarray,
    history_returns: np.ndarray,
    current_x: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> RelativeForecast:
    """Fit two causal contrast Ridge models with train-only normalization."""

    train_x = np.asarray(history_x, dtype=float)
    train_returns = np.asarray(history_returns, dtype=float)
    candidate_x = np.asarray(current_x, dtype=float).reshape(1, -1)
    if train_x.ndim != 2 or train_returns.shape != (len(train_x), 3) or candidate_x.shape[1] != train_x.shape[1]:
        raise ValueError("invalid Ridge feature/return shapes")
    if not np.isfinite(train_x).all() or not np.isfinite(train_returns).all() or not np.isfinite(candidate_x).all():
        raise ValueError("Ridge inputs must be finite")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=0)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    scaled_train = (train_x - mean) / std
    scaled_current = (candidate_x - mean) / std
    utilities = train_returns - train_returns.mean(axis=1, keepdims=True)
    contrasts = utilities @ np.asarray(
        [[1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)], [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)], [0.0, -2.0 / math.sqrt(6.0)]],
        dtype=float,
    )
    predicted = np.empty(2, dtype=float)
    for index in range(2):
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(scaled_train, contrasts[:, index])
        predicted[index] = float(estimator.predict(scaled_current)[0])
    predicted_utilities = contrasts_to_utilities(predicted)
    return RelativeForecast(
        utilities=predicted_utilities,
        contrasts=predicted,
        pairwise=pairwise_from_utilities(predicted_utilities),
        history_days=int(len(train_x)),
        training_end=None,
    )


def predict_pairwise_logit(
    history_x: np.ndarray,
    history_returns: np.ndarray,
    current_x: np.ndarray,
    *,
    c_value: float = PAIRWISE_LOGIT_C,
) -> RelativeForecast:
    """Fixed-C pairwise win-probability ranker with a direct Copeland argmax."""

    train_x = np.asarray(history_x, dtype=float)
    train_returns = np.asarray(history_returns, dtype=float)
    candidate_x = np.asarray(current_x, dtype=float).reshape(1, -1)
    if train_x.ndim != 2 or train_returns.shape != (len(train_x), 3) or candidate_x.shape[1] != train_x.shape[1]:
        raise ValueError("invalid pairwise ranker feature/return shapes")
    if not np.isfinite(train_x).all() or not np.isfinite(train_returns).all() or not np.isfinite(candidate_x).all():
        raise ValueError("pairwise ranker inputs must be finite")
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=0)
    std = np.where(np.isfinite(std) & (std > 0.0), std, 1.0)
    scaled_train = (train_x - mean) / std
    scaled_current = (candidate_x - mean) / std
    win_scores = np.zeros(3, dtype=float)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        difference = train_returns[:, left] - train_returns[:, right]
        valid = np.abs(difference) > 1e-15
        labels = (difference[valid] > 0.0).astype(int)
        if len(labels) < 2 or len(np.unique(labels)) < 2:
            raise ValueError("pairwise ranker has a degenerate causal label history")
        estimator = LogisticRegression(C=c_value, solver="lbfgs", max_iter=300, random_state=0)
        estimator.fit(scaled_train[valid], labels)
        positive_index = int(np.where(estimator.classes_ == 1)[0][0])
        probability = float(estimator.predict_proba(scaled_current)[0, positive_index])
        win_scores[left] += probability
        win_scores[right] += 1.0 - probability
    utilities = win_scores - float(win_scores.mean())
    return RelativeForecast(
        utilities=utilities,
        contrasts=utilities_to_contrasts(utilities),
        pairwise=pairwise_from_utilities(utilities),
        history_days=int(len(train_x)),
        training_end=None,
    )


def _selector_metrics(returns: pd.Series, dates: pd.Series, selected: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(returns, errors="raise").to_numpy(dtype=float)
    if not len(values):
        return {"days": 0}
    nav = np.cumprod(1.0 + values)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return {
        "days": int(len(values)),
        "start_score_day": str(dates.iloc[0]),
        "end_score_day": str(dates.iloc[-1]),
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": float(nav[-1] ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": float(std * math.sqrt(252.0)),
        "sharpe": float(np.mean(values) / std * math.sqrt(252.0)) if math.isfinite(std) and std > 0.0 else float("nan"),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "win_rate": float((values > 0.0).mean()),
        "switch_count": int((selected.astype(str).to_numpy()[1:] != selected.astype(str).to_numpy()[:-1]).sum()) if len(values) > 1 else 0,
    }


def _paired_metrics(delta: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    nonzero = values[np.abs(values) > 1e-15]
    wins = int((nonzero > 0.0).sum())
    result: dict[str, object] = {
        "paired_days": int(len(values)),
        "nonzero_paired_days": int(len(nonzero)),
        "wins": wins,
        "losses": int((nonzero < 0.0).sum()),
        "mean_delta_bp": float(values.mean() * 1e4) if len(values) else float("nan"),
        "sum_delta_bp": float(values.sum() * 1e4) if len(values) else float("nan"),
    }
    if len(values) > 1:
        result["paired_t_p_two_sided"] = float(stats.ttest_1samp(values, 0.0).pvalue)
    if len(nonzero) > 0:
        result["sign_p_two_sided"] = float(stats.binomtest(wins, len(nonzero), 0.5, alternative="two-sided").pvalue)
    return result


def reproduce_direct_basegap(snapshot: Snapshot, score_days: Iterable[object]) -> pd.DataFrame:
    """Run the production direct BaseGap function against frozen input paths."""

    rows: list[dict[str, object]] = []
    for raw_day in score_days:
        score_day = pd.Timestamp(raw_day).date()
        decision = decide_scoreopt_t1430_dispersion(snapshot.frozen_switch_config, score_day=score_day)
        score_by_model = {MODEL_IDS["Regsim"]: "Regsim", MODEL_IDS["Ensemble"]: "Ensemble", MODEL_IDS["HL20"]: "HL20"}
        candidate_scores = {score_by_model[item["model_id"]]: item.get("score") for item in decision.candidate_scores}
        rows.append(
            {
                "score_day": score_day,
                "basegap_selected_name": score_by_model[decision.selected_model_id],
                "basegap_reason": decision.reason,
                "basegap_history_days": decision.history_days,
                "basegap_history_end": decision.history_end,
                "basegap_score_gap": decision.score_diff,
                **{f"basegap_score_{model}": candidate_scores.get(model) for model in MODELS},
            }
        )
    return pd.DataFrame(rows)


def _prepare_panel(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = _read_return_panel(snapshot.return_paths)
    state = _load_state(snapshot.state_path)
    geometry = build_score_disagreement_features(snapshot.score_root, returns["score_day"].tolist())
    panel = returns.merge(state, on="score_day", how="left", validate="one_to_one")
    panel = panel.merge(geometry, on="score_day", how="left", validate="one_to_one")
    panel = _add_lag_relative_features(panel)
    basegap = reproduce_direct_basegap(snapshot, panel["score_day"].tolist())
    panel = panel.merge(basegap, on="score_day", how="left", validate="one_to_one")
    return panel.sort_values("score_day").reset_index(drop=True), geometry


def run_selector(
    panel: pd.DataFrame,
    *,
    variant: str,
    feature_cols: Sequence[str],
    method: str = "coherent_ridge",
) -> pd.DataFrame:
    """Strict daily walk-forward direct argmax selector for one fixed variant."""

    rows: list[dict[str, object]] = []
    features = list(feature_cols)
    for position, current in panel.iterrows():
        score_day = current["score_day"]
        history = panel.loc[panel["score_day"] < score_day, ["score_day", *MODELS, *features]].tail(MAX_HISTORY).copy()
        history_complete = np.isfinite(history[[*MODELS, *features]].to_numpy(dtype=float)).all(axis=1)
        history = history.loc[history_complete].copy()
        current_features = current[features].to_numpy(dtype=float)
        ready = bool(np.isfinite(current_features).all() and len(history) >= MIN_PERIODS)
        record: dict[str, object] = {
            "score_day": score_day,
            "variant": variant,
            "feature_count": len(features),
            "training_rows": int(len(history)),
            "training_end": history["score_day"].max() if len(history) else None,
            "prediction_ready": ready,
            "forecast_method": method,
            "selection_reason": "relative_utility_argmax" if ready else "warmup_or_input_unavailable_fixed_regsim",
            "selected_name": "Regsim",
            "selected_model_id": MODEL_IDS["Regsim"],
        }
        for model in MODELS:
            record[f"realized_return_{model}"] = float(current[model])
        if ready:
            try:
                if method == "coherent_ridge":
                    forecast = predict_coherent_ridge(
                        history[features].to_numpy(dtype=float),
                        history[list(MODELS)].to_numpy(dtype=float),
                        current_features,
                    )
                elif method == "pairwise_logit":
                    forecast = predict_pairwise_logit(
                        history[features].to_numpy(dtype=float),
                        history[list(MODELS)].to_numpy(dtype=float),
                        current_features,
                    )
                else:
                    raise ValueError(f"unsupported forecast method: {method}")
            except ValueError:
                ready = False
                record["prediction_ready"] = False
                record["selection_reason"] = "degenerate_causal_label_history_fixed_regsim"
        if ready:
            selected_index = select_highest_utility(forecast.utilities)
            selected_name = MODELS[selected_index]
            record.update(
                {
                    "selected_name": selected_name,
                    "selected_model_id": MODEL_IDS[selected_name],
                    "utility_regsim": float(forecast.utilities[0]),
                    "utility_ensemble": float(forecast.utilities[1]),
                    "utility_hl20": float(forecast.utilities[2]),
                    "contrast_1": float(forecast.contrasts[0]),
                    "contrast_2": float(forecast.contrasts[1]),
                    "pair_regsim_ensemble": float(forecast.pairwise[0, 1]),
                    "pair_regsim_hl20": float(forecast.pairwise[0, 2]),
                    "pair_ensemble_hl20": float(forecast.pairwise[1, 2]),
                }
            )
        else:
            record.update({
                "utility_regsim": np.nan,
                "utility_ensemble": np.nan,
                "utility_hl20": np.nan,
                "contrast_1": np.nan,
                "contrast_2": np.nan,
                "pair_regsim_ensemble": np.nan,
                "pair_regsim_hl20": np.nan,
                "pair_ensemble_hl20": np.nan,
            })
        record["selected_return"] = float(current[record["selected_name"]])
        record["regsim_return"] = float(current["Regsim"])
        record["basegap_selected_name"] = current["basegap_selected_name"]
        record["basegap_return"] = float(current[str(current["basegap_selected_name"])])
        record["basegap_history_days"] = current.get("basegap_history_days", np.nan)
        record["basegap_reason"] = current.get("basegap_reason", None)
        record["execution_metadata_complete"] = bool(current.get("execution_metadata_complete", False))
        rows.append(record)
    return pd.DataFrame(rows)


def _slice_metrics(frame: pd.DataFrame, *, strategy: str, return_col: str, selected_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, indices in zip(("early", "validation", "holdout"), np.array_split(np.arange(len(frame)), 3), strict=True):
        subset = frame.iloc[indices]
        rows.append({"strategy": strategy, "slice": name, **_selector_metrics(subset[return_col], subset["score_day"], subset[selected_col])})
    return rows


def _metadata_sensitivity(reference: pd.DataFrame, variants: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Report the known execution-metadata-incomplete suffix separately."""

    complete = reference["execution_metadata_complete"].astype(bool)
    rows: list[dict[str, object]] = []
    for scope, mask in (("all_return_rows", pd.Series(True, index=reference.index)), ("complete_execution_metadata_only", complete)):
        excluded = int((~mask).sum())
        base = reference.loc[mask]
        rows.append(
            {
                "scope": scope,
                "strategy": "Regsim",
                "excluded_metadata_days": excluded,
                **_selector_metrics(base["regsim_return"], base["score_day"], pd.Series("Regsim", index=base.index)),
            }
        )
        rows.append(
            {
                "scope": scope,
                "strategy": "direct_basegap",
                "excluded_metadata_days": excluded,
                **_selector_metrics(base["basegap_return"], base["score_day"], base["basegap_selected_name"]),
            }
        )
        for variant, frame in variants.items():
            subset = frame.loc[mask]
            rows.append(
                {
                    "scope": scope,
                    "strategy": variant,
                    "excluded_metadata_days": excluded,
                    **_selector_metrics(subset["selected_return"], subset["score_day"], subset["selected_name"]),
                }
            )
    return pd.DataFrame(rows)


def _prediction_diagnostics(variants: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pairs = (
        ("Regsim", "Ensemble", "pair_regsim_ensemble"),
        ("Regsim", "HL20", "pair_regsim_hl20"),
        ("Ensemble", "HL20", "pair_ensemble_hl20"),
    )
    for variant, frame in variants.items():
        ready = frame.loc[frame["prediction_ready"].astype(bool)].copy()
        method = str(ready["forecast_method"].iloc[0]) if len(ready) else "unknown"
        for left, right, prediction_column in pairs:
            predicted = pd.to_numeric(ready[prediction_column], errors="coerce")
            actual = pd.to_numeric(ready[f"realized_return_{left}"], errors="coerce") - pd.to_numeric(ready[f"realized_return_{right}"], errors="coerce")
            valid = predicted.notna() & actual.notna()
            y_pred = predicted.loc[valid].to_numpy(dtype=float)
            y_true = actual.loc[valid].to_numpy(dtype=float)
            nonzero = np.abs(y_true) > 1e-15
            rows.append(
                {
                    "variant": variant,
                    "diagnostic": "pairwise_relative_return" if method == "coherent_ridge" else "pairwise_win_probability_margin",
                    "pair": f"{left}_minus_{right}",
                    "days": int(len(y_true)),
                    "correlation": float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan"),
                    "oos_r2_zero": float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum(y_true**2)) if method == "coherent_ridge" and np.sum(y_true**2) > 0.0 else float("nan"),
                    "direction_hit_rate": float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))) if np.any(nonzero) else float("nan"),
                }
            )
        if len(ready):
            actual_winner = [MODELS[select_highest_utility(row)] for row in ready[[f"realized_return_{model}" for model in MODELS]].to_numpy(dtype=float)]
            rows.append(
                {
                    "variant": variant,
                    "diagnostic": "realized_oracle_winner_hit",
                    "pair": None,
                    "days": int(len(ready)),
                    "correlation": float("nan"),
                    "oos_r2_zero": float("nan"),
                    "direction_hit_rate": float(np.mean(ready["selected_name"].to_numpy(dtype=str) == np.asarray(actual_winner, dtype=str))),
                }
            )
    return pd.DataFrame(rows)


def _write_nav_plot(run_root: Path, reference: pd.DataFrame, variants: Mapping[str, pd.DataFrame]) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(reference["score_day"], np.cumprod(1.0 + reference["regsim_return"]), label="Regsim", color="black", linewidth=2.0)
    ax.plot(reference["score_day"], np.cumprod(1.0 + reference["basegap_return"]), label="direct BaseGap", color="#777777", linewidth=1.3)
    for variant, frame in variants.items():
        ax.plot(frame["score_day"], np.cumprod(1.0 + frame["selected_return"]), label=variant, linewidth=1.25)
    ax.set_title("Research-only no-threshold relative-utility selector")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_root / "nav_compare.png", dpi=160)
    plt.close(fig)


def _write_results(
    run_root: Path,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    slices: pd.DataFrame,
    sensitivity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    def block(frame: pd.DataFrame) -> str:
        return "```csv\n" + frame.to_csv(index=False, float_format="%.6f").rstrip() + "\n```"

    text = [
        "# No-threshold relative-utility model selector",
        "",
        "## Contract",
        "",
        "- Research-only frozen replay. No live config, scheduler, DB, model state, trade list, `results/live`, or `results/analysis` was written.",
        "- Every ready day directly selects the highest coherent relative utility. It does not consume BaseGap, Robust, Fusion, LCB, Champion role, or a top-one/top-two threshold.",
        "- The direct BaseGap series is an aligned comparator rebuilt with the production function against frozen return/state paths.",
        "- Warmup/input-unavailable dates use disclosed fixed Regsim availability routing; they are reported separately and are not an alpha-confidence gate.",
        "- Legacy state provenance remains 14:30 / not-strict-1429-certified. This replay preserves the caveat and cannot establish live readiness.",
        "",
        "## Summary",
        "",
        block(summary),
        "",
        "## Paired daily delta versus Regsim",
        "",
        block(paired),
        "",
        "## Chronological slices",
        "",
        block(slices),
        "",
        "## Execution-metadata suffix sensitivity",
        "",
        block(sensitivity),
        "",
        "## Prediction diagnostics",
        "",
        block(diagnostics),
        "",
        "## Artifacts",
        "",
        "- `input_snapshot/input_manifest.json`: copied input hashes and research contract.",
        "- `daily_score_disagreement.csv`, `daily_direct_basegap.csv`, `daily_selector_replay.csv`.",
        "- `summary_metrics.csv`, `paired_vs_regsim_basegap.csv`, `summary_slices.csv`, `metadata_suffix_sensitivity.csv`, `prediction_diagnostics.csv`, `nav_compare.png`.",
        "",
        "## Caveat",
        "",
        "This is a selector shadow-return composition under the existing full-liquidation contract, not realised live NAV and not evidence for promotion without an untouched chronological/prospective confirmation.",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run_replay(snapshot: Snapshot) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    verify_snapshot(snapshot)
    panel, geometry = _prepare_panel(snapshot)
    geometry.to_csv(snapshot.run_root / "daily_score_disagreement.csv", index=False)
    panel[["score_day", "basegap_selected_name", "basegap_reason", "basegap_history_days", "basegap_history_end", "basegap_score_gap", *[f"basegap_score_{model}" for model in MODELS]]].to_csv(snapshot.run_root / "daily_direct_basegap.csv", index=False)
    variants = {
        name: run_selector(panel, variant=name, feature_cols=features, method=method)
        for name, (method, features) in VARIANTS.items()
    }
    all_daily = pd.concat(variants.values(), ignore_index=True)
    all_daily.to_csv(snapshot.run_root / "daily_selector_replay.csv", index=False)

    reference = next(iter(variants.values()))
    summary_rows = [
        {"strategy": "Regsim", **_selector_metrics(reference["regsim_return"], reference["score_day"], pd.Series("Regsim", index=reference.index)), "prediction_ready_days": int(reference["prediction_ready"].sum()), "availability_fallback_days": 0},
        {"strategy": "direct_basegap", **_selector_metrics(reference["basegap_return"], reference["score_day"], reference["basegap_selected_name"]), "prediction_ready_days": int(reference["basegap_history_days"].ge(40).sum()), "availability_fallback_days": int(reference["basegap_history_days"].lt(40).sum())},
    ]
    paired_rows = [
        {"strategy": "direct_basegap", "baseline": "Regsim", **_paired_metrics(reference["basegap_return"] - reference["regsim_return"])},
    ]
    slice_rows = [
        *_slice_metrics(reference, strategy="Regsim", return_col="regsim_return", selected_col="selected_name"),
        *_slice_metrics(reference, strategy="direct_basegap", return_col="basegap_return", selected_col="basegap_selected_name"),
    ]
    for variant, frame in variants.items():
        summary_rows.append(
            {
                "strategy": variant,
                **_selector_metrics(frame["selected_return"], frame["score_day"], frame["selected_name"]),
                "prediction_ready_days": int(frame["prediction_ready"].sum()),
                "availability_fallback_days": int((~frame["prediction_ready"]).sum()),
            }
        )
        paired_rows.append({"strategy": variant, "baseline": "Regsim", **_paired_metrics(frame["selected_return"] - frame["regsim_return"])})
        paired_rows.append({"strategy": variant, "baseline": "direct_basegap", **_paired_metrics(frame["selected_return"] - frame["basegap_return"])})
        slice_rows.extend(_slice_metrics(frame, strategy=variant, return_col="selected_return", selected_col="selected_name"))
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    slices = pd.DataFrame(slice_rows)
    sensitivity = _metadata_sensitivity(reference, variants)
    diagnostics = _prediction_diagnostics(variants)
    summary.to_csv(snapshot.run_root / "summary_metrics.csv", index=False)
    paired.to_csv(snapshot.run_root / "paired_vs_regsim_basegap.csv", index=False)
    slices.to_csv(snapshot.run_root / "summary_slices.csv", index=False)
    sensitivity.to_csv(snapshot.run_root / "metadata_suffix_sensitivity.csv", index=False)
    diagnostics.to_csv(snapshot.run_root / "prediction_diagnostics.csv", index=False)
    run_manifest = dict(snapshot.manifest)
    run_manifest["analysis"] = {
        "metadata_incomplete_return_days": int((~panel["execution_metadata_complete"].astype(bool)).sum()),
        "metadata_incomplete_dates": [str(day) for day in panel.loc[~panel["execution_metadata_complete"].astype(bool), "score_day"].tolist()],
    }
    (snapshot.run_root / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_nav_plot(snapshot.run_root, reference, variants)
    _write_results(snapshot.run_root, summary, paired, slices, sensitivity, diagnostics, run_manifest)
    return all_daily, summary, paired, slices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="new leaf under the isolated research root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _assert_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_safe_git(['rev-parse', '--short', 'HEAD']) or 'nogit'}"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must be a single directory leaf")
    snapshot = capture_input_snapshot(output_root / run_name)
    _, summary, paired, _ = run_replay(snapshot)
    print(f"OUTPUT_ROOT {snapshot.run_root}")
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("PAIRED")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
