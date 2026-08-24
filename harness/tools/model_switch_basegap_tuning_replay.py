"""Frozen, research-only replay for pure three-candidate BaseGap tuning.

This module deliberately does *not* call the live runtime.  It reads the
already frozen 50-factor three-candidate snapshot created by the relative
utility research run, verifies every frozen input hash, reproduces the
production BaseGap decision day by day, and then evaluates a preregistered
small grid entirely in memory.

It is intentionally limited to the direct, symmetric three-candidate
``scoreopt_t1430_dispersion`` selector:

* no Champion preference after the production-baseline parity gate;
* no Robust/Fusion/threshold/gate logic;
* no factor, model, strategy, mask, cost, DB, scheduler, or live-output write;
* every selector decision uses only state and candidate returns dated strictly
  before its score day.

The result is a selector-shadow-return research replay.  It is not a new
single-book execution backtest and is never a promotion mechanism.
"""

from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cbond_on.infra.live.model_switch import (
    T1430_DISPERSION_FEATURE_SETS,
    _scoreopt_score_sample,
    decide_scoreopt_t1430_dispersion,
)


DEFAULT_SOURCE_RUN = Path(
    "D:/cbond_on/research_scratch/model_switch_relative_utility_20260811/"
    "run_20260811_relative_utility_v5_with_ranker"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_basegap_tuning_20260811")

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

# This grid was fixed before running it.  The current production point is
# (60, 40, trim20_lcb10).  ``nearest_k`` is the final Top-K after the lookback
# candidate pool; it is not the candidate-pool length.
LOOKBACK_GRID = (40, 60, 90, 120, 180, 240)
NEAREST_K_GRID = (20, 40, 60)
METRIC_GRID = ("mean", "lcb10", "trim20_lcb10")
BASELINE_POINT = {"lookback_days": 60, "nearest_k": 40, "metric": "trim20_lcb10"}

SPLITS: tuple[tuple[str, str, str], ...] = (
    ("design", "2024-05-08", "2025-06-30"),
    ("validation", "2025-07-01", "2025-12-31"),
    ("final_oos", "2026-01-01", "2026-07-30"),
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file at once."""

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


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, description: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be under {root}: {resolved}") from exc
    return resolved


def _assert_source_run(source_run: Path) -> Path:
    """Reject any source other than the specified frozen research snapshot."""

    resolved = source_run.resolve()
    expected = DEFAULT_SOURCE_RUN.resolve()
    if resolved != expected:
        raise ValueError(f"source run is locked to the verified frozen snapshot: {expected}")
    if not (resolved / "input_snapshot").is_dir() or not (resolved / "run_manifest.json").is_file():
        raise FileNotFoundError(f"frozen source run is incomplete: {resolved}")
    return resolved


def _assert_output_root(path: Path) -> Path:
    return _resolve_inside(path, DEFAULT_OUTPUT_ROOT, description="research output")


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _assert_output_root(output_root)
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _assert_output_root(root / name)
    if run_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("a new run leaf below the research output root is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing research run: {run_root}")
    return run_root


def _path_from_value(value: object) -> Path:
    if isinstance(value, Mapping):
        value = value.get("windows")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"expected a non-empty local path, got {value!r}")
    return Path(value)


def _load_frozen_config(input_root: Path) -> dict[str, Any]:
    path = input_root / "configs" / "frozen_model_switch_config.json"
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"frozen switch config must be a mapping: {path}")
    return value


def _candidate_configurations(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    champion = config.get("champion")
    challengers = config.get("challengers")
    if not isinstance(champion, Mapping) or not isinstance(challengers, list) or len(challengers) != 2:
        raise ValueError("expected one champion and exactly two challengers in frozen config")
    if not all(isinstance(item, Mapping) for item in challengers):
        raise ValueError("frozen challenger definitions must be mappings")
    configured = {
        "Regsim": dict(champion),
        "Ensemble": dict(challengers[0]),
        "HL20": dict(challengers[1]),
    }
    for model, candidate in configured.items():
        if str(candidate.get("model_id", "")).strip() != MODEL_IDS[model]:
            raise ValueError(f"unexpected {model} model id in frozen config: {candidate.get('model_id')!r}")
    return configured


def validate_frozen_source(source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash-check every recorded frozen file and validate the read boundary."""

    source_run = _assert_source_run(source_run)
    input_root = source_run / "input_snapshot"
    manifest_path = source_run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"source manifest has no frozen file entries: {manifest_path}")

    verified: list[dict[str, object]] = []
    for index, item in enumerate(entries, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"invalid source manifest entry {index}")
        snapshot = _resolve_inside(Path(str(item.get("snapshot", ""))), input_root, description="frozen input")
        expected_hash = str(item.get("sha256", ""))
        if not snapshot.is_file() or len(expected_hash) != 64:
            raise FileNotFoundError(f"invalid frozen manifest entry: {snapshot}")
        actual_hash = sha256(snapshot)
        if actual_hash != expected_hash:
            raise RuntimeError(f"frozen input SHA-256 mismatch: {snapshot}")
        verified.append(
            {
                "kind": str(item.get("kind", "")),
                "snapshot": str(snapshot),
                "sha256": actual_hash,
                "bytes": int(snapshot.stat().st_size),
            }
        )

    config = _load_frozen_config(input_root)
    if config.get("mode") != "scoreopt_t1430_dispersion":
        raise ValueError(f"expected direct BaseGap mode, got {config.get('mode')!r}")
    if str(config.get("feature_set", "")).strip() != "path_full_t1430":
        raise ValueError("this research is locked to the frozen path_full_t1430 state")
    if float(config.get("margin", float("nan"))) != 0.0:
        raise ValueError("this research requires direct BaseGap margin=0.0")
    if str(config.get("metric", "")).strip().lower() != "trim20_lcb10":
        raise ValueError("frozen source is not the expected trim20_lcb10 production baseline")

    paths_to_check = [_path_from_value(config.get("state_feature_path"))]
    for candidate in _candidate_configurations(config).values():
        paths_to_check.append(_path_from_value(candidate.get("return_path")))
    for path in paths_to_check:
        checked = _resolve_inside(path, input_root, description="frozen selector input")
        if not checked.is_file():
            raise FileNotFoundError(f"frozen selector input missing: {checked}")

    verification = {
        "source_run": str(source_run),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "verified_file_count": len(verified),
        "verified_total_bytes": int(sum(int(item["bytes"]) for item in verified)),
        "files": verified,
        "read_boundary": str(input_root.resolve()),
    }
    return config, verification


def _return_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        model: _path_from_value(candidate.get("return_path"))
        for model, candidate in _candidate_configurations(config).items()
    }


def read_return_panel(return_paths: Mapping[str, Path]) -> pd.DataFrame:
    """Load aligned standalone candidate returns and metadata completeness."""

    panel: pd.DataFrame | None = None
    for model in MODELS:
        path = return_paths[model]
        header = pd.read_csv(path, nrows=0).columns.tolist()
        metadata_cols = [column for column in ("score_day", "signal_day", "buy_day", "sell_day") if column in header]
        frame = pd.read_csv(path, usecols=["trade_date", "day_return", *metadata_cols])
        complete = frame[metadata_cols].notna().all(axis=1) if metadata_cols else pd.Series(False, index=frame.index)
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[model] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame[f"execution_metadata_complete_{model}"] = complete.astype(bool)
        frame = frame[["score_day", model, f"execution_metadata_complete_{model}"]]
        frame = frame.dropna(subset=["score_day", model]).drop_duplicates("score_day", keep="last")
        if frame.empty:
            raise ValueError(f"no usable return history for {model}: {path}")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if panel is None or panel.empty:
        raise ValueError("no aligned candidate return history")
    metadata_flags = [f"execution_metadata_complete_{model}" for model in MODELS]
    if not panel[metadata_flags].nunique(axis=1).eq(1).all():
        raise ValueError("candidate return histories disagree on execution metadata completeness")
    panel["execution_metadata_complete"] = panel[metadata_flags].all(axis=1)
    return panel.sort_values("score_day").reset_index(drop=True)


def read_state_panel(state_path: Path, feature_cols: Sequence[str]) -> pd.DataFrame:
    required = ["trade_date", *feature_cols]
    frame = pd.read_csv(state_path, usecols=required)
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["state_available"] = True
    frame = frame[["score_day", "state_available", *feature_cols]].dropna(subset=["score_day"])
    return frame.drop_duplicates("score_day", keep="last").sort_values("score_day").reset_index(drop=True)


def prepare_panel(config: Mapping[str, Any]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    feature_set = str(config["feature_set"]).strip().lower()
    feature_cols = tuple(T1430_DISPERSION_FEATURE_SETS[feature_set])
    returns = read_return_panel(_return_paths(config))
    state = read_state_panel(_path_from_value(config["state_feature_path"]), feature_cols)
    panel = returns.merge(state, on="score_day", how="left", validate="one_to_one")
    return panel.sort_values("score_day").reset_index(drop=True), feature_cols


def _canonical_model_from_id(model_id: str) -> str:
    for name, expected in MODEL_IDS.items():
        if model_id == expected:
            return name
    raise KeyError(f"unrecognised frozen candidate id: {model_id}")


def reproduce_production_basegap(config: Mapping[str, Any], score_days: Iterable[object]) -> pd.DataFrame:
    """Call the production function on frozen paths for the identity gate."""

    rows: list[dict[str, object]] = []
    frozen_config = copy.deepcopy(dict(config))
    for raw_day in score_days:
        score_day = pd.Timestamp(raw_day).date()
        decision = decide_scoreopt_t1430_dispersion(frozen_config, score_day=score_day)
        score_by_model = {
            _canonical_model_from_id(str(item["model_id"])): item.get("score")
            for item in (decision.candidate_scores or [])
        }
        rows.append(
            {
                "score_day": score_day,
                "basegap_selected_name": _canonical_model_from_id(decision.selected_model_id),
                "basegap_reason": decision.reason,
                "basegap_history_days": int(decision.history_days),
                "basegap_history_end": decision.history_end,
                "basegap_score_gap": decision.score_diff,
                **{f"basegap_score_{model}": score_by_model.get(model) for model in MODELS},
            }
        )
    return pd.DataFrame(rows)


def _normalise_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce").dt.date


def _float_mismatch_mask(left: pd.Series, right: pd.Series, *, atol: float = 1e-14) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
    equal = np.isclose(left_values, right_values, rtol=0.0, atol=atol, equal_nan=True)
    return pd.Series(~equal, index=left.index)


def parity_audit(actual: pd.DataFrame, expected: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return field-level parity diagnostics and fail closed on any mismatch."""

    fields = (
        "basegap_selected_name",
        "basegap_reason",
        "basegap_history_days",
        "basegap_history_end",
        "basegap_score_gap",
        *(f"basegap_score_{model}" for model in MODELS),
    )
    left = actual.copy()
    right = expected.copy()
    left["score_day"] = _normalise_date_column(left, "score_day")
    right["score_day"] = _normalise_date_column(right, "score_day")
    left = left.sort_values("score_day").reset_index(drop=True)
    right = right.sort_values("score_day").reset_index(drop=True)
    if left["score_day"].tolist() != right["score_day"].tolist():
        raise RuntimeError(f"{label} has a score-day coverage mismatch")
    rows: list[dict[str, object]] = []
    for field in fields:
        if field not in left.columns or field not in right.columns:
            raise KeyError(f"{label} missing parity field {field}")
        if field in {"basegap_score_gap", *(f"basegap_score_{model}" for model in MODELS)}:
            mismatch = _float_mismatch_mask(left[field], right[field])
            abs_delta = (
                pd.to_numeric(left[field], errors="coerce") - pd.to_numeric(right[field], errors="coerce")
            ).abs()
            max_abs_delta = float(abs_delta.max(skipna=True)) if abs_delta.notna().any() else 0.0
        elif field == "basegap_history_end":
            left_dates = _normalise_date_column(left, field)
            right_dates = _normalise_date_column(right, field)
            mismatch = ~(left_dates.eq(right_dates) | (left_dates.isna() & right_dates.isna()))
            max_abs_delta = None
        else:
            mismatch = left[field].astype(str).ne(right[field].astype(str))
            max_abs_delta = None
        rows.append({"comparison": label, "field": field, "mismatch_rows": int(mismatch.sum()), "max_abs_delta": max_abs_delta})
    audit = pd.DataFrame(rows)
    failures = audit.loc[audit["mismatch_rows"] > 0]
    if not failures.empty:
        fields_text = ", ".join(failures["field"].tolist())
        raise RuntimeError(f"{label} parity failed in fields: {fields_text}")
    return audit


def _variant_name(*, lookback_days: int, nearest_k: int, metric: str) -> str:
    return f"lb{lookback_days}_k{nearest_k}_{metric}"


def preregistered_grid() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for lookback_days in LOOKBACK_GRID:
        for nearest_k in NEAREST_K_GRID:
            if nearest_k > lookback_days:
                continue
            for metric in METRIC_GRID:
                rows.append(
                    {
                        "variant": _variant_name(lookback_days=lookback_days, nearest_k=nearest_k, metric=metric),
                        "lookback_days": lookback_days,
                        "nearest_k": nearest_k,
                        "min_periods": nearest_k,
                        "metric": metric,
                        "distance_standardization": "rolling_window_zscore_ddof1",
                        "neighbor_weighting": "equal",
                        "objective": "candidate_absolute_day_return_statistic",
                        "margin": 0.0,
                        "is_production_baseline": bool(
                            lookback_days == BASELINE_POINT["lookback_days"]
                            and nearest_k == BASELINE_POINT["nearest_k"]
                            and metric == BASELINE_POINT["metric"]
                        ),
                    }
                )
    if len(rows) != 51:
        raise AssertionError(f"unexpected preregistered grid size: {len(rows)}")
    return rows


def run_basegap_variant(
    panel: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    lookback_days: int,
    nearest_k: int,
    min_periods: int,
    metric: str,
    variant: str,
) -> pd.DataFrame:
    """Pure in-memory BaseGap replay with a strict predecessor-only history."""

    if lookback_days <= 0 or nearest_k <= 0 or min_periods <= 0:
        raise ValueError("lookback_days, nearest_k, and min_periods must be positive")
    if nearest_k > lookback_days:
        raise ValueError("nearest_k cannot exceed lookback_days")
    if metric not in METRIC_GRID:
        raise ValueError(f"unsupported metric: {metric}")

    features = list(feature_cols)
    required = ["score_day", *MODELS, *features]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required BaseGap columns: {missing}")
    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    # ``state_available`` is distinct from a present state row with one or
    # more missing feature cells.  Production first inner-joins state rows,
    # then tails the lookback window, and only then filters feature-NA rows.
    # Keeping that distinction is required for byte-for-byte baseline parity
    # around days missing entirely from the state history.
    if "state_available" not in ordered.columns:
        ordered["state_available"] = True
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []

    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        returns_history = ordered.iloc[:position][["score_day", *MODELS]].copy()
        history_end = returns_history["score_day"].max() if not returns_history.empty else None
        record: dict[str, object] = {
            "score_day": score_day,
            "variant": variant,
            "lookback_days": int(lookback_days),
            "nearest_k": int(nearest_k),
            "min_periods": int(min_periods),
            "metric": metric,
            "basegap_history_end": history_end,
            "basegap_history_days": 0,
            "basegap_reason": None,
            "basegap_selected_name": "Regsim",
            "basegap_score_gap": float("nan"),
            "neighbor_min_day": None,
            "neighbor_max_day": None,
            "neighbor_count": 0,
        }
        for model in MODELS:
            record[f"basegap_score_{model}"] = float("nan")

        current_features = current[features].to_numpy(dtype=float)
        if not bool(current["state_available"]):
            record["basegap_reason"] = "feature_missing"
        elif not np.isfinite(current_features).all():
            record["basegap_reason"] = "feature_na"
        else:
            joined = ordered.iloc[:position][["score_day", "state_available", *MODELS, *features]].copy()
            joined = joined.loc[joined["state_available"]].tail(lookback_days).copy()
            history_features = joined[features]
            history_returns = joined[list(MODELS)]
            valid_mask = ~(history_features.isna().any(axis=1) | history_returns.isna().any(axis=1))
            history_features = history_features.loc[valid_mask]
            history_returns = history_returns.loc[valid_mask]
            observations = int(len(history_features))
            record["basegap_history_days"] = observations
            if observations < nearest_k or observations < min_periods:
                record["basegap_reason"] = "insufficient_history"
            else:
                mean = history_features.mean(axis=0)
                std = history_features.std(axis=0).replace(0, math.nan).fillna(1.0)
                distances = (((history_features - mean) / std - (current[features] - mean) / std) ** 2).sum(axis=1).pow(0.5)
                nearest_index = distances.sort_values().index[:nearest_k]
                sample = history_returns.loc[nearest_index]
                scores = _scoreopt_score_sample(sample, model_cols=list(MODELS), score_mode=metric).sort_values(ascending=False)
                selected = str(scores.index[0])
                best_score = float(scores.iloc[0])
                second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
                score_gap = best_score - second_score if math.isfinite(second_score) else float("nan")
                nearest_days = pd.to_datetime(joined.loc[nearest_index, "score_day"], errors="coerce").dt.date
                record.update(
                    {
                        "basegap_selected_name": selected,
                        "basegap_reason": "score_best" if score_gap > 0.0 else "margin_default",
                        "basegap_score_gap": score_gap,
                        "neighbor_min_day": nearest_days.min(),
                        "neighbor_max_day": nearest_days.max(),
                        "neighbor_count": int(len(sample)),
                        "basegap_history_days": int(len(sample)),
                    }
                )
                for model in MODELS:
                    record[f"basegap_score_{model}"] = float(scores[model])
        selected_name = str(record["basegap_selected_name"])
        realised = {model: float(current[model]) for model in MODELS}
        selected_return = realised[selected_name]
        best_name = max(MODELS, key=lambda model: realised[model])
        rank = 1 + sum(realised[model] > selected_return for model in MODELS)
        record.update(
            {
                **{f"realized_return_{model}": realised[model] for model in MODELS},
                "selected_return": selected_return,
                "equal_weight_return": float(np.mean(list(realised.values()))),
                "best_name": best_name,
                "best_return": realised[best_name],
                "selection_alpha": selected_return - float(np.mean(list(realised.values()))),
                "regret": realised[best_name] - selected_return,
                "selected_rank": int(rank),
                "selection_active": bool(record["basegap_reason"] in {"score_best", "margin_default"}),
                "execution_metadata_complete": bool(current.get("execution_metadata_complete", False)),
            }
        )
        rows.append(record)

    result = pd.DataFrame(rows)
    ready = result.loc[result["selection_active"]]
    if not ready.empty and not (pd.to_datetime(ready["neighbor_max_day"]).dt.date < pd.to_datetime(ready["score_day"]).dt.date).all():
        raise RuntimeError("BaseGap replay leakage: a neighbour is not strictly before score_day")
    return result


def add_variant_metadata(daily: pd.DataFrame, variant_row: Mapping[str, object]) -> pd.DataFrame:
    result = daily.copy()
    for key in ("distance_standardization", "neighbor_weighting", "objective", "margin", "is_production_baseline"):
        result[key] = variant_row[key]
    return result


def _nav_statistics(values: pd.Series) -> dict[str, float | None]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if series.empty:
        return {
            "mean_bp": None,
            "cumulative_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "max_drawdown": None,
            "win_rate": None,
        }
    nav = (1.0 + series).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    annualized_volatility = float(series.std(ddof=1) * math.sqrt(252.0)) if len(series) > 1 else 0.0
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    annualized_return = float(nav.iloc[-1] ** (252.0 / len(series)) - 1.0)
    return {
        "mean_bp": float(series.mean() * 10_000.0),
        "cumulative_return": float(nav.iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": float(series.mean() / std * math.sqrt(252.0)) if math.isfinite(std) and std > 0.0 else None,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((series > 0.0).mean()),
    }


def _scope_frame(daily: pd.DataFrame, *, start: str | None, end: str | None, active_only: bool) -> pd.DataFrame:
    frame = daily.loc[daily["execution_metadata_complete"]].copy()
    days = pd.to_datetime(frame["score_day"])
    if start is not None:
        frame = frame.loc[days >= pd.Timestamp(start)].copy()
        days = pd.to_datetime(frame["score_day"])
    if end is not None:
        frame = frame.loc[days <= pd.Timestamp(end)].copy()
    if active_only:
        frame = frame.loc[frame["selection_active"]].copy()
    return frame


def _scope_metrics(frame: pd.DataFrame, *, variant_row: Mapping[str, object], scope: str, active_only: bool) -> dict[str, object]:
    result: dict[str, object] = {
        **dict(variant_row),
        "scope": scope,
        "active_only": bool(active_only),
        "n_days": int(len(frame)),
        "start": None if frame.empty else str(pd.to_datetime(frame["score_day"]).min().date()),
        "end": None if frame.empty else str(pd.to_datetime(frame["score_day"]).max().date()),
        "selection_active_days": int(frame["selection_active"].sum()) if not frame.empty else 0,
        "fallback_days": int((~frame["selection_active"]).sum()) if not frame.empty else 0,
    }
    selected = _nav_statistics(frame["selected_return"])
    equal_weight = _nav_statistics(frame["equal_weight_return"])
    result.update({f"selected_{key}": value for key, value in selected.items()})
    result.update({f"equal_weight_{key}": value for key, value in equal_weight.items()})
    for model in MODELS:
        stats = _nav_statistics(frame[f"realized_return_{model}"])
        result.update({f"{model}_{key}": value for key, value in stats.items()})
        result[f"vs_{model}_mean_bp"] = (
            float((frame["selected_return"] - frame[f"realized_return_{model}"]).mean() * 10_000.0)
            if not frame.empty
            else None
        )
    result.update(
        {
            "selection_alpha_mean_bp": float(frame["selection_alpha"].mean() * 10_000.0) if not frame.empty else None,
            "selection_alpha_sum_bp": float(frame["selection_alpha"].sum() * 10_000.0) if not frame.empty else None,
            "hit_rate": float((frame["basegap_selected_name"] == frame["best_name"]).mean()) if not frame.empty else None,
            "mean_rank": float(frame["selected_rank"].mean()) if not frame.empty else None,
            "mean_regret_bp": float(frame["regret"].mean() * 10_000.0) if not frame.empty else None,
            "model_switches": int(frame["basegap_selected_name"].ne(frame["basegap_selected_name"].shift()).sum() - 1)
            if len(frame) > 1
            else 0,
        }
    )
    return result


def build_summary_metrics(all_daily: pd.DataFrame, grid: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None, str | None]] = [("full", None, None), *SPLITS]
    for variant_row in grid:
        variant_daily = all_daily.loc[all_daily["variant"] == variant_row["variant"]].copy()
        for scope, start, end in scopes:
            rows.append(_scope_metrics(_scope_frame(variant_daily, start=start, end=end, active_only=False), variant_row=variant_row, scope=scope, active_only=False))
            rows.append(_scope_metrics(_scope_frame(variant_daily, start=start, end=end, active_only=True), variant_row=variant_row, scope=scope, active_only=True))
    return pd.DataFrame(rows)


def rank_design_variants(summary: pd.DataFrame) -> pd.DataFrame:
    """Choose exactly once using *only* active design-period selection alpha."""

    ranking = summary.loc[(summary["scope"] == "design") & summary["active_only"]].copy()
    ranking = ranking.loc[ranking["n_days"] > 0].copy()
    ranking["metric_order"] = ranking["metric"].map({"trim20_lcb10": 0, "lcb10": 1, "mean": 2}).fillna(99)
    ranking = ranking.sort_values(
        ["selection_alpha_mean_bp", "hit_rate", "mean_regret_bp", "lookback_days", "nearest_k", "metric_order"],
        ascending=[False, False, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "design_rank", np.arange(1, len(ranking) + 1))
    return ranking.drop(columns=["metric_order"])


def _plot_nav(run_root: Path, daily: pd.DataFrame, design_winner: str) -> None:
    baseline_name = _variant_name(**BASELINE_POINT)
    relevant = daily.loc[
        daily["variant"].isin([baseline_name, design_winner]) & daily["execution_metadata_complete"]
    ].copy()
    if relevant.empty:
        return
    figure, axis = plt.subplots(figsize=(13, 6))
    for variant, label in ((baseline_name, "production BaseGap point"), (design_winner, "design-selected grid point")):
        frame = relevant.loc[relevant["variant"] == variant].sort_values("score_day")
        if frame.empty:
            continue
        nav = (1.0 + frame["selected_return"].astype(float)).cumprod()
        axis.plot(pd.to_datetime(frame["score_day"]), nav, label=label)
    equal = relevant.loc[relevant["variant"] == baseline_name].sort_values("score_day")
    if not equal.empty:
        axis.plot(
            pd.to_datetime(equal["score_day"]),
            (1.0 + equal["equal_weight_return"].astype(float)).cumprod(),
            label="three-candidate arithmetic equal weight",
            linestyle="--",
            color="black",
        )
    for _, _, boundary in SPLITS[:-1]:
        axis.axvline(pd.Timestamp(boundary), color="grey", linewidth=0.7, alpha=0.7)
    axis.set_title("Pure BaseGap selector-shadow-return replay (metadata-complete days)")
    axis.set_ylabel("Cumulative NAV")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(run_root / "nav_selector_vs_equal_weight.png", dpi=160)
    plt.close(figure)


def _plot_heatmaps(run_root: Path, summary: pd.DataFrame) -> None:
    for scope in ("design", "validation"):
        frame = summary.loc[(summary["scope"] == scope) & summary["active_only"]].copy()
        if frame.empty:
            continue
        figure, axes = plt.subplots(1, len(METRIC_GRID), figsize=(5.2 * len(METRIC_GRID), 4.5), sharey=True)
        for axis, metric in zip(np.atleast_1d(axes), METRIC_GRID):
            data = frame.loc[frame["metric"] == metric]
            table = data.pivot(index="nearest_k", columns="lookback_days", values="selection_alpha_mean_bp")
            table = table.reindex(index=NEAREST_K_GRID, columns=LOOKBACK_GRID)
            image = axis.imshow(table.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
            axis.set_title(metric)
            axis.set_xlabel("lookback days")
            axis.set_xticks(np.arange(len(LOOKBACK_GRID)), labels=LOOKBACK_GRID)
            axis.set_yticks(np.arange(len(NEAREST_K_GRID)), labels=NEAREST_K_GRID)
            if axis is np.atleast_1d(axes)[0]:
                axis.set_ylabel("nearest K")
            for row_index in range(table.shape[0]):
                for col_index in range(table.shape[1]):
                    value = table.iat[row_index, col_index]
                    if math.isfinite(value):
                        axis.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8)
            figure.colorbar(image, ax=axis, label="selection alpha (bp/day)")
        figure.suptitle(f"Active BaseGap selection alpha: {scope} (not a promotion result)")
        figure.tight_layout()
        figure.savefig(run_root / f"grid_{scope}_selection_alpha_heatmap.png", dpi=160)
        plt.close(figure)


def _task_state_text(*, phase: str, run_root: Path, source_run: Path, detail: str) -> str:
    return f"""# Task State

## Objective

- Frozen, research-only pure three-candidate BaseGap internal tuning on the live50 snapshot.

## Risk Level

- medium: research conclusions can be overfit; all runtime/live assets remain read-only.

## Current Verified Facts

- source is the fixed frozen relative-utility run: `{source_run}`
- output is isolated beneath `{run_root}`
- no DB, scheduler, live runtime, model state, factor store, live config, or live result root is written.

## Files Read

- source `run_manifest.json`, frozen switch config, frozen return histories, and frozen T1430 state.

## Files Changed

- only files under this isolated research run root.

## Commands Run

- `py harness/tools/model_switch_basegap_tuning_replay.py ...`

## Artifacts

- pending / see this run root.

## Open Risks

- selector-shadow-return replay is not a full new execution backtest; 2026 final OOS must not be used for parameter selection.

## Next Action

- {detail}

## Handoff Summary

- phase: {phase}
"""


def _write_results(
    run_root: Path,
    *,
    source_run: Path,
    design_ranking: pd.DataFrame,
    summary: pd.DataFrame,
    baseline_variant: str,
) -> None:
    winner = design_ranking.iloc[0]
    selected_variant = str(winner["variant"])
    lines = [
        "# Pure BaseGap internal tuning (frozen research replay)",
        "",
        "## Scope",
        "",
        f"- Source snapshot: `{source_run}`",
        "- Candidate order: Regsim, Ensemble, HL20; selection is direct argmax with no Champion/Robust/Fusion/threshold logic.",
        "- Grid: 51 preregistered (lookback, K, metric) points.  State, candidates, standalone returns, mask, trade rule, cost and factor inputs were not changed.",
        "- Evaluation is selector-shadow-return only.  It is not a live promotion or a new full single-book execution backtest.",
        "",
        "## Parameter selection rule",
        "",
        "- The design winner is selected once by active design-period mean selection alpha versus the three-candidate arithmetic equal weight; ties use hit rate, lower regret, then smaller window/K and fixed metric order.",
        "- Validation and final OOS are reporting-only; final OOS was not used for selection.",
        "",
        "## Design-selected point",
        "",
        f"- `{selected_variant}`: lookback={int(winner['lookback_days'])}, K={int(winner['nearest_k'])}, metric={winner['metric']}; design active alpha={float(winner['selection_alpha_mean_bp']):.3f} bp/day.",
        "",
        "## Aligned summary",
        "",
        "| variant | scope | n | alpha vs equal weight (bp/day) | hit rate | mean regret (bp) | selected Sharpe | selected MDD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    report_rows = summary.loc[
        (summary["variant"].isin([baseline_variant, selected_variant]))
        & summary["scope"].isin(["design", "validation", "final_oos", "full"])
        & summary["active_only"]
    ].sort_values(["variant", "scope"])
    for _, row in report_rows.iterrows():
        lines.append(
            "| {variant} | {scope} | {n} | {alpha:.3f} | {hit:.2%} | {regret:.3f} | {sharpe} | {mdd} |".format(
                variant=row["variant"],
                scope=row["scope"],
                n=int(row["n_days"]),
                alpha=float(row["selection_alpha_mean_bp"]),
                hit=float(row["hit_rate"]),
                regret=float(row["mean_regret_bp"]),
                sharpe="" if pd.isna(row["selected_sharpe"]) else f"{float(row['selected_sharpe']):.3f}",
                mdd="" if pd.isna(row["selected_max_drawdown"]) else f"{float(row['selected_max_drawdown']):.2%}",
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- The source T1430 state keeps its existing PIT caveat; it is not new strict-14:29 evidence.",
            "- Do not choose another grid point using the reported validation/final-OOS values.  Any later score-geometry or 50-factor-state family must be preregistered as a separate stage.",
            "- `daily_selector_replay.csv`, `summary_metrics.csv`, `design_ranking.csv`, `leakage_audit.json`, and plots contain the detailed evidence.",
            "",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_replay(*, source_run: Path, run_root: Path) -> dict[str, object]:
    """Execute the bounded Stage-A grid and write isolated evidence."""

    source_run = _assert_source_run(source_run)
    run_root = _assert_output_root(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "task_state.md").write_text(
        _task_state_text(
            phase="running",
            run_root=run_root,
            source_run=source_run,
            detail="verify source hashes, reproduce production BaseGap, then run the preregistered 51-point in-memory grid.",
        ),
        encoding="utf-8",
    )
    _write_json(
        run_root / "run_status.json",
        {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc),
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "research_output_only": True,
        },
    )

    config, verification = validate_frozen_source(source_run)
    _write_json(run_root / "input_verification.json", verification)
    panel, feature_cols = prepare_panel(config)
    grid = preregistered_grid()
    pd.DataFrame(grid).to_csv(run_root / "preregistered_grid.csv", index=False)

    # Gate 1: actual production function against frozen inputs must reproduce
    # the already recorded direct BaseGap file exactly.
    production = reproduce_production_basegap(config, panel["score_day"].tolist())
    expected_path = source_run / "daily_direct_basegap.csv"
    expected = pd.read_csv(expected_path)
    expected["score_day"] = _normalise_date_column(expected, "score_day")
    production_audit = parity_audit(production, expected, label="production_vs_source_direct_basegap")
    production.to_csv(run_root / "production_direct_basegap.csv", index=False)

    # Gate 2: the fast in-memory reconstruction must be exact at the current
    # production point before it is allowed to run any variant.
    baseline_row = next(item for item in grid if item["is_production_baseline"])
    in_memory_baseline = run_basegap_variant(
        panel,
        feature_cols=feature_cols,
        lookback_days=int(baseline_row["lookback_days"]),
        nearest_k=int(baseline_row["nearest_k"]),
        min_periods=int(baseline_row["min_periods"]),
        metric=str(baseline_row["metric"]),
        variant=str(baseline_row["variant"]),
    )
    in_memory_direct = in_memory_baseline[
        [
            "score_day",
            "basegap_selected_name",
            "basegap_reason",
            "basegap_history_days",
            "basegap_history_end",
            "basegap_score_gap",
            *(f"basegap_score_{model}" for model in MODELS),
        ]
    ]
    memory_audit = parity_audit(in_memory_direct, production, label="in_memory_vs_production_direct_basegap")
    pd.concat([production_audit, memory_audit], ignore_index=True).to_csv(run_root / "baseline_parity_audit.csv", index=False)

    all_variants: list[pd.DataFrame] = [add_variant_metadata(in_memory_baseline, baseline_row)]
    for variant_row in grid:
        if variant_row["variant"] == baseline_row["variant"]:
            continue
        daily = run_basegap_variant(
            panel,
            feature_cols=feature_cols,
            lookback_days=int(variant_row["lookback_days"]),
            nearest_k=int(variant_row["nearest_k"]),
            min_periods=int(variant_row["min_periods"]),
            metric=str(variant_row["metric"]),
            variant=str(variant_row["variant"]),
        )
        all_variants.append(add_variant_metadata(daily, variant_row))
    all_daily = pd.concat(all_variants, ignore_index=True)
    all_daily.to_csv(run_root / "daily_selector_replay.csv", index=False)

    summary = build_summary_metrics(all_daily, grid)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    design_ranking = rank_design_variants(summary)
    design_ranking.to_csv(run_root / "design_ranking.csv", index=False)
    winner = design_ranking.iloc[0]

    active = all_daily.loc[all_daily["selection_active"]].copy()
    leakage_audit = {
        "strict_predecessor_history": bool(
            (pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all()
        ),
        "active_rows": int(len(active)),
        "variants": int(all_daily["variant"].nunique()),
        "source_manifest_hash_verified": True,
        "production_parity_verified": True,
        "in_memory_parity_verified": True,
        "return_day_not_used_for_same_day_selection": True,
        "future_state_not_used_for_prior_selection": True,
        "final_oos_selection_rule": "not used; reported once after design-only winner selection",
    }
    _write_json(run_root / "leakage_audit.json", leakage_audit)
    _plot_nav(run_root, all_daily, str(winner["variant"]))
    _plot_heatmaps(run_root, summary)
    _write_results(
        run_root,
        source_run=source_run,
        design_ranking=design_ranking,
        summary=summary,
        baseline_variant=str(baseline_row["variant"]),
    )

    manifest = {
        "run_class": "research_only_pure_three_candidate_basegap_internal_grid",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "live_config_written": False,
        "live_result_written": False,
        "model_state_written": False,
        "frozen_source_run": str(source_run),
        "source_input_verification": {
            "source_manifest_sha256": verification["source_manifest_sha256"],
            "verified_file_count": verification["verified_file_count"],
        },
        "comparison_contract": {
            "candidate_returns": "aligned standalone full-cycle day_return",
            "selector": "pure symmetric direct BaseGap argmax",
            "state": "frozen path_full_t1430 only",
            "candidate_order": list(MODELS),
            "threshold_or_margin_gate": False,
            "champion_robust_fusion_logic": False,
            "evaluation_days": "only execution_metadata_complete dates",
            "main_objective": "selected_return - arithmetic mean(Regsim, Ensemble, HL20)",
            "final_oos_policy": "reported, not used for selection",
        },
        "splits": [{"name": name, "start": start, "end": end} for name, start, end in SPLITS],
        "grid": grid,
        "design_selected_variant": str(winner["variant"]),
        "source_code": [
            {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
            {
                "path": str(REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py"),
                "sha256": sha256(REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py"),
            },
        ],
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
        "completed_at_utc": datetime.now(timezone.utc),
    }
    _write_json(run_root / "run_manifest.json", manifest)
    _write_json(
        run_root / "run_status.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc),
            "design_selected_variant": str(winner["variant"]),
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "research_output_only": True,
        },
    )
    (run_root / "task_state.md").write_text(
        _task_state_text(
            phase="completed",
            run_root=run_root,
            source_run=source_run,
            detail="Stage A completed.  Interpret design/validation/final-OOS separately; do not use final OOS to select another point.",
        ),
        encoding="utf-8",
    )
    return {
        "run_root": str(run_root),
        "design_selected_variant": str(winner["variant"]),
        "design_selected_alpha_bp_per_day": float(winner["selection_alpha_mean_bp"]),
        "verified_frozen_files": int(verification["verified_file_count"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN, help="locked frozen source run")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="isolated research root")
    parser.add_argument("--run-name", default=None, help="new plain-name leaf below --output-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = _make_run_root(args.output_root, args.run_name)
    result = run_replay(source_run=args.source_run, run_root=run_root)
    print(json.dumps(result, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
