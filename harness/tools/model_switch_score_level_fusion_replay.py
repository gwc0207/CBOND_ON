"""Research-only score-level continuous three-model fusion replay.

This is deliberately not part of the live model-switch path.  It consumes the
already frozen v5 relative-utility replay inputs, creates two daily score trees
under a fresh research root, and sends them to the existing generic
``backtest_runtime.run`` implementation.  The strategy, allowlist, masks,
costs, benchmark, Top20 selection, and execution-cycle accounting are thus
unchanged by this tool.

The formulas are fixed before the run:

* frozen Regsim score execution identity baseline;
* equal-weight fusion of three within-day percentile score ranks;
* causal Ridge-utility softmax fusion, with temperature equal to the RMS of
  realised three-model relative utilities over up to 360 strictly prior,
  complete-metadata rows.

There is no score-gap threshold, confidence gate, LCB, Champion preference,
BaseGap/Robust routing, clipping, or parameter selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json5
import numpy as np
import pandas as pd

from cbond_on.app.usecases import backtest_runtime
from cbond_on.config.loader import load_config_file
from cbond_on.core.config import resolve_config_file_path
from cbond_on.core.fees import load_fees_buy_sell_bps


SOURCE_RUN = Path(
    r"D:\cbond_on\research_scratch\model_switch_relative_utility_20260811"
    r"\run_20260811_relative_utility_v5_with_ranker"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\cbond_on\research_scratch\model_switch_dynamic_weight_20260811\score_level_v1"
)
DEFAULT_RUN_NAME = "run_main_20260811"
SOURCE_VARIANT = "state_score_geometry_lags"
MODELS = ("Regsim", "Ensemble", "HL20")
MODEL_UTILITY_COLUMNS = {
    "Regsim": "utility_regsim",
    "Ensemble": "utility_ensemble",
    "HL20": "utility_hl20",
}
MODEL_RETURN_COLUMNS = {model: f"realized_return_{model}" for model in MODELS}
MAIN_START = date(2024, 5, 8)
MAIN_END = date(2026, 7, 30)
EXPECTED_MAIN_DAYS = 541
MAX_HISTORY_DAYS = 360
TOP_K = 20
O005_ALLOWLIST_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"
EPSILON = 1e-12

EQUAL_VARIANT = "equal_weight_rank_score_fusion"
RIDGE_VARIANT = "ridge_softmax_rank_score_fusion"
SCORE_VARIANTS = (EQUAL_VARIANT, RIDGE_VARIANT)

PLAN = {
    "run_class": "research_only_score_level_continuous_weight_fusion_v1",
    "source_variant": SOURCE_VARIANT,
    "main_window": {"start": str(MAIN_START), "end": str(MAIN_END), "complete_metadata_days": EXPECTED_MAIN_DAYS},
    "baseline": "frozen Regsim scores through generic backtest runtime; exact frozen-return identity required",
    "score_normalization": "each model's full shared frozen same-day score universe is percentile-ranked before fusion",
    "equal_weight": [1.0 / 3.0] * 3,
    "ridge_softmax": {
        "forecast": "precomputed causal state_score_geometry_lags utility; source asserts training_end < score_day",
        "temperature": "sqrt(mean(relative_utility^2)) over at most 360 complete rows strictly before score_day",
        "warmup_or_input_unavailable": "equal weights",
    },
    "prohibited": [
        "top1_top2_gap",
        "confidence_threshold",
        "LCB",
        "BaseGap_route",
        "Robust_route",
        "Champion_preference",
        "veto",
        "weight_clip",
        "hyperparameter_search",
    ],
    "execution": "existing generic backtest_runtime; original o_0005, strategy01 Top20, cost, mask, benchmark, and cycle-return logic",
}


@dataclass(frozen=True)
class SourceInputs:
    run_root: Path
    snapshot_root: Path
    snapshot_manifest_path: Path
    daily_forecast_path: Path
    regsim_return_path: Path
    frozen_live_config_path: Path
    frozen_strategy_config_path: Path
    score_root: Path


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _assert_output_root(path: Path) -> Path:
    """Reject any write target outside the dedicated score-level v1 root."""

    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"score-level research output must stay under {allowed}: {resolved}") from exc
    return resolved


def _read_json5(path: Path) -> dict[str, Any]:
    value = json5.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping config: {path}")
    return value


def _score_path(root: Path, score_day: object) -> Path:
    timestamp = pd.Timestamp(score_day)
    return root / timestamp.strftime("%Y-%m") / f"{timestamp:%Y-%m-%d}.csv"


def _require_file(path: Path, *, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def load_source_inputs(source_run: Path = SOURCE_RUN) -> SourceInputs:
    run_root = Path(source_run).resolve()
    snapshot_root = run_root / "input_snapshot"
    source = SourceInputs(
        run_root=run_root,
        snapshot_root=snapshot_root,
        snapshot_manifest_path=_require_file(snapshot_root / "input_manifest.json", label="frozen source manifest"),
        daily_forecast_path=_require_file(run_root / "daily_selector_replay.csv", label="source daily forecasts"),
        regsim_return_path=_require_file(snapshot_root / "returns" / "Challenger_Regsim.csv", label="frozen Regsim returns"),
        frozen_live_config_path=_require_file(snapshot_root / "configs" / "live_config.json5", label="frozen live config"),
        frozen_strategy_config_path=_require_file(
            snapshot_root / "configs" / "strategy01_config.json5", label="frozen strategy01 config"
        ),
        score_root=snapshot_root / "scores",
    )
    for model in MODELS:
        if not (source.score_root / model).is_dir():
            raise FileNotFoundError(f"missing frozen score root for {model}: {source.score_root / model}")
    return source


def verify_source_snapshot(source: SourceInputs) -> int:
    """Verify all predecessor snapshot hashes before using any frozen input."""

    raw = json.loads(source.snapshot_manifest_path.read_text(encoding="utf-8"))
    entries = raw.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"frozen manifest has no files list: {source.snapshot_manifest_path}")
    source_root = source.snapshot_root.resolve()
    verified = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("frozen manifest has a non-mapping file entry")
        snapshot_text = entry.get("snapshot")
        expected = entry.get("sha256")
        if not isinstance(snapshot_text, str) or not isinstance(expected, str):
            raise ValueError("frozen manifest entry lacks snapshot or sha256")
        path = Path(snapshot_text).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"frozen manifest points outside its snapshot root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"frozen snapshot file missing: {path}")
        if sha256(path) != expected:
            raise RuntimeError(f"frozen snapshot hash mismatch: {path}")
        verified += 1
    return verified


def _coerce_bool(series: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False, "nan": False, "": False}
    result = lowered.map(mapping)
    if result.isna().any():
        values = sorted(lowered.loc[result.isna()].unique().tolist())[:5]
        raise ValueError(f"cannot parse boolean {name}: {values}")
    return result.astype(bool)


def _strictly_increasing_days(days: pd.Series, *, label: str) -> None:
    values = pd.to_datetime(days, errors="coerce")
    if values.isna().any() or values.duplicated().any() or not values.is_monotonic_increasing:
        raise ValueError(f"{label} must contain unique strictly increasing valid dates")


def load_main_source_daily(source: SourceInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the predeclared complete-metadata main window and frozen baseline."""

    daily = pd.read_csv(source.daily_forecast_path)
    required = {
        "score_day",
        "variant",
        "prediction_ready",
        "training_end",
        "execution_metadata_complete",
        *MODEL_UTILITY_COLUMNS.values(),
        *MODEL_RETURN_COLUMNS.values(),
    }
    missing = sorted(required - set(daily.columns))
    if missing:
        raise ValueError(f"source forecast lacks columns: {missing}")
    daily["score_day"] = pd.to_datetime(daily["score_day"], errors="coerce").dt.date
    daily["training_end"] = pd.to_datetime(daily["training_end"], errors="coerce").dt.date
    daily["prediction_ready"] = _coerce_bool(daily["prediction_ready"], name="prediction_ready")
    daily["execution_metadata_complete"] = _coerce_bool(
        daily["execution_metadata_complete"], name="execution_metadata_complete"
    )
    selected = daily.loc[daily["variant"].eq(SOURCE_VARIANT)].copy()
    if len(selected) != 547:
        raise ValueError(f"expected 547 frozen {SOURCE_VARIANT} rows, got {len(selected)}")
    selected = selected.loc[
        selected["execution_metadata_complete"]
        & selected["score_day"].ge(MAIN_START)
        & selected["score_day"].le(MAIN_END)
    ].sort_values("score_day").reset_index(drop=True)
    _strictly_increasing_days(selected["score_day"], label="main source score_day")
    if len(selected) != EXPECTED_MAIN_DAYS:
        raise ValueError(f"expected {EXPECTED_MAIN_DAYS} complete-metadata main rows, got {len(selected)}")
    ready = selected["prediction_ready"]
    if ready.any():
        training_end = selected.loc[ready, "training_end"]
        if training_end.isna().any() or not (training_end.to_numpy() < selected.loc[ready, "score_day"].to_numpy()).all():
            raise ValueError("source Ridge utility violates training_end < score_day")
        utilities = selected.loc[ready, list(MODEL_UTILITY_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(utilities.to_numpy(dtype=float)).all():
            raise ValueError("source marks a Ridge forecast ready with non-finite utilities")

    returns = pd.read_csv(source.regsim_return_path, usecols=["trade_date", "day_return"])
    returns["score_day"] = pd.to_datetime(returns["trade_date"], errors="coerce").dt.date
    returns["frozen_regsim_return"] = pd.to_numeric(returns["day_return"], errors="coerce")
    returns = returns.dropna(subset=["score_day", "frozen_regsim_return"])[["score_day", "frozen_regsim_return"]]
    returns = returns.loc[returns["score_day"].isin(selected["score_day"])].sort_values("score_day").reset_index(drop=True)
    _strictly_increasing_days(returns["score_day"], label="frozen Regsim score_day")
    if len(returns) != len(selected) or not returns["score_day"].tolist() == selected["score_day"].tolist():
        raise ValueError("frozen Regsim history does not exactly cover the complete-metadata main window")
    return selected, returns


def equal_weights() -> np.ndarray:
    return np.full(len(MODELS), 1.0 / len(MODELS), dtype=float)


def causal_relative_utility_scale(history_returns: np.ndarray) -> float | None:
    """RMS cross-model relative utility using only a caller-provided prior window."""

    values = np.asarray(history_returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(MODELS) or len(values) == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError("non-finite realised return in causal temperature history")
    utilities = values - values.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(np.square(utilities))))
    return scale if math.isfinite(scale) and scale > 0.0 else None


def softmax_weights(utilities: Sequence[float], *, temperature: float) -> np.ndarray:
    """Stable continuous simplex map; it deliberately has no winner rule."""

    values = np.asarray(utilities, dtype=float)
    if values.shape != (len(MODELS),) or not np.isfinite(values).all():
        raise ValueError("softmax utilities must be a finite length-three vector")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("softmax temperature must be positive and finite")
    logits = values / float(temperature)
    logits = logits - float(logits.max())
    weights = np.exp(logits)
    weights /= float(weights.sum())
    if not np.isfinite(weights).all() or np.any(weights <= 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise RuntimeError("softmax produced an invalid simplex weight vector")
    return weights


def build_daily_weights(main_daily: pd.DataFrame) -> pd.DataFrame:
    """Build every day before touching that day's realised returns.

    ``main_daily`` is sorted. The only realised returns consumed for the day at
    index ``i`` are ``iloc[max(0, i-360):i]``.  The current source utility is a
    frozen T-day Ridge prediction, not the current realised return.
    """

    frame = main_daily.copy().sort_values("score_day").reset_index(drop=True)
    _strictly_increasing_days(frame["score_day"], label="weight input score_day")
    return_columns = [MODEL_RETURN_COLUMNS[model] for model in MODELS]
    utility_columns = [MODEL_UTILITY_COLUMNS[model] for model in MODELS]
    for column in [*return_columns, *utility_columns]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for index, current in frame.iterrows():
        prior = frame.iloc[max(0, index - MAX_HISTORY_DAYS) : index]
        history = prior[return_columns].to_numpy(dtype=float)
        temperature = causal_relative_utility_scale(history)
        ready = bool(current["prediction_ready"])
        utilities = current[utility_columns].to_numpy(dtype=float)
        equal = equal_weights()
        if ready:
            training_end = current["training_end"]
            if pd.isna(training_end) or not pd.Timestamp(training_end).date() < current["score_day"]:
                raise ValueError(f"ready Ridge row is not strictly prior-trained: {current['score_day']}")
            if not np.isfinite(utilities).all():
                raise ValueError(f"ready Ridge row has non-finite utilities: {current['score_day']}")
            if not math.isclose(float(utilities.sum()), 0.0, abs_tol=1e-8):
                raise ValueError(f"Ridge utilities are not coherent on {current['score_day']}: {utilities}")
        if ready and temperature is not None:
            ridge = softmax_weights(utilities, temperature=temperature)
            ridge_reason = "causal_ridge_utility_softmax"
        else:
            ridge = equal.copy()
            ridge_reason = "warmup_or_input_unavailable_equal_weight"
        for label, weights in ((EQUAL_VARIANT, equal), (RIDGE_VARIANT, ridge)):
            if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12) or np.any(weights < -EPSILON):
                raise RuntimeError(f"invalid {label} weights on {current['score_day']}")
        record: dict[str, object] = {
            "score_day": current["score_day"],
            "prediction_ready": ready,
            "training_end": current["training_end"],
            "history_complete_rows": int(len(prior)),
            "history_end": prior["score_day"].iloc[-1] if len(prior) else None,
            "causal_temperature": temperature,
            "ridge_weight_reason": ridge_reason,
            **{f"forecast_utility_{model}": float(utilities[position]) if np.isfinite(utilities[position]) else np.nan for position, model in enumerate(MODELS)},
            **{f"realized_return_{model}": float(current[MODEL_RETURN_COLUMNS[model]]) for model in MODELS},
        }
        for variant, weights in ((EQUAL_VARIANT, equal), (RIDGE_VARIANT, ridge)):
            record[f"effective_models_{variant}"] = float(1.0 / np.sum(np.square(weights)))
            for position, model in enumerate(MODELS):
                record[f"weight_{variant}_{model}"] = float(weights[position])
        rows.append(record)
    return pd.DataFrame(rows)


def _load_frozen_score(source: SourceInputs, model: str, score_day: object) -> pd.Series:
    path = _score_path(source.score_root / model, score_day)
    _require_file(path, label=f"frozen {model} score")
    frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    expected = pd.Timestamp(score_day).date()
    observed = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    if frame.empty or observed.isna().any() or not observed.eq(expected).all():
        raise ValueError(f"frozen {model} score day mismatch: {path}; expected {expected}")
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame["code"].eq("").any() or frame["score"].isna().any() or not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError(f"invalid code or score in frozen {model} input: {path}")
    if frame["code"].duplicated().any():
        raise ValueError(f"duplicate code in frozen {model} score: {path}")
    series = frame.set_index("code")["score"].astype(float)
    if len(series) < TOP_K + 1:
        raise ValueError(f"frozen {model} score has too few names on {expected}: {len(series)}")
    return series


def rank_blend_scores(
    score_by_model: Mapping[str, pd.Series],
    weights: Sequence[float],
    *,
    score_day: object,
) -> pd.DataFrame:
    """Percentile-rank fusion on the full *identical* frozen daily universe.

    No intersection is silently taken: a model-universe discrepancy is a hard
    failure, before the downstream generic runtime applies its normal o_0005
    filter and trading masks.
    """

    weight = np.asarray(weights, dtype=float)
    if weight.shape != (len(MODELS),) or not np.isfinite(weight).all() or np.any(weight < -EPSILON):
        raise ValueError("rank fusion weights must be a finite non-negative length-three vector")
    if not math.isclose(float(weight.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("rank fusion weights must sum to one")
    expected_codes: set[str] | None = None
    for model in MODELS:
        series = score_by_model.get(model)
        if not isinstance(series, pd.Series):
            raise ValueError(f"missing frozen score series for {model}")
        if series.index.has_duplicates or series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
            raise ValueError(f"invalid frozen score series for {model}")
        codes = set(series.index.astype(str))
        if expected_codes is None:
            expected_codes = codes
        elif codes != expected_codes:
            raise ValueError("rank-level fusion requires identical model score universes")
    if not expected_codes or len(expected_codes) < TOP_K + 1:
        raise ValueError("rank-level fusion has insufficient shared score universe")
    codes = sorted(expected_codes)
    ranked = pd.DataFrame({model: score_by_model[model].reindex(codes).astype(float) for model in MODELS}, index=codes)
    percentile = ranked.rank(axis=0, method="average", pct=True)
    blended = percentile.to_numpy(dtype=float) @ weight
    if not np.isfinite(blended).all():
        raise RuntimeError("rank-level fusion produced non-finite scores")
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp(score_day).strftime("%Y-%m-%d"),
            "code": codes,
            "score": blended,
        }
    )


def _codes_digest(codes: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()


def write_fused_score_inputs(source: SourceInputs, weights: pd.DataFrame, *, run_root: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    """Materialise only the two pre-registered isolated score trees."""

    score_roots = {variant: run_root / "fused_score_inputs" / variant for variant in SCORE_VARIANTS}
    audit_rows: list[dict[str, object]] = []
    for _, row in weights.iterrows():
        score_day = row["score_day"]
        score_by_model = {model: _load_frozen_score(source, model, score_day) for model in MODELS}
        code_order = sorted(score_by_model[MODELS[0]].index.astype(str).tolist())
        for variant in SCORE_VARIANTS:
            values = [float(row[f"weight_{variant}_{model}"]) for model in MODELS]
            blended = rank_blend_scores(score_by_model, values, score_day=score_day)
            output = _score_path(score_roots[variant], score_day)
            output.parent.mkdir(parents=True, exist_ok=True)
            blended.to_csv(output, index=False)
            audit_rows.append(
                {
                    "score_day": score_day,
                    "variant": variant,
                    "score_rows": int(len(blended)),
                    "shared_universe_sha256": _codes_digest(code_order),
                    "fused_score_sha256": sha256(output),
                    **{f"weight_{model}": values[position] for position, model in enumerate(MODELS)},
                }
            )
    return score_roots, pd.DataFrame(audit_rows)


def _validate_frozen_execution_contract(live: Mapping[str, Any], strategy: Mapping[str, Any]) -> None:
    allowlist = live.get("allowlist")
    if not isinstance(allowlist, Mapping):
        raise ValueError("frozen live config lacks allowlist mapping")
    if not bool(allowlist.get("enabled", True)) or str(allowlist.get("table", "")) != O005_ALLOWLIST_TABLE:
        raise ValueError("frozen live config is not the required o_0005-only allowlist contract")
    if not math.isclose(float(strategy.get("top_k", float("nan"))), float(TOP_K), abs_tol=0.0):
        raise ValueError("frozen strategy no longer represents Top20")
    if not math.isclose(float(strategy.get("max_weight", float("nan"))), 0.05, abs_tol=0.0):
        raise ValueError("frozen strategy no longer represents max 5% single-name weight")
    if not math.isclose(float(strategy.get("turnover_ratio", float("nan"))), 1.0, abs_tol=0.0):
        raise ValueError("frozen strategy no longer represents full daily turnover")


def build_generic_backtest_config(
    *,
    source: SourceInputs,
    score_root: Path,
    output_root: Path,
    batch_id: str,
) -> dict[str, Any]:
    """Create an in-memory generic backtest config; no repository config is changed."""

    _assert_output_root(output_root)
    live = _read_json5(source.frozen_live_config_path)
    strategy = _read_json5(source.frozen_strategy_config_path)
    _validate_frozen_execution_contract(live, strategy)
    output = live.get("output")
    if not isinstance(output, Mapping):
        raise ValueError("frozen live config lacks output mapping")
    return {
        "start": str(MAIN_START),
        "end": str(MAIN_END),
        "batch_id": batch_id,
        "score_source": {"score_root": str(score_root)},
        "strategy_id": "strategy01_topk_turnover",
        "strategy_config": dict(strategy),
        "buy_twap_col": str(output.get("buy_twap_col", "twap_1442_1457")),
        "sell_twap_col": str(output.get("sell_twap_col", "twap_0930_0939")),
        "allowlist": dict(live["allowlist"]),
        "execution_lag_trading_days": 0,
        "freeze_signal_universe": False,
        "output_root": str(output_root),
    }


def _run_generic_backtest(*, source: SourceInputs, score_root: Path, run_root: Path, batch_id: str) -> tuple[Path, dict[str, Any]]:
    output_root = _assert_output_root(run_root / "generic_backtests")
    cfg = build_generic_backtest_config(source=source, score_root=score_root, output_root=output_root, batch_id=batch_id)
    result = backtest_runtime.run(start=MAIN_START, end=MAIN_END, cfg=cfg)
    return result.out_dir, cfg


def _load_generic_daily(out_dir: Path) -> pd.DataFrame:
    path = _require_file(out_dir / "daily_returns.csv", label="generic daily returns")
    frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["day_return"] = pd.to_numeric(frame["day_return"], errors="coerce")
    frame = frame.dropna(subset=["score_day", "day_return"])[["score_day", "day_return"]].sort_values("score_day")
    _strictly_increasing_days(frame["score_day"], label=f"generic output {out_dir}")
    return frame.reset_index(drop=True)


def compare_identity_parity(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    comparison = expected.merge(actual, on="score_day", how="outer", validate="one_to_one", indicator=True)
    comparison = comparison.rename(columns={"day_return": "generic_regsim_return"})
    both = comparison.loc[comparison["_merge"].eq("both")].copy()
    both["return_difference"] = both["generic_regsim_return"] - both["frozen_regsim_return"]
    max_abs = float(np.abs(both["return_difference"]).max()) if len(both) else float("inf")
    passed = bool(
        len(comparison) == len(expected)
        and comparison["_merge"].eq("both").all()
        and len(both) == len(expected)
        and math.isfinite(max_abs)
        and max_abs <= EPSILON
    )
    return comparison, {
        "status": "passed" if passed else "failed",
        "expected_days": int(len(expected)),
        "generic_days": int(len(actual)),
        "aligned_days": int(len(both)),
        "max_abs_return_difference": max_abs,
    }


def _portfolio_metrics(daily: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(daily["day_return"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("cannot summarise an empty backtest return series")
    nav = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(nav)
    drawdown = nav / peaks - 1.0
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 and np.std(values, ddof=1) > 0 else float("nan")
    return {
        "days": int(len(values)),
        "start_score_day": str(daily["score_day"].iloc[0]),
        "end_score_day": str(daily["score_day"].iloc[-1]),
        "total_return": float(nav[-1] - 1.0),
        "annualized_return": float(nav[-1] ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float(np.mean(values > 0.0)),
    }


def _paired_metrics(baseline: pd.DataFrame, candidate: pd.DataFrame, *, strategy: str) -> dict[str, object]:
    joined = baseline.merge(candidate, on="score_day", how="inner", suffixes=("_regsim", "_variant"), validate="one_to_one")
    delta = joined["day_return_variant"].to_numpy(dtype=float) - joined["day_return_regsim"].to_numpy(dtype=float)
    return {
        "strategy": strategy,
        "paired_days": int(len(joined)),
        "mean_daily_delta_bp": float(delta.mean() * 10_000.0),
        "sum_daily_delta_bp": float(delta.sum() * 10_000.0),
        "wins": int((delta > 0.0).sum()),
        "losses": int((delta < 0.0).sum()),
    }


def _weight_diagnostics(weights: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for variant in SCORE_VARIANTS:
        columns = [f"weight_{variant}_{model}" for model in MODELS]
        matrix = weights[columns].to_numpy(dtype=float)
        entropy = -np.sum(np.where(matrix > 0.0, matrix * np.log(matrix), 0.0), axis=1)
        records.append(
            {
                "strategy": variant,
                "days": int(len(matrix)),
                "mean_effective_models": float((1.0 / np.sum(np.square(matrix), axis=1)).mean()),
                "mean_weight_entropy": float(entropy.mean()),
                "min_weight_entropy": float(entropy.min()),
                "max_weight_entropy": float(entropy.max()),
                **{f"mean_weight_{model}": float(matrix[:, index].mean()) for index, model in enumerate(MODELS)},
                **{f"max_weight_{model}": float(matrix[:, index].max()) for index, model in enumerate(MODELS)},
            }
        )
    return pd.DataFrame(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False, float_format="%.6f").rstrip() + "\n```"


def _write_results(
    *,
    run_root: Path,
    status: Mapping[str, Any],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    identity = status.get("identity_regsim_parity", {})
    if summary.empty:
        score_section = [
            "## Score-level results",
            "",
            "No fused-score return is reported because the frozen Regsim generic-runtime identity check failed or the run was interrupted.",
        ]
    else:
        score_section = [
            "## Score-level summary",
            "",
            _markdown_table(summary),
            "",
            "## Paired daily delta versus frozen Regsim execution baseline",
            "",
            _markdown_table(paired),
            "",
            "## Weight diagnostics",
            "",
            _markdown_table(diagnostics),
        ]
    lines = [
        "# Research-only score-level continuous fusion v1",
        "",
        "## Locked contract",
        "",
        "- Main evaluation is `2024-05-08` through `2026-07-30` only: 541 score days with complete execution metadata.",
        "- Variants were fixed before results: frozen Regsim execution, equal-weight percentile-rank score fusion, and causal Ridge-utility softmax percentile-rank score fusion.",
        "- Each model is ranked on the complete shared frozen score universe for that day. The generic runtime then applies the unchanged o_0005 allowlist, market mask, Top20 strategy, 5% maximum weight, full turnover, costs, benchmark, and strict buy/sell cycle logic.",
        "- Ridge softmax uses only a current frozen T-day prediction and at most 360 complete strictly prior realised return rows for its temperature. Warmup/input-unavailable rows are exactly equal weight.",
        "- No BaseGap/Robust route, Champion preference, top-one/top-two gap, LCB, confidence threshold, veto, clip, or parameter sweep is used.",
        "",
        "## Required frozen Regsim identity check",
        "",
        "```json",
        json.dumps(identity, ensure_ascii=False, indent=2, default=_json_default),
        "```",
        "",
        *score_section,
        "",
        "## Caveats",
        "",
        "- The current generic raw calendar contains 543 dates in this interval, while the frozen score/return contract has 541 executable score days. `2026-06-11` and `2026-06-12` have no frozen score file for any of the three candidates, so the generic runtime records `missing_score` and skips them for every strategy. All identity and score-fusion tables are explicitly the common 541 executed score days, not all 543 calendar dates.",
        "- The v5 source freezes score, return, config, and T1430 state inputs, but not an immutable copy of daily raw execution prices or o_0005 pool snapshots. The generic runtime therefore reads the current configured DataHub raw/pool inputs. Baseline identity is checked before any fusion result is allowed, but it remains a historical replay boundary.",
        "- Although the in-memory generic config carries buy/sell TWAP names, the strict generic backtest does not consume those passed fields: it resolves execution price fields from the current benchmark config and reads the current fees config directly. Their paths and SHA-256 values at run start are recorded in `run_manifest.json`; identity parity proves current-contract equality, not that benchmark/fees/raw/pool were frozen and injected.",
        "- The source state provenance is T1430 and is not strict-1429-certified. This result is research-only and cannot authorize a live change.",
        "- No incomplete-metadata suffix day (`2026-07-31` through `2026-08-07`) enters this main run or its causal temperature histories.",
        "",
        "## Artifacts",
        "",
        "- `run_manifest.json`, `run_status.json`, `generic_backtest_configs.json`, and `daily_weights.csv` retain the contract and causal audit.",
        "- `identity_regsim_parity.csv` is the exact precondition for fused-score execution.",
        "- `fused_score_inputs/` contains isolated daily equal/Ridge rank-score files; `generic_backtests/` contains only scratch backtest outputs.",
        "- `daily_rank_score_audit.csv`, `daily_score_level_returns.csv`, `summary_metrics.csv`, `paired_vs_regsim.csv`, and `weight_diagnostics.csv` are the aligned research evidence.",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_status(path: Path, status: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _prepare_run_root(output_root: Path, run_name: str) -> Path:
    output_root = _assert_output_root(output_root)
    if Path(run_name).name != run_name or not run_name.strip():
        raise ValueError("run-name must be one non-empty directory leaf")
    run_root = _assert_output_root(output_root / run_name)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=False, exist_ok=False)
    return run_root


def run_replay(
    *,
    source_run: Path = SOURCE_RUN,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
) -> dict[str, Any]:
    """Execute the locked 541-day research replay, all writes under output_root."""

    output_root = _assert_output_root(Path(output_root))
    source = load_source_inputs(source_run)
    verified_files = verify_source_snapshot(source)
    main_daily, frozen_regsim = load_main_source_daily(source)
    run_root = _prepare_run_root(output_root, run_name)
    status_path = run_root / "run_status.json"
    paths_config_path = resolve_config_file_path("paths")
    fees_config_path = resolve_config_file_path("fees/fees")
    benchmark_config_path = resolve_config_file_path("benchmark/benchmark")
    buy_bps, sell_bps, fee_source = load_fees_buy_sell_bps()
    manifest: dict[str, Any] = {
        "run_class": PLAN["run_class"],
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "production_result_root_written": False,
        "source_run": str(source.run_root),
        "source_input_snapshot_manifest": {"path": str(source.snapshot_manifest_path), "sha256": sha256(source.snapshot_manifest_path)},
        "source_daily_forecast": {"path": str(source.daily_forecast_path), "sha256": sha256(source.daily_forecast_path)},
        "source_frozen_regsim_return": {"path": str(source.regsim_return_path), "sha256": sha256(source.regsim_return_path)},
        "source_snapshot_verified_file_count": verified_files,
        "frozen_config_hashes": {
            "live_config": sha256(source.frozen_live_config_path),
            "strategy01": sha256(source.frozen_strategy_config_path),
        },
        "current_generic_runtime_inputs": {
            "paths_config": {"path": str(paths_config_path), "sha256": sha256(paths_config_path), "loaded": load_config_file("paths")},
            "benchmark_config": {
                "path": str(benchmark_config_path),
                "sha256": sha256(benchmark_config_path),
                "runtime_note": "Generic strict backtest resolves execution price fields from this current config; passed buy_twap_col/sell_twap_col are not consumed.",
            },
            "fees_config": {
                "path": str(fees_config_path),
                "sha256": sha256(fees_config_path),
                "runtime_note": "Generic strict backtest reads this current config directly.",
            },
            "fees": {"buy_bps": buy_bps, "sell_bps": sell_bps, "source": fee_source},
        },
        "plan": PLAN,
        "created_at_utc": datetime.now(timezone.utc),
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
    }
    (run_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    weights = build_daily_weights(main_daily)
    weights.to_csv(run_root / "daily_weights.csv", index=False)
    diagnostics = _weight_diagnostics(weights)
    diagnostics.to_csv(run_root / "weight_diagnostics.csv", index=False)

    status: dict[str, Any] = {
        "status": "running_identity_baseline",
        "run_root": str(run_root),
        "main_days": int(len(main_daily)),
        "source_snapshot_verified_file_count": verified_files,
        "started_at_utc": datetime.now(timezone.utc),
    }
    _write_status(status_path, status)
    baseline_out, baseline_cfg = _run_generic_backtest(
        source=source,
        score_root=source.score_root / "Regsim",
        run_root=run_root,
        batch_id="score_level_v1_identity_frozen_regsim",
    )
    baseline = _load_generic_daily(baseline_out)
    comparison, identity = compare_identity_parity(frozen_regsim, baseline)
    identity["generic_backtest_output"] = str(baseline_out)
    comparison.to_csv(run_root / "identity_regsim_parity.csv", index=False)
    configs: dict[str, Any] = {"Regsim": baseline_cfg}
    status["identity_regsim_parity"] = identity
    if identity["status"] != "passed":
        status["status"] = "blocked_nonparity"
        status["completed_at_utc"] = datetime.now(timezone.utc)
        _write_status(status_path, status)
        pd.DataFrame().to_csv(run_root / "summary_metrics.csv", index=False)
        pd.DataFrame().to_csv(run_root / "paired_vs_regsim.csv", index=False)
        pd.DataFrame().to_csv(run_root / "daily_score_level_returns.csv", index=False)
        (run_root / "generic_backtest_configs.json").write_text(
            json.dumps(configs, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        _write_results(run_root=run_root, status=status, summary=pd.DataFrame(), paired=pd.DataFrame(), diagnostics=diagnostics)
        return status

    status["status"] = "writing_fused_scores"
    _write_status(status_path, status)
    score_roots, score_audit = write_fused_score_inputs(source, weights, run_root=run_root)
    score_audit.to_csv(run_root / "daily_rank_score_audit.csv", index=False)
    daily_frames = [baseline.assign(strategy="Regsim")]
    summary_rows = [{"strategy": "Regsim", "backtest_output": str(baseline_out), **_portfolio_metrics(baseline)}]
    paired_rows: list[dict[str, object]] = []
    for variant in SCORE_VARIANTS:
        status["status"] = f"running_{variant}"
        _write_status(status_path, status)
        out_dir, cfg = _run_generic_backtest(
            source=source,
            score_root=score_roots[variant],
            run_root=run_root,
            batch_id=f"score_level_v1_{variant}",
        )
        configs[variant] = cfg
        daily = _load_generic_daily(out_dir)
        if len(daily) != EXPECTED_MAIN_DAYS or daily["score_day"].tolist() != frozen_regsim["score_day"].tolist():
            raise RuntimeError(f"generic score-level {variant} did not cover the locked main date window")
        daily_frames.append(daily.assign(strategy=variant))
        summary_rows.append({"strategy": variant, "backtest_output": str(out_dir), **_portfolio_metrics(daily)})
        paired_rows.append(_paired_metrics(baseline, daily, strategy=variant))
    daily_score_level = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    daily_score_level.to_csv(run_root / "daily_score_level_returns.csv", index=False)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    paired.to_csv(run_root / "paired_vs_regsim.csv", index=False)
    (run_root / "generic_backtest_configs.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    status["status"] = "completed"
    status["score_level_days_by_strategy"] = {str(row["strategy"]): int(row["days"]) for row in summary_rows}
    status["completed_at_utc"] = datetime.now(timezone.utc)
    _write_status(status_path, status)
    _write_results(run_root=run_root, status=status, summary=summary, paired=paired, diagnostics=diagnostics)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = run_replay(source_run=args.source_run, output_root=args.output_root, run_name=str(args.run_name))
    print("OUTPUT_ROOT", status["run_root"])
    print(json.dumps(status, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":  # pragma: no cover - command line entrypoint
    main()
