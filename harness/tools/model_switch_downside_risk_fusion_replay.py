"""Research-only causal downside-risk score-level fusion replay.

This is a deliberately separate experiment from the existing equal-rank and
Ridge-utility score fusion.  It does not change a live score, configuration,
scheduler, database, state, or result root.

The pre-registered variants are:

* frozen Regsim raw-score execution identity baseline;
* equal-weight fusion of the three same-day within-universe percentile ranks;
* causal EWMA downside-risk-parity rank fusion.

For a score day ``t``, the risk-parity vector uses only the three standalone
candidate returns strictly before ``t``.  For model ``i``:

``d_i,t = sqrt(EWMA_lambda(min(r_i,s, 0)^2, s < t))``

and ``w_i,t`` is proportional to ``1 / d_i,t``.  The first 20 score days are
exactly equal weight.  Lambda=0.94 is a fixed daily RiskMetrics-style decay,
not the outcome of a parameter search.  There are no thresholds, guards,
Champion roles, LCBs, clipping, or score-gap rules.

Each score tree is sent through the existing generic backtest runtime, so the
unchanged o_0005 filter, market masks, Top20 rule, turnover, cost, benchmark,
and strict cycle-return accounting are applied after fusion.  The frozen
Regsim score identity must reproduce the frozen Regsim return series exactly
before either candidate is executed.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import pandas as pd

from cbond_on.app.usecases import backtest_runtime
from cbond_on.config.loader import load_config_file
from cbond_on.core.config import resolve_config_file_path
from cbond_on.core.fees import load_fees_buy_sell_bps
from harness.tools import model_switch_score_level_fusion_replay as score_base


SOURCE_RUN = score_base.SOURCE_RUN
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\cbond_on\research_scratch\model_switch_downside_fusion_20260811"
)
DEFAULT_RUN_NAME = "run_main_20260811"

MODELS = score_base.MODELS
MODEL_RETURN_COLUMNS = score_base.MODEL_RETURN_COLUMNS
MAIN_START = score_base.MAIN_START
MAIN_END = score_base.MAIN_END
EXPECTED_MAIN_DAYS = score_base.EXPECTED_MAIN_DAYS
TOP_K = score_base.TOP_K
EPSILON = score_base.EPSILON

EQUAL_VARIANT = "equal_weight_rank_score_fusion"
DOWNSIDE_RISK_PARITY_VARIANT = "ewma_downside_risk_parity_rank_score_fusion"
SCORE_VARIANTS = (EQUAL_VARIANT, DOWNSIDE_RISK_PARITY_VARIANT)

# Fixed before the run.  0.94 is the conventional daily RiskMetrics decay.
EWMA_LAMBDA = 0.94
MIN_HISTORY_DAYS = 20

PLAN = {
    "run_class": "research_only_score_level_downside_risk_fusion_v1",
    "source_run": str(SOURCE_RUN),
    "main_window": {
        "start": str(MAIN_START),
        "end": str(MAIN_END),
        "complete_metadata_days": EXPECTED_MAIN_DAYS,
    },
    "baseline": "frozen Regsim raw scores through generic backtest runtime; exact frozen-return identity required",
    "score_normalization": "each model's full shared frozen same-day score universe is percentile-ranked before fusion",
    "equal_weight": [1.0 / len(MODELS)] * len(MODELS),
    "downside_risk_parity": {
        "return_target": 0.0,
        "semivariance": "EWMA of min(standalone_model_day_return, 0)^2",
        "ewma_lambda": EWMA_LAMBDA,
        "history": "all available complete main-window rows strictly before score_day",
        "weights": "inverse downside deviation, normalized to simplex",
        "warmup": f"first {MIN_HISTORY_DAYS} score days exactly equal weight",
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
        "utility_tilt",
    ],
    "execution": "existing generic backtest_runtime; original o_0005, strategy01 Top20, cost, mask, benchmark, and cycle-return logic",
}


def _json_default(value: object) -> object:
    return score_base._json_default(value)


def sha256(path: Path) -> str:
    return score_base.sha256(path)


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _assert_output_root(path: Path) -> Path:
    """Reject every write target outside this experiment's dedicated root."""

    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"downside-risk research output must stay under {allowed}: {resolved}") from exc
    return resolved


def equal_weights() -> np.ndarray:
    return np.full(len(MODELS), 1.0 / len(MODELS), dtype=float)


def _ewma_observation_weights(length: int) -> np.ndarray:
    """Return normalized weights with the most recent observation largest."""

    if length <= 0:
        raise ValueError("EWMA history length must be positive")
    exponents = np.arange(length - 1, -1, -1, dtype=float)
    values = np.power(EWMA_LAMBDA, exponents)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("EWMA observation weights are invalid")
    return values / total


def causal_ewma_downside_deviation(history_returns: np.ndarray) -> np.ndarray:
    """Calculate the strictly-prior EWMA downside deviation per model.

    The caller supplies only prior standalone returns.  Positive returns enter
    as zero, which is the usual target-zero downside semivariance definition.
    This function deliberately has no data-dependent cap, floor, or clip.
    """

    values = np.asarray(history_returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(MODELS) or len(values) == 0:
        raise ValueError("downside-risk history must be a non-empty N-by-model matrix")
    if not np.isfinite(values).all():
        raise ValueError("non-finite standalone return in downside-risk history")
    negative_square = np.square(np.minimum(values, 0.0))
    weights = _ewma_observation_weights(len(values))
    semivariance = np.average(negative_square, axis=0, weights=weights)
    deviation = np.sqrt(semivariance)
    if not np.isfinite(deviation).all() or np.any(deviation <= 0.0):
        raise RuntimeError("downside-risk history has undefined non-positive downside deviation")
    return deviation.astype(float)


def inverse_downside_risk_parity_weights(downside_deviation: Sequence[float]) -> np.ndarray:
    """Normalize inverse downside deviations to a positive simplex vector."""

    deviation = np.asarray(downside_deviation, dtype=float)
    if deviation.shape != (len(MODELS),) or not np.isfinite(deviation).all() or np.any(deviation <= 0.0):
        raise ValueError("downside deviations must be positive and finite for every model")
    inverse = 1.0 / deviation
    weights = inverse / float(inverse.sum())
    if not np.isfinite(weights).all() or np.any(weights <= 0.0) or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12):
        raise RuntimeError("inverse downside-risk parity produced an invalid simplex")
    return weights.astype(float)


def build_daily_weights(main_daily: pd.DataFrame) -> pd.DataFrame:
    """Build both score-fusion weights without reading a same/future return.

    On score-day row ``i`` only ``frame.iloc[:i]`` is passed to the EWMA
    calculation.  The `history_end` audit field makes this explicit for every
    row and focused tests poison future returns to assert the causal boundary.
    """

    frame = main_daily.copy().sort_values("score_day").reset_index(drop=True)
    score_base._strictly_increasing_days(frame["score_day"], label="downside-risk weight input score_day")
    return_columns = [MODEL_RETURN_COLUMNS[model] for model in MODELS]
    missing = sorted(set(return_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"downside-risk weight input lacks standalone return columns: {missing}")
    for column in return_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[return_columns].to_numpy(dtype=float)).all():
        raise ValueError("downside-risk main return frame contains non-finite standalone return")

    rows: list[dict[str, object]] = []
    for index, current in frame.iterrows():
        prior = frame.iloc[:index]
        history = prior[return_columns].to_numpy(dtype=float)
        equal = equal_weights()
        if len(prior) < MIN_HISTORY_DAYS:
            risk_parity = equal.copy()
            risk_reason = "warmup_equal_weight"
            deviation = np.full(len(MODELS), np.nan, dtype=float)
        else:
            deviation = causal_ewma_downside_deviation(history)
            risk_parity = inverse_downside_risk_parity_weights(deviation)
            risk_reason = "causal_ewma_inverse_downside_deviation"

        for label, values in ((EQUAL_VARIANT, equal), (DOWNSIDE_RISK_PARITY_VARIANT, risk_parity)):
            if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-12) or np.any(values < -EPSILON):
                raise RuntimeError(f"invalid {label} weights on {current['score_day']}")
        record: dict[str, object] = {
            "score_day": current["score_day"],
            "history_return_rows": int(len(prior)),
            "history_start": prior["score_day"].iloc[0] if len(prior) else None,
            "history_end": prior["score_day"].iloc[-1] if len(prior) else None,
            "downside_risk_reason": risk_reason,
            "ewma_lambda": EWMA_LAMBDA,
            **{
                f"realized_return_{model}": float(current[MODEL_RETURN_COLUMNS[model]])
                for model in MODELS
            },
        }
        for position, model in enumerate(MODELS):
            record[f"downside_deviation_{model}"] = float(deviation[position]) if np.isfinite(deviation[position]) else np.nan
        for variant, values in ((EQUAL_VARIANT, equal), (DOWNSIDE_RISK_PARITY_VARIANT, risk_parity)):
            record[f"effective_models_{variant}"] = float(1.0 / np.sum(np.square(values)))
            for position, model in enumerate(MODELS):
                record[f"weight_{variant}_{model}"] = float(values[position])
        rows.append(record)
    return pd.DataFrame(rows)


def _codes_digest(codes: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest()


def write_fused_score_inputs(
    source: score_base.SourceInputs,
    weights: pd.DataFrame,
    *,
    run_root: Path,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """Materialise only the pre-registered isolated daily score trees."""

    score_roots = {variant: run_root / "fused_score_inputs" / variant for variant in SCORE_VARIANTS}
    audit_rows: list[dict[str, object]] = []
    for _, row in weights.iterrows():
        score_day = row["score_day"]
        score_by_model = {model: score_base._load_frozen_score(source, model, score_day) for model in MODELS}
        code_order = sorted(score_by_model[MODELS[0]].index.astype(str).tolist())
        for variant in SCORE_VARIANTS:
            values = [float(row[f"weight_{variant}_{model}"]) for model in MODELS]
            blended = score_base.rank_blend_scores(score_by_model, values, score_day=score_day)
            output = score_base._score_path(score_roots[variant], score_day)
            _assert_output_root(output)
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


def build_generic_backtest_config(
    *,
    source: score_base.SourceInputs,
    score_root: Path,
    output_root: Path,
    batch_id: str,
) -> dict[str, Any]:
    """Build an in-memory generic backtest config without touching repository config."""

    _assert_output_root(output_root)
    live = score_base._read_json5(source.frozen_live_config_path)
    strategy = score_base._read_json5(source.frozen_strategy_config_path)
    score_base._validate_frozen_execution_contract(live, strategy)
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
        # These are retained for audit; strict generic runtime currently resolves
        # the actual fields via its current benchmark configuration.
        "buy_twap_col": str(output.get("buy_twap_col", "twap_1442_1457")),
        "sell_twap_col": str(output.get("sell_twap_col", "twap_0930_0939")),
        "allowlist": dict(live["allowlist"]),
        "execution_lag_trading_days": 0,
        "freeze_signal_universe": False,
        "output_root": str(output_root),
    }


def _run_generic_backtest(
    *,
    source: score_base.SourceInputs,
    score_root: Path,
    run_root: Path,
    batch_id: str,
) -> tuple[Path, dict[str, Any]]:
    output_root = _assert_output_root(run_root / "generic_backtests")
    cfg = build_generic_backtest_config(
        source=source,
        score_root=score_root,
        output_root=output_root,
        batch_id=batch_id,
    )
    result = backtest_runtime.run(start=MAIN_START, end=MAIN_END, cfg=cfg)
    return result.out_dir, cfg


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
                **{f"min_weight_{model}": float(matrix[:, index].min()) for index, model in enumerate(MODELS)},
                **{f"max_weight_{model}": float(matrix[:, index].max()) for index, model in enumerate(MODELS)},
            }
        )
    return pd.DataFrame(records)


def _daily_drawdown(daily: pd.DataFrame, *, strategy: str) -> pd.DataFrame:
    frame = daily[["score_day", "day_return"]].copy().sort_values("score_day").reset_index(drop=True)
    values = frame["day_return"].to_numpy(dtype=float)
    nav = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(nav)
    frame["strategy"] = strategy
    frame["nav"] = nav
    frame["running_peak"] = peaks
    frame["drawdown"] = nav / peaks - 1.0
    return frame


def _drawdown_summary(drawdown: pd.DataFrame, *, strategy: str) -> dict[str, object]:
    if drawdown.empty:
        raise ValueError("cannot summarize empty drawdown history")
    trough_index = int(drawdown["drawdown"].to_numpy(dtype=float).argmin())
    nav = drawdown["nav"].to_numpy(dtype=float)
    peak_index = int(nav[: trough_index + 1].argmax())
    peak_nav = float(nav[peak_index])
    recover_indices = np.flatnonzero(nav[trough_index:] >= peak_nav)
    recovery_index = int(trough_index + recover_indices[0]) if len(recover_indices) else None
    return {
        "strategy": strategy,
        "max_drawdown": float(drawdown.loc[trough_index, "drawdown"]),
        "peak_score_day": str(drawdown.loc[peak_index, "score_day"]),
        "trough_score_day": str(drawdown.loc[trough_index, "score_day"]),
        "peak_to_trough_score_days": int(trough_index - peak_index),
        "recovered_within_window": recovery_index is not None,
        "recovery_score_day": str(drawdown.loc[recovery_index, "score_day"]) if recovery_index is not None else None,
        "trough_to_recovery_score_days": int(recovery_index - trough_index) if recovery_index is not None else None,
    }


def _relative_path_diagnostics(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    strategy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Audit realised path risk versus Regsim without using it in weights.

    This is diagnostic-only and is calculated *after* the candidate generic
    replay completes.  It cannot feed back into the pre-registered risk-parity
    allocation.  Rolling relative returns use compounded candidate versus
    baseline returns over a fixed 5- or 20-executed-day trailing window.
    """

    joined = baseline.merge(
        candidate,
        on="score_day",
        how="inner",
        suffixes=("_regsim", "_candidate"),
        validate="one_to_one",
    ).sort_values("score_day").reset_index(drop=True)
    if len(joined) != len(baseline) or len(joined) != len(candidate):
        raise ValueError(f"relative-path {strategy} has an unaligned return history")
    regsim = joined["day_return_regsim"].to_numpy(dtype=float)
    selected = joined["day_return_candidate"].to_numpy(dtype=float)
    if not np.isfinite(regsim).all() or not np.isfinite(selected).all():
        raise ValueError("relative-path inputs contain non-finite return")
    regsim_nav = np.cumprod(1.0 + regsim)
    candidate_nav = np.cumprod(1.0 + selected)
    relative_nav = candidate_nav / regsim_nav
    relative_peak = np.maximum.accumulate(relative_nav)
    relative_drawdown = relative_nav / relative_peak - 1.0
    output = pd.DataFrame(
        {
            "score_day": joined["score_day"],
            "strategy": strategy,
            "regsim_day_return": regsim,
            "candidate_day_return": selected,
            "day_return_delta": selected - regsim,
            "regsim_nav": regsim_nav,
            "candidate_nav": candidate_nav,
            "relative_nav_candidate_over_regsim": relative_nav,
            "relative_running_peak": relative_peak,
            "relative_drawdown": relative_drawdown,
        }
    )
    for window in (5, 20):
        regsim_window = pd.Series(1.0 + regsim).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        candidate_window = pd.Series(1.0 + selected).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        output[f"regsim_{window}d_return"] = regsim_window.to_numpy(dtype=float)
        output[f"candidate_{window}d_return"] = candidate_window.to_numpy(dtype=float)
        output[f"relative_{window}d_return"] = (1.0 + candidate_window).to_numpy(dtype=float) / (
            1.0 + regsim_window
        ).to_numpy(dtype=float) - 1.0

    trough_index = int(relative_drawdown.argmin())
    peak_index = int(relative_nav[: trough_index + 1].argmax())
    summary: dict[str, object] = {
        "strategy": strategy,
        "relative_mdd_candidate_over_regsim": float(relative_drawdown[trough_index]),
        "relative_mdd_peak_score_day": str(output.loc[peak_index, "score_day"]),
        "relative_mdd_trough_score_day": str(output.loc[trough_index, "score_day"]),
        "relative_mdd_peak_to_trough_score_days": int(trough_index - peak_index),
    }
    for window in (5, 20):
        column = f"relative_{window}d_return"
        available = output.loc[output[column].notna()]
        if available.empty:
            summary[f"worst_relative_{window}d_return"] = np.nan
            summary[f"worst_relative_{window}d_end_score_day"] = None
        else:
            worst_index = int(available[column].idxmin())
            summary[f"worst_relative_{window}d_return"] = float(output.loc[worst_index, column])
            summary[f"worst_relative_{window}d_end_score_day"] = str(output.loc[worst_index, "score_day"])
    return output, summary


def _markdown_table(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False, float_format="%.6f").rstrip() + "\n```"


def _write_results(
    *,
    run_root: Path,
    status: Mapping[str, Any],
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    diagnostics: pd.DataFrame,
    drawdown_summary: pd.DataFrame,
    relative_path_summary: pd.DataFrame,
) -> None:
    identity = status.get("identity_regsim_parity", {})
    if summary.empty:
        score_section = [
            "## Score-level results",
            "",
            "No candidate return is reported because the frozen Regsim generic-runtime identity check failed or the run was interrupted.",
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
            "",
            "## Drawdown-path diagnostics",
            "",
            _markdown_table(drawdown_summary),
            "",
            "## Relative path-risk diagnostics versus Regsim",
            "",
            _markdown_table(relative_path_summary),
        ]
    lines = [
        "# Research-only causal downside-risk score-level fusion v1",
        "",
        "## Locked contract",
        "",
        "- Main evaluation is `2024-05-08` through `2026-07-30` only: 541 score days with complete execution metadata.",
        "- Variants were fixed before results: frozen Regsim raw-score execution, equal-weight percentile-rank score fusion, and causal EWMA downside-risk-parity percentile-rank score fusion.",
        f"- The risk-parity formula uses target-zero downside deviation `sqrt(EWMA_{{lambda={EWMA_LAMBDA}}}(min(r,0)^2))` for each standalone model.  It consumes all available complete rows strictly before the score day and is equal weight for the first {MIN_HISTORY_DAYS} score days.",
        "- Each model is ranked on the complete shared frozen score universe for that day. The generic runtime then applies the unchanged o_0005 allowlist, market mask, Top20 strategy, 5% maximum weight, full turnover, costs, benchmark, and strict buy/sell cycle logic.",
        "- No BaseGap/Robust route, Champion preference, top-one/top-two gap, LCB, confidence threshold, veto, clip, utility tilt, or parameter sweep is used.",
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
        "- This is a volatility-style risk allocation across model score ranks, not a direct forecast of the next drawdown. A lower standalone downside deviation does not guarantee that its current score basket has lower next-day downside risk.",
        "- The current generic raw calendar contains 543 dates in this interval, while the frozen score/return contract has 541 executable score days. `2026-06-11` and `2026-06-12` have no frozen score file for any candidate, so the generic runtime records `missing_score` and skips them for every strategy. All tables are explicitly common 541 executed score days, not all 543 calendar dates.",
        "- The v5 source freezes score, candidate return, config, and T1430 state inputs, but not an immutable copy of daily raw execution prices or o_0005 pool snapshots. The generic runtime therefore reads the current configured DataHub raw/pool inputs. Baseline identity is checked before any fusion result is allowed, but it remains a historical replay boundary.",
        "- Strict generic backtesting resolves execution fields from the current benchmark configuration and reads the current fees configuration directly. Their paths and SHA-256 values at run start are retained in `run_manifest.json`; baseline identity proves current-contract equality, not immutable raw/pool/benchmark/fee injection.",
        "- The source state provenance is T1430 and is not strict-1429-certified. This result is research-only and cannot authorize a live change.",
        "",
        "## Artifacts",
        "",
        "- `run_manifest.json`, `run_status.json`, `generic_backtest_configs.json`, and `daily_weights.csv` retain the contract and causal audit.",
        "- `identity_regsim_parity.csv` is the exact precondition for candidate score execution.",
        "- `daily_downside_risk_inputs.csv` contains each prior-only downside-deviation input and all daily weights; `weight_diagnostics.csv` summarizes allocation concentration.",
        "- `daily_drawdown.csv` and `drawdown_summary.csv` retain the actual executed score-level drawdown paths, rather than an independent model-return proxy.",
        "- `relative_path_vs_regsim.csv` and `relative_path_summary.csv` report post-replay relative NAV, relative MDD, and fixed 5/20-day compounded loss windows versus Regsim. They are diagnostic only and never enter a candidate weight.",
        "- `fused_score_inputs/` contains isolated score files; `generic_backtests/` contains only scratch backtest outputs.",
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
    """Execute the locked 541-day research replay under the dedicated scratch root."""

    output_root = _assert_output_root(Path(output_root))
    source = score_base.load_source_inputs(source_run)
    verified_files = score_base.verify_source_snapshot(source)
    main_daily, frozen_regsim = score_base.load_main_source_daily(source)
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
    weights.to_csv(run_root / "daily_downside_risk_inputs.csv", index=False)
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
        batch_id="downside_risk_v1_identity_frozen_regsim",
    )
    baseline = score_base._load_generic_daily(baseline_out)
    comparison, identity = score_base.compare_identity_parity(frozen_regsim, baseline)
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
        pd.DataFrame().to_csv(run_root / "daily_downside_risk_returns.csv", index=False)
        pd.DataFrame().to_csv(run_root / "daily_drawdown.csv", index=False)
        pd.DataFrame().to_csv(run_root / "drawdown_summary.csv", index=False)
        pd.DataFrame().to_csv(run_root / "relative_path_vs_regsim.csv", index=False)
        pd.DataFrame().to_csv(run_root / "relative_path_summary.csv", index=False)
        (run_root / "generic_backtest_configs.json").write_text(
            json.dumps(configs, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        _write_results(
            run_root=run_root,
            status=status,
            summary=pd.DataFrame(),
            paired=pd.DataFrame(),
            diagnostics=diagnostics,
            drawdown_summary=pd.DataFrame(),
            relative_path_summary=pd.DataFrame(),
        )
        return status

    status["status"] = "writing_fused_scores"
    _write_status(status_path, status)
    score_roots, score_audit = write_fused_score_inputs(source, weights, run_root=run_root)
    score_audit.to_csv(run_root / "daily_rank_score_audit.csv", index=False)

    daily_frames = [baseline.assign(strategy="Regsim")]
    summary_rows = [{"strategy": "Regsim", "backtest_output": str(baseline_out), **score_base._portfolio_metrics(baseline)}]
    paired_rows: list[dict[str, object]] = []
    drawdown_frames = [_daily_drawdown(baseline, strategy="Regsim")]
    drawdown_rows = [_drawdown_summary(drawdown_frames[0], strategy="Regsim")]
    relative_path_frames: list[pd.DataFrame] = []
    relative_path_rows: list[dict[str, object]] = []
    for variant in SCORE_VARIANTS:
        status["status"] = f"running_{variant}"
        _write_status(status_path, status)
        out_dir, cfg = _run_generic_backtest(
            source=source,
            score_root=score_roots[variant],
            run_root=run_root,
            batch_id=f"downside_risk_v1_{variant}",
        )
        configs[variant] = cfg
        daily = score_base._load_generic_daily(out_dir)
        if len(daily) != EXPECTED_MAIN_DAYS or daily["score_day"].tolist() != frozen_regsim["score_day"].tolist():
            raise RuntimeError(f"generic downside-risk {variant} did not cover the locked main date window")
        daily_frames.append(daily.assign(strategy=variant))
        summary_rows.append({"strategy": variant, "backtest_output": str(out_dir), **score_base._portfolio_metrics(daily)})
        paired_rows.append(score_base._paired_metrics(baseline, daily, strategy=variant))
        drawdown = _daily_drawdown(daily, strategy=variant)
        drawdown_frames.append(drawdown)
        drawdown_rows.append(_drawdown_summary(drawdown, strategy=variant))
        relative_path, relative_summary = _relative_path_diagnostics(baseline, daily, strategy=variant)
        relative_path_frames.append(relative_path)
        relative_path_rows.append(relative_summary)

    daily_returns = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    daily_drawdown = pd.concat(drawdown_frames, ignore_index=True)
    drawdown_summary = pd.DataFrame(drawdown_rows)
    relative_path_diagnostics = pd.concat(relative_path_frames, ignore_index=True)
    relative_path_summary = pd.DataFrame(relative_path_rows)
    daily_returns.to_csv(run_root / "daily_downside_risk_returns.csv", index=False)
    summary.to_csv(run_root / "summary_metrics.csv", index=False)
    paired.to_csv(run_root / "paired_vs_regsim.csv", index=False)
    daily_drawdown.to_csv(run_root / "daily_drawdown.csv", index=False)
    drawdown_summary.to_csv(run_root / "drawdown_summary.csv", index=False)
    relative_path_diagnostics.to_csv(run_root / "relative_path_vs_regsim.csv", index=False)
    relative_path_summary.to_csv(run_root / "relative_path_summary.csv", index=False)
    (run_root / "generic_backtest_configs.json").write_text(
        json.dumps(configs, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    status["status"] = "completed"
    status["score_level_days_by_strategy"] = {str(row["strategy"]): int(row["days"]) for row in summary_rows}
    status["completed_at_utc"] = datetime.now(timezone.utc)
    _write_status(status_path, status)
    _write_results(
        run_root=run_root,
        status=status,
        summary=summary,
        paired=paired,
        diagnostics=diagnostics,
        drawdown_summary=drawdown_summary,
        relative_path_summary=relative_path_summary,
    )
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


if __name__ == "__main__":  # pragma: no cover - command-line entrypoint
    main()
