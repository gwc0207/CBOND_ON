"""Research-only, no-threshold continuous three-model fusion replay.

The source is a *frozen* no-threshold relative-utility replay.  Its causal
``state_score_geometry_lags`` forecast is converted to daily continuous model
weights.  This module deliberately has no interaction with the live scheduler,
live configuration, database, model states, or live trade lists.

Two result semantics are deliberately kept separate:

* ``sleeve_proxy`` is the weighted sum of the three existing standalone daily
  returns.  With the current full-liquidation contract it describes a
  three-sleeve portfolio allocation, but it is **not** a single Top20 book.
* ``score_level_rank_blend`` (optional) blends frozen score ranks, then invokes
  the existing generic backtest runtime to form one fixed-strategy Top20 book.
  It is run only after an identity-score parity replay proves that the generic
  runtime still reproduces the frozen Regsim return stream.

There are no confidence margins, LCBs, Champion preferences, BaseGap/Robust
routes, vetoes, or post-hoc parameter sweeps.  Every dynamic input for day T is
from the source forecast produced at T or realised returns strictly before T.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json5
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from cbond_on.app.usecases import backtest_runtime
from cbond_on.core.config import load_config_file, resolve_config_file_path


MODELS = ("Regsim", "Ensemble", "HL20")
SOURCE_VARIANT = "state_score_geometry_lags"
SOURCE_RUN = Path(
    r"D:\cbond_on\research_scratch\model_switch_relative_utility_20260811"
    r"\run_20260811_relative_utility_v5_with_ranker"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_dynamic_weight_20260811")

# All formula choices are fixed before looking at this run's result.
SCALE_LOOKBACK_DAYS = 360
EWM_HALFLIFE_DAYS = 20.0
SOFTMAX_TEMPERATURE = 1.0
SOFTMAX_SHRINKAGE_TO_EQUAL = 0.50
EPSILON = 1e-12

VARIANTS = (
    "equal_weight_three_models",
    "forecast_softmax_causal_scale",
    "forecast_softmax_causal_scale_shrink50",
    "history_relative_ewm20_softmax",
)

PLAN: dict[str, Any] = {
    "run_class": "research_only_no_threshold_dynamic_continuous_weight_fusion",
    "source_variant": SOURCE_VARIANT,
    "models": list(MODELS),
    "no_threshold": True,
    "no_lcb": True,
    "no_champion_preference": True,
    "no_basegap_route": True,
    "no_robust_route": True,
    "no_veto": True,
    "weight_variants": {
        "equal_weight_three_models": "w=(1/3,1/3,1/3)",
        "forecast_softmax_causal_scale": (
            "softmax(predicted_relative_utility / causal_cross_model_scale), "
            "temperature=1.0"
        ),
        "forecast_softmax_causal_scale_shrink50": (
            "0.50 * forecast_softmax + 0.50 * equal_weight"
        ),
        "history_relative_ewm20_softmax": (
            "softmax(EWM_halflife20(past_relative_returns) / causal_cross_model_scale), "
            "temperature=1.0"
        ),
    },
    "causal_scale": {
        "definition": "median daily RMS of cross-model demeaned standalone returns",
        "lookback_days": SCALE_LOOKBACK_DAYS,
        "strictly_before_score_day": True,
    },
    "forecast_warmup": "equal weight only when frozen forecast is unavailable",
    "history_control_warmup": "equal weight only when no prior realised return exists",
    "score_level_formula": "weighted average of each frozen model's within-day percentile rank",
    "score_level_execution": "existing generic backtest_runtime with frozen strategy and allowlist config",
}


@dataclass(frozen=True)
class SourceInputs:
    run_root: Path
    daily_path: Path
    manifest_path: Path
    snapshot_root: Path
    score_root: Path
    frozen_live_config_path: Path
    frozen_strategy_config_path: Path
    manifest: Mapping[str, Any]


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
    import subprocess

    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _assert_output_root(path: Path) -> Path:
    root = DEFAULT_OUTPUT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"research output must be under {root}: {resolved}") from exc
    return resolved


def _load_json5(path: Path) -> dict[str, Any]:
    value = json5.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def load_source_inputs(source_run: Path) -> SourceInputs:
    run_root = source_run.resolve()
    daily_path = run_root / "daily_selector_replay.csv"
    manifest_path = run_root / "run_manifest.json"
    snapshot_root = run_root / "input_snapshot"
    score_root = snapshot_root / "scores"
    frozen_live_config_path = snapshot_root / "configs" / "live_config.json5"
    frozen_strategy_config_path = snapshot_root / "configs" / "strategy01_config.json5"
    required = (
        daily_path,
        manifest_path,
        score_root,
        frozen_live_config_path,
        frozen_strategy_config_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("frozen relative-utility source is incomplete: " + "; ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"expected mapping manifest: {manifest_path}")
    return SourceInputs(
        run_root=run_root,
        daily_path=daily_path,
        manifest_path=manifest_path,
        snapshot_root=snapshot_root,
        score_root=score_root,
        frozen_live_config_path=frozen_live_config_path,
        frozen_strategy_config_path=frozen_strategy_config_path,
        manifest=manifest,
    )


def verify_source_snapshot(source: SourceInputs) -> int:
    """Verify every frozen input hash from the predecessor experiment."""

    files = source.manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source run_manifest has no frozen file entries")
    checked = 0
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("invalid source manifest file entry")
        path = Path(str(item.get("snapshot", "")))
        expected = str(item.get("sha256", ""))
        if not path.is_file():
            raise FileNotFoundError(f"source snapshot file missing: {path}")
        if len(expected) != 64 or sha256(path) != expected:
            raise RuntimeError(f"source snapshot hash mismatch: {path}")
        checked += 1
    return checked


def _required_columns() -> list[str]:
    return [
        "score_day",
        "variant",
        "prediction_ready",
        "training_end",
        "execution_metadata_complete",
        "basegap_return",
        *[f"realized_return_{model}" for model in MODELS],
        *[f"utility_{model.lower()}" for model in MODELS],
    ]


def load_source_daily(source: SourceInputs) -> pd.DataFrame:
    frame = pd.read_csv(source.daily_path)
    missing = sorted(set(_required_columns()) - set(frame.columns))
    if missing:
        raise KeyError(f"source daily selector replay is missing columns: {missing}")
    frame = frame.loc[frame["variant"].astype(str) == SOURCE_VARIANT].copy()
    if frame.empty:
        raise ValueError(f"source daily selector replay has no {SOURCE_VARIANT!r} rows")
    frame["score_day"] = pd.to_datetime(frame["score_day"], errors="coerce").dt.date
    frame = frame.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    if frame["score_day"].duplicated().any():
        raise ValueError("source utility forecast has duplicate score days")
    for model in MODELS:
        frame[f"realized_return_{model}"] = pd.to_numeric(
            frame[f"realized_return_{model}"], errors="coerce"
        )
        frame[f"utility_{model.lower()}"] = pd.to_numeric(
            frame[f"utility_{model.lower()}"], errors="coerce"
        )
    frame["basegap_return"] = pd.to_numeric(frame["basegap_return"], errors="coerce")
    raw_metadata_complete = frame["execution_metadata_complete"]
    if pd.api.types.is_bool_dtype(raw_metadata_complete):
        frame["execution_metadata_complete"] = raw_metadata_complete.astype(bool)
    else:
        normalized_metadata = raw_metadata_complete.astype(str).str.strip().str.lower()
        valid_metadata = normalized_metadata.isin(("true", "false", "1", "0"))
        if not valid_metadata.all():
            bad = raw_metadata_complete.loc[~valid_metadata].head(5).tolist()
            raise ValueError(f"invalid execution_metadata_complete values: {bad}")
        frame["execution_metadata_complete"] = normalized_metadata.isin(("true", "1"))
    if not np.isfinite(frame[[f"realized_return_{model}" for model in MODELS]].to_numpy(dtype=float)).all():
        raise ValueError("source daily selector replay has a non-finite standalone return")
    ready = frame["prediction_ready"].astype(bool)
    if ready.any():
        training_end = pd.to_datetime(frame.loc[ready, "training_end"], errors="coerce").dt.date
        if training_end.isna().any() or not (training_end.to_numpy() < frame.loc[ready, "score_day"].to_numpy()).all():
            raise ValueError("source forecast violates training_end < score_day")
        utilities = frame.loc[ready, [f"utility_{model.lower()}" for model in MODELS]].to_numpy(dtype=float)
        if not np.isfinite(utilities).all():
            raise ValueError("source marked a forecast ready while utilities are non-finite")
    return frame


def equal_weights() -> np.ndarray:
    return np.full(len(MODELS), 1.0 / len(MODELS), dtype=float)


def causal_cross_model_scale(history_returns: np.ndarray, *, lookback_days: int = SCALE_LOOKBACK_DAYS) -> float | None:
    """Return a strictly-past scale for dimensionless utility softmaxes."""

    values = np.asarray(history_returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(MODELS):
        raise ValueError("history_returns must have shape (n, 3)")
    if len(values) == 0:
        return None
    values = values[-int(lookback_days) :]
    if not np.isfinite(values).all():
        raise ValueError("history_returns must be finite")
    centered = values - values.mean(axis=1, keepdims=True)
    per_day_rms = np.sqrt(np.mean(centered**2, axis=1))
    scale = float(np.median(per_day_rms))
    return scale if math.isfinite(scale) and scale > EPSILON else None


def softmax_weights(
    utilities: np.ndarray,
    *,
    scale: float,
    temperature: float = SOFTMAX_TEMPERATURE,
) -> np.ndarray:
    """Continuous simplex weights; no winner rule or confidence condition."""

    utility = np.asarray(utilities, dtype=float)
    if utility.shape != (len(MODELS),) or not np.isfinite(utility).all():
        raise ValueError("utilities must be a finite vector with three models")
    if not math.isfinite(scale) or scale <= EPSILON:
        raise ValueError("scale must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = utility / (float(scale) * float(temperature))
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights /= float(weights.sum())
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("softmax produced invalid weights")
    return weights


def ewm_relative_utility(history_returns: np.ndarray, *, halflife_days: float = EWM_HALFLIFE_DAYS) -> np.ndarray | None:
    """Past-only realised relative-return EWM used as a non-model control."""

    values = np.asarray(history_returns, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(MODELS):
        raise ValueError("history_returns must have shape (n, 3)")
    if len(values) == 0:
        return None
    if not np.isfinite(values).all():
        raise ValueError("history_returns must be finite")
    relative = values - values.mean(axis=1, keepdims=True)
    utility = pd.DataFrame(relative, columns=MODELS).ewm(
        halflife=float(halflife_days), adjust=False, min_periods=1
    ).mean().iloc[-1].to_numpy(dtype=float)
    return utility if np.isfinite(utility).all() else None


def _utility_vector(current: pd.Series) -> np.ndarray:
    return current[[f"utility_{model.lower()}" for model in MODELS]].to_numpy(dtype=float)


def build_weight_replay(source_daily: pd.DataFrame) -> pd.DataFrame:
    """Produce every dynamic weight before consuming that row's realised return."""

    rows: list[dict[str, object]] = []
    return_columns = [f"realized_return_{model}" for model in MODELS]
    for position, current in source_daily.iterrows():
        # This slice is the entire anti-leakage boundary: only dates < score_day.
        history = source_daily.iloc[:position][return_columns].to_numpy(dtype=float)
        scale = causal_cross_model_scale(history)
        current_utility = _utility_vector(current)
        forecast_ready = bool(current["prediction_ready"]) and np.isfinite(current_utility).all() and scale is not None

        equal = equal_weights()
        if forecast_ready:
            forecast_softmax = softmax_weights(current_utility, scale=float(scale))
            forecast_reason = "frozen_causal_relative_utility_softmax"
        else:
            forecast_softmax = equal.copy()
            forecast_reason = "forecast_or_past_scale_unavailable_equal_weight"
        forecast_shrink = SOFTMAX_SHRINKAGE_TO_EQUAL * forecast_softmax + (1.0 - SOFTMAX_SHRINKAGE_TO_EQUAL) * equal

        ewm_utility = ewm_relative_utility(history)
        if ewm_utility is not None and scale is not None:
            history_ewm = softmax_weights(ewm_utility, scale=float(scale))
            history_reason = "past_relative_return_ewm20_softmax"
        else:
            history_ewm = equal.copy()
            history_reason = "past_return_or_scale_unavailable_equal_weight"

        realized = current[return_columns].to_numpy(dtype=float)
        record: dict[str, object] = {
            "score_day": current["score_day"],
            "source_forecast_ready": bool(current["prediction_ready"]),
            "source_training_end": current.get("training_end"),
            "causal_cross_model_scale": scale,
            "forecast_reason": forecast_reason,
            "history_ewm_reason": history_reason,
            **{f"realized_return_{model}": float(realized[index]) for index, model in enumerate(MODELS)},
            **{f"forecast_utility_{model}": float(current_utility[index]) if np.isfinite(current_utility[index]) else np.nan for index, model in enumerate(MODELS)},
            **{f"history_ewm_utility_{model}": float(ewm_utility[index]) if ewm_utility is not None else np.nan for index, model in enumerate(MODELS)},
        }
        weights_by_variant = {
            "equal_weight_three_models": equal,
            "forecast_softmax_causal_scale": forecast_softmax,
            "forecast_softmax_causal_scale_shrink50": forecast_shrink,
            "history_relative_ewm20_softmax": history_ewm,
        }
        for variant, weights in weights_by_variant.items():
            if not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12) or np.any(weights < -EPSILON):
                raise RuntimeError(f"invalid {variant} weights on {current['score_day']}")
            record[f"return_{variant}"] = float(np.dot(weights, realized))
            record[f"effective_models_{variant}"] = float(1.0 / np.sum(weights**2))
            for index, model in enumerate(MODELS):
                record[f"weight_{variant}_{model}"] = float(weights[index])
        rows.append(record)
    return pd.DataFrame(rows)


def _portfolio_metrics(returns: pd.Series, dates: pd.Series) -> dict[str, object]:
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
    if len(nonzero):
        result["sign_p_two_sided"] = float(stats.binomtest(wins, len(nonzero), 0.5, alternative="two-sided").pvalue)
    return result


def _slice_metrics(frame: pd.DataFrame, *, strategy: str, return_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, indices in zip(("early", "validation", "holdout"), np.array_split(np.arange(len(frame)), 3), strict=True):
        subset = frame.iloc[indices]
        rows.append({"strategy": strategy, "slice": name, **_portfolio_metrics(subset[return_col], subset["score_day"])})
    return rows


def _weight_summary(frame: pd.DataFrame, variant: str) -> dict[str, object]:
    matrix = frame[[f"weight_{variant}_{model}" for model in MODELS]].to_numpy(dtype=float)
    daily_turnover = 0.5 * np.abs(np.diff(matrix, axis=0)).sum(axis=1) if len(matrix) > 1 else np.empty(0)
    entropy = -np.sum(np.where(matrix > 0.0, matrix * np.log(matrix), 0.0), axis=1)
    return {
        "strategy": variant,
        "mean_effective_models": float(frame[f"effective_models_{variant}"].mean()),
        "min_effective_models": float(frame[f"effective_models_{variant}"].min()),
        "mean_weight_entropy": float(entropy.mean()),
        "min_weight_entropy": float(entropy.min()),
        "max_weight_entropy": float(entropy.max()),
        "mean_daily_weight_turnover": float(daily_turnover.mean()) if len(daily_turnover) else 0.0,
        "total_weight_turnover": float(daily_turnover.sum()) if len(daily_turnover) else 0.0,
        **{f"mean_weight_{model}": float(matrix[:, index].mean()) for index, model in enumerate(MODELS)},
        **{f"min_weight_{model}": float(matrix[:, index].min()) for index, model in enumerate(MODELS)},
        **{f"max_weight_{model}": float(matrix[:, index].max()) for index, model in enumerate(MODELS)},
    }


def _metadata_sensitivity(result: pd.DataFrame) -> pd.DataFrame:
    """Show all 547 rows and the 541-row complete-metadata execution subset."""

    complete = result["execution_metadata_complete"].astype(bool)
    rows: list[dict[str, object]] = []
    strategies = {
        "Regsim": "return_regsim",
        "direct_basegap": "return_direct_basegap",
        **{variant: f"return_{variant}" for variant in VARIANTS},
    }
    for scope, mask in (
        ("all_return_rows", pd.Series(True, index=result.index)),
        ("complete_execution_metadata_only", complete),
    ):
        subset = result.loc[mask]
        excluded = int((~mask).sum())
        for strategy, column in strategies.items():
            rows.append(
                {
                    "scope": scope,
                    "strategy": strategy,
                    "excluded_metadata_days": excluded,
                    **_portfolio_metrics(subset[column], subset["score_day"]),
                }
            )
    return pd.DataFrame(rows)


def summarise_sleeve_proxy(weights: pd.DataFrame, source_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = weights.copy()
    result["return_regsim"] = source_daily["realized_return_Regsim"].to_numpy(dtype=float)
    result["return_direct_basegap"] = source_daily["basegap_return"].to_numpy(dtype=float)
    result["execution_metadata_complete"] = source_daily["execution_metadata_complete"].to_numpy(dtype=bool)
    rows = [
        {"evaluation_kind": "standalone_reference", "strategy": "Regsim", **_portfolio_metrics(result["return_regsim"], result["score_day"])},
        {"evaluation_kind": "selector_reference", "strategy": "direct_basegap", **_portfolio_metrics(result["return_direct_basegap"], result["score_day"])},
    ]
    paired = [
        {"evaluation_kind": "selector_reference", "strategy": "direct_basegap", "baseline": "Regsim", **_paired_metrics(result["return_direct_basegap"] - result["return_regsim"])},
    ]
    slices = [
        *_slice_metrics(result, strategy="Regsim", return_col="return_regsim"),
        *_slice_metrics(result, strategy="direct_basegap", return_col="return_direct_basegap"),
    ]
    weight_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        return_col = f"return_{variant}"
        rows.append({"evaluation_kind": "sleeve_proxy", "strategy": variant, **_portfolio_metrics(result[return_col], result["score_day"])})
        paired.append({"evaluation_kind": "sleeve_proxy", "strategy": variant, "baseline": "Regsim", **_paired_metrics(result[return_col] - result["return_regsim"])})
        paired.append({"evaluation_kind": "sleeve_proxy", "strategy": variant, "baseline": "direct_basegap", **_paired_metrics(result[return_col] - result["return_direct_basegap"])})
        slices.extend(_slice_metrics(result, strategy=variant, return_col=return_col))
        weight_rows.append(_weight_summary(result, variant))
    return (
        pd.DataFrame(rows),
        pd.DataFrame(paired),
        pd.DataFrame(slices),
        pd.DataFrame(weight_rows),
        _metadata_sensitivity(result),
    )


def _score_path(score_root: Path, score_day: object) -> Path:
    timestamp = pd.Timestamp(score_day)
    return score_root / timestamp.strftime("%Y-%m") / f"{timestamp:%Y-%m-%d}.csv"


def _load_frozen_score(source: SourceInputs, model: str, score_day: object) -> pd.Series:
    path = _score_path(source.score_root / model, score_day)
    if not path.is_file():
        raise FileNotFoundError(f"frozen score missing: {path}")
    frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    expected = pd.Timestamp(score_day).date()
    reported = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    if reported.isna().any() or not reported.eq(expected).all():
        raise ValueError(f"frozen score date mismatch in {path}; expected {expected}")
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["code", "score"])
    frame = frame.loc[frame["code"].ne("")]
    if frame["code"].duplicated().any():
        raise ValueError(f"frozen score has duplicate codes: {path}")
    if len(frame) < 21:
        raise ValueError(f"insufficient frozen score rows: {path}")
    return frame.set_index("code")["score"].astype(float)


def rank_blend_scores(score_by_model: Mapping[str, pd.Series], weights: Sequence[float], *, score_day: object) -> pd.DataFrame:
    """Blend rank percentiles on an identical frozen score universe."""

    weight = np.asarray(weights, dtype=float)
    if weight.shape != (len(MODELS),) or not np.isfinite(weight).all() or np.any(weight < -EPSILON):
        raise ValueError("rank blend weights must be a finite non-negative length-three vector")
    if not math.isclose(float(weight.sum()), 1.0, abs_tol=1e-12):
        raise ValueError("rank blend weights must sum to one")
    reference_codes: set[str] | None = None
    for model in MODELS:
        series = score_by_model.get(model)
        if not isinstance(series, pd.Series):
            raise KeyError(f"missing score series: {model}")
        codes = set(series.index.astype(str))
        if reference_codes is None:
            reference_codes = codes
        elif codes != reference_codes:
            raise ValueError("rank-level blending requires identical model score universes")
    assert reference_codes is not None
    codes = sorted(reference_codes)
    ranked = pd.DataFrame({model: score_by_model[model].reindex(codes) for model in MODELS})
    percentile = ranked.rank(method="average", pct=True, ascending=True)
    blend = percentile.to_numpy(dtype=float) @ weight
    output = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(score_day).strftime("%Y-%m-%d"),
            "code": codes,
            "score": blend,
        }
    )
    if not np.isfinite(output["score"].to_numpy(dtype=float)).all():
        raise RuntimeError("rank-level blend contains non-finite values")
    return output


def write_rank_blend_scores(source: SourceInputs, weights: pd.DataFrame, *, run_root: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    """Write isolated daily rank-blend scores for each predeclared variant."""

    score_roots = {variant: run_root / "score_level" / "score_inputs" / variant for variant in VARIANTS}
    audit_rows: list[dict[str, object]] = []
    for _, row in weights.iterrows():
        score_day = row["score_day"]
        score_by_model = {model: _load_frozen_score(source, model, score_day) for model in MODELS}
        for variant, root in score_roots.items():
            values = [float(row[f"weight_{variant}_{model}"]) for model in MODELS]
            blended = rank_blend_scores(score_by_model, values, score_day=score_day)
            out_path = _score_path(root, score_day)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            blended.to_csv(out_path, index=False)
            audit_rows.append(
                {
                    "score_day": score_day,
                    "variant": variant,
                    "score_rows": int(len(blended)),
                    "score_sha256": sha256(out_path),
                    **{f"weight_{model}": values[index] for index, model in enumerate(MODELS)},
                }
            )
    return score_roots, pd.DataFrame(audit_rows)


def _score_level_config(
    *,
    score_root: Path,
    output_root: Path,
    batch_id: str,
    frozen_live_config: Mapping[str, Any],
    frozen_strategy_config: Mapping[str, Any],
    start_day: date,
    end_day: date,
) -> dict[str, Any]:
    allowlist = frozen_live_config.get("allowlist")
    output = frozen_live_config.get("output")
    if not isinstance(allowlist, Mapping) or not isinstance(output, Mapping):
        raise ValueError("frozen live config lacks allowlist/output mappings")
    return {
        "start": str(start_day),
        "end": str(end_day),
        "batch_id": batch_id,
        "score_source": {"score_root": str(score_root)},
        "strategy_id": "strategy01_topk_turnover",
        "strategy_config": dict(frozen_strategy_config),
        "buy_twap_col": str(output.get("buy_twap_col", "twap_1442_1457")),
        "sell_twap_col": str(output.get("sell_twap_col", "twap_0930_0939")),
        "allowlist": dict(allowlist),
        "execution_lag_trading_days": 0,
        "freeze_signal_universe": False,
        "output_root": str(output_root),
    }


def _run_generic_score_backtest(
    *,
    score_root: Path,
    output_root: Path,
    batch_id: str,
    frozen_live_config: Mapping[str, Any],
    frozen_strategy_config: Mapping[str, Any],
    start_day: date,
    end_day: date,
) -> Path:
    cfg = _score_level_config(
        score_root=score_root,
        output_root=output_root,
        batch_id=batch_id,
        frozen_live_config=frozen_live_config,
        frozen_strategy_config=frozen_strategy_config,
        start_day=start_day,
        end_day=end_day,
    )
    result = backtest_runtime.run(start=start_day, end=end_day, cfg=cfg)
    return result.out_dir


def _load_backtest_returns(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "daily_returns.csv"
    if not path.is_file():
        raise FileNotFoundError(f"generic score-level backtest wrote no daily returns: {path}")
    frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["day_return"] = pd.to_numeric(frame["day_return"], errors="coerce")
    frame = frame.dropna(subset=["score_day", "day_return"]).drop_duplicates("score_day", keep="last")
    return frame[["score_day", "day_return"]].sort_values("score_day").reset_index(drop=True)


def _identity_parity(
    *,
    source: SourceInputs,
    source_daily: pd.DataFrame,
    run_root: Path,
    frozen_live_config: Mapping[str, Any],
    frozen_strategy_config: Mapping[str, Any],
) -> tuple[bool, dict[str, object], Path]:
    """Only permit score-level results after generic Regsim parity is exact."""

    start_day = source_daily["score_day"].iloc[0]
    end_day = source_daily["score_day"].iloc[-1]
    out_dir = _run_generic_score_backtest(
        score_root=source.score_root / "Regsim",
        output_root=run_root / "score_level" / "backtest_outputs",
        batch_id="identity_frozen_regsim_parity",
        frozen_live_config=frozen_live_config,
        frozen_strategy_config=frozen_strategy_config,
        start_day=start_day,
        end_day=end_day,
    )
    actual = _load_backtest_returns(out_dir).rename(columns={"day_return": "generic_regsim_return"})
    expected = source_daily[["score_day", "realized_return_Regsim"]].rename(columns={"realized_return_Regsim": "frozen_regsim_return"})
    comparison = expected.merge(actual, on="score_day", how="outer", validate="one_to_one", indicator=True)
    matched = comparison.loc[comparison["_merge"].eq("both")].copy()
    matched["difference"] = matched["generic_regsim_return"] - matched["frozen_regsim_return"]
    max_abs = float(np.abs(matched["difference"]).max()) if len(matched) else float("inf")
    parity = bool(
        len(comparison) == len(expected)
        and comparison["_merge"].eq("both").all()
        and math.isfinite(max_abs)
        and max_abs <= 1e-12
    )
    detail = {
        "status": "passed" if parity else "failed",
        "expected_days": int(len(expected)),
        "generic_days": int(len(actual)),
        "aligned_days": int(len(matched)),
        "max_abs_return_difference": max_abs,
        "generic_backtest_output": str(out_dir),
    }
    comparison.to_csv(run_root / "score_level" / "identity_regsim_parity.csv", index=False)
    return parity, detail, out_dir


def run_score_level_replays(
    *,
    source: SourceInputs,
    source_daily: pd.DataFrame,
    weights: pd.DataFrame,
    run_root: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run verified single-book rank blends through the generic backtest runtime."""

    frozen_live = _load_json5(source.frozen_live_config_path)
    frozen_strategy = _load_json5(source.frozen_strategy_config_path)
    score_level_root = run_root / "score_level"
    score_level_root.mkdir(parents=True, exist_ok=True)
    parity, parity_detail, _ = _identity_parity(
        source=source,
        source_daily=source_daily,
        run_root=run_root,
        frozen_live_config=frozen_live,
        frozen_strategy_config=frozen_strategy,
    )
    status: dict[str, object] = {
        "evaluation_kind": "score_level_rank_blend_generic_backtest",
        "identity_regsim_parity": parity_detail,
        "status": "blocked_nonparity" if not parity else "running",
    }
    if not parity:
        (score_level_root / "score_level_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        return pd.DataFrame(), status

    score_roots, score_audit = write_rank_blend_scores(source, weights, run_root=run_root)
    score_audit.to_csv(score_level_root / "daily_rank_blend_audit.csv", index=False)
    start_day = source_daily["score_day"].iloc[0]
    end_day = source_daily["score_day"].iloc[-1]
    summary_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    for variant in VARIANTS:
        out_dir = _run_generic_score_backtest(
            score_root=score_roots[variant],
            output_root=score_level_root / "backtest_outputs",
            batch_id=f"rank_blend_{variant}",
            frozen_live_config=frozen_live,
            frozen_strategy_config=frozen_strategy,
            start_day=start_day,
            end_day=end_day,
        )
        daily = _load_backtest_returns(out_dir)
        daily["strategy"] = variant
        daily_frames.append(daily)
        summary_rows.append(
            {
                "evaluation_kind": "score_level_rank_blend_generic_backtest",
                "strategy": variant,
                "backtest_output": str(out_dir),
                **_portfolio_metrics(daily["day_return"], daily["score_day"]),
            }
        )
    daily_score_level = pd.concat(daily_frames, ignore_index=True)
    daily_score_level.to_csv(score_level_root / "daily_score_level_returns.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(score_level_root / "score_level_summary_metrics.csv", index=False)
    status["status"] = "completed"
    status["score_level_days_by_variant"] = {
        row["strategy"]: int(row["days"]) for row in summary_rows
    }
    (score_level_root / "score_level_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return summary, status


def _write_nav_plot(run_root: Path, sleeve_daily: pd.DataFrame) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(sleeve_daily["score_day"], np.cumprod(1.0 + sleeve_daily["return_regsim"]), label="Regsim", color="black", linewidth=2.0)
    ax.plot(sleeve_daily["score_day"], np.cumprod(1.0 + sleeve_daily["return_direct_basegap"]), label="direct BaseGap", color="#777777", linewidth=1.2)
    for variant in VARIANTS:
        ax.plot(sleeve_daily["score_day"], np.cumprod(1.0 + sleeve_daily[f"return_{variant}"]), label=variant, linewidth=1.2)
    ax.set_title("Research-only continuous three-model sleeve fusion")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_root / "sleeve_proxy_nav_compare.png", dpi=160)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame) -> str:
    return "```csv\n" + frame.to_csv(index=False, float_format="%.6f").rstrip() + "\n```"


def write_results(
    *,
    run_root: Path,
    sleeve_summary: pd.DataFrame,
    sleeve_paired: pd.DataFrame,
    sleeve_slices: pd.DataFrame,
    weight_summary: pd.DataFrame,
    metadata_sensitivity: pd.DataFrame,
    score_level_summary: pd.DataFrame,
    score_level_status: Mapping[str, Any] | None,
    manifest: Mapping[str, Any],
) -> None:
    score_section: list[str]
    if score_level_status is None:
        score_section = [
            "## Score-level rank blend",
            "",
            "Not requested in this run. No assertion about a single Top20 execution book is made.",
        ]
    elif score_level_summary.empty:
        score_section = [
            "## Score-level rank blend",
            "",
            "Blocked: the generic Regsim identity replay did not exactly reproduce frozen standalone returns.",
            "",
            "```json",
            json.dumps(score_level_status, ensure_ascii=False, indent=2, default=_json_default),
            "```",
        ]
    else:
        score_section = [
            "## Score-level rank blend",
            "",
            "These are a distinct, single-Top20-book execution replay. They must not be mixed with the sleeve proxy table.",
            "",
            _markdown_table(score_level_summary),
        ]
    text = [
        "# No-threshold continuous dynamic three-model fusion",
        "",
        "## Contract",
        "",
        "- Research-only. No live config, scheduler, database, model state, score root, trade list, `results/live`, `results/analysis`, or `results/backtest` was written.",
        "- Source forecasts are the frozen causal `state_score_geometry_lags` utility output. Every source ready row asserts `training_end < score_day`.",
        "- The formulas and all hyperparameters were predeclared: a causal 360-day scale, softmax temperature 1.0, 50% equal-weight shrinkage, and a 20-day EWM control.",
        "- On the frozen source's 126 unavailable forecast/state rows, forecast-derived weights are fixed to equal weight; this is disclosed input-availability handling, not a Regsim fallback or confidence threshold.",
        "- No margin, LCB, top-one/top-two gap, Champion role, BaseGap route, Robust route, veto, or parameter search is present.",
        "",
        "## Critical semantic boundary",
        "",
        "`sleeve_proxy` below is `sum_i(weight_i * standalone_day_return_i)`. It represents a fully liquidated multi-sleeve portfolio allocation only. It is not mathematically identical to blending scores then selecting one fixed Top20 book, so it is not an executable single-book live result.",
        "",
        "## Sleeve proxy summary",
        "",
        _markdown_table(sleeve_summary),
        "",
        "## Sleeve proxy paired daily delta",
        "",
        _markdown_table(sleeve_paired),
        "",
        "## Sleeve proxy chronological slices",
        "",
        _markdown_table(sleeve_slices),
        "",
        "## Continuous-weight diagnostics",
        "",
        _markdown_table(weight_summary),
        "",
        "## Execution-metadata sensitivity",
        "",
        "The main sleeve table uses all 547 aligned return rows. The second scope reports only the 541 rows with complete execution metadata; the six incomplete suffix rows are not silently treated as production-grade evidence.",
        "",
        _markdown_table(metadata_sensitivity),
        "",
        *score_section,
        "",
        "## Inputs and artifacts",
        "",
        "- `input_manifest.json`: source frozen-run paths, hashes, current generic-runtime config hashes, and contract.",
        "- `daily_dynamic_weights.csv`: daily weight and sleeve-return audit.",
        "- `sleeve_proxy_summary_metrics.csv`, `sleeve_proxy_paired_metrics.csv`, `sleeve_proxy_slices.csv`, `sleeve_proxy_metadata_sensitivity.csv`, `weight_diagnostics.csv`, `sleeve_proxy_nav_compare.png`.",
        "- If score-level was requested: `score_level/identity_regsim_parity.csv`, `score_level/score_level_status.json`, isolated score inputs, and generic-backtest output paths.",
        "",
        "## Caveats",
        "",
        "- The frozen T1430 state retains its existing not-strict-1429 provenance caveat.",
        "- The score-level branch can only establish a historical replay if its identity parity passes. It still has no forward/prospective evidence and does not authorize a live change.",
        "- Do not select a formula from this same result table; formula choices were deliberately fixed before execution.",
    ]
    (run_root / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def run_replay(
    *,
    source_run: Path,
    run_root: Path,
    run_score_level: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, Any] | None]:
    run_root = _assert_output_root(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    source = load_source_inputs(source_run)
    verified_files = verify_source_snapshot(source)
    source_daily = load_source_daily(source)

    run_root.mkdir(parents=True, exist_ok=False)
    weights = build_weight_replay(source_daily)
    sleeve_summary, sleeve_paired, sleeve_slices, weight_summary, metadata_sensitivity = summarise_sleeve_proxy(weights, source_daily)
    weights.to_csv(run_root / "daily_dynamic_weights.csv", index=False)
    sleeve_summary.to_csv(run_root / "sleeve_proxy_summary_metrics.csv", index=False)
    sleeve_paired.to_csv(run_root / "sleeve_proxy_paired_metrics.csv", index=False)
    sleeve_slices.to_csv(run_root / "sleeve_proxy_slices.csv", index=False)
    weight_summary.to_csv(run_root / "weight_diagnostics.csv", index=False)
    metadata_sensitivity.to_csv(run_root / "sleeve_proxy_metadata_sensitivity.csv", index=False)
    _write_nav_plot(run_root, weights.assign(return_regsim=source_daily["realized_return_Regsim"].to_numpy(dtype=float), return_direct_basegap=source_daily["basegap_return"].to_numpy(dtype=float)))

    paths_config_path = resolve_config_file_path("paths")
    manifest: dict[str, Any] = {
        "run_class": PLAN["run_class"],
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "live_config_written": False,
        "source_run": str(source.run_root),
        "source_daily_selector_replay": {"path": str(source.daily_path), "sha256": sha256(source.daily_path)},
        "source_run_manifest": {"path": str(source.manifest_path), "sha256": sha256(source.manifest_path)},
        "source_snapshot_verified_file_count": verified_files,
        "source_date_coverage": {"days": int(len(source_daily)), "start": str(source_daily["score_day"].iloc[0]), "end": str(source_daily["score_day"].iloc[-1])},
        "execution_metadata": {
            "all_return_rows": int(len(source_daily)),
            "complete_execution_metadata_rows": int(source_daily["execution_metadata_complete"].astype(bool).sum()),
            "incomplete_execution_metadata_rows": int((~source_daily["execution_metadata_complete"].astype(bool)).sum()),
        },
        "frozen_forecast_availability": {
            "ready_days": int(source_daily["prediction_ready"].astype(bool).sum()),
            "unavailable_days_fixed_equal_weight": int((~source_daily["prediction_ready"].astype(bool)).sum()),
        },
        "current_generic_runtime_paths_config": {"path": str(paths_config_path), "sha256": sha256(paths_config_path), "loaded": load_config_file("paths")},
        "plan": PLAN,
        "git": {"head": _safe_git(["rev-parse", "HEAD"]), "short_head": _safe_git(["rev-parse", "--short", "HEAD"]), "status_porcelain": _safe_git(["status", "--porcelain"])},
        "created_at_utc": datetime.now(timezone.utc),
    }
    (run_root / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    score_level_summary = pd.DataFrame()
    score_level_status: Mapping[str, Any] | None = None
    if run_score_level:
        score_level_summary, score_level_status = run_score_level_replays(
            source=source,
            source_daily=source_daily,
            weights=weights,
            run_root=run_root,
        )
    write_results(
        run_root=run_root,
        sleeve_summary=sleeve_summary,
        sleeve_paired=sleeve_paired,
        sleeve_slices=sleeve_slices,
        weight_summary=weight_summary,
        metadata_sensitivity=metadata_sensitivity,
        score_level_summary=score_level_summary,
        score_level_status=score_level_status,
        manifest=manifest,
    )
    return weights, sleeve_summary, score_level_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None, help="new leaf under the isolated research root")
    parser.add_argument(
        "--score-level",
        action="store_true",
        help="after exact frozen-Regsim parity, run generic single-book rank-blend replays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _assert_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_safe_git(['rev-parse', '--short', 'HEAD']) or 'nogit'}"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must be a single directory leaf")
    run_root = output_root / run_name
    _, summary, score_status = run_replay(source_run=args.source_run, run_root=run_root, run_score_level=bool(args.score_level))
    print(f"OUTPUT_ROOT {run_root}")
    print("SLEEVE_PROXY_SUMMARY")
    print(summary.to_string(index=False))
    if score_status is not None:
        print("SCORE_LEVEL_STATUS")
        print(json.dumps(score_status, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
