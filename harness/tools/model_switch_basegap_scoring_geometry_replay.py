"""Frozen, research-only replay of twelve BaseGap scoring-geometry variants.

This is deliberately a *selector shadow-return* experiment.  It is locked to
the frozen three-candidate live50 selector snapshot and to the completed
Stage-A r3 parameter-selection run.  It does not load live configuration,
call a scheduler, write a database, mutate a factor store, or emit an output
outside its dedicated ``research_scratch`` root.

All twelve variants keep the Stage-A parameters fixed:

* state scale: rolling z-score or rolling median/MAD;
* neighbour score aggregation: equal or fixed median-distance-half Gaussian;
* symmetric historical target: absolute return, return minus the three-model
  equal-weight return, or mean of the two pairwise return differences.

The ``zscore/equal/absolute`` member is a mandatory day-by-day parity gate
against Stage-A ``lb240_k60_trim20_lcb10``.  The other eleven variants are not
run unless that gate succeeds.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.infra.live.model_switch import _scoreopt_score_sample  # noqa: E402
from harness.tools.model_switch_basegap_tuning_replay import (  # noqa: E402
    DEFAULT_SOURCE_RUN,
    MODELS,
    SPLITS,
    _scope_frame,
    _scope_metrics,
    parity_audit,
    prepare_panel,
    sha256,
    validate_frozen_source,
)


DEFAULT_STAGE_A_RUN = Path(
    "D:/cbond_on/research_scratch/model_switch_basegap_tuning_20260811/"
    "run_stage_a_20260811_r3"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:\cbond_on\research_scratch\model_switch_basegap_scoring_geometry_20260811"
)

FIXED_LOOKBACK_DAYS = 240
FIXED_NEAREST_K = 60
FIXED_MIN_PERIODS = 60
FIXED_METRIC = "trim20_lcb10"
BASELINE_VARIANT = "zscore__equal__absolute"

STATE_STANDARDIZATIONS = ("zscore", "median_mad")
NEIGHBOR_WEIGHTINGS = ("equal", "median_half_gaussian")
SYMMETRIC_TARGETS = ("absolute", "relative_equal_weight", "pairwise_mean")


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, description: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be under {root}: {resolved}") from exc
    return resolved


def _assert_output_root(path: Path) -> Path:
    return _resolve_inside(path, DEFAULT_OUTPUT_ROOT, description="research output")


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _assert_output_root(output_root)
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _assert_output_root(root / name)
    if run_root == DEFAULT_OUTPUT_ROOT.resolve() or run_root.exists():
        raise FileExistsError(f"refusing to overwrite research output: {run_root}")
    return run_root


def _assert_locked_stage_a(path: Path) -> Path:
    resolved = path.resolve()
    expected = DEFAULT_STAGE_A_RUN.resolve()
    if resolved != expected:
        raise ValueError(f"Stage-A run is locked to {expected}, got {resolved}")
    required = ("run_status.json", "run_manifest.json", "daily_selector_replay.csv")
    for relative in required:
        if not (resolved / relative).is_file():
            raise FileNotFoundError(f"Stage-A r3 is incomplete; missing {relative}: {resolved}")
    return resolved


def preregistered_variants() -> list[dict[str, object]]:
    """Return the fixed 2 x 2 x 3 scoring-geometry family, in stable order."""

    rows: list[dict[str, object]] = []
    for standardization in STATE_STANDARDIZATIONS:
        for weighting in NEIGHBOR_WEIGHTINGS:
            for target in SYMMETRIC_TARGETS:
                variant = f"{standardization}__{weighting}__{target}"
                rows.append(
                    {
                        "variant": variant,
                        "state_standardization": standardization,
                        "neighbor_weighting": weighting,
                        "target": target,
                        "lookback_days": FIXED_LOOKBACK_DAYS,
                        "nearest_k": FIXED_NEAREST_K,
                        "min_periods": FIXED_MIN_PERIODS,
                        "metric": FIXED_METRIC,
                        "margin": 0.0,
                        "is_stage_a_parity_reference": variant == BASELINE_VARIANT,
                    }
                )
    if len(rows) != 12:
        raise AssertionError(f"expected exactly 12 preregistered variants, got {len(rows)}")
    if sum(bool(row["is_stage_a_parity_reference"]) for row in rows) != 1:
        raise AssertionError("the Stage-A parity reference must occur exactly once")
    return rows


def _standardised_distances(
    history_features: pd.DataFrame,
    current_features: pd.Series,
    *,
    standardization: str,
) -> pd.Series:
    """Return history-to-current Euclidean distances without using future rows."""

    if standardization == "zscore":
        # This spelling intentionally mirrors Stage-A's in-memory kernel so
        # that zscore/equal/absolute can serve as an exact parity reference.
        center = history_features.mean(axis=0)
        scale = history_features.std(axis=0).replace(0, math.nan).fillna(1.0)
    elif standardization == "median_mad":
        center = history_features.median(axis=0)
        mad = history_features.sub(center, axis=1).abs().median(axis=0)
        # 1.4826 makes the MAD comparable to a Gaussian standard deviation.
        scale = (1.4826 * mad).replace(0, math.nan).fillna(1.0)
    else:
        raise ValueError(f"unsupported state standardization: {standardization}")
    history_z = (history_features - center) / scale
    current_z = (current_features - center) / scale
    distances = (history_z.sub(current_z, axis=1).pow(2).sum(axis=1)).pow(0.5)
    if not np.isfinite(distances.to_numpy(dtype=float)).all():
        raise ValueError("non-finite BaseGap state distance")
    return distances


def _median_half_gaussian_weights(distances: pd.Series) -> pd.Series:
    """Fixed Gaussian kernel whose selected-neighbour median has weight 1/2."""

    values = distances.astype(float)
    if values.empty or not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0.0).any():
        raise ValueError("Gaussian weights require finite non-negative distances")
    scale = float(values.median())
    if not math.isfinite(scale) or scale <= np.finfo(float).eps:
        return pd.Series(1.0, index=values.index, dtype=float)
    weights = np.exp(-math.log(2.0) * (values / scale).pow(2))
    if not np.isfinite(weights.to_numpy(dtype=float)).all() or float(weights.sum()) <= 0.0:
        raise ValueError("invalid median-half Gaussian weights")
    return weights.astype(float)


def _target_sample(sample: pd.DataFrame, *, target: str) -> pd.DataFrame:
    """Build a symmetric historical-only candidate target panel."""

    values = sample.loc[:, list(MODELS)].astype(float)
    if target == "absolute":
        return values.copy()
    equal = values.mean(axis=1)
    if target == "relative_equal_weight":
        return values.sub(equal, axis=0)
    if target == "pairwise_mean":
        result = pd.DataFrame(index=values.index, columns=list(MODELS), dtype=float)
        for model in MODELS:
            others = [other for other in MODELS if other != model]
            result[model] = values[model].sub(values[others].mean(axis=1), fill_value=np.nan)
        return result
    raise ValueError(f"unsupported symmetric target: {target}")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("weighted quantile must be within [0, 1]")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(ordered_weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weighted quantile requires positive finite weights")
    index = int(np.searchsorted(np.cumsum(ordered_weights), quantile * total, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _weighted_trimmed_mean(values: np.ndarray, weights: np.ndarray, trim_fraction: float) -> float:
    """Weighted mean after partial-weight trimming from both ordered tails."""

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    retained = weights[order].astype(float).copy()
    total = float(retained.sum())
    tail = trim_fraction * total
    for index in range(len(retained)):
        removed = min(float(retained[index]), tail)
        retained[index] -= removed
        tail -= removed
        if tail <= 0.0:
            break
    tail = trim_fraction * total
    for index in range(len(retained) - 1, -1, -1):
        removed = min(float(retained[index]), tail)
        retained[index] -= removed
        tail -= removed
        if tail <= 0.0:
            break
    retained_total = float(retained.sum())
    if retained_total <= 0.0:
        return float(np.average(ordered_values, weights=weights[order]))
    return float(np.average(ordered_values, weights=retained))


def _weighted_trim20_lcb10(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & np.isfinite(weights) & weights.gt(0.0)
    value_array = values.loc[valid].astype(float).to_numpy()
    weight_array = weights.loc[valid].astype(float).to_numpy()
    if len(value_array) == 0:
        return float("nan")
    center = _weighted_trimmed_mean(value_array, weight_array, 0.20)
    lower = _weighted_quantile(value_array, weight_array, 0.10)
    upper = _weighted_quantile(value_array, weight_array, 0.90)
    clipped = np.clip(value_array, lower, upper)
    total = float(weight_array.sum())
    mean = float(np.average(clipped, weights=weight_array))
    correction = total - float(np.square(weight_array).sum()) / total
    variance = (
        float(np.sum(weight_array * np.square(clipped - mean)) / correction)
        if correction > np.finfo(float).eps
        else 0.0
    )
    effective_n = total * total / float(np.square(weight_array).sum())
    standard_error = math.sqrt(max(variance, 0.0) / effective_n) if effective_n > 0.0 else 0.0
    return float(center - standard_error)


def _score_targets(
    targets: pd.DataFrame,
    weights: pd.Series,
    *,
    neighbor_weighting: str,
    metric: str,
) -> pd.Series:
    if neighbor_weighting == "equal":
        # Use the production helper verbatim for the parity reference.
        return _scoreopt_score_sample(targets, model_cols=list(MODELS), score_mode=metric)
    if neighbor_weighting != "median_half_gaussian":
        raise ValueError(f"unsupported neighbour weighting: {neighbor_weighting}")
    if metric != "trim20_lcb10":
        raise ValueError(f"unsupported weighted score metric: {metric}")
    return pd.Series(
        {model: _weighted_trim20_lcb10(targets[model], weights) for model in MODELS},
        dtype=float,
    )


def run_scoring_geometry_variant(
    panel: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    standardization: str,
    neighbor_weighting: str,
    target: str,
    lookback_days: int = FIXED_LOOKBACK_DAYS,
    nearest_k: int = FIXED_NEAREST_K,
    min_periods: int = FIXED_MIN_PERIODS,
    metric: str = FIXED_METRIC,
    variant: str,
) -> pd.DataFrame:
    """Replay one fixed variant with strictly predecessor-only state/returns."""

    if lookback_days <= 0 or nearest_k <= 0 or min_periods <= 0 or nearest_k > lookback_days:
        raise ValueError("invalid BaseGap fixed parameters")
    if standardization not in STATE_STANDARDIZATIONS:
        raise ValueError(f"unsupported state standardization: {standardization}")
    if neighbor_weighting not in NEIGHBOR_WEIGHTINGS:
        raise ValueError(f"unsupported neighbour weighting: {neighbor_weighting}")
    if target not in SYMMETRIC_TARGETS:
        raise ValueError(f"unsupported symmetric target: {target}")
    if metric not in {"trim20_lcb10", "lcb10", "mean"}:
        raise ValueError(f"unsupported score metric: {metric}")
    if neighbor_weighting != "equal" and metric != "trim20_lcb10":
        raise ValueError("Gaussian weighting is preregistered only for trim20_lcb10")

    features = list(feature_cols)
    required = ["score_day", *MODELS, *features]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"panel missing scoring-geometry columns: {missing}")
    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    if "state_available" not in ordered.columns:
        ordered["state_available"] = True
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)

    rows: list[dict[str, object]] = []
    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        returns_history = ordered.iloc[:position][["score_day", *MODELS]]
        history_end = returns_history["score_day"].max() if not returns_history.empty else None
        record: dict[str, object] = {
            "score_day": score_day,
            "variant": variant,
            "lookback_days": int(lookback_days),
            "nearest_k": int(nearest_k),
            "min_periods": int(min_periods),
            "metric": metric,
            "state_standardization": standardization,
            "neighbor_weighting": neighbor_weighting,
            "target": target,
            "basegap_history_end": history_end,
            "basegap_history_days": 0,
            "basegap_reason": None,
            # This inactive warm-up default reproduces the old BaseGap audit
            # shape only; it is excluded from active selection evaluation.
            "basegap_selected_name": "Regsim",
            "basegap_score_gap": float("nan"),
            "neighbor_min_day": None,
            "neighbor_max_day": None,
            "neighbor_count": 0,
            "neighbor_distance_median": float("nan"),
            "neighbor_weight_sum": float("nan"),
            "neighbor_effective_n": float("nan"),
        }
        for model in MODELS:
            record[f"basegap_score_{model}"] = float("nan")

        current_values = current[features].to_numpy(dtype=float)
        if not bool(current["state_available"]):
            record["basegap_reason"] = "feature_missing"
        elif not np.isfinite(current_values).all():
            record["basegap_reason"] = "feature_na"
        else:
            # Exactly as Stage A: retain state-present rows, tail the lookback
            # first, then discard rows whose state/return cells are incomplete.
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
                distances = _standardised_distances(
                    history_features, current[features], standardization=standardization
                )
                nearest_index = distances.sort_values().index[:nearest_k]
                nearest_distances = distances.loc[nearest_index]
                sample = history_returns.loc[nearest_index]
                weights = (
                    pd.Series(1.0, index=nearest_index, dtype=float)
                    if neighbor_weighting == "equal"
                    else _median_half_gaussian_weights(nearest_distances)
                )
                target_sample = _target_sample(sample, target=target)
                scores = _score_targets(
                    target_sample,
                    weights,
                    neighbor_weighting=neighbor_weighting,
                    metric=metric,
                ).sort_values(ascending=False)
                selected = str(scores.index[0])
                best_score = float(scores.iloc[0])
                second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
                score_gap = best_score - second_score if math.isfinite(second_score) else float("nan")
                neighbour_days = pd.to_datetime(joined.loc[nearest_index, "score_day"], errors="coerce").dt.date
                weight_sum = float(weights.sum())
                effective_n = weight_sum * weight_sum / float(np.square(weights).sum())
                record.update(
                    {
                        "basegap_selected_name": selected,
                        "basegap_reason": "score_best" if score_gap > 0.0 else "margin_default",
                        "basegap_history_days": int(len(sample)),
                        "basegap_score_gap": score_gap,
                        "neighbor_min_day": neighbour_days.min(),
                        "neighbor_max_day": neighbour_days.max(),
                        "neighbor_count": int(len(sample)),
                        "neighbor_distance_median": float(nearest_distances.median()),
                        "neighbor_weight_sum": weight_sum,
                        "neighbor_effective_n": effective_n,
                    }
                )
                for model in MODELS:
                    record[f"basegap_score_{model}"] = float(scores[model])

        selected_name = str(record["basegap_selected_name"])
        realised = {model: float(current[model]) for model in MODELS}
        selected_return = realised[selected_name]
        best_name = max(MODELS, key=lambda model: realised[model])
        record.update(
            {
                **{f"realized_return_{model}": realised[model] for model in MODELS},
                "selected_return": selected_return,
                "equal_weight_return": float(np.mean(list(realised.values()))),
                "best_name": best_name,
                "best_return": realised[best_name],
                "selection_alpha": selected_return - float(np.mean(list(realised.values()))),
                "regret": realised[best_name] - selected_return,
                "selected_rank": int(1 + sum(realised[model] > selected_return for model in MODELS)),
                "selection_active": bool(record["basegap_reason"] in {"score_best", "margin_default"}),
                "execution_metadata_complete": bool(current.get("execution_metadata_complete", False)),
            }
        )
        rows.append(record)

    result = pd.DataFrame(rows)
    active = result.loc[result["selection_active"]]
    if not active.empty and not (
        pd.to_datetime(active["neighbor_max_day"]).dt.date
        < pd.to_datetime(active["score_day"]).dt.date
    ).all():
        raise RuntimeError("scoring-geometry replay leakage: neighbour is not strictly before score_day")
    return result


def _variant_summary(all_daily: pd.DataFrame, variants: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None, str | None]] = [("full", None, None), *SPLITS]
    for variant in variants:
        daily = all_daily.loc[all_daily["variant"] == variant["variant"]].copy()
        for scope, start, end in scopes:
            rows.append(
                _scope_metrics(
                    _scope_frame(daily, start=start, end=end, active_only=False),
                    variant_row=variant,
                    scope=scope,
                    active_only=False,
                )
            )
            rows.append(
                _scope_metrics(
                    _scope_frame(daily, start=start, end=end, active_only=True),
                    variant_row=variant,
                    scope=scope,
                    active_only=True,
                )
            )
    return pd.DataFrame(rows)


def rank_design_variants(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank once using only active design dates; validation/final OOS are report-only."""

    ranking = summary.loc[(summary["scope"] == "design") & summary["active_only"]].copy()
    ranking = ranking.loc[ranking["n_days"] > 0].copy()
    ranking = ranking.sort_values(
        ["selection_alpha_mean_bp", "hit_rate", "mean_regret_bp", "variant"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "design_rank", np.arange(1, len(ranking) + 1))
    return ranking


def validate_stage_a_run(
    stage_a_run: Path,
    *,
    source_verification: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate the completed Stage-A r3 lineage and load its fixed baseline."""

    stage_a_run = _assert_locked_stage_a(stage_a_run)
    status_path = stage_a_run / "run_status.json"
    manifest_path = stage_a_run / "run_manifest.json"
    daily_path = stage_a_run / "daily_selector_replay.csv"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"Stage-A r3 is not completed: {status.get('status')!r}")
    if manifest.get("design_selected_variant") != "lb240_k60_trim20_lcb10":
        raise ValueError("Stage-A r3 selected parameter point drifted from lb240_k60_trim20_lcb10")
    source_info = manifest.get("source_input_verification")
    if not isinstance(source_info, Mapping):
        raise ValueError("Stage-A r3 lacks frozen source verification metadata")
    expected_hash = str(source_verification.get("source_manifest_sha256", ""))
    if str(source_info.get("source_manifest_sha256", "")) != expected_hash:
        raise RuntimeError("Stage-A r3 frozen source hash does not match this verified source")
    daily = pd.read_csv(daily_path)
    if "variant" not in daily.columns or "score_day" not in daily.columns:
        raise ValueError("Stage-A daily selector replay has no variant/score_day columns")
    baseline = daily.loc[daily["variant"] == "lb240_k60_trim20_lcb10"].copy()
    if baseline.empty or baseline["score_day"].duplicated().any():
        raise ValueError("Stage-A r3 baseline is absent or has duplicate score days")
    return baseline, {
        "stage_a_run": str(stage_a_run),
        "stage_a_run_manifest_sha256": sha256(manifest_path),
        "stage_a_run_status_sha256": sha256(status_path),
        "stage_a_daily_selector_sha256": sha256(daily_path),
        "stage_a_daily_rows": int(len(baseline)),
        "stage_a_design_selected_variant": str(manifest.get("design_selected_variant")),
        "stage_a_source_manifest_sha256": str(source_info.get("source_manifest_sha256")),
    }


def _task_state(*, phase: str, run_root: Path, source_run: Path, detail: str) -> str:
    return f"""# Task State

## Objective

- Frozen, research-only 12-variant BaseGap scoring-geometry replay after Stage-A r3 parameter selection.

## Risk Level

- medium: a scoring-geometry family can overfit; every live/runtime asset is read-only.

## Current Verified Facts

- selector source is locked to `{source_run}`
- Stage-A source is locked to `{DEFAULT_STAGE_A_RUN}`
- output is isolated beneath `{run_root}`
- no DB, scheduler, live runtime, live config, factor store, model state, or live result directory is written.

## Files Changed

- only files beneath this isolated research run root.

## Open Risks

- this is a selector-shadow-return replay, not a full single-book execution backtest.
- validation/final-OOS values are report-only after the one design-period ranking.

## Next Action

- {detail}

## Handoff Summary

- phase: {phase}
"""


def _plot_summary(run_root: Path, summary: pd.DataFrame) -> None:
    scopes = ("design", "validation", "final_oos")
    plot = summary.loc[summary["active_only"] & summary["scope"].isin(scopes)].copy()
    variants = plot["variant"].drop_duplicates().tolist()
    if not variants:
        return
    figure, axis = plt.subplots(figsize=(15, 6))
    positions = np.arange(len(variants))
    width = 0.24
    for offset, scope in enumerate(scopes):
        values = [
            float(
                plot.loc[(plot["variant"] == variant) & (plot["scope"] == scope), "selection_alpha_mean_bp"].iloc[0]
            )
            for variant in variants
        ]
        axis.bar(positions + (offset - 1) * width, values, width=width, label=scope)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, labels=variants, rotation=25, ha="right")
    axis.set_ylabel("selection alpha vs three-candidate arithmetic equal weight (bp/day)")
    axis.set_title("Preregistered BaseGap scoring geometry; validation/final OOS are report-only")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_root / "scoring_geometry_selection_alpha.png", dpi=160)
    plt.close(figure)


def _write_results(run_root: Path, *, source_run: Path, ranking: pd.DataFrame, summary: pd.DataFrame) -> None:
    winner = ranking.iloc[0]
    report = summary.loc[
        summary["variant"].isin([BASELINE_VARIANT, str(winner["variant"])])
        & summary["active_only"]
        & summary["scope"].isin(["full", "design", "validation", "final_oos"])
    ].sort_values(["variant", "scope"])
    lines = [
        "# Frozen pure-BaseGap scoring-geometry replay",
        "",
        "## Scope",
        "",
        f"- Frozen selector source: `{source_run}`",
        f"- Stage-A fixed parameters: lookback={FIXED_LOOKBACK_DAYS}, K={FIXED_NEAREST_K}, metric={FIXED_METRIC}.",
        "- Twelve preregistered variants: 2 state scalings x 2 neighbour-weight rules x 3 symmetric target definitions.",
        "- No Champion, Robust, Fusion, threshold/gate, factor/model/strategy change, or live mutation is present.",
        "",
        "## Mandatory parity gate",
        "",
        f"- `{BASELINE_VARIANT}` matched Stage-A `lb240_k60_trim20_lcb10` day by day before the other eleven variants ran.",
        "",
        "## Design-only ranking",
        "",
        f"- winner: `{winner['variant']}`; active design selection alpha={float(winner['selection_alpha_mean_bp']):.3f} bp/day.",
        "- Validation and final OOS are report-only and must not be used for another selection pass.",
        "",
        "## Aligned active-day summary",
        "",
        "| variant | scope | n | alpha (bp/day) | hit rate | mean regret (bp) | selected Sharpe | selected MDD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in report.iterrows():
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
            "## Definitions",
            "",
            "- `median_half_gaussian`: after selecting the same Top-K neighbours, weight is `exp(-ln(2) * (distance / median_distance)^2)`; a zero median uses equal weights fail-safe.",
            "- `relative_equal_weight`: each candidate historical return minus that day's three-candidate arithmetic mean.",
            "- `pairwise_mean`: each candidate's mean historical return difference against the other two candidates; it is a symmetric contrast diagnostic.",
            "- With exactly three candidates, `pairwise_mean` is algebraically 1.5 times `relative_equal_weight`; under the same score statistic it should select identically apart from numerical ties. It remains in the preregistered family as an explicit symmetry audit, not an independent degree of freedom.",
            "",
            "## Evidence",
            "",
            "- `input_verification.json`, `stage_a_verification.json`, `stage_a_parity_audit.csv`, `preregistered_variants.csv`, `daily_selector_replay.csv`, `summary_metrics.csv`, `design_ranking.csv`, `leakage_audit.json`, and `scoring_geometry_selection_alpha.png`.",
            "",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_replay(*, source_run: Path, stage_a_run: Path, run_root: Path) -> dict[str, object]:
    """Execute all 12 preregistered variants in an isolated research root."""

    # Verify both frozen lineages before creating an output directory.
    config, source_verification = validate_frozen_source(source_run)
    stage_a_baseline, stage_a_verification = validate_stage_a_run(
        stage_a_run, source_verification=source_verification
    )
    run_root = _assert_output_root(run_root)
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite research output: {run_root}")
    variants = preregistered_variants()
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "task_state.md").write_text(
        _task_state(
            phase="running",
            run_root=run_root,
            source_run=source_run,
            detail="prepare frozen panel and require Stage-A parity before the 12 in-memory replays.",
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
    try:
        _write_json(run_root / "input_verification.json", source_verification)
        _write_json(run_root / "stage_a_verification.json", stage_a_verification)
        pd.DataFrame(variants).to_csv(run_root / "preregistered_variants.csv", index=False)
        panel, feature_cols = prepare_panel(config)

        baseline_row = next(row for row in variants if row["is_stage_a_parity_reference"])
        baseline = run_scoring_geometry_variant(
            panel,
            feature_cols=feature_cols,
            standardization=str(baseline_row["state_standardization"]),
            neighbor_weighting=str(baseline_row["neighbor_weighting"]),
            target=str(baseline_row["target"]),
            lookback_days=int(baseline_row["lookback_days"]),
            nearest_k=int(baseline_row["nearest_k"]),
            min_periods=int(baseline_row["min_periods"]),
            metric=str(baseline_row["metric"]),
            variant=str(baseline_row["variant"]),
        )
        parity_fields = [
            "score_day",
            "basegap_selected_name",
            "basegap_reason",
            "basegap_history_days",
            "basegap_history_end",
            "basegap_score_gap",
            *(f"basegap_score_{model}" for model in MODELS),
        ]
        stage_a_parity = parity_audit(
            baseline[parity_fields],
            stage_a_baseline[parity_fields],
            label="zscore_equal_absolute_vs_stage_a_r3",
        )
        stage_a_parity.to_csv(run_root / "stage_a_parity_audit.csv", index=False)

        all_daily: list[pd.DataFrame] = [baseline]
        for variant in variants:
            if variant["variant"] == baseline_row["variant"]:
                continue
            all_daily.append(
                run_scoring_geometry_variant(
                    panel,
                    feature_cols=feature_cols,
                    standardization=str(variant["state_standardization"]),
                    neighbor_weighting=str(variant["neighbor_weighting"]),
                    target=str(variant["target"]),
                    lookback_days=int(variant["lookback_days"]),
                    nearest_k=int(variant["nearest_k"]),
                    min_periods=int(variant["min_periods"]),
                    metric=str(variant["metric"]),
                    variant=str(variant["variant"]),
                )
            )
        combined = pd.concat(all_daily, ignore_index=True)
        if combined["variant"].nunique() != 12:
            raise RuntimeError("not every preregistered scoring-geometry variant ran")
        combined.to_csv(run_root / "daily_selector_replay.csv", index=False)
        summary = _variant_summary(combined, variants)
        summary.to_csv(run_root / "summary_metrics.csv", index=False)
        ranking = rank_design_variants(summary)
        ranking.to_csv(run_root / "design_ranking.csv", index=False)
        _plot_summary(run_root, summary)
        active = combined.loc[combined["selection_active"]]
        _write_json(
            run_root / "leakage_audit.json",
            {
                "strict_predecessor_neighbours": bool(
                    (
                        pd.to_datetime(active["neighbor_max_day"]).dt.date
                        < pd.to_datetime(active["score_day"]).dt.date
                    ).all()
                ),
                "source_manifest_hash_verified": True,
                "stage_a_lineage_hash_verified": True,
                "stage_a_parity_verified": True,
                "return_day_not_used_for_same_day_selection": True,
                "future_state_not_used_for_prior_selection": True,
                "variants": int(combined["variant"].nunique()),
                "active_rows": int(len(active)),
                "final_oos_selection_rule": "not used; reported once after design-only ranking",
            },
        )
        _write_results(run_root, source_run=source_run, ranking=ranking, summary=summary)
        manifest = {
            "run_class": "research_only_pure_basegap_scoring_geometry_replay",
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "live_config_written": False,
            "live_result_written": False,
            "model_state_written": False,
            "frozen_source_run": str(source_run),
            "stage_a_run": str(stage_a_run),
            "source_input_verification": {
                "source_manifest_sha256": source_verification["source_manifest_sha256"],
                "verified_file_count": source_verification["verified_file_count"],
            },
            "stage_a_verification": stage_a_verification,
            "comparison_contract": {
                "candidate_returns": "aligned standalone full-cycle day_return",
                "selector": "pure symmetric direct BaseGap argmax",
                "state": "frozen path_full_t1430 only",
                "candidate_order": list(MODELS),
                "fixed_basegap_parameters": {
                    "lookback_days": FIXED_LOOKBACK_DAYS,
                    "nearest_k": FIXED_NEAREST_K,
                    "min_periods": FIXED_MIN_PERIODS,
                    "metric": FIXED_METRIC,
                    "margin": 0.0,
                },
                "threshold_or_margin_gate": False,
                "champion_robust_fusion_logic": False,
                "evaluation_days": "only execution_metadata_complete dates",
                "main_objective": "selected_return - arithmetic mean(Regsim, Ensemble, HL20)",
                "final_oos_policy": "reported, not used for selection",
            },
            "variants": variants,
            "design_selected_variant": str(ranking.iloc[0]["variant"]),
            "splits": [{"name": name, "start": start, "end": end} for name, start, end in SPLITS],
            "source_code": [
                {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
                {
                    "path": str(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_tuning_replay.py"),
                    "sha256": sha256(REPO_ROOT / "harness" / "tools" / "model_switch_basegap_tuning_replay.py"),
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
                "design_selected_variant": str(ranking.iloc[0]["variant"]),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
                "research_output_only": True,
            },
        )
        (run_root / "task_state.md").write_text(
            _task_state(
                phase="completed",
                run_root=run_root,
                source_run=source_run,
                detail="completed; do not select again using validation/final-OOS values.",
            ),
            encoding="utf-8",
        )
        return {
            "run_root": str(run_root),
            "design_selected_variant": str(ranking.iloc[0]["variant"]),
            "design_selected_alpha_bp_per_day": float(ranking.iloc[0]["selection_alpha_mean_bp"]),
            "verified_frozen_files": int(source_verification["verified_file_count"]),
            "variants": int(combined["variant"].nunique()),
        }
    except Exception:
        _write_json(
            run_root / "run_status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
                "research_output_only": True,
            },
        )
        (run_root / "task_state.md").write_text(
            _task_state(
                phase="failed",
                run_root=run_root,
                source_run=source_run,
                detail="failed closed before a complete research result; inspect the traceback and frozen-input audit.",
            ),
            encoding="utf-8",
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--stage-a-run", type=Path, default=DEFAULT_STAGE_A_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = _make_run_root(args.output_root, args.run_name)
    result = run_replay(source_run=args.source_run, stage_a_run=args.stage_a_run, run_root=run_root)
    print(json.dumps(result, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
