"""Frozen, research-only BaseGap replay with direct factor-rank similarity.

This is deliberately separate from the prior F12 state-family experiment.
F12 compressed each daily 50-factor cross-section into twelve aggregate
statistics.  Here, a current day and a historical day are compared directly:
for every live50 factor, the factor's percentile ranks across the two dates'
shared score-code universe are correlated.  The median of those fifty
correlations is a factor-similarity signal for that *pair* of dates.

The experiment is bounded and read-only with respect to production:

* selector source and factor store are locked frozen snapshots;
* BaseGap stays 240-day history / Top-60 / trim20_lcb10 / direct argmax;
* only neighbour ordering changes, never candidate returns or scoring;
* no Champion, Robust, Fusion, threshold, model, factor, DB, scheduler, or
  production-result path is changed;
* all artefacts are written below one new ``research_scratch`` run root.

The result is selector-shadow-return research, not a new single-book
execution backtest and never a live-promotion mechanism.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cbond_on.infra.live.model_switch import _scoreopt_score_sample  # noqa: E402
from harness.tools.model_switch_basegap_state_family_replay import (  # noqa: E402
    FACTOR_CONTRACT_PATH,
    load_common_score_codes,
    load_factor_contract,
    load_factor_cross_section,
    verify_factor_freeze,
)
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
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_basegap_factor_similarity_20260811")

FIXED_LOOKBACK_DAYS = 240
FIXED_NEAREST_K = 60
FIXED_MIN_PERIODS = 60
FIXED_METRIC = "trim20_lcb10"
MIN_SHARED_CODES = 80
MIN_VALID_FACTORS = 45
RECENT_DIAGNOSTIC_DAYS = 60

# P25/P50 are the only eligible factor-assisted candidates.  P0 is the exact
# Stage-A path44 reference; F100 and Recent60 are diagnostics only and cannot
# be selected after looking at any result segment.
VARIANTS: tuple[dict[str, object], ...] = (
    {
        "variant": "P0_path44_only",
        "path_weight": 1.0,
        "factor_weight": 0.0,
        "selection_eligible": True,
        "role": "parity_baseline",
    },
    {
        "variant": "P25_path75_factor25",
        "path_weight": 0.75,
        "factor_weight": 0.25,
        "selection_eligible": True,
        "role": "factor_assisted_candidate",
    },
    {
        "variant": "P50_path50_factor50",
        "path_weight": 0.50,
        "factor_weight": 0.50,
        "selection_eligible": True,
        "role": "factor_assisted_candidate",
    },
    {
        "variant": "F100_factor_rank_only_diagnostic",
        "path_weight": 0.0,
        "factor_weight": 1.0,
        "selection_eligible": False,
        "role": "diagnostic_only",
    },
    {
        "variant": "Recent60_diagnostic",
        "path_weight": None,
        "factor_weight": None,
        "selection_eligible": False,
        "role": "recentness_diagnostic",
    },
)


@dataclass(frozen=True)
class FactorRankSlice:
    """One score-day's live50 cross-section after within-day percentile rank."""

    codes: pd.Index
    values: np.ndarray


@dataclass(frozen=True)
class FactorPairDistance:
    """Direct factor-rank distance for one current/history pair."""

    distance: float | None
    median_rho: float | None
    shared_codes: int
    valid_factors: int


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
        raise ValueError(f"{description} must stay below {root}: {resolved}") from exc
    return resolved


def _assert_locked_source(source_run: Path) -> Path:
    source = source_run.resolve()
    expected = DEFAULT_SOURCE_RUN.resolve()
    if source != expected:
        raise ValueError(f"selector source is locked to {expected}, got {source}")
    return source


def _assert_locked_stage_a(stage_a_run: Path) -> Path:
    source = stage_a_run.resolve()
    expected = DEFAULT_STAGE_A_RUN.resolve()
    if source != expected:
        raise ValueError(f"Stage-A parity source is locked to {expected}, got {source}")
    if not (source / "daily_selector_replay.csv").is_file():
        raise FileNotFoundError(f"Stage-A daily replay is missing: {source}")
    return source


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _resolve_inside(output_root, DEFAULT_OUTPUT_ROOT, description="research output root")
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _resolve_inside(root / name, DEFAULT_OUTPUT_ROOT, description="research run")
    if run_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("a new run leaf under the research output root is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    return run_root


def make_factor_rank_slice(frame: pd.DataFrame) -> FactorRankSlice:
    """Convert a code-indexed factor matrix to independent daily percentile ranks."""

    if frame.index.has_duplicates:
        raise ValueError("factor cross-section has duplicate codes")
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ranks = numeric.rank(axis=0, method="average", pct=True, na_option="keep")
    return FactorRankSlice(codes=pd.Index(frame.index.astype(str), name="code"), values=ranks.to_numpy(dtype=float))


def factor_rank_distance(
    current: FactorRankSlice,
    historical: FactorRankSlice,
    *,
    min_shared_codes: int = MIN_SHARED_CODES,
    min_valid_factors: int = MIN_VALID_FACTORS,
) -> FactorPairDistance:
    """Return ``1 - median(rank-correlation)`` across the live50 contract.

    Each date is ranked only within its own frozen three-model common score
    universe.  The two matrices are then aligned by actual code (never parquet
    row order), restricted to their code intersection, and correlated factor
    by factor.  It uses no candidate return, label, or future input.
    """

    if min_shared_codes <= 1 or min_valid_factors <= 0:
        raise ValueError("invalid factor-pair completeness thresholds")
    common = current.codes.intersection(historical.codes, sort=False)
    if len(common) < min_shared_codes:
        return FactorPairDistance(distance=None, median_rho=None, shared_codes=int(len(common)), valid_factors=0)
    current_positions = current.codes.get_indexer(common)
    historical_positions = historical.codes.get_indexer(common)
    if (current_positions < 0).any() or (historical_positions < 0).any():
        raise RuntimeError("factor code intersection cannot be aligned")
    left = current.values[current_positions]
    right = historical.values[historical_positions]
    finite = np.isfinite(left) & np.isfinite(right)
    count = finite.sum(axis=0).astype(float)
    if left.shape[1] != right.shape[1]:
        raise ValueError("factor contracts differ across dates")
    safe_left = np.where(finite, left, 0.0)
    safe_right = np.where(finite, right, 0.0)
    left_mean = np.divide(safe_left.sum(axis=0), count, out=np.full(left.shape[1], np.nan), where=count > 0.0)
    right_mean = np.divide(safe_right.sum(axis=0), count, out=np.full(right.shape[1], np.nan), where=count > 0.0)
    left_centered = np.where(finite, left - left_mean, 0.0)
    right_centered = np.where(finite, right - right_mean, 0.0)
    numerator = (left_centered * right_centered).sum(axis=0)
    denominator = np.sqrt((left_centered**2).sum(axis=0) * (right_centered**2).sum(axis=0))
    rho = np.divide(numerator, denominator, out=np.full(left.shape[1], np.nan), where=denominator > 1e-12)
    valid = (count >= float(min_shared_codes)) & np.isfinite(rho)
    valid_count = int(valid.sum())
    if valid_count < min_valid_factors:
        return FactorPairDistance(distance=None, median_rho=None, shared_codes=int(len(common)), valid_factors=valid_count)
    median_rho = float(np.median(rho[valid]))
    return FactorPairDistance(
        distance=float(1.0 - median_rho),
        median_rho=median_rho,
        shared_codes=int(len(common)),
        valid_factors=valid_count,
    )


def build_factor_rank_slices(
    *,
    input_root: Path,
    factor_root: Path,
    score_days: Sequence[object],
    factor_names: Sequence[str],
) -> tuple[dict[date, FactorRankSlice], pd.DataFrame]:
    """Load every frozen T1430 slice and record its strict score alignment."""

    slices: dict[date, FactorRankSlice] = {}
    audit_rows: list[dict[str, object]] = []
    for raw_day in score_days:
        score_day = pd.Timestamp(raw_day).date()
        codes = load_common_score_codes(input_root, score_day)
        cross_section = load_factor_cross_section(
            factor_root=factor_root,
            score_day=score_day,
            score_codes=codes,
            factor_names=factor_names,
        )
        if len(cross_section.columns) != len(factor_names):
            raise ValueError(f"factor column count drift on {score_day}")
        slices[score_day] = make_factor_rank_slice(cross_section)
        audit_rows.append(
            {
                "score_day": score_day,
                "score_universe_n": int(len(codes)),
                "factor_rows": int(len(cross_section)),
                "factor_file": str(
                    factor_root / "factors" / "T1430" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"
                ),
                "factor_cell_missing_rate": float(np.isnan(cross_section.to_numpy(dtype=float)).mean()),
            }
        )
    return slices, pd.DataFrame(audit_rows)


def _valid_history(
    ordered: pd.DataFrame,
    *,
    position: int,
    path_features: Sequence[str],
) -> pd.DataFrame:
    columns = ["score_day", "state_available", *MODELS, *path_features]
    history = ordered.iloc[:position][columns].copy()
    history = history.loc[history["state_available"]].tail(FIXED_LOOKBACK_DAYS).copy()
    valid = ~(history[list(path_features)].isna().any(axis=1) | history[list(MODELS)].isna().any(axis=1))
    return history.loc[valid].copy()


def build_factor_distance_cache(
    panel: pd.DataFrame,
    *,
    path_features: Sequence[str],
    factor_slices: Mapping[date, FactorRankSlice],
    min_shared_codes: int = MIN_SHARED_CODES,
    min_valid_factors: int = MIN_VALID_FACTORS,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Precompute only strictly-predecessor current/history factor distances."""

    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    ordered["state_available"] = ordered.get("state_available", True)
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)
    cache: dict[int, pd.DataFrame] = {}
    audit_rows: list[dict[str, object]] = []
    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        current_slice = factor_slices.get(score_day)
        if current_slice is None:
            cache[position] = pd.DataFrame(
                columns=["history_index", "history_day", "factor_distance", "factor_median_rho", "shared_codes", "valid_factors"]
            )
            continue
        history = _valid_history(ordered, position=position, path_features=path_features)
        rows: list[dict[str, object]] = []
        for history_index, history_row in history.iterrows():
            history_day = pd.Timestamp(history_row["score_day"]).date()
            historical_slice = factor_slices.get(history_day)
            if historical_slice is None:
                pair = FactorPairDistance(distance=None, median_rho=None, shared_codes=0, valid_factors=0)
            else:
                pair = factor_rank_distance(
                    current_slice,
                    historical_slice,
                    min_shared_codes=min_shared_codes,
                    min_valid_factors=min_valid_factors,
                )
            row = {
                "score_day": score_day,
                "history_index": int(history_index),
                "history_day": history_day,
                "factor_distance": pair.distance,
                "factor_median_rho": pair.median_rho,
                "shared_codes": pair.shared_codes,
                "valid_factors": pair.valid_factors,
                "age_trading_days": int(position - history_index),
            }
            rows.append(row)
            audit_rows.append(row)
        cache[position] = pd.DataFrame(rows)
        if position and position % 50 == 0:
            print(f"factor-distance cache: {position}/{len(ordered)} score days", flush=True)
    return cache, pd.DataFrame(audit_rows)


def combine_neighbour_distances(
    path_distance: pd.Series,
    factor_distance: pd.Series,
    *,
    path_weight: float,
    factor_weight: float,
) -> pd.Series:
    """Fuse history-day ranks, not raw distances, to avoid unit domination."""

    if path_weight < 0.0 or factor_weight < 0.0 or not math.isclose(path_weight + factor_weight, 1.0):
        raise ValueError("path/factor weights must be non-negative and sum to one")
    if path_distance.index.tolist() != factor_distance.index.tolist():
        raise ValueError("path and factor distances must share the same ordered history index")
    if path_distance.empty:
        return pd.Series(dtype=float, index=path_distance.index)
    if factor_weight == 0.0:
        return path_distance.astype(float).copy()
    if path_weight == 0.0:
        return factor_distance.astype(float).copy()
    path_rank = path_distance.astype(float).rank(method="first", ascending=True, pct=True)
    factor_rank = factor_distance.astype(float).rank(method="first", ascending=True, pct=True)
    return path_weight * path_rank + factor_weight * factor_rank


def _new_record(
    *,
    score_day: date,
    variant: str,
    position: int,
    ordered: pd.DataFrame,
    path_weight: float | None,
    factor_weight: float | None,
) -> dict[str, object]:
    returns_history = ordered.iloc[:position][["score_day", *MODELS]]
    history_end = returns_history["score_day"].max() if not returns_history.empty else None
    record: dict[str, object] = {
        "score_day": score_day,
        "variant": variant,
        "lookback_days": FIXED_LOOKBACK_DAYS,
        "nearest_k": FIXED_NEAREST_K,
        "min_periods": FIXED_MIN_PERIODS,
        "metric": FIXED_METRIC,
        "path_weight": path_weight,
        "factor_weight": factor_weight,
        "basegap_history_end": history_end,
        "basegap_history_days": 0,
        "basegap_reason": None,
        "basegap_selected_name": "Regsim",
        "basegap_score_gap": float("nan"),
        "neighbor_min_day": None,
        "neighbor_max_day": None,
        "neighbor_count": 0,
        "factor_pair_valid_history_days": 0,
        "neighbor_mean_path_distance": float("nan"),
        "neighbor_mean_factor_distance": float("nan"),
        "neighbor_median_factor_rho": float("nan"),
        "neighbor_mean_age_trading_days": float("nan"),
        "neighbor_median_age_trading_days": float("nan"),
        "neighbor_recent20_share": float("nan"),
        "neighbor_min_shared_codes": float("nan"),
        "neighbor_min_valid_factors": float("nan"),
    }
    for model in MODELS:
        record[f"basegap_score_{model}"] = float("nan")
    return record


def _path_distance(history: pd.DataFrame, *, current: pd.Series, path_features: Sequence[str]) -> pd.Series:
    features = list(path_features)
    mean = history[features].mean(axis=0)
    std = history[features].std(axis=0).replace(0, math.nan).fillna(1.0)
    history_z = (history[features] - mean) / std
    current_z = (current[features] - mean) / std
    return ((history_z.sub(current_z, axis=1) ** 2).sum(axis=1)).pow(0.5)


def _finalise_realised_outcome(record: dict[str, object], current: pd.Series) -> None:
    realised = {model: float(current[model]) for model in MODELS}
    selected_name = str(record["basegap_selected_name"])
    selected_return = realised[selected_name]
    best_name = max(MODELS, key=lambda model: realised[model])
    selected_rank = int(1 + sum(realised[model] > selected_return for model in MODELS))
    record.update(
        {
            **{f"realized_return_{model}": realised[model] for model in MODELS},
            "selected_return": selected_return,
            "equal_weight_return": float(np.mean(list(realised.values()))),
            "best_name": best_name,
            "best_return": realised[best_name],
            "selection_alpha": selected_return - float(np.mean(list(realised.values()))),
            "regret": realised[best_name] - selected_return,
            "selected_rank": selected_rank,
            "selection_active": bool(record["basegap_reason"] in {"score_best", "margin_default"}),
            "execution_metadata_complete": bool(current.get("execution_metadata_complete", False)),
        }
    )


def _apply_neighbours(
    *,
    record: dict[str, object],
    ordered: pd.DataFrame,
    current_position: int,
    history: pd.DataFrame,
    nearest_index: pd.Index,
    path_distance: pd.Series | None,
    factor_information: pd.DataFrame | None,
) -> None:
    sample = history.loc[nearest_index, list(MODELS)]
    scores = _scoreopt_score_sample(sample, model_cols=list(MODELS), score_mode=FIXED_METRIC).sort_values(ascending=False)
    selected = str(scores.index[0])
    best_score = float(scores.iloc[0])
    second_score = float(scores.iloc[1]) if len(scores) > 1 else float("nan")
    score_gap = best_score - second_score if math.isfinite(second_score) else float("nan")
    neighbour_days = pd.to_datetime(history.loc[nearest_index, "score_day"], errors="coerce").dt.date
    ages = np.asarray([current_position - int(index) for index in nearest_index], dtype=float)
    record.update(
        {
            "basegap_selected_name": selected,
            "basegap_reason": "score_best" if score_gap > 0.0 else "margin_default",
            "basegap_history_days": int(len(sample)),
            "basegap_score_gap": score_gap,
            "neighbor_min_day": neighbour_days.min(),
            "neighbor_max_day": neighbour_days.max(),
            "neighbor_count": int(len(sample)),
            "neighbor_mean_age_trading_days": float(np.mean(ages)),
            "neighbor_median_age_trading_days": float(np.median(ages)),
            "neighbor_recent20_share": float((ages <= 20.0).mean()),
        }
    )
    if path_distance is not None:
        record["neighbor_mean_path_distance"] = float(path_distance.loc[nearest_index].mean())
    if factor_information is not None:
        selected_factor = factor_information.loc[nearest_index]
        record.update(
            {
                "factor_pair_valid_history_days": int(len(factor_information)),
                "neighbor_mean_factor_distance": float(selected_factor["factor_distance"].mean()),
                "neighbor_median_factor_rho": float(selected_factor["factor_median_rho"].median()),
                "neighbor_min_shared_codes": float(selected_factor["shared_codes"].min()),
                "neighbor_min_valid_factors": float(selected_factor["valid_factors"].min()),
            }
        )
    for model in MODELS:
        record[f"basegap_score_{model}"] = float(scores[model])


def run_factor_similarity_variant(
    panel: pd.DataFrame,
    *,
    path_features: Sequence[str],
    factor_cache: Mapping[int, pd.DataFrame],
    variant: str,
    path_weight: float,
    factor_weight: float,
) -> pd.DataFrame:
    """Run one fixed path/factor rank-fusion variant entirely in memory."""

    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    ordered["state_available"] = ordered.get("state_available", True)
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        record = _new_record(
            score_day=score_day,
            variant=variant,
            position=position,
            ordered=ordered,
            path_weight=path_weight,
            factor_weight=factor_weight,
        )
        current_values = current[list(path_features)].to_numpy(dtype=float)
        if not bool(current["state_available"]):
            record["basegap_reason"] = "feature_missing"
        elif not np.isfinite(current_values).all():
            record["basegap_reason"] = "feature_na"
        else:
            history = _valid_history(ordered, position=position, path_features=path_features)
            if len(history) < FIXED_NEAREST_K:
                record["basegap_history_days"] = int(len(history))
                record["basegap_reason"] = "insufficient_history"
            else:
                factor_information: pd.DataFrame | None = None
                if factor_weight > 0.0:
                    pair_rows = factor_cache.get(position)
                    if pair_rows is None or pair_rows.empty:
                        record["basegap_reason"] = "factor_pair_missing"
                        _finalise_realised_outcome(record, current)
                        rows.append(record)
                        continue
                    factor_information = pair_rows.set_index("history_index").reindex(history.index)
                    valid_factor = (
                        factor_information["factor_distance"].notna()
                        & factor_information["factor_median_rho"].notna()
                    )
                    history = history.loc[valid_factor].copy()
                    factor_information = factor_information.loc[valid_factor].copy()
                    if len(history) < FIXED_NEAREST_K:
                        record["basegap_history_days"] = int(len(history))
                        record["factor_pair_valid_history_days"] = int(len(history))
                        record["basegap_reason"] = "factor_pair_insufficient_history"
                        _finalise_realised_outcome(record, current)
                        rows.append(record)
                        continue
                path_distances = _path_distance(history, current=current, path_features=path_features)
                if factor_weight == 0.0:
                    combined = path_distances
                else:
                    factor_distances = factor_information["factor_distance"].astype(float)
                    combined = combine_neighbour_distances(
                        path_distances,
                        factor_distances,
                        path_weight=path_weight,
                        factor_weight=factor_weight,
                    )
                nearest_index = combined.sort_values(kind="mergesort").index[:FIXED_NEAREST_K]
                _apply_neighbours(
                    record=record,
                    ordered=ordered,
                    current_position=position,
                    history=history,
                    nearest_index=nearest_index,
                    path_distance=path_distances,
                    factor_information=factor_information,
                )
        _finalise_realised_outcome(record, current)
        rows.append(record)
    result = pd.DataFrame(rows)
    active = result.loc[result["selection_active"]]
    if not active.empty and not (
        pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date
    ).all():
        raise RuntimeError("factor-similarity replay leakage: a neighbour is not strictly before its score day")
    return result


def run_recent60_diagnostic(panel: pd.DataFrame, *, path_features: Sequence[str], variant: str) -> pd.DataFrame:
    """Diagnostic that always scores the newest 60 valid history days."""

    ordered = panel.sort_values("score_day").reset_index(drop=True).copy()
    ordered["state_available"] = ordered.get("state_available", True)
    ordered["state_available"] = ordered["state_available"].fillna(False).astype(bool)
    rows: list[dict[str, object]] = []
    for position, current in ordered.iterrows():
        score_day = pd.Timestamp(current["score_day"]).date()
        record = _new_record(
            score_day=score_day,
            variant=variant,
            position=position,
            ordered=ordered,
            path_weight=None,
            factor_weight=None,
        )
        current_values = current[list(path_features)].to_numpy(dtype=float)
        if not bool(current["state_available"]):
            record["basegap_reason"] = "feature_missing"
        elif not np.isfinite(current_values).all():
            record["basegap_reason"] = "feature_na"
        else:
            history = _valid_history(ordered, position=position, path_features=path_features)
            if len(history) < RECENT_DIAGNOSTIC_DAYS:
                record["basegap_history_days"] = int(len(history))
                record["basegap_reason"] = "insufficient_history"
            else:
                nearest_index = history.tail(RECENT_DIAGNOSTIC_DAYS).index
                _apply_neighbours(
                    record=record,
                    ordered=ordered,
                    current_position=position,
                    history=history,
                    nearest_index=nearest_index,
                    path_distance=None,
                    factor_information=None,
                )
        _finalise_realised_outcome(record, current)
        rows.append(record)
    result = pd.DataFrame(rows)
    active = result.loc[result["selection_active"]]
    if not active.empty and not (
        pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date
    ).all():
        raise RuntimeError("recentness diagnostic leakage: a neighbour is not strictly before its score day")
    return result


def build_summary_metrics(all_daily: pd.DataFrame, variants: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Summarise every pre-registered variant, retaining rank as a first-class metric."""

    diagnostic_columns = (
        "neighbor_mean_age_trading_days",
        "neighbor_median_age_trading_days",
        "neighbor_recent20_share",
        "neighbor_mean_path_distance",
        "neighbor_mean_factor_distance",
        "neighbor_median_factor_rho",
        "neighbor_min_shared_codes",
        "neighbor_min_valid_factors",
    )
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, str | None, str | None]] = [("full", None, None), *SPLITS]
    for variant in variants:
        variant_daily = all_daily.loc[all_daily["variant"] == variant["variant"]].copy()
        metadata = {
            **dict(variant),
            "lookback_days": FIXED_LOOKBACK_DAYS,
            "nearest_k": FIXED_NEAREST_K,
            "min_periods": FIXED_MIN_PERIODS,
            "metric": FIXED_METRIC,
            "factor_distance_definition": "1-median_f(rank_pearson_on_shared_score_codes)",
        }
        for scope, start, end in scopes:
            for active_only in (False, True):
                frame = _scope_frame(variant_daily, start=start, end=end, active_only=active_only)
                row = _scope_metrics(frame, variant_row=metadata, scope=scope, active_only=active_only)
                for column in diagnostic_columns:
                    values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
                    row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else None
                rows.append(row)
    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["variant"].eq("P0_path44_only")].set_index(["scope", "active_only"])
    delta_rows: list[float | None] = []
    for _, row in summary.iterrows():
        key = (row["scope"], row["active_only"])
        baseline_rank = baseline.at[key, "mean_rank"] if key in baseline.index else None
        own_rank = row.get("mean_rank")
        delta_rows.append(
            float(baseline_rank - own_rank)
            if pd.notna(baseline_rank) and pd.notna(own_rank)
            else None
        )
    summary["mean_rank_improvement_vs_p0"] = delta_rows
    return summary


def rank_design_variants(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Select once, on design mean rank only, among P0/P25/P50."""

    design = summary.loc[
        (summary["scope"] == "design")
        & summary["active_only"]
        & summary["selection_eligible"]
        & (summary["n_days"] > 0)
    ].copy()
    if design.empty:
        raise RuntimeError("no eligible active design variants")
    baseline = design.loc[design["variant"] == "P0_path44_only"]
    if len(baseline) != 1:
        raise RuntimeError("expected exactly one active design P0 baseline")
    baseline_rank = float(baseline.iloc[0]["mean_rank"])
    design = design.sort_values(
        ["mean_rank", "hit_rate", "mean_regret_bp", "selection_alpha_mean_bp", "factor_weight", "variant"],
        ascending=[True, False, True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    design.insert(0, "design_rank", np.arange(1, len(design) + 1))
    factor_candidates = design.loc[
        design["variant"].isin(["P25_path75_factor25", "P50_path50_factor50"])
        & (design["mean_rank"] < baseline_rank)
    ]
    if factor_candidates.empty:
        selection = {
            "design_selected_variant": "P0_path44_only",
            "selection_reason": "no_factor_assisted_variant_strictly_reduced_design_mean_rank",
            "baseline_mean_rank": baseline_rank,
            "design_selected_mean_rank": baseline_rank,
        }
    else:
        winner = factor_candidates.iloc[0]
        selection = {
            "design_selected_variant": str(winner["variant"]),
            "selection_reason": "lowest_design_mean_rank_among_strict_factor_improvers",
            "baseline_mean_rank": baseline_rank,
            "design_selected_mean_rank": float(winner["mean_rank"]),
            "design_mean_rank_improvement": float(baseline_rank - float(winner["mean_rank"])),
        }
    return design, selection


def _plot_mean_rank(run_root: Path, summary: pd.DataFrame) -> None:
    active = summary.loc[summary["active_only"] & summary["scope"].isin(["design", "validation", "final_oos"])].copy()
    if active.empty:
        return
    variants = [item["variant"] for item in VARIANTS]
    scopes = ["design", "validation", "final_oos"]
    figure, axes = plt.subplots(1, len(scopes), figsize=(5.2 * len(scopes), 4.8), sharey=True)
    for axis, scope in zip(np.atleast_1d(axes), scopes):
        frame = active.loc[active["scope"] == scope].set_index("variant").reindex(variants)
        values = pd.to_numeric(frame["mean_rank"], errors="coerce")
        bars = axis.bar(np.arange(len(frame)), values, color=["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#9C755F"])
        axis.set_title(scope)
        axis.set_xticks(np.arange(len(frame)), labels=[value.replace("_", "\n") for value in variants], rotation=0, fontsize=8)
        axis.set_ylabel("mean selected rank (lower is better)")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            if pd.notna(value):
                axis.text(bar.get_x() + bar.get_width() / 2.0, float(value) + 0.01, f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    figure.suptitle("Pre-registered factor-rank neighbour experiments")
    figure.tight_layout()
    figure.savefig(run_root / "mean_rank_by_scope.png", dpi=160)
    plt.close(figure)


def _plot_nav(run_root: Path, daily: pd.DataFrame, selected_variant: str) -> None:
    variants = ["P0_path44_only"]
    if selected_variant not in variants:
        variants.append(selected_variant)
    relevant = daily.loc[daily["variant"].isin(variants) & daily["execution_metadata_complete"]].copy()
    if relevant.empty:
        return
    figure, axis = plt.subplots(figsize=(13, 6))
    for variant in variants:
        frame = relevant.loc[relevant["variant"] == variant].sort_values("score_day")
        if frame.empty:
            continue
        axis.plot(pd.to_datetime(frame["score_day"]), (1.0 + frame["selected_return"].astype(float)).cumprod(), label=variant)
    reference = relevant.loc[relevant["variant"] == "P0_path44_only"].sort_values("score_day")
    if not reference.empty:
        axis.plot(pd.to_datetime(reference["score_day"]), (1.0 + reference["realized_return_Regsim"].astype(float)).cumprod(), label="Regsim", linestyle="--")
        axis.plot(pd.to_datetime(reference["score_day"]), (1.0 + reference["equal_weight_return"].astype(float)).cumprod(), label="three-candidate equal weight", linestyle=":", color="black")
    for _, _, boundary in SPLITS[:-1]:
        axis.axvline(pd.Timestamp(boundary), color="grey", linewidth=0.7, alpha=0.7)
    axis.set_title("Factor-rank BaseGap selector-shadow-return")
    axis.set_ylabel("Cumulative NAV")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(run_root / "nav_baseline_selected_regsim_equal.png", dpi=160)
    plt.close(figure)


def _plot_neighbor_age(run_root: Path, summary: pd.DataFrame) -> None:
    active = summary.loc[summary["active_only"] & (summary["scope"] == "full")].copy()
    if active.empty:
        return
    frame = active.set_index("variant").reindex([item["variant"] for item in VARIANTS])
    values = pd.to_numeric(frame["neighbor_mean_age_trading_days_mean"], errors="coerce")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    bars = axis.bar(np.arange(len(frame)), values, color="#76B7B2")
    axis.set_xticks(np.arange(len(frame)), labels=[value.replace("_", "\n") for value in frame.index], fontsize=8)
    axis.set_ylabel("mean neighbour age (trading days)")
    axis.set_title("Neighbour age audit: factor similarity versus recentness")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if pd.notna(value):
            axis.text(bar.get_x() + bar.get_width() / 2.0, float(value) + 1.0, f"{float(value):.1f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    figure.savefig(run_root / "neighbor_age_by_variant.png", dpi=160)
    plt.close(figure)


def _task_state(*, phase: str, run_root: Path, selection: Mapping[str, object] | None = None) -> str:
    selected = str(selection.get("design_selected_variant")) if selection else "pending"
    return "\n".join(
        [
            "# Task State",
            "",
            "## Objective",
            "",
            "- Test whether direct live50 factor-rank similarity improves BaseGap mean selected rank without changing the selector contract.",
            "",
            "## Risk Level",
            "",
            "- medium research-only; live assets remain untouched.",
            "",
            "## Current Verified Facts",
            "",
            f"- phase: {phase}",
            f"- output: {run_root}",
            "- fixed selector: path44 / 240 / Top60 / trim20_lcb10 / direct three-candidate argmax.",
            "- factor distance: median 1-rho over 50 day-local rank vectors aligned by shared score code.",
            "",
            "## Files Changed",
            "",
            "- only this isolated research run root; no live/config/DB/scheduler/state/result path is written.",
            "",
            "## Open Risks",
            "",
            "- factor ranks may encode stable instrument identity or recency rather than a transferable regime signal.",
            "- T1430 input is not a proof of strict 14:29 operational availability.",
            "",
            "## Next Action",
            "",
            f"- design-selected variant: {selected}; validation/final OOS are reporting-only.",
            "",
        ]
    )


def _write_results_markdown(
    *,
    run_root: Path,
    selection: Mapping[str, object],
    summary: pd.DataFrame,
) -> None:
    active = summary.loc[summary["active_only"] & summary["scope"].isin(["design", "validation", "final_oos"])].copy()
    lines = [
        "# Frozen direct factor-rank BaseGap replay",
        "",
        "## Contract",
        "",
        "- Frozen three-candidate selector source and frozen live50 T1430 factor store only.",
        "- Fixed BaseGap: path44, lookback=240, Top-60, trim20_lcb10, equal neighbour weighting, direct argmax.",
        "- No threshold, Champion, Robust, Fusion, model, factor, strategy, mask, DB, scheduler, or production write.",
        "- Factor distance for T/history H: median over 50 factors of `1 - Pearson(day-local percentile ranks on U(T) intersection U(H))`.",
        "- P25/P50 are eligible only on active design mean rank; factor-only and Recent60 are diagnostics only.",
        "",
        "## Design Selection",
        "",
        f"- selected: `{selection['design_selected_variant']}`",
        f"- reason: `{selection['selection_reason']}`",
        "",
        "## Active Metrics",
        "",
        "| variant | scope | n | mean rank | delta rank vs P0 | hit rate | regret bp | alpha vs EW bp/day | vs Regsim bp/day | MDD | mean neighbour age |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in active.iterrows():
        lines.append(
            "| {variant} | {scope} | {n_days} | {mean_rank:.3f} | {delta:.3f} | {hit_rate:.3f} | {regret:.3f} | {alpha:.3f} | {regsim:.3f} | {mdd:.3%} | {age:.1f} |".format(
                variant=row["variant"],
                scope=row["scope"],
                n_days=int(row["n_days"]),
                mean_rank=float(row["mean_rank"]),
                delta=float(row["mean_rank_improvement_vs_p0"]),
                hit_rate=float(row["hit_rate"]),
                regret=float(row["mean_regret_bp"]),
                alpha=float(row["selection_alpha_mean_bp"]),
                regsim=float(row["vs_Regsim_mean_bp"]),
                mdd=float(row["selected_max_drawdown"]),
                age=float(row["neighbor_mean_age_trading_days_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is selector-shadow-return research, not a new single-account execution replay.",
            "- Final OOS is reporting-only and has previously been viewed in earlier BaseGap research; it is not a clean project-level blind holdout.",
            "- No result here authorises a live change. A fixed winner would require fresh forward-shadow evidence and separate live-change approval.",
            "",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def run_replay(
    *,
    source_run: Path = DEFAULT_SOURCE_RUN,
    stage_a_run: Path = DEFAULT_STAGE_A_RUN,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str | None = None,
) -> dict[str, object]:
    """Run the entire pre-registered research experiment once."""

    source_run = _assert_locked_source(source_run)
    stage_a_run = _assert_locked_stage_a(stage_a_run)
    run_root = _make_run_root(output_root, run_name)
    run_root.mkdir(parents=True, exist_ok=False)
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
    (run_root / "task_state.md").write_text(_task_state(phase="running", run_root=run_root), encoding="utf-8")
    try:
        config, selector_verification = validate_frozen_source(source_run)
        panel, path_features = prepare_panel(config)
        factor_names = load_factor_contract()
        factor_root, factor_verification = verify_factor_freeze(panel["score_day"].tolist())
        factor_slices, factor_input_audit = build_factor_rank_slices(
            input_root=source_run / "input_snapshot",
            factor_root=factor_root,
            score_days=panel["score_day"].tolist(),
            factor_names=factor_names,
        )
        factor_cache, factor_pair_audit = build_factor_distance_cache(
            panel,
            path_features=path_features,
            factor_slices=factor_slices,
        )
        daily_variants: list[pd.DataFrame] = []
        for variant in VARIANTS:
            name = str(variant["variant"])
            if name == "Recent60_diagnostic":
                daily = run_recent60_diagnostic(panel, path_features=path_features, variant=name)
            else:
                daily = run_factor_similarity_variant(
                    panel,
                    path_features=path_features,
                    factor_cache=factor_cache,
                    variant=name,
                    path_weight=float(variant["path_weight"]),
                    factor_weight=float(variant["factor_weight"]),
                )
            daily_variants.append(daily)
        all_daily = pd.concat(daily_variants, ignore_index=True)

        baseline_daily = all_daily.loc[all_daily["variant"] == "P0_path44_only"].copy()
        stage_a_daily = pd.read_csv(stage_a_run / "daily_selector_replay.csv")
        stage_a_daily = stage_a_daily.loc[stage_a_daily["variant"] == "lb240_k60_trim20_lcb10"].copy()
        parity = parity_audit(baseline_daily, stage_a_daily, label="P0_path44_vs_stage_a_240_60")

        factor_input_audit.to_csv(run_root / "factor_rank_input_audit.csv", index=False)
        factor_pair_audit.to_csv(run_root / "factor_rank_pair_audit.csv", index=False)
        parity.to_csv(run_root / "p0_stage_a_parity_audit.csv", index=False)
        all_daily.to_csv(run_root / "daily_factor_rank_selector_replay.csv", index=False)
        summary = build_summary_metrics(all_daily, VARIANTS)
        summary.to_csv(run_root / "summary_metrics.csv", index=False)
        design_ranking, selection = rank_design_variants(summary)
        design_ranking.to_csv(run_root / "design_mean_rank_ranking.csv", index=False)
        _write_json(run_root / "design_selection.json", selection)

        _plot_mean_rank(run_root, summary)
        _plot_nav(run_root, all_daily, str(selection["design_selected_variant"]))
        _plot_neighbor_age(run_root, summary)

        factor_pairs = factor_pair_audit.copy()
        factor_pairs["score_day"] = pd.to_datetime(factor_pairs["score_day"], errors="coerce")
        factor_pairs["history_day"] = pd.to_datetime(factor_pairs["history_day"], errors="coerce")
        finite_pairs = factor_pairs.loc[factor_pairs["factor_distance"].notna()].copy()
        active = all_daily.loc[all_daily["selection_active"]].copy()
        leakage_audit = {
            "strict_predecessor_factor_pairs": bool((factor_pairs["history_day"] < factor_pairs["score_day"]).all()),
            "strict_predecessor_selected_neighbours": bool(
                (pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all()
            ),
            "pair_distance_uses_day_local_factor_ranks": True,
            "factor_score_universe_rule": "exact frozen three-way common score codes on each date, aligned by code intersection",
            "min_shared_codes": MIN_SHARED_CODES,
            "min_valid_factors": MIN_VALID_FACTORS,
            "valid_pair_count": int(len(finite_pairs)),
            "invalid_pair_count": int(len(factor_pairs) - len(finite_pairs)),
            "observed_min_shared_codes_valid_pairs": int(finite_pairs["shared_codes"].min()) if not finite_pairs.empty else None,
            "observed_min_valid_factors_valid_pairs": int(finite_pairs["valid_factors"].min()) if not finite_pairs.empty else None,
            "no_return_or_label_input_to_factor_distance": True,
            "current_return_used_only_after_neighbour_selection": True,
            "p0_stage_a_parity_verified": True,
            "final_oos_not_used_for_variant_selection": True,
        }
        _write_json(run_root / "leakage_audit.json", leakage_audit)
        _write_results_markdown(run_root=run_root, selection=selection, summary=summary)

        manifest = {
            "run_class": "research_only_direct_factor_rank_similarity",
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "live_config_written": False,
            "live_result_written": False,
            "model_state_written": False,
            "selector_source_run": str(source_run),
            "stage_a_source_run": str(stage_a_run),
            "selector_source_verification": selector_verification,
            "factor_freeze_verification": factor_verification,
            "factor_contract": {
                "path": str(FACTOR_CONTRACT_PATH),
                "sha256": sha256(FACTOR_CONTRACT_PATH),
                "factor_count": len(factor_names),
            },
            "fixed_basegap_contract": {
                "lookback_days": FIXED_LOOKBACK_DAYS,
                "nearest_k": FIXED_NEAREST_K,
                "min_periods": FIXED_MIN_PERIODS,
                "metric": FIXED_METRIC,
                "margin": 0.0,
                "candidate_order": list(MODELS),
                "selector": "pure symmetric direct BaseGap argmax",
            },
            "factor_distance_contract": {
                "definition": "1 - median Pearson correlation of per-day factor percentile ranks over shared score codes",
                "factor_count": len(factor_names),
                "min_shared_codes": MIN_SHARED_CODES,
                "min_valid_factors": MIN_VALID_FACTORS,
                "uses_factor_return_or_label": False,
            },
            "variants": [dict(item) for item in VARIANTS],
            "selection_rule": {
                "eligible_variants": ["P0_path44_only", "P25_path75_factor25", "P50_path50_factor50"],
                "primary_metric": "lowest active design mean_rank",
                "strict_improvement_required_vs_p0": True,
                "tie_breaks": ["higher_hit_rate", "lower_mean_regret_bp", "higher_selection_alpha", "lower_factor_weight"],
                "validation_and_final_oos": "report_only",
            },
            "splits": [{"name": name, "start": start, "end": end} for name, start, end in SPLITS],
            "design_selection": selection,
            "source_code": [{"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())}],
            "git_head": _safe_git(["rev-parse", "HEAD"]),
            "git_status_porcelain": _safe_git(["status", "--short"]),
        }
        _write_json(run_root / "run_manifest.json", manifest)
        _write_json(
            run_root / "run_status.json",
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
                "research_output_only": True,
                **selection,
            },
        )
        (run_root / "task_state.md").write_text(
            _task_state(phase="completed", run_root=run_root, selection=selection), encoding="utf-8"
        )
        return {
            "run_root": str(run_root),
            "design_selected_variant": str(selection["design_selected_variant"]),
            "baseline_design_mean_rank": float(selection["baseline_mean_rank"]),
            "verified_factor_files": int(factor_verification["verified_used_factor_files"]),
        }
    except Exception as exc:
        _write_json(
            run_root / "run_status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
                "research_output_only": True,
            },
        )
        (run_root / "task_state.md").write_text(_task_state(phase="failed", run_root=run_root), encoding="utf-8")
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
    result = run_replay(
        source_run=args.source_run,
        stage_a_run=args.stage_a_run,
        output_root=args.output_root,
        run_name=args.run_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
