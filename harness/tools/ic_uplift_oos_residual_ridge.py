"""Research-only chronological residual-feature IC experiment.

The experiment changes only the score model.  It does not vary, read, or
optimise any strategy rule, trading window, fee, benchmark, universe filter,
or mask.  The fixed production mask remains a later, unchanged strategy
evaluation control; this tool reports raw score IC only.

``score`` and ``evaluate`` are intentionally separate commands.  During
``score``, a score-day T only reads feature rows for T and labels dated strictly
before T.  Same-day labels are opened only by ``evaluate`` after all scores are
already written.  This makes the no-target-label-read boundary auditable.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_FACTOR_ROOT = Path(r"D:\cbond_on\factor_data\factors\T1430")
DEFAULT_LABEL_ROOT = Path(r"D:\cbond_on\label_data")
DEFAULT_SCORE_ROOT = Path(r"D:\cbond_on\results\scores\live")
REGSIM_ID = "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708"

BASE_CANDIDATE_FEATURES = (
    "cb_overnight_return_mean_120d",
    "vwap_30m",
    "volatility_scaled_return_v1",
)
PARITY_V2_FEATURE = "parity_adjusted_stock_lag_v2"
WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE = "t1430_w80_liq_vol_balance_45m_l5_v1"
DELTA_AMOUNT_ACCEL_DEPTH_V2_FEATURE = "t1430_w80_amount_accel_depth_delta_10m_l1_v2"
PRIOR_INTRADAY_SHARPE_120D_FEATURE = "daily_prior_intraday_sharpe_120d_v1"
PRIOR_INTRADAY_RETURN_SURPRISE_120D_FEATURE = "daily_prior_intraday_return_surprise_120d_v1"
TAIL_PATH_EFFICIENCY_5M_FEATURE = "tail_path_efficiency_5m_v1"

# These columns are read only from the existing primary FactorStore.  Any
# other frozen research feature must be supplied through an explicit sidecar
# root, so a new factor can never silently fall back to production storage.
PRODUCTION_CANDIDATE_FEATURES = frozenset(
    (*BASE_CANDIDATE_FEATURES, WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE)
)


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    feature_columns: tuple[str, ...]
    target_mode: str  # anchored_residual | joint
    lookback_days: int
    alpha: float
    # ``strict`` retains the original no-gap score/feature contract.  The
    # fallback policy is deliberately opt-in for a separately pre-registered
    # candidate and never changes legacy experiment behavior.
    availability_policy: str = "strict"
    min_train_feature_days: int | None = None
    min_score_coverage: float = 0.0


# Fixed before execution.  The factors were chosen from an earlier residual
# diagnostic, not by the eventual final-reporting metrics produced here.
CANDIDATE_PLAN_SETS: dict[str, tuple[CandidatePlan, ...]] = {
    "base3_v1": (
    CandidatePlan(
        "anchored_ridge_120d_w120_a20",
        ("cb_overnight_return_mean_120d",),
        "anchored_residual",
        120,
        20.0,
    ),
    CandidatePlan(
        "anchored_ridge_120d_vwap_w120_a20",
        ("cb_overnight_return_mean_120d", "vwap_30m"),
        "anchored_residual",
        120,
        20.0,
    ),
    CandidatePlan(
        "anchored_ridge_all3_w120_a20",
        BASE_CANDIDATE_FEATURES,
        "anchored_residual",
        120,
        20.0,
    ),
    CandidatePlan(
        "joint_ridge_all3_w120_a20",
        BASE_CANDIDATE_FEATURES,
        "joint",
        120,
        20.0,
    ),
    ),
    # Frozen before the corrected pool-free factor is scored.  The principal
    # arm is a conservative residual correction; vwap is the only existing
    # companion retained because it passed the earlier residual holdout screen.
    "parity_v2_r1": (
        CandidatePlan(
            "anchored_ridge_parity_v2_w120_a20",
            (PARITY_V2_FEATURE,),
            "anchored_residual",
            120,
            20.0,
        ),
        CandidatePlan(
            "anchored_ridge_parity_v2_vwap_w120_a20",
            (PARITY_V2_FEATURE, "vwap_30m"),
            "anchored_residual",
            120,
            20.0,
        ),
    ),
    # A frozen, read-only diagnostic using a legacy research factor already
    # present in the primary FactorStore.  It is not evidence for live use:
    # the historical FactorStore does not independently prove a 14:29 cutoff.
    "wave80_r1": (
        CandidatePlan(
            "anchored_ridge_w80_liqvol45_w120_a20",
            (WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE,),
            "anchored_residual",
            120,
            20.0,
        ),
        CandidatePlan(
            "anchored_ridge_w80_liqvol45_vwap_w120_a20",
            (WAVE80_LIQ_VOL_BALANCE_45M_L5_FEATURE, "vwap_30m"),
            "anchored_residual",
            120,
            20.0,
        ),
    ),
    # Frozen before the new delta-amount scratch FactorStore is built.  The
    # companion vwap feature is retained solely as the previously frozen
    # residual-model companion, not selected from this candidate's results.
    "delta_amount_v2_r1": (
        CandidatePlan(
            "anchored_ridge_delta_amount_v2_w120_a20",
            (DELTA_AMOUNT_ACCEL_DEPTH_V2_FEATURE,),
            "anchored_residual",
            120,
            20.0,
        ),
        CandidatePlan(
            "anchored_ridge_delta_amount_v2_vwap_w120_a20",
            (DELTA_AMOUNT_ACCEL_DEPTH_V2_FEATURE, "vwap_30m"),
            "anchored_residual",
            120,
            20.0,
        ),
    ),
    # Pre-registered after the delta-amount candidate was rejected for
    # structural T1430 availability gaps.  This one keeps the full Regsim
    # code universe: unavailable feature codes/days fall back exactly to the
    # raw Regsim score rather than being dropped or imputed.
    "prior_intraday_sharpe_120d_fallback_r1": (
        CandidatePlan(
            "anchored_ridge_prior_intraday_sharpe_120d_calendar_fallback_w120_a20",
            (PRIOR_INTRADAY_SHARPE_120D_FEATURE,),
            "anchored_residual",
            120,
            20.0,
            availability_policy="calendar_regsim_fallback",
            min_train_feature_days=96,
            min_score_coverage=0.80,
        ),
    ),
    # Pre-registered independently of the static prior-session Sharpe result.
    # The numerator is current T1430 return, while daily_twap supplies only the
    # strictly prior per-code reference distribution.  The fallback policy is
    # retained so no feature gap can alter Regsim's code universe.
    "prior_intraday_return_surprise_120d_fallback_r1": (
        CandidatePlan(
            "anchored_ridge_prior_intraday_return_surprise_120d_calendar_fallback_w120_a20",
            (PRIOR_INTRADAY_RETURN_SURPRISE_120D_FEATURE,),
            "anchored_residual",
            120,
            20.0,
            availability_policy="calendar_regsim_fallback",
            min_train_feature_days=96,
            min_score_coverage=0.80,
        ),
    ),
    # The exact L1-midpoint formula, six clock-minute buckets, sidecar root,
    # residual target, and fallback policy are fixed before any factor artifact
    # or score/label metric is inspected.
    "tail_path_efficiency_5m_fallback_r1": (
        CandidatePlan(
            "anchored_ridge_tail_path_efficiency_5m_calendar_fallback_w120_a20",
            (TAIL_PATH_EFFICIENCY_5M_FEATURE,),
            "anchored_residual",
            120,
            20.0,
            availability_policy="calendar_regsim_fallback",
            min_train_feature_days=96,
            min_score_coverage=0.80,
        ),
    ),
}


def _plans_for_set(candidate_set: str) -> tuple[CandidatePlan, ...]:
    try:
        return CANDIDATE_PLAN_SETS[candidate_set]
    except KeyError as exc:
        raise ValueError(f"unknown candidate set: {candidate_set}") from exc


def _feature_columns_for_plans(plans: Sequence[CandidatePlan]) -> tuple[str, ...]:
    ordered: list[str] = []
    for plan in plans:
        for column in plan.feature_columns:
            if column not in ordered:
                ordered.append(column)
    return tuple(ordered)


def _finite(value: object) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std <= 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - numeric.mean()) / std


def _factor_path(root: Path, day: str) -> Path:
    return root / day[:7] / f"{day.replace('-', '')}.parquet"


def _score_path(root: Path, day: str) -> Path:
    return root / REGSIM_ID / day[:7] / f"{day}.csv"


def _label_path(root: Path, day: str) -> Path:
    return root / day[:7] / f"{day.replace('-', '')}.parquet"


def _read_factor_store_frame(
    factor_root: Path,
    day: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame | None:
    path = _factor_path(factor_root, day)
    if not path.is_file():
        return None
    requested = ["dt", "code", *feature_columns]
    try:
        frame = pd.read_parquet(path, columns=requested)
    except (KeyError, ValueError):
        # Pandas restores dt/code as a MultiIndex for the current FactorStore.
        # Reading only value columns then resetting index is the compatible path.
        try:
            frame = pd.read_parquet(path, columns=list(feature_columns))
        except (KeyError, ValueError):
            return None
    if isinstance(frame.index, pd.MultiIndex) or frame.index.name in {"dt", "code"}:
        frame = frame.reset_index()
    if "code" not in frame.columns:
        return None
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        return None
    frame = frame[["code", *feature_columns]].copy()
    frame["code"] = frame["code"].astype(str)
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.drop_duplicates("code", keep=False)


def _read_factor_frame(
    factor_root: Path,
    sidecar_factor_root: Path | None,
    day: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame | None:
    primary_features = tuple(
        column for column in feature_columns if column in PRODUCTION_CANDIDATE_FEATURES
    )
    sidecar_features = tuple(
        column for column in feature_columns if column not in PRODUCTION_CANDIDATE_FEATURES
    )
    frames: list[pd.DataFrame] = []
    if primary_features:
        primary = _read_factor_store_frame(factor_root, day, primary_features)
        if primary is None:
            return None
        frames.append(primary)
    if sidecar_features:
        if sidecar_factor_root is None:
            raise ValueError("candidate set requires --sidecar-factor-root")
        sidecar = _read_factor_store_frame(sidecar_factor_root, day, sidecar_features)
        if sidecar is None:
            return None
        frames.append(sidecar)
    if not frames:
        raise ValueError("candidate set has no feature columns")
    frame = frames[0]
    for other in frames[1:]:
        frame = frame.merge(other, on="code", how="inner", validate="one_to_one")
    for column in feature_columns:
        frame[f"{column}__z"] = _zscore(frame[column])
    frame = frame.dropna(subset=[f"{column}__z" for column in feature_columns])
    return frame.drop_duplicates("code", keep=False)


def _read_regsim_score(score_root: Path, day: str) -> pd.DataFrame | None:
    path = _score_path(score_root, day)
    if not path.is_file():
        return None
    score = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    if set(score["trade_date"].astype(str).dropna().unique()) != {day}:
        raise ValueError(f"Regsim score date mismatch: {path}")
    score["code"] = score["code"].astype(str)
    score["regsim_score"] = pd.to_numeric(score["score"], errors="coerce")
    score["regsim_score__z"] = _zscore(score["regsim_score"])
    score = score[["code", "regsim_score", "regsim_score__z"]].dropna()
    return score.drop_duplicates("code", keep=False)


def _read_label(label_root: Path, day: str) -> pd.DataFrame | None:
    path = _label_path(label_root, day)
    if not path.is_file():
        return None
    label = pd.read_parquet(path, columns=["code", "trade_time", "y"])
    timestamp = pd.to_datetime(label["trade_time"], errors="coerce")
    label = label.loc[(timestamp.dt.hour == 14) & (timestamp.dt.minute == 42), ["code", "y"]].copy()
    label["code"] = label["code"].astype(str)
    label["y"] = pd.to_numeric(label["y"], errors="coerce")
    label = label.dropna().drop_duplicates("code", keep=False)
    if len(label) < 30:
        return None
    label["y__z"] = _zscore(label["y"])
    return label.dropna(subset=["y__z"])


def discover_feature_days(factor_root: Path, score_root: Path, start: str, end: str) -> list[str]:
    """Return primary-FactorStore dates that also have a Regsim score."""

    days: list[str] = []
    for path in sorted(factor_root.glob("*/*.parquet")):
        compact = path.stem
        if len(compact) != 8 or not compact.isdigit():
            continue
        day = f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
        if not (start <= day <= end):
            continue
        if _score_path(score_root, day).is_file():
            days.append(day)
    return days


def discover_score_days(score_root: Path, start: str, end: str) -> list[str]:
    """Return the explicit Regsim score calendar used to reject date bridging."""

    model_root = score_root / REGSIM_ID
    days: list[str] = []
    for path in sorted(model_root.glob("*/*.csv")):
        day = path.stem
        if (
            len(day) == 10
            and day[4] == "-"
            and day[7] == "-"
            and day.replace("-", "").isdigit()
            and start <= day <= end
        ):
            days.append(day)
    return days


def _build_feature_sections(
    days: Iterable[str],
    factor_root: Path,
    sidecar_factor_root: Path | None,
    score_root: Path,
    feature_columns: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    sections: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    for day in days:
        factor = _read_factor_frame(factor_root, sidecar_factor_root, day, feature_columns)
        if factor is None:
            skipped[day] = "missing candidate factor columns or unusable factor frame"
            continue
        score = _read_regsim_score(score_root, day)
        if score is None:
            skipped[day] = "missing Regsim score"
            continue
        merged = score.merge(factor, on="code", how="inner", validate="one_to_one")
        if len(merged) < 30:
            skipped[day] = f"only {len(merged)} score/factor matches"
            continue
        merged.insert(0, "score_day", day)
        sections[day] = merged
    return sections, skipped


def training_days(days: Sequence[str], score_index: int, lookback_days: int) -> list[str]:
    """The central causal boundary: never return the current score day."""

    return list(days[max(0, score_index - lookback_days) : score_index])


def _fit_daily_candidates(
    sections: dict[str, pd.DataFrame],
    label_root: Path,
    *,
    plans: Sequence[CandidatePlan],
    feature_columns: Sequence[str],
    score_calendar: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate scores without opening a same-day target label.

    A cache is populated only for days selected into an earlier training window.
    That restriction is intentionally local to this function rather than a
    convention imposed on callers.
    """

    days = list(sections)
    calendar_days = list(score_calendar)
    if calendar_days != sorted(set(calendar_days)):
        raise ValueError("score_calendar must be sorted and unique")
    calendar_index = {day: index for index, day in enumerate(calendar_days)}
    missing_calendar_days = [day for day in days if day not in calendar_index]
    if missing_calendar_days:
        raise ValueError(f"section days missing from score calendar: {missing_calendar_days[:3]}")
    max_lookback = max(plan.lookback_days for plan in plans)
    # A labelled frame arrives only after its score day has passed.  Keeping a
    # bounded deque avoids re-reading/re-merging every one of the 120 training
    # days for every candidate and score day.
    history: deque[tuple[str, pd.DataFrame]] = deque()
    score_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    skipped: dict[str, str] = {}
    for idx, day in enumerate(days):
        if idx > 0:
            prior_day = days[idx - 1]
            # This is intentionally the only label read introduced before
            # scoring ``day``.  It is strictly earlier than ``day``.
            prior_label = _read_label(label_root, prior_day)
            if prior_label is not None:
                prior_train = sections[prior_day].merge(
                    prior_label, on="code", how="inner", validate="one_to_one"
                )
                required = [
                    "regsim_score__z",
                    *(f"{column}__z" for column in feature_columns),
                    "y__z",
                ]
                prior_train = prior_train.dropna(subset=required)
                if len(prior_train) >= 30:
                    history.append((prior_day, prior_train))

        current_calendar_index = calendar_index[day]
        oldest_allowed = calendar_days[max(0, current_calendar_index - max_lookback)]
        while history and history[0][0] < oldest_allowed:
            history.popleft()

        history_by_day = {item_day: item_frame for item_day, item_frame in history}
        history_by_lookback: dict[int, tuple[list[str], pd.DataFrame]] = {}
        unavailable: list[str] = []
        for lookback in sorted({plan.lookback_days for plan in plans}):
            requested_days = training_days(calendar_days, current_calendar_index, lookback)
            if len(requested_days) != lookback:
                unavailable.append(f"requires {lookback} prior score-calendar days")
                continue
            missing_sections = [item_day for item_day in requested_days if item_day not in sections]
            if missing_sections:
                unavailable.append(
                    f"missing score/factor section inside {lookback}-day calendar window: "
                    f"{missing_sections[0]}"
                )
                continue
            missing_labels = [item_day for item_day in requested_days if item_day not in history_by_day]
            if missing_labels:
                unavailable.append(
                    f"missing usable prior label inside {lookback}-day calendar window: "
                    f"{missing_labels[0]}"
                )
                continue
            selected = [(item_day, history_by_day[item_day]) for item_day in requested_days]
            history_by_lookback[lookback] = (
                requested_days,
                pd.concat([item_frame for _, item_frame in selected], ignore_index=True),
            )
        if unavailable:
            skipped[day] = "; ".join(unavailable)
            continue

        current = sections[day].copy()
        current["regsim"] = current["regsim_score"]
        for plan in plans:
            cached_history = history_by_lookback.get(plan.lookback_days)
            if cached_history is None:
                continue
            train_days, train = cached_history
            input_columns = [f"{column}__z" for column in plan.feature_columns]
            if plan.target_mode == "joint":
                input_columns = ["regsim_score__z", *input_columns]
                target = train["y__z"]
            elif plan.target_mode == "anchored_residual":
                target = train["y__z"] - train["regsim_score__z"]
            else:
                raise ValueError(f"unknown target mode: {plan.target_mode}")
            day_sizes = train.groupby("score_day")["code"].transform("size")
            weights = 1.0 / day_sizes.to_numpy(dtype=float)
            model = Ridge(alpha=plan.alpha, fit_intercept=True)
            model.fit(train.loc[:, input_columns], target, sample_weight=weights)
            raw_prediction = pd.Series(model.predict(current.loc[:, input_columns]), index=current.index)
            if plan.target_mode == "anchored_residual":
                current[plan.name] = current["regsim_score__z"] + raw_prediction
            else:
                current[plan.name] = raw_prediction
            audit: dict[str, object] = {
                "score_day": day,
                "candidate": plan.name,
                "target_mode": plan.target_mode,
                "lookback_days": plan.lookback_days,
                "alpha": plan.alpha,
                "train_days_requested": plan.lookback_days,
                "train_days_used": len(train_days),
                "train_days_strict_calendar_window": True,
                "train_feature_max_day": train_days[-1],
                "train_label_max_day": train_days[-1],
                "train_label_max_is_strictly_before_score_day": train_days[-1] < day,
                "train_rows": len(train),
                "intercept": float(model.intercept_),
            }
            for column, coefficient in zip(input_columns, model.coef_, strict=True):
                audit[f"coef__{column}"] = float(coefficient)
            audit_rows.append(audit)
        keep = [
            "score_day",
            "code",
            "regsim",
            *[plan.name for plan in plans if plan.name in current.columns],
        ]
        score_rows.append(current.loc[:, keep])
    if not score_rows:
        raise RuntimeError("score stage produced no rows")
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(audit_rows), {
        "score_stage_skipped": skipped,
        "feature_days": len(days),
        "score_calendar_days": len(calendar_days),
        "score_days": sorted(set(pd.concat(score_rows, ignore_index=True)["score_day"])),
    }


def _build_calendar_fallback_inputs(
    score_calendar: Sequence[str],
    factor_root: Path,
    sidecar_factor_root: Path | None,
    score_root: Path,
    *,
    plan: CandidatePlan,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str], pd.DataFrame]:
    """Read all Regsim days and retain factor sections only when quality passes.

    Unlike the strict path, an unavailable factor section does not remove a
    Regsim day from the output universe.  It is recorded here and becomes a
    score-level fallback later; no missing input is filled.
    """
    base_sections: dict[str, pd.DataFrame] = {}
    feature_sections: dict[str, pd.DataFrame] = {}
    feature_skips: dict[str, str] = {}
    availability_rows: list[dict[str, object]] = []
    for day in score_calendar:
        base = _read_regsim_score(score_root, day)
        if base is None:
            feature_skips[day] = "missing or unusable Regsim score"
            availability_rows.append(
                {
                    "score_day": day,
                    "regsim_codes": 0,
                    "feature_codes": 0,
                    "feature_coverage": 0.0,
                    "feature_usable": False,
                    "feature_status": feature_skips[day],
                }
            )
            continue
        base_sections[day] = base
        factor = _read_factor_frame(
            factor_root,
            sidecar_factor_root,
            day,
            plan.feature_columns,
        )
        if factor is None:
            feature_skips[day] = "missing candidate factor columns or unusable factor frame"
            availability_rows.append(
                {
                    "score_day": day,
                    "regsim_codes": len(base),
                    "feature_codes": 0,
                    "feature_coverage": 0.0,
                    "feature_usable": False,
                    "feature_status": feature_skips[day],
                }
            )
            continue
        merged = base.merge(factor, on="code", how="inner", validate="one_to_one")
        coverage = len(merged) / len(base) if len(base) else 0.0
        if len(merged) < 30:
            feature_skips[day] = f"only {len(merged)} score/factor matches"
        elif coverage < plan.min_score_coverage:
            feature_skips[day] = (
                f"feature coverage {coverage:.6f} below frozen minimum {plan.min_score_coverage:.6f}"
            )
        else:
            merged.insert(0, "score_day", day)
            feature_sections[day] = merged
        availability_rows.append(
            {
                "score_day": day,
                "regsim_codes": len(base),
                "feature_codes": len(merged),
                "feature_coverage": coverage,
                "feature_usable": day in feature_sections,
                "feature_status": feature_skips.get(day, "usable"),
            }
        )
    return base_sections, feature_sections, feature_skips, pd.DataFrame(availability_rows)


def _fit_calendar_fallback_candidate(
    base_sections: dict[str, pd.DataFrame],
    feature_sections: dict[str, pd.DataFrame],
    label_root: Path,
    *,
    plan: CandidatePlan,
    score_calendar: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit one pre-registered no-imputation, full-Regsim-calendar candidate."""
    _validate_calendar_fallback_plan(plan)
    if plan.target_mode != "anchored_residual":
        raise ValueError("calendar fallback scorer currently supports anchored_residual only")

    calendar_days = list(score_calendar)
    if calendar_days != sorted(set(calendar_days)):
        raise ValueError("score_calendar must be sorted and unique")
    history_by_day: dict[str, pd.DataFrame] = {}
    score_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    fallback_days: list[str] = []

    feature_z_columns = [f"{column}__z" for column in plan.feature_columns]
    for current_index, day in enumerate(calendar_days):
        # At T, the only label newly opened is exactly T-1.  A current-day
        # label is never read, including when the current feature is missing.
        if current_index > 0:
            prior_day = calendar_days[current_index - 1]
            prior_section = feature_sections.get(prior_day)
            if prior_section is not None:
                prior_label = _read_label(label_root, prior_day)
                if prior_label is not None:
                    prior_train = prior_section.merge(
                        prior_label,
                        on="code",
                        how="inner",
                        validate="one_to_one",
                    )
                    required = ["regsim_score__z", *feature_z_columns, "y__z"]
                    prior_train = prior_train.dropna(subset=required)
                    if len(prior_train) >= 30:
                        history_by_day[prior_day] = prior_train

        base = base_sections.get(day)
        if base is None:
            # A missing base score cannot have a candidate or a paired metric.
            continue
        candidate = base["regsim_score"].copy()
        requested_days = training_days(calendar_days, current_index, plan.lookback_days)
        current_section = feature_sections.get(day)
        current_coverage = len(current_section) / len(base) if current_section is not None else 0.0
        fitted = False
        fallback_reason: str | None = None
        selected_days: list[str] = []
        train_rows = 0
        intercept: float | None = None
        coefficients: dict[str, float] = {}

        if len(requested_days) != plan.lookback_days:
            fallback_reason = f"requires {plan.lookback_days} prior score-calendar days"
        elif current_section is None:
            fallback_reason = "current score day has no usable factor section"
        else:
            selected_days = [item_day for item_day in requested_days if item_day in history_by_day]
            if len(selected_days) < plan.min_train_feature_days:
                fallback_reason = (
                    f"only {len(selected_days)} usable factor/label days in fixed "
                    f"{plan.lookback_days}-day calendar window; requires {plan.min_train_feature_days}"
                )
            else:
                train = pd.concat([history_by_day[item_day] for item_day in selected_days], ignore_index=True)
                train_rows = len(train)
                target = train["y__z"] - train["regsim_score__z"]
                day_sizes = train.groupby("score_day")["code"].transform("size")
                weights = 1.0 / day_sizes.to_numpy(dtype=float)
                model = Ridge(alpha=plan.alpha, fit_intercept=True)
                model.fit(train.loc[:, feature_z_columns], target, sample_weight=weights)
                intercept = float(model.intercept_)
                coefficients = {
                    f"coef__{column}": float(coefficient)
                    for column, coefficient in zip(feature_z_columns, model.coef_, strict=True)
                }
                correction_z = np.asarray(model.predict(current_section.loc[:, feature_z_columns]), dtype=float)
                if not np.isfinite(correction_z).all():
                    raise RuntimeError("calendar fallback Ridge produced a non-finite correction")
                score_mean = float(base["regsim_score"].mean())
                score_std = float(base["regsim_score"].std(ddof=0))
                if not math.isfinite(score_std) or score_std <= 0.0:
                    raise RuntimeError("Regsim score standard deviation is invalid for raw-scale fallback mapping")
                corrected_raw = (
                    current_section["regsim_score__z"].to_numpy(dtype=float) + correction_z
                ) * score_std + score_mean
                replacement = pd.Series(corrected_raw, index=current_section["code"].to_numpy())
                mapped = base["code"].map(replacement)
                candidate = candidate.where(mapped.isna(), mapped)
                fitted = True
        if not fitted:
            fallback_days.append(day)

        result = pd.DataFrame(
            {
                "score_day": day,
                "code": base["code"].to_numpy(),
                "regsim": base["regsim_score"].to_numpy(dtype=float),
                plan.name: candidate.to_numpy(dtype=float),
            }
        )
        score_rows.append(result)
        audit: dict[str, object] = {
            "score_day": day,
            "candidate": plan.name,
            "target_mode": plan.target_mode,
            "availability_policy": plan.availability_policy,
            "lookback_days": plan.lookback_days,
            "alpha": plan.alpha,
            "train_days_requested": plan.lookback_days,
            "train_days_used": len(selected_days),
            # This legacy field means a complete usable training window for
            # the fallback plan.  The next field separately proves that the
            # 120 candidate slots themselves were exact and never bridged.
            "train_days_strict_calendar_window": len(selected_days) == plan.lookback_days,
            "train_calendar_slots_exact": True,
            "train_feature_days_complete": len(selected_days) == plan.lookback_days,
            "train_feature_days_minimum": plan.min_train_feature_days,
            "train_feature_days_missing": max(0, len(requested_days) - len(selected_days)),
            "train_feature_max_day": max(selected_days) if selected_days else None,
            "train_label_max_day": max(selected_days) if selected_days else None,
            "train_label_max_is_strictly_before_score_day": (
                max(selected_days) < day if selected_days else True
            ),
            "current_regsim_codes": len(base),
            "current_feature_codes": len(current_section) if current_section is not None else 0,
            "current_feature_coverage": current_coverage,
            "fitted": fitted,
            "fallback_reason": fallback_reason,
            "train_rows": train_rows,
            "intercept": intercept,
        }
        audit.update(coefficients)
        audit_rows.append(audit)

    if not score_rows:
        raise RuntimeError("calendar fallback score stage produced no Regsim rows")
    scores = pd.concat(score_rows, ignore_index=True)
    return scores, pd.DataFrame(audit_rows), {
        "score_stage_skipped": {},
        "feature_days": len(feature_sections),
        "score_calendar_days": len(calendar_days),
        "score_days": sorted(set(scores["score_day"])),
        "candidate_availability_policy": plan.availability_policy,
        "fallback_score_days": fallback_days,
        "fallback_score_day_count": len(fallback_days),
    }


def _validate_calendar_fallback_plan(plan: CandidatePlan) -> None:
    """Reject malformed fallback plans before any experiment output is created."""
    if plan.availability_policy != "calendar_regsim_fallback":
        raise ValueError("expected a calendar_regsim_fallback candidate plan")
    if plan.min_train_feature_days is None:
        raise ValueError("calendar fallback plan requires min_train_feature_days")
    if not 1 <= plan.min_train_feature_days <= plan.lookback_days:
        raise ValueError(
            "calendar fallback min_train_feature_days must be within "
            "[1, lookback_days]"
        )
    if not 0.0 <= plan.min_score_coverage <= 1.0:
        raise ValueError("calendar fallback min_score_coverage must be within [0, 1]")


def _build_calendar_fallback_code_audit(
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    feature_sections: dict[str, pd.DataFrame],
    *,
    plan: CandidatePlan,
) -> pd.DataFrame:
    """Make code-level no-imputation/fallback behavior independently auditable."""
    required_score_columns = {"score_day", "code", "regsim", plan.name}
    missing_score_columns = required_score_columns.difference(scores.columns)
    if missing_score_columns:
        raise ValueError(f"fallback score audit missing columns: {sorted(missing_score_columns)}")
    if scores.duplicated(["score_day", "code"]).any():
        raise ValueError("fallback score audit received duplicate score-day/code rows")
    if audit.duplicated(["score_day", "candidate"]).any():
        raise ValueError("fallback walk-forward audit has duplicate score-day/candidate rows")

    day_audit = audit.loc[audit["candidate"] == plan.name].copy()
    if set(day_audit["score_day"].astype(str)) != set(scores["score_day"].astype(str)):
        raise ValueError("fallback walk-forward audit does not cover every emitted score day")
    day_audit = day_audit.set_index("score_day")

    availability_by_day = {
        str(day): set(section["code"].astype(str))
        for day, section in feature_sections.items()
    }
    output = scores.loc[:, ["score_day", "code", "regsim", plan.name]].copy()
    output["score_day"] = output["score_day"].astype(str)
    output["code"] = output["code"].astype(str)
    output["candidate"] = plan.name
    output["availability_policy"] = plan.availability_policy
    output["feature_available"] = [
        code in availability_by_day.get(day, set())
        for day, code in zip(output["score_day"], output["code"], strict=True)
    ]
    output["model_fitted"] = output["score_day"].map(day_audit["fitted"]).astype(bool)
    output["fallback_reason"] = output["score_day"].map(day_audit["fallback_reason"])
    output["used_regsim_fallback"] = (~output["model_fitted"]) | (~output["feature_available"])
    output.loc[
        output["model_fitted"] & ~output["feature_available"], "fallback_reason"
    ] = "code has no usable candidate factor"

    fallback_rows = output.loc[output["used_regsim_fallback"]]
    if not np.array_equal(
        fallback_rows[plan.name].to_numpy(dtype=float),
        fallback_rows["regsim"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError("calendar fallback changed a code that should equal raw Regsim")
    return output.loc[
        :,
        [
            "score_day",
            "code",
            "candidate",
            "availability_policy",
            "feature_available",
            "model_fitted",
            "used_regsim_fallback",
            "fallback_reason",
        ],
    ]


def score(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment root: {root}")
    factor_root = Path(args.factor_root)
    sidecar_factor_root = Path(args.sidecar_factor_root) if args.sidecar_factor_root else None
    score_root = Path(args.score_root)
    label_root = Path(args.label_root)
    plans = _plans_for_set(args.candidate_set)
    feature_columns = _feature_columns_for_plans(plans)
    sidecar_features = [
        column for column in feature_columns if column not in PRODUCTION_CANDIDATE_FEATURES
    ]
    if sidecar_features and sidecar_factor_root is None:
        raise ValueError(
            f"candidate_set {args.candidate_set} requires --sidecar-factor-root for {sidecar_features}"
        )
    score_calendar = discover_score_days(score_root, args.start, args.end)
    if not score_calendar:
        raise RuntimeError("no Regsim score calendar days in the requested range")
    policies = {plan.availability_policy for plan in plans}
    availability_audit: pd.DataFrame | None = None
    fallback_code_audit: pd.DataFrame | None = None
    if policies == {"strict"}:
        # Preserve the legacy strict behavior exactly.  In particular, a
        # missing factor/label section omits a score day rather than turning
        # that day into an implicit Regsim fallback.
        days = discover_feature_days(factor_root, score_root, args.start, args.end)
        sections, feature_skips = _build_feature_sections(
            days,
            factor_root,
            sidecar_factor_root,
            score_root,
            feature_columns,
        )
        if not sections:
            raise RuntimeError("no aligned score/factor sections")
        scores, audit, score_metadata = _fit_daily_candidates(
            sections,
            label_root,
            plans=plans,
            feature_columns=feature_columns,
            score_calendar=score_calendar,
        )
        available_feature_days = len(days)
        usable_feature_sections = len(sections)
        feature_stage_skipped = feature_skips
    elif policies == {"calendar_regsim_fallback"}:
        if len(plans) != 1:
            raise ValueError("calendar_regsim_fallback currently requires exactly one candidate plan")
        plan = plans[0]
        _validate_calendar_fallback_plan(plan)
        base_sections, sections, feature_skips, availability_audit = _build_calendar_fallback_inputs(
            score_calendar,
            factor_root,
            sidecar_factor_root,
            score_root,
            plan=plan,
        )
        invalid_base_days = [
            day
            for day in score_calendar
            if day not in base_sections or len(base_sections[day]) < 30
        ]
        if invalid_base_days:
            raise RuntimeError(
                "calendar fallback requires a valid Regsim score universe on every score-calendar "
                f"day; first invalid day: {invalid_base_days[0]}"
            )
        scores, audit, score_metadata = _fit_calendar_fallback_candidate(
            base_sections,
            sections,
            label_root,
            plan=plan,
            score_calendar=score_calendar,
        )
        expected_pairs = pd.concat(
            [
                section.assign(score_day=day).loc[:, ["score_day", "code"]]
                for day, section in base_sections.items()
            ],
            ignore_index=True,
        )
        actual_pairs = scores.loc[:, ["score_day", "code"]].copy()
        if actual_pairs.duplicated(["score_day", "code"]).any():
            raise RuntimeError("calendar fallback emitted duplicate score-day/code rows")
        if not actual_pairs.equals(expected_pairs):
            raise RuntimeError("calendar fallback OOF universe does not exactly match the Regsim universe")
        availability_audit.insert(1, "candidate", plan.name)
        availability_audit.insert(2, "availability_policy", plan.availability_policy)
        fallback_code_audit = _build_calendar_fallback_code_audit(
            scores,
            audit,
            sections,
            plan=plan,
        )
        fallback_code_count = int(fallback_code_audit["used_regsim_fallback"].sum())
        score_metadata.update(
            {
                "base_regsim_score_days": len(base_sections),
                "base_regsim_rows": int(len(expected_pairs)),
                "oof_score_days": int(scores["score_day"].nunique()),
                "oof_rows": int(len(scores)),
                "oof_matches_base_regsim_universe": True,
                "fallback_code_count": fallback_code_count,
                "fallback_code_share": float(fallback_code_count / len(scores)),
                "availability_audit_path": "availability_audit.csv",
                "fallback_code_audit_path": "fallback_code_audit.csv",
            }
        )
        available_feature_days = len(sections)
        usable_feature_sections = len(sections)
        feature_stage_skipped = feature_skips
    else:
        raise ValueError(
            "candidate set cannot mix strict and calendar_regsim_fallback availability policies"
        )
    root.mkdir(parents=True, exist_ok=False)
    scores.to_parquet(root / "oof_scores.parquet", index=False)
    audit.to_csv(root / "walk_forward_audit.csv", index=False, encoding="utf-8")
    if availability_audit is not None:
        availability_audit.to_csv(root / "availability_audit.csv", index=False, encoding="utf-8")
    if fallback_code_audit is not None:
        fallback_code_audit.to_csv(root / "fallback_code_audit.csv", index=False, encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "stage": "score",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "score-only OOF residual ridge research; fixed strategy rules and masks are out of scope",
        "input_contract": {
            "factor_root": str(factor_root),
            "sidecar_factor_root": str(sidecar_factor_root) if sidecar_factor_root else None,
            "score_root": str(score_root),
            "label_root": str(label_root),
            "base_score_model_id": REGSIM_ID,
            "candidate_set": args.candidate_set,
            "candidate_features": list(feature_columns),
            "feature_time": "factor-store T1430 rows for score day T",
            "label_access": "only training label days strictly before score day T",
            "training_window": "exactly N immediately preceding Regsim score-calendar days; no date bridging",
        },
        "frozen_candidate_plans": [plan.__dict__ for plan in plans],
        "fixed_non_interventions": [
            "strategy01_topk_turnover and every trading rule",
            "buy/sell TWAP windows, fees, benchmark, and turnover",
            "o_0005 and every existing mask/universe filter",
            "live config, scheduler, database, live state, and live output",
        ],
        "availability_policy": sorted(policies),
        "available_feature_days": available_feature_days,
        "usable_feature_sections": usable_feature_sections,
        "feature_stage_skipped": feature_stage_skipped,
        **score_metadata,
        "next_stage": "run evaluate separately; only that stage may open same-day labels",
    }
    (root / "score_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    return root


def _partition(day: str, validation_start: str, final_start: str) -> str:
    if day < validation_start:
        return "development"
    if day < final_start:
        return "validation"
    return "final_reporting"


def evaluate(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    score_path = root / "oof_scores.parquet"
    manifest_path = root / "score_manifest.json"
    if not score_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("score stage artifacts missing; run score first")
    score_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score_frame = pd.read_parquet(score_path)
    if score_frame.duplicated(["score_day", "code"]).any():
        raise ValueError("score artifact contains duplicate score-day/code rows")
    label_root = Path(args.label_root)
    daily_rows: list[dict[str, object]] = []
    predictions = [column for column in score_frame.columns if column not in {"score_day", "code"}]
    availability_policies = score_manifest.get("availability_policy", ["strict"])
    calendar_fallback = availability_policies == ["calendar_regsim_fallback"]
    if calendar_fallback:
        if "regsim" not in predictions:
            raise ValueError("calendar fallback score artifact is missing the Regsim baseline")
        candidate_columns = [column for column in predictions if column != "regsim"]
        if len(candidate_columns) != 1:
            raise ValueError("calendar fallback score artifact must contain exactly one candidate")
        if score_frame[["regsim", *candidate_columns]].isna().any().any():
            raise ValueError("calendar fallback score artifact has missing baseline/candidate scores")
    for day, day_scores in score_frame.groupby("score_day", sort=True):
        # This is the first same-day label access, after score parquet exists.
        label = _read_label(label_root, str(day))
        if label is None:
            continue
        merged = day_scores.merge(label[["code", "y"]], on="code", how="inner", validate="one_to_one")
        if len(merged) < 30:
            continue
        for prediction in predictions:
            subset = merged[[prediction, "y"]].dropna()
            if len(subset) < 30 or subset[prediction].nunique() < 2 or subset["y"].nunique() < 2:
                continue
            top = subset.nlargest(20, prediction)["y"]
            daily_rows.append(
                {
                    "score_day": str(day),
                    "partition": _partition(str(day), args.validation_start, args.final_start),
                    "prediction": prediction,
                    "n": len(subset),
                    "pearson_ic": _finite(subset[prediction].corr(subset["y"], method="pearson")),
                    "rank_ic": _finite(subset[prediction].corr(subset["y"], method="spearman")),
                    "top20_mean_y": _finite(top.mean()),
                }
            )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("evaluation produced no aligned metrics")
    summary_rows: list[dict[str, object]] = []
    for (prediction, partition), group in daily.groupby(["prediction", "partition"], sort=True):
        row: dict[str, object] = {
            "prediction": prediction,
            "partition": partition,
            "valid_days": int(len(group)),
            "first_score_day": str(group["score_day"].min()),
            "last_score_day": str(group["score_day"].max()),
            "avg_n": float(group["n"].mean()),
        }
        for metric in ("pearson_ic", "rank_ic", "top20_mean_y"):
            values = group[metric].dropna()
            row[f"mean_{metric}"] = _finite(values.mean()) if len(values) else None
            row[f"t_{metric}"] = (
                _finite(values.mean() / values.std(ddof=1) * math.sqrt(len(values)))
                if len(values) > 1 and values.std(ddof=1) > 0
                else None
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["partition", "prediction"])
    baseline = daily.loc[daily["prediction"] == "regsim"].set_index(["score_day", "partition"])
    paired_rows: list[dict[str, object]] = []
    for prediction in sorted(set(daily["prediction"]) - {"regsim"}):
        candidate = daily.loc[daily["prediction"] == prediction].set_index(["score_day", "partition"])
        common = candidate.join(baseline[["pearson_ic", "rank_ic", "top20_mean_y"]], how="inner", lsuffix="__candidate", rsuffix="__regsim")
        for partition, group in common.groupby(level="partition", sort=True):
            for metric in ("pearson_ic", "rank_ic", "top20_mean_y"):
                delta = group[f"{metric}__candidate"] - group[f"{metric}__regsim"]
                paired_rows.append(
                    {
                        "prediction": prediction,
                        "partition": partition,
                        "metric": metric,
                        "shared_days": int(len(delta)),
                        "mean_delta": _finite(delta.mean()),
                        "t_delta": _finite(delta.mean() / delta.std(ddof=1) * math.sqrt(len(delta))) if len(delta) > 1 and delta.std(ddof=1) > 0 else None,
                    }
                )
    daily.to_csv(root / "daily_metrics.csv", index=False, encoding="utf-8")
    summary.to_csv(root / "summary_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(paired_rows).to_csv(root / "paired_vs_regsim.csv", index=False, encoding="utf-8")
    universe_scope = (
        "full Regsim score universe on every score-calendar day; unavailable candidate "
        "feature codes/days are retained as exact raw-Regsim fallbacks"
        if calendar_fallback
        else "full score/factor intersection; no mask or strategy rule is varied or optimised"
    )
    evaluation_manifest = {
        "schema_version": 1,
        "stage": "evaluate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "same_day_label_access": "begins only in this command after oof_scores.parquet existed",
        "metric": "daily cross-sectional score vs same-score-day 14:42 y, equal-weighted by day",
        "universe_scope": universe_scope,
        "availability_policy": availability_policies,
        "score_universe_integrity": {
            "oof_matches_base_regsim_universe": score_manifest.get(
                "oof_matches_base_regsim_universe"
            ),
            "fallback_score_day_count": score_manifest.get("fallback_score_day_count"),
            "fallback_code_count": score_manifest.get("fallback_code_count"),
            "fallback_code_share": score_manifest.get("fallback_code_share"),
            "availability_audit_path": score_manifest.get("availability_audit_path"),
            "fallback_code_audit_path": score_manifest.get("fallback_code_audit_path"),
        },
        "fixed_strategy_contract": "not evaluated here; any later strategy evaluation must use the unchanged existing mask and execution contract",
        "time_splits": {
            "validation_start": args.validation_start,
            "final_reporting_start": args.final_start,
            "warning": "final reporting is chronological research evidence, not a prospective live shadow",
        },
    }
    (root / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--output-root", required=True)
    score_parser.add_argument("--factor-root", default=str(DEFAULT_FACTOR_ROOT))
    score_parser.add_argument(
        "--sidecar-factor-root",
        help="research-only FactorStore root for features outside the production factor store",
    )
    score_parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    score_parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    score_parser.add_argument("--start", default="2024-05-08")
    score_parser.add_argument("--end", default="2026-06-05")
    score_parser.add_argument("--candidate-set", choices=sorted(CANDIDATE_PLAN_SETS), default="base3_v1")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--output-root", required=True)
    evaluate_parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    evaluate_parser.add_argument("--validation-start", default="2025-10-08")
    evaluate_parser.add_argument("--final-start", default="2026-05-06")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = score(args) if args.command == "score" else evaluate(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
