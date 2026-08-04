"""Research-only, flow-corrected 14:29 amount-acceleration depth factor.

The source panel exposes cumulative intraday ``amount``.  This factor first
differences that cumulative series within one `(dt, code)` day, then compares
the early and recent portions of a fixed 10-minute T1430 tail.  It deliberately
does not read a label, a mask/pool, daily data, stock data, a database, or a
filesystem path.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _iter_dt_code_groups


EPS = 1e-8


def _parse_cutoff_time(value: object) -> time:
    if isinstance(value, time):
        return value
    try:
        return time.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "t1430_amount_accel_depth_delta_v2 cutoff_time must be an ISO time"
        ) from exc


def _positive_int(params: dict, key: str, default: int) -> int:
    value = int(params.get(key, default))
    if value <= 0:
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {key} must be > 0")
    return value


def _nonnegative_float(params: dict, key: str, default: float) -> float:
    value = float(params.get(key, default))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {key} must be finite and >= 0")
    return value


def _segment_flow(
    timestamps: pd.Series,
    deltas: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, int]:
    """Allocate each cumulative-amount increment to its overlapping interval.

    A snapshot delta belongs to the elapsed interval ending at that snapshot.
    At a segment boundary we allocate proportionally by elapsed seconds; the
    caller separately requires fresh boundary snapshots to keep this limited
    interpolation local and visible.
    """

    total = 0.0
    contributing_intervals = 0
    for position in range(1, len(timestamps)):
        left = pd.Timestamp(timestamps.iloc[position - 1])
        right = pd.Timestamp(timestamps.iloc[position])
        interval_seconds = (right - left).total_seconds()
        if interval_seconds <= 0.0:
            raise ValueError("t1430_amount_accel_depth_delta_v2 timestamps must be strictly increasing")
        overlap_start = max(left, start)
        overlap_end = min(right, end)
        overlap_seconds = (overlap_end - overlap_start).total_seconds()
        if overlap_seconds <= 0.0:
            continue
        total += float(deltas.iloc[position]) * overlap_seconds / interval_seconds
        contributing_intervals += 1
    return total, contributing_intervals


def _boundary_is_fresh(
    timestamps: pd.Series,
    boundary: pd.Timestamp,
    max_staleness_seconds: int,
) -> bool:
    distances = (timestamps - boundary).abs().dt.total_seconds()
    return bool(not distances.empty and float(distances.min()) <= float(max_staleness_seconds))


def _group_value(
    dt: object,
    code: object,
    group: pd.DataFrame,
    *,
    window_minutes: int,
    recent_minutes: int,
    min_obs_per_segment: int,
    max_boundary_staleness_seconds: int,
    max_cumulative_correction_abs: float,
    max_cumulative_correction_ratio: float,
    cutoff_time: time,
) -> float:
    signal_dt = pd.Timestamp(dt)
    if pd.isna(signal_dt):
        raise ValueError("t1430_amount_accel_depth_delta_v2 received an invalid panel dt")
    signal_day = signal_dt.date()
    cutoff = pd.Timestamp.combine(signal_day, cutoff_time)
    recent_start = cutoff - pd.Timedelta(minutes=recent_minutes)
    window_start = cutoff - pd.Timedelta(minutes=window_minutes)

    frame = group.reset_index().copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    if frame["trade_time"].isna().any():
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {code} has invalid trade_time")
    frame = frame.loc[
        (frame["trade_time"].dt.date == signal_day) & (frame["trade_time"] <= cutoff)
    ].copy()
    # The clean-direct panel intentionally retains a long intraday sequence.
    # This factor only needs a fresh seed immediately before its 10-minute
    # window, so earlier-session counters and auction/session transitions must
    # not enter the cumulative-difference validation.
    required_start = window_start - pd.Timedelta(seconds=max_boundary_staleness_seconds)
    frame = frame.loc[frame["trade_time"] >= required_start].copy()
    if frame.empty:
        return float("nan")
    frame = frame.sort_values(["trade_time", "seq"], kind="mergesort")
    # Multiple snapshots at an identical timestamp are one cumulative-state
    # observation.  Keeping the final sequence value preserves monotonicity
    # without inventing a zero-duration flow interval.
    frame = frame.drop_duplicates("trade_time", keep="last").reset_index(drop=True)

    amount = pd.to_numeric(frame["amount"], errors="coerce")
    if not np.isfinite(amount.to_numpy(dtype=float, copy=False)).all():
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {code} has non-finite cumulative amount")
    if (amount < 0.0).any():
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {code} has negative cumulative amount")
    deltas = amount.diff()
    negative = deltas.iloc[1:] < 0.0
    if negative.any():
        # Vendor snapshots can contain tiny signed adjustments to a cumulative
        # amount.  They remain signed increments in the flow calculation (no
        # clipping or fill).  A material decrease is a reset/data fault and
        # must stop the build rather than silently redefining the factor.
        previous = amount.shift(1).iloc[1:]
        correction_abs = -deltas.iloc[1:]
        within_absolute = correction_abs <= max_cumulative_correction_abs
        within_relative = correction_abs <= previous.abs() * max_cumulative_correction_ratio
        if not (within_absolute.loc[negative] & within_relative.loc[negative]).all():
            raise ValueError(
                f"t1430_amount_accel_depth_delta_v2 {code} cumulative amount correction exceeds tolerance"
            )

    timestamps = frame["trade_time"]
    seed = timestamps.loc[timestamps <= window_start]
    if seed.empty:
        return float("nan")
    if (window_start - pd.Timestamp(seed.iloc[-1])).total_seconds() > max_boundary_staleness_seconds:
        return float("nan")
    if (cutoff - pd.Timestamp(timestamps.iloc[-1])).total_seconds() > max_boundary_staleness_seconds:
        return float("nan")
    if not _boundary_is_fresh(timestamps, recent_start, max_boundary_staleness_seconds):
        return float("nan")

    early_amount, early_intervals = _segment_flow(
        timestamps,
        deltas,
        start=window_start,
        end=recent_start,
    )
    recent_amount, recent_intervals = _segment_flow(
        timestamps,
        deltas,
        start=recent_start,
        end=cutoff,
    )
    if early_intervals < min_obs_per_segment or recent_intervals < min_obs_per_segment:
        return float("nan")
    if early_amount <= 0.0:
        return float("nan")

    recent = frame.loc[(timestamps > recent_start) & (timestamps <= cutoff)].copy()
    bid = pd.to_numeric(recent["bid_volume1"], errors="coerce")
    ask = pd.to_numeric(recent["ask_volume1"], errors="coerce")
    invalid_depth = (bid.notna() & (bid < 0.0)) | (ask.notna() & (ask < 0.0))
    if invalid_depth.any():
        raise ValueError(f"t1430_amount_accel_depth_delta_v2 {code} has negative L1 depth")
    depth = (bid - ask) / (bid + ask + EPS)
    depth = depth.loc[np.isfinite(depth)]
    if len(depth) < min_obs_per_segment:
        return float("nan")

    early_seconds = (recent_start - window_start).total_seconds()
    recent_seconds = (cutoff - recent_start).total_seconds()
    rate_ratio = (recent_amount / recent_seconds) / (early_amount / early_seconds)
    raw = rate_ratio * float(depth.mean())
    if not np.isfinite(raw):
        return float("nan")
    return float(np.sign(raw) * np.log1p(abs(raw)))


@FactorRegistry.register("t1430_amount_accel_depth_delta_v2")
class T1430AmountAccelDepthDeltaV2(Factor):
    """Delta-amount acceleration times recent L1 depth, for research only."""

    name = "t1430_amount_accel_depth_delta_v2"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = {"amount", "bid_volume1", "ask_volume1", "trade_time"}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise KeyError(f"t1430_amount_accel_depth_delta_v2 missing panel columns: {missing}")

        window_minutes = _positive_int(ctx.params, "window_minutes", 10)
        recent_minutes = _positive_int(ctx.params, "recent_minutes", 5)
        if recent_minutes >= window_minutes:
            raise ValueError(
                "t1430_amount_accel_depth_delta_v2 requires window_minutes > recent_minutes"
            )
        min_obs_per_segment = _positive_int(ctx.params, "min_obs_per_segment", 2)
        max_boundary_staleness_seconds = _positive_int(
            ctx.params, "max_boundary_staleness_seconds", 90
        )
        max_cumulative_correction_abs = _nonnegative_float(
            ctx.params, "max_cumulative_correction_abs", 100.0
        )
        max_cumulative_correction_ratio = _nonnegative_float(
            ctx.params, "max_cumulative_correction_ratio", 1e-5
        )
        cutoff_time = _parse_cutoff_time(ctx.params.get("cutoff_time", "14:29:00"))

        rows: list[tuple[object, str, float]] = []
        for dt, code, group in _iter_dt_code_groups(panel):
            value = _group_value(
                dt,
                code,
                group,
                window_minutes=window_minutes,
                recent_minutes=recent_minutes,
                min_obs_per_segment=min_obs_per_segment,
                max_boundary_staleness_seconds=max_boundary_staleness_seconds,
                max_cumulative_correction_abs=max_cumulative_correction_abs,
                max_cumulative_correction_ratio=max_cumulative_correction_ratio,
                cutoff_time=cutoff_time,
            )
            rows.append((dt, str(code), value))
        if not rows:
            raise ValueError("t1430_amount_accel_depth_delta_v2 received an empty panel")
        index = pd.MultiIndex.from_tuples(
            [(dt, code) for dt, code, _ in rows], names=["dt", "code"]
        )
        output = pd.Series([value for _, _, value in rows], index=index, dtype="float64")
        output = output.replace([np.inf, -np.inf], np.nan)
        # A structurally unusable T1430 window (for example, no fresh early
        # segment snapshots anywhere in the market) is a legitimate missing
        # factor date. Preserve its per-code NaNs so downstream OOS auditing
        # can record the gap explicitly; do not turn it into zeros or invent a
        # seed. Input corruption remains fail-fast above.
        output.name = self.output_name(self.name)
        return output
