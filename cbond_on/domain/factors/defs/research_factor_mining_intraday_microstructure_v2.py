"""Research-only strict-PIT intraday microstructure candidate catalogue.

This module is deliberately not imported by :mod:`defs.__init__`.  It keeps
three candidate families isolated from the production factor pack and consumes
only the panel supplied by ``FactorComputeContext``.  No files, labels, pools,
or results are opened here.

The on-demand ``clean_direct`` panel can label carried historical snapshots as
the requested day.  Every metric below therefore requires both the index date
and the physical ``trade_time`` date to be the score date, retains only the
continuous trading sessions through 14:30, and uses a stable time/sequence
ordering.  Missing or unsafe inputs remain ``NaN``; no missing path is
replaced with a neutral value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_microstructure_v2"
MICROSTRUCTURE_V2_VERSION = "20260803_microstructure_v2"
# The generic research launcher records this conventional name in its
# immutable run manifest; retain the more specific constant in cache keys.
CATALOG_VERSION = MICROSTRUCTURE_V2_VERSION
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 30)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research candidate in the microstructure catalogue."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str,
    signals: Iterable[str],
    hypothesis: str,
) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(
            family=family,
            signal=signal,
            kernel=KERNEL_NAME,
            hypothesis=hypothesis,
        )
        for signal in signals
    )


_QUOTE_PATH_SIGNALS = (
    "micro_quote_mid_leads_last_lag1",
    "micro_quote_last_leads_mid_lag1",
    "micro_quote_high_time_gap",
    "micro_quote_low_time_gap",
    "micro_quote_direction_disagreement_share",
    "micro_quote_divergence_crossing_rate",
    "micro_quote_disagreement_persistence",
    "micro_quote_phase_lead_shift",
)
_EVENT_CLOCK_SIGNALS = (
    "micro_event_intensity_dispersion",
    "micro_event_early_late_intensity_shift",
    "micro_event_longest_lull_share",
    "micro_event_lull_release_move_ratio",
    "micro_event_per_trade_impact_phase_shift",
    "micro_event_trade_abs_return_corr",
    "micro_event_zero_trade_move_share",
    "micro_event_burst_followthrough",
)
_DEPTH_CENTROID_SIGNALS = (
    "micro_depth_bid_centroid_relocation",
    "micro_depth_ask_centroid_relocation",
    "micro_depth_centroid_asymmetry_switch",
    "micro_depth_centroid_side_comovement",
    "micro_depth_centroid_asymmetry_leads_return",
    "micro_depth_centroid_extreme_order_gap",
    "micro_depth_centroid_joint_dispersion",
    "micro_depth_centroid_price_rank_coupling",
)


_CATALOG = (
    _entries(
        "quote_trade_path_asynchrony",
        _QUOTE_PATH_SIGNALS,
        "The timing and direction of best-quote price discovery can differ from the traded last-price path.",
    )
    + _entries(
        "event_clock_price_discovery",
        _EVENT_CLOCK_SIGNALS,
        "Safe cumulative trade-count increments define an event clock whose price response need not match clock-time activity.",
    )
    + _entries(
        "depth_centroid_relocation",
        _DEPTH_CENTROID_SIGNALS,
        "Five-level liquidity can relocate away from or toward the touch without changing total displayed depth.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "quote_trade_path_asynchrony": _QUOTE_PATH_SIGNALS,
    "event_clock_price_discovery": _EVENT_CLOCK_SIGNALS,
    "depth_centroid_relocation": _DEPTH_CENTROID_SIGNALS,
}
_FAMILY_REQUIRED_COLUMNS = {
    "quote_trade_path_asynchrony": ("last", "ask_price1", "bid_price1"),
    "event_clock_price_discovery": ("last", "num_trades"),
    "depth_centroid_relocation": (
        "last",
        *tuple(
            column
            for level in range(1, 6)
            for column in (
                f"ask_price{level}",
                f"bid_price{level}",
                f"ask_volume{level}",
                f"bid_volume{level}",
            )
        ),
    ),
}
_FULL_BATCH_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            "trade_time",
            *(
                column
                for family_columns in _FAMILY_REQUIRED_COLUMNS.values()
                for column in family_columns
            ),
        )
    )
)


def factor_mining_intraday_microstructure_v2_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only microstructure catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return factor_mining_intraday_microstructure_v2_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve one score date without trusting rolling panel labels alone."""

    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        day = pd.Timestamp(raw_day)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
    if panel.empty:
        return None
    labels = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(labels[labels.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _is_continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _score_day_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Return only physical score-day continuous-session snapshots through 14:30."""

    cache_key = f"{KERNEL_NAME}:{MICROSTRUCTURE_V2_VERSION}:score_day_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    panel = ensure_trade_time(ctx.panel)
    frame = panel.reset_index().copy(deep=False)
    target = _signal_date_from_panel(panel)
    if target is None:
        out = frame.iloc[0:0].copy()
    else:
        labels = pd.to_datetime(frame["dt"], errors="coerce")
        if not frame.empty and not bool((labels.dt.normalize() == target).any()):
            raise ValueError(
                f"{KERNEL_NAME} has no indexed panel rows for signal day {target.date().isoformat()}"
            )
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        clocks = timestamps.dt.time
        continuous = timestamps.notna() & clocks.map(
            lambda clock: _is_continuous_session(clock) if pd.notna(clock) else False
        )
        keep = (
            labels.notna()
            & (labels.dt.normalize() == target)
            & timestamps.notna()
            & (timestamps.dt.normalize() == target)
            & continuous
        )
        out = frame.loc[keep].copy()
        out["trade_time"] = timestamps.loc[keep]
        if not out.empty:
            out = out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = out
    return out


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _numeric(g: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")


def _valid_prices(g: pd.DataFrame, *, min_count: int) -> np.ndarray | None:
    prices = _numeric(g, "last")
    if prices.size < min_count or not (np.isfinite(prices).all() and (prices > 0.0).all()):
        return None
    return prices


def _continuous_seconds(g: pd.DataFrame) -> np.ndarray | None:
    times = pd.to_datetime(g["trade_time"], errors="coerce")
    if len(times) < 2 or times.isna().any():
        return None
    morning_start = times.dt.normalize() + pd.Timedelta(hours=9, minutes=30)
    afternoon_start = times.dt.normalize() + pd.Timedelta(hours=13)
    clocks = times.dt.time.to_numpy()
    seconds = np.empty(len(times), dtype="float64")
    morning_seconds = 2.0 * 60.0 * 60.0
    for index, clock in enumerate(clocks):
        if _MORNING_START <= clock <= _MORNING_END:
            seconds[index] = float((times.iloc[index] - morning_start.iloc[index]).total_seconds())
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            seconds[index] = morning_seconds + float(
                (times.iloc[index] - afternoon_start.iloc[index]).total_seconds()
            )
        else:
            return None
    differences = np.diff(seconds)
    # The 11:30 and 13:00 session endpoints intentionally share the same
    # compressed-clock position.  A zero-length interval therefore marks a
    # session boundary, rather than a reversed time path.  Event-clock
    # metrics discard it below because an intensity is not defined there.
    if not np.isfinite(seconds).all() or (differences < 0.0).any():
        return None
    return seconds


def _time_positions(g: pd.DataFrame) -> np.ndarray | None:
    seconds = _continuous_seconds(g)
    if seconds is None:
        return None
    total = float(seconds[-1] - seconds[0])
    if total <= _EPS:
        return None
    return (seconds - seconds[0]) / total


def _early_late_masks(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    times = pd.to_datetime(g["trade_time"], errors="coerce")
    if times.isna().any():
        return None
    clocks = times.dt.time.to_numpy()
    return clocks <= _EARLY_END, clocks >= _LATE_START


def _corr(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    a = left[valid]
    b = right[valid]
    if float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _coverage_required(total_rows: int, *, minimum: int = 12) -> int:
    return max(minimum, int(np.ceil(0.75 * total_rows)))


def _valid_best_quote_frame(g: pd.DataFrame) -> pd.DataFrame | None:
    prices = _numeric(g, "last")
    ask = _numeric(g, "ask_price1")
    bid = _numeric(g, "bid_price1")
    valid = (
        np.isfinite(prices)
        & (prices > 0.0)
        & np.isfinite(ask)
        & np.isfinite(bid)
        & (ask > 0.0)
        & (bid > 0.0)
        & (ask >= bid)
    )
    if int(valid.sum()) < _coverage_required(len(g)):
        return None
    out = g.loc[valid].copy()
    if _continuous_seconds(out) is None:
        return None
    return out


def _safe_num_trades_increments(g: pd.DataFrame) -> np.ndarray | None:
    values = _numeric(g, "num_trades")
    if values.size < 12 or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    return increments


def _valid_depth_frame(g: pd.DataFrame) -> pd.DataFrame | None:
    prices = _numeric(g, "last")
    ask_price = np.column_stack([_numeric(g, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(g, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(g, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(g, f"bid_volume{level}") for level in range(1, 6)])
    valid = (
        np.isfinite(prices)
        & (prices > 0.0)
        & np.isfinite(ask_price).all(axis=1)
        & np.isfinite(bid_price).all(axis=1)
        & np.isfinite(ask_depth).all(axis=1)
        & np.isfinite(bid_depth).all(axis=1)
        & (ask_price > 0.0).all(axis=1)
        & (bid_price > 0.0).all(axis=1)
        & (ask_depth >= 0.0).all(axis=1)
        & (bid_depth >= 0.0).all(axis=1)
        & (ask_price[:, 0] >= bid_price[:, 0])
        & (np.diff(ask_price, axis=1) >= 0.0).all(axis=1)
        & (np.diff(bid_price, axis=1) <= 0.0).all(axis=1)
        & (ask_depth.sum(axis=1) > _EPS)
        & (bid_depth.sum(axis=1) > _EPS)
    )
    if int(valid.sum()) < _coverage_required(len(g)):
        return None
    out = g.loc[valid].copy()
    if _continuous_seconds(out) is None:
        return None
    return out


def _quote_trade_path_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_QUOTE_PATH_SIGNALS)
    frame = _valid_best_quote_frame(g)
    if frame is None:
        return out
    last = _valid_prices(frame, min_count=12)
    positions = _time_positions(frame)
    masks = _early_late_masks(frame)
    if last is None or positions is None or masks is None:
        return out
    ask = _numeric(frame, "ask_price1")
    bid = _numeric(frame, "bid_price1")
    mid = (ask + bid) / 2.0
    if not (np.isfinite(mid).all() and (mid > 0.0).all()):
        return out
    last_return = np.diff(np.log(last))
    mid_return = np.diff(np.log(mid))
    if len(last_return) < 10 or not (np.isfinite(last_return).all() and np.isfinite(mid_return).all()):
        return out
    out["micro_quote_mid_leads_last_lag1"] = _corr(mid_return[:-1], last_return[1:])
    out["micro_quote_last_leads_mid_lag1"] = _corr(last_return[:-1], mid_return[1:])
    out["micro_quote_high_time_gap"] = float(positions[int(np.argmax(mid))] - positions[int(np.argmax(last))])
    out["micro_quote_low_time_gap"] = float(positions[int(np.argmin(mid))] - positions[int(np.argmin(last))])

    nonzero_direction = (np.sign(last_return) != 0.0) & (np.sign(mid_return) != 0.0)
    if int(nonzero_direction.sum()) >= 3:
        out["micro_quote_direction_disagreement_share"] = float(
            np.mean(np.sign(last_return[nonzero_direction]) != np.sign(mid_return[nonzero_direction]))
        )
    divergence = (last - mid) / mid
    divergence_sign = np.sign(divergence)
    divergence_sign = divergence_sign[divergence_sign != 0.0]
    if len(divergence_sign) >= 3:
        out["micro_quote_divergence_crossing_rate"] = float(
            np.mean(divergence_sign[1:] != divergence_sign[:-1])
        )
    return_difference = mid_return - last_return
    out["micro_quote_disagreement_persistence"] = _corr(
        return_difference[:-1],
        return_difference[1:],
    )

    early_rows, late_rows = masks
    early_pair = early_rows[1:-1] & early_rows[2:]
    late_pair = late_rows[1:-1] & late_rows[2:]
    early_lead = _corr(mid_return[:-1][early_pair], last_return[1:][early_pair])
    late_lead = _corr(mid_return[:-1][late_pair], last_return[1:][late_pair])
    if np.isfinite(early_lead) and np.isfinite(late_lead):
        out["micro_quote_phase_lead_shift"] = late_lead - early_lead
    return out


def _longest_lull_share(increments: np.ndarray, seconds: np.ndarray) -> float:
    if len(increments) != len(seconds) or len(increments) < 3:
        return float("nan")
    if not (
        np.isfinite(increments).all()
        and np.isfinite(seconds).all()
        and (seconds >= 0.0).all()
    ):
        return float("nan")
    longest = 0.0
    current = 0.0
    for increment, duration in zip(increments, seconds, strict=True):
        if duration <= _EPS:
            # Do not let a lunch/duplicate-time boundary join two event-time
            # lulls.  There is no duration to attribute to this increment.
            current = 0.0
            continue
        if increment == 0.0:
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    return _safe_div(longest, float(seconds.sum()))


def _lull_release_ratio(increments: np.ndarray, seconds: np.ndarray, returns: np.ndarray) -> float:
    if not (len(increments) == len(seconds) == len(returns)):
        return float("nan")
    if len(increments) < 5:
        return float("nan")
    release_moves: list[float] = []
    positive_moves: list[float] = []
    lull_seconds = 0.0
    for increment, duration, ret in zip(increments, seconds, returns, strict=True):
        if duration <= _EPS:
            # A compressed-clock boundary has no event-time return.  It also
            # breaks a preceding lull rather than carrying it across lunch.
            lull_seconds = 0.0
            continue
        if increment == 0.0:
            lull_seconds += float(duration)
            continue
        move = abs(float(ret))
        positive_moves.append(move)
        if lull_seconds > 0.0:
            release_moves.append(move)
        lull_seconds = 0.0
    if len(release_moves) < 3 or len(positive_moves) < 3:
        return float("nan")
    baseline = float(np.median(positive_moves))
    return _safe_div(float(np.mean(release_moves)), baseline)


def _event_clock_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_EVENT_CLOCK_SIGNALS)
    prices = _valid_prices(g, min_count=12)
    seconds = _continuous_seconds(g)
    increments = _safe_num_trades_increments(g)
    masks = _early_late_masks(g)
    if prices is None or seconds is None or increments is None or masks is None:
        return out
    gaps = np.diff(seconds)
    returns = np.diff(np.log(prices))
    if (
        len(gaps) != len(increments)
        or len(returns) != len(increments)
        or not (np.isfinite(gaps).all() and (gaps >= 0.0).all() and np.isfinite(returns).all())
    ):
        return out
    event_intervals = gaps > _EPS
    if int(event_intervals.sum()) < 3:
        return out
    intensity = increments[event_intervals] / gaps[event_intervals]
    if not np.isfinite(intensity).all():
        return out
    mean_intensity = float(np.mean(intensity))
    out["micro_event_intensity_dispersion"] = _safe_div(float(np.std(intensity, ddof=1)), mean_intensity)

    early_rows, late_rows = masks
    early_intervals = event_intervals & early_rows[1:]
    late_intervals = event_intervals & late_rows[1:]
    if int(early_intervals.sum()) >= 3 and int(late_intervals.sum()) >= 3:
        early_rate = _safe_div(float(increments[early_intervals].sum()), float(gaps[early_intervals].sum()))
        late_rate = _safe_div(float(increments[late_intervals].sum()), float(gaps[late_intervals].sum()))
        if np.isfinite(early_rate) and np.isfinite(late_rate) and early_rate > 0.0 and late_rate > 0.0:
            out["micro_event_early_late_intensity_shift"] = float(np.log(late_rate / early_rate))

        early_impact = _safe_div(float(np.abs(returns[early_intervals]).sum()), float(increments[early_intervals].sum()))
        late_impact = _safe_div(float(np.abs(returns[late_intervals]).sum()), float(increments[late_intervals].sum()))
        if np.isfinite(early_impact) and np.isfinite(late_impact) and early_impact > 0.0 and late_impact > 0.0:
            out["micro_event_per_trade_impact_phase_shift"] = float(np.log(late_impact / early_impact))

    out["micro_event_longest_lull_share"] = _longest_lull_share(increments, gaps)
    out["micro_event_lull_release_move_ratio"] = _lull_release_ratio(increments, gaps, returns)
    out["micro_event_trade_abs_return_corr"] = _corr(
        np.log1p(increments[event_intervals]),
        np.abs(returns[event_intervals]),
    )

    zero_events = event_intervals & (increments == 0.0)
    baseline_move = float(np.median(np.abs(returns[event_intervals])))
    if int(zero_events.sum()) >= 3 and baseline_move > _EPS:
        out["micro_event_zero_trade_move_share"] = float(
            np.mean(np.abs(returns[zero_events]) > baseline_move)
        )

    positive_events = increments[event_intervals & (increments > 0.0)]
    if len(positive_events) >= 6 and len(returns) >= 8:
        threshold = float(np.quantile(positive_events, 0.75))
        consecutive_event_pairs = event_intervals[:-1] & event_intervals[1:]
        current_increments = increments[:-1][consecutive_event_pairs]
        burst = current_increments >= threshold
        contrast = (current_increments > 0.0) & ~burst
        next_returns = returns[1:][consecutive_event_pairs]
        if int(burst.sum()) >= 3 and int(contrast.sum()) >= 3:
            out["micro_event_burst_followthrough"] = float(
                np.mean(next_returns[burst]) - np.mean(next_returns[contrast])
            )
    return out


def _depth_centroids(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    frame = _valid_depth_frame(g)
    if frame is None:
        return None
    last = _valid_prices(frame, min_count=12)
    positions = _time_positions(frame)
    if last is None or positions is None:
        return None
    ask_price = np.column_stack([_numeric(frame, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(frame, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(frame, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(frame, f"bid_volume{level}") for level in range(1, 6)])
    mid = (ask_price[:, 0] + bid_price[:, 0]) / 2.0
    bid_distance = (mid[:, None] - bid_price) / mid[:, None]
    ask_distance = (ask_price - mid[:, None]) / mid[:, None]
    bid_centroid = (bid_depth * bid_distance).sum(axis=1) / bid_depth.sum(axis=1)
    ask_centroid = (ask_depth * ask_distance).sum(axis=1) / ask_depth.sum(axis=1)
    if not (np.isfinite(bid_centroid).all() and np.isfinite(ask_centroid).all()):
        return None
    return last, positions, bid_centroid, ask_centroid


def _depth_centroid_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_DEPTH_CENTROID_SIGNALS)
    frame = _valid_depth_frame(g)
    if frame is None:
        return out
    arrays = _depth_centroids(g)
    masks = _early_late_masks(frame)
    if arrays is None or masks is None:
        return out
    last, positions, bid_centroid, ask_centroid = arrays
    if len(last) != len(frame):
        # Arrays are built from the same independently validated source rows.
        return out
    early, late = masks
    if min(int(early.sum()), int(late.sum())) < 3:
        return out
    asymmetry = bid_centroid - ask_centroid
    joint = bid_centroid + ask_centroid
    out["micro_depth_bid_centroid_relocation"] = float(np.mean(bid_centroid[late]) - np.mean(bid_centroid[early]))
    out["micro_depth_ask_centroid_relocation"] = float(np.mean(ask_centroid[late]) - np.mean(ask_centroid[early]))
    out["micro_depth_centroid_asymmetry_switch"] = float(
        np.mean(asymmetry[late]) - np.mean(asymmetry[early])
    )
    out["micro_depth_centroid_side_comovement"] = _corr(bid_centroid, ask_centroid)
    returns = np.diff(np.log(last))
    if len(returns) >= 10:
        out["micro_depth_centroid_asymmetry_leads_return"] = _corr(asymmetry[:-2], returns[1:])
    out["micro_depth_centroid_extreme_order_gap"] = float(
        positions[int(np.argmax(bid_centroid))] - positions[int(np.argmax(ask_centroid))]
    )
    out["micro_depth_centroid_joint_dispersion"] = _safe_div(
        float(np.std(joint, ddof=1)),
        float(np.mean(joint)),
    )
    price_rank = pd.Series(last, dtype="float64").rank(pct=True, method="average").to_numpy()
    out["micro_depth_centroid_price_rank_coupling"] = _corr(asymmetry, price_rank)
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "quote_trade_path_asynchrony": _quote_trade_path_metrics,
    "event_clock_price_discovery": _event_clock_metrics,
    "depth_centroid_relocation": _depth_centroid_metrics,
}


def _full_batch_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Materialise every family in one grouped traversal when its schema is complete.

    The generic batch asks one concrete signal at a time.  With the full
    microstructure schema present, letting each first family signal separately
    group a multi-million-row panel triples the dominant cost while producing
    identical values.  This cache is intentionally only an optimization: the
    family-specific fallback below retains its narrower missing-column contract
    for standalone calls.
    """

    cache_key = f"{KERNEL_NAME}:{MICROSTRUCTURE_V2_VERSION}:full_batch_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    _require_columns(frame, _FULL_BATCH_REQUIRED_COLUMNS)
    all_signals = tuple(entry.signal for entry in _CATALOG)
    if frame.empty:
        built = pd.DataFrame(index=_empty_index(), columns=all_signals, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            for family, calculator in _FAMILY_CALCULATORS.items():
                row.update(calculator(group))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(all_signals)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache all family signals so specs share a single panel scan."""

    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{MICROSTRUCTURE_V2_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    signals = _FAMILY_SIGNALS[family]
    _require_columns(frame, ("trade_time", *_FAMILY_REQUIRED_COLUMNS[family]))
    if set(_FULL_BATCH_REQUIRED_COLUMNS).issubset(frame.columns):
        # Preserve the family-specific check above.  When the complete schema
        # exists, all concrete batch specs share the one full-frame traversal.
        built = _full_batch_feature_frame(ctx).loc[:, list(signals)].copy()
    elif frame.empty:
        built = pd.DataFrame(index=_empty_index(), columns=signals, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        calculator = _FAMILY_CALCULATORS[family]
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            values = calculator(group)
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            row.update({signal: float(values.get(signal, np.nan)) for signal in signals})
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(signals)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires an explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayMicrostructureV2(Factor):
    """Research-only T1430 microstructure candidate family kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
