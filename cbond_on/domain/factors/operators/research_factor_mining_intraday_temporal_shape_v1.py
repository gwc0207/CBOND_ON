"""Research-only strict-PIT intraday temporal-shape factor catalogue.

This module is intentionally import-only: it is neither imported from
``defs.__init__`` nor referenced by a production, model, or live configuration.
It is a research candidate for the fixed T1430 / 14:30 factor-mining contract.

The sole market-data input is ``FactorComputeContext.panel``.  For each bond it
uses trade-confirmed changes in ``last`` and changes in displayed L1 total depth
to describe *where in the intraday clock* those changes occur.  It does not use
an endpoint return, quote staleness, queue state, event threshold, price-ladder
reprice, path-transform signal, label, score, PnL, file, database, or any daily
or stock context.

All observations must be physical score-day continuous-session snapshots at or
before 14:29.  A duplicate/reversed timestamp, an in-session gap longer than
20 minutes, a ``num_trades`` counter reset, or an invalid trade/quote/depth
observation invalidates the entire bond-day and returns NaN.  Lunch is a known
session boundary and is never treated as an intraday return/depth-adjustment
interval.  The local ``ctx.cache`` is only memoization of values derived from
``ctx.panel``; it is not an additional input source.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_temporal_shape_v1"
CATALOG_VERSION = "20260803_intraday_temporal_shape_v1"
_EPS = 1e-12
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MAX_IN_SESSION_GAP_SECONDS = 20.0 * 60.0
_MIN_ROWS = 12
_MIN_PRICE_EVENTS = 4
_MIN_DEPTH_ADJUSTMENTS = 8
_N_CLOCK_BINS = 6
_MORNING_SECONDS = 2.0 * 60.0 * 60.0
_TOTAL_SESSION_SECONDS = _MORNING_SECONDS + 89.0 * 60.0


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only temporal-shape signal."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_PRICE_ALLOCATION_SIGNALS = (
    "tshape_trade_variation_concentration",
    "tshape_trade_variation_late_tilt",
    "tshape_trade_variation_edge_core",
)
_DEPTH_ALLOCATION_SIGNALS = (
    "tshape_depth_adjustment_concentration",
    "tshape_depth_adjustment_late_tilt",
    "tshape_depth_adjustment_edge_core",
)
_COUPLING_SIGNALS = (
    "tshape_trade_depth_profile_alignment",
    "tshape_trade_depth_temporal_offset",
    "tshape_trade_depth_profile_divergence",
)

_CATALOG = (
    _entries(
        "temporal_trade_variation_allocation",
        _PRICE_ALLOCATION_SIGNALS,
        "Trade-confirmed realized price variation is informative through its allocation across the intraday clock, rather than through a raw endpoint return or a path-transform statistic.",
    )
    + _entries(
        "temporal_book_adjustment_allocation",
        _DEPTH_ALLOCATION_SIGNALS,
        "Displayed L1 depth adjustment has a temporal concentration and clock shape distinct from a static depth level, queue lifecycle, or order-book repricing event.",
    )
    + _entries(
        "temporal_trade_depth_profile_coupling",
        _COUPLING_SIGNALS,
        "The normalized intraday profiles of trade-confirmed price variation and depth adjustment can align, lead, or diverge without using an event clock or a raw return level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# Let a_i be a trade-confirmed absolute log change in last price at interval i,
# d_i be the absolute log change in L1 total depth, and c_j be the center of
# equal-width continuous-session clock bin j.  p_j and q_j normalize a_i and
# d_i by their respective total mass in bin j.  A counter reset, invalid quote,
# invalid price, timestamp gap, or lunch-crossing interval is never repaired.
FORMULAS: dict[str, str] = {
    "tshape_trade_variation_concentration": "sum_j p_j^2, p_j=sum(a_i in bin j)/sum_i a_i",
    "tshape_trade_variation_late_tilt": "sum_j p_j*(c_j-0.5)",
    "tshape_trade_variation_edge_core": "sum_j p_j*cos(2*pi*c_j)",
    "tshape_depth_adjustment_concentration": "sum_j q_j^2, q_j=sum(d_i in bin j)/sum_i d_i",
    "tshape_depth_adjustment_late_tilt": "sum_j q_j*(c_j-0.5)",
    "tshape_depth_adjustment_edge_core": "sum_j q_j*cos(2*pi*c_j)",
    "tshape_trade_depth_profile_alignment": "dot(p,q)/(||p||*||q||)",
    "tshape_trade_depth_temporal_offset": "sum_j (p_j-q_j)*c_j",
    "tshape_trade_depth_profile_divergence": "0.5*sum_j[p_j*log(p_j/m_j)+q_j*log(q_j/m_j)], m_j=(p_j+q_j)/2",
}

_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "last",
    "num_trades",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
)


def factor_mining_intraday_temporal_shape_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research catalogue loaders."""

    return factor_mining_intraday_temporal_shape_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
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


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return only physical score-day observations visible by 14:29:00."""

    checked = ensure_panel_index(panel)
    missing = [column for column in _REQUIRED_PANEL_COLUMNS if column not in checked.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing panel columns: {missing}")
    checked = ensure_trade_time(checked)
    frame = checked.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
    keep = (
        indexed.notna()
        & (indexed.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    return out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")


def _nan_record() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _numeric(group: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(group[column], errors="coerce").to_numpy(dtype="float64")


def _continuous_clock_seconds(times: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Return compressed-session seconds and an explicit morning/afternoon id."""

    parsed = pd.to_datetime(times, errors="coerce")
    if len(parsed) < _MIN_ROWS or parsed.isna().any() or parsed.duplicated().any():
        return None
    if (parsed.diff().dropna() <= pd.Timedelta(0)).any():
        return None

    clocks = parsed.dt.time.to_numpy()
    seconds = np.empty(len(parsed), dtype="float64")
    sessions = np.empty(len(parsed), dtype="int8")
    for index, (stamp, clock) in enumerate(zip(parsed, clocks, strict=True)):
        day = stamp.normalize()
        if _MORNING_START <= clock <= _MORNING_END:
            seconds[index] = float((stamp - (day + pd.Timedelta(hours=9, minutes=30))).total_seconds())
            sessions[index] = 0
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            seconds[index] = _MORNING_SECONDS + float(
                (stamp - (day + pd.Timedelta(hours=13))).total_seconds()
            )
            sessions[index] = 1
        else:
            return None
    if not np.isfinite(seconds).all() or (np.diff(seconds) < 0.0).any():
        return None

    for session in (0, 1):
        local = parsed.iloc[sessions == session]
        if local.empty:
            return None
        gaps = local.diff().dropna().dt.total_seconds().to_numpy(dtype="float64")
        if not np.isfinite(gaps).all() or (gaps <= 0.0).any() or (gaps > _MAX_IN_SESSION_GAP_SECONDS).any():
            return None
    return seconds, sessions


def _validated_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Validate the full per-bond contract before deriving any profile mass."""

    if len(group) < _MIN_ROWS:
        return None
    clock = _continuous_clock_seconds(group["trade_time"])
    if clock is None:
        return None
    seconds, sessions = clock

    last = _numeric(group, "last")
    count = _numeric(group, "num_trades")
    ask_price = _numeric(group, "ask_price1")
    bid_price = _numeric(group, "bid_price1")
    ask_depth = _numeric(group, "ask_volume1")
    bid_depth = _numeric(group, "bid_volume1")
    total_depth = ask_depth + bid_depth
    valid = (
        np.isfinite(last).all()
        and np.isfinite(count).all()
        and np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and (last > 0.0).all()
        and (count >= 0.0).all()
        and (ask_price > 0.0).all()
        and (bid_price > 0.0).all()
        and (ask_price >= bid_price).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (total_depth > _EPS).all()
    )
    if not valid:
        return None
    increments = np.diff(count)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    return seconds, sessions, last, total_depth


def _profile_from_interval_mass(
    seconds: np.ndarray,
    sessions: np.ndarray,
    mass: np.ndarray,
    *,
    minimum_active: int,
) -> np.ndarray | None:
    """Normalize nonnegative interval mass into fixed continuous-clock bins."""

    if len(mass) != len(seconds) - 1 or len(sessions) != len(seconds):
        return None
    same_session = sessions[1:] == sessions[:-1]
    valid_mass = np.isfinite(mass) & (mass >= 0.0) & same_session
    active = valid_mass & (mass > _EPS)
    if int(active.sum()) < minimum_active:
        return None
    endpoints = seconds[1:]
    if not np.isfinite(endpoints).all() or ((endpoints < 0.0) | (endpoints > _TOTAL_SESSION_SECONDS)).any():
        return None
    bin_index = np.floor(endpoints / _TOTAL_SESSION_SECONDS * _N_CLOCK_BINS).astype("int64")
    bin_index = np.clip(bin_index, 0, _N_CLOCK_BINS - 1)
    weights = np.where(valid_mass, mass, 0.0)
    binned = np.bincount(bin_index, weights=weights, minlength=_N_CLOCK_BINS).astype("float64")
    total = float(binned.sum())
    if not np.isfinite(total) or total <= _EPS:
        return None
    profile = binned / total
    if not np.isfinite(profile).all() or (profile < 0.0).any():
        return None
    return profile


def _profile_summary(profile: np.ndarray) -> tuple[float, float, float] | None:
    if profile.shape != (_N_CLOCK_BINS,) or not np.isfinite(profile).all() or (profile < 0.0).any():
        return None
    total = float(profile.sum())
    if total <= _EPS:
        return None
    normalized = profile / total
    centers = (np.arange(_N_CLOCK_BINS, dtype="float64") + 0.5) / float(_N_CLOCK_BINS)
    concentration = float(np.dot(normalized, normalized))
    late_tilt = float(np.dot(normalized, centers - 0.5))
    edge_core = float(np.dot(normalized, np.cos(2.0 * np.pi * centers)))
    if not (np.isfinite(concentration) and np.isfinite(late_tilt) and np.isfinite(edge_core)):
        return None
    return concentration, late_tilt, edge_core


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = (left + right) / 2.0
    valid_left = left > _EPS
    valid_right = right > _EPS
    left_term = float(np.sum(left[valid_left] * np.log(left[valid_left] / midpoint[valid_left])))
    right_term = float(np.sum(right[valid_right] * np.log(right[valid_right] / midpoint[valid_right])))
    value = 0.5 * (left_term + right_term)
    return value if np.isfinite(value) and value >= -_EPS else float("nan")


def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    arrays = _validated_arrays(group)
    if arrays is None:
        return out
    seconds, sessions, last, total_depth = arrays
    same_session = sessions[1:] == sessions[:-1]
    # Count increments are checked above and then used only to certify that a
    # realized last-price movement occurred alongside new trades.
    count = _numeric(group, "num_trades")
    trade_increment = np.diff(count)
    price_mass = np.zeros(len(last) - 1, dtype="float64")
    price_mass[same_session & (trade_increment > 0.0)] = np.abs(
        np.diff(np.log(last))[same_session & (trade_increment > 0.0)]
    )
    depth_mass = np.zeros(len(total_depth) - 1, dtype="float64")
    depth_mass[same_session] = np.abs(np.diff(np.log(total_depth))[same_session])

    price_profile = _profile_from_interval_mass(
        seconds,
        sessions,
        price_mass,
        minimum_active=_MIN_PRICE_EVENTS,
    )
    depth_profile = _profile_from_interval_mass(
        seconds,
        sessions,
        depth_mass,
        minimum_active=_MIN_DEPTH_ADJUSTMENTS,
    )
    if price_profile is not None:
        price_summary = _profile_summary(price_profile)
        if price_summary is not None:
            out.update(dict(zip(_PRICE_ALLOCATION_SIGNALS, price_summary, strict=True)))
    if depth_profile is not None:
        depth_summary = _profile_summary(depth_profile)
        if depth_summary is not None:
            out.update(dict(zip(_DEPTH_ALLOCATION_SIGNALS, depth_summary, strict=True)))
    if price_profile is not None and depth_profile is not None:
        price_norm = float(np.linalg.norm(price_profile))
        depth_norm = float(np.linalg.norm(depth_profile))
        if price_norm > _EPS and depth_norm > _EPS:
            centers = (np.arange(_N_CLOCK_BINS, dtype="float64") + 0.5) / float(_N_CLOCK_BINS)
            alignment = float(np.dot(price_profile, depth_profile) / (price_norm * depth_norm))
            offset = float(np.dot(price_profile - depth_profile, centers))
            divergence = _jensen_shannon(price_profile, depth_profile)
            if np.isfinite(alignment) and np.isfinite(offset) and np.isfinite(divergence):
                out.update(dict(zip(_COUPLING_SIGNALS, (alignment, offset, divergence), strict=True)))
    return {signal: (float(value) if np.isfinite(value) else float("nan")) for signal, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        strict = _strict_physical_frame(ctx.panel, score_date=score_date)
        if strict.empty:
            built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
        else:
            rows: list[dict[str, object]] = []
            for (dt, code), group in strict.groupby(["dt", "code"], sort=False):
                row: dict[str, object] = {"dt": dt, "code": code}
                row.update(_group_metrics(group))
                rows.append(row)
            built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
            built = built.replace([np.inf, -np.inf], np.nan).reindex(output_index)

    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayTemporalShapeV1(Factor):
    """Research-only T1430 temporal allocation and profile-coupling catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = False
    requires_bond_stock_map = False

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
