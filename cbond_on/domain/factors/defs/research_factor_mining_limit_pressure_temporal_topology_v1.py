"""Research-only strict-PIT topology of non-lock limit-pressure approaches.

This import-only module is not in ``defs.__init__`` or any factor/model/live
configuration.  Its sole market input is ``FactorComputeContext.panel`` and it
uses only physical score-day snapshots through 14:29.  It reads no label,
return target, PnL, score, pool, file, database, Redis, daily data, or stock
panel.

The established ``bond_limit_pressure`` family already supplies terminal limit
distances, an endpoint pressure change, near-touch occupancy, and a directional
path.  The established limit-queue family already supplies *actual* executable
limit-lock terminal state, occupancy, and transition rate.  This module does
not repeat those fields.  It instead excludes actual locks and studies the
topology of a non-lock price approach to a limit: entry timing and episodes,
the immediate exit/re-approach sequence, and L1 liquidity reconfiguration at
the entry into an approach.

Duplicate/reversed timestamps, in-session gaps longer than twenty minutes,
missing/invalid limits or L1 observations, crossed quotes, or prices materially
outside their published limit range fail closed to NaN for the entire bond-day.
``ctx.cache`` only memoizes quantities derived from ``ctx.panel``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_limit_pressure_temporal_topology_v1"
CATALOG_VERSION = "20260803_limit_pressure_temporal_topology_v1"
_EPS = 1e-12
_LIMIT_REL_TOL = 0.005
_QUOTE_REL_TOL = 1e-8
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MAX_IN_SESSION_GAP_SECONDS = 20.0 * 60.0
_MIN_ROWS = 12
_REAPPROACH_HORIZON = 3


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable non-lock limit-approach topology candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_ENTRY_TOPOLOGY_SIGNALS = (
    "lpt_nonlock_approach_first_entry_clock",
    "lpt_nonlock_approach_longest_dwell_share",
    "lpt_nonlock_approach_entry_rate",
)
_EXIT_SEQUENCE_SIGNALS = (
    "lpt_nonlock_exit_price_reversal",
    "lpt_nonlock_exit_limit_release",
    "lpt_nonlock_exit_reapproach_share",
)
_ENTRY_LIQUIDITY_SIGNALS = (
    "lpt_nonlock_entry_same_side_depth_log_change",
    "lpt_nonlock_entry_opposite_side_depth_log_change",
    "lpt_nonlock_entry_oriented_imbalance",
)

_CATALOG = (
    _entries(
        "limit_nonlock_approach_entry_topology",
        _ENTRY_TOPOLOGY_SIGNALS,
        "The entry time, longest contiguous dwell, and recurrence rate of a near-limit price regime are distinct from endpoint pressure or near-touch occupancy.",
    )
    + _entries(
        "limit_nonlock_approach_exit_sequence",
        _EXIT_SEQUENCE_SIGNALS,
        "After a non-lock approach exits, its directed price reversal, distance release, and short-horizon return-to-approach sequence describe recovery topology rather than an end-of-day path.",
    )
    + _entries(
        "limit_nonlock_approach_entry_liquidity",
        _ENTRY_LIQUIDITY_SIGNALS,
        "L1 same-side and opposite-side depth adjustments at non-lock approach entry describe conditional liquidity behavior without classifying a passive limit queue.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# s_t in {-1,0,+1} is a *non-lock* approach state: price lies within 0.5% of
# the lower/upper limit but is not an executable L1 limit lock.  E indexes
# same-side contiguous episodes of s_t.  u_t is compressed continuous-session
# clock position, d_s is the distance to the approached side's limit, and
# I_t=(bid_depth-ask_depth)/(bid_depth+ask_depth).
FORMULAS: dict[str, str] = {
    "lpt_nonlock_approach_first_entry_clock": "s_entry1*u_entry1",
    "lpt_nonlock_approach_longest_dwell_share": "max_E(|E|)/T",
    "lpt_nonlock_approach_entry_rate": "count(E)/(T-1)",
    "lpt_nonlock_exit_price_reversal": "mean_E[-s_E*log(last_exit/last_end)]",
    "lpt_nonlock_exit_limit_release": "mean_E[d_s,exit-d_s,end]",
    "lpt_nonlock_exit_reapproach_share": "mean_E(1[s reappears within next 3 snapshots])",
    "lpt_nonlock_entry_same_side_depth_log_change": "mean_entry(log(depth_same,t/depth_same,t-1))",
    "lpt_nonlock_entry_opposite_side_depth_log_change": "mean_entry(log(depth_opposite,t/depth_opposite,t-1))",
    "lpt_nonlock_entry_oriented_imbalance": "mean_entry(s_entry*I_entry)",
}

_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "last",
    "high_limited",
    "low_limited",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
)


def factor_mining_limit_pressure_temporal_topology_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research catalogue loaders."""

    return factor_mining_limit_pressure_temporal_topology_catalog()


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
    """Keep physical score-day continuous-session observations only through 14:29."""

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


def _continuous_clock(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Return normalized compressed session time and a morning/afternoon id."""

    times = pd.to_datetime(group["trade_time"], errors="coerce")
    if len(times) < _MIN_ROWS or times.isna().any() or times.duplicated().any():
        return None
    if (times.diff().dropna() <= pd.Timedelta(0)).any():
        return None
    sessions = np.empty(len(times), dtype="int8")
    seconds = np.empty(len(times), dtype="float64")
    morning_seconds = 2.0 * 60.0 * 60.0
    total_seconds = morning_seconds + 89.0 * 60.0
    for index, stamp in enumerate(times):
        clock = stamp.time()
        day = stamp.normalize()
        if _MORNING_START <= clock <= _MORNING_END:
            seconds[index] = float((stamp - (day + pd.Timedelta(hours=9, minutes=30))).total_seconds())
            sessions[index] = 0
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            seconds[index] = morning_seconds + float(
                (stamp - (day + pd.Timedelta(hours=13))).total_seconds()
            )
            sessions[index] = 1
        else:
            return None
    if not np.isfinite(seconds).all() or (np.diff(seconds) < 0.0).any():
        return None
    for session in (0, 1):
        local = times.iloc[sessions == session]
        if local.empty:
            continue
        gaps = local.diff().dropna().dt.total_seconds().to_numpy(dtype="float64")
        if not np.isfinite(gaps).all() or (gaps <= 0.0).any() or (gaps > _MAX_IN_SESSION_GAP_SECONDS).any():
            return None
    return seconds / total_seconds, sessions


def _validated_arrays(group: pd.DataFrame) -> tuple[np.ndarray, ...] | None:
    """Validate all required panel fields before any state classification."""

    clock = _continuous_clock(group)
    if clock is None:
        return None
    positions, _ = clock
    last = _numeric(group, "last")
    high = _numeric(group, "high_limited")
    low = _numeric(group, "low_limited")
    ask = _numeric(group, "ask_price1")
    bid = _numeric(group, "bid_price1")
    ask_depth = _numeric(group, "ask_volume1")
    bid_depth = _numeric(group, "bid_volume1")
    total_depth = ask_depth + bid_depth
    valid = (
        np.isfinite(last).all()
        and np.isfinite(high).all()
        and np.isfinite(low).all()
        and np.isfinite(ask).all()
        and np.isfinite(bid).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and (last > 0.0).all()
        and (high > 0.0).all()
        and (low > 0.0).all()
        and (high > low).all()
        and (ask > 0.0).all()
        and (bid > 0.0).all()
        and (ask >= bid).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (total_depth > _EPS).all()
        and (last >= low * (1.0 - _LIMIT_REL_TOL)).all()
        and (last <= high * (1.0 + _LIMIT_REL_TOL)).all()
    )
    if not valid:
        return None
    return positions, last, high, low, ask, bid, ask_depth, bid_depth


def _nonlock_approach_states(
    last: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ask: np.ndarray,
    bid: np.ndarray,
    ask_depth: np.ndarray,
    bid_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Classify price-near-limit states while explicitly excluding L1 locks."""

    up_distance = (high - last) / last
    down_distance = (last - low) / last
    if not (np.isfinite(up_distance).all() and np.isfinite(down_distance).all()):
        return None
    upper_near = np.abs(up_distance) <= _LIMIT_REL_TOL
    lower_near = np.abs(down_distance) <= _LIMIT_REL_TOL
    upper_lock = (
        (np.abs(last - high) <= high * _QUOTE_REL_TOL)
        & (np.abs(bid - high) <= high * _QUOTE_REL_TOL)
        & (bid_depth > _EPS)
    )
    lower_lock = (
        (np.abs(last - low) <= low * _QUOTE_REL_TOL)
        & (np.abs(ask - low) <= low * _QUOTE_REL_TOL)
        & (ask_depth > _EPS)
    )
    if (upper_near & lower_near).any() or (upper_lock & lower_lock).any():
        return None
    states = np.where(
        upper_near & ~upper_lock,
        1.0,
        np.where(lower_near & ~lower_lock, -1.0, 0.0),
    )
    return states, up_distance, down_distance


def _episodes(states: np.ndarray) -> list[tuple[int, int, float]]:
    out: list[tuple[int, int, float]] = []
    start: int | None = None
    state = 0.0
    for index, value in enumerate(states):
        if value == 0.0:
            if start is not None:
                out.append((start, index - 1, state))
                start = None
                state = 0.0
            continue
        if start is None:
            start = index
            state = float(value)
        elif value != state:
            out.append((start, index - 1, state))
            start = index
            state = float(value)
    if start is not None:
        out.append((start, len(states) - 1, state))
    return out


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator) and numerator > _EPS and denominator > _EPS):
        return float("nan")
    value = float(np.log(numerator / denominator))
    return value if np.isfinite(value) else float("nan")


def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    arrays = _validated_arrays(group)
    if arrays is None:
        return out
    positions, last, high, low, ask, bid, ask_depth, bid_depth = arrays
    classified = _nonlock_approach_states(last, high, low, ask, bid, ask_depth, bid_depth)
    if classified is None:
        return out
    states, up_distance, down_distance = classified
    episodes = _episodes(states)
    if not episodes:
        return out

    first_start, _, first_side = episodes[0]
    longest = max(end - start + 1 for start, end, _ in episodes)
    denominator = max(1, len(states) - 1)
    out.update(
        {
            "lpt_nonlock_approach_first_entry_clock": float(first_side * positions[first_start]),
            "lpt_nonlock_approach_longest_dwell_share": float(longest / len(states)),
            "lpt_nonlock_approach_entry_rate": float(len(episodes) / denominator),
        }
    )

    exit_reversals: list[float] = []
    exit_releases: list[float] = []
    reapproach: list[float] = []
    for _, end, side in episodes:
        exit_index = end + 1
        if exit_index >= len(states) or states[exit_index] != 0.0:
            continue
        price_move = _safe_log_ratio(float(last[exit_index]), float(last[end]))
        if np.isfinite(price_move):
            exit_reversals.append(float(-side * price_move))
        distance = up_distance if side > 0.0 else down_distance
        release = float(distance[exit_index] - distance[end])
        if np.isfinite(release):
            exit_releases.append(release)
        horizon_end = min(len(states), exit_index + 1 + _REAPPROACH_HORIZON)
        if horizon_end > exit_index + 1:
            reapproach.append(float(np.any(states[exit_index + 1 : horizon_end] == side)))
    if exit_reversals:
        out["lpt_nonlock_exit_price_reversal"] = float(np.mean(exit_reversals))
    if exit_releases:
        out["lpt_nonlock_exit_limit_release"] = float(np.mean(exit_releases))
    if reapproach:
        out["lpt_nonlock_exit_reapproach_share"] = float(np.mean(reapproach))

    same_side_changes: list[float] = []
    opposite_side_changes: list[float] = []
    oriented_imbalances: list[float] = []
    for start, _, side in episodes:
        if start <= 0 or states[start - 1] != 0.0:
            continue
        same_now, same_prior = (bid_depth[start], bid_depth[start - 1]) if side > 0.0 else (
            ask_depth[start],
            ask_depth[start - 1],
        )
        opposite_now, opposite_prior = (ask_depth[start], ask_depth[start - 1]) if side > 0.0 else (
            bid_depth[start],
            bid_depth[start - 1],
        )
        same_change = _safe_log_ratio(float(same_now), float(same_prior))
        opposite_change = _safe_log_ratio(float(opposite_now), float(opposite_prior))
        if np.isfinite(same_change):
            same_side_changes.append(same_change)
        if np.isfinite(opposite_change):
            opposite_side_changes.append(opposite_change)
        total_depth = float(ask_depth[start] + bid_depth[start])
        if total_depth > _EPS:
            imbalance = float((bid_depth[start] - ask_depth[start]) / total_depth)
            if np.isfinite(imbalance):
                oriented_imbalances.append(float(side * imbalance))
    if same_side_changes:
        out["lpt_nonlock_entry_same_side_depth_log_change"] = float(np.mean(same_side_changes))
    if opposite_side_changes:
        out["lpt_nonlock_entry_opposite_side_depth_log_change"] = float(np.mean(opposite_side_changes))
    if oriented_imbalances:
        out["lpt_nonlock_entry_oriented_imbalance"] = float(np.mean(oriented_imbalances))
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
class FactorMiningLimitPressureTemporalTopologyV1(Factor):
    """Research-only T1430 non-lock limit-approach topology catalogue."""

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
