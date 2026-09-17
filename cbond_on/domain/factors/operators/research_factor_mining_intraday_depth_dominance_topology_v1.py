"""Research-only strict-PIT topology of five-level depth dominance.

This catalogue uses only the physical score-day T1430 bond panel.  It studies
the ordinal identity of the deepest displayed level on each side of the book,
not depth magnitude, depth centroid, quote-price geometry, a price-ladder
reprice, or L1 queue events.  It is intentionally not imported by
``defs.__init__`` and has no live, I/O, configuration, FactorStore, database,
or scheduler effect.
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


KERNEL_NAME = "factor_mining_intraday_depth_dominance_topology_v1"
CATALOG_VERSION = "20260803_intraday_depth_dominance_topology_v1"
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MIN_ROWS = 12
_MIN_TRANSITIONS = 3
_DEPTH_LEVELS = 5


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable depth-dominance topology candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_STATE_OCCUPANCY_SIGNALS = (
    "ddso_bid_dominant_level_entropy",
    "ddso_ask_dominant_level_entropy",
    "ddso_cross_side_dominant_level_alignment",
)
_SWITCH_TOPOLOGY_SIGNALS = (
    "ddst_bid_dominant_level_switch_rate",
    "ddst_ask_dominant_level_switch_rate",
    "ddst_cross_side_switch_synchrony",
)
_TRANSITION_ORIENTATION_SIGNALS = (
    "ddto_bid_dominant_level_inward_bias",
    "ddto_ask_dominant_level_inward_bias",
    "ddto_cross_side_touch_orientation",
)

_CATALOG = (
    _entries(
        "depth_dominance_state_occupancy",
        _STATE_OCCUPANCY_SIGNALS,
        "The discrete identity of the deepest displayed level can be diverse or aligned across sides even when total depth, centroid, and continuous depth shape are unchanged.",
    )
    + _entries(
        "depth_dominance_switch_topology",
        _SWITCH_TOPOLOGY_SIGNALS,
        "Changes in the ordinal level carrying the most displayed depth describe a liquidity-state transition rather than a quote reprice or a magnitude-weighted depth update.",
    )
    + _entries(
        "depth_dominance_transition_orientation",
        _TRANSITION_ORIENTATION_SIGNALS,
        "When the deepest level relocates toward or away from the touch, bid and ask sides can provide an oriented relative-touch-liquidity signal without inferring trade direction.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# r_side,t is the unique argmax depth level in {0,...,4}, where 0 is L1.  A
# transition d_side,t=r_side,t-r_side,t-1 is only evaluated within a continuous
# session.  q_side,t=-sign(d_side,t), so +1 means the deepest level moved
# toward the touch and -1 means it moved away.
FORMULAS: dict[str, str] = {
    "ddso_bid_dominant_level_entropy": "-sum_{k=0..4} p_bid,k*log(p_bid,k)/log(5), p_bid,k=mean(1[r_bid,t=k]).",
    "ddso_ask_dominant_level_entropy": "-sum_{k=0..4} p_ask,k*log(p_ask,k)/log(5), p_ask,k=mean(1[r_ask,t=k]).",
    "ddso_cross_side_dominant_level_alignment": "mean(1[r_bid,t=r_ask,t]) over valid physical score-day snapshots.",
    "ddst_bid_dominant_level_switch_rate": "mean(1[d_bid,t!=0]) over adjacent within-session snapshots.",
    "ddst_ask_dominant_level_switch_rate": "mean(1[d_ask,t!=0]) over adjacent within-session snapshots.",
    "ddst_cross_side_switch_synchrony": "mean(1[d_bid,t!=0 and d_ask,t!=0]) over adjacent within-session snapshots.",
    "ddto_bid_dominant_level_inward_bias": "mean(q_bid,t | d_bid,t!=0), q_bid,t=-sign(d_bid,t).",
    "ddto_ask_dominant_level_inward_bias": "mean(q_ask,t | d_ask,t!=0), q_ask,t=-sign(d_ask,t).",
    "ddto_cross_side_touch_orientation": "mean((q_bid,t-q_ask,t)/2 | d_bid,t!=0 and d_ask,t!=0).",
}


def factor_mining_intraday_depth_dominance_topology_v1_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable three-family research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research factor-mining runners."""

    return factor_mining_intraday_depth_dominance_topology_v1_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve the score day without trusting a relabelled rolling panel alone."""

    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        day = pd.Timestamp(raw_day)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _session_label(clock: dt_time) -> int | None:
    if _MORNING_START <= clock <= _MORNING_END:
        return 0
    if _AFTERNOON_START <= clock <= _CUTOFF:
        return 1
    return None


def _score_day_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Keep only physical score-day continuous-session observations through 14:29."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:score_day_frame"
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
        indexed = pd.to_datetime(frame["dt"], errors="coerce")
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        clocks = timestamps.dt.time
        sessions = clocks.map(lambda clock: _session_label(clock) if pd.notna(clock) else None)
        if not frame.empty and not bool((indexed.dt.normalize() == target).any()):
            raise ValueError(f"{KERNEL_NAME} has no indexed rows for signal day {target.date().isoformat()}")
        keep = (
            indexed.notna()
            & (indexed.dt.normalize() == target)
            & timestamps.notna()
            & (timestamps.dt.normalize() == target)
            & sessions.notna()
        )
        out = frame.loc[keep].copy()
        out["trade_time"] = timestamps.loc[keep]
        out["__session"] = sessions.loc[keep].astype("int64")
        if not out.empty:
            out = out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")

    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
        ctx.cache[cache_key] = out
    return out


def _depth_columns() -> tuple[str, ...]:
    return tuple(
        f"{side}_volume{level}"
        for side in ("bid", "ask")
        for level in range(1, _DEPTH_LEVELS + 1)
    )


_REQUIRED_COLUMNS = ("trade_time", *_depth_columns())


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _nan_record() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _unique_dominant_level(depth: np.ndarray) -> np.ndarray | None:
    if depth.ndim != 2 or depth.shape[1] != _DEPTH_LEVELS:
        return None
    if not np.isfinite(depth).all() or (depth < 0.0).any():
        return None
    maxima = depth.max(axis=1)
    if (np.sum(depth == maxima[:, None], axis=1) != 1).any():
        return None
    return np.argmax(depth, axis=1).astype("int64")


def _state_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return unique deepest-level states, failing closed on a tied hierarchy."""

    if len(group) < _MIN_ROWS:
        return None
    trade_times = pd.to_datetime(group["trade_time"], errors="coerce")
    if trade_times.isna().any() or trade_times.duplicated().any() or not trade_times.is_monotonic_increasing:
        return None
    sessions = pd.to_numeric(group["__session"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(sessions).all() or not np.isin(sessions, (0.0, 1.0)).all():
        return None
    bid_depth = np.column_stack(
        [pd.to_numeric(group[f"bid_volume{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    ask_depth = np.column_stack(
        [pd.to_numeric(group[f"ask_volume{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    bid_level = _unique_dominant_level(bid_depth)
    ask_level = _unique_dominant_level(ask_depth)
    if bid_level is None or ask_level is None:
        return None
    return bid_level, ask_level, sessions.astype("int64")


def _state_entropy(levels: np.ndarray) -> float:
    counts = np.bincount(levels, minlength=_DEPTH_LEVELS).astype("float64")
    probabilities = counts[counts > 0.0] / float(counts.sum())
    value = -float(np.sum(probabilities * np.log(probabilities))) / float(np.log(_DEPTH_LEVELS))
    return value if np.isfinite(value) else float("nan")


def _metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    arrays = _state_arrays(group)
    if arrays is None:
        return out
    bid_level, ask_level, sessions = arrays

    out["ddso_bid_dominant_level_entropy"] = _state_entropy(bid_level)
    out["ddso_ask_dominant_level_entropy"] = _state_entropy(ask_level)
    out["ddso_cross_side_dominant_level_alignment"] = float(np.mean(bid_level == ask_level))

    same_session = sessions[1:] == sessions[:-1]
    if int(same_session.sum()) < _MIN_TRANSITIONS:
        return out
    bid_delta = np.diff(bid_level)
    ask_delta = np.diff(ask_level)
    bid_switch = same_session & (bid_delta != 0)
    ask_switch = same_session & (ask_delta != 0)
    both_switch = bid_switch & ask_switch

    out["ddst_bid_dominant_level_switch_rate"] = float(np.mean(bid_switch[same_session]))
    out["ddst_ask_dominant_level_switch_rate"] = float(np.mean(ask_switch[same_session]))
    out["ddst_cross_side_switch_synchrony"] = float(np.mean(both_switch[same_session]))

    if int(bid_switch.sum()) >= _MIN_TRANSITIONS:
        out["ddto_bid_dominant_level_inward_bias"] = float(np.mean(-np.sign(bid_delta[bid_switch])))
    if int(ask_switch.sum()) >= _MIN_TRANSITIONS:
        out["ddto_ask_dominant_level_inward_bias"] = float(np.mean(-np.sign(ask_delta[ask_switch])))
    if int(both_switch.sum()) >= _MIN_TRANSITIONS:
        bid_orientation = -np.sign(bid_delta[both_switch])
        ask_orientation = -np.sign(ask_delta[both_switch])
        out["ddto_cross_side_touch_orientation"] = float(np.mean((bid_orientation - ask_orientation) / 2.0))
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Compute all topology signals once and cache the immutable frame in context."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:feature_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    _require_columns(frame)
    if frame.empty:
        built = pd.DataFrame(index=_empty_index(), columns=_ALL_SIGNALS, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            row.update(_metrics(group))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
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
class FactorMiningIntradayDepthDominanceTopologyV1(Factor):
    """Research-only T1430 five-level depth-dominance topology kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _feature_frame(ctx)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
