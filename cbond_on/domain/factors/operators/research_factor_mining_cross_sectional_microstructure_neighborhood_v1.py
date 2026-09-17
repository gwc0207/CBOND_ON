"""Research-only strict-PIT cross-sectional microstructure neighbourhood factors.

This import-only module is deliberately outside :mod:`defs.__init__` and all
live, configuration, database, mask, and scratch-output paths.  It consumes
only ``FactorComputeContext.panel``.  Each signal is a non-parametric local
cross-sectional contrast: a bond's rank for a visible microstructure state
minus the median rank of its nearest same-day peers in two other visible
microstructure coordinates.  It is neither a return factor nor a linear
cross-sectional residual.

``clean_direct`` can carry an old physical snapshot beneath a requested score
date.  Rows must therefore agree on index date and physical ``trade_time``,
fall inside a continuous session, and be no later than physical ``14:29:00``.
Missing fields, duplicate clocks, crossed books, invalid depth, counter
resets, or an insufficient finite cross-section fail closed to ``NaN``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_cross_sectional_microstructure_neighborhood_v1"
CATALOG_VERSION = "20260803_cross_sectional_microstructure_neighborhood_v1"
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EPS = 1e-12
_QUOTE_REL_TOL = 1e-8
_MIN_PATH_ROWS = 12
_MIN_CROSS_SECTION = 30
_NEIGHBOR_COUNT = 12


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only neighbourhood candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class NeighborhoodDefinition:
    """One target contrasted against same-day local microstructure peers."""

    signal: str
    target: str
    anchors: tuple[str, str]


def _definition(signal: str, target: str, anchor_a: str, anchor_b: str) -> NeighborhoodDefinition:
    return NeighborhoodDefinition(signal=signal, target=target, anchors=(anchor_a, anchor_b))


_FAMILY_DEFINITIONS: dict[str, tuple[NeighborhoodDefinition, ...]] = {
    "csn_limit_queue_local_dislocation": (
        _definition(
            "csn_lql_terminal_lock_neighbor_gap",
            "limit_terminal_lock",
            "quote_update_rate",
            "trade_event_rate",
        ),
        _definition(
            "csn_lql_occupancy_neighbor_gap",
            "limit_lock_occupancy",
            "quote_update_rate",
            "trade_event_rate",
        ),
        _definition(
            "csn_lql_transition_neighbor_gap",
            "limit_transition_rate",
            "quote_update_rate",
            "trade_event_rate",
        ),
    ),
    "csn_passive_queue_local_dislocation": (
        _definition(
            "csn_pql_retention_neighbor_gap",
            "passive_depth_retention",
            "trade_event_rate",
            "top_depth_update_intensity",
        ),
        _definition(
            "csn_pql_refill_neighbor_gap",
            "passive_refill_imbalance",
            "trade_event_rate",
            "top_depth_update_intensity",
        ),
        _definition(
            "csn_pql_churn_neighbor_gap",
            "passive_churn_per_trade",
            "trade_event_rate",
            "top_depth_update_intensity",
        ),
    ),
    "csn_quote_initiation_local_dislocation": (
        _definition(
            "csn_taq_buy_initiation_neighbor_gap",
            "quote_buy_initiation_share",
            "quote_update_rate",
            "queue_imbalance_change_rate",
        ),
        _definition(
            "csn_taq_sell_initiation_neighbor_gap",
            "quote_sell_initiation_share",
            "quote_update_rate",
            "queue_imbalance_change_rate",
        ),
        _definition(
            "csn_taq_imbalance_neighbor_gap",
            "quote_initiation_imbalance",
            "quote_update_rate",
            "queue_imbalance_change_rate",
        ),
    ),
    "csn_depth_cascade_local_dislocation": (
        _definition(
            "csn_dlc_bid_coherence_neighbor_gap",
            "depth_bid_adjacent_coherence",
            "layer_update_intensity",
            "queue_imbalance_change_rate",
        ),
        _definition(
            "csn_dlc_ask_coherence_neighbor_gap",
            "depth_ask_adjacent_coherence",
            "layer_update_intensity",
            "queue_imbalance_change_rate",
        ),
        _definition(
            "csn_dlc_crosslag_neighbor_gap",
            "depth_cross_side_lag_asymmetry",
            "layer_update_intensity",
            "queue_imbalance_change_rate",
        ),
    ),
}

_FAMILY_HYPOTHESES = {
    "csn_limit_queue_local_dislocation": "Limit-queue lock state can be unusual relative to peers with the same visible quote and trade clocks, without using a limit-distance or return transform.",
    "csn_passive_queue_local_dislocation": "Trade-conditioned passive queue retention, refill, and churn can be locally unusual relative to peers with similar event and touch-depth update activity.",
    "csn_quote_initiation_local_dislocation": "Prior-quote buy/sell initiation state can differ from peers with similar quote-clock and queue-imbalance-change states, without an aggressor label.",
    "csn_depth_cascade_local_dislocation": "Five-level update cascade can be locally unusual relative to peers with similar layer-update intensity and queue-imbalance motion rather than static depth or spread.",
}


def _catalog_entries() -> tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for family, definitions in _FAMILY_DEFINITIONS.items():
        entries.extend(
            CatalogEntry(
                family=family,
                signal=definition.signal,
                kernel=KERNEL_NAME,
                hypothesis=_FAMILY_HYPOTHESES[family],
            )
            for definition in definitions
        )
    return tuple(entries)


_CATALOG = _catalog_entries()
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    family: tuple(definition.signal for definition in definitions)
    for family, definitions in _FAMILY_DEFINITIONS.items()
}

_TOP_BOOK_COLUMNS = ("ask_price1", "bid_price1", "ask_volume1", "bid_volume1")


def _five_level_book_columns() -> tuple[str, ...]:
    return tuple(
        column
        for level in range(1, 6)
        for column in (
            f"ask_price{level}",
            f"bid_price{level}",
            f"ask_volume{level}",
            f"bid_volume{level}",
        )
    )


_FAMILY_REQUIRED_COLUMNS = {
    "csn_limit_queue_local_dislocation": (
        "last",
        "high_limited",
        "low_limited",
        "num_trades",
        *_TOP_BOOK_COLUMNS,
    ),
    "csn_passive_queue_local_dislocation": ("num_trades", *_TOP_BOOK_COLUMNS),
    "csn_quote_initiation_local_dislocation": ("last", "num_trades", *_TOP_BOOK_COLUMNS),
    "csn_depth_cascade_local_dislocation": _five_level_book_columns(),
}
_FAMILY_SUMMARY_COLUMNS = {
    family: tuple(
        dict.fromkeys(
            (
                *(definition.target for definition in definitions),
                *(anchor for definition in definitions for anchor in definition.anchors),
            )
        )
    )
    for family, definitions in _FAMILY_DEFINITIONS.items()
}


FORMULAS = {
    "csn_lql_terminal_lock_neighbor_gap": "rank(s_T) - median(rank(s_j): j in N_i(quote_update_rate, trade_event_rate)), s_T in {-1,0,+1}.",
    "csn_lql_occupancy_neighbor_gap": "rank(mean(1[s_t != 0])) minus the local-neighbour median rank.",
    "csn_lql_transition_neighbor_gap": "rank(mean(1[s_t != s_{t-1}])) minus the local-neighbour median rank.",
    "csn_pql_retention_neighbor_gap": "rank(mean(min(Dprev,Dnow)/Dprev | delta trades>0)) minus local-neighbour median rank.",
    "csn_pql_refill_neighbor_gap": "rank(mean((max(delta bid,0)-max(delta ask,0))/Dprev | delta trades>0)) minus local-neighbour median rank.",
    "csn_pql_churn_neighbor_gap": "rank(sum(|delta bid|+|delta ask|)/sum(delta trades)) minus local-neighbour median rank.",
    "csn_taq_buy_initiation_neighbor_gap": "rank(mean(1[last_t>=ask_{t-1}] | delta trades>0)) minus local-neighbour median rank.",
    "csn_taq_sell_initiation_neighbor_gap": "rank(mean(1[last_t<=bid_{t-1}] | delta trades>0)) minus local-neighbour median rank.",
    "csn_taq_imbalance_neighbor_gap": "rank((Nbuy-Nsell)/(Nbuy+Nsell)) minus local-neighbour median rank.",
    "csn_dlc_bid_coherence_neighbor_gap": "rank(mean corr(u_bid,k,u_bid,k+1)) minus local-neighbour median rank, u=2*delta depth/(new+old).",
    "csn_dlc_ask_coherence_neighbor_gap": "rank(mean corr(u_ask,k,u_ask,k+1)) minus local-neighbour median rank.",
    "csn_dlc_crosslag_neighbor_gap": "rank(corr(sum u_bid,t,sum u_ask,t+1)-corr(sum u_ask,t,sum u_bid,t+1)) minus local-neighbour median rank.",
}


def factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable four-family research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return factor_mining_cross_sectional_microstructure_neighborhood_v1_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
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
    """Keep only physical score-day, continuous-session rows through 14:29."""

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


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    """Keep labelled score-day codes even when one path must fail closed."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:output_index"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.MultiIndex):
            return cached

    panel = ensure_panel_index(ctx.panel)
    target = _signal_date_from_panel(panel)
    if target is None:
        out = _empty_index()
    else:
        frame = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
        labels = pd.to_datetime(frame["dt"], errors="coerce")
        frame = frame.loc[labels.notna() & (labels.dt.normalize() == target)].copy()
        frame = frame.sort_values(["dt", "code"], kind="mergesort").drop_duplicates()
        out = pd.MultiIndex.from_frame(frame, names=["dt", "code"]) if not frame.empty else _empty_index()

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.MultiIndex):
            return existing
        ctx.cache[cache_key] = out
    return out


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _nan_record(columns: Iterable[str]) -> dict[str, float]:
    return {column: float("nan") for column in columns}


def _numeric(g: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")


def _valid_strict_path(g: pd.DataFrame) -> bool:
    if len(g) < _MIN_PATH_ROWS:
        return False
    if g["seq"].duplicated(keep=False).any() or g["trade_time"].duplicated(keep=False).any():
        return False
    times = pd.to_datetime(g["trade_time"], errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing:
        return False
    raw = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    return bool((np.diff(raw) > 0).all())


def _positive_prices(g: pd.DataFrame, column: str) -> np.ndarray | None:
    values = _numeric(g, column)
    if values.size < _MIN_PATH_ROWS or not (np.isfinite(values).all() and (values > 0.0).all()):
        return None
    return values


def _top_book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask = _positive_prices(g, "ask_price1")
    bid = _positive_prices(g, "bid_price1")
    ask_depth = _numeric(g, "ask_volume1")
    bid_depth = _numeric(g, "bid_volume1")
    if (
        ask is None
        or bid is None
        or ask_depth.size != len(ask)
        or bid_depth.size != len(ask)
        or not (np.isfinite(ask_depth).all() and np.isfinite(bid_depth).all())
        or (ask_depth < 0.0).any()
        or (bid_depth < 0.0).any()
        or (ask < bid).any()
    ):
        return None
    return ask, bid, ask_depth, bid_depth


def _strict_trade_increments(g: pd.DataFrame) -> np.ndarray | None:
    values = _numeric(g, "num_trades")
    if values.size < _MIN_PATH_ROWS or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    return increments


def _quote_update_rate(ask: np.ndarray, bid: np.ndarray) -> float:
    relative_ask = np.abs(np.diff(ask)) / ask[:-1]
    relative_bid = np.abs(np.diff(bid)) / bid[:-1]
    if not (np.isfinite(relative_ask).all() and np.isfinite(relative_bid).all()):
        return float("nan")
    return float(np.mean(np.maximum(relative_ask, relative_bid) > _QUOTE_REL_TOL))


def _queue_imbalance_change_rate(ask_depth: np.ndarray, bid_depth: np.ndarray) -> float:
    total = ask_depth + bid_depth
    if not (np.isfinite(total).all() and (total > _EPS).all()):
        return float("nan")
    imbalance = (bid_depth - ask_depth) / total
    return float(np.mean(np.abs(np.diff(imbalance)))) if np.isfinite(imbalance).all() else float("nan")


def _top_depth_update_intensity(ask_depth: np.ndarray, bid_depth: np.ndarray) -> float:
    ask_sum = ask_depth[1:] + ask_depth[:-1]
    bid_sum = bid_depth[1:] + bid_depth[:-1]
    valid = (ask_sum > _EPS) & (bid_sum > _EPS)
    if int(valid.sum()) < 3:
        return float("nan")
    ask_update = 2.0 * (ask_depth[1:] - ask_depth[:-1]) / ask_sum
    bid_update = 2.0 * (bid_depth[1:] - bid_depth[:-1]) / bid_sum
    values = np.concatenate([np.abs(ask_update[valid]), np.abs(bid_update[valid])])
    return float(np.mean(values)) if np.isfinite(values).all() else float("nan")


def _limit_states(
    g: pd.DataFrame,
    book: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray | None:
    last = _positive_prices(g, "last")
    high = _positive_prices(g, "high_limited")
    low = _positive_prices(g, "low_limited")
    if last is None or high is None or low is None or not (high > low).all():
        return None
    ask, bid, ask_depth, bid_depth = book
    if (last < low * (1.0 - _QUOTE_REL_TOL)).any() or (last > high * (1.0 + _QUOTE_REL_TOL)).any():
        return None
    up_lock = (
        (np.abs(last - high) <= high * _QUOTE_REL_TOL)
        & (np.abs(bid - high) <= high * _QUOTE_REL_TOL)
        & (bid_depth > _EPS)
    )
    down_lock = (
        (np.abs(last - low) <= low * _QUOTE_REL_TOL)
        & (np.abs(ask - low) <= low * _QUOTE_REL_TOL)
        & (ask_depth > _EPS)
    )
    if (up_lock & down_lock).any():
        return None
    return np.where(up_lock, 1.0, np.where(down_lock, -1.0, 0.0))


def _limit_queue_summary(g: pd.DataFrame) -> dict[str, float]:
    columns = _FAMILY_SUMMARY_COLUMNS["csn_limit_queue_local_dislocation"]
    out = _nan_record(columns)
    if not _valid_strict_path(g):
        return out
    book = _top_book_arrays(g)
    trades = _strict_trade_increments(g)
    if book is None or trades is None:
        return out
    states = _limit_states(g, book)
    if states is None:
        return out
    ask, bid, _, _ = book
    out.update(
        {
            "limit_terminal_lock": float(states[-1]),
            "limit_lock_occupancy": float(np.mean(states != 0.0)),
            "limit_transition_rate": float(np.mean(states[1:] != states[:-1])),
            "quote_update_rate": _quote_update_rate(ask, bid),
            "trade_event_rate": float(np.mean(trades > 0.0)),
        }
    )
    return out


def _passive_queue_summary(g: pd.DataFrame) -> dict[str, float]:
    columns = _FAMILY_SUMMARY_COLUMNS["csn_passive_queue_local_dislocation"]
    out = _nan_record(columns)
    if not _valid_strict_path(g):
        return out
    book = _top_book_arrays(g)
    trades = _strict_trade_increments(g)
    if book is None or trades is None:
        return out
    _, _, ask_depth, bid_depth = book
    prior_depth = ask_depth[:-1] + bid_depth[:-1]
    current_depth = ask_depth[1:] + bid_depth[1:]
    bid_delta = np.diff(bid_depth)
    ask_delta = np.diff(ask_depth)
    event = trades > 0.0
    valid_event = event & (prior_depth > _EPS) & np.isfinite(current_depth)
    if int(valid_event.sum()) < 3:
        return out
    retention = np.minimum(prior_depth[valid_event], current_depth[valid_event]) / prior_depth[valid_event]
    refill = (
        np.maximum(bid_delta[valid_event], 0.0) - np.maximum(ask_delta[valid_event], 0.0)
    ) / prior_depth[valid_event]
    churn = np.abs(bid_delta[valid_event]) + np.abs(ask_delta[valid_event])
    trade_total = float(trades[valid_event].sum())
    if not (np.isfinite(retention).all() and np.isfinite(refill).all()):
        return out
    out.update(
        {
            "passive_depth_retention": float(np.mean(retention)),
            "passive_refill_imbalance": float(np.mean(refill)),
            "passive_churn_per_trade": _safe_div(float(churn.sum()), trade_total),
            "trade_event_rate": float(np.mean(trades > 0.0)),
            "top_depth_update_intensity": _top_depth_update_intensity(ask_depth, bid_depth),
        }
    )
    return out


def _quote_initiation_summary(g: pd.DataFrame) -> dict[str, float]:
    columns = _FAMILY_SUMMARY_COLUMNS["csn_quote_initiation_local_dislocation"]
    out = _nan_record(columns)
    if not _valid_strict_path(g):
        return out
    last = _positive_prices(g, "last")
    book = _top_book_arrays(g)
    trades = _strict_trade_increments(g)
    if last is None or book is None or trades is None:
        return out
    ask, bid, ask_depth, bid_depth = book
    event = trades > 0.0
    if int(event.sum()) < 3:
        return out
    at_ask = last[1:] >= ask[:-1] * (1.0 - _QUOTE_REL_TOL)
    at_bid = last[1:] <= bid[:-1] * (1.0 + _QUOTE_REL_TOL)
    ambiguous = at_ask & at_bid
    buy = event & at_ask & ~ambiguous
    sell = event & at_bid & ~ambiguous
    event_count = int(event.sum())
    buy_count = int(buy.sum())
    sell_count = int(sell.sum())
    out.update(
        {
            "quote_buy_initiation_share": float(buy_count / event_count),
            "quote_sell_initiation_share": float(sell_count / event_count),
            "quote_initiation_imbalance": _safe_div(float(buy_count - sell_count), float(buy_count + sell_count)),
            "quote_update_rate": _quote_update_rate(ask, bid),
            "queue_imbalance_change_rate": _queue_imbalance_change_rate(ask_depth, bid_depth),
        }
    )
    return out


def _five_level_book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask_price = np.column_stack([_numeric(g, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(g, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(g, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(g, f"bid_volume{level}") for level in range(1, 6)])
    valid = (
        len(ask_depth) >= _MIN_PATH_ROWS
        and np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and (ask_price > 0.0).all()
        and (bid_price > 0.0).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (ask_price[:, 0] >= bid_price[:, 0]).all()
        and (np.diff(ask_price, axis=1) >= 0.0).all()
        and (np.diff(bid_price, axis=1) <= 0.0).all()
    )
    return (ask_price, bid_price, ask_depth, bid_depth) if valid else None


def _normalized_depth_updates(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    denominator = depth[1:] + depth[:-1]
    valid = (denominator > _EPS).all(axis=1)
    if int(valid.sum()) < 3:
        return None
    updates = np.full_like(denominator, np.nan, dtype="float64")
    np.divide(2.0 * (depth[1:] - depth[:-1]), denominator, out=updates, where=denominator > _EPS)
    return (updates, valid) if np.isfinite(updates[valid]).all() else None


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
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


def _adjacent_update_coherence(updates: np.ndarray, valid: np.ndarray) -> float:
    values = [_corr(updates[valid, level], updates[valid, level + 1]) for level in range(4)]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if len(finite) >= 2 else float("nan")


def _lagged_correlation(
    left: np.ndarray,
    left_valid: np.ndarray,
    right: np.ndarray,
    right_valid: np.ndarray,
) -> float:
    mask = left_valid[:-1] & right_valid[1:]
    return _corr(left[:-1][mask], right[1:][mask])


def _depth_cascade_summary(g: pd.DataFrame) -> dict[str, float]:
    columns = _FAMILY_SUMMARY_COLUMNS["csn_depth_cascade_local_dislocation"]
    out = _nan_record(columns)
    if not _valid_strict_path(g):
        return out
    book = _five_level_book_arrays(g)
    if book is None:
        return out
    _, _, ask_depth, bid_depth = book
    ask = _normalized_depth_updates(ask_depth)
    bid = _normalized_depth_updates(bid_depth)
    if ask is None or bid is None:
        return out
    ask_updates, ask_valid = ask
    bid_updates, bid_valid = bid
    ask_total = ask_updates.sum(axis=1)
    bid_total = bid_updates.sum(axis=1)
    forward = _lagged_correlation(bid_total, bid_valid, ask_total, ask_valid)
    reverse = _lagged_correlation(ask_total, ask_valid, bid_total, bid_valid)
    intensity_values = np.concatenate(
        [np.abs(ask_updates[ask_valid]).ravel(), np.abs(bid_updates[bid_valid]).ravel()]
    )
    if not np.isfinite(intensity_values).all():
        return out
    out.update(
        {
            "depth_bid_adjacent_coherence": _adjacent_update_coherence(bid_updates, bid_valid),
            "depth_ask_adjacent_coherence": _adjacent_update_coherence(ask_updates, ask_valid),
            "depth_cross_side_lag_asymmetry": float(forward - reverse)
            if np.isfinite(forward) and np.isfinite(reverse)
            else float("nan"),
            "layer_update_intensity": float(np.mean(intensity_values)),
            "queue_imbalance_change_rate": _queue_imbalance_change_rate(ask_depth[:, 0], bid_depth[:, 0]),
        }
    )
    return out


_FAMILY_SUMMARIZERS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "csn_limit_queue_local_dislocation": _limit_queue_summary,
    "csn_passive_queue_local_dislocation": _passive_queue_summary,
    "csn_quote_initiation_local_dislocation": _quote_initiation_summary,
    "csn_depth_cascade_local_dislocation": _depth_cascade_summary,
}


def _family_summary_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache the family target and two visible neighbour coordinates."""

    if family not in _FAMILY_SUMMARIZERS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:summary:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score = _score_day_frame(ctx)
    _require_columns(score, ("trade_time", *_FAMILY_REQUIRED_COLUMNS[family]))
    output_index = _output_index(ctx)
    columns = _FAMILY_SUMMARY_COLUMNS[family]
    if score.empty:
        built = pd.DataFrame(index=output_index, columns=columns, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        summarize = _FAMILY_SUMMARIZERS[family]
        for (dt, code), group in score.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            row.update(summarize(group))
            rows.append(row)
        raw = pd.DataFrame(rows).set_index(["dt", "code"])[list(columns)].sort_index()
        built = raw.reindex(output_index).replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _rank_neighborhood_gap(
    summary: pd.DataFrame,
    *,
    target: str,
    anchors: tuple[str, str],
) -> pd.Series:
    """Return a local peer-median rank gap without fitting a regression model."""

    requested = (target, *anchors)
    missing = [column for column in requested if column not in summary.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} neighbourhood definition missing fields: {missing}")
    out = pd.Series(np.nan, index=summary.index, dtype="float64")
    source = summary.loc[:, list(requested)]
    finite = np.isfinite(source.to_numpy(dtype="float64")).all(axis=1)
    min_required = max(_MIN_CROSS_SECTION, _NEIGHBOR_COUNT + 1)
    if int(finite.sum()) < min_required:
        return out
    work = source.loc[finite].copy()
    if any(int(work[column].nunique(dropna=True)) < 2 for column in requested):
        return out
    ranked = work.rank(method="average", pct=True)
    coordinates = ranked.loc[:, list(anchors)].to_numpy(dtype="float64")
    distance = np.sqrt(np.square(coordinates[:, None, :] - coordinates[None, :, :]).sum(axis=2))
    if not np.isfinite(distance).all():
        return out
    np.fill_diagonal(distance, np.inf)
    neighbor_positions = np.argsort(distance, axis=1, kind="stable")[:, :_NEIGHBOR_COUNT]
    target_ranks = ranked[target].to_numpy(dtype="float64")
    local_median = np.median(target_ranks[neighbor_positions], axis=1)
    gaps = target_ranks - local_median
    if not np.isfinite(gaps).all():
        return out
    out.loc[work.index] = gaps
    return out


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache the three local-neighbour contrasts in one family."""

    definitions = _FAMILY_DEFINITIONS.get(family)
    if definitions is None:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    summary = _family_summary_frame(ctx, family)
    built = pd.DataFrame(index=summary.index, columns=_FAMILY_SIGNALS[family], dtype="float64")
    for definition in definitions:
        built[definition.signal] = _rank_neighborhood_gap(
            summary,
            target=definition.target,
            anchors=definition.anchors,
        )
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
class FactorMiningCrossSectionalMicrostructureNeighborhoodV1(Factor):
    """Research-only non-parametric cross-sectional microstructure kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
