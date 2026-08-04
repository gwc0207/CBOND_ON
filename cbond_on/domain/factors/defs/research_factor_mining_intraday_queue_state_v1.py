"""Research-only strict-PIT single-bond queue-state candidate catalogue.

This module is intentionally not imported by :mod:`defs.__init__`.  It only
consumes :class:`~cbond_on.domain.factors.base.FactorComputeContext.panel` and
does not open files, use a pool or label, access a database, or alter a live
factor contract.  It is a candidate catalogue for a later research screen.

The on-demand ``clean_direct`` panel can carry a historical snapshot under a
new index day.  Every calculation therefore requires both the indexed ``dt``
and physical ``trade_time`` to be the score date, retains only continuous
sessions, and stops at physical ``14:29:00``.  A missing field, crossed book,
counter reset, or undefined denominator is explicit ``NaN`` rather than an
imputed neutral state.
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


KERNEL_NAME = "factor_mining_intraday_queue_state_v1"
QUEUE_STATE_V1_VERSION = "20260803_intraday_queue_state_v1"
CATALOG_VERSION = QUEUE_STATE_V1_VERSION
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EPS = 1e-12
_LIMIT_REL_TOL = 1e-8
_MIN_ROWS = 12
_MIN_EVENT_INTERVALS = 3


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only queue-state candidate."""

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


_LIMIT_QUEUE_SIGNALS = (
    "lqls_signed_terminal_lock",
    "lqls_lock_occupancy_share",
    "lqls_state_transition_rate",
)
_PASSIVE_QUEUE_SIGNALS = (
    "pql_trade_depth_retention",
    "pql_trade_refill_imbalance",
    "pql_trade_queue_churn_per_trade",
)
_TRADE_AT_QUOTE_SIGNALS = (
    "taqi_buy_quote_initiation_share",
    "taqi_sell_quote_initiation_share",
    "taqi_signed_quote_initiation_imbalance",
)
_DEPTH_CASCADE_SIGNALS = (
    "dluc_bid_adjacent_update_coherence",
    "dluc_ask_adjacent_update_coherence",
    "dluc_cross_side_update_lag_asymmetry",
)


_CATALOG = (
    _entries(
        "limit_queue_lock_state_machine",
        _LIMIT_QUEUE_SIGNALS,
        "A limit-touch becomes a queue lock only when the executable touch and passive same-side L1 queue agree; its state path is distinct from limit distance.",
    )
    + _entries(
        "passive_queue_lifecycle_after_trade",
        _PASSIVE_QUEUE_SIGNALS,
        "Top-of-book retention, refill asymmetry, and churn conditional on an observed trade event describe passive queue lifecycle rather than unconditional depth change.",
    )
    + _entries(
        "trade_at_quote_initiation_state",
        _TRADE_AT_QUOTE_SIGNALS,
        "A cumulative-trade-certified last price at the prior best ask or bid identifies quote-side initiation without inferring unavailable aggressor flags.",
    )
    + _entries(
        "depth_layer_update_cascade",
        _DEPTH_CASCADE_SIGNALS,
        "Coherent five-level depth updates and their cross-side lead/lag measure queue-update cascade rather than static book geometry.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "limit_queue_lock_state_machine": _LIMIT_QUEUE_SIGNALS,
    "passive_queue_lifecycle_after_trade": _PASSIVE_QUEUE_SIGNALS,
    "trade_at_quote_initiation_state": _TRADE_AT_QUOTE_SIGNALS,
    "depth_layer_update_cascade": _DEPTH_CASCADE_SIGNALS,
}


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
    "limit_queue_lock_state_machine": (
        "last",
        "high_limited",
        "low_limited",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "bid_volume1",
    ),
    "passive_queue_lifecycle_after_trade": (
        "num_trades",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "bid_volume1",
    ),
    "trade_at_quote_initiation_state": (
        "last",
        "num_trades",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "bid_volume1",
    ),
    "depth_layer_update_cascade": _five_level_book_columns(),
}
_FULL_BATCH_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            "trade_time",
            *(
                column
                for columns in _FAMILY_REQUIRED_COLUMNS.values()
                for column in columns
            ),
        )
    )
)


FORMULAS = {
    "lqls_signed_terminal_lock": "s_T, where s_t=+1 for an upper-limit bid queue lock, -1 for a lower-limit ask queue lock, else 0.",
    "lqls_lock_occupancy_share": "mean(1[s_t != 0]) over validated physical score-day snapshots.",
    "lqls_state_transition_rate": "mean(1[s_t != s_{t-1}]) over consecutive validated snapshots.",
    "pql_trade_depth_retention": "mean(min(D_{t-1}, D_t) / D_{t-1} | delta num_trades_t > 0), D=bid_volume1+ask_volume1.",
    "pql_trade_refill_imbalance": "mean((max(delta bid,0)-max(delta ask,0))/D_{t-1} | delta num_trades_t > 0).",
    "pql_trade_queue_churn_per_trade": "sum(|delta bid|+|delta ask|) / sum(delta num_trades) over observed trade-event intervals.",
    "taqi_buy_quote_initiation_share": "mean(1[last_t >= ask_price1_{t-1}] | delta num_trades_t > 0), excluding an ambiguous locked quote.",
    "taqi_sell_quote_initiation_share": "mean(1[last_t <= bid_price1_{t-1}] | delta num_trades_t > 0), excluding an ambiguous locked quote.",
    "taqi_signed_quote_initiation_imbalance": "(N_buy-N_sell)/(N_buy+N_sell) on quote-initiating trade events.",
    "dluc_bid_adjacent_update_coherence": "mean corr(u_bid,k, u_bid,k+1), k=1..4; u=2*(depth_t-depth_{t-1})/(depth_t+depth_{t-1}).",
    "dluc_ask_adjacent_update_coherence": "mean corr(u_ask,k, u_ask,k+1), k=1..4 under the same validated-update contract.",
    "dluc_cross_side_update_lag_asymmetry": "corr(sum u_bid,t, sum u_ask,t+1) - corr(sum u_ask,t, sum u_bid,t+1).",
}


def factor_mining_intraday_queue_state_v1_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only queue-state catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return factor_mining_intraday_queue_state_v1_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve one score date without trusting rolling-panel labels alone."""

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
    """Return only physical score-day continuous-session rows through 14:29."""

    cache_key = f"{KERNEL_NAME}:{QUEUE_STATE_V1_VERSION}:score_day_frame"
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
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _numeric(g: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")


def _positive_prices(g: pd.DataFrame, column: str, *, min_count: int = _MIN_ROWS) -> np.ndarray | None:
    values = _numeric(g, column)
    if values.size < min_count or not (np.isfinite(values).all() and (values > 0.0).all()):
        return None
    return values


def _top_book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask = _positive_prices(g, "ask_price1")
    bid = _positive_prices(g, "bid_price1")
    ask_volume = _numeric(g, "ask_volume1")
    bid_volume = _numeric(g, "bid_volume1")
    if (
        ask is None
        or bid is None
        or ask_volume.size != len(ask)
        or bid_volume.size != len(ask)
        or not (np.isfinite(ask_volume).all() and np.isfinite(bid_volume).all())
        or (ask_volume < 0.0).any()
        or (bid_volume < 0.0).any()
        or (ask < bid).any()
    ):
        return None
    return ask, bid, ask_volume, bid_volume


def _strict_counter_increments(g: pd.DataFrame, column: str) -> np.ndarray | None:
    """Return only strict non-negative increments of a cumulative field."""

    values = _numeric(g, column)
    if values.size < _MIN_ROWS or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    return increments


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < _MIN_EVENT_INTERVALS:
        return float("nan")
    a = left[valid]
    b = right[valid]
    if float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _limit_queue_states(g: pd.DataFrame) -> np.ndarray | None:
    last = _positive_prices(g, "last")
    high = _positive_prices(g, "high_limited")
    low = _positive_prices(g, "low_limited")
    book = _top_book_arrays(g)
    if (
        last is None
        or high is None
        or low is None
        or book is None
        or not (high > low).all()
        or (last < low * (1.0 - _LIMIT_REL_TOL)).any()
        or (last > high * (1.0 + _LIMIT_REL_TOL)).any()
    ):
        return None
    ask, bid, ask_volume, bid_volume = book
    up_lock = (
        (np.abs(last - high) <= high * _LIMIT_REL_TOL)
        & (np.abs(bid - high) <= high * _LIMIT_REL_TOL)
        & (bid_volume > _EPS)
    )
    down_lock = (
        (np.abs(last - low) <= low * _LIMIT_REL_TOL)
        & (np.abs(ask - low) <= low * _LIMIT_REL_TOL)
        & (ask_volume > _EPS)
    )
    if (up_lock & down_lock).any():
        return None
    return np.where(up_lock, 1.0, np.where(down_lock, -1.0, 0.0))


def _limit_queue_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LIMIT_QUEUE_SIGNALS)
    states = _limit_queue_states(g)
    if states is None or len(states) < _MIN_ROWS:
        return out
    out["lqls_signed_terminal_lock"] = float(states[-1])
    out["lqls_lock_occupancy_share"] = float(np.mean(states != 0.0))
    out["lqls_state_transition_rate"] = float(np.mean(states[1:] != states[:-1]))
    return out


def _passive_queue_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_PASSIVE_QUEUE_SIGNALS)
    book = _top_book_arrays(g)
    increments = _strict_counter_increments(g, "num_trades")
    if book is None or increments is None:
        return out
    _, _, ask_volume, bid_volume = book
    prior_depth = ask_volume[:-1] + bid_volume[:-1]
    current_depth = ask_volume[1:] + bid_volume[1:]
    bid_delta = np.diff(bid_volume)
    ask_delta = np.diff(ask_volume)
    event = increments > 0.0
    valid_event = event & (prior_depth > _EPS) & np.isfinite(current_depth)
    if int(valid_event.sum()) < _MIN_EVENT_INTERVALS:
        return out
    retention = np.minimum(prior_depth[valid_event], current_depth[valid_event]) / prior_depth[valid_event]
    refill = (
        np.maximum(bid_delta[valid_event], 0.0) - np.maximum(ask_delta[valid_event], 0.0)
    ) / prior_depth[valid_event]
    churn = np.abs(bid_delta[valid_event]) + np.abs(ask_delta[valid_event])
    trade_total = float(increments[valid_event].sum())
    out["pql_trade_depth_retention"] = float(np.mean(retention)) if np.isfinite(retention).all() else float("nan")
    out["pql_trade_refill_imbalance"] = float(np.mean(refill)) if np.isfinite(refill).all() else float("nan")
    out["pql_trade_queue_churn_per_trade"] = _safe_div(float(churn.sum()), trade_total)
    return out


def _trade_at_quote_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_TRADE_AT_QUOTE_SIGNALS)
    last = _positive_prices(g, "last")
    book = _top_book_arrays(g)
    increments = _strict_counter_increments(g, "num_trades")
    if last is None or book is None or increments is None:
        return out
    ask, bid, _, _ = book
    event = increments > 0.0
    if int(event.sum()) < _MIN_EVENT_INTERVALS:
        return out
    at_ask = last[1:] >= ask[:-1] * (1.0 - _LIMIT_REL_TOL)
    at_bid = last[1:] <= bid[:-1] * (1.0 + _LIMIT_REL_TOL)
    ambiguous = at_ask & at_bid
    buy = event & at_ask & ~ambiguous
    sell = event & at_bid & ~ambiguous
    event_count = int(event.sum())
    buy_count = int(buy.sum())
    sell_count = int(sell.sum())
    out["taqi_buy_quote_initiation_share"] = float(buy_count / event_count)
    out["taqi_sell_quote_initiation_share"] = float(sell_count / event_count)
    out["taqi_signed_quote_initiation_imbalance"] = _safe_div(
        float(buy_count - sell_count),
        float(buy_count + sell_count),
    )
    return out


def _five_level_book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    ask_price = np.column_stack([_numeric(g, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(g, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(g, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(g, f"bid_volume{level}") for level in range(1, 6)])
    valid = (
        len(ask_depth) >= _MIN_ROWS
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
    return (ask_depth, bid_depth) if valid else None


def _normalized_depth_updates(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(depth) < _MIN_ROWS or depth.ndim != 2 or depth.shape[1] != 5:
        return None
    denominator = depth[1:] + depth[:-1]
    valid = (denominator > _EPS).all(axis=1)
    if int(valid.sum()) < _MIN_EVENT_INTERVALS:
        return None
    updates = np.full_like(denominator, np.nan, dtype="float64")
    np.divide(2.0 * (depth[1:] - depth[:-1]), denominator, out=updates, where=denominator > _EPS)
    if not np.isfinite(updates[valid]).all():
        return None
    return updates, valid


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


def _depth_cascade_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_DEPTH_CASCADE_SIGNALS)
    book = _five_level_book_arrays(g)
    if book is None:
        return out
    ask_depth, bid_depth = book
    ask = _normalized_depth_updates(ask_depth)
    bid = _normalized_depth_updates(bid_depth)
    if ask is None or bid is None:
        return out
    ask_updates, ask_valid = ask
    bid_updates, bid_valid = bid
    bid_total = bid_updates.sum(axis=1)
    ask_total = ask_updates.sum(axis=1)
    forward = _lagged_correlation(bid_total, bid_valid, ask_total, ask_valid)
    reverse = _lagged_correlation(ask_total, ask_valid, bid_total, bid_valid)
    out["dluc_bid_adjacent_update_coherence"] = _adjacent_update_coherence(bid_updates, bid_valid)
    out["dluc_ask_adjacent_update_coherence"] = _adjacent_update_coherence(ask_updates, ask_valid)
    if np.isfinite(forward) and np.isfinite(reverse):
        out["dluc_cross_side_update_lag_asymmetry"] = float(forward - reverse)
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "limit_queue_lock_state_machine": _limit_queue_metrics,
    "passive_queue_lifecycle_after_trade": _passive_queue_metrics,
    "trade_at_quote_initiation_state": _trade_at_quote_metrics,
    "depth_layer_update_cascade": _depth_cascade_metrics,
}


def _full_batch_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Compute all families once when their complete schema is available."""

    cache_key = f"{KERNEL_NAME}:{QUEUE_STATE_V1_VERSION}:full_batch_frame"
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
            for calculator in _FAMILY_CALCULATORS.values():
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
    """Compute/cache all three signals in one requested family."""

    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{QUEUE_STATE_V1_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    signals = _FAMILY_SIGNALS[family]
    _require_columns(frame, ("trade_time", *_FAMILY_REQUIRED_COLUMNS[family]))
    if set(_FULL_BATCH_REQUIRED_COLUMNS).issubset(frame.columns):
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
class FactorMiningIntradayQueueStateV1(Factor):
    """Research-only T1430 single-bond queue-state factor kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
