"""Research-only strict-PIT L1--L5 order-book repricing catalogue.

The factors here are deliberately about *dynamic price-ladder repricing*, not
static spread, imbalance, or depth.  Each repricing event at snapshot ``t``
is defined solely by the validated current ladder and its immediately previous
snapshot ``t-1`` in the same continuous auction session.  No later quote is
used to classify or explain the event.  Depth metrics are likewise the
immediate ``t-1 -> t`` relocation observed after that already-defined event.

Only physical score-day snapshots in the 09:30--11:30 / 13:00--14:29 sessions
are eligible.  The implementation consumes only ``FactorComputeContext.panel``
and does not read labels, pools, PnL, files, databases, Redis, or live state.
Missing columns, crossed/invalid books, non-finite values, zero denominators,
or too few validated repricing events fail closed to ``NaN``.
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


KERNEL_NAME = "factor_mining_orderbook_repricing_v1"
CATALOG_VERSION = "20260803_orderbook_repricing_v1"
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MIN_ROWS = 12
_MIN_EVENT_INTERVALS = 3
_MIN_MOVED_LEVELS = 3
_PRICE_TOL = 1e-12
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_LADDER_DIRECTION_SIGNALS = (
    "lrd_bid_reprice_direction_persistence",
    "lrd_ask_reprice_direction_persistence",
    "lrd_cross_side_reprice_symmetry",
)
_REPRICE_DEPTH_SIGNALS = (
    "rdm_bid_inner_depth_migration",
    "rdm_ask_inner_depth_migration",
    "rdm_joint_reprice_depth_retention",
)

_CATALOG = (
    _entries(
        "price_ladder_reprice_direction",
        _LADDER_DIRECTION_SIGNALS,
        "Persistent versus reversing L1--L5 bid/ask ladder migration and same-interval cross-side symmetry are distinct from a static spread or quote level.",
    )
    + _entries(
        "reprice_conditioned_depth_relocation",
        _REPRICE_DEPTH_SIGNALS,
        "Immediate post-reprice inward-depth relocation and joint-book retention measure how displayed liquidity moves with a validated ladder reprice, not unconditional depth.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "price_ladder_reprice_direction": _LADDER_DIRECTION_SIGNALS,
    "reprice_conditioned_depth_relocation": _REPRICE_DEPTH_SIGNALS,
}


def _book_columns() -> tuple[str, ...]:
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


_REQUIRED_COLUMNS = ("trade_time", *_book_columns())

FORMULAS = {
    "lrd_bid_reprice_direction_persistence": "mean(d_bid,t*d_bid,t-1 | both adjacent intervals are validated bid ladder reprices), where d_t is the sign of the t-1 -> t median L1-L5 log-price move with >=3 moved levels.",
    "lrd_ask_reprice_direction_persistence": "mean(d_ask,t*d_ask,t-1 | both adjacent intervals are validated ask ladder reprices), under the same adjacent-snapshot contract.",
    "lrd_cross_side_reprice_symmetry": "mean(d_bid,t*d_ask,t | both L1-L5 ladders reprice in the same t-1 -> t interval); +1 is symmetric and -1 is opposed.",
    "rdm_bid_inner_depth_migration": "mean(I_bid,t-I_bid,t-1 | validated bid ladder reprice), I=bid_volume1/sum_{k=1..5} bid_volumek.",
    "rdm_ask_inner_depth_migration": "mean(I_ask,t-I_ask,t-1 | validated ask ladder reprice), I=ask_volume1/sum_{k=1..5} ask_volumek.",
    "rdm_joint_reprice_depth_retention": "mean(min(D_t,D_t-1)/D_t-1 | both ladders reprice), D=sum of all L1-L5 bid and ask depth.",
}


def factor_mining_orderbook_repricing_v1_catalog() -> tuple[CatalogEntry, ...]:
    """Return this immutable research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return factor_mining_orderbook_repricing_v1_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        date = pd.Timestamp(raw)
        if pd.isna(date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return date.normalize()
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
    """Keep physical score-day continuous-session snapshots through 14:29."""

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
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        clocks = timestamps.dt.time
        sessions = clocks.map(lambda value: _session_label(value) if pd.notna(value) else None)
        if not frame.empty and not bool((labels.dt.normalize() == target).any()):
            raise ValueError(f"{KERNEL_NAME} has no indexed rows for signal day {target.date().isoformat()}")
        keep = (
            labels.notna()
            & (labels.dt.normalize() == target)
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


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


@dataclass(frozen=True)
class _Book:
    ask_price: np.ndarray
    bid_price: np.ndarray
    ask_depth: np.ndarray
    bid_depth: np.ndarray
    sessions: np.ndarray


def _book(frame: pd.DataFrame) -> _Book | None:
    ask_price = np.column_stack([_numeric(frame, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(frame, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(frame, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(frame, f"bid_volume{level}") for level in range(1, 6)])
    sessions = pd.to_numeric(frame["__session"], errors="coerce").to_numpy(dtype="float64")
    valid = (
        len(frame) >= _MIN_ROWS
        and np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and np.isfinite(sessions).all()
        and (ask_price > 0.0).all()
        and (bid_price > 0.0).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (ask_price[:, 0] >= bid_price[:, 0]).all()
        and (np.diff(ask_price, axis=1) >= 0.0).all()
        and (np.diff(bid_price, axis=1) <= 0.0).all()
    )
    if not valid:
        return None
    return _Book(ask_price, bid_price, ask_depth, bid_depth, sessions.astype("int64"))


def _ladder_directions(prices: np.ndarray, sessions: np.ndarray) -> np.ndarray | None:
    """Classify only t-1 -> t reprices in the same auction session.

    The returned entry at interval ``i`` is based exclusively on prices at
    snapshots ``i`` and ``i+1``.  It never accesses ``i+2`` or later quotes.
    A side reprices only if at least three of five levels move coherently.
    """

    if prices.ndim != 2 or prices.shape[1] != 5 or len(prices) < _MIN_ROWS:
        return None
    log_move = np.log(prices[1:] / prices[:-1])
    if not np.isfinite(log_move).all():
        return None
    same_session = sessions[1:] == sessions[:-1]
    positive = np.sum(log_move > _PRICE_TOL, axis=1)
    negative = np.sum(log_move < -_PRICE_TOL, axis=1)
    upward = same_session & (positive >= _MIN_MOVED_LEVELS) & (positive > negative)
    downward = same_session & (negative >= _MIN_MOVED_LEVELS) & (negative > positive)
    return np.where(upward, 1.0, np.where(downward, -1.0, 0.0))


def _direction_persistence(directions: np.ndarray) -> float:
    if len(directions) < 2:
        return float("nan")
    pairs = (directions[:-1] != 0.0) & (directions[1:] != 0.0)
    if int(pairs.sum()) < _MIN_EVENT_INTERVALS:
        return float("nan")
    return float(np.mean(directions[:-1][pairs] * directions[1:][pairs]))


def _cross_side_symmetry(bid_directions: np.ndarray, ask_directions: np.ndarray) -> float:
    joint = (bid_directions != 0.0) & (ask_directions != 0.0)
    if int(joint.sum()) < _MIN_EVENT_INTERVALS:
        return float("nan")
    return float(np.mean(bid_directions[joint] * ask_directions[joint]))


def _inner_depth_migration(depth: np.ndarray, directions: np.ndarray) -> float:
    totals = depth.sum(axis=1)
    prior = totals[:-1]
    current = totals[1:]
    valid = (directions != 0.0) & np.isfinite(prior) & np.isfinite(current) & (prior > _EPS) & (current > _EPS)
    if int(valid.sum()) < _MIN_EVENT_INTERVALS:
        return float("nan")
    prior_share = depth[:-1, 0] / prior
    current_share = depth[1:, 0] / current
    movement = current_share[valid] - prior_share[valid]
    return float(np.mean(movement)) if np.isfinite(movement).all() else float("nan")


def _joint_depth_retention(book: _Book, bid_directions: np.ndarray, ask_directions: np.ndarray) -> float:
    prior = book.bid_depth[:-1].sum(axis=1) + book.ask_depth[:-1].sum(axis=1)
    current = book.bid_depth[1:].sum(axis=1) + book.ask_depth[1:].sum(axis=1)
    joint = (
        (bid_directions != 0.0)
        & (ask_directions != 0.0)
        & np.isfinite(prior)
        & np.isfinite(current)
        & (prior > _EPS)
        & (current >= 0.0)
    )
    if int(joint.sum()) < _MIN_EVENT_INTERVALS:
        return float("nan")
    retention = np.minimum(prior[joint], current[joint]) / prior[joint]
    return float(np.mean(retention)) if np.isfinite(retention).all() else float("nan")


def _direction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LADDER_DIRECTION_SIGNALS)
    book = _book(frame)
    if book is None:
        return out
    bid_directions = _ladder_directions(book.bid_price, book.sessions)
    ask_directions = _ladder_directions(book.ask_price, book.sessions)
    if bid_directions is None or ask_directions is None:
        return out
    out["lrd_bid_reprice_direction_persistence"] = _direction_persistence(bid_directions)
    out["lrd_ask_reprice_direction_persistence"] = _direction_persistence(ask_directions)
    out["lrd_cross_side_reprice_symmetry"] = _cross_side_symmetry(bid_directions, ask_directions)
    return out


def _depth_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_REPRICE_DEPTH_SIGNALS)
    book = _book(frame)
    if book is None:
        return out
    bid_directions = _ladder_directions(book.bid_price, book.sessions)
    ask_directions = _ladder_directions(book.ask_price, book.sessions)
    if bid_directions is None or ask_directions is None:
        return out
    out["rdm_bid_inner_depth_migration"] = _inner_depth_migration(book.bid_depth, bid_directions)
    out["rdm_ask_inner_depth_migration"] = _inner_depth_migration(book.ask_depth, ask_directions)
    out["rdm_joint_reprice_depth_retention"] = _joint_depth_retention(book, bid_directions, ask_directions)
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "price_ladder_reprice_direction": _direction_metrics,
    "reprice_conditioned_depth_relocation": _depth_metrics,
}


def _full_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:full_feature_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    _require_columns(frame)
    signals = tuple(entry.signal for entry in _CATALOG)
    if frame.empty:
        built = pd.DataFrame(index=_empty_index(), columns=signals, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            values: dict[str, object] = {"dt": dt, "code": str(code)}
            for calculator in _FAMILY_CALCULATORS.values():
                values.update(calculator(group))
            rows.append(values)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(signals)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
        ctx.cache[cache_key] = built
    return built


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _full_feature_frame(ctx).loc[:, list(_FAMILY_SIGNALS[family])].copy()
    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
        ctx.cache[cache_key] = built
    return built


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningOrderbookRepricingV1(Factor):
    """Research-only T1430 L1--L5 repricing factor kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningOrderbookRepricingV1",
    "FORMULAS",
    "factor_mining_catalog",
    "factor_mining_orderbook_repricing_v1_catalog",
]
