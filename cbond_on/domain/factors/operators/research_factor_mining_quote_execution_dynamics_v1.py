"""Research-only strict-PIT prior-quote execution-dynamics factors.

This module replaces neither the existing queue-state catalogue nor any live
factor.  It uses only a bond's physical T-day panel through 14:29:00 and
attributes a transaction snapshot at t against the *strictly previous*
top-of-book quote at t-1.  No price return, label, score, PnL, file, database,
or execution result is read.

The first family describes the continuous distribution of transaction
locations within or beyond the prior spread.  The second describes the
sequence of only quote-side-verified directions.  Missing fields, invalid
book values, counter resets, insufficient observed events, or insufficient
directional events fail closed to NaN.
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


KERNEL_NAME = "factor_mining_quote_execution_dynamics_v1"
CATALOG_VERSION = "20260803_quote_execution_dynamics_v1"
_EPS = 1e-12
_AT_QUOTE_TOL = 1e-8
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MIN_ROWS = 12
_MIN_QUOTE_EVENTS = 4
_MIN_DIRECTION_EVENTS = 4


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only quote-execution candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis) for signal in signals)


_LOCATION_SHAPE_SIGNALS = (
    "qed_prior_quote_location_dispersion",
    "qed_prior_quote_inside_spread_share",
    "qed_prior_quote_tail_penetration",
)
_DIRECTION_DYNAMICS_SIGNALS = (
    "qed_prior_quote_direction_switch_rate",
    "qed_prior_quote_longest_run_share",
    "qed_prior_quote_lag2_agreement",
)

_CATALOG = (
    _entries(
        "quote_execution_location_shape",
        _LOCATION_SHAPE_SIGNALS,
        "Prior-quote normalized execution location has dispersion, inside-spread mass, and beyond-quote penetration that are distinct from buy/sell shares.",
    )
    + _entries(
        "quote_side_sequence_dynamics",
        _DIRECTION_DYNAMICS_SIGNALS,
        "Only prior-quote-verified buy/sell directions enter switch, run-concentration, and two-event agreement dynamics; no aggressor flag is inferred.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# E_t denotes an interval with positive delta num_trades and a valid prior
# spread. z_t is measured with last_t against the prior L1 midpoint/spread.
FORMULAS: dict[str, str] = {
    "qed_prior_quote_location_dispersion": "weighted_sd(z_t | E_t), z_t=(last_t-mid_{t-1})/(spread_{t-1}/2)",
    "qed_prior_quote_inside_spread_share": "sum(deltaN_t * 1[abs(z_t)<1]) / sum(deltaN_t) over E_t",
    "qed_prior_quote_tail_penetration": "weighted_mean(log1p(max(abs(z_t)-1,0)) | E_t)",
    "qed_prior_quote_direction_switch_rate": "mean(1[d_t != d_{t-1}]), d_t in {+1,-1} only at/beyond prior ask/bid",
    "qed_prior_quote_longest_run_share": "max contiguous same-side run length / number of verified directions",
    "qed_prior_quote_lag2_agreement": "mean(d_t * d_{t-2}) over ordered verified prior-quote directions",
}

_REQUIRED_PANEL_COLUMNS = ("trade_time", "last", "ask_price1", "bid_price1", "num_trades")


def factor_mining_quote_execution_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research catalogue loaders."""

    return factor_mining_quote_execution_dynamics_catalog()


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


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.DataFrame | None:
    """Return physical T-day continuous-session rows through 14:29:00 only."""

    checked = ensure_panel_index(panel)
    if any(column not in checked.columns for column in _REQUIRED_PANEL_COLUMNS):
        return None
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


def _event_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Return prior-quote locations and positive counter-increment weights."""

    if len(group) < _MIN_ROWS:
        return None
    trade_times = pd.to_datetime(group["trade_time"], errors="coerce")
    if trade_times.isna().any() or trade_times.duplicated().any():
        return None
    last = pd.to_numeric(group["last"], errors="coerce").to_numpy(dtype="float64")
    ask = pd.to_numeric(group["ask_price1"], errors="coerce").to_numpy(dtype="float64")
    bid = pd.to_numeric(group["bid_price1"], errors="coerce").to_numpy(dtype="float64")
    count = pd.to_numeric(group["num_trades"], errors="coerce").to_numpy(dtype="float64")
    if (
        not np.isfinite(last).all()
        or not np.isfinite(ask).all()
        or not np.isfinite(bid).all()
        or not np.isfinite(count).all()
        or (last <= 0.0).any()
        or (ask <= 0.0).any()
        or (bid <= 0.0).any()
        or (count < 0.0).any()
        or (ask < bid).any()
    ):
        return None
    increments = np.diff(count)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    prior_spread = ask[:-1] - bid[:-1]
    event = (increments > 0.0) & (prior_spread > _EPS)
    if int(event.sum()) < _MIN_QUOTE_EVENTS:
        return None
    prior_midpoint = (ask[:-1] + bid[:-1]) / 2.0
    location = (last[1:] - prior_midpoint) / (prior_spread / 2.0)
    location = location[event]
    weights = increments[event]
    if (
        len(location) < _MIN_QUOTE_EVENTS
        or not np.isfinite(location).all()
        or not np.isfinite(weights).all()
        or (weights <= _EPS).any()
    ):
        return None
    return location, weights


def _location_metrics(location: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    out = _nan_record()
    total = float(weights.sum())
    if not np.isfinite(total) or total <= _EPS:
        return out
    center = float(np.dot(weights, location) / total)
    variance = float(np.dot(weights, (location - center) ** 2) / total)
    if not np.isfinite(variance) or variance < 0.0:
        return out
    out["qed_prior_quote_location_dispersion"] = float(np.sqrt(max(variance, 0.0)))
    out["qed_prior_quote_inside_spread_share"] = float(
        weights[np.abs(location) < (1.0 - _AT_QUOTE_TOL)].sum() / total
    )
    penetration = np.log1p(np.maximum(np.abs(location) - 1.0, 0.0))
    out["qed_prior_quote_tail_penetration"] = float(np.dot(weights, penetration) / total)
    return out


def _longest_run_share(direction: np.ndarray) -> float:
    if direction.size == 0:
        return float("nan")
    longest = 1
    current = 1
    for value, previous in zip(direction[1:], direction[:-1], strict=True):
        if value == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    longest = max(longest, current)
    return float(longest / len(direction))


def _direction_metrics(location: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _DIRECTION_DYNAMICS_SIGNALS}
    direction = np.where(
        location >= (1.0 - _AT_QUOTE_TOL),
        1.0,
        np.where(location <= (-1.0 + _AT_QUOTE_TOL), -1.0, 0.0),
    )
    direction = direction[direction != 0.0]
    if len(direction) < _MIN_DIRECTION_EVENTS:
        return out
    out["qed_prior_quote_direction_switch_rate"] = float(np.mean(direction[1:] != direction[:-1]))
    out["qed_prior_quote_longest_run_share"] = _longest_run_share(direction)
    out["qed_prior_quote_lag2_agreement"] = float(np.mean(direction[2:] * direction[:-2]))
    return out


def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
    arrays = _event_arrays(group)
    if arrays is None:
        return _nan_record()
    location, weights = arrays
    out = _location_metrics(location, weights)
    out.update(_direction_metrics(location))
    return {signal: float(value) if np.isfinite(value) else float("nan") for signal, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    strict = None if score_date is None else _strict_physical_frame(ctx.panel, score_date=score_date)
    if score_date is None or output_index.empty or strict is None or strict.empty:
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
class FactorMiningQuoteExecutionDynamicsV1(Factor):
    """Research-only prior-quote execution-distribution and sequence factors."""

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
