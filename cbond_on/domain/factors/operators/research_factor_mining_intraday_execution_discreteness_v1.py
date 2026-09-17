"""Research-only strict-PIT intraday execution-discreteness factors.

This module is deliberately import-only: it is not part of ``defs.__init__``
and is not referenced by a production, model, or live configuration.  It uses
only physical score-day T1430 panel observations through 14:29:00.  The input
fields are ``trade_time``, ``last``, and ``num_trades``; no quote, daily,
stock, label, score, PnL, file, database, or execution output is read.

The catalogue is intentionally not a renamed price-update rate, longest
stable-price run, ordinary return autocorrelation, or largest-jump share.
It first restricts all calculations to trade-confirmed, same-session price
intervals.  It then describes (1) the *count distribution* of non-zero price
step multiples relative to a robust small-step scale, (2) the topology of the
compressed non-zero step direction sequence, and (3) the distribution of
stationary and moving run lengths among trade-confirmed intervals.

Any absent required field, mixed physical date, duplicate or non-monotonic
timestamp, in-session gap above twenty minutes, invalid price/counter, or
``num_trades`` reset fails closed to ``NaN`` for that bond-day.  Lunch is an
explicit session boundary and no interval is allowed to bridge it.
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


KERNEL_NAME = "factor_mining_intraday_execution_discreteness_v1"
CATALOG_VERSION = "20260803_intraday_execution_discreteness_v1"
_EPS = 1e-12
_PRICE_REL_TOL = 1e-8
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MAX_IN_SESSION_GAP_SECONDS = 20.0 * 60.0
_MIN_ROWS = 12
_MIN_TRADE_EVENTS = 8
_MIN_NONZERO_UPDATES = 6
_MIN_DIRECTION_RUNS = 3
_MIN_STATIONARY_RUNS = 2
_MIN_MOVING_RUNS = 2


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only execution-discreteness candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis)
        for signal in signals
    )


_STEP_MULTIPLICITY_SIGNALS = (
    "execdisc_step_unit_share",
    "execdisc_step_multiplicity_entropy",
    "execdisc_step_large_multiple_share",
)
_DIRECTION_TOPOLOGY_SIGNALS = (
    "execdisc_direction_reversal_rate",
    "execdisc_direction_run_concentration",
    "execdisc_direction_run_length_entropy",
)
_STATIONARY_MOVING_SIGNALS = (
    "execdisc_stationary_run_length_entropy",
    "execdisc_stationary_moving_mean_length_ratio",
    "execdisc_stationary_run_concentration",
)

_CATALOG = (
    _entries(
        "execution_step_multiplicity_geometry",
        _STEP_MULTIPLICITY_SIGNALS,
        "The count geometry of trade-confirmed non-zero price steps, expressed as robust small-step multiples, can distinguish granular updating from sparse multi-step repricing without using return magnitude shares.",
    )
    + _entries(
        "execution_nonzero_direction_topology",
        _DIRECTION_TOPOLOGY_SIGNALS,
        "The compressed signs of actual non-zero trade-confirmed price steps encode reversal and run topology, rather than quote-side aggressor labels or endpoint return.",
    )
    + _entries(
        "execution_stationary_moving_regime",
        _STATIONARY_MOVING_SIGNALS,
        "The distribution of stable and moving run lengths among trade-confirmed intervals describes an execution regime without returning a price-change rate, stationary share, or longest stable run.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# Let E_t denote a same-continuous-session interval with delta num_trades > 0,
# x_t = last_t - last_{t-1}, M_t = 1[abs(x_t) exceeds the scale-aware price
# tolerance], and u be the median of the smallest max(3, ceil(25%)) values of
# abs(x_t) over non-zero E_t intervals.  m_t=max(1, round(abs(x_t)/u)).
FORMULAS: dict[str, str] = {
    "execdisc_step_unit_share": "mean(1[m_t=1]) over non-zero E_t; u=median(lower-quartile abs(x_t))",
    "execdisc_step_multiplicity_entropy": "-sum_k p_k*log(p_k)/log(K), p_k=mean(1[m_t=k])",
    "execdisc_step_large_multiple_share": "mean(1[m_t>=3]) over non-zero E_t",
    "execdisc_direction_reversal_rate": "mean(1[d_j!=d_{j-1}]) within session, d_j=sign(x_t) on compressed non-zero E_t",
    "execdisc_direction_run_concentration": "sum_r (L_r/N)^2 over within-session runs r of compressed d_j",
    "execdisc_direction_run_length_entropy": "entropy of the empirical within-session direction-run-length histogram",
    "execdisc_stationary_run_length_entropy": "entropy of stationary-run length shares over E_t, M_t=0, without lunch bridging",
    "execdisc_stationary_moving_mean_length_ratio": "mean(stationary run length)/mean(moving run length) over E_t",
    "execdisc_stationary_run_concentration": "sum_r (L_r/sum_q L_q)^2 over stationary E_t runs",
}

_REQUIRED_PANEL_COLUMNS = ("trade_time", "last", "num_trades")


def factor_mining_intraday_execution_discreteness_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research catalogue loaders."""

    return factor_mining_intraday_execution_discreteness_catalog()


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
    """Return only physical T-day continuous-session rows visible by 14:29."""

    checked = ensure_panel_index(panel)
    if any(column not in checked.columns for column in _REQUIRED_PANEL_COLUMNS):
        return None
    checked = ensure_trade_time(checked)
    frame = checked.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(
        lambda clock: _continuous_session(clock) if pd.notna(clock) else False
    )
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


def _sessions(group: pd.DataFrame) -> np.ndarray | None:
    """Validate clocks and return an explicit morning/afternoon session id."""

    parsed = pd.to_datetime(group["trade_time"], errors="coerce")
    if len(parsed) < _MIN_ROWS or parsed.isna().any() or parsed.duplicated().any():
        return None
    if (parsed.diff().dropna() <= pd.Timedelta(0)).any():
        return None

    out = np.full(len(parsed), -1, dtype="int8")
    for index, clock in enumerate(parsed.dt.time.to_numpy()):
        if _MORNING_START <= clock <= _MORNING_END:
            out[index] = 0
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            out[index] = 1
        else:
            return None
    for session in (0, 1):
        local = parsed.iloc[out == session]
        if local.empty:
            return None
        gaps = local.diff().dropna().dt.total_seconds().to_numpy(dtype="float64")
        if not np.isfinite(gaps).all() or (gaps <= 0.0).any() or (gaps > _MAX_IN_SESSION_GAP_SECONDS).any():
            return None
    return out


def _event_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return trade-confirmed price deltas, movement states, and session ids."""

    if len(group) < _MIN_ROWS:
        return None
    sessions = _sessions(group)
    if sessions is None:
        return None
    last = _numeric(group, "last")
    count = _numeric(group, "num_trades")
    if (
        not np.isfinite(last).all()
        or not np.isfinite(count).all()
        or (last <= 0.0).any()
        or (count < 0.0).any()
    ):
        return None
    increments = np.diff(count)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    same_session = sessions[1:] == sessions[:-1]
    event = same_session & (increments > 0.0)
    if int(event.sum()) < _MIN_TRADE_EVENTS:
        return None
    delta = np.diff(last)
    scale = np.maximum(last[1:], last[:-1])
    moving = np.abs(delta) > (_PRICE_REL_TOL * scale)
    event_delta = delta[event]
    event_moving = moving[event]
    event_sessions = sessions[1:][event]
    if (
        len(event_delta) < _MIN_TRADE_EVENTS
        or not np.isfinite(event_delta).all()
        or not np.isfinite(event_sessions).all()
    ):
        return None
    return event_delta, event_moving, event_sessions


def _normalized_entropy(weights: np.ndarray) -> float:
    """Return normalized Shannon entropy of non-negative discrete weights."""

    weights = np.asarray(weights, dtype="float64")
    if weights.size == 0 or not np.isfinite(weights).all() or (weights < 0.0).any():
        return float("nan")
    total = float(weights.sum())
    if total <= _EPS:
        return float("nan")
    positive = weights[weights > _EPS]
    if positive.size <= 1:
        return 0.0
    probabilities = positive / float(positive.sum())
    value = -float(np.dot(probabilities, np.log(probabilities))) / float(np.log(len(probabilities)))
    return value if np.isfinite(value) and value >= -_EPS else float("nan")


def _step_multiplicity_metrics(abs_steps: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _STEP_MULTIPLICITY_SIGNALS}
    steps = np.asarray(abs_steps, dtype="float64")
    if (
        len(steps) < _MIN_NONZERO_UPDATES
        or not np.isfinite(steps).all()
        or (steps <= _EPS).any()
    ):
        return out
    lower_count = min(len(steps), max(3, int(np.ceil(len(steps) * 0.25))))
    small_scale = float(np.median(np.sort(steps)[:lower_count]))
    if not np.isfinite(small_scale) or small_scale <= _EPS:
        return out
    ratio = steps / small_scale
    if not np.isfinite(ratio).all() or (ratio <= 0.0).any():
        return out
    multiple = np.maximum(1, np.rint(ratio).astype("int64"))
    _, counts = np.unique(multiple, return_counts=True)
    out.update(
        {
            "execdisc_step_unit_share": float(np.mean(multiple == 1)),
            "execdisc_step_multiplicity_entropy": _normalized_entropy(counts.astype("float64")),
            "execdisc_step_large_multiple_share": float(np.mean(multiple >= 3)),
        }
    )
    return out


def _run_lengths(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype="int64")
    lengths: list[int] = []
    current = 1
    for left, right in zip(values[:-1], values[1:], strict=True):
        if right == left:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return np.asarray(lengths, dtype="int64")


def _direction_metrics(nonzero_delta: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _DIRECTION_TOPOLOGY_SIGNALS}
    delta = np.asarray(nonzero_delta, dtype="float64")
    session_id = np.asarray(sessions, dtype="int8")
    if (
        len(delta) < _MIN_NONZERO_UPDATES
        or len(delta) != len(session_id)
        or not np.isfinite(delta).all()
        or (np.abs(delta) <= _EPS).any()
    ):
        return out
    directions = np.sign(delta).astype("int8")
    total = 0
    transitions = 0
    reversals = 0
    run_lengths: list[int] = []
    for session in (0, 1):
        local = directions[session_id == session]
        if len(local) == 0:
            continue
        total += len(local)
        if len(local) > 1:
            transitions += len(local) - 1
            reversals += int(np.sum(local[1:] != local[:-1]))
        run_lengths.extend(_run_lengths(local).tolist())
    if total < _MIN_NONZERO_UPDATES or transitions <= 0 or len(run_lengths) < _MIN_DIRECTION_RUNS:
        return out
    length_array = np.asarray(run_lengths, dtype="float64")
    coverage = length_array / float(total)
    _, counts = np.unique(length_array.astype("int64"), return_counts=True)
    out.update(
        {
            "execdisc_direction_reversal_rate": float(reversals / transitions),
            "execdisc_direction_run_concentration": float(np.square(coverage).sum()),
            "execdisc_direction_run_length_entropy": _normalized_entropy(counts.astype("float64")),
        }
    )
    return out


def _state_runs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(states) == 0:
        return np.asarray([], dtype=bool), np.asarray([], dtype="int64")
    values: list[bool] = []
    lengths: list[int] = []
    current = bool(states[0])
    length = 1
    for value in states[1:]:
        current_value = bool(value)
        if current_value == current:
            length += 1
        else:
            values.append(current)
            lengths.append(length)
            current = current_value
            length = 1
    values.append(current)
    lengths.append(length)
    return np.asarray(values, dtype=bool), np.asarray(lengths, dtype="int64")


def _stationary_metrics(moving: np.ndarray, sessions: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _STATIONARY_MOVING_SIGNALS}
    movement = np.asarray(moving, dtype=bool)
    session_id = np.asarray(sessions, dtype="int8")
    if len(movement) < _MIN_TRADE_EVENTS or len(movement) != len(session_id):
        return out
    states: list[bool] = []
    lengths: list[int] = []
    for session in (0, 1):
        local = movement[session_id == session]
        if len(local) == 0:
            continue
        local_states, local_lengths = _state_runs(local)
        states.extend(local_states.tolist())
        lengths.extend(local_lengths.tolist())
    if not lengths:
        return out
    state_array = np.asarray(states, dtype=bool)
    length_array = np.asarray(lengths, dtype="float64")
    stationary = length_array[~state_array]
    moving_lengths = length_array[state_array]
    if len(stationary) >= _MIN_STATIONARY_RUNS:
        stationary_share = stationary / float(stationary.sum())
        out["execdisc_stationary_run_length_entropy"] = _normalized_entropy(stationary_share)
        out["execdisc_stationary_run_concentration"] = float(np.square(stationary_share).sum())
    if len(stationary) >= _MIN_STATIONARY_RUNS and len(moving_lengths) >= _MIN_MOVING_RUNS:
        denominator = float(np.mean(moving_lengths))
        if np.isfinite(denominator) and denominator > _EPS:
            out["execdisc_stationary_moving_mean_length_ratio"] = float(np.mean(stationary) / denominator)
    return out


def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    arrays = _event_arrays(group)
    if arrays is None:
        return out
    event_delta, event_moving, event_sessions = arrays
    nonzero_delta = event_delta[event_moving]
    nonzero_sessions = event_sessions[event_moving]
    if len(nonzero_delta) < _MIN_NONZERO_UPDATES:
        return out
    out.update(_step_multiplicity_metrics(np.abs(nonzero_delta)))
    out.update(_direction_metrics(nonzero_delta, nonzero_sessions))
    out.update(_stationary_metrics(event_moving, event_sessions))
    return {signal: (float(value) if np.isfinite(value) else float("nan")) for signal, value in out.items()}


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
class FactorMiningIntradayExecutionDiscretenessV1(Factor):
    """Research-only price-step, direction, and stationary-run candidates."""

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
