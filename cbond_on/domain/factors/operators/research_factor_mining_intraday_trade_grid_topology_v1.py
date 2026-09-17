"""Research-only strict-PIT traded-price increment and grid-topology catalogue.

The module consumes only the physical score-day T1430 bond panel.  It uses
trade-confirmed changes in ``last`` to study phase-resolved step scale,
zero-step spell exits, and magnitude-weighted sign transitions.  It does not
use displayed depth, quote levels, an endpoint price-path return, files,
daily data, stock data, or any live contract.  It is intentionally not
imported by ``defs.__init__``.
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


KERNEL_NAME = "factor_mining_intraday_trade_grid_topology_v1"
CATALOG_VERSION = "20260803_intraday_trade_grid_topology_v1"
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)
_ZERO_REL_TOL = 1e-10
_EPS = 1e-12
_MIN_ROWS = 12
_MIN_MOVING_STEPS = 6
_MIN_PHASE_STEPS = 3
_MIN_ZERO_SPELL_LENGTH = 2
_MIN_EXIT_SPELLS = 3
_MIN_TRANSITION_PAIRS = 3


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable traded-price grid-topology candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class _TradeGridEvents:
    """Strictly validated same-session, trade-confirmed price increments."""

    steps: np.ndarray
    moving: np.ndarray
    zero: np.ndarray
    clocks: np.ndarray


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_PHASE_SCALE_SIGNALS = (
    "tgps_positive_negative_step_scale_asymmetry",
    "tgps_large_step_late_tilt",
    "tgps_median_step_scale_phase_shift",
)
_ZERO_SPELL_EXIT_SIGNALS = (
    "tgze_zero_spell_exit_magnitude",
    "tgze_zero_spell_exit_directional_imbalance",
    "tgze_zero_spell_exit_reversal_share",
)
_TRANSITION_AMPLITUDE_SIGNALS = (
    "tgta_reversal_log_step_amplification",
    "tgta_continuation_log_step_amplification",
    "tgta_updown_reversal_amplitude_asymmetry",
)

_CATALOG = (
    _entries(
        "trade_grid_phase_scale_topology",
        _PHASE_SCALE_SIGNALS,
        "The scale and signed asymmetry of trade-confirmed increments can rotate between early and late sessions without being a raw return or a count of integer step multiples.",
    )
    + _entries(
        "trade_grid_zero_spell_exit",
        _ZERO_SPELL_EXIT_SIGNALS,
        "A consecutive zero-price-change spell has a distinct trade-confirmed exit magnitude, direction, and reversal relation to the pre-spell move; this is not a stationary-run length statistic.",
    )
    + _entries(
        "trade_grid_transition_amplitude",
        _TRANSITION_AMPLITUDE_SIGNALS,
        "The size of a next step relative to its adjacent predecessor can amplify differently after a directional reversal or continuation, beyond sign-only run topology.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# E_t denotes an adjacent, same-continuous-session interval with positive
# delta(num_trades). x_t=last_t-last_t-1; Z_t is an E_t whose magnitude is no
# larger than a scale-aware numerical tolerance; M_t is E_t and not Z_t. The
# robust normalizer u is median(abs(x_t) | M_t). A zero spell has at least two
# consecutive Z_t observations and an immediately adjacent M_t exit.
FORMULAS: dict[str, str] = {
    "tgps_positive_negative_step_scale_asymmetry": "(median(abs(x_t)|M_t,x_t>0)-median(abs(x_t)|M_t,x_t<0))/u.",
    "tgps_large_step_late_tilt": "mean(1[abs(x_t)>1.5u]|M_t,late)-mean(1[abs(x_t)>1.5u]|M_t,early).",
    "tgps_median_step_scale_phase_shift": "log(median(abs(x_t)|M_t,late)/median(abs(x_t)|M_t,early)).",
    "tgze_zero_spell_exit_magnitude": "mean(abs(x_exit)/u) over valid zero-spell exits.",
    "tgze_zero_spell_exit_directional_imbalance": "mean(sign(x_exit)) over valid zero-spell exits.",
    "tgze_zero_spell_exit_reversal_share": "mean(1[sign(x_exit)!=sign(x_pre)]) over exits with an immediately pre-spell M_t.",
    "tgta_reversal_log_step_amplification": "mean(log(abs(x_t)/abs(x_t-1)) | adjacent M_t,M_t-1 and sign(x_t)!=sign(x_t-1)).",
    "tgta_continuation_log_step_amplification": "mean(log(abs(x_t)/abs(x_t-1)) | adjacent M_t,M_t-1 and sign(x_t)=sign(x_t-1)).",
    "tgta_updown_reversal_amplitude_asymmetry": "mean(log(abs(x_t)/abs(x_t-1)) | x_t-1>0,x_t<0)-mean(same | x_t-1<0,x_t>0).",
}


def factor_mining_intraday_trade_grid_topology_v1_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable three-family research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research factor-mining runners."""

    return factor_mining_intraday_trade_grid_topology_v1_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve one score day without trusting a relabelled rolling panel alone."""

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
    """Keep only physical score-day continuous-session rows through 14:29."""

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


_REQUIRED_COLUMNS = ("trade_time", "last", "num_trades")


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _nan_record() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _events(group: pd.DataFrame) -> _TradeGridEvents | None:
    """Build fail-closed trade-confirmed increments without lunch bridging."""

    if len(group) < _MIN_ROWS:
        return None
    trade_times = pd.to_datetime(group["trade_time"], errors="coerce")
    if trade_times.isna().any() or trade_times.duplicated().any() or not trade_times.is_monotonic_increasing:
        return None
    last = pd.to_numeric(group["last"], errors="coerce").to_numpy(dtype="float64")
    trades = pd.to_numeric(group["num_trades"], errors="coerce").to_numpy(dtype="float64")
    sessions = pd.to_numeric(group["__session"], errors="coerce").to_numpy(dtype="float64")
    if (
        not np.isfinite(last).all()
        or not np.isfinite(trades).all()
        or not np.isfinite(sessions).all()
        or (last <= 0.0).any()
        or (trades < 0.0).any()
        or not np.isin(sessions, (0.0, 1.0)).all()
    ):
        return None
    increments = np.diff(trades)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    steps = np.diff(last)
    if not np.isfinite(steps).all():
        return None
    same_session = sessions[1:] == sessions[:-1]
    event = same_session & (increments > 0.0)
    tolerance = _ZERO_REL_TOL * np.maximum(last[1:], last[:-1])
    zero = event & (np.abs(steps) <= tolerance)
    moving = event & ~zero
    if int(moving.sum()) < _MIN_MOVING_STEPS:
        return None
    clocks = trade_times.dt.time.to_numpy()[1:]
    return _TradeGridEvents(steps=steps, moving=moving, zero=zero, clocks=clocks)


def _median(values: np.ndarray) -> float:
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    value = float(np.median(values))
    return value if np.isfinite(value) else float("nan")


def _phase_scale_metrics(events: _TradeGridEvents, *, unit: float) -> dict[str, float]:
    out = {signal: float("nan") for signal in _PHASE_SCALE_SIGNALS}
    steps = events.steps
    magnitudes = np.abs(steps)
    positive = events.moving & (steps > 0.0)
    negative = events.moving & (steps < 0.0)
    if int(positive.sum()) >= _MIN_PHASE_STEPS and int(negative.sum()) >= _MIN_PHASE_STEPS:
        positive_scale = _median(magnitudes[positive])
        negative_scale = _median(magnitudes[negative])
        if np.isfinite(positive_scale) and np.isfinite(negative_scale) and unit > _EPS:
            out["tgps_positive_negative_step_scale_asymmetry"] = float((positive_scale - negative_scale) / unit)

    early = events.moving & np.fromiter((clock <= _EARLY_END for clock in events.clocks), dtype=bool)
    late = events.moving & np.fromiter((clock >= _LATE_START for clock in events.clocks), dtype=bool)
    if int(early.sum()) >= _MIN_PHASE_STEPS and int(late.sum()) >= _MIN_PHASE_STEPS:
        early_scale = _median(magnitudes[early])
        late_scale = _median(magnitudes[late])
        if np.isfinite(early_scale) and np.isfinite(late_scale) and early_scale > _EPS and late_scale > _EPS:
            out["tgps_median_step_scale_phase_shift"] = float(np.log(late_scale / early_scale))
        large_threshold = 1.5 * unit
        if large_threshold > _EPS:
            out["tgps_large_step_late_tilt"] = float(
                np.mean(magnitudes[late] > large_threshold) - np.mean(magnitudes[early] > large_threshold)
            )
    return out


def _zero_spell_exit_indices(events: _TradeGridEvents) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-spell exits and their immediately pre-spell moving steps."""

    exits: list[int] = []
    pre_spell: list[int] = []
    index = 0
    while index < len(events.zero):
        if not bool(events.zero[index]):
            index += 1
            continue
        start = index
        while index < len(events.zero) and bool(events.zero[index]):
            index += 1
        end = index - 1
        if end - start + 1 < _MIN_ZERO_SPELL_LENGTH:
            continue
        if index < len(events.moving) and bool(events.moving[index]):
            exits.append(index)
            if start > 0 and bool(events.moving[start - 1]):
                pre_spell.append(start - 1)
            else:
                pre_spell.append(-1)
    return np.asarray(exits, dtype="int64"), np.asarray(pre_spell, dtype="int64")


def _zero_spell_exit_metrics(events: _TradeGridEvents, *, unit: float) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ZERO_SPELL_EXIT_SIGNALS}
    exits, pre_spell = _zero_spell_exit_indices(events)
    if exits.size < _MIN_EXIT_SPELLS or unit <= _EPS:
        return out
    exit_steps = events.steps[exits]
    out["tgze_zero_spell_exit_magnitude"] = float(np.mean(np.abs(exit_steps) / unit))
    out["tgze_zero_spell_exit_directional_imbalance"] = float(np.mean(np.sign(exit_steps)))
    valid_pre = pre_spell >= 0
    if int(valid_pre.sum()) >= _MIN_EXIT_SPELLS:
        prior_steps = events.steps[pre_spell[valid_pre]]
        paired_exits = exit_steps[valid_pre]
        out["tgze_zero_spell_exit_reversal_share"] = float(np.mean(np.sign(prior_steps) != np.sign(paired_exits)))
    return out


def _transition_amplitude_metrics(events: _TradeGridEvents) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TRANSITION_AMPLITUDE_SIGNALS}
    adjacent = events.moving[:-1] & events.moving[1:]
    if int(adjacent.sum()) < _MIN_TRANSITION_PAIRS:
        return out
    prior = events.steps[:-1][adjacent]
    current = events.steps[1:][adjacent]
    log_ratio = np.log(np.abs(current) / np.abs(prior))
    if not np.isfinite(log_ratio).all():
        return out
    reversal = np.sign(prior) != np.sign(current)
    continuation = ~reversal
    if int(reversal.sum()) >= _MIN_TRANSITION_PAIRS:
        out["tgta_reversal_log_step_amplification"] = float(np.mean(log_ratio[reversal]))
    if int(continuation.sum()) >= _MIN_TRANSITION_PAIRS:
        out["tgta_continuation_log_step_amplification"] = float(np.mean(log_ratio[continuation]))
    up_down = (prior > 0.0) & (current < 0.0)
    down_up = (prior < 0.0) & (current > 0.0)
    if int(up_down.sum()) >= _MIN_TRANSITION_PAIRS and int(down_up.sum()) >= _MIN_TRANSITION_PAIRS:
        out["tgta_updown_reversal_amplitude_asymmetry"] = float(
            np.mean(log_ratio[up_down]) - np.mean(log_ratio[down_up])
        )
    return out


def _metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    events = _events(group)
    if events is None:
        return out
    unit = _median(np.abs(events.steps[events.moving]))
    if not np.isfinite(unit) or unit <= _EPS:
        return out
    out.update(_phase_scale_metrics(events, unit=unit))
    out.update(_zero_spell_exit_metrics(events, unit=unit))
    out.update(_transition_amplitude_metrics(events))
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Compute all grid-topology signals once and cache the frame in context."""

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
class FactorMiningIntradayTradeGridTopologyV1(Factor):
    """Research-only T1430 traded-price increment/grid topology kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _feature_frame(ctx)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
