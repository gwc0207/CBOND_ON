"""Research-only strict-PIT intraday cross-sectional residual catalogue.

This import-only module intentionally stays outside ``defs.__init__`` and all
live/configuration paths.  It derives *same-score-day* T1430 summaries from
the physical ``trade_time`` path through 14:30, then emits rank-Ridge
cross-sectional residuals.  A residual is an abnormal intraday state after
conditioning on other visible scale, liquidity, and volatility states; it is
not a label fit, a pool fit, or a historical prediction model.

The clean-direct panel can label carried observations with the requested score
date.  Index labels are therefore never treated as sufficient point-in-time
evidence: an observation must also have a physical ``trade_time`` on the
score date, within a continuous market session, and no later than 14:30.
Unsafe per-bond paths (missing required values, duplicate clock/sequence
points, material counter resets, crossed books, or invalid depth) fail closed
for the signals that depend on the affected input.  A bounded signed vendor
correction to cumulative ``amount`` is retained as a signed flow increment;
no value is clipped or imputed as zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_cross_section_residual_v1"
CATALOG_VERSION = "20260803_intraday_cross_section_residual_v1_r1"

_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 30)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)

_EPS = 1e-12
_MIN_PATH_ROWS = 12
_MIN_CROSS_SECTION = 30
_RIDGE_ALPHA = 1.0
_AMOUNT_CORRECTION_ABS = 100.0
_AMOUNT_CORRECTION_RATIO = 1e-5


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only factor candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class ResidualDefinition:
    """A same-day target and its visible, non-label conditioning variables."""

    signal: str
    target: str
    covariates: tuple[str, ...]


def _definition(
    signal: str,
    target: str,
    *covariates: str,
) -> ResidualDefinition:
    return ResidualDefinition(signal=signal, target=target, covariates=tuple(covariates))


# The six families deliberately use different *targets*.  They share a
# conservative, same-day cross-sectional conditioning engine, but are not
# simple re-scalings of one raw return, VWAP, spread, depth, or flow factor.
_FAMILY_DEFINITIONS: dict[str, tuple[ResidualDefinition, ...]] = {
    "cs_price_action_abnormality": (
        _definition(
            "csr_price_terminal_return_residual",
            "terminal_return",
            "log_pre_close",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_price_path_efficiency_residual",
            "path_efficiency",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_price_range_residual",
            "log_path_range",
            "terminal_return",
            "log_pre_close",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_price_late_return_residual",
            "late_return",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
        _definition(
            "csr_price_open_gap_residual",
            "open_gap",
            "log_pre_close",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_price_terminal_location_residual",
            "terminal_location",
            "terminal_return",
            "log_path_range",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
        ),
    ),
    "cs_flow_activity_anomaly": (
        _definition(
            "csr_flow_amount_residual",
            "log_amount_total",
            "log_pre_close",
            "terminal_return",
            "realized_volatility",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_flow_trade_count_residual",
            "log_trade_total",
            "log_pre_close",
            "terminal_return",
            "realized_volatility",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_flow_trade_notional_residual",
            "log_average_trade_notional",
            "log_pre_close",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "spread_mean",
        ),
        _definition(
            "csr_flow_concentration_residual",
            "amount_concentration",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_flow_tail_share_residual",
            "tail_amount_share",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "book_imbalance_last",
            "spread_mean",
        ),
        _definition(
            "csr_flow_phase_acceleration_residual",
            "flow_phase_acceleration",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "log_trade_total",
            "spread_mean",
        ),
    ),
    "cs_book_state_anomaly": (
        _definition(
            "csr_book_spread_residual",
            "spread_last",
            "log_pre_close",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "log_depth_last",
        ),
        _definition(
            "csr_book_imbalance_residual",
            "book_imbalance_last",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_book_depth_residual",
            "log_depth_last",
            "log_pre_close",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_book_touch_share_residual",
            "book_touch_share_last",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_book_imbalance_shift_residual",
            "book_imbalance_shift",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_book_depth_rotation_residual",
            "book_depth_rotation",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
    ),
    "cs_phase_rotation_anomaly": (
        _definition(
            "csr_phase_early_late_turn_residual",
            "early_late_turn",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_phase_lunch_reopen_residual",
            "lunch_reopen_return",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
        _definition(
            "csr_phase_late_efficiency_residual",
            "late_path_efficiency",
            "terminal_return",
            "late_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_phase_volatility_ratio_residual",
            "late_early_volatility_ratio",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_phase_extreme_timing_residual",
            "time_of_abs_extreme",
            "terminal_return",
            "log_path_range",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_phase_terminal_location_residual",
            "terminal_location",
            "terminal_return",
            "log_path_range",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
    ),
    "cs_impact_response_anomaly": (
        _definition(
            "csr_impact_amount_return_correlation_residual",
            "amount_return_correlation",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_impact_volume_return_correlation_residual",
            "volume_return_correlation",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_impact_price_per_amount_residual",
            "price_impact_per_amount",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_impact_trade_size_return_residual",
            "trade_size_return_correlation",
            "terminal_return",
            "realized_volatility",
            "log_amount_total",
            "log_trade_total",
            "spread_mean",
        ),
        _definition(
            "csr_impact_depth_return_residual",
            "depth_return_coupling",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_impact_terminal_mid_residual",
            "terminal_mid_dislocation",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
    ),
    "cs_path_risk_anomaly": (
        _definition(
            "csr_risk_realized_volatility_residual",
            "realized_volatility",
            "log_pre_close",
            "terminal_return",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_risk_downside_volatility_residual",
            "downside_volatility",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_risk_max_drawdown_residual",
            "max_drawdown",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "book_imbalance_last",
        ),
        _definition(
            "csr_risk_jump_share_residual",
            "jump_share",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_risk_variance_ratio_residual",
            "variance_ratio_two",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
        _definition(
            "csr_risk_max_move_residual",
            "max_abs_return",
            "terminal_return",
            "realized_volatility",
            "log_trade_total",
            "spread_mean",
            "log_depth_last",
        ),
    ),
}

_FAMILY_HYPOTHESES = {
    "cs_price_action_abnormality": "A price path can be unusual relative to the same-day scale, volatility, visible liquidity, and book state without using a future return.",
    "cs_flow_activity_anomaly": "Abnormal amount, trade, concentration, and late-session activity are separated from visible price movement and displayed liquidity.",
    "cs_book_state_anomaly": "Spread, depth, touch concentration, and imbalance can be unusual after conditioning on same-day price and trading conditions.",
    "cs_phase_rotation_anomaly": "The timing and phase rotation of a path can contain distinct state information after total path and liquidity conditions are removed.",
    "cs_impact_response_anomaly": "Price response to contemporaneous flow and book changes can be abnormal relative to visible activity, not inferred from a label.",
    "cs_path_risk_anomaly": "Path risk measures can be abnormal relative to level, return, and visible liquidity rather than merely restating raw volatility.",
}


def _catalog_entries() -> tuple[CatalogEntry, ...]:
    out: list[CatalogEntry] = []
    for family, definitions in _FAMILY_DEFINITIONS.items():
        hypothesis = _FAMILY_HYPOTHESES[family]
        out.extend(
            CatalogEntry(family=family, signal=item.signal, kernel=KERNEL_NAME, hypothesis=hypothesis)
            for item in definitions
        )
    return tuple(out)


_CATALOG = _catalog_entries()
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    family: tuple(item.signal for item in definitions) for family, definitions in _FAMILY_DEFINITIONS.items()
}
_SUMMARY_COLUMNS = tuple(
    dict.fromkeys(
        (
            "log_pre_close",
            "terminal_return",
            "path_efficiency",
            "log_path_range",
            "late_return",
            "open_gap",
            "terminal_location",
            "realized_volatility",
            "downside_volatility",
            "max_drawdown",
            "jump_share",
            "variance_ratio_two",
            "max_abs_return",
            "log_amount_total",
            "log_trade_total",
            "log_average_trade_notional",
            "amount_concentration",
            "tail_amount_share",
            "flow_phase_acceleration",
            "spread_mean",
            "spread_last",
            "book_imbalance_last",
            "log_depth_last",
            "book_touch_share_last",
            "book_imbalance_shift",
            "book_depth_rotation",
            "early_late_turn",
            "lunch_reopen_return",
            "late_path_efficiency",
            "late_early_volatility_ratio",
            "time_of_abs_extreme",
            "amount_return_correlation",
            "volume_return_correlation",
            "price_impact_per_amount",
            "trade_size_return_correlation",
            "depth_return_coupling",
            "terminal_mid_dislocation",
        )
    )
)
_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            "trade_time",
            "last",
            "pre_close",
            "volume",
            "amount",
            "num_trades",
            *(
                field
                for level in range(1, 6)
                for field in (
                    f"ask_price{level}",
                    f"bid_price{level}",
                    f"ask_volume{level}",
                    f"bid_volume{level}",
                )
            ),
        )
    )
)


def factor_mining_intraday_cross_section_residual_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable six-family, 36-signal research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-launcher compatibility alias."""

    return factor_mining_intraday_cross_section_residual_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve exactly one requested day without trusting rolling index labels."""

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
    """Keep physical score-day, continuous-session observations through 14:30."""

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
    """Return every labelled score-day code, including unsafe codes as NaN rows."""

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


def _safe_log(value: float) -> float:
    return float(np.log(value)) if np.isfinite(value) and value > _EPS else float("nan")


def _safe_corr(left: np.ndarray, right: np.ndarray, *, min_count: int = 4) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < min_count:
        return float("nan")
    a = left[valid]
    b = right[valid]
    if float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _counter_increments(
    g: pd.DataFrame,
    column: str,
    *,
    allow_bounded_amount_correction: bool = False,
) -> np.ndarray | None:
    """Return causal cumulative-counter increments without inventing flow.

    ``amount`` has an existing repository contract for small vendor
    corrections: retain a correction that satisfies both the absolute and
    relative limits as a *signed* increment.  ``volume`` and ``num_trades``
    have no corresponding established correction contract, so any decrease
    leaves only signals requiring that field unavailable.  Neither branch
    clips, fills, or substitutes a counter value.
    """

    values = pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")
    if values.size < _MIN_PATH_ROWS or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all():
        return None
    negative = increments < 0.0
    if negative.any():
        if column != "amount" or not allow_bounded_amount_correction:
            return None
        previous = values[:-1][negative]
        correction = -increments[negative]
        within_absolute = correction <= _AMOUNT_CORRECTION_ABS
        within_relative = correction <= np.abs(previous) * _AMOUNT_CORRECTION_RATIO
        if not bool((within_absolute & within_relative).all()):
            return None
    return increments


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress signed flow values without dropping a bounded correction."""

    out = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values)
    out[valid] = np.sign(values[valid]) * np.log1p(np.abs(values[valid]))
    return out


def _book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask = np.column_stack(
        [pd.to_numeric(g[f"ask_price{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    bid = np.column_stack(
        [pd.to_numeric(g[f"bid_price{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    ask_depth = np.column_stack(
        [pd.to_numeric(g[f"ask_volume{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    bid_depth = np.column_stack(
        [pd.to_numeric(g[f"bid_volume{level}"], errors="coerce").to_numpy(dtype="float64") for level in range(1, 6)]
    )
    valid = (
        np.isfinite(ask).all()
        and np.isfinite(bid).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and (ask > 0.0).all()
        and (bid > 0.0).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (ask[:, 0] >= bid[:, 0]).all()
        and (np.diff(ask, axis=1) >= 0.0).all()
        and (np.diff(bid, axis=1) <= 0.0).all()
        and ((ask_depth.sum(axis=1) + bid_depth.sum(axis=1)) > _EPS).all()
    )
    return (ask, bid, ask_depth, bid_depth) if valid else None


def _masked_return(values: np.ndarray, mask: np.ndarray) -> float:
    part = values[mask]
    if len(part) < 2 or not (np.isfinite(part).all() and (part > 0.0).all()):
        return float("nan")
    return float(np.log(part[-1] / part[0]))


def _masked_efficiency(returns: np.ndarray, interval_mask: np.ndarray) -> float:
    part = returns[interval_mask]
    if len(part) < 2 or not np.isfinite(part).all():
        return float("nan")
    return _safe_div(float(part.sum()), float(np.abs(part).sum()))


def _variance_ratio_two(returns: np.ndarray) -> float:
    count = (len(returns) // 2) * 2
    if count < 8:
        return float("nan")
    fine = returns[:count]
    fine_var = float(np.var(fine, ddof=1))
    if not np.isfinite(fine_var) or fine_var <= _EPS:
        return float("nan")
    coarse = fine.reshape(-1, 2).sum(axis=1)
    if len(coarse) < 4:
        return float("nan")
    return _safe_div(float(np.var(coarse, ddof=1)), 2.0 * fine_var)


def _empty_summary_record() -> dict[str, float]:
    return {column: float("nan") for column in _SUMMARY_COLUMNS}


def _summarize_group(g: pd.DataFrame) -> dict[str, float]:
    """Build same-day summaries, isolating each source field's failure domain."""

    out = _empty_summary_record()
    if len(g) < _MIN_PATH_ROWS:
        return out
    if g["seq"].duplicated(keep=False).any() or g["trade_time"].duplicated(keep=False).any():
        return out
    times = pd.to_datetime(g["trade_time"], errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing:
        return out
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    if (np.diff(time_ns) <= 0).any():
        return out

    prices = pd.to_numeric(g["last"], errors="coerce").to_numpy(dtype="float64")
    pre_close = pd.to_numeric(g["pre_close"], errors="coerce").to_numpy(dtype="float64")
    if not (
        np.isfinite(prices).all()
        and np.isfinite(pre_close).all()
        and (prices > 0.0).all()
        and (pre_close > 0.0).all()
    ):
        return out
    amount = _counter_increments(g, "amount", allow_bounded_amount_correction=True)
    volume = _counter_increments(g, "volume")
    trades = _counter_increments(g, "num_trades")
    book = _book_arrays(g)

    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    if len(returns) < _MIN_PATH_ROWS - 1 or not np.isfinite(returns).all():
        return out

    clocks = times.dt.time.to_numpy()
    early_rows = clocks <= _EARLY_END
    late_rows = clocks >= _LATE_START
    morning_rows = clocks <= _MORNING_END
    afternoon_rows = clocks >= _AFTERNOON_START
    early_intervals = early_rows[1:]
    late_intervals = late_rows[1:]
    total_abs_return = float(np.abs(returns).sum())
    high = float(np.max(prices))
    low = float(np.min(prices))
    path_range = high - low
    max_running = np.maximum.accumulate(log_prices)
    drawdown = log_prices - max_running
    realized_variance = float(np.square(returns).sum())
    bipower = float(np.pi / 2.0 * np.abs(returns[1:] * returns[:-1]).sum()) if len(returns) >= 2 else float("nan")

    out.update(
        {
            "log_pre_close": _safe_log(float(pre_close[0])),
            "terminal_return": float(log_prices[-1] - np.log(pre_close[0])),
            "path_efficiency": _safe_div(float(returns.sum()), total_abs_return),
            "log_path_range": _safe_log(high / low),
            "late_return": _masked_return(prices, late_rows),
            "open_gap": float(log_prices[0] - np.log(pre_close[0])),
            "terminal_location": _safe_div(float(prices[-1] - low), path_range),
            "realized_volatility": float(np.sqrt(realized_variance)),
            "downside_volatility": float(np.sqrt(np.square(np.minimum(returns, 0.0)).sum())),
            "max_drawdown": float(np.min(drawdown)),
            "jump_share": _safe_div(realized_variance - bipower, realized_variance),
            "variance_ratio_two": _variance_ratio_two(returns),
            "max_abs_return": float(np.max(np.abs(returns))),
            "early_late_turn": _masked_return(prices, late_rows) - _masked_return(prices, early_rows),
            "lunch_reopen_return": _safe_log(
                float(prices[afternoon_rows][0] / prices[morning_rows][-1])
            )
            if int(afternoon_rows.sum()) >= 1 and int(morning_rows.sum()) >= 1
            else float("nan"),
            "late_path_efficiency": _masked_efficiency(returns, late_intervals),
            "time_of_abs_extreme": _safe_div(
                float(np.argmax(np.abs(log_prices - np.log(pre_close[0])))), float(len(prices) - 1)
            ),
        }
    )

    if int(early_intervals.sum()) >= 3 and int(late_intervals.sum()) >= 3:
        early_vol = float(np.std(returns[early_intervals], ddof=1))
        late_vol = float(np.std(returns[late_intervals], ddof=1))
        out["late_early_volatility_ratio"] = _safe_div(late_vol, early_vol)

    if trades is not None:
        trade_total = float(trades.sum())
        if trade_total >= 0.0:
            out["log_trade_total"] = float(np.log1p(trade_total))

    if amount is not None:
        amount_total = float(amount.sum())
        if amount_total > _EPS:
            log_amount_total = float(np.log1p(amount_total))
            out.update(
                {
                    "log_amount_total": log_amount_total,
                    "amount_concentration": _safe_div(float(np.max(amount)), amount_total),
                    "tail_amount_share": _safe_div(float(amount[late_intervals].sum()), amount_total),
                    "amount_return_correlation": _safe_corr(_signed_log1p(amount), returns),
                    "price_impact_per_amount": _safe_div(total_abs_return, log_amount_total),
                }
            )
            if trades is not None and trade_total > _EPS:
                out["log_average_trade_notional"] = _safe_log(amount_total / trade_total)
                positive_trade = trades > 0.0
                if int(positive_trade.sum()) >= 4:
                    trade_size = _signed_log1p(amount[positive_trade] / trades[positive_trade])
                    out["trade_size_return_correlation"] = _safe_corr(
                        trade_size,
                        returns[positive_trade],
                    )
        if int(early_intervals.sum()) >= 3 and int(late_intervals.sum()) >= 3:
            early_amount = float(np.mean(amount[early_intervals]))
            late_amount = float(np.mean(amount[late_intervals]))
            out["flow_phase_acceleration"] = (
                _safe_log(late_amount / early_amount) if early_amount > _EPS else float("nan")
            )

    if volume is not None:
        out["volume_return_correlation"] = _safe_corr(np.log1p(volume), returns)

    if book is not None:
        ask, bid, ask_depth, bid_depth = book
        mid = (ask[:, 0] + bid[:, 0]) / 2.0
        spread = (ask[:, 0] - bid[:, 0]) / mid
        total_depth = ask_depth.sum(axis=1) + bid_depth.sum(axis=1)
        imbalance = (bid_depth.sum(axis=1) - ask_depth.sum(axis=1)) / total_depth
        if (
            np.isfinite(mid).all()
            and np.isfinite(spread).all()
            and np.isfinite(total_depth).all()
            and np.isfinite(imbalance).all()
            and (mid > 0.0).all()
            and (spread >= 0.0).all()
            and (total_depth > _EPS).all()
        ):
            touch_share = _safe_div(
                float(ask_depth[-1, 0] + bid_depth[-1, 0]),
                float(total_depth[-1]),
            )
            out.update(
                {
                    "spread_mean": float(np.mean(spread)),
                    "spread_last": float(spread[-1]),
                    "book_imbalance_last": float(imbalance[-1]),
                    "log_depth_last": float(np.log1p(total_depth[-1])),
                    "book_touch_share_last": touch_share,
                    "book_imbalance_shift": float(imbalance[-1] - imbalance[0]),
                    "book_depth_rotation": _safe_log(float(total_depth[-1] / total_depth[0])),
                    "depth_return_coupling": _safe_corr(np.diff(np.log(total_depth)), returns),
                    "terminal_mid_dislocation": _safe_log(float(prices[-1] / mid[-1])),
                }
            )

    return {column: float(out.get(column, np.nan)) for column in _SUMMARY_COLUMNS}


def _summary_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Compute/cache all physical-path summaries once for the 36 signal specs."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:summary_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score = _score_day_frame(ctx)
    _require_columns(score, _REQUIRED_COLUMNS)
    output_index = _output_index(ctx)
    if score.empty:
        built = pd.DataFrame(index=output_index, columns=_SUMMARY_COLUMNS, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        for (dt, code), group in score.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            row.update(_summarize_group(group))
            rows.append(row)
        raw = pd.DataFrame(rows).set_index(["dt", "code"])[list(_SUMMARY_COLUMNS)].sort_index()
        built = raw.reindex(output_index).replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _rank_ridge_residual(
    summary: pd.DataFrame,
    *,
    target: str,
    covariates: tuple[str, ...],
) -> pd.Series:
    """Residualize one raw summary across the same visible cross-section.

    Every target and covariate is independently rank-transformed *on the
    exact finite common cross-section*.  The intercept is unpenalized while
    the fixed Ridge penalty applies only to the visible covariates.  This is
    a deterministic feature transform, never a fit to labels or returns.
    """

    requested = (target, *covariates)
    missing = [column for column in requested if column not in summary.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} residual definition missing summary fields: {missing}")
    out = pd.Series(np.nan, index=summary.index, dtype="float64")
    source = summary.loc[:, list(requested)]
    values = source.to_numpy(dtype="float64")
    valid = np.isfinite(values).all(axis=1)
    min_required = max(_MIN_CROSS_SECTION, 4 * len(covariates) + 8)
    if int(valid.sum()) < min_required:
        return out
    work = source.loc[valid].copy()
    ranked = pd.DataFrame(index=work.index)
    for column in requested:
        values = work[column]
        if int(values.nunique(dropna=True)) < 2:
            return out
        ranked[column] = values.rank(method="average", pct=True) - 0.5
    y = ranked[target].to_numpy(dtype="float64")
    x_covariates = ranked.loc[:, list(covariates)].to_numpy(dtype="float64")
    design = np.column_stack([np.ones(len(work), dtype="float64"), x_covariates])
    penalty = np.eye(design.shape[1], dtype="float64") * _RIDGE_ALPHA
    penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        return out
    residual = y - design @ beta
    if not np.isfinite(residual).all():
        return out
    out.loc[work.index] = residual
    return out


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache six conditionally residualized signals for one family."""

    definitions = _FAMILY_DEFINITIONS.get(family)
    if definitions is None:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    summary = _summary_frame(ctx)
    signals = _FAMILY_SIGNALS[family]
    built = pd.DataFrame(index=summary.index, columns=signals, dtype="float64")
    for definition in definitions:
        built[definition.signal] = _rank_ridge_residual(
            summary,
            target=definition.target,
            covariates=definition.covariates,
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
class FactorMiningIntradayCrossSectionResidualV1(Factor):
    """Research-only T1430 rank-Ridge cross-sectional residual kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
