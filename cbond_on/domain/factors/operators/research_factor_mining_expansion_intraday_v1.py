"""Research-only T1430 intraday expansion catalogue.

This module is deliberately *not* imported by :mod:`defs.__init__`: it is an
isolated candidate pool for IC mining, not a production factor pack.  It
contains one registered kernel and sixty explicit signals, arranged as ten
economic/microstructure families.  The caller must import this module and
declare a ``signal`` parameter for every factor spec.

The implementation consumes only ``FactorComputeContext.panel``.  In
particular, it does not read files, databases, labels, scores, or results.
Every path is physically restricted to the score date and to snapshots at or
before 14:30.  This matters for ``clean_direct`` panels: their index ``dt`` may
label carried prior-day rows as the score day, so ``dt`` alone is not a valid
point-in-time proof.
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


KERNEL_NAME = "factor_mining_intraday_expansion_v1"
EXPANSION_VERSION = "20260803_intraday_v1"
_CUTOFF = dt_time(14, 30)
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research candidate in the expansion catalogue."""

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


_OPEN_GAP_SIGNALS = (
    "exp_gap_initial",
    "exp_gap_absorption",
    "exp_gap_residual",
    "exp_gap_extreme_extension",
    "exp_gap_reversal_efficiency",
    "exp_gap_volume_weighted_reversal",
)
_ROTATION_SIGNALS = (
    "exp_rotation_early_late_return_gap",
    "exp_rotation_mid_late_return_turn",
    "exp_rotation_time_of_extreme",
    "exp_rotation_return_sign_flip_rate",
    "exp_rotation_late_abs_move_share",
    "exp_rotation_segment_return_dispersion",
)
_NOISE_SIGNALS = (
    "exp_noise_variance_ratio_2",
    "exp_noise_variance_ratio_5",
    "exp_noise_coarse_fine_vol_ratio",
    "exp_noise_return_abs_autocorr",
    "exp_noise_bipower_jump_share",
    "exp_noise_median_mean_abs_return_ratio",
)
_EXECUTION_SIGNALS = (
    "exp_exec_amount_price_dispersion",
    "exp_exec_trade_price_dispersion",
    "exp_exec_tail_vwap_shift",
    "exp_exec_vwap_last_residual",
    "exp_exec_trade_size_return_corr",
    "exp_exec_amount_concentration_impact",
)
_BOOK_CURVE_SIGNALS = (
    "exp_book_bid_curve_convexity",
    "exp_book_ask_curve_convexity",
    "exp_book_curve_asymmetry",
    "exp_book_bid_top_cliff",
    "exp_book_ask_top_cliff",
    "exp_book_depth_spread_elasticity",
)
_SUPPLY_SIGNALS = (
    "exp_supply_bid_replenishment",
    "exp_supply_ask_replenishment",
    "exp_supply_bid_depletion_share",
    "exp_supply_ask_depletion_share",
    "exp_supply_imbalance_trend",
    "exp_supply_refill_price_coupling",
)
_LIMIT_SIGNALS = (
    "exp_limit_up_distance",
    "exp_limit_down_distance",
    "exp_limit_nearest_signed_pressure",
    "exp_limit_pressure_change",
    "exp_limit_near_touch_share",
    "exp_limit_directional_path",
)
_IOPV_SIGNALS = (
    "exp_iopv_basis_last",
    "exp_iopv_basis_mean",
    "exp_iopv_basis_convergence",
    "exp_iopv_basis_volatility",
    "exp_iopv_basis_return_correlation",
    "exp_iopv_basis_flow_correlation",
)
_STICKINESS_SIGNALS = (
    "exp_stick_price_change_rate",
    "exp_stick_price_run_length",
    "exp_stick_quote_update_rate",
    "exp_stick_quote_trade_gap",
    "exp_stick_mid_trade_dislocation",
    "exp_stick_spread_staleness",
)
_LEAD_LAG_SIGNALS = (
    "exp_leadlag_flow_leads_return",
    "exp_leadlag_return_leads_flow",
    "exp_leadlag_trade_size_leads_return",
    "exp_leadlag_amount_return_asymmetry",
    "exp_leadlag_flow_feedback",
    "exp_leadlag_tail_flow_response",
)


_CATALOG = (
    _entries(
        "open_gap_absorption",
        _OPEN_GAP_SIGNALS,
        "Opening displacement is informative only through the way the T-day path absorbs, extends, or reverses it.",
    )
    + _entries(
        "clock_time_rotation",
        _ROTATION_SIGNALS,
        "Early, mid-session, and pre-14:30 returns can rotate even when their full-path endpoint is similar.",
    )
    + _entries(
        "multiscale_noise_variance",
        _NOISE_SIGNALS,
        "Fine-versus-coarse variation distinguishes diffusive price discovery from noisy or jump-dominated paths.",
    )
    + _entries(
        "execution_price_dispersion",
        _EXECUTION_SIGNALS,
        "The distribution of realized trade prices and incremental execution flow contains information beyond a raw VWAP gap.",
    )
    + _entries(
        "book_depth_curve_cliff",
        _BOOK_CURVE_SIGNALS,
        "Five-level depth geometry measures whether displayed liquidity is smoothly distributed or concentrated at a fragile top level.",
    )
    + _entries(
        "liquidity_supply_depletion",
        _SUPPLY_SIGNALS,
        "Dynamic bid/ask replenishment and depletion describe liquidity supply rather than a static book imbalance.",
    )
    + _entries(
        "bond_limit_pressure",
        _LIMIT_SIGNALS,
        "Convertible-bond-specific upper/lower limit proximity captures asymmetric price constraints before 14:30.",
    )
    + _entries(
        "iopv_basis_convergence",
        _IOPV_SIGNALS,
        "The executable bond price can converge to or diverge from contemporaneous IOPV through the session.",
    )
    + _entries(
        "quote_trade_stickiness",
        _STICKINESS_SIGNALS,
        "The relative update clocks of trades and best quotes distinguish stale displayed liquidity from active price discovery.",
    )
    + _entries(
        "flow_return_lead_lag",
        _LEAD_LAG_SIGNALS,
        "Lagged incremental flow/return relations target intraday transmission timing rather than contemporaneous correlation.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}


def factor_mining_expansion_intraday_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only expansion catalogue in family order."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return factor_mining_expansion_intraday_catalog()


_FAMILY_SIGNALS = {
    "open_gap_absorption": _OPEN_GAP_SIGNALS,
    "clock_time_rotation": _ROTATION_SIGNALS,
    "multiscale_noise_variance": _NOISE_SIGNALS,
    "execution_price_dispersion": _EXECUTION_SIGNALS,
    "book_depth_curve_cliff": _BOOK_CURVE_SIGNALS,
    "liquidity_supply_depletion": _SUPPLY_SIGNALS,
    "bond_limit_pressure": _LIMIT_SIGNALS,
    "iopv_basis_convergence": _IOPV_SIGNALS,
    "quote_trade_stickiness": _STICKINESS_SIGNALS,
    "flow_return_lead_lag": _LEAD_LAG_SIGNALS,
}


def _book_columns() -> tuple[str, ...]:
    columns: list[str] = []
    for level in range(1, 6):
        columns.extend(
            (
                f"ask_price{level}",
                f"ask_volume{level}",
                f"bid_price{level}",
                f"bid_volume{level}",
            )
        )
    return tuple(columns)


_FAMILY_REQUIRED_COLUMNS = {
    "open_gap_absorption": ("last", "pre_close", "volume"),
    "clock_time_rotation": ("last",),
    "multiscale_noise_variance": ("last",),
    "execution_price_dispersion": ("last", "volume", "amount", "num_trades"),
    "book_depth_curve_cliff": ("last", *_book_columns()),
    "liquidity_supply_depletion": ("last", *_book_columns()),
    "bond_limit_pressure": ("last", "high_limited", "low_limited"),
    "iopv_basis_convergence": ("last", "iopv", "volume"),
    "quote_trade_stickiness": (
        "last",
        "ask_price1",
        "bid_price1",
        "ask_volume1",
        "bid_volume1",
    ),
    "flow_return_lead_lag": ("last", "volume", "amount", "num_trades"),
}


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve a unique score date without trusting rolling-panel labels alone."""

    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        date = pd.Timestamp(raw_day)
        if pd.isna(date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return date.normalize()
    if panel.empty:
        return None
    labels = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(labels[labels.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(
            f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel"
        )
    return pd.Timestamp(unique[0]).normalize()


def _score_day_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Return only physical score-day snapshots observed no later than 14:30.

    ``dt`` selects the requested score partition.  ``trade_time`` then proves
    the observation belongs to the same physical date and predates the cutoff.
    We intentionally discard, rather than impute over, invalid/late rows.
    """

    cache_key = f"{KERNEL_NAME}:{EXPANSION_VERSION}:score_day_frame"
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
        keep = (
            labels.notna()
            & (labels.dt.normalize() == target)
            & timestamps.notna()
            & (timestamps.dt.normalize() == target)
            & (timestamps.dt.time <= _CUTOFF)
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
        joined = ", ".join(missing)
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {joined}")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _all_finite_positive(values: np.ndarray, *, min_count: int = 2) -> np.ndarray | None:
    if values.size < min_count:
        return None
    if not (np.isfinite(values).all() and (values > 0.0).all()):
        return None
    return values.astype("float64", copy=False)


def _numeric(g: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")


def _prices(g: pd.DataFrame) -> np.ndarray | None:
    return _all_finite_positive(_numeric(g, "last"))


def _returns(prices: np.ndarray | None) -> np.ndarray | None:
    if prices is None or len(prices) < 3:
        return None
    out = prices[1:] / prices[:-1] - 1.0
    if not np.isfinite(out).all():
        return None
    return out


def _counter_increments(g: pd.DataFrame, column: str) -> np.ndarray | None:
    """Use increments of a cumulative field; a reset remains explicitly missing."""

    values = _numeric(g, column)
    if values.size < 3 or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all() or (increments < 0.0).any():
        return None
    return increments


def _corr(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    a = left[valid]
    b = right[valid]
    if float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None or len(values) != len(weights):
        return float("nan")
    if not (np.isfinite(values).all() and np.isfinite(weights).all() and (weights >= 0.0).all()):
        return float("nan")
    total = float(weights.sum())
    if total <= _EPS:
        return float("nan")
    return float(np.dot(values, weights) / total)


def _weighted_std(values: np.ndarray, weights: np.ndarray | None) -> float:
    mean = _weighted_mean(values, weights)
    if not np.isfinite(mean) or weights is None:
        return float("nan")
    total = float(weights.sum())
    if total <= _EPS:
        return float("nan")
    variance = float(np.dot(weights, (values - mean) ** 2) / total)
    return float(np.sqrt(variance)) if variance >= 0.0 else float("nan")


def _variance_ratio(returns: np.ndarray | None, horizon: int) -> float:
    if returns is None or horizon <= 1:
        return float("nan")
    count = (len(returns) // horizon) * horizon
    if count < horizon * 3:
        return float("nan")
    fine = returns[:count]
    fine_var = float(np.var(fine, ddof=1))
    if fine_var <= _EPS:
        return float("nan")
    coarse = fine.reshape(-1, horizon).sum(axis=1)
    if len(coarse) < 3:
        return float("nan")
    return _safe_div(float(np.var(coarse, ddof=1)), float(horizon) * fine_var)


def _book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask_price = np.column_stack([_numeric(g, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(g, f"bid_price{level}") for level in range(1, 6)])
    ask_volume = np.column_stack([_numeric(g, f"ask_volume{level}") for level in range(1, 6)])
    bid_volume = np.column_stack([_numeric(g, f"bid_volume{level}") for level in range(1, 6)])
    if not (
        np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_volume).all()
        and np.isfinite(bid_volume).all()
        and (ask_price > 0.0).all()
        and (bid_price > 0.0).all()
        and (ask_volume >= 0.0).all()
        and (bid_volume >= 0.0).all()
        and (ask_price[:, 0] >= bid_price[:, 0]).all()
    ):
        return None
    return ask_price, bid_price, ask_volume, bid_volume


def _best_book_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    ask = _all_finite_positive(_numeric(g, "ask_price1"))
    bid = _all_finite_positive(_numeric(g, "bid_price1"))
    ask_vol = _numeric(g, "ask_volume1")
    bid_vol = _numeric(g, "bid_volume1")
    if (
        ask is None
        or bid is None
        or ask_vol.size != len(ask)
        or bid_vol.size != len(ask)
        or not (np.isfinite(ask_vol).all() and np.isfinite(bid_vol).all())
        or (ask_vol < 0.0).any()
        or (bid_vol < 0.0).any()
        or (ask < bid).any()
    ):
        return None
    return ask, bid, ask_vol, bid_vol


def _longest_stable_run(values: np.ndarray, tolerance: float) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    longest = 1
    current = 1
    for previous, current_value in zip(values[:-1], values[1:], strict=True):
        if abs(current_value - previous) <= tolerance:
            current += 1
        else:
            current = 1
        longest = max(longest, current)
    return _safe_div(float(longest), float(len(values)))


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _open_gap_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_OPEN_GAP_SIGNALS)
    prices = _prices(g)
    if prices is None:
        return out
    pre_close = _numeric(g, "pre_close")
    if pre_close.size == 0 or not np.isfinite(pre_close[0]) or pre_close[0] <= 0.0:
        return out
    gap = _safe_div(float(prices[0] - pre_close[0]), float(pre_close[0]))
    gap_value = float(prices[0] - pre_close[0])
    if not np.isfinite(gap) or abs(gap_value) <= _EPS:
        return out
    direction = float(np.sign(gap_value))
    path_abs = float(np.abs(np.diff(prices)).sum())
    reversal_extent = (
        float(prices[0] - np.min(prices))
        if direction > 0.0
        else float(np.max(prices) - prices[0])
    )
    extension = (
        float(np.max(prices) - prices[0])
        if direction > 0.0
        else float(prices[0] - np.min(prices))
    )
    volume = _counter_increments(g, "volume")
    returns = _returns(prices)
    weighted_reversal = float("nan")
    if volume is not None and returns is not None:
        weighted_reversal = -direction * _weighted_mean(returns, volume)
    out.update(
        {
            "exp_gap_initial": gap,
            "exp_gap_absorption": _safe_div(float(-(prices[-1] - prices[0])), gap_value),
            "exp_gap_residual": _safe_div(float(prices[-1] - pre_close[0]), gap_value),
            "exp_gap_extreme_extension": _safe_div(extension, abs(gap_value)),
            "exp_gap_reversal_efficiency": _safe_div(reversal_extent, path_abs),
            "exp_gap_volume_weighted_reversal": weighted_reversal,
        }
    )
    return out


def _session_returns(g: pd.DataFrame) -> tuple[float, float, float] | None:
    clocks = pd.to_datetime(g["trade_time"], errors="coerce")
    prices = _prices(g)
    if prices is None or clocks.isna().any():
        return None
    early_mask = (clocks.dt.time < dt_time(10, 30)).to_numpy(dtype=bool)
    middle_mask = (
        (clocks.dt.time >= dt_time(10, 30)) & (clocks.dt.time < dt_time(13, 30))
    ).to_numpy(dtype=bool)
    late_mask = (
        (clocks.dt.time >= dt_time(13, 30)) & (clocks.dt.time <= _CUTOFF)
    ).to_numpy(dtype=bool)
    early = prices[early_mask]
    middle = prices[middle_mask]
    late = prices[late_mask]
    if min(len(early), len(middle), len(late)) < 2:
        return None
    returns = tuple(_safe_div(float(block[-1] - block[0]), float(block[0])) for block in (early, middle, late))
    return returns if all(np.isfinite(value) for value in returns) else None


def _rotation_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_ROTATION_SIGNALS)
    prices = _prices(g)
    returns = _returns(prices)
    if prices is None or returns is None:
        return out
    session = _session_returns(g)
    flips = float("nan")
    nonzero_sign = np.sign(returns[np.abs(returns) > _EPS])
    if len(nonzero_sign) >= 2:
        flips = float(np.mean(nonzero_sign[1:] != nonzero_sign[:-1]))
    tail_start = max(1, int(len(returns) * 2 / 3))
    total_abs = float(np.abs(returns).sum())
    time_of_extreme = _safe_div(
        float(np.argmax(np.abs(prices / prices[0] - 1.0))),
        float(max(1, len(prices) - 1)),
    )
    out.update(
        {
            "exp_rotation_time_of_extreme": time_of_extreme,
            "exp_rotation_return_sign_flip_rate": flips,
            "exp_rotation_late_abs_move_share": _safe_div(float(np.abs(returns[tail_start:]).sum()), total_abs),
        }
    )
    if session is not None:
        early, middle, late = session
        out.update(
            {
                "exp_rotation_early_late_return_gap": late - early,
                "exp_rotation_mid_late_return_turn": late - middle,
                "exp_rotation_segment_return_dispersion": float(np.std([early, middle, late], ddof=0)),
            }
        )
    return out


def _noise_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_NOISE_SIGNALS)
    returns = _returns(_prices(g))
    if returns is None or len(returns) < 6:
        return out
    vr2 = _variance_ratio(returns, 2)
    vr5 = _variance_ratio(returns, 5)
    coarse_fine = _safe_div(float(np.sqrt(vr5)), 1.0) if np.isfinite(vr5) and vr5 >= 0.0 else float("nan")
    abs_autocorr = _corr(np.abs(returns[:-1]), np.abs(returns[1:]))
    realized_variance = float(np.square(returns).sum())
    bipower = float(np.pi / 2.0 * np.abs(returns[1:] * returns[:-1]).sum())
    jump_share = _safe_div(realized_variance - bipower, realized_variance)
    mean_abs = float(np.abs(returns).mean())
    median_mean = _safe_div(float(np.median(np.abs(returns))), mean_abs)
    out.update(
        {
            "exp_noise_variance_ratio_2": vr2,
            "exp_noise_variance_ratio_5": vr5,
            "exp_noise_coarse_fine_vol_ratio": coarse_fine,
            "exp_noise_return_abs_autocorr": abs_autocorr,
            "exp_noise_bipower_jump_share": jump_share,
            "exp_noise_median_mean_abs_return_ratio": median_mean,
        }
    )
    return out


def _execution_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_EXECUTION_SIGNALS)
    prices = _prices(g)
    if prices is None:
        return out
    volume = _counter_increments(g, "volume")
    amount = _counter_increments(g, "amount")
    deals = _counter_increments(g, "num_trades")
    interval_prices = prices[1:]
    amount_vwap = _weighted_mean(interval_prices, amount)
    trade_vwap = _weighted_mean(interval_prices, deals)
    tail_start = max(1, int(len(interval_prices) * 2 / 3))
    tail_vwap = _weighted_mean(interval_prices[tail_start:], amount[tail_start:] if amount is not None else None)
    returns = _returns(prices)
    trade_size_corr = float("nan")
    if volume is not None and amount is not None and returns is not None and (volume > _EPS).all():
        trade_size_corr = _corr(np.log(amount / volume), returns)
    concentration_impact = float("nan")
    if amount is not None and returns is not None:
        total = float(amount.sum())
        if total > _EPS:
            concentration_impact = float(np.dot(amount / total, np.abs(returns)))
    out.update(
        {
            "exp_exec_amount_price_dispersion": _safe_div(_weighted_std(interval_prices, amount), amount_vwap),
            "exp_exec_trade_price_dispersion": _safe_div(_weighted_std(interval_prices, deals), trade_vwap),
            "exp_exec_tail_vwap_shift": _safe_div(tail_vwap - amount_vwap, amount_vwap),
            "exp_exec_vwap_last_residual": _safe_div(float(prices[-1] - amount_vwap), amount_vwap),
            "exp_exec_trade_size_return_corr": trade_size_corr,
            "exp_exec_amount_concentration_impact": concentration_impact,
        }
    )
    return out


def _curve_ratio(depth: np.ndarray) -> float:
    if len(depth) != 5 or not np.isfinite(depth).all():
        return float("nan")
    return _safe_div(float(depth[0] + depth[1]), float(depth[3] + depth[4]))


def _book_curve_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_BOOK_CURVE_SIGNALS)
    book = _book_arrays(g)
    if book is None:
        return out
    ask_price, bid_price, ask_volume, bid_volume = book
    bid_ratio = _curve_ratio(bid_volume[-1])
    ask_ratio = _curve_ratio(ask_volume[-1])
    bid_convexity = float(np.log(bid_ratio)) if np.isfinite(bid_ratio) and bid_ratio > 0.0 else float("nan")
    ask_convexity = float(np.log(ask_ratio)) if np.isfinite(ask_ratio) and ask_ratio > 0.0 else float("nan")
    bid_cliff = _safe_div(float(bid_volume[-1, 0]), float(np.mean(bid_volume[-1, 1:])))
    ask_cliff = _safe_div(float(ask_volume[-1, 0]), float(np.mean(ask_volume[-1, 1:])))
    mid = (ask_price[:, 0] + bid_price[:, 0]) / 2.0
    spread = (ask_price[:, 0] - bid_price[:, 0]) / mid
    total_depth = ask_volume.sum(axis=1) + bid_volume.sum(axis=1)
    elasticity = _corr(spread, np.log(total_depth)) if (total_depth > 0.0).all() else float("nan")
    out.update(
        {
            "exp_book_bid_curve_convexity": bid_convexity,
            "exp_book_ask_curve_convexity": ask_convexity,
            "exp_book_curve_asymmetry": bid_convexity - ask_convexity
            if np.isfinite(bid_convexity) and np.isfinite(ask_convexity)
            else float("nan"),
            "exp_book_bid_top_cliff": bid_cliff,
            "exp_book_ask_top_cliff": ask_cliff,
            "exp_book_depth_spread_elasticity": elasticity,
        }
    )
    return out


def _supply_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_SUPPLY_SIGNALS)
    book = _book_arrays(g)
    prices = _prices(g)
    if book is None or prices is None:
        return out
    _, _, ask_volume, bid_volume = book
    bid_total = bid_volume.sum(axis=1)
    ask_total = ask_volume.sum(axis=1)
    bid_delta = np.diff(bid_total)
    ask_delta = np.diff(ask_total)
    bid_abs_change = float(np.abs(bid_delta).sum())
    ask_abs_change = float(np.abs(ask_delta).sum())
    combined_depth = bid_total + ask_total
    bid_imbalance = (
        (bid_total - ask_total) / combined_depth
        if (combined_depth > _EPS).all()
        else np.full_like(combined_depth, np.nan)
    )
    returns = _returns(prices)
    refill = bid_delta - ask_delta
    out.update(
        {
            "exp_supply_bid_replenishment": _safe_div(float(np.maximum(bid_delta, 0.0).sum()), float(bid_total[:-1].sum())),
            "exp_supply_ask_replenishment": _safe_div(float(np.maximum(ask_delta, 0.0).sum()), float(ask_total[:-1].sum())),
            "exp_supply_bid_depletion_share": _safe_div(float(np.maximum(-bid_delta, 0.0).sum()), bid_abs_change),
            "exp_supply_ask_depletion_share": _safe_div(float(np.maximum(-ask_delta, 0.0).sum()), ask_abs_change),
            "exp_supply_imbalance_trend": float(bid_imbalance[-1] - bid_imbalance[0])
            if np.isfinite(bid_imbalance[[0, -1]]).all()
            else float("nan"),
            "exp_supply_refill_price_coupling": _corr(refill, returns),
        }
    )
    return out


def _limit_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LIMIT_SIGNALS)
    prices = _prices(g)
    high = _numeric(g, "high_limited")
    low = _numeric(g, "low_limited")
    if (
        prices is None
        or high.size != len(prices)
        or low.size != len(prices)
        or not (np.isfinite(high).all() and np.isfinite(low).all())
        or (high <= 0.0).any()
        or (low <= 0.0).any()
        or (high <= low).any()
    ):
        # Limit zeros are missing/invalid data, never a valid "at-limit" state.
        return out
    up = (high - prices) / prices
    down = (prices - low) / prices
    if not (np.isfinite(up).all() and np.isfinite(down).all()):
        return out
    pressure = (down - up) / (down + up)
    if not np.isfinite(pressure).all():
        return out
    nearest = np.minimum(np.abs(up), np.abs(down))
    full_return = _safe_div(float(prices[-1] - prices[0]), float(prices[0]))
    out.update(
        {
            "exp_limit_up_distance": float(up[-1]),
            "exp_limit_down_distance": float(down[-1]),
            "exp_limit_nearest_signed_pressure": float(pressure[-1]),
            "exp_limit_pressure_change": float(pressure[-1] - pressure[0]),
            "exp_limit_near_touch_share": float(np.mean(nearest <= 0.005)),
            "exp_limit_directional_path": full_return * float(pressure[-1])
            if np.isfinite(full_return)
            else float("nan"),
        }
    )
    return out


def _iopv_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_IOPV_SIGNALS)
    prices = _prices(g)
    iopv = _all_finite_positive(_numeric(g, "iopv"))
    if prices is None or iopv is None or len(iopv) != len(prices):
        return out
    basis = prices / iopv - 1.0
    returns = _returns(prices)
    volume = _counter_increments(g, "volume")
    out.update(
        {
            "exp_iopv_basis_last": float(basis[-1]),
            "exp_iopv_basis_mean": float(np.mean(basis)),
            "exp_iopv_basis_convergence": float(abs(basis[0]) - abs(basis[-1])),
            "exp_iopv_basis_volatility": float(np.std(basis, ddof=1)) if len(basis) >= 2 else float("nan"),
            "exp_iopv_basis_return_correlation": _corr(basis[1:], returns),
            "exp_iopv_basis_flow_correlation": _corr(basis[1:], volume),
        }
    )
    return out


def _stickiness_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_STICKINESS_SIGNALS)
    prices = _prices(g)
    best = _best_book_arrays(g)
    if prices is None or best is None:
        return out
    ask, bid, _, _ = best
    mid = (ask + bid) / 2.0
    spread = (ask - bid) / mid
    price_tol = max(_EPS, abs(float(prices[0])) * 1e-6)
    mid_tol = max(_EPS, abs(float(mid[0])) * 1e-6)
    spread_tol = max(_EPS, abs(float(spread[0])) * 1e-6)
    price_change_rate = float(np.mean(np.abs(np.diff(prices)) > price_tol))
    quote_change_rate = float(np.mean(np.abs(np.diff(mid)) > mid_tol))
    spread_staleness = 1.0 - float(np.mean(np.abs(np.diff(spread)) > spread_tol))
    out.update(
        {
            "exp_stick_price_change_rate": price_change_rate,
            "exp_stick_price_run_length": _longest_stable_run(prices, price_tol),
            "exp_stick_quote_update_rate": quote_change_rate,
            "exp_stick_quote_trade_gap": quote_change_rate - price_change_rate,
            "exp_stick_mid_trade_dislocation": _safe_div(float(prices[-1] - mid[-1]), float(mid[-1])),
            "exp_stick_spread_staleness": spread_staleness,
        }
    )
    return out


def _lead_lag_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LEAD_LAG_SIGNALS)
    prices = _prices(g)
    returns = _returns(prices)
    volume = _counter_increments(g, "volume")
    amount = _counter_increments(g, "amount")
    if prices is None or returns is None or volume is None or amount is None:
        return out
    log_volume = np.log1p(volume)
    log_amount = np.log1p(amount)
    flow_leads = _corr(log_volume[:-1], returns[1:])
    return_leads = _corr(returns[:-1], log_volume[1:])
    trade_size_leads = float("nan")
    if (volume > _EPS).all():
        trade_size_leads = _corr(np.log(amount[:-1] / volume[:-1]), returns[1:])
    amount_forward = _corr(log_amount[:-1], returns[1:])
    amount_backward = _corr(returns[:-1], log_amount[1:])
    flow_change = _corr(np.diff(log_volume), returns[1:])
    tail_start = max(0, int(len(returns) * 2 / 3))
    tail_response = _corr(log_volume[tail_start:-1], returns[tail_start + 1 :])
    out.update(
        {
            "exp_leadlag_flow_leads_return": flow_leads,
            "exp_leadlag_return_leads_flow": return_leads,
            "exp_leadlag_trade_size_leads_return": trade_size_leads,
            "exp_leadlag_amount_return_asymmetry": amount_forward - amount_backward
            if np.isfinite(amount_forward) and np.isfinite(amount_backward)
            else float("nan"),
            "exp_leadlag_flow_feedback": flow_change,
            "exp_leadlag_tail_flow_response": tail_response,
        }
    )
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "open_gap_absorption": _open_gap_metrics,
    "clock_time_rotation": _rotation_metrics,
    "multiscale_noise_variance": _noise_metrics,
    "execution_price_dispersion": _execution_metrics,
    "book_depth_curve_cliff": _book_curve_metrics,
    "liquidity_supply_depletion": _supply_metrics,
    "bond_limit_pressure": _limit_metrics,
    "iopv_basis_convergence": _iopv_metrics,
    "quote_trade_stickiness": _stickiness_metrics,
    "flow_return_lead_lag": _lead_lag_metrics,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache one whole family so six signal specs share one panel scan."""

    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{EXPANSION_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    signals = _FAMILY_SIGNALS[family]
    _require_columns(frame, ("trade_time", *_FAMILY_REQUIRED_COLUMNS[family]))
    if frame.empty:
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
class FactorMiningIntradayExpansionV1(Factor):
    """Research-only, T-day ``ctx.panel`` candidate family kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
