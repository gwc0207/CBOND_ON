"""Research-only factor-family catalogue for the 2026-08 IC mining run.

The module deliberately exposes a small number of registered kernels and many
explicit ``signal`` instances.  The batch builder shares one
``FactorComputeContext.cache`` per score day, so a catalogue can compute its
common panel/daily state once and then materialise each auditable signal
without re-reading any data.

All factor code consumes only FactorComputeContext.  In particular, it never
reads a file, database, pool, label, score, or trading result.  Historical
daily inputs are explicitly restricted to dates strictly before the signal
date because the daily-context loader intentionally supplies the score-day
file too.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


EPS = 1e-12
# v2 makes the clean-direct panel contract explicit: an index ``dt`` is a
# score-day label, not proof that each retained snapshot was observed that day.
# The on-demand panel builder can include prior-day rows to satisfy its generic
# rolling-window length, so research-only T1430 signals must additionally gate
# on the physical ``trade_time`` date below.
CATALOG_VERSION = "20260802_v3"


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    kernel: str,
    family: str,
    signals: Iterable[str],
    hypothesis: str,
) -> list[CatalogEntry]:
    return [CatalogEntry(family, signal, kernel, hypothesis) for signal in signals]


_INTRADAY_ENTRIES = (
    _entries(
        "factor_mining_intraday_catalog_v1",
        "intraday_price_path_risk",
        (
            "intra_full_return",
            "intra_path_efficiency",
            "intra_signed_efficiency",
            "intra_path_roughness",
            "intra_return_autocorr",
            "intra_return_skew",
            "intra_return_kurtosis",
            "intra_up_down_semivol_ratio",
            "intra_drawdown",
            "intra_drawup",
            "intra_recovery_from_low",
            "intra_close_position",
            "intra_tail_return_5m",
            "intra_tail_return_15m",
            "intra_tail_vs_full_return",
            "intra_tail_vol_ratio",
            "intra_tail_path_efficiency",
            "intra_tail_jump_share",
            "intra_time_under_vwap",
            "intra_tail_reversal_ratio",
            "intra_price_entropy",
        ),
        "Price-path shape, tail risk, and path quality are distinct from a simple endpoint return.",
    )
    + _entries(
        "factor_mining_intraday_catalog_v1",
        "intraday_trade_price_impact",
        (
            "intra_volume_accel_5m",
            "intra_volume_accel_15m",
            "intra_amount_accel_5m",
            "intra_amount_accel_15m",
            "intra_trade_accel_5m",
            "intra_trade_accel_15m",
            "intra_avg_trade_size_tail",
            "intra_volume_entropy",
            "intra_volume_hhi",
            "intra_volume_gini",
            "intra_flow_time_center",
            "intra_flow_interarrival_cv",
            "intra_burst_cluster_count",
            "intra_volume_price_corr",
            "intra_volume_return_corr",
            "intra_amount_return_corr",
            "intra_trade_size_return_corr",
            "intra_impact_per_amount",
            "intra_impact_per_trade",
            "intra_impact_asymmetry",
            "intra_flow_price_divergence",
        ),
        "Incremental flow and price-impact mechanisms use cumulative-field differences, never levels.",
    )
    + _entries(
        "factor_mining_intraday_catalog_v1",
        "intraday_quote_resilience",
        (
            "book_spread_last",
            "book_spread_mean",
            "book_spread_std",
            "book_spread_tail_change",
            "book_depth_total",
            "book_depth_concentration",
            "book_depth_slope",
            "book_imbalance_l1_last",
            "book_imbalance_l5_last",
            "book_imbalance_l1_mean",
            "book_imbalance_l5_mean",
            "book_imbalance_l1_change",
            "book_imbalance_l5_change",
            "book_imbalance_dispersion",
            "book_microprice_bias_last",
            "book_microprice_bias_mean",
            "book_microprice_bias_change",
            "book_imbalance_return_corr",
            "book_microprice_return_corr",
            "book_spread_return_corr",
            "book_depth_return_corr",
            "book_bid_depth_recovery",
            "book_ask_depth_recovery",
            "book_liquidity_recovery",
            "book_pressure_impact",
            "book_pressure_tail_reversal",
            "book_quote_staleness",
            "book_quote_dislocation",
        ),
        "Order-book pressure, replenishment, and adverse-selection dynamics are different from static spread aliases.",
    )
)


_DAILY_ENTRIES = (
    _entries(
        "factor_mining_daily_catalog_v1",
        "daily_price_path_geometry",
        (
            "dpx_body_range",
            "dpx_upper_wick",
            "dpx_lower_wick",
            "dpx_close_location",
            "dpx_path_efficiency",
            "dpx_range_expansion",
            "dpx_breakout_distance",
        ),
        "Prior daily candle geometry measures historical price-path state rather than current-day information.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_return_distribution",
        (
            "dret_tstat",
            "dret_downside_semivar_ratio",
            "dret_upside_semivar_surprise",
            "dret_skew",
            "dret_tail_loss_frequency",
            "dret_drawup_drawdown_asym",
            "dret_sign_run_reversal",
        ),
        "Historical return distribution and sequencing are separate from a rolling mean or Sharpe ratio.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_gap_reaction",
        (
            "gap_open_z",
            "gap_close_followthrough",
            "gap_reversal",
            "gap_range_amplification",
            "gap_persistence",
            "gap_extreme_recovery",
            "gap_vs_prior_vol",
        ),
        "Prior opening gap reactions capture a distinct close-to-open and session-follow-through mechanism.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_price_impact",
        (
            "impact_amihud_absret",
            "impact_signed_sqrt_dollar",
            "impact_range_per_amount",
            "impact_body_per_amount",
            "impact_close_dislocation_per_deal",
            "impact_illiquidity_shock",
            "impact_up_down_asymmetry",
        ),
        "Historical price impact is based on daily liquidity-normalized moves, not a renamed volume factor.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_liquidity_state",
        (
            "liq_amount_shock",
            "liq_volume_shock",
            "liq_deal_count_shock",
            "liq_avg_trade_size_shock",
            "liq_notional_per_volume",
            "liq_volume_persistence",
            "liq_drought_streak",
        ),
        "Prior liquidity shocks and droughts are modeled as historical state, not same-day raw turnover.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_twap_curve_shape",
        (
            "twap_morning_slope",
            "twap_lunch_reopen_gap",
            "twap_afternoon_slope",
            "twap_curve_curvature",
            "twap_morning_afternoon_disagreement",
            "twap_segment_volatility",
            "twap_path_efficiency",
        ),
        "The historical intraday TWAP curve contains timing information beyond an overnight return mean.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "daily_twap_late_execution",
        (
            "late_preclose_ramp",
            "late_execution_premium",
            "late_vs_midday_reversal",
            "late_extension",
            "late_return_concentration",
            "late_price_impact_z",
            "late_path_sign_stability",
        ),
        "Historical late-session curve and execution premium are distinct from current T1430 prices.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "historical_overnight_conditionals",
        (
            "overnight_on_prior_intraday_beta",
            "overnight_reversal_after_tail",
            "overnight_win_rate",
            "overnight_downside_semisharpe",
            "overnight_tail_loss_share",
            "overnight_autocorr",
            "overnight_vs_prior_range_sensitivity",
        ),
        "Overnight conditionals use only completed D+1 mornings strictly before T.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_premium_level",
        (
            "base_premium_z",
            "base_puredebt_premium_z",
            "base_premium_spread",
            "base_premium_percentile_break",
            "base_premium_crosssection_residual",
            "base_redemption_premium_z",
            "base_premium_dispersion",
        ),
        "Prior-day premium level and distribution are valuation state, not a current price-derived factor.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_premium_dynamics",
        (
            "base_bond_premium_delta1",
            "base_puredebt_premium_delta1",
            "base_premium_acceleration",
            "base_premium_mean_revert_distance",
            "base_premium_breakout",
            "base_premium_vs_conv_value_change",
            "base_premium_vs_stock_vol_change",
        ),
        "Premium state transitions use the T-1 and earlier base history only.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_credit_duration_state",
        (
            "base_yield_spread",
            "base_ytm_change",
            "base_duration_adjusted_yield",
            "base_convexity_duration_ratio",
            "base_time_to_mat_residual",
            "base_stock_volatility_z",
            "base_duration_stockvol_interaction",
        ),
        "Prior credit, duration, and option-risk state is structurally different from intraday microstructure.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_call_redemption_state",
        (
            "base_call_price_distance",
            "base_trigger_progress_ratio",
            "base_trigger_days_remaining",
            "base_in_trigger_process",
            "base_trigger_progress_delta",
            "base_trigger_revision_gap",
            "base_call_state_transition",
        ),
        "Call and redemption event state is a pre-existing structural attribute, never inferred from labels.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_conversion_optionality",
        (
            "base_conversion_moneyness",
            "base_conversion_delta_proxy",
            "base_conv_price_distance",
            "base_stockvol_per_moneyness",
            "base_moneyness_time_decay",
            "base_conversion_value_change",
            "base_optionality_premium_residual",
        ),
        "Conversion-option state joins only prior daily-base observations.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_relative_liquidity",
        (
            "base_turnover_z",
            "base_float_adjusted_amount",
            "base_bond_stock_amount_ratio",
            "base_bond_stock_trade_size_ratio",
            "base_liquidity_mismatch_z",
            "base_turnover_change",
            "base_float_liquidity_drought",
        ),
        "Bond-versus-stock historical liquidity mismatch is separate from a same-day flow ratio.",
    )
    + _entries(
        "factor_mining_daily_catalog_v1",
        "base_balance_sheet_defensive",
        (
            "base_bond_floor_distance",
            "base_debt_premium_floor_gap",
            "base_yield_to_floor_ratio",
            "base_duration_adjusted_floor",
            "base_floor_change",
            "base_floor_volatility",
            "base_floor_conversion_residual",
        ),
        "Pure-debt floor and defensive-state signals are distinct from equity-option premium signals.",
    )
)


_CROSS_ENTRIES = (
    _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_intraday_lead_lag",
        (
            "cross_stock_early_bond_late_response",
            "cross_bond_early_stock_late_response",
            "cross_lagged_return_corr",
            "cross_stock_impulse_bond_decay",
            "cross_bond_impulse_stock_decay",
            "cross_response_asym_up",
            "cross_response_asym_down",
        ),
        "T-1 mapped stock and bond path timing, not a static endpoint return gap.",
    )
    + _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_bond_beta_residual",
        (
            "cross_intraday_beta_resid",
            "cross_tail_beta_resid",
            "cross_rolling_cov_ratio",
            "cross_residual_vol_ratio",
            "cross_residual_drawdown",
            "cross_residual_efficiency_gap",
            "cross_bond_stock_return_gap",
        ),
        "Dynamic bond-stock beta residuals are separated from conversion-value scaling.",
    )
    + _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_bond_path_asynchrony",
        (
            "cross_time_to_extreme_gap",
            "cross_time_of_high_gap",
            "cross_time_of_low_gap",
            "cross_turning_point_mismatch",
            "cross_path_entropy_gap",
            "cross_drawdown_timing_gap",
            "cross_recovery_timing_gap",
        ),
        "Cross-asset path timing mechanisms do not reduce to contemporaneous price correlation.",
    )
    + _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_bond_microstructure_divergence",
        (
            "cross_mid_gap",
            "cross_relative_spread_gap",
            "cross_depth_imbalance_gap",
            "cross_microprice_bias_gap",
            "cross_quote_update_intensity_gap",
            "cross_book_slope_gap",
            "cross_spread_resilience_gap",
        ),
        "Mapped order-book divergence measures two-asset microstructure, not one-asset aliases.",
    )
    + _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_bond_liquidity_transmission",
        (
            "cross_stock_flow_to_bond_flow_beta",
            "cross_bond_flow_to_stock_flow_beta",
            "cross_flow_lead_lag_corr",
            "cross_flow_shock_pass_through",
            "cross_trade_size_transmission",
            "cross_liquidity_drought_transmission",
            "cross_flow_return_elasticity_gap",
        ),
        "Flow transmission is calculated from per-interval differences on both assets.",
    )
    + _entries(
        "factor_mining_cross_asset_catalog_v1",
        "stock_bond_limit_stress",
        (
            "cross_stock_up_limit_response",
            "cross_stock_down_limit_response",
            "cross_limit_distance_gap",
            "cross_limit_proximity_change_gap",
            "cross_one_side_limit_indicator",
            "cross_limit_proximity_vol_scaled",
            "cross_limit_stress_premium",
        ),
        "Current stock limit stress is conditioned only on T-1 structural premium state.",
    )
)


_HYBRID_ENTRIES = (
    _entries(
        "factor_mining_hybrid_catalog_v1",
        "hybrid_daily_intraday_surprise",
        (
            "hybrid_current_return_vs_hist_intraday",
            "hybrid_current_range_vs_hist_range",
            "hybrid_current_flow_vs_hist_amount",
            "hybrid_current_spread_vs_hist_illiquidity",
            "hybrid_current_tail_vs_hist_tail",
            "hybrid_current_position_vs_hist_premium",
            "hybrid_current_stock_response_vs_hist_beta",
        ),
        "T1430 surprise is benchmarked only against completed historical state of the same instrument.",
    )
    + _entries(
        "factor_mining_hybrid_catalog_v1",
        "hybrid_structural_conditional_flow",
        (
            "hybrid_flow_x_premium_level",
            "hybrid_flow_x_moneyness",
            "hybrid_flow_x_bond_floor_distance",
            "hybrid_flow_x_duration",
            "hybrid_quote_pressure_x_stockvol",
            "hybrid_return_x_redemption_state",
            "hybrid_spread_resilience_x_remain_size",
        ),
        "T1430 flow is gated by prior structural state; it is not a label-trained interaction.",
    )
    + _entries(
        "factor_mining_hybrid_catalog_v1",
        "hybrid_execution_curve",
        (
            "hybrid_pre1430_move_vs_hist_late_beta",
            "hybrid_current_flow_vs_hist_overnight_response",
            "hybrid_quote_pressure_vs_late_premium",
            "hybrid_stock_lead_vs_prior_premium",
            "hybrid_current_range_vs_hist_twap_curve",
            "hybrid_current_momentum_vs_call_state",
            "hybrid_current_illiquidity_vs_prior_turnover",
        ),
        "Current T1430 state is connected to completed historical execution curves only.",
    )
)


_CATALOG = tuple(_INTRADAY_ENTRIES + _DAILY_ENTRIES + _CROSS_ENTRIES + _HYBRID_ENTRIES)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research catalogue in family-first order."""

    return _CATALOG


def _bare_code(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.upper().str.split(".", n=1).str[0]


_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def _canonical_market_code(
    values: pd.Series,
    exchanges: pd.Series | None = None,
) -> pd.Series:
    """Preserve the exchange suffix that disambiguates DataHub instruments."""

    code = values.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    if exchanges is None:
        return code
    exchange = exchanges.astype(str).str.strip().str.upper().map(
        lambda value: _EXCHANGE_ALIASES.get(value, value)
    )
    has_suffix = code.str.contains(r"\.(?:SH|SZ|BJ)$", regex=True, na=False)
    attachable = (~has_suffix) & exchange.isin(_MARKET_EXCHANGES)
    out = code.copy()
    out.loc[attachable] = out.loc[attachable] + "." + exchange.loc[attachable]
    return out


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve the score date and fail closed on an unlabelled multi-day panel."""

    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        day = pd.Timestamp(raw)
        if pd.isna(day):
            raise ValueError("factor mining catalogue has invalid panel __build_day__")
        return day.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(
            "factor mining catalogue requires panel __build_day__ when the input contains multiple dates"
        )
    return pd.Timestamp(unique[0]).normalize()


def _signal_day_panel(
    panel: pd.DataFrame,
    *,
    signal_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only the score-day T1430 snapshots from a rolling clean-direct panel."""

    panel = ensure_panel_index(panel)
    if panel.empty:
        return panel
    target = signal_date if signal_date is not None else _signal_date_from_panel(panel)
    if target is None:
        return panel.iloc[0:0]
    target = pd.Timestamp(target).normalize()
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    keep = dates == target
    if not bool(np.any(keep)):
        raise ValueError(f"factor mining catalogue has no panel rows for signal day {target.date().isoformat()}")
    out = panel.loc[keep].copy(deep=False)
    # The clean-direct builder gives all retained rolling rows the target ``dt``
    # label.  Retain only snapshots whose physical exchange timestamp belongs
    # to that score day before using the panel even as an output-code index.
    # This is deliberately fail-closed: every supported T1430 panel contract
    # provides ``trade_time``.
    if "trade_time" not in out.columns:
        raise KeyError("factor mining catalogue panel missing required trade_time")
    trade_times = pd.to_datetime(out["trade_time"], errors="coerce")
    out = out.loc[trade_times.notna() & (trade_times.dt.normalize() == target)].copy(deep=False)
    out["trade_time"] = trade_times.loc[out.index]
    out.attrs = dict(getattr(panel, "attrs", {}) or {})
    out.attrs["__build_day__"] = target.date().isoformat()
    return out


def _strict_score_day_snapshots(
    frame: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    """Keep physically score-day snapshots at or before the T1430 cutoff.

    ``clean_direct`` may deliberately assemble a rolling path and label each
    retained row with the target ``dt``.  The label is useful for the generic
    factor engine but cannot establish point-in-time membership for this
    catalogue.  These factors describe current intraday state, so prior-day
    snapshots must never enter their path, flow, or stock/bond alignment.
    """

    target = pd.Timestamp(signal_date).normalize()
    times = pd.to_datetime(frame["trade_time"], errors="coerce")
    keep = (
        times.notna()
        & (times.dt.normalize() == target)
        & (times.dt.time <= dt_time(14, 30))
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = times.loc[keep]
    return out


def _empty_output(panel: pd.DataFrame, *, name: str) -> pd.Series:
    panel = _signal_day_panel(panel)
    keys = panel.index.droplevel("seq").unique()
    keys = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"])
    out = pd.Series(np.nan, index=keys, dtype="float64", name=name)
    return out


def _cached(ctx: FactorComputeContext, key: str, build: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    with ctx.cache_lock:
        existing = ctx.cache.get(key)
        if isinstance(existing, pd.DataFrame):
            return existing
        value = build()
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"factor mining cache builder {key} returned {type(value).__name__}")
        ctx.cache[key] = value
        return value


def _cached_output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    """Resolve the physical score-day output universe once per build day.

    A factor family shares one :class:`FactorComputeContext.cache` across all
    238 specs.  Re-running the physical-date filter for each concrete signal
    scans the same multi-million-row rolling panel repeatedly; cache a tiny
    frame solely to reuse its validated ``(dt, code)`` index without changing
    any factor values or widening the output universe.
    """

    def _build() -> pd.DataFrame:
        panel = _signal_day_panel(ctx.panel)
        keys = panel.index.droplevel("seq").unique()
        index = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"])
        return pd.DataFrame(index=index)

    return _cached(ctx, f"factor_mining_output_index:{CATALOG_VERSION}", _build).index


def _empty_output_for_context(ctx: FactorComputeContext, *, name: str) -> pd.Series:
    return pd.Series(np.nan, index=_cached_output_index(ctx), dtype="float64", name=name)


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    pair = np.column_stack([left, right])
    pair = pair[np.isfinite(pair).all(axis=1)]
    if len(pair) < 3:
        return float("nan")
    if np.nanstd(pair[:, 0]) <= EPS or np.nanstd(pair[:, 1]) <= EPS:
        return float("nan")
    return float(np.corrcoef(pair[:, 0], pair[:, 1])[0, 1])


def _safe_autocorr(values: np.ndarray) -> float:
    if len(values) < 4:
        return float("nan")
    return _safe_corr(values[1:], values[:-1])


def _safe_skew(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 4:
        return float("nan")
    std = float(np.std(values, ddof=0))
    if std <= EPS:
        return float("nan")
    return float(np.mean(((values - np.mean(values)) / std) ** 3))


def _safe_kurtosis(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 5:
        return float("nan")
    std = float(np.std(values, ddof=0))
    if std <= EPS:
        return float("nan")
    return float(np.mean(((values - np.mean(values)) / std) ** 4) - 3.0)


def _entropy(weights: np.ndarray) -> float:
    weights = _finite(weights)
    weights = weights[weights > 0.0]
    total = float(weights.sum())
    if len(weights) < 2 or total <= EPS:
        return float("nan")
    probs = weights / total
    return float(-(probs * np.log(probs)).sum() / np.log(len(probs)))


def _gini(weights: np.ndarray) -> float:
    weights = _finite(weights)
    weights = weights[weights >= 0.0]
    if len(weights) < 2 or float(weights.sum()) <= EPS:
        return float("nan")
    ordered = np.sort(weights)
    n = len(ordered)
    return float((2.0 * np.dot(np.arange(1, n + 1), ordered) / (n * ordered.sum())) - (n + 1.0) / n)


def _time_tail(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    end = frame["trade_time"].max()
    if pd.isna(end):
        return frame.iloc[0:0]
    return frame.loc[frame["trade_time"] >= end - pd.Timedelta(minutes=int(minutes))]


def _incremental(cumulative: np.ndarray) -> np.ndarray | None:
    """Return interval increments or explicit missing when a counter resets."""

    if len(cumulative) < 3 or not np.isfinite(cumulative).all():
        return None
    diff = np.diff(cumulative)
    tolerance = max(1e-8, float(np.nanmax(np.abs(cumulative))) * 1e-10)
    if np.any(diff < -tolerance):
        return None
    if not np.any(diff > tolerance):
        return None
    return diff


def _last_valid(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    return float(valid.iloc[-1]) if not valid.empty else float("nan")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{owner} missing required columns: {missing}")


def _assert_catalog_request(ctx: FactorComputeContext, kernel: str) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{kernel} received unknown research signal: {signal!r}")
    if entry.kernel != kernel:
        raise ValueError(f"{signal} belongs to {entry.kernel}, not {kernel}")
    declared_family = str(ctx.params.get("family", entry.family)).strip()
    if declared_family != entry.family:
        raise ValueError(
            f"{signal} expected family={entry.family!r}, received family={declared_family!r}"
        )
    return entry


def _output_for_signal(ctx: FactorComputeContext, *, kernel: str, feature_key: str) -> pd.Series:
    entry = _assert_catalog_request(ctx, kernel)
    out = _empty_output_for_context(ctx, name=ctx.params.get("output_name", entry.signal))
    features = {
        "factor_mining_intraday_catalog_v1": _intraday_feature_frame,
        "factor_mining_daily_catalog_v1": _daily_feature_frame,
        "factor_mining_cross_asset_catalog_v1": _cross_feature_frame,
        "factor_mining_hybrid_catalog_v1": _hybrid_feature_frame,
    }[kernel](ctx)
    if feature_key not in features.columns:
        raise KeyError(f"{kernel} did not materialise requested signal {feature_key}")
    values = pd.to_numeric(features[feature_key], errors="coerce").reindex(out.index)
    out.iloc[:] = values.to_numpy(dtype="float64")
    out.name = ctx.params.get("output_name") or entry.signal
    return out.replace([np.inf, -np.inf], np.nan)


# Definitions below deliberately appear after the generic output helper; Python
# resolves them at call time and this keeps every registered class compact.


_INTRADAY_SIGNALS = tuple(entry.signal for entry in _INTRADAY_ENTRIES)
_BOOK_PRICE_COLUMNS = tuple(f"{side}_price{level}" for side in ("ask", "bid") for level in range(1, 6))
_BOOK_VOLUME_COLUMNS = tuple(f"{side}_volume{level}" for side in ("ask", "bid") for level in range(1, 6))
_INTRADAY_REQUIRED = (
    "trade_time",
    "open",
    "last",
    "volume",
    "amount",
    "num_trades",
    "pre_close",
    "high_limited",
    "low_limited",
    *_BOOK_PRICE_COLUMNS,
    *_BOOK_VOLUME_COLUMNS,
)


def _tail_mask(times: pd.Series, minutes: int) -> np.ndarray:
    if times.empty:
        return np.zeros(0, dtype=bool)
    end = times.max()
    if pd.isna(end):
        return np.zeros(len(times), dtype=bool)
    return (times >= end - pd.Timedelta(minutes=int(minutes))).to_numpy(dtype=bool)


def _window_return(prices: np.ndarray, mask: np.ndarray) -> float:
    values = prices[mask]
    values = _finite(values)
    if len(values) < 2 or values[0] <= 0.0:
        return float("nan")
    return float(values[-1] / values[0] - 1.0)


def _window_efficiency(prices: np.ndarray, mask: np.ndarray) -> float:
    values = prices[mask]
    values = _finite(values)
    if len(values) < 3 or values[0] <= 0.0:
        return float("nan")
    ret = np.diff(values) / values[:-1]
    denom = float(np.abs(ret).sum())
    return _safe_div(abs(float(values[-1] / values[0] - 1.0)), denom)


def _feature_row_intraday(group: pd.DataFrame) -> dict[str, float]:
    """Materialise all panel-only research features for one (dt, code) path."""

    result = {signal: float("nan") for signal in _INTRADAY_SIGNALS}
    group = group.sort_values("trade_time", kind="mergesort").reset_index(drop=True)
    if len(group) < 5:
        return result

    px = pd.to_numeric(group["last"], errors="coerce").to_numpy(dtype="float64")
    op = pd.to_numeric(group["open"], errors="coerce").to_numpy(dtype="float64")
    times = pd.to_datetime(group["trade_time"], errors="coerce")
    valid_px = np.isfinite(px) & (px > 0.0)
    if valid_px.sum() < 5:
        return result
    # Keep raw time alignment intact: an invalid intermediate quote renders
    # path-only measurements missing rather than silently compressing time.
    if not valid_px.all() or times.isna().any():
        return result

    full_ret = _safe_div(float(px[-1] - px[0]), float(px[0]))
    log_ret = np.diff(np.log(px))
    simple_ret = np.diff(px) / px[:-1]
    tail5 = _tail_mask(times, 5)
    tail15 = _tail_mask(times, 15)
    tail5_ret = _window_return(px, tail5)
    tail15_ret = _window_return(px, tail15)
    path_abs = float(np.abs(simple_ret).sum())
    signed_eff = _safe_div(full_ret, path_abs)
    path_eff = _safe_div(abs(full_ret), path_abs)
    roughness = _safe_div(path_abs, abs(full_ret))

    high = float(np.max(px))
    low = float(np.min(px))
    range_ratio = _safe_div(high - low, px[0])
    close_pos = _safe_div(px[-1] - low, high - low)
    wealth = px / px[0]
    running_peak = np.maximum.accumulate(wealth)
    running_floor = np.minimum.accumulate(wealth)
    drawdown = float(np.min(wealth / running_peak - 1.0))
    drawup = float(np.max(wealth / running_floor - 1.0))
    recovery = _safe_div(float(px[-1] - low), low)
    up_vol = float(np.sqrt(np.mean(np.square(simple_ret[simple_ret > 0.0])))) if np.any(simple_ret > 0.0) else float("nan")
    down_vol = float(np.sqrt(np.mean(np.square(simple_ret[simple_ret < 0.0])))) if np.any(simple_ret < 0.0) else float("nan")
    vol = float(np.std(simple_ret, ddof=1)) if len(simple_ret) >= 2 else float("nan")
    tail_vol = float(np.std(simple_ret[tail15[1:]], ddof=1)) if tail15[1:].sum() >= 3 else float("nan")
    med_abs_ret = float(np.median(np.abs(simple_ret))) if len(simple_ret) else float("nan")
    jump_share = _safe_div(float((np.abs(simple_ret) > 3.0 * med_abs_ret).sum()), float(len(simple_ret))) if med_abs_ret > EPS else float("nan")

    volume = pd.to_numeric(group["volume"], errors="coerce").to_numpy(dtype="float64")
    amount = pd.to_numeric(group["amount"], errors="coerce").to_numpy(dtype="float64")
    deals = pd.to_numeric(group["num_trades"], errors="coerce").to_numpy(dtype="float64")
    volume_inc = _incremental(volume)
    amount_inc = _incremental(amount)
    deal_inc = _incremental(deals)

    vwap = float("nan")
    tail_vwap = float("nan")
    flow_time_center = float("nan")
    volume_entropy = float("nan")
    volume_hhi = float("nan")
    volume_gini = float("nan")
    total_amount = float("nan")
    flow_interarrival_cv = float("nan")
    burst_cluster_count = float("nan")
    vol_accel5 = amount_accel5 = trade_accel5 = float("nan")
    vol_accel15 = amount_accel15 = trade_accel15 = float("nan")
    avg_trade_size_tail = float("nan")
    impact_per_amount = impact_per_trade = impact_asymmetry = float("nan")
    volume_price_corr = volume_return_corr = amount_return_corr = trade_size_return_corr = float("nan")
    flow_price_divergence = float("nan")

    if volume_inc is not None and amount_inc is not None and deal_inc is not None:
        valid_flow = (volume_inc > 0.0) & (amount_inc > 0.0) & (deal_inc >= 0.0)
        if valid_flow.sum() >= 3:
            inc_px = np.where(volume_inc > 0.0, amount_inc / volume_inc, np.nan)
            total_volume = float(volume_inc[valid_flow].sum())
            total_amount = float(amount_inc[valid_flow].sum())
            vwap = _safe_div(total_amount, total_volume)
            tail_intervals5 = tail5[1:]
            tail_intervals15 = tail15[1:]
            for mask, suffix in ((tail_intervals5, "5"), (tail_intervals15, "15")):
                in_mask = mask & valid_flow
                out_mask = (~mask) & valid_flow
                if in_mask.sum() >= 2 and out_mask.sum() >= 2:
                    time_ratio = _safe_div(float(in_mask.sum()), float(valid_flow.sum()))
                    if suffix == "5":
                        vol_accel5 = _safe_div(float(volume_inc[in_mask].sum() / total_volume), time_ratio)
                        amount_accel5 = _safe_div(float(amount_inc[in_mask].sum() / total_amount), time_ratio)
                        trade_accel5 = _safe_div(float(deal_inc[in_mask].sum() / max(float(deal_inc[valid_flow].sum()), EPS)), time_ratio)
                    else:
                        vol_accel15 = _safe_div(float(volume_inc[in_mask].sum() / total_volume), time_ratio)
                        amount_accel15 = _safe_div(float(amount_inc[in_mask].sum() / total_amount), time_ratio)
                        trade_accel15 = _safe_div(float(deal_inc[in_mask].sum() / max(float(deal_inc[valid_flow].sum()), EPS)), time_ratio)
            tail_valid = tail_intervals15 & valid_flow
            if tail_valid.sum() >= 2:
                tail_vwap = _safe_div(float(amount_inc[tail_valid].sum()), float(volume_inc[tail_valid].sum()))
                avg_trade_size_tail = _safe_div(
                    _safe_div(float(amount_inc[tail_valid].sum()), float(deal_inc[tail_valid].sum())),
                    _safe_div(float(amount_inc[valid_flow].sum()), float(deal_inc[valid_flow].sum())),
                )
            volume_entropy = _entropy(volume_inc[valid_flow])
            weights = volume_inc[valid_flow] / total_volume
            volume_hhi = float(np.square(weights).sum()) if len(weights) >= 2 else float("nan")
            volume_gini = _gini(volume_inc[valid_flow])
            seconds = (times.iloc[1:] - times.iloc[1]).dt.total_seconds().to_numpy(dtype="float64")
            flow_time_center = _safe_div(float(np.dot(seconds[valid_flow], volume_inc[valid_flow])), float(total_volume))
            positive_times = seconds[valid_flow]
            if len(positive_times) >= 3:
                gaps = np.diff(positive_times)
                flow_interarrival_cv = _safe_div(float(np.std(gaps, ddof=1)), float(np.mean(gaps))) if np.all(gaps > 0.0) else float("nan")
            burst_threshold = float(np.quantile(volume_inc[valid_flow], 0.75))
            burst = (volume_inc > burst_threshold) & valid_flow
            burst_cluster_count = float(np.sum(burst & np.r_[True, ~burst[:-1]]))
            aligned_ret = simple_ret
            volume_price_corr = _safe_corr(volume_inc, px[1:])
            volume_return_corr = _safe_corr(volume_inc, aligned_ret)
            amount_return_corr = _safe_corr(amount_inc, aligned_ret)
            trade_size = np.where(deal_inc > 0.0, amount_inc / deal_inc, np.nan)
            trade_size_return_corr = _safe_corr(trade_size, aligned_ret)
            impact_per_amount = _safe_div(float(np.abs(aligned_ret[valid_flow]).sum()), float(amount_inc[valid_flow].sum()))
            impact_per_trade = _safe_div(float(np.abs(aligned_ret[valid_flow]).sum()), float(deal_inc[valid_flow].sum()))
            positive_impact = np.abs(aligned_ret[(aligned_ret > 0.0) & valid_flow])
            negative_impact = np.abs(aligned_ret[(aligned_ret < 0.0) & valid_flow])
            impact_asymmetry = _safe_div(
                float(np.mean(positive_impact)) if len(positive_impact) else float("nan"),
                float(np.mean(negative_impact)) if len(negative_impact) else float("nan"),
            )
            flow_price_divergence = _safe_div(
                float(np.sum(np.sign(aligned_ret[valid_flow]) * volume_inc[valid_flow])), total_volume
            )

    ask_price = np.column_stack(
        [pd.to_numeric(group[f"ask_price{i}"], errors="coerce").to_numpy(dtype="float64") for i in range(1, 6)]
    )
    bid_price = np.column_stack(
        [pd.to_numeric(group[f"bid_price{i}"], errors="coerce").to_numpy(dtype="float64") for i in range(1, 6)]
    )
    ask_volume = np.column_stack(
        [pd.to_numeric(group[f"ask_volume{i}"], errors="coerce").to_numpy(dtype="float64") for i in range(1, 6)]
    )
    bid_volume = np.column_stack(
        [pd.to_numeric(group[f"bid_volume{i}"], errors="coerce").to_numpy(dtype="float64") for i in range(1, 6)]
    )
    mid = (ask_price[:, 0] + bid_price[:, 0]) / 2.0
    spread = np.where(mid > 0.0, (ask_price[:, 0] - bid_price[:, 0]) / mid, np.nan)
    bid_depth = np.nansum(bid_volume, axis=1)
    ask_depth = np.nansum(ask_volume, axis=1)
    total_depth = bid_depth + ask_depth
    imbalance1 = np.where(
        bid_volume[:, 0] + ask_volume[:, 0] > 0.0,
        (bid_volume[:, 0] - ask_volume[:, 0]) / (bid_volume[:, 0] + ask_volume[:, 0]),
        np.nan,
    )
    imbalance5 = np.where(total_depth > 0.0, (bid_depth - ask_depth) / total_depth, np.nan)
    depth_concentration = np.where(total_depth > 0.0, (bid_volume[:, 0] + ask_volume[:, 0]) / total_depth, np.nan)
    book_slope = np.where(
        mid > 0.0,
        ((ask_price[:, 4] - ask_price[:, 0]) + (bid_price[:, 0] - bid_price[:, 4])) / mid,
        np.nan,
    )
    microprice = np.where(
        bid_volume[:, 0] + ask_volume[:, 0] > 0.0,
        (ask_price[:, 0] * bid_volume[:, 0] + bid_price[:, 0] * ask_volume[:, 0]) / (bid_volume[:, 0] + ask_volume[:, 0]),
        np.nan,
    )
    micro_bias = np.where(np.abs(spread) > EPS, (microprice - mid) / (mid * spread), np.nan)
    tail_spread = spread[tail15]
    spread_tail_change = _safe_div(float(tail_spread[-1] - tail_spread[0]), abs(float(tail_spread[0]))) if len(tail_spread) >= 2 else float("nan")
    depth_change = _safe_div(float(total_depth[-1] - total_depth[0]), float(total_depth[0]))
    bid_recovery = _safe_div(float(bid_depth[-1] - bid_depth[0]), float(bid_depth[0])) * (-np.sign(full_ret))
    ask_recovery = _safe_div(float(ask_depth[-1] - ask_depth[0]), float(ask_depth[0])) * np.sign(full_ret)
    liquidity_recovery = _safe_div(depth_change, abs(full_ret))
    quote_staleness = float(np.mean(np.isclose(np.diff(mid), 0.0, rtol=0.0, atol=1e-10))) if len(mid) >= 2 else float("nan")
    quote_dislocation = float(np.nanmean(np.abs(px - mid) / mid)) if np.isfinite(mid).any() else float("nan")
    pressure_impact = _safe_corr(imbalance5[:-1], simple_ret)
    pressure_tail_reversal = -float(imbalance5[-1]) * tail15_ret if np.isfinite(imbalance5[-1]) and np.isfinite(tail15_ret) else float("nan")

    result.update(
        {
            "intra_full_return": full_ret,
            "intra_path_efficiency": path_eff,
            "intra_signed_efficiency": signed_eff,
            "intra_path_roughness": roughness,
            "intra_return_autocorr": _safe_autocorr(simple_ret),
            "intra_return_skew": _safe_skew(simple_ret),
            "intra_return_kurtosis": _safe_kurtosis(simple_ret),
            "intra_up_down_semivol_ratio": _safe_div(up_vol, down_vol),
            "intra_drawdown": drawdown,
            "intra_drawup": drawup,
            "intra_recovery_from_low": recovery,
            "intra_close_position": close_pos,
            "intra_tail_return_5m": tail5_ret,
            "intra_tail_return_15m": tail15_ret,
            "intra_tail_vs_full_return": full_ret - tail15_ret if np.isfinite(full_ret) and np.isfinite(tail15_ret) else float("nan"),
            "intra_tail_vol_ratio": _safe_div(tail_vol, vol),
            "intra_tail_path_efficiency": _window_efficiency(px, tail15),
            "intra_tail_jump_share": jump_share,
            "intra_time_under_vwap": float(np.mean(px < vwap)) if np.isfinite(vwap) else float("nan"),
            "intra_tail_reversal_ratio": _safe_div(-tail15_ret, full_ret - tail15_ret) if np.isfinite(tail15_ret) else float("nan"),
            "intra_price_entropy": _entropy(np.abs(simple_ret)),
            "intra_volume_accel_5m": vol_accel5,
            "intra_volume_accel_15m": vol_accel15,
            "intra_amount_accel_5m": amount_accel5,
            "intra_amount_accel_15m": amount_accel15,
            "intra_trade_accel_5m": trade_accel5,
            "intra_trade_accel_15m": trade_accel15,
            "intra_avg_trade_size_tail": avg_trade_size_tail,
            "intra_volume_entropy": volume_entropy,
            "intra_volume_hhi": volume_hhi,
            "intra_volume_gini": volume_gini,
            "intra_flow_time_center": flow_time_center,
            "intra_flow_interarrival_cv": flow_interarrival_cv,
            "intra_burst_cluster_count": burst_cluster_count,
            "intra_volume_price_corr": volume_price_corr,
            "intra_volume_return_corr": volume_return_corr,
            "intra_amount_return_corr": amount_return_corr,
            "intra_trade_size_return_corr": trade_size_return_corr,
            "intra_impact_per_amount": impact_per_amount,
            "intra_impact_per_trade": impact_per_trade,
            "intra_impact_asymmetry": impact_asymmetry,
            "intra_flow_price_divergence": flow_price_divergence,
            "book_spread_last": float(spread[-1]),
            "book_spread_mean": float(np.nanmean(spread)),
            "book_spread_std": float(np.nanstd(spread, ddof=1)) if np.isfinite(spread).sum() >= 2 else float("nan"),
            "book_spread_tail_change": spread_tail_change,
            "book_depth_total": float(total_depth[-1]),
            "book_depth_concentration": float(depth_concentration[-1]),
            "book_depth_slope": float(book_slope[-1]),
            "book_imbalance_l1_last": float(imbalance1[-1]),
            "book_imbalance_l5_last": float(imbalance5[-1]),
            "book_imbalance_l1_mean": float(np.nanmean(imbalance1)),
            "book_imbalance_l5_mean": float(np.nanmean(imbalance5)),
            "book_imbalance_l1_change": float(imbalance1[-1] - imbalance1[0]) if np.isfinite(imbalance1[[0, -1]]).all() else float("nan"),
            "book_imbalance_l5_change": float(imbalance5[-1] - imbalance5[0]) if np.isfinite(imbalance5[[0, -1]]).all() else float("nan"),
            "book_imbalance_dispersion": float(np.nanstd(imbalance5, ddof=1)) if np.isfinite(imbalance5).sum() >= 2 else float("nan"),
            "book_microprice_bias_last": float(micro_bias[-1]),
            "book_microprice_bias_mean": float(np.nanmean(micro_bias)),
            "book_microprice_bias_change": float(micro_bias[-1] - micro_bias[0]) if np.isfinite(micro_bias[[0, -1]]).all() else float("nan"),
            "book_imbalance_return_corr": _safe_corr(imbalance5[:-1], simple_ret),
            "book_microprice_return_corr": _safe_corr(micro_bias[:-1], simple_ret),
            "book_spread_return_corr": _safe_corr(spread[:-1], simple_ret),
            "book_depth_return_corr": _safe_corr(total_depth[:-1], simple_ret),
            "book_bid_depth_recovery": bid_recovery,
            "book_ask_depth_recovery": ask_recovery,
            "book_liquidity_recovery": liquidity_recovery,
            "book_pressure_impact": pressure_impact,
            "book_pressure_tail_reversal": pressure_tail_reversal,
            "book_quote_staleness": quote_staleness,
            "book_quote_dislocation": quote_dislocation,
        }
    )
    # Extra state required by cross/hybrid kernels.  They do not appear in the
    # intraday catalogue and are kept out of factor specs.
    result["__intra_vwap"] = vwap
    result["__intra_tail_vwap"] = tail_vwap
    result["__intra_amount_total"] = total_amount
    result["__intra_last_to_mid"] = _safe_div(float(px[-1] - mid[-1]), float(mid[-1]))
    result["__intra_range_ratio"] = range_ratio
    result["__intra_high_time_fraction"] = _safe_div(float(np.argmax(px)), float(max(1, len(px) - 1)))
    result["__intra_low_time_fraction"] = _safe_div(float(np.argmin(px)), float(max(1, len(px) - 1)))
    result["__intra_limit_up_distance"] = _safe_div(float(pd.to_numeric(group["high_limited"], errors="coerce").iloc[-1] - px[-1]), px[-1])
    result["__intra_limit_down_distance"] = _safe_div(float(px[-1] - pd.to_numeric(group["low_limited"], errors="coerce").iloc[-1]), px[-1])
    return result


def _build_intraday_features(
    panel: pd.DataFrame,
    *,
    signal_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    panel = _signal_day_panel(panel, signal_date=signal_date)
    score_day = _signal_date_from_panel(panel)
    if score_day is None:
        return pd.DataFrame(
            columns=[*_INTRADAY_SIGNALS],
            index=pd.MultiIndex.from_arrays([[], []], names=["dt", "code"]),
        )
    frame = panel.reset_index().copy()
    _require_columns(frame, _INTRADAY_REQUIRED, owner="factor_mining_intraday_catalog_v1")
    frame = _strict_score_day_snapshots(frame, signal_date=score_day)
    rows: list[dict[str, object]] = []
    for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
        row: dict[str, object] = {"dt": dt, "code": str(code)}
        row.update(_feature_row_intraday(group))
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[*_INTRADAY_SIGNALS],
            index=pd.MultiIndex.from_arrays([[], []], names=["dt", "code"]),
        )
    out = pd.DataFrame(rows).set_index(["dt", "code"]).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def _intraday_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    return _cached(ctx, f"factor_mining_intraday_features:{CATALOG_VERSION}", lambda: _build_intraday_features(ctx.panel))


@FactorRegistry.register("factor_mining_intraday_catalog_v1")
class FactorMiningIntradayCatalogV1(Factor):
    """Research-only T1430 price-path, flow, and order-book catalogue."""

    name = "factor_mining_intraday_catalog_v1"
    kernel_name = "factor_mining_intraday_catalog_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _assert_catalog_request(ctx, self.kernel_name)
        out = _output_for_signal(ctx, kernel=self.kernel_name, feature_key=entry.signal)
        out.name = self.output_name(entry.signal)
        return out


_DAILY_SIGNALS = tuple(entry.signal for entry in _DAILY_ENTRIES)
_DAILY_INTERNAL_DEFAULTS: dict[str, object] = {
    "__daily_return_mean20": float("nan"),
    "__daily_return_std20": float("nan"),
    "__daily_intraday_mean20": float("nan"),
    "__daily_range_mean20": float("nan"),
    "__daily_range_std20": float("nan"),
    "__daily_amount_z20": float("nan"),
    "__daily_amihud_z20": float("nan"),
    "__daily_price_range_ret_corr": float("nan"),
    "__daily_volume_return_corr": float("nan"),
    "__daily_amount_return_corr": float("nan"),
    "__daily_deal_return_corr": float("nan"),
    "__late_mean20": float("nan"),
    "__overnight_mean20": float("nan"),
    "__twap_curve_mean20": float("nan"),
    "__base_stock_code": "",
    "__base_premium_raw": float("nan"),
    "__base_premium_z": float("nan"),
    "__base_duration": float("nan"),
    "__base_stockvol": float("nan"),
    "__base_moneyness": float("nan"),
    "__base_floor_distance": float("nan"),
    "__base_call_state": float("nan"),
    "__base_trigger_progress": float("nan"),
    "__base_remain_size": float("nan"),
    "__base_turnover": float("nan"),
    "__base_stock_amount": float("nan"),
    "__base_cb_amount": float("nan"),
    "__base_stock_bond_beta20": float("nan"),
}
_DAILY_TWAP_COLUMNS = (
    "twap_0930_0935",
    "twap_0935_1000",
    "twap_1000_1030",
    "twap_1100_1130",
    "twap_1300_1330",
    "twap_1330_1400",
    "twap_1400_1430",
    "twap_1430_1442",
    "twap_1430_1500",
    "twap_1442_1457",
)
_DAILY_PRICE_COLUMNS = (
    "prev_close_price",
    "act_prev_close_price",
    "close_price",
    "open_price",
    "high_price",
    "low_price",
    "volume",
    "amount",
    "deal",
)
_DAILY_BASE_COLUMNS = (
    "year_to_mat",
    "remain_size",
    "current_yield",
    "cb_conv_price",
    "turnover_rate",
    "stock_code",
    "stock_close_price",
    "bond_prem_ratio",
    "debt_puredebt_ratio",
    "puredebt_prem_ratio",
    "conv_value",
    "ytm",
    "duration",
    "modify_duration",
    "convexity",
    "base_rate",
    "stock_volatility",
    "pure_redemption_value",
    "redemption_prem_ratio",
    "cb_prev_close_price",
    "cb_close_price",
    "cb_volume",
    "cb_amount",
    "cb_deal",
    "stk_volume",
    "stk_amount",
    "stk_deal",
    "cb_call_price",
    "trigger_is_price",
    "trigger_cum_days",
    "trigger_reach_days",
    "in_trigger_process",
    "trigger_price_revise",
    "trigger_cum_days_revise",
    "trigger_reach_days_revise",
)


def _daily_requirements(params: dict | None = None) -> list[DailyFactorRequirement]:
    params = dict(params or {})
    lookback = max(65, int(params.get("context_lookback_days", 65) or 65))
    return [
        DailyFactorRequirement("market_cbond.daily_twap", ("exchange_code", *_DAILY_TWAP_COLUMNS), lookback),
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", *_DAILY_PRICE_COLUMNS), lookback),
        DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", *_DAILY_BASE_COLUMNS), lookback),
    ]


def _strict_history_source(
    ctx: FactorComputeContext,
    source: str,
    required_columns: Iterable[str],
    signal_date: pd.Timestamp,
) -> pd.DataFrame:
    source_df = ctx.daily_data.get(source)
    if source_df is None or source_df.empty:
        return pd.DataFrame(columns=["trade_date", "code", *required_columns])
    _require_columns(source_df, ("trade_date", "code", *required_columns), owner=f"{source} research context")
    selected_columns = ["trade_date", "code", *required_columns]
    if "exchange_code" in source_df.columns and "exchange_code" not in selected_columns:
        selected_columns.append("exchange_code")
    frame = source_df[selected_columns].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(
        frame["code"],
        frame["exchange_code"] if "exchange_code" in frame.columns else None,
    )
    frame = frame.loc[frame["trade_date"].notna() & (frame["trade_date"] < signal_date)].copy()
    frame = frame.loc[frame["code"] != ""].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]].head(3)
        raise ValueError(f"{source} has duplicate pre-signal daily rows: {examples.to_dict('records')}")
    frame = frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
    return frame


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _tail(values: np.ndarray, count: int, *, min_count: int) -> np.ndarray:
    values = _finite(values)
    if len(values) < min_count:
        return np.array([], dtype="float64")
    return values[-count:]


def _mean(values: np.ndarray, count: int, *, min_count: int) -> float:
    values = _tail(values, count, min_count=min_count)
    return float(values.mean()) if len(values) else float("nan")


def _std(values: np.ndarray, count: int, *, min_count: int) -> float:
    values = _tail(values, count, min_count=min_count)
    return float(values.std(ddof=1)) if len(values) >= 2 else float("nan")


def _z_last(values: np.ndarray, count: int = 20, *, min_count: int = 15) -> float:
    values = _tail(values, count, min_count=min_count)
    if len(values) < 2:
        return float("nan")
    return _safe_div(float(values[-1] - values.mean()), float(values.std(ddof=1)))


def _last_delta(values: np.ndarray, periods: int) -> float:
    values = _finite(values)
    if len(values) <= periods:
        return float("nan")
    return float(values[-1] - values[-1 - periods])


def _last_pct_change(values: np.ndarray, periods: int) -> float:
    values = _finite(values)
    if len(values) <= periods or abs(values[-1 - periods]) <= EPS:
        return float("nan")
    return float(values[-1] / values[-1 - periods] - 1.0)


def _trailing_corr(left: np.ndarray, right: np.ndarray, *, count: int = 20, min_count: int = 12) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna().tail(count)
    if len(pair) < min_count:
        return float("nan")
    return _safe_corr(pair["left"].to_numpy(dtype="float64"), pair["right"].to_numpy(dtype="float64"))


def _run_length(values: np.ndarray, predicate: Callable[[float], bool]) -> float:
    count = 0
    for value in values[::-1]:
        if not np.isfinite(value) or not predicate(float(value)):
            break
        count += 1
    return float(count) if count else float("nan")


def _daily_row(
    price: pd.DataFrame,
    twap: pd.DataFrame,
    base: pd.DataFrame,
    *,
    strict_mapping_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {signal: float("nan") for signal in _DAILY_SIGNALS}
    result.update(_DAILY_INTERNAL_DEFAULTS)

    # ---- Daily OHLC and liquidity histories ---------------------------------
    if not price.empty:
        p = price.copy()
        for column in _DAILY_PRICE_COLUMNS:
            p[column] = pd.to_numeric(p[column], errors="coerce")
        prior_close = p["act_prev_close_price"].where(p["act_prev_close_price"] > 0.0, p["prev_close_price"])
        close = p["close_price"]
        open_px = p["open_price"]
        high = p["high_price"]
        low = p["low_price"]
        p["ret"] = np.where((prior_close > 0.0) & (close > 0.0), close / prior_close - 1.0, np.nan)
        p["gap"] = np.where((prior_close > 0.0) & (open_px > 0.0), open_px / prior_close - 1.0, np.nan)
        p["intraday"] = np.where((open_px > 0.0) & (close > 0.0), close / open_px - 1.0, np.nan)
        p["range"] = np.where(prior_close > 0.0, (high - low) / prior_close, np.nan)
        denom = high - low
        p["body_range"] = np.where(np.abs(denom) > EPS, (close - open_px) / denom, np.nan)
        p["upper_wick"] = np.where(np.abs(denom) > EPS, (high - np.maximum(open_px, close)) / denom, np.nan)
        p["lower_wick"] = np.where(np.abs(denom) > EPS, (np.minimum(open_px, close) - low) / denom, np.nan)
        p["close_loc"] = np.where(np.abs(denom) > EPS, (close - low) / denom, np.nan)
        p["efficiency"] = np.where(np.abs(denom) > EPS, np.abs(close - open_px) / denom, np.nan)
        p["log_amount"] = np.where(p["amount"] > 0.0, np.log(p["amount"]), np.nan)
        p["log_volume"] = np.where(p["volume"] > 0.0, np.log(p["volume"]), np.nan)
        p["log_deal"] = np.where(p["deal"] > 0.0, np.log(p["deal"]), np.nan)
        p["trade_size"] = np.where(p["deal"] > 0.0, p["amount"] / p["deal"], np.nan)
        p["amihud"] = np.where(p["amount"] > 0.0, np.abs(p["ret"]) / p["amount"], np.nan)
        p["sqrt_impact"] = np.where(p["amount"] > 0.0, np.sign(p["ret"]) * np.abs(p["ret"]) / np.sqrt(p["amount"]), np.nan)

        ret = _numeric(p, "ret")
        gap = _numeric(p, "gap")
        intraday = _numeric(p, "intraday")
        rng = _numeric(p, "range")
        body = _numeric(p, "body_range")
        close_loc = _numeric(p, "close_loc")
        close_values = _numeric(p, "close_price")
        result.update(
            {
                "dpx_body_range": _last_valid(p["body_range"]),
                "dpx_upper_wick": _last_valid(p["upper_wick"]),
                "dpx_lower_wick": _last_valid(p["lower_wick"]),
                "dpx_close_location": _last_valid(p["close_loc"]),
                "dpx_path_efficiency": _last_valid(p["efficiency"]),
                "dpx_range_expansion": _safe_div(
                    _last_valid(p["range"]), float(np.nanmedian(_tail(rng[:-1], 20, min_count=12))) if len(rng) > 1 else float("nan")
                ),
                "dpx_breakout_distance": _z_last(close_values, 20),
                "dret_tstat": _safe_div(_mean(ret, 20, min_count=15), _std(ret, 20, min_count=15)),
                "dret_downside_semivar_ratio": _safe_div(
                    float(np.sqrt(np.mean(np.square(_tail(ret, 20, min_count=15)[_tail(ret, 20, min_count=15) < 0.0])))) if np.any(_tail(ret, 20, min_count=15) < 0.0) else float("nan"),
                    float(np.sqrt(np.mean(np.square(_tail(ret, 20, min_count=15)[_tail(ret, 20, min_count=15) > 0.0])))) if np.any(_tail(ret, 20, min_count=15) > 0.0) else float("nan"),
                ),
                "dret_upside_semivar_surprise": _safe_div(
                    max(float(ret[-1]), 0.0) if len(ret) else float("nan"),
                    float(np.sqrt(np.mean(np.square(_tail(ret, 20, min_count=15)[_tail(ret, 20, min_count=15) > 0.0])))) if np.any(_tail(ret, 20, min_count=15) > 0.0) else float("nan"),
                ),
                "dret_skew": _safe_skew(_tail(ret, 20, min_count=15)),
                "dret_tail_loss_frequency": float(np.mean(_tail(ret, 20, min_count=15) < np.nanquantile(_tail(ret, 20, min_count=15), 0.1))) if len(_tail(ret, 20, min_count=15)) else float("nan"),
                "dret_drawup_drawdown_asym": _drawup_drawdown_asym(_tail(ret, 20, min_count=15)),
                "dret_sign_run_reversal": _sign_run_reversal(_tail(ret, 20, min_count=15)),
                "gap_open_z": _z_last(gap, 20),
                "gap_close_followthrough": _last_valid(p["gap"]) * _last_valid(p["intraday"]),
                "gap_reversal": -_last_valid(p["gap"]) * (1.0 - _last_valid(p["close_loc"])),
                "gap_range_amplification": abs(_last_valid(p["gap"])) * _last_valid(p["range"]),
                "gap_persistence": _safe_autocorr(_tail(gap, 20, min_count=12)),
                "gap_extreme_recovery": _last_valid(p["close_loc"]) * abs(_z_last(gap, 20)),
                "gap_vs_prior_vol": _safe_div(_last_valid(p["gap"]), _std(ret, 20, min_count=15)),
                "impact_amihud_absret": _mean(_numeric(p, "amihud"), 20, min_count=15),
                "impact_signed_sqrt_dollar": _mean(_numeric(p, "sqrt_impact"), 20, min_count=15),
                "impact_range_per_amount": _safe_div(_last_valid(p["range"]), _last_valid(p["amount"])),
                "impact_body_per_amount": _safe_div(abs(_last_valid(p["intraday"])), _last_valid(p["amount"])),
                "impact_close_dislocation_per_deal": _safe_div(abs(_last_valid(p["ret"])), _last_valid(p["deal"])),
                "impact_illiquidity_shock": _z_last(_numeric(p, "amihud"), 20),
                "impact_up_down_asymmetry": _impact_asymmetry(ret, _numeric(p, "amount")),
                "liq_amount_shock": _z_last(_numeric(p, "log_amount"), 20),
                "liq_volume_shock": _z_last(_numeric(p, "log_volume"), 20),
                "liq_deal_count_shock": _z_last(_numeric(p, "log_deal"), 20),
                "liq_avg_trade_size_shock": _z_last(_numeric(p, "trade_size"), 20),
                "liq_notional_per_volume": _safe_div(_last_valid(p["amount"]), _last_valid(p["volume"])),
                "liq_volume_persistence": _safe_autocorr(_tail(_numeric(p, "log_volume"), 20, min_count=12)),
                "liq_drought_streak": _drought_streak(_numeric(p, "log_amount")),
                "__daily_return_mean20": _mean(ret, 20, min_count=15),
                "__daily_return_std20": _std(ret, 20, min_count=15),
                "__daily_intraday_mean20": _mean(intraday, 20, min_count=15),
                "__daily_range_mean20": _mean(rng, 20, min_count=15),
                "__daily_range_std20": _std(rng, 20, min_count=15),
                "__daily_amount_z20": _z_last(_numeric(p, "log_amount"), 20),
                "__daily_amihud_z20": _z_last(_numeric(p, "amihud"), 20),
                "__daily_price_range_ret_corr": _trailing_corr(rng, ret),
                "__daily_volume_return_corr": _trailing_corr(_numeric(p, "log_volume"), ret),
                "__daily_amount_return_corr": _trailing_corr(_numeric(p, "log_amount"), ret),
                "__daily_deal_return_corr": _trailing_corr(_numeric(p, "log_deal"), ret),
            }
        )

    # ---- Historical TWAP curve ------------------------------------------------
    if not twap.empty:
        t = twap.copy()
        for column in _DAILY_TWAP_COLUMNS:
            t[column] = pd.to_numeric(t[column], errors="coerce")
        def _ratio(num: str, den: str) -> np.ndarray:
            a = _numeric(t, num)
            b = _numeric(t, den)
            return np.where((a > 0.0) & (b > 0.0), a / b - 1.0, np.nan)
        morning = _ratio("twap_0935_1000", "twap_0930_0935")
        lunch_gap = _ratio("twap_1300_1330", "twap_1100_1130")
        afternoon = _ratio("twap_1400_1430", "twap_1300_1330")
        late = _ratio("twap_1430_1442", "twap_1400_1430")
        late_extension = _ratio("twap_1430_1500", "twap_1430_1442")
        execution_premium = _ratio("twap_1442_1457", "twap_1430_1442")
        curve = morning + afternoon - 2.0 * late
        prior_late = t["twap_1430_1442"].shift(1).to_numpy(dtype="float64")
        morning_open = _numeric(t, "twap_0930_0935")
        overnight = np.where((prior_late > 0.0) & (morning_open > 0.0), morning_open / prior_late - 1.0, np.nan)
        twap_segments = np.column_stack([morning, lunch_gap, afternoon, late])
        segment_vol = np.nanstd(twap_segments, axis=1)
        segment_eff = np.where(
            np.nansum(np.abs(twap_segments), axis=1) > EPS,
            np.abs(morning + lunch_gap + afternoon + late) / np.nansum(np.abs(twap_segments), axis=1),
            np.nan,
        )
        result.update(
            {
                "twap_morning_slope": _mean(morning, 20, min_count=15),
                "twap_lunch_reopen_gap": _mean(lunch_gap, 20, min_count=15),
                "twap_afternoon_slope": _mean(afternoon, 20, min_count=15),
                "twap_curve_curvature": _mean(curve, 20, min_count=15),
                "twap_morning_afternoon_disagreement": _mean(morning * afternoon, 20, min_count=15),
                "twap_segment_volatility": _mean(segment_vol, 20, min_count=15),
                "twap_path_efficiency": _mean(segment_eff, 20, min_count=15),
                "late_preclose_ramp": _mean(late, 20, min_count=15),
                "late_execution_premium": _mean(execution_premium, 20, min_count=15),
                "late_vs_midday_reversal": _mean(late - afternoon, 20, min_count=15),
                "late_extension": _mean(late_extension, 20, min_count=15),
                "late_return_concentration": _mean(np.abs(late) / (np.abs(morning) + np.abs(afternoon) + np.abs(late) + EPS), 20, min_count=15),
                "late_price_impact_z": _z_last(late, 20),
                "late_path_sign_stability": float(np.abs(np.mean(np.sign(_tail(late, 20, min_count=15))))) if len(_tail(late, 20, min_count=15)) else float("nan"),
                "overnight_on_prior_intraday_beta": _trailing_corr(overnight, late),
                "overnight_reversal_after_tail": _mean(-overnight * late, 20, min_count=15),
                "overnight_win_rate": float(np.mean(_tail(overnight, 20, min_count=15) > 0.0)) if len(_tail(overnight, 20, min_count=15)) else float("nan"),
                "overnight_downside_semisharpe": _downside_semisharpe(_tail(overnight, 20, min_count=15)),
                "overnight_tail_loss_share": _tail_loss_share(_tail(overnight, 20, min_count=15)),
                "overnight_autocorr": _safe_autocorr(_tail(overnight, 20, min_count=12)),
                "overnight_vs_prior_range_sensitivity": _trailing_corr(overnight[1:], np.abs(late[:-1]), count=19, min_count=12),
                "__late_mean20": _mean(late, 20, min_count=15),
                "__overnight_mean20": _mean(overnight, 20, min_count=15),
                "__twap_curve_mean20": _mean(curve, 20, min_count=15),
            }
        )

    # ---- Structural T-1 base history -----------------------------------------
    if not base.empty:
        b = base.copy()
        for column in _DAILY_BASE_COLUMNS:
            if column != "stock_code":
                b[column] = pd.to_numeric(b[column], errors="coerce")
        def arr(column: str) -> np.ndarray:
            return _numeric(b, column)
        premium = arr("bond_prem_ratio")
        pure_premium = arr("puredebt_prem_ratio")
        conv_value = arr("conv_value")
        stock_vol = arr("stock_volatility")
        duration = arr("duration")
        ytm = arr("ytm")
        current_yield = arr("current_yield")
        floor = arr("pure_redemption_value")
        cb_close = arr("cb_close_price")
        stk_close = arr("stock_close_price")
        cb_amount = arr("cb_amount")
        cb_deal = arr("cb_deal")
        stk_amount = arr("stk_amount")
        stk_deal = arr("stk_deal")
        remain = arr("remain_size")
        trigger_cum = arr("trigger_cum_days")
        trigger_reach = arr("trigger_reach_days")
        trigger_revise = arr("trigger_cum_days_revise")
        call_price = arr("cb_call_price")
        premium_tail = _tail(premium, 20, min_count=15)
        premium_revert_distance = (
            _safe_div(
                float(premium_tail[-1] - np.nanmedian(premium_tail)),
                abs(float(np.nanmedian(premium_tail))),
            )
            if len(premium_tail)
            else float("nan")
        )
        cb_ret = np.full_like(cb_close, np.nan)
        stk_ret = np.full_like(stk_close, np.nan)
        if len(cb_close) >= 2:
            cb_ret[1:] = np.where(
                (cb_close[:-1] > 0.0) & (cb_close[1:] > 0.0),
                cb_close[1:] / cb_close[:-1] - 1.0,
                np.nan,
            )
        if len(stk_close) >= 2:
            stk_ret[1:] = np.where(
                (stk_close[:-1] > 0.0) & (stk_close[1:] > 0.0),
                stk_close[1:] / stk_close[:-1] - 1.0,
                np.nan,
            )
        beta_pair = np.column_stack([cb_ret, stk_ret])
        beta_pair = beta_pair[np.isfinite(beta_pair).all(axis=1)]
        stock_bond_beta20 = (
            _safe_div(
                float(np.cov(beta_pair[-20:, 0], beta_pair[-20:, 1], ddof=1)[0, 1]),
                float(np.var(beta_pair[-20:, 1], ddof=1)),
            )
            if len(beta_pair) >= 12
            else float("nan")
        )
        mapped_stock_code = ""
        if len(b) and "trade_date" in b.columns:
            latest_base_date = pd.Timestamp(b["trade_date"].iloc[-1])
            mapping_is_current = (
                strict_mapping_date is None
                or (
                    pd.notna(latest_base_date)
                    and latest_base_date.normalize() == pd.Timestamp(strict_mapping_date).normalize()
                )
            )
            latest_stock_code = b["stock_code"].iloc[-1]
            if mapping_is_current and pd.notna(latest_stock_code):
                mapped_stock_code = str(latest_stock_code).strip().upper()
        moneyness = np.where(cb_close > 0.0, conv_value / cb_close, np.nan)
        premium_spread = premium - pure_premium
        result.update(
            {
                "base_premium_z": _z_last(premium, 20),
                "base_puredebt_premium_z": _z_last(pure_premium, 20),
                "base_premium_spread": _last_valid(pd.Series(premium_spread)),
                "base_premium_percentile_break": _percentile_last(premium, 20),
                "base_redemption_premium_z": _z_last(arr("redemption_prem_ratio"), 20),
                "base_premium_dispersion": _std(premium, 20, min_count=15),
                "base_bond_premium_delta1": _last_delta(premium, 1),
                "base_puredebt_premium_delta1": _last_delta(pure_premium, 1),
                "base_premium_acceleration": _last_delta(premium, 1) - _last_delta(premium, 5),
                "base_premium_mean_revert_distance": premium_revert_distance,
                "base_premium_breakout": _breakout_last(premium, 20),
                "base_premium_vs_conv_value_change": _last_delta(premium, 1) - _last_pct_change(conv_value, 1),
                "base_premium_vs_stock_vol_change": _last_delta(premium, 1) - _last_pct_change(stock_vol, 1),
                "base_yield_spread": _last_valid(b["current_yield"] - b["base_rate"]),
                "base_ytm_change": _last_delta(ytm, 1),
                "base_duration_adjusted_yield": _safe_div(_last_valid(b["current_yield"]), _last_valid(b["duration"])),
                "base_convexity_duration_ratio": _safe_div(_last_valid(b["convexity"]), _last_valid(b["duration"])),
                "base_time_to_mat_residual": _z_last(arr("year_to_mat"), 20),
                "base_stock_volatility_z": _z_last(stock_vol, 20),
                "base_duration_stockvol_interaction": _last_valid(b["duration"]) * _last_valid(b["stock_volatility"]),
                "base_call_price_distance": _safe_div(_last_valid(b["cb_call_price"]) - _last_valid(b["cb_close_price"]), _last_valid(b["cb_close_price"])),
                "base_trigger_progress_ratio": _safe_div(_last_valid(b["trigger_cum_days"]), _last_valid(b["trigger_reach_days"])),
                "base_trigger_days_remaining": _last_valid(b["trigger_reach_days"]) - _last_valid(b["trigger_cum_days"]),
                "base_in_trigger_process": _last_valid(b["in_trigger_process"]),
                "base_trigger_progress_delta": _last_delta(trigger_cum, 1),
                "base_trigger_revision_gap": _last_valid(b["trigger_cum_days"]) - _last_valid(b["trigger_cum_days_revise"]),
                "base_call_state_transition": _last_delta(arr("in_trigger_process"), 1),
                "base_conversion_moneyness": _last_valid(pd.Series(moneyness)),
                "base_conversion_delta_proxy": _last_pct_change(moneyness, 1),
                "base_conv_price_distance": _safe_div(_last_valid(b["stock_close_price"]) - _last_valid(b["cb_conv_price"]), _last_valid(b["cb_conv_price"])),
                "base_stockvol_per_moneyness": _safe_div(_last_valid(b["stock_volatility"]), _last_valid(pd.Series(moneyness))),
                "base_moneyness_time_decay": _safe_div(_last_valid(pd.Series(moneyness)), _last_valid(b["year_to_mat"])),
                "base_conversion_value_change": _last_pct_change(conv_value, 1),
                "base_optionality_premium_residual": _last_valid(b["bond_prem_ratio"]) - _last_valid(pd.Series(moneyness)),
                "base_turnover_z": _z_last(arr("turnover_rate"), 20),
                "base_float_adjusted_amount": _safe_div(_last_valid(b["cb_amount"]), _last_valid(b["remain_size"])),
                "base_bond_stock_amount_ratio": _safe_div(_last_valid(b["cb_amount"]), _last_valid(b["stk_amount"])),
                "base_bond_stock_trade_size_ratio": _safe_div(
                    _safe_div(_last_valid(b["cb_amount"]), _last_valid(b["cb_deal"])),
                    _safe_div(_last_valid(b["stk_amount"]), _last_valid(b["stk_deal"])),
                ),
                "base_liquidity_mismatch_z": _z_last(np.log(np.where((cb_amount > 0.0) & (stk_amount > 0.0), cb_amount / stk_amount, np.nan)), 20),
                "base_turnover_change": _last_delta(arr("turnover_rate"), 1),
                "base_float_liquidity_drought": _drought_streak(np.log(np.where(cb_amount > 0.0, cb_amount / np.maximum(remain, EPS), np.nan))),
                "base_bond_floor_distance": _safe_div(_last_valid(b["cb_close_price"]) - _last_valid(b["pure_redemption_value"]), _last_valid(b["pure_redemption_value"])),
                "base_debt_premium_floor_gap": _last_valid(b["debt_puredebt_ratio"]) - _last_valid(b["puredebt_prem_ratio"]),
                "base_yield_to_floor_ratio": _safe_div(_last_valid(b["ytm"]), _last_valid(b["pure_redemption_value"])),
                "base_duration_adjusted_floor": _safe_div(_last_valid(b["pure_redemption_value"]), _last_valid(b["duration"])),
                "base_floor_change": _last_pct_change(floor, 1),
                "base_floor_volatility": _std(floor, 20, min_count=15),
                "base_floor_conversion_residual": _safe_div(_last_valid(b["conv_value"]) - _last_valid(b["pure_redemption_value"]), _last_valid(b["pure_redemption_value"])),
                "__base_stock_code": mapped_stock_code,
                "__base_premium_raw": _last_valid(b["bond_prem_ratio"]),
                "__base_premium_z": _z_last(premium, 20),
                "__base_duration": _last_valid(b["duration"]),
                "__base_stockvol": _last_valid(b["stock_volatility"]),
                "__base_moneyness": _last_valid(pd.Series(moneyness)),
                "__base_floor_distance": _safe_div(_last_valid(b["cb_close_price"]) - _last_valid(b["pure_redemption_value"]), _last_valid(b["pure_redemption_value"])),
                "__base_call_state": _last_valid(b["in_trigger_process"]),
                "__base_trigger_progress": _safe_div(
                    _last_valid(b["trigger_cum_days"]),
                    _last_valid(b["trigger_reach_days"]),
                ),
                "__base_remain_size": _last_valid(b["remain_size"]),
                "__base_turnover": _last_valid(b["turnover_rate"]),
                "__base_stock_amount": _last_valid(b["stk_amount"]),
                "__base_cb_amount": _last_valid(b["cb_amount"]),
                "__base_stock_bond_beta20": stock_bond_beta20,
            }
        )
    return result


def _drawup_drawdown_asym(returns: np.ndarray) -> float:
    returns = _finite(returns)
    if len(returns) < 5:
        return float("nan")
    wealth = np.cumprod(1.0 + returns)
    dd = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
    du = float(np.max(wealth / np.minimum.accumulate(wealth) - 1.0))
    return _safe_div(du, abs(dd))


def _sign_run_reversal(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 3:
        return float("nan")
    signs = np.sign(values)
    sign = signs[-1]
    if sign == 0.0:
        return float("nan")
    run = 0
    for value in signs[::-1]:
        if value != sign:
            break
        run += 1
    return float(-sign * run / len(signs))


def _impact_asymmetry(returns: np.ndarray, amount: np.ndarray) -> float:
    up = np.abs(returns[(returns > 0.0) & np.isfinite(amount) & (amount > 0.0)] / amount[(returns > 0.0) & np.isfinite(amount) & (amount > 0.0)])
    down = np.abs(returns[(returns < 0.0) & np.isfinite(amount) & (amount > 0.0)] / amount[(returns < 0.0) & np.isfinite(amount) & (amount > 0.0)])
    return _safe_div(float(np.mean(up)) if len(up) else float("nan"), float(np.mean(down)) if len(down) else float("nan"))


def _drought_streak(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 10:
        return float("nan")
    threshold = float(np.nanmedian(values[-20:]))
    return _run_length(values, lambda value: value < threshold)


def _percentile_last(values: np.ndarray, count: int) -> float:
    values = _tail(values, count, min_count=max(12, count // 2))
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values <= values[-1]))


def _breakout_last(values: np.ndarray, count: int) -> float:
    values = _finite(values)
    if len(values) <= 12:
        return float("nan")
    prior = values[-count - 1 : -1]
    if len(prior) < 12:
        return float("nan")
    if values[-1] > np.max(prior):
        return 1.0
    if values[-1] < np.min(prior):
        return -1.0
    return 0.0


def _downside_semisharpe(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 12:
        return float("nan")
    downside = values[values < 0.0]
    if len(downside) < 2:
        return float("nan")
    return _safe_div(float(np.mean(values)), float(np.sqrt(np.mean(np.square(downside)))))


def _tail_loss_share(values: np.ndarray) -> float:
    values = _finite(values)
    if len(values) < 12:
        return float("nan")
    losses = -values[values < 0.0]
    if len(losses) < 2 or float(losses.sum()) <= EPS:
        return float("nan")
    cutoff = float(np.quantile(losses, 0.8))
    return float(losses[losses >= cutoff].sum() / losses.sum())


def _build_daily_features(ctx: FactorComputeContext) -> pd.DataFrame:
    out_index = _cached_output_index(ctx)
    if out_index.empty:
        return pd.DataFrame(
            columns=[*_DAILY_SIGNALS, *_DAILY_INTERNAL_DEFAULTS],
            index=out_index,
        )
    signal_dates = pd.to_datetime(out_index.get_level_values("dt"), errors="coerce").normalize().unique()
    if len(signal_dates) != 1 or pd.isna(signal_dates[0]):
        raise ValueError("factor mining daily catalogue requires one valid signal date per context")
    signal_date = pd.Timestamp(signal_dates[0])
    price = _strict_history_source(ctx, "market_cbond.daily_price", _DAILY_PRICE_COLUMNS, signal_date)
    twap = _strict_history_source(ctx, "market_cbond.daily_twap", _DAILY_TWAP_COLUMNS, signal_date)
    base = _strict_history_source(ctx, "market_cbond.daily_base", _DAILY_BASE_COLUMNS, signal_date)
    # The daily_price history is an independent market-calendar anchor for the
    # latest permitted prior session.  A base row for an older session may
    # still be useful to daily features, but it must not silently supply a
    # stale stock mapping to a current-day stock/bond path factor.
    strict_mapping_date = (
        pd.Timestamp(price["trade_date"].max()).normalize()
        if not price.empty and price["trade_date"].notna().any()
        else None
    )
    price_groups = {str(code): frame for code, frame in price.groupby("code", sort=False)}
    twap_groups = {str(code): frame for code, frame in twap.groupby("code", sort=False)}
    base_groups = {str(code): frame for code, frame in base.groupby("code", sort=False)}
    rows: list[dict[str, object]] = []
    for dt, code in out_index:
        market_code = _canonical_market_code(pd.Series([code])).iloc[0]
        row: dict[str, object] = {"dt": dt, "code": code}
        row.update(_daily_row(
            price_groups.get(market_code, pd.DataFrame(columns=["trade_date", "code", *_DAILY_PRICE_COLUMNS])),
            twap_groups.get(market_code, pd.DataFrame(columns=["trade_date", "code", *_DAILY_TWAP_COLUMNS])),
            base_groups.get(market_code, pd.DataFrame(columns=["trade_date", "code", *_DAILY_BASE_COLUMNS])),
            strict_mapping_date=strict_mapping_date,
        ))
        rows.append(row)
    out = pd.DataFrame(rows).set_index(["dt", "code"]).sort_index()
    # A genuine cross-sectional residual uses only prior-day structural values
    # available to every bond on this score day.
    if "__base_premium_raw" in out.columns:
        median = out.groupby(level="dt")["__base_premium_raw"].transform("median")
        out["base_premium_crosssection_residual"] = out["__base_premium_raw"] - median
    return out.replace([np.inf, -np.inf], np.nan)


def _daily_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    return _cached(ctx, f"factor_mining_daily_features:{CATALOG_VERSION}", lambda: _build_daily_features(ctx))


@FactorRegistry.register("factor_mining_daily_catalog_v1")
class FactorMiningDailyCatalogV1(Factor):
    """Research-only historical daily price, TWAP, and structural-state catalogue."""

    name = "factor_mining_daily_catalog_v1"
    kernel_name = "factor_mining_daily_catalog_v1"

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        return _daily_requirements(params)

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _assert_catalog_request(ctx, self.kernel_name)
        out = _output_for_signal(ctx, kernel=self.kernel_name, feature_key=entry.signal)
        out.name = self.output_name(entry.signal)
        return out


_CROSS_SIGNALS = tuple(entry.signal for entry in _CROSS_ENTRIES)


def _stock_bare(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) < 6 else text


def _panel_path_for_cross(
    panel: pd.DataFrame,
    *,
    signal_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = _signal_day_panel(panel, signal_date=signal_date).reset_index().copy()
    _require_columns(frame, ("trade_time", "last", "volume", "amount", "num_trades"), owner="factor_mining_cross_asset_catalog_v1")
    score_day = pd.Timestamp(signal_date).normalize() if signal_date is not None else _signal_date_from_panel(panel)
    if score_day is None:
        return frame.iloc[0:0]
    frame = _strict_score_day_snapshots(frame, signal_date=score_day)
    return frame.sort_values(["dt", "code", "trade_time"], kind="mergesort")


def _cross_path_metrics(bond: pd.DataFrame, stock: pd.DataFrame) -> dict[str, float]:
    empty = {
        "cross_stock_early_bond_late_response": float("nan"),
        "cross_bond_early_stock_late_response": float("nan"),
        "cross_lagged_return_corr": float("nan"),
        "cross_stock_impulse_bond_decay": float("nan"),
        "cross_bond_impulse_stock_decay": float("nan"),
        "cross_response_asym_up": float("nan"),
        "cross_response_asym_down": float("nan"),
        "cross_intraday_beta_resid": float("nan"),
        "cross_tail_beta_resid": float("nan"),
        "cross_rolling_cov_ratio": float("nan"),
        "cross_residual_vol_ratio": float("nan"),
        "cross_residual_drawdown": float("nan"),
        "cross_time_to_extreme_gap": float("nan"),
        "cross_time_of_high_gap": float("nan"),
        "cross_time_of_low_gap": float("nan"),
        "cross_turning_point_mismatch": float("nan"),
        "cross_drawdown_timing_gap": float("nan"),
        "cross_recovery_timing_gap": float("nan"),
        "cross_stock_flow_to_bond_flow_beta": float("nan"),
        "cross_bond_flow_to_stock_flow_beta": float("nan"),
        "cross_flow_lead_lag_corr": float("nan"),
        "cross_flow_shock_pass_through": float("nan"),
        "cross_trade_size_transmission": float("nan"),
        "cross_liquidity_drought_transmission": float("nan"),
        "cross_flow_return_elasticity_gap": float("nan"),
    }
    if bond.empty or stock.empty:
        return empty
    b = bond[["trade_time", "last", "volume", "amount", "num_trades"]].copy()
    s = stock[["trade_time", "last", "volume", "amount", "num_trades"]].copy().rename(
        columns={
            "last": "stock_last",
            "volume": "stock_volume",
            "amount": "stock_amount",
            "num_trades": "stock_num_trades",
        }
    )
    b = b.sort_values("trade_time", kind="mergesort")
    s = s.sort_values("trade_time", kind="mergesort")
    aligned = pd.merge_asof(
        b,
        s,
        on="trade_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=90),
    )
    aligned = aligned.dropna(subset=["last", "stock_last"]).reset_index(drop=True)
    if len(aligned) < 8:
        return empty
    bp = pd.to_numeric(aligned["last"], errors="coerce").to_numpy(dtype="float64")
    sp = pd.to_numeric(aligned["stock_last"], errors="coerce").to_numpy(dtype="float64")
    if not ((bp > 0.0).all() and (sp > 0.0).all()):
        return empty
    br = np.diff(bp) / bp[:-1]
    sr = np.diff(sp) / sp[:-1]
    half = max(2, len(bp) // 2)
    b_early = _safe_div(float(bp[half - 1] - bp[0]), float(bp[0]))
    s_early = _safe_div(float(sp[half - 1] - sp[0]), float(sp[0]))
    b_late = _safe_div(float(bp[-1] - bp[half - 1]), float(bp[half - 1]))
    s_late = _safe_div(float(sp[-1] - sp[half - 1]), float(sp[half - 1]))
    beta = _safe_div(float(np.cov(br, sr, ddof=1)[0, 1]), float(np.var(sr, ddof=1))) if len(sr) >= 3 else float("nan")
    b_full = _safe_div(float(bp[-1] - bp[0]), float(bp[0]))
    s_full = _safe_div(float(sp[-1] - sp[0]), float(sp[0]))
    tail_start = max(2, int(len(br) * 0.7))
    beta_tail = _safe_div(
        float(np.cov(br[tail_start:], sr[tail_start:], ddof=1)[0, 1]), float(np.var(sr[tail_start:], ddof=1))
    ) if len(sr[tail_start:]) >= 3 else float("nan")
    residual_path = br - beta * sr if np.isfinite(beta) else np.full_like(br, np.nan)
    b_wealth = bp / bp[0]
    s_wealth = sp / sp[0]
    b_draw = b_wealth / np.maximum.accumulate(b_wealth) - 1.0
    s_draw = s_wealth / np.maximum.accumulate(s_wealth) - 1.0
    b_rec = b_wealth / np.minimum.accumulate(b_wealth) - 1.0
    s_rec = s_wealth / np.minimum.accumulate(s_wealth) - 1.0
    bvol = _incremental(pd.to_numeric(aligned["volume"], errors="coerce").to_numpy(dtype="float64"))
    svol = _incremental(pd.to_numeric(aligned["stock_volume"], errors="coerce").to_numpy(dtype="float64"))
    bam = _incremental(pd.to_numeric(aligned["amount"], errors="coerce").to_numpy(dtype="float64"))
    sam = _incremental(pd.to_numeric(aligned["stock_amount"], errors="coerce").to_numpy(dtype="float64"))
    bdeals = _incremental(pd.to_numeric(aligned["num_trades"], errors="coerce").to_numpy(dtype="float64"))
    sdeals = _incremental(pd.to_numeric(aligned["stock_num_trades"], errors="coerce").to_numpy(dtype="float64"))
    flow_corr = _safe_corr(bvol, svol) if bvol is not None and svol is not None else float("nan")
    flow_lag = _safe_corr(bvol[1:], svol[:-1]) if bvol is not None and svol is not None and len(bvol) >= 4 else float("nan")
    beta_sb = _safe_div(float(np.cov(bvol, svol, ddof=1)[0, 1]), float(np.var(svol, ddof=1))) if bvol is not None and svol is not None and len(bvol) >= 3 else float("nan")
    beta_bs = _safe_div(float(np.cov(bvol, svol, ddof=1)[0, 1]), float(np.var(bvol, ddof=1))) if bvol is not None and svol is not None and len(bvol) >= 3 else float("nan")
    impact_b = _safe_corr(bam, br) if bam is not None else float("nan")
    impact_s = _safe_corr(sam, sr) if sam is not None else float("nan")
    empty.update(
        {
            "cross_stock_early_bond_late_response": s_early * b_late,
            "cross_bond_early_stock_late_response": b_early * s_late,
            "cross_lagged_return_corr": _safe_corr(br[1:], sr[:-1]),
            "cross_stock_impulse_bond_decay": s_early * (b_late - b_early),
            "cross_bond_impulse_stock_decay": b_early * (s_late - s_early),
            "cross_response_asym_up": b_late - s_late if s_early > 0.0 else float("nan"),
            "cross_response_asym_down": b_late - s_late if s_early < 0.0 else float("nan"),
            "cross_intraday_beta_resid": b_full - beta * s_full if np.isfinite(beta) else float("nan"),
            "cross_tail_beta_resid": _safe_div(float(bp[-1] - bp[-tail_start - 1]), float(bp[-tail_start - 1])) - beta_tail * _safe_div(float(sp[-1] - sp[-tail_start - 1]), float(sp[-tail_start - 1])) if np.isfinite(beta_tail) else float("nan"),
            "cross_rolling_cov_ratio": beta,
            "cross_residual_vol_ratio": _safe_div(float(np.nanstd(residual_path, ddof=1)), float(np.nanstd(br, ddof=1))) if np.isfinite(residual_path).sum() >= 3 else float("nan"),
            "cross_residual_drawdown": float(np.nanmin(residual_path)) if np.isfinite(residual_path).any() else float("nan"),
            "cross_time_to_extreme_gap": _safe_div(float(np.argmax(np.abs(bp / bp[0] - 1.0)) - np.argmax(np.abs(sp / sp[0] - 1.0))), float(max(1, len(bp) - 1))),
            "cross_time_of_high_gap": _safe_div(float(np.argmax(bp) - np.argmax(sp)), float(max(1, len(bp) - 1))),
            "cross_time_of_low_gap": _safe_div(float(np.argmin(bp) - np.argmin(sp)), float(max(1, len(bp) - 1))),
            "cross_turning_point_mismatch": _safe_div(float(np.argmax(np.abs(br)) - np.argmax(np.abs(sr))), float(max(1, len(br) - 1))),
            "cross_drawdown_timing_gap": _safe_div(float(np.argmin(b_draw) - np.argmin(s_draw)), float(max(1, len(b_draw) - 1))),
            "cross_recovery_timing_gap": _safe_div(float(np.argmax(b_rec) - np.argmax(s_rec)), float(max(1, len(b_rec) - 1))),
            "cross_stock_flow_to_bond_flow_beta": beta_sb,
            "cross_bond_flow_to_stock_flow_beta": beta_bs,
            "cross_flow_lead_lag_corr": flow_lag,
            "cross_flow_shock_pass_through": flow_corr,
            "cross_trade_size_transmission": _safe_corr(
                np.where(bdeals > 0.0, bam / bdeals, np.nan), np.where(sdeals > 0.0, sam / sdeals, np.nan)
            ) if bdeals is not None and sdeals is not None and bam is not None and sam is not None else float("nan"),
            "cross_liquidity_drought_transmission": _safe_corr(np.log(np.where(bam > 0.0, bam, np.nan)), np.log(np.where(sam > 0.0, sam, np.nan))) if bam is not None and sam is not None else float("nan"),
            "cross_flow_return_elasticity_gap": impact_b - impact_s if np.isfinite(impact_b) and np.isfinite(impact_s) else float("nan"),
        }
    )
    return empty


def _build_cross_features(ctx: FactorComputeContext) -> pd.DataFrame:
    if ctx.stock_panel is None or ctx.stock_panel.empty:
        raise RuntimeError("factor_mining_cross_asset_catalog_v1 requires a non-empty stock_panel")
    signal_date = _signal_date_from_panel(ctx.panel)
    bond_features = _intraday_feature_frame(ctx).reset_index()
    stock_features = _cached(
        ctx,
        f"factor_mining_stock_intraday_features:{CATALOG_VERSION}:{signal_date}",
        lambda: _build_intraday_features(
            ctx.stock_panel if ctx.stock_panel is not None else pd.DataFrame(),
            signal_date=signal_date,
        ),
    ).reset_index()
    daily = _daily_feature_frame(ctx).reset_index()
    mapping = daily[["dt", "code", "__base_stock_code", "__base_premium_raw"]].copy()
    mapping["bond_bare"] = _bare_code(mapping["code"])
    mapping["stock_bare"] = mapping["__base_stock_code"].map(_stock_bare)
    stock_features["stock_bare"] = _bare_code(stock_features["code"])
    merged = bond_features.merge(mapping, on=["dt", "code"], how="left")
    merged = merged.merge(
        stock_features.add_prefix("stock_").rename(columns={"stock_dt": "dt", "stock_stock_bare": "stock_bare"}),
        on=["dt", "stock_bare"],
        how="left",
    )
    bond_paths = _panel_path_for_cross(ctx.panel, signal_date=signal_date)
    stock_paths = _panel_path_for_cross(ctx.stock_panel, signal_date=signal_date)
    bond_path_groups = {(dt, str(code)): group for (dt, code), group in bond_paths.groupby(["dt", "code"], sort=False)}
    stock_path_groups = {(dt, _bare_code(pd.Series([code])).iloc[0]): group for (dt, code), group in stock_paths.groupby(["dt", "code"], sort=False)}
    rows: list[dict[str, object]] = []
    for record in merged.to_dict("records"):
        row: dict[str, object] = {"dt": record["dt"], "code": record["code"]}
        row.update({signal: float("nan") for signal in _CROSS_SIGNALS})
        stock_bare = _stock_bare(record.get("stock_bare"))
        bond_path = bond_path_groups.get((record["dt"], str(record["code"])), pd.DataFrame())
        stock_path = stock_path_groups.get((record["dt"], stock_bare), pd.DataFrame())
        pair = _cross_path_metrics(bond_path, stock_path)
        row.update(pair)
        b = record
        def get(name: str) -> float:
            value = b.get(name)
            return float(value) if value is not None and pd.notna(value) else float("nan")
        row.update(
            {
                "cross_residual_efficiency_gap": get("intra_path_efficiency") - get("stock_intra_path_efficiency"),
                "cross_bond_stock_return_gap": get("intra_full_return") - get("stock_intra_full_return"),
                "cross_path_entropy_gap": get("intra_price_entropy") - get("stock_intra_price_entropy"),
                "cross_mid_gap": get("__intra_last_to_mid") - get("stock___intra_last_to_mid"),
                "cross_relative_spread_gap": _safe_div(get("book_spread_last"), get("stock_book_spread_last")),
                "cross_depth_imbalance_gap": get("book_imbalance_l5_last") - get("stock_book_imbalance_l5_last"),
                "cross_microprice_bias_gap": get("book_microprice_bias_last") - get("stock_book_microprice_bias_last"),
                "cross_quote_update_intensity_gap": get("book_quote_staleness") - get("stock_book_quote_staleness"),
                "cross_book_slope_gap": get("book_depth_slope") - get("stock_book_depth_slope"),
                "cross_spread_resilience_gap": get("book_liquidity_recovery") - get("stock_book_liquidity_recovery"),
                "cross_stock_up_limit_response": -get("stock___intra_limit_up_distance") * get("intra_full_return"),
                "cross_stock_down_limit_response": -get("stock___intra_limit_down_distance") * get("intra_full_return"),
                "cross_limit_distance_gap": get("__intra_limit_up_distance") - get("stock___intra_limit_up_distance"),
                "cross_limit_proximity_change_gap": get("__intra_limit_down_distance") - get("stock___intra_limit_down_distance"),
                "cross_one_side_limit_indicator": float(
                    (get("stock___intra_limit_up_distance") < 0.002) != (get("stock___intra_limit_down_distance") < 0.002)
                ) if np.isfinite(get("stock___intra_limit_up_distance")) and np.isfinite(get("stock___intra_limit_down_distance")) else float("nan"),
                "cross_limit_proximity_vol_scaled": _safe_div(
                    min(abs(get("stock___intra_limit_up_distance")), abs(get("stock___intra_limit_down_distance"))),
                    abs(get("stock_intra_full_return")),
                ),
                "cross_limit_stress_premium": min(abs(get("stock___intra_limit_up_distance")), abs(get("stock___intra_limit_down_distance"))) * get("__base_premium_raw"),
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=_CROSS_SIGNALS,
            index=_cached_output_index(ctx),
        )
    out = pd.DataFrame(rows).set_index(["dt", "code"]).sort_index()
    return out.replace([np.inf, -np.inf], np.nan)


def _cross_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    return _cached(ctx, f"factor_mining_cross_features:{CATALOG_VERSION}", lambda: _build_cross_features(ctx))


@FactorRegistry.register("factor_mining_cross_asset_catalog_v1")
class FactorMiningCrossAssetCatalogV1(Factor):
    """Research-only T-1-mapped stock/bond path, book, and flow catalogue."""

    name = "factor_mining_cross_asset_catalog_v1"
    kernel_name = "factor_mining_cross_asset_catalog_v1"
    requires_stock_panel = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        return _daily_requirements(params)

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _assert_catalog_request(ctx, self.kernel_name)
        out = _output_for_signal(ctx, kernel=self.kernel_name, feature_key=entry.signal)
        out.name = self.output_name(entry.signal)
        return out


_HYBRID_SIGNALS = tuple(entry.signal for entry in _HYBRID_ENTRIES)


def _as_float(record: dict[str, object], name: str) -> float:
    value = record.get(name)
    return float(value) if value is not None and pd.notna(value) else float("nan")


def _build_hybrid_features(ctx: FactorComputeContext) -> pd.DataFrame:
    """Join current T1430 state with strictly pre-signal historical state.

    The daily and cross frames have already applied their own point-in-time
    constraints.  This builder only combines their current-row outputs; it
    never reads source data or shifts a completed score-day field into history.
    """

    out_index = _cached_output_index(ctx)
    intraday = _intraday_feature_frame(ctx).reindex(out_index)
    daily = _daily_feature_frame(ctx).reindex(out_index)
    cross = _cross_feature_frame(ctx).reindex(out_index)
    joined = pd.concat([intraday, daily, cross], axis=1)
    rows: list[dict[str, object]] = []
    for (dt, code), record in joined.iterrows():
        values = record.to_dict()
        get = lambda name: _as_float(values, name)
        current_return = get("intra_full_return")
        current_range = get("__intra_range_ratio")
        current_amount = get("__intra_amount_total")
        current_flow = get("intra_amount_accel_15m")
        amount_ratio = _safe_div(current_amount, get("__base_cb_amount"))
        quote_pressure = get("book_imbalance_l5_last")
        spread = get("book_spread_last")
        tail_return = get("intra_tail_return_15m")
        close_position = get("intra_close_position")
        liquidity_recovery = get("book_liquidity_recovery")
        result: dict[str, object] = {"dt": dt, "code": code}
        result.update(
            {
                "hybrid_current_return_vs_hist_intraday": current_return - get("__daily_intraday_mean20"),
                "hybrid_current_range_vs_hist_range": current_range - get("__daily_range_mean20"),
                "hybrid_current_flow_vs_hist_amount": amount_ratio - 1.0 if np.isfinite(amount_ratio) else float("nan"),
                "hybrid_current_spread_vs_hist_illiquidity": spread * get("__daily_amihud_z20"),
                "hybrid_current_tail_vs_hist_tail": tail_return - get("__late_mean20"),
                "hybrid_current_position_vs_hist_premium": close_position * get("__base_premium_z"),
                "hybrid_current_stock_response_vs_hist_beta": get("cross_rolling_cov_ratio") - get("__base_stock_bond_beta20"),
                "hybrid_flow_x_premium_level": amount_ratio * get("__base_premium_z"),
                "hybrid_flow_x_moneyness": current_flow * get("__base_moneyness"),
                "hybrid_flow_x_bond_floor_distance": current_flow * get("__base_floor_distance"),
                "hybrid_flow_x_duration": current_flow * get("__base_duration"),
                "hybrid_quote_pressure_x_stockvol": quote_pressure * get("__base_stockvol"),
                "hybrid_return_x_redemption_state": current_return * get("__base_trigger_progress"),
                "hybrid_spread_resilience_x_remain_size": liquidity_recovery * np.log1p(get("__base_remain_size"))
                if get("__base_remain_size") >= 0.0
                else float("nan"),
                "hybrid_pre1430_move_vs_hist_late_beta": current_return - get("__late_mean20"),
                "hybrid_current_flow_vs_hist_overnight_response": amount_ratio * get("__overnight_mean20"),
                "hybrid_quote_pressure_vs_late_premium": quote_pressure * get("late_execution_premium"),
                "hybrid_stock_lead_vs_prior_premium": get("cross_stock_early_bond_late_response") * get("__base_premium_z"),
                "hybrid_current_range_vs_hist_twap_curve": current_range * get("__twap_curve_mean20"),
                "hybrid_current_momentum_vs_call_state": current_return * get("__base_call_state"),
                "hybrid_current_illiquidity_vs_prior_turnover": get("intra_impact_per_amount") * get("__base_turnover"),
            }
        )
        rows.append(result)
    if not rows:
        return pd.DataFrame(
            columns=_HYBRID_SIGNALS,
            index=pd.MultiIndex.from_arrays([[], []], names=["dt", "code"]),
        )
    return pd.DataFrame(rows).set_index(["dt", "code"]).sort_index().replace([np.inf, -np.inf], np.nan)


def _hybrid_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    return _cached(ctx, f"factor_mining_hybrid_features:{CATALOG_VERSION}", lambda: _build_hybrid_features(ctx))


@FactorRegistry.register("factor_mining_hybrid_catalog_v1")
class FactorMiningHybridCatalogV1(Factor):
    """Research-only current-state interactions with T-1 historical state."""

    name = "factor_mining_hybrid_catalog_v1"
    kernel_name = "factor_mining_hybrid_catalog_v1"
    requires_stock_panel = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        return _daily_requirements(params)

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _assert_catalog_request(ctx, self.kernel_name)
        out = _output_for_signal(ctx, kernel=self.kernel_name, feature_key=entry.signal)
        out.name = self.output_name(entry.signal)
        return out
