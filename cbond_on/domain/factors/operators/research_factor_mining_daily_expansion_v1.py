"""Research-only strict T-1 daily and structural factor expansion.

This module is deliberately not imported by defs.__init__.  It is an explicit
import research catalogue and never reads files, databases, labels, scores,
pool or mask data, model outputs, results, or live artefacts.

The only calculated inputs are declared DataHub daily fields from
market_cbond.daily_price, market_cbond.daily_twap, and
market_cbond.daily_base.  The context loader can include the score-day file,
so every source is independently restricted to trade_date < score date here.
Missing sources/columns and duplicate strict-prior observations are errors;
missing numeric observations remain missing and are never zero-filled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    DailyFactorRequirement,
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)


KERNEL_NAME = "factor_mining_daily_expansion_v1"
CATALOG_VERSION = "20260803_daily_expansion_v1"
_LOOKBACK_DAYS = 65
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    """One research-only signal with a distinct family-level hypothesis."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis)
        for signal in signals
    )


_RETURN_REGIME_SIGNALS = (
    "dret_last_return",
    "dret_momentum_5",
    "dret_reversal_1_5",
    "dret_volatility_20",
    "dret_downside_share_20",
    "dret_range_z_20",
    "dret_close_location_last",
    "dret_gap_followthrough_last",
)
_LIQUIDITY_QUALITY_SIGNALS = (
    "dliq_log_amount_z20",
    "dliq_log_volume_z20",
    "dliq_log_deal_z20",
    "dliq_trade_size_z20",
    "dliq_amount_cv20",
    "dliq_amount_persistence20",
    "dliq_amihud_z20",
    "dliq_volume_return_corr20",
)
_TWAP_MEMORY_SIGNALS = (
    "dtwap_morning_slope20",
    "dtwap_lunch_gap20",
    "dtwap_afternoon_slope20",
    "dtwap_late_ramp20",
    "dtwap_execution_premium20",
    "dtwap_curve_curvature20",
    "dtwap_late_z20",
    "dtwap_overnight_tail_reversal20",
)
_PREMIUM_YIELD_SIGNALS = (
    "dstruct_premium_z20",
    "dstruct_premium_delta1",
    "dstruct_premium_acceleration",
    "dstruct_premium_volatility20",
    "dstruct_yield_spread_last",
    "dstruct_ytm_delta1",
    "dstruct_current_yield_z20",
    "dstruct_premium_yield_corr20",
)
_REDEMPTION_SIGNALS = (
    "dredemption_premium_z20",
    "dredemption_premium_delta1",
    "dredemption_premium_volatility20",
    "dredemption_floor_z20",
    "dredemption_floor_delta1",
    "dredemption_premium_percentile20",
    "dredemption_bondpremium_interaction",
    "dredemption_yield_per_premium",
)
_CROSS_SECTIONAL_SIGNALS = (
    "dcs_premium_residual",
    "dcs_ytm_residual",
    "dcs_current_yield_residual",
    "dcs_duration_residual",
    "dcs_convexity_residual",
    "dcs_log_remain_size_residual",
    "dcs_turnover_residual",
    "dcs_maturity_residual",
)
_DURATION_CONVEXITY_SIGNALS = (
    "dduration_level",
    "dduration_z20",
    "dduration_delta1",
    "dduration_modified_ratio",
    "dduration_convexity_per_duration",
    "dduration_convexity_z20",
    "dduration_yield_sensitivity",
    "dduration_term_gap",
)
_MATURITY_ROLLDOWN_SIGNALS = (
    "dmaturity_level",
    "dmaturity_z20",
    "dmaturity_delta1",
    "dmaturity_volatility20",
    "dmaturity_duration_ratio",
    "dmaturity_modified_ratio",
    "dmaturity_convexity_ratio",
    "dmaturity_yield_roll_ratio",
)


_CATALOG = (
    _entries(
        "daily_return_regime_transition",
        _RETURN_REGIME_SIGNALS,
        "Prior completed daily return, range, and gap regimes can transition before the next score day.",
    )
    + _entries(
        "daily_liquidity_quality_stability",
        _LIQUIDITY_QUALITY_SIGNALS,
        "Prior completed turnover quality and price-impact stability differ from raw price direction.",
    )
    + _entries(
        "daily_twap_session_memory",
        _TWAP_MEMORY_SIGNALS,
        "Historical morning, lunch, afternoon, and execution-window TWAP shapes summarize session structure.",
    )
    + _entries(
        "daily_premium_yield_curve",
        _PREMIUM_YIELD_SIGNALS,
        "Convertible-bond premium and yield states form a structural valuation regime without stock inputs.",
    )
    + _entries(
        "daily_redemption_hazard_surface",
        _REDEMPTION_SIGNALS,
        "Prior redemption-premium and floor states describe structural pressure without trigger fields.",
    )
    + _entries(
        "daily_cross_sectional_structural_residual",
        _CROSS_SECTIONAL_SIGNALS,
        "T-1 structural values are demeaned within the score-day output universe.",
    )
    + _entries(
        "daily_duration_convexity_structure",
        _DURATION_CONVEXITY_SIGNALS,
        "Duration, modified duration, convexity, and yield describe historical rate-risk shape.",
    )
    + _entries(
        "daily_maturity_roll_down_state",
        _MATURITY_ROLLDOWN_SIGNALS,
        "Time-to-maturity and its rate-risk relations capture roll-down state rather than a price path.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "daily_return_regime_transition": _RETURN_REGIME_SIGNALS,
    "daily_liquidity_quality_stability": _LIQUIDITY_QUALITY_SIGNALS,
    "daily_twap_session_memory": _TWAP_MEMORY_SIGNALS,
    "daily_premium_yield_curve": _PREMIUM_YIELD_SIGNALS,
    "daily_redemption_hazard_surface": _REDEMPTION_SIGNALS,
    "daily_cross_sectional_structural_residual": _CROSS_SECTIONAL_SIGNALS,
    "daily_duration_convexity_structure": _DURATION_CONVEXITY_SIGNALS,
    "daily_maturity_roll_down_state": _MATURITY_ROLLDOWN_SIGNALS,
}

# Every family uses one source.  Base fields intentionally exclude stock,
# trigger, and pure-debt-premium fields with known weak coverage.
_FAMILY_SOURCE_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "daily_return_regime_transition": (
        "market_cbond.daily_price",
        ("prev_close_price", "open_price", "high_price", "low_price", "close_price"),
    ),
    "daily_liquidity_quality_stability": (
        "market_cbond.daily_price",
        ("prev_close_price", "close_price", "volume", "amount", "deal"),
    ),
    "daily_twap_session_memory": (
        "market_cbond.daily_twap",
        (
            "twap_0930_0935",
            "twap_0935_1000",
            "twap_1100_1130",
            "twap_1300_1330",
            "twap_1400_1430",
            "twap_1430_1442",
            "twap_1442_1457",
        ),
    ),
    "daily_premium_yield_curve": (
        "market_cbond.daily_base",
        ("bond_prem_ratio", "ytm", "current_yield", "base_rate"),
    ),
    "daily_redemption_hazard_surface": (
        "market_cbond.daily_base",
        ("bond_prem_ratio", "ytm", "redemption_prem_ratio", "pure_redemption_value"),
    ),
    "daily_cross_sectional_structural_residual": (
        "market_cbond.daily_base",
        (
            "bond_prem_ratio",
            "ytm",
            "current_yield",
            "duration",
            "convexity",
            "remain_size",
            "turnover_rate",
            "year_to_mat",
        ),
    ),
    "daily_duration_convexity_structure": (
        "market_cbond.daily_base",
        ("duration", "modify_duration", "convexity", "year_to_mat", "current_yield"),
    ),
    "daily_maturity_roll_down_state": (
        "market_cbond.daily_base",
        ("year_to_mat", "duration", "modify_duration", "convexity", "ytm", "current_yield"),
    ),
}

_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def daily_expansion_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first daily expansion catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return daily_expansion_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirements_for_family(family: str) -> DailyFactorRequirement:
    source, fields = _FAMILY_SOURCE_FIELDS[family]
    return DailyFactorRequirement(
        source=source,
        columns=("exchange_code", *fields),
        lookback_days=_LOOKBACK_DAYS,
    )


def _all_requirements() -> list[DailyFactorRequirement]:
    grouped: dict[str, set[str]] = {}
    for source, fields in _FAMILY_SOURCE_FIELDS.values():
        grouped.setdefault(source, {"exchange_code"}).update(fields)
    return [
        DailyFactorRequirement(
            source=source,
            columns=tuple(sorted(columns)),
            lookback_days=_LOOKBACK_DAYS,
        )
        for source, columns in sorted(grouped.items())
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series) -> pd.Series:
    """Keep exchange suffixes so bare instrument-code collisions cannot merge."""

    codes = values.astype(str).str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    exchange = exchanges.astype(str).str.strip().str.upper().map(
        lambda value: _EXCHANGE_ALIASES.get(value, value)
    )
    has_suffix = codes.str.contains(r"\.(?:SH|SZ|BJ)$", regex=True, na=False)
    attach = (~has_suffix) & exchange.isin(_MARKET_EXCHANGES)
    out = codes.copy()
    out.loc[attach] = out.loc[attach] + "." + exchange.loc[attach]
    return out


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        score_day = pd.Timestamp(raw)
        if pd.isna(score_day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_day.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    """Use panel keys only as an output index, never a panel value as a signal."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:output_index"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached.index
    panel = ensure_panel_index(ctx.panel)
    score_day = _score_date_from_panel(panel)
    if score_day is None:
        index = _empty_index()
    else:
        dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
        keep = dates == score_day
        if not panel.empty and not bool(np.any(keep)):
            raise ValueError(
                f"{KERNEL_NAME} has no indexed panel rows for score date {score_day.date().isoformat()}"
            )
        selected = panel.loc[keep]
        keys = selected.index.droplevel("seq").unique()
        index = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"]).sort_values()
    built = pd.DataFrame(index=index)
    with ctx.cache_lock:
        previous = ctx.cache.get(cache_key)
        if isinstance(previous, pd.DataFrame):
            return previous.index
        ctx.cache[cache_key] = built
    return index


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required columns: {missing}")


def _strict_history_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Validate, normalize, and strictly cut one daily DataHub source."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), owner=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False),
            ["trade_date", "code"],
        ].head(3)
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _history_for_family(ctx: FactorComputeContext, family: str, score_date: pd.Timestamp) -> pd.DataFrame:
    source, fields = _FAMILY_SOURCE_FIELDS[family]
    cache_key = (
        f"{KERNEL_NAME}:{CATALOG_VERSION}:history:{source}:{score_date.date().isoformat()}:"
        + ",".join(fields)
    )
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _strict_history_source(ctx, source=source, fields=fields, score_date=score_date)
    with ctx.cache_lock:
        previous = ctx.cache.get(cache_key)
        if isinstance(previous, pd.DataFrame):
            return previous
        ctx.cache[cache_key] = built
    return built


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_div_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > _EPS)
    )
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return _safe_div_array(numerator, denominator) - 1.0


def _last(values: np.ndarray) -> float:
    if not len(values) or not np.isfinite(values[-1]):
        return float("nan")
    return float(values[-1])


def _tail(values: np.ndarray, count: int, *, min_count: int) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if len(valid) < min_count:
        return np.array([], dtype="float64")
    return valid[-count:]


def _mean(values: np.ndarray, count: int, *, min_count: int) -> float:
    tail = _tail(values, count, min_count=min_count)
    return float(tail.mean()) if len(tail) else float("nan")


def _std(values: np.ndarray, count: int, *, min_count: int) -> float:
    tail = _tail(values, count, min_count=min_count)
    return float(tail.std(ddof=1)) if len(tail) >= 2 else float("nan")


def _z_last(values: np.ndarray, count: int = 20, *, min_count: int = 12) -> float:
    latest = _last(values)
    tail = _tail(values, count, min_count=min_count)
    if not np.isfinite(latest) or len(tail) < 2:
        return float("nan")
    return _safe_div(latest - float(tail.mean()), float(tail.std(ddof=1)))


def _delta(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    later = values[-1]
    earlier = values[-1 - periods]
    if not (np.isfinite(later) and np.isfinite(earlier)):
        return float("nan")
    return float(later - earlier)


def _pct_change(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    return _safe_div(float(values[-1] - values[-1 - periods]), float(values[-1 - periods]))


def _autocorr(values: np.ndarray, count: int = 20, *, min_count: int = 12) -> float:
    tail = _tail(values, count, min_count=min_count)
    if len(tail) < 3 or float(np.std(tail[:-1])) <= _EPS or float(np.std(tail[1:])) <= _EPS:
        return float("nan")
    return float(np.corrcoef(tail[:-1], tail[1:])[0, 1])


def _trailing_corr(
    left: np.ndarray,
    right: np.ndarray,
    count: int = 20,
    *,
    min_count: int = 12,
) -> float:
    pair = pd.DataFrame({"left": left, "right": right}).dropna().tail(count)
    if len(pair) < min_count:
        return float("nan")
    a = pair["left"].to_numpy(dtype="float64")
    b = pair["right"].to_numpy(dtype="float64")
    if float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _percentile_last(values: np.ndarray, count: int = 20, *, min_count: int = 12) -> float:
    latest = _last(values)
    tail = _tail(values, count, min_count=min_count)
    if not np.isfinite(latest) or not len(tail):
        return float("nan")
    return float(np.mean(tail <= latest))


def _downside_share(values: np.ndarray) -> float:
    tail = _tail(values, 20, min_count=12)
    return float(np.mean(tail < 0.0)) if len(tail) else float("nan")


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _return_regime_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_RETURN_REGIME_SIGNALS)
    prev_close = _numeric(frame, "prev_close_price")
    open_px = _numeric(frame, "open_price")
    high = _numeric(frame, "high_price")
    low = _numeric(frame, "low_price")
    close = _numeric(frame, "close_price")
    ret = _ratio(close, prev_close)
    gap = _ratio(open_px, prev_close)
    intraday = _ratio(close, open_px)
    daily_range = _safe_div_array(high - low, prev_close)
    close_location = _safe_div_array(close - low, high - low)
    reversal = _last(ret) - _mean(ret[:-1], 5, min_count=4) if len(ret) > 1 else float("nan")
    gap_follow = _last(gap) * _last(intraday)
    out.update(
        {
            "dret_last_return": _last(ret),
            "dret_momentum_5": _pct_change(close, 5),
            "dret_reversal_1_5": reversal if np.isfinite(reversal) else float("nan"),
            "dret_volatility_20": _std(ret, 20, min_count=12),
            "dret_downside_share_20": _downside_share(ret),
            "dret_range_z_20": _z_last(daily_range, 20),
            "dret_close_location_last": _last(close_location),
            "dret_gap_followthrough_last": gap_follow if np.isfinite(gap_follow) else float("nan"),
        }
    )
    return out


def _positive_log(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values) & (values > 0.0)
    out[valid] = np.log(values[valid])
    return out


def _liquidity_quality_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LIQUIDITY_QUALITY_SIGNALS)
    amount = _numeric(frame, "amount")
    volume = _numeric(frame, "volume")
    deal = _numeric(frame, "deal")
    close = _numeric(frame, "close_price")
    prev_close = _numeric(frame, "prev_close_price")
    ret = _ratio(close, prev_close)
    log_amount = _positive_log(amount)
    log_volume = _positive_log(volume)
    log_deal = _positive_log(deal)
    trade_size = _safe_div_array(amount, deal)
    amihud = _safe_div_array(np.abs(ret), amount)
    out.update(
        {
            "dliq_log_amount_z20": _z_last(log_amount, 20),
            "dliq_log_volume_z20": _z_last(log_volume, 20),
            "dliq_log_deal_z20": _z_last(log_deal, 20),
            "dliq_trade_size_z20": _z_last(trade_size, 20),
            "dliq_amount_cv20": _safe_div(
                _std(log_amount, 20, min_count=12),
                _mean(log_amount, 20, min_count=12),
            ),
            "dliq_amount_persistence20": _autocorr(log_amount, 20),
            "dliq_amihud_z20": _z_last(amihud, 20),
            "dliq_volume_return_corr20": _trailing_corr(log_volume, ret, 20),
        }
    )
    return out


def _twap_session_memory_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_TWAP_MEMORY_SIGNALS)
    morning_open = _numeric(frame, "twap_0930_0935")
    morning_end = _numeric(frame, "twap_0935_1000")
    noon = _numeric(frame, "twap_1100_1130")
    afternoon_open = _numeric(frame, "twap_1300_1330")
    afternoon_end = _numeric(frame, "twap_1400_1430")
    late = _numeric(frame, "twap_1430_1442")
    execution = _numeric(frame, "twap_1442_1457")
    morning = _ratio(morning_end, morning_open)
    lunch = _ratio(afternoon_open, noon)
    afternoon = _ratio(afternoon_end, afternoon_open)
    late_ramp = _ratio(late, afternoon_end)
    execution_premium = _ratio(execution, late)
    curvature = morning + afternoon - 2.0 * late_ramp
    overnight = np.full(len(frame), np.nan, dtype="float64")
    if len(frame) >= 2:
        overnight[1:] = _ratio(morning_open[1:], late[:-1])
    prior_late_ramp = np.concatenate(([np.nan], late_ramp[:-1]))
    tail_reversal = -overnight * prior_late_ramp
    out.update(
        {
            "dtwap_morning_slope20": _mean(morning, 20, min_count=12),
            "dtwap_lunch_gap20": _mean(lunch, 20, min_count=12),
            "dtwap_afternoon_slope20": _mean(afternoon, 20, min_count=12),
            "dtwap_late_ramp20": _mean(late_ramp, 20, min_count=12),
            "dtwap_execution_premium20": _mean(execution_premium, 20, min_count=12),
            "dtwap_curve_curvature20": _mean(curvature, 20, min_count=12),
            "dtwap_late_z20": _z_last(late_ramp, 20),
            "dtwap_overnight_tail_reversal20": _mean(tail_reversal, 20, min_count=12),
        }
    )
    return out


def _premium_yield_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_PREMIUM_YIELD_SIGNALS)
    premium = _numeric(frame, "bond_prem_ratio")
    ytm = _numeric(frame, "ytm")
    current_yield = _numeric(frame, "current_yield")
    base_rate = _numeric(frame, "base_rate")
    delta1 = _delta(premium, 1)
    delta5 = _delta(premium, 5)
    acceleration = delta1 - delta5 if np.isfinite(delta1) and np.isfinite(delta5) else float("nan")
    out.update(
        {
            "dstruct_premium_z20": _z_last(premium, 20),
            "dstruct_premium_delta1": delta1,
            "dstruct_premium_acceleration": acceleration,
            "dstruct_premium_volatility20": _std(premium, 20, min_count=12),
            "dstruct_yield_spread_last": _last(current_yield - base_rate),
            "dstruct_ytm_delta1": _delta(ytm, 1),
            "dstruct_current_yield_z20": _z_last(current_yield, 20),
            "dstruct_premium_yield_corr20": _trailing_corr(premium, ytm, 20),
        }
    )
    return out


def _redemption_hazard_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_REDEMPTION_SIGNALS)
    bond_premium = _numeric(frame, "bond_prem_ratio")
    ytm = _numeric(frame, "ytm")
    redemption_premium = _numeric(frame, "redemption_prem_ratio")
    floor = _numeric(frame, "pure_redemption_value")
    interaction = _last(bond_premium) * _last(redemption_premium)
    out.update(
        {
            "dredemption_premium_z20": _z_last(redemption_premium, 20),
            "dredemption_premium_delta1": _delta(redemption_premium, 1),
            "dredemption_premium_volatility20": _std(redemption_premium, 20, min_count=12),
            "dredemption_floor_z20": _z_last(floor, 20),
            "dredemption_floor_delta1": _delta(floor, 1),
            "dredemption_premium_percentile20": _percentile_last(redemption_premium, 20),
            "dredemption_bondpremium_interaction": interaction if np.isfinite(interaction) else float("nan"),
            "dredemption_yield_per_premium": _safe_div(_last(ytm), _last(redemption_premium)),
        }
    )
    return out


def _duration_convexity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_DURATION_CONVEXITY_SIGNALS)
    duration = _numeric(frame, "duration")
    modified = _numeric(frame, "modify_duration")
    convexity = _numeric(frame, "convexity")
    maturity = _numeric(frame, "year_to_mat")
    current_yield = _numeric(frame, "current_yield")
    out.update(
        {
            "dduration_level": _last(duration),
            "dduration_z20": _z_last(duration, 20),
            "dduration_delta1": _delta(duration, 1),
            "dduration_modified_ratio": _safe_div(_last(modified), _last(duration)),
            "dduration_convexity_per_duration": _safe_div(_last(convexity), _last(duration)),
            "dduration_convexity_z20": _z_last(convexity, 20),
            "dduration_yield_sensitivity": _last(duration) * _last(current_yield),
            "dduration_term_gap": _last(maturity) - _last(duration),
        }
    )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _maturity_rolldown_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_MATURITY_ROLLDOWN_SIGNALS)
    maturity = _numeric(frame, "year_to_mat")
    duration = _numeric(frame, "duration")
    modified = _numeric(frame, "modify_duration")
    convexity = _numeric(frame, "convexity")
    ytm = _numeric(frame, "ytm")
    current_yield = _numeric(frame, "current_yield")
    out.update(
        {
            "dmaturity_level": _last(maturity),
            "dmaturity_z20": _z_last(maturity, 20),
            "dmaturity_delta1": _delta(maturity, 1),
            "dmaturity_volatility20": _std(maturity, 20, min_count=12),
            "dmaturity_duration_ratio": _safe_div(_last(duration), _last(maturity)),
            "dmaturity_modified_ratio": _safe_div(_last(modified), _last(maturity)),
            "dmaturity_convexity_ratio": _safe_div(_last(convexity), _last(maturity)),
            "dmaturity_yield_roll_ratio": _safe_div(_last(ytm) - _last(current_yield), _last(maturity)),
        }
    )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "daily_return_regime_transition": _return_regime_metrics,
    "daily_liquidity_quality_stability": _liquidity_quality_metrics,
    "daily_twap_session_memory": _twap_session_memory_metrics,
    "daily_premium_yield_curve": _premium_yield_metrics,
    "daily_redemption_hazard_surface": _redemption_hazard_metrics,
    "daily_duration_convexity_structure": _duration_convexity_metrics,
    "daily_maturity_roll_down_state": _maturity_rolldown_metrics,
}


def _cross_sectional_feature_frame(out_index: pd.MultiIndex, history: pd.DataFrame) -> pd.DataFrame:
    """Create residuals only across the output-code universe."""

    signal_to_field = {
        "dcs_premium_residual": "bond_prem_ratio",
        "dcs_ytm_residual": "ytm",
        "dcs_current_yield_residual": "current_yield",
        "dcs_duration_residual": "duration",
        "dcs_convexity_residual": "convexity",
        "dcs_log_remain_size_residual": "remain_size",
        "dcs_turnover_residual": "turnover_rate",
        "dcs_maturity_residual": "year_to_mat",
    }
    groups = {str(code): group for code, group in history.groupby("code", sort=False)}
    values: dict[str, list[float]] = {signal: [] for signal in _CROSS_SECTIONAL_SIGNALS}
    for _, code in out_index:
        market_code = _canonical_market_code(pd.Series([code]), pd.Series([""])).iloc[0]
        group = groups.get(market_code)
        for signal, field in signal_to_field.items():
            raw = _last(_numeric(group, field)) if group is not None else float("nan")
            if signal == "dcs_log_remain_size_residual":
                raw = float(np.log(raw)) if np.isfinite(raw) and raw > 0.0 else float("nan")
            values[signal].append(raw)
    out = pd.DataFrame(values, index=out_index, dtype="float64")
    for signal in _CROSS_SECTIONAL_SIGNALS:
        median = out[signal].median(skipna=True)
        out[signal] = out[signal] - median if np.isfinite(median) else np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache one whole family so eight concrete specs share one scan."""

    if family not in _FAMILY_SIGNALS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    out_index = _output_index(ctx)
    score_date = _score_date_from_panel(ctx.panel)
    if score_date is None:
        built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
    else:
        history = _history_for_family(ctx, family, score_date)
        if family == "daily_cross_sectional_structural_residual":
            built = _cross_sectional_feature_frame(out_index, history)
        elif out_index.empty:
            built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
        else:
            _, fields = _FAMILY_SOURCE_FIELDS[family]
            groups = {str(code): group for code, group in history.groupby("code", sort=False)}
            calculator = _FAMILY_CALCULATORS[family]
            rows: list[dict[str, object]] = []
            for dt, code in out_index:
                market_code = _canonical_market_code(pd.Series([code]), pd.Series([""])).iloc[0]
                group = groups.get(
                    market_code,
                    pd.DataFrame(columns=["trade_date", "code", "exchange_code", *fields]),
                )
                row: dict[str, object] = {"dt": dt, "code": code}
                row.update(calculator(group))
                rows.append(row)
            built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_FAMILY_SIGNALS[family])]
            built = built.sort_index().replace([np.inf, -np.inf], np.nan)
    with ctx.cache_lock:
        previous = ctx.cache.get(cache_key)
        if isinstance(previous, pd.DataFrame):
            return previous
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyExpansionV1(Factor):
    """Research-only strict T-1 daily and structural candidate kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        values = dict(params or {})
        if not str(values.get("signal", "")).strip():
            return _all_requirements()
        return [_requirements_for_family(_requested_entry(values).family)]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)
