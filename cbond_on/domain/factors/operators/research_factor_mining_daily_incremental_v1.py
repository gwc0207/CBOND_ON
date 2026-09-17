"""Research-only strict T-1 daily structural factor families.

The module is intentionally not imported by defs.__init__.  It is an
explicit-import scratch catalogue and cannot read files, databases, labels,
pools, masks, scores, results, raw snapshots, or live artefacts.  All
calculation inputs come from declared DataHub daily sources only.

The daily context loader may include a score-date parquet.  Every source is
therefore validated and cut to trade_date strictly before the score date here.
Missing sources/columns, duplicate strict-prior rows, an insufficient
cross-section, or a rank-deficient robust ridge all fail closed.  Numeric data
is never zero-filled or forward-filled.
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


KERNEL_NAME = "factor_mining_daily_incremental_v1"
CATALOG_VERSION = "20260803_daily_incremental_v1"
_LOOKBACK_DAYS = 75
_MIN_CROSS_SECTION = 30
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis)
        for signal in signals
    )


_CONDITIONAL_PREMIUM_SIGNALS = (
    "cprr_premium_level_residual",
    "cprr_premium_delta1_residual",
    "cprr_premium_delta5_residual",
    "cprr_premium_acceleration_residual",
    "cprr_premium_volatility_residual",
)
_PREMIUM_YIELD_REGIME_SIGNALS = (
    "pyreg_beta_60",
    "pyreg_beta_shift_20_60",
    "pyreg_latest_innovation_60",
    "pyreg_residual_vol_ratio_20_60",
    "pyreg_current_yield_ytm_basis_z20",
)
_LAGGED_LIQUIDITY_SIGNALS = (
    "llpt_signed_flow_lag_beta20",
    "llpt_signed_flow_lag_beta60",
    "llpt_signed_flow_beta_shift",
    "llpt_amihud_to_absret_beta20",
    "llpt_highflow_next_return_spread20",
    "llpt_impact_memory20",
)
_DRAWDOWN_RECOVERY_SIGNALS = (
    "drt_drawdown_from_high20",
    "drt_rebound_from_low20",
    "drt_peak_recency20",
    "drt_trough_recency20",
    "drt_recovery_path_efficiency20",
    "drt_liquidity_recovery_ratio20",
)
_TWAP_TRANSITION_SIGNALS = (
    "tpt_state_entropy60",
    "tpt_morning_late_alignment60",
    "tpt_late_overnight_alignment60",
    "tpt_latest_transition_surprise",
    "tpt_intraday_turn_rate60",
    "tpt_execution_late_alignment60",
)


_CATALOG = (
    _entries(
        "conditional_premium_repricing_residual",
        _CONDITIONAL_PREMIUM_SIGNALS,
        "Cross-sectional premium level and repricing residuals are conditioned on structural state through robust rank ridge.",
    )
    + _entries(
        "premium_yield_regime_break",
        _PREMIUM_YIELD_REGIME_SIGNALS,
        "Short-versus-long premium-yield transmission and innovation describe a structural regime break rather than a raw premium move.",
    )
    + _entries(
        "lagged_liquidity_price_transmission",
        _LAGGED_LIQUIDITY_SIGNALS,
        "Completed prior-day flow and impact can transmit to the following completed daily return.",
    )
    + _entries(
        "drawdown_recovery_timing",
        _DRAWDOWN_RECOVERY_SIGNALS,
        "Peak/trough recency and recovery efficiency retain daily path timing absent from endpoint statistics.",
    )
    + _entries(
        "twap_phase_transition_entropy",
        _TWAP_TRANSITION_SIGNALS,
        "Multi-phase daily TWAP state transitions capture temporal structure rather than a static session average.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "conditional_premium_repricing_residual": _CONDITIONAL_PREMIUM_SIGNALS,
    "premium_yield_regime_break": _PREMIUM_YIELD_REGIME_SIGNALS,
    "lagged_liquidity_price_transmission": _LAGGED_LIQUIDITY_SIGNALS,
    "drawdown_recovery_timing": _DRAWDOWN_RECOVERY_SIGNALS,
    "twap_phase_transition_entropy": _TWAP_TRANSITION_SIGNALS,
}

_FAMILY_SOURCE_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "conditional_premium_repricing_residual": (
        "market_cbond.daily_base",
        ("bond_prem_ratio", "ytm", "duration", "year_to_mat", "remain_size"),
    ),
    "premium_yield_regime_break": (
        "market_cbond.daily_base",
        ("bond_prem_ratio", "ytm", "current_yield"),
    ),
    "lagged_liquidity_price_transmission": (
        "market_cbond.daily_price",
        ("prev_close_price", "close_price", "amount"),
    ),
    "drawdown_recovery_timing": (
        "market_cbond.daily_price",
        ("close_price", "high_price", "low_price", "amount"),
    ),
    "twap_phase_transition_entropy": (
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


def daily_incremental_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first incremental research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return daily_incremental_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirement_for_family(family: str) -> DailyFactorRequirement:
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
        day = pd.Timestamp(raw)
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


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    """Use only output keys from the panel; no panel value is a factor input."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:output_index"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached.index

    panel = ensure_panel_index(ctx.panel)
    score_date = _score_date_from_panel(panel)
    if score_date is None:
        index = _empty_index()
    else:
        dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
        keep = dates == score_date
        if not panel.empty and not bool(np.any(keep)):
            raise ValueError(
                f"{KERNEL_NAME} has no indexed panel rows for score date {score_date.date().isoformat()}"
            )
        selected = panel.loc[keep]
        keys = selected.index.droplevel("seq").unique()
        index = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"]).sort_values()

    frame = pd.DataFrame(index=index)
    with ctx.cache_lock:
        previous = ctx.cache.get(cache_key)
        if isinstance(previous, pd.DataFrame):
            return previous.index
        ctx.cache[cache_key] = frame
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


def _history_for_family(
    ctx: FactorComputeContext,
    family: str,
    score_date: pd.Timestamp,
) -> pd.DataFrame:
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


def _history_groups_at_anchor(history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reject stale per-code rows when a common latest daily source date exists."""

    if history.empty:
        return {}
    anchor = pd.Timestamp(history["trade_date"].max()).normalize()
    groups: dict[str, pd.DataFrame] = {}
    for code, group in history.groupby("code", sort=False):
        ordered = group.sort_values("trade_date", kind="mergesort")
        if not ordered.empty and pd.Timestamp(ordered["trade_date"].iloc[-1]).normalize() == anchor:
            groups[str(code)] = ordered
    return groups


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_div_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype="float64")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > _EPS)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return _safe_div_array(numerator, denominator) - 1.0


def _last(values: np.ndarray) -> float:
    if not len(values) or not np.isfinite(values[-1]):
        return float("nan")
    return float(values[-1])


def _delta(values: np.ndarray, periods: int) -> float:
    if len(values) <= periods:
        return float("nan")
    later = values[-1]
    earlier = values[-1 - periods]
    if not (np.isfinite(later) and np.isfinite(earlier)):
        return float("nan")
    return float(later - earlier)


def _tail(values: np.ndarray, count: int, *, min_count: int) -> np.ndarray:
    valid = values[np.isfinite(values)]
    if len(valid) < min_count:
        return np.array([], dtype="float64")
    return valid[-count:]


def _std(values: np.ndarray, count: int, *, min_count: int) -> float:
    tail = _tail(values, count, min_count=min_count)
    return float(tail.std(ddof=1)) if len(tail) >= 2 else float("nan")


def _z_last(values: np.ndarray, count: int, *, min_count: int) -> float:
    latest = _last(values)
    tail = _tail(values, count, min_count=min_count)
    if not np.isfinite(latest) or len(tail) < 2:
        return float("nan")
    return _safe_div(latest - float(tail.mean()), float(tail.std(ddof=1)))


def _complete_tail(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    count: int,
    min_count: int,
) -> pd.DataFrame:
    """Return an aligned, complete tail without using stale terminal values."""

    if frame.empty:
        return pd.DataFrame(columns=columns, dtype="float64")
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    if numeric.empty or numeric.iloc[-1].isna().any():
        return pd.DataFrame(columns=columns, dtype="float64")
    complete = numeric.dropna()
    if len(complete) < min_count:
        return pd.DataFrame(columns=columns, dtype="float64")
    return complete.tail(count).reset_index(drop=True)


def _ols_fit(x: np.ndarray, y: np.ndarray, *, min_count: int) -> tuple[float, float, np.ndarray] | None:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_count or float(np.std(x)) <= _EPS:
        return None
    design = np.column_stack([np.ones(len(x), dtype="float64"), x])
    if np.linalg.matrix_rank(design) < 2:
        return None
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    residual = y - design @ beta
    if not np.isfinite(beta).all() or not np.isfinite(residual).all():
        return None
    return float(beta[0]), float(beta[1]), residual


def _slope(x: np.ndarray, y: np.ndarray, *, min_count: int) -> float:
    fitted = _ols_fit(x, y, min_count=min_count)
    return fitted[1] if fitted is not None else float("nan")


def _autocorr(values: np.ndarray, *, count: int, min_count: int) -> float:
    tail = _tail(values, count, min_count=min_count)
    if len(tail) < 3:
        return float("nan")
    return _slope(tail[:-1], tail[1:], min_count=max(3, min_count - 1))


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _robust_rank_ridge_residuals(
    snapshot: pd.DataFrame,
    *,
    target: str,
    features: tuple[str, ...],
) -> pd.Series:
    """Fit a cross-sectional rank-ridge with Huber reweighting, or return NaN.

    The rank and sample guards deliberately fail closed.  This prevents a
    degenerate small cross-section from silently becoming a constant or an
    arbitrary pseudo-residual.
    """

    output = pd.Series(np.nan, index=snapshot.index, dtype="float64")
    valid = snapshot.loc[:, [target, *features]].dropna()
    if len(valid) < _MIN_CROSS_SECTION:
        return output
    ranked = valid.loc[:, list(features)].rank(method="average", pct=True).to_numpy(dtype="float64") - 0.5
    design = np.column_stack([np.ones(len(valid), dtype="float64"), ranked])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return output
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > 1e8:
        return output
    y = valid[target].to_numpy(dtype="float64")
    weights = np.ones(len(valid), dtype="float64")
    ridge = np.diag(np.r_[0.0, np.full(len(features), 0.05, dtype="float64")])
    beta: np.ndarray | None = None
    residual = np.full(len(valid), np.nan, dtype="float64")
    for _ in range(3):
        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + ridge
        rhs = design.T @ (weights * y)
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return output
        residual = y - design @ beta
        scale = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
        if not np.isfinite(scale) or scale <= _EPS:
            return output
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), _EPS))
    if beta is None or not np.isfinite(residual).all():
        return output
    output.loc[valid.index] = residual
    return output


def _conditional_premium_feature_frame(
    out_index: pd.MultiIndex,
    history: pd.DataFrame,
) -> pd.DataFrame:
    fields = ("bond_prem_ratio", "ytm", "duration", "year_to_mat", "remain_size")
    groups = _history_groups_at_anchor(history)
    records: list[dict[str, object]] = []
    for dt, code in out_index:
        market_code = _canonical_market_code(pd.Series([code]), pd.Series([""])).iloc[0]
        group = groups.get(market_code)
        record: dict[str, object] = {"dt": dt, "code": code}
        if group is not None:
            values = _complete_tail(group, fields, count=30, min_count=21)
            if not values.empty:
                premium = values["bond_prem_ratio"].to_numpy(dtype="float64")
                remain_size = values["remain_size"].to_numpy(dtype="float64")
                latest_remain = _last(remain_size)
                record.update(
                    {
                        "__premium_level": _last(premium),
                        "__premium_delta1": _delta(premium, 1),
                        "__premium_delta5": _delta(premium, 5),
                        "__premium_acceleration": _delta(premium, 1) - _delta(premium, 5),
                        "__premium_volatility": _std(premium, 20, min_count=15),
                        "__ytm": _last(values["ytm"].to_numpy(dtype="float64")),
                        "__duration": _last(values["duration"].to_numpy(dtype="float64")),
                        "__maturity": _last(values["year_to_mat"].to_numpy(dtype="float64")),
                        "__log_remain_size": (
                            float(np.log(latest_remain))
                            if np.isfinite(latest_remain) and latest_remain > 0.0
                            else float("nan")
                        ),
                    }
                )
        records.append(record)
    snapshot = pd.DataFrame(records).set_index(["dt", "code"]).reindex(out_index)
    feature_columns = ("__ytm", "__duration", "__maturity", "__log_remain_size")
    targets = {
        "cprr_premium_level_residual": "__premium_level",
        "cprr_premium_delta1_residual": "__premium_delta1",
        "cprr_premium_delta5_residual": "__premium_delta5",
        "cprr_premium_acceleration_residual": "__premium_acceleration",
        "cprr_premium_volatility_residual": "__premium_volatility",
    }
    out = pd.DataFrame(index=out_index, columns=_CONDITIONAL_PREMIUM_SIGNALS, dtype="float64")
    for signal, target in targets.items():
        out[signal] = _robust_rank_ridge_residuals(
            snapshot,
            target=target,
            features=feature_columns,
        ).reindex(out_index)
    return out.replace([np.inf, -np.inf], np.nan)


def _premium_yield_regime_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_PREMIUM_YIELD_REGIME_SIGNALS)
    tail = _complete_tail(
        frame,
        ("bond_prem_ratio", "ytm", "current_yield"),
        count=75,
        min_count=62,
    )
    if tail.empty:
        return out
    premium = tail["bond_prem_ratio"].to_numpy(dtype="float64")
    ytm = tail["ytm"].to_numpy(dtype="float64")
    current_yield = tail["current_yield"].to_numpy(dtype="float64")
    dpremium = np.diff(premium)
    dytm = np.diff(ytm)
    long_x = dytm[-60:]
    long_y = dpremium[-60:]
    short_fit = _ols_fit(dytm[-20:], dpremium[-20:], min_count=12)
    long_fit = _ols_fit(long_x, long_y, min_count=30)
    innovation = float("nan")
    if len(long_x) >= 31:
        innovation_fit = _ols_fit(long_x[:-1], long_y[:-1], min_count=30)
        if innovation_fit is not None:
            intercept, beta, _ = innovation_fit
            innovation = float(long_y[-1] - (intercept + beta * long_x[-1]))
    residual_vol_ratio = float("nan")
    if long_fit is not None:
        _, _, residual_long = long_fit
        residual_short = residual_long[-20:]
        residual_vol_ratio = _safe_div(
            float(np.std(residual_short, ddof=1)),
            float(np.std(residual_long, ddof=1)),
        )
    out.update(
        {
            "pyreg_beta_60": long_fit[1] if long_fit is not None else float("nan"),
            "pyreg_beta_shift_20_60": (
                short_fit[1] - long_fit[1]
                if short_fit is not None and long_fit is not None
                else float("nan")
            ),
            "pyreg_latest_innovation_60": innovation,
            "pyreg_residual_vol_ratio_20_60": residual_vol_ratio,
            "pyreg_current_yield_ytm_basis_z20": _z_last(
                current_yield - ytm,
                20,
                min_count=15,
            ),
        }
    )
    return out


def _positive_log(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values) & (values > 0.0)
    out[valid] = np.log(values[valid])
    return out


def _lagged_liquidity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LAGGED_LIQUIDITY_SIGNALS)
    tail = _complete_tail(
        frame,
        ("prev_close_price", "close_price", "amount"),
        count=75,
        min_count=62,
    )
    if tail.empty:
        return out
    prev_close = tail["prev_close_price"].to_numpy(dtype="float64")
    close = tail["close_price"].to_numpy(dtype="float64")
    amount = tail["amount"].to_numpy(dtype="float64")
    ret = _ratio(close, prev_close)
    log_amount = _positive_log(amount)
    signed_flow = np.sign(ret) * log_amount
    amihud = _safe_div_array(np.abs(ret), amount)
    flow_x = signed_flow[:-1]
    response_y = ret[1:]
    beta20 = _slope(flow_x[-20:], response_y[-20:], min_count=12)
    beta60 = _slope(flow_x[-60:], response_y[-60:], min_count=30)
    impact_beta = _slope(amihud[:-1][-20:], np.abs(ret[1:])[-20:], min_count=12)
    highflow_spread = float("nan")
    recent_flow = log_amount[:-1][-20:]
    recent_response = response_y[-20:]
    if np.isfinite(recent_flow).all() and np.isfinite(recent_response).all():
        median = float(np.median(recent_flow))
        high = recent_response[recent_flow > median]
        low = recent_response[recent_flow <= median]
        if len(high) >= 4 and len(low) >= 4:
            highflow_spread = float(high.mean() - low.mean())
    out.update(
        {
            "llpt_signed_flow_lag_beta20": beta20,
            "llpt_signed_flow_lag_beta60": beta60,
            "llpt_signed_flow_beta_shift": (
                beta20 - beta60 if np.isfinite(beta20) and np.isfinite(beta60) else float("nan")
            ),
            "llpt_amihud_to_absret_beta20": impact_beta,
            "llpt_highflow_next_return_spread20": highflow_spread,
            "llpt_impact_memory20": _autocorr(amihud, count=20, min_count=12),
        }
    )
    return out


def _drawdown_recovery_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_DRAWDOWN_RECOVERY_SIGNALS)
    tail = _complete_tail(
        frame,
        ("close_price", "high_price", "low_price", "amount"),
        count=20,
        min_count=12,
    )
    if tail.empty:
        return out
    close = tail["close_price"].to_numpy(dtype="float64")
    high = tail["high_price"].to_numpy(dtype="float64")
    low = tail["low_price"].to_numpy(dtype="float64")
    amount = tail["amount"].to_numpy(dtype="float64")
    peak_index = int(np.argmax(high))
    trough_index = int(np.argmin(low))
    recovery_efficiency = float("nan")
    if trough_index < len(close) - 1:
        path = close[trough_index:]
        path_total = float(np.abs(np.diff(path)).sum())
        recovery_efficiency = _safe_div(abs(float(path[-1] - path[0])), path_total)
    amount_after_trough = float(np.mean(amount[trough_index:]))
    out.update(
        {
            "drt_drawdown_from_high20": _safe_div(float(close[-1] - high[peak_index]), float(high[peak_index])),
            "drt_rebound_from_low20": _safe_div(float(close[-1] - low[trough_index]), float(low[trough_index])),
            "drt_peak_recency20": _safe_div(float(len(close) - 1 - peak_index), float(len(close) - 1)),
            "drt_trough_recency20": _safe_div(float(len(close) - 1 - trough_index), float(len(close) - 1)),
            "drt_recovery_path_efficiency20": recovery_efficiency,
            "drt_liquidity_recovery_ratio20": _safe_div(amount_after_trough, float(np.mean(amount))) - 1.0,
        }
    )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _phase_returns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    morning_open = _numeric(frame, "twap_0930_0935")
    morning_end = _numeric(frame, "twap_0935_1000")
    noon = _numeric(frame, "twap_1100_1130")
    afternoon_open = _numeric(frame, "twap_1300_1330")
    afternoon_end = _numeric(frame, "twap_1400_1430")
    late = _numeric(frame, "twap_1430_1442")
    execution = _numeric(frame, "twap_1442_1457")
    return (
        _ratio(morning_end, morning_open),
        _ratio(afternoon_open, noon),
        _ratio(afternoon_end, afternoon_open),
        _ratio(late, afternoon_end),
        _ratio(execution, late),
        _ratio(morning_open[1:], late[:-1]),
    )


def _normalized_entropy(states: np.ndarray) -> float:
    if len(states) < 20:
        return float("nan")
    _, counts = np.unique(states, return_counts=True)
    probabilities = counts.astype("float64") / float(counts.sum())
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(16.0))


def _twap_phase_transition_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_TWAP_TRANSITION_SIGNALS)
    tail = _complete_tail(
        frame,
        (
            "twap_0930_0935",
            "twap_0935_1000",
            "twap_1100_1130",
            "twap_1300_1330",
            "twap_1400_1430",
            "twap_1430_1442",
            "twap_1442_1457",
        ),
        count=75,
        min_count=25,
    )
    if tail.empty:
        return out
    morning, lunch, afternoon, late, execution, overnight = _phase_returns(tail)
    phase = np.column_stack([morning, lunch, afternoon, late])
    valid_phase = np.isfinite(phase).all(axis=1)
    phase = phase[valid_phase][-60:]
    execution_aligned = execution[valid_phase][-60:]
    if len(phase) < 20:
        return out
    direction = (phase > 0.0).astype("int64")
    state = direction[:, 0] * 8 + direction[:, 1] * 4 + direction[:, 2] * 2 + direction[:, 3]
    transition_surprise = float("nan")
    if len(state) >= 8:
        previous_state = int(state[-2])
        current_state = int(state[-1])
        train = state[:-1]
        starts = train[:-1]
        ends = train[1:]
        denominator = int(np.sum(starts == previous_state))
        count = int(np.sum((starts == previous_state) & (ends == current_state)))
        if denominator >= 3 and count > 0:
            transition_surprise = float(-np.log(float(count) / float(denominator)))
    morning_late = direction[:, 0] * 2 - 1
    late_direction = direction[:, 3] * 2 - 1
    execution_direction = np.sign(execution_aligned)
    execution_late_alignment = float("nan")
    if len(execution_direction) == len(phase) and np.isfinite(execution_direction).all():
        execution_late_alignment = float(np.mean(execution_direction * late_direction))
    late_overnight_alignment = float("nan")
    if len(overnight) >= 20 and np.isfinite(overnight[-60:]).all():
        overnight_direction = np.sign(overnight[-60:])
        late_for_overnight = np.sign(late[:-1][-60:])
        if len(overnight_direction) == len(late_for_overnight):
            late_overnight_alignment = float(np.mean(overnight_direction * late_for_overnight))
    out.update(
        {
            "tpt_state_entropy60": _normalized_entropy(state),
            "tpt_morning_late_alignment60": float(np.mean(morning_late * late_direction)),
            "tpt_late_overnight_alignment60": late_overnight_alignment,
            "tpt_latest_transition_surprise": transition_surprise,
            "tpt_intraday_turn_rate60": float(np.mean(direction[:, 1:] != direction[:, :-1])),
            "tpt_execution_late_alignment60": execution_late_alignment,
        }
    )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "premium_yield_regime_break": _premium_yield_regime_metrics,
    "lagged_liquidity_price_transmission": _lagged_liquidity_metrics,
    "drawdown_recovery_timing": _drawdown_recovery_metrics,
    "twap_phase_transition_entropy": _twap_phase_transition_metrics,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute/cache a full family for all of its concrete signal specs."""

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
        if family == "conditional_premium_repricing_residual":
            built = _conditional_premium_feature_frame(out_index, history)
        elif out_index.empty:
            built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
        else:
            _, fields = _FAMILY_SOURCE_FIELDS[family]
            groups = _history_groups_at_anchor(history)
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
class FactorMiningDailyIncrementalV1(Factor):
    """Research-only strict T-1 daily structural candidate kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        values = dict(params or {})
        if not str(values.get("signal", "")).strip():
            return _all_requirements()
        return [_requirement_for_family(_requested_entry(values).family)]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)
