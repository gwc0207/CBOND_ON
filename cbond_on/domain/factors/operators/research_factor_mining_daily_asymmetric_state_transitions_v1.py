"""Research-only strict T-1 asymmetric daily price/base state transitions.

This catalogue is economically separate from capacity, valuation-flow, and
intrinsic-anchor factors.  It models completed daily price-path responses to
three changing structural states:

* an increase versus decrease in current yield;
* an increase versus decrease in convexity density; and
* a stock-volatility stress/release state interacting with bond gaps and
  intraday candle bodies.

The module consumes only declared rows from market_cbond.daily_price and
market_cbond.daily_base strictly before the score date.  The T1430 panel
supplies output keys only; it contributes no factor values.  There is no file,
database, network, live, configuration, or scheduler I/O.  Invalid or stale
source histories fail closed to NaN and are never replaced by zeros.
"""

from __future__ import annotations

from collections.abc import Iterable
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


KERNEL_NAME = "factor_mining_daily_asymmetric_state_transitions_v1"
CATALOG_VERSION = "20260803_daily_asymmetric_state_transitions_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_DIRECTIONAL_OBSERVATIONS = 8
_MIN_EVENT_OBSERVATIONS = 4
_MIN_SIGN_OBSERVATIONS = 12
_EPS = 1e-12
_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})
_PRICE_FIELDS = (
    "act_prev_close_price",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
)
_BASE_FIELDS = ("current_yield", "duration", "convexity", "stock_volatility")


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only candidate signal."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str, signals: Iterable[str], hypothesis: str
) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals
    )


_YIELD_SIGNALS = (
    "ydpt_yield_rise_return_beta60",
    "ydpt_yield_fall_return_beta60",
    "ydpt_yield_return_beta_asymmetry60",
)
_RISK_SHAPE_SIGNALS = (
    "dcrt_density_body_asymmetry60",
    "dcrt_density_range_asymmetry60",
    "dcrt_density_sign_agreement60",
)
_STOCK_VOL_SIGNALS = (
    "svcr_stress_gap_reversal_rate60",
    "svcr_relief_gap_fade_rate60",
    "svcr_stress_relief_reversal_spread60",
)

_CATALOG = (
    _entries(
        "prior_yield_directional_price_pass_through",
        _YIELD_SIGNALS,
        "Completed bond-price response can differ after a rising versus falling current-yield state transition.",
    )
    + _entries(
        "prior_duration_convexity_range_transition",
        _RISK_SHAPE_SIGNALS,
        "Daily bond candle bodies and ranges can react asymmetrically to changes in convexity density rather than its static level.",
    )
    + _entries(
        "prior_stockvol_candle_reversal_state",
        _STOCK_VOL_SIGNALS,
        "Bond gap reversals during rising stock-volatility stress can differ from gap fades during volatility relief.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "prior_yield_directional_price_pass_through": _YIELD_SIGNALS,
    "prior_duration_convexity_range_transition": _RISK_SHAPE_SIGNALS,
    "prior_stockvol_candle_reversal_state": _STOCK_VOL_SIGNALS,
}

FORMULAS: dict[str, str] = {
    "ydpt_yield_rise_return_beta60": "OLS beta of adjusted daily log return on Delta current_yield over the latest 60 strict-prior changes with Delta current_yield > 0",
    "ydpt_yield_fall_return_beta60": "OLS beta of adjusted daily log return on Delta current_yield over the latest 60 strict-prior changes with Delta current_yield < 0",
    "ydpt_yield_return_beta_asymmetry60": "ydpt_yield_rise_return_beta60 - ydpt_yield_fall_return_beta60",
    "dcrt_density_body_asymmetry60": "mean(log(close/open) | Delta log(convexity/duration^2) > 0) - mean(log(close/open) | Delta log(convexity/duration^2) < 0) over 60 strict-prior changes",
    "dcrt_density_range_asymmetry60": "mean(log(high/low) | Delta log(convexity/duration^2) > 0) - mean(log(high/low) | Delta log(convexity/duration^2) < 0) over 60 strict-prior changes",
    "dcrt_density_sign_agreement60": "mean(sign(log(close/open)) * sign(Delta log(convexity/duration^2))) over nonzero latest 60 strict-prior changes",
    "svcr_stress_gap_reversal_rate60": "mean(1[log(open/act_prev_close) < 0 and Delta log(stock_volatility) > 0 and log(close/open) > 0]) conditional on stress-gap events over 60 strict-prior changes",
    "svcr_relief_gap_fade_rate60": "mean(1[log(open/act_prev_close) > 0 and Delta log(stock_volatility) < 0 and log(close/open) < 0]) conditional on relief-gap events over 60 strict-prior changes",
    "svcr_stress_relief_reversal_spread60": "svcr_stress_gap_reversal_rate60 - svcr_relief_gap_fade_rate60",
}


def daily_asymmetric_state_transitions_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first strict-T-1 research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_asymmetric_state_transitions_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series) -> pd.Series:
    """Normalize only codes whose exchange is explicit or already valid."""

    def _one(value: object, exchange: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().upper()
        if not text or text == "NAN":
            return ""
        if text.endswith(".0"):
            text = text[:-2]
        if "." in text:
            bare, suffix = text.rsplit(".", 1)
            suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
            if bare and suffix in _MARKET_EXCHANGES:
                return f"{bare}.{suffix}"
        raw_exchange = "" if pd.isna(exchange) else str(exchange).strip().upper()
        suffix = _EXCHANGE_ALIASES.get(raw_exchange, raw_exchange)
        return f"{text}.{suffix}" if suffix in _MARKET_EXCHANGES else ""

    return pd.Series(
        [
            _one(value, exchange)
            for value, exchange in zip(values, exchanges, strict=False)
        ],
        index=values.index,
        dtype="string",
    )


def _panel_code(value: object) -> str:
    """A panel key must already carry a valid exchange suffix."""

    return str(_canonical_market_code(pd.Series([value]), pd.Series([""])).iloc[0])


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        score_date = pd.Timestamp(raw_day)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(
        panel.index.get_level_values("dt"), errors="coerce"
    ).normalize()
    unique_dates = pd.Index(dates[dates.notna()]).unique()
    if len(unique_dates) != 1:
        raise ValueError(
            f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel"
        )
    return pd.Timestamp(unique_dates[0]).normalize()


def _output_index(
    ctx: FactorComputeContext, score_date: pd.Timestamp | None
) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = (
        keys.loc[dates == score_date]
        .drop_duplicates()
        .sort_values(["dt", "code"], kind="mergesort")
    )
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _strict_history_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load only completed strict-prior source rows with unique normalized keys."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(
        raw, ("trade_date", "code", "exchange_code", *fields), source=source
    )
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=["trade_date", "code", *_PRICE_FIELDS, *_BASE_FIELDS])


def _aligned_history(
    ctx: FactorComputeContext, *, score_date: pd.Timestamp
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Join the two strict-prior sources only when their latest sessions agree."""

    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=_PRICE_FIELDS,
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_BASE_FIELDS,
        score_date=score_date,
    )
    if price.empty or base.empty:
        return _empty_history(), None
    price_anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    base_anchor = pd.Timestamp(base["trade_date"].max()).normalize()
    if price_anchor != base_anchor:
        return _empty_history(), None
    keys = ["trade_date", "code"]
    history = price.loc[:, [*keys, *_PRICE_FIELDS]].merge(
        base.loc[:, [*keys, *_BASE_FIELDS]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    return (
        history.sort_values(["code", "trade_date"], kind="mergesort").reset_index(
            drop=True
        ),
        price_anchor,
    )


def _complete_tail(
    frame: pd.DataFrame, columns: tuple[str, ...], count: int
) -> tuple[np.ndarray, ...] | None:
    if len(frame) < count:
        return None
    tail = frame.tail(count)
    values = tuple(
        pd.to_numeric(tail[column], errors="coerce").to_numpy(dtype="float64")
        for column in columns
    )
    return values if all(np.isfinite(value).all() for value in values) else None


def _positive_log(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values) & (values > _EPS)
    out[valid] = np.log(values[valid])
    return out


def _slope(left: np.ndarray, right: np.ndarray, *, min_count: int) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < min_count:
        return float("nan")
    x = left[valid]
    y = right[valid]
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    value = float(np.dot(centered, y - float(np.mean(y))) / denominator)
    return value if np.isfinite(value) else float("nan")


def _conditional_slope(
    state_change: np.ndarray, response: np.ndarray, *, direction: int
) -> float:
    if direction > 0:
        keep = state_change > _EPS
    else:
        keep = state_change < -_EPS
    return _slope(
        state_change[keep],
        response[keep],
        min_count=_MIN_DIRECTIONAL_OBSERVATIONS,
    )


def _conditional_mean_gap(state_change: np.ndarray, response: np.ndarray) -> float:
    rising = response[state_change > _EPS]
    falling = response[state_change < -_EPS]
    if (
        len(rising) < _MIN_DIRECTIONAL_OBSERVATIONS
        or len(falling) < _MIN_DIRECTIONAL_OBSERVATIONS
    ):
        return float("nan")
    if not (np.isfinite(rising).all() and np.isfinite(falling).all()):
        return float("nan")
    return float(np.mean(rising) - np.mean(falling))


def _empty_metrics() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _yield_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _YIELD_SIGNALS}
    values = _complete_tail(
        frame,
        ("act_prev_close_price", "close_price", "current_yield"),
        _WINDOW + 1,
    )
    if values is None:
        return out
    adjusted_prev_close, close_price, current_yield = values
    price_return = _positive_log(close_price) - _positive_log(adjusted_prev_close)
    yield_change = np.diff(current_yield)
    if not np.isfinite(price_return).all():
        return out
    rise = _conditional_slope(yield_change, price_return[1:], direction=1)
    fall = _conditional_slope(yield_change, price_return[1:], direction=-1)
    out["ydpt_yield_rise_return_beta60"] = rise
    out["ydpt_yield_fall_return_beta60"] = fall
    if np.isfinite(rise) and np.isfinite(fall):
        out["ydpt_yield_return_beta_asymmetry60"] = float(rise - fall)
    return out


def _risk_shape_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _RISK_SHAPE_SIGNALS}
    values = _complete_tail(
        frame,
        (
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "duration",
            "convexity",
        ),
        _WINDOW + 1,
    )
    if values is None:
        return out
    open_price, high_price, low_price, close_price, duration, convexity = values
    body = _positive_log(close_price) - _positive_log(open_price)
    price_range = _positive_log(high_price) - _positive_log(low_price)
    risk_density = _positive_log(convexity) - 2.0 * _positive_log(duration)
    if not (
        np.isfinite(body).all()
        and np.isfinite(price_range).all()
        and np.isfinite(risk_density).all()
    ):
        return out
    density_change = np.diff(risk_density)
    body_tail = body[1:]
    range_tail = price_range[1:]
    out["dcrt_density_body_asymmetry60"] = _conditional_mean_gap(
        density_change, body_tail
    )
    out["dcrt_density_range_asymmetry60"] = _conditional_mean_gap(
        density_change, range_tail
    )
    valid = (
        np.isfinite(density_change)
        & np.isfinite(body_tail)
        & (np.abs(density_change) > _EPS)
        & (np.abs(body_tail) > _EPS)
    )
    if int(valid.sum()) >= _MIN_SIGN_OBSERVATIONS:
        out["dcrt_density_sign_agreement60"] = float(
            np.mean(np.sign(density_change[valid]) * np.sign(body_tail[valid]))
        )
    return out


def _stock_volatility_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _STOCK_VOL_SIGNALS}
    values = _complete_tail(
        frame,
        ("act_prev_close_price", "open_price", "close_price", "stock_volatility"),
        _WINDOW + 1,
    )
    if values is None:
        return out
    adjusted_prev_close, open_price, close_price, stock_volatility = values
    gap = _positive_log(open_price) - _positive_log(adjusted_prev_close)
    body = _positive_log(close_price) - _positive_log(open_price)
    volatility_change = np.diff(_positive_log(stock_volatility))
    if not (
        np.isfinite(gap).all()
        and np.isfinite(body).all()
        and np.isfinite(volatility_change).all()
    ):
        return out
    gap_tail = gap[1:]
    body_tail = body[1:]
    stress = (volatility_change > _EPS) & (gap_tail < -_EPS)
    relief = (volatility_change < -_EPS) & (gap_tail > _EPS)
    stress_rate = float("nan")
    relief_rate = float("nan")
    if int(stress.sum()) >= _MIN_EVENT_OBSERVATIONS:
        stress_rate = float(np.mean(body_tail[stress] > _EPS))
        out["svcr_stress_gap_reversal_rate60"] = stress_rate
    if int(relief.sum()) >= _MIN_EVENT_OBSERVATIONS:
        relief_rate = float(np.mean(body_tail[relief] < -_EPS))
        out["svcr_relief_gap_fade_rate60"] = relief_rate
    if np.isfinite(stress_rate) and np.isfinite(relief_rate):
        out["svcr_stress_relief_reversal_spread60"] = float(stress_rate - relief_rate)
    return out


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics()
    out.update(_yield_metrics(frame))
    out.update(_risk_shape_metrics(frame))
    out.update(_stock_volatility_metrics(frame))
    return {
        signal: value if np.isfinite(value) else float("nan")
        for signal, value in out.items()
    }


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build and cache all nine strict-prior signals for one T-day context."""

    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = _output_index(ctx, score_date)
    if score_date is None or out_index.empty:
        built = pd.DataFrame(index=out_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        history, anchor = _aligned_history(ctx, score_date=score_date)
        groups = {
            str(code): group
            for code, group in history.groupby("code", sort=False)
            if anchor is not None
            and pd.Timestamp(group["trade_date"].max()).normalize() == anchor
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in out_index:
            code = _panel_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update(_empty_metrics())
            group = groups.get(code)
            if group is not None:
                row.update(_metrics(group))
            rows.append(row)
        built = (
            pd.DataFrame(rows)
            .set_index(["dt", "code"])[list(_ALL_SIGNALS)]
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyAsymmetricStateTransitionsV1(Factor):
    """Research-only strict-T-1 asymmetric price/base state-transition kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(
        cls, params: dict | None = None
    ) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement(
                "market_cbond.daily_price",
                ("exchange_code", *_PRICE_FIELDS),
                _LOOKBACK_DAYS,
            ),
            DailyFactorRequirement(
                "market_cbond.daily_base",
                ("exchange_code", *_BASE_FIELDS),
                _LOOKBACK_DAYS,
            ),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        out = _feature_frame(ctx)[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out


__all__ = [
    "CATALOG_VERSION",
    "FORMULAS",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningDailyAsymmetricStateTransitionsV1",
    "daily_asymmetric_state_transitions_catalog",
    "factor_mining_catalog",
]
