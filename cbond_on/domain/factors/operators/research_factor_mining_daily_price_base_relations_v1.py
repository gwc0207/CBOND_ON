"""Research-only strict T-1 daily price/base relation catalogue.

This module consumes only declared prior rows from market_cbond.daily_price
and market_cbond.daily_base.  It intentionally has no live import,
configuration, file I/O, database I/O, or fallback data source.  The panel is
used solely to establish the T-day output (dt, code) index.

The three families are deliberately dynamic relations, not static price,
liquidity, premium, or floor levels:

* prior structural capacity utilization normalizes executed daily flow by
  remaining issue size;
* prior valuation-flow elasticity estimates historical co-movement between
  changes in structural valuation and normalized flow; and
* prior intrinsic-anchor topology compares the conversion and debt anchors to
  one another and to the prior market close.

Every source row at or after the score date is excluded before any join.  A
security is additionally rejected when either source is stale relative to the
latest common strict-prior daily session.
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


KERNEL_NAME = "factor_mining_daily_price_base_relations_v1"
CATALOG_VERSION = "20260803_daily_price_base_relations_v1"
_LOOKBACK_DAYS = 45
_WINDOW = 20
_MIN_REGRESSION_OBSERVATIONS = 12
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
_PRICE_FIELDS = ("close_price", "volume", "amount")
_BASE_FIELDS = (
    "remain_size",
    "bond_prem_ratio",
    "current_yield",
    "conv_value",
    "pure_redemption_value",
)


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


_CAPACITY_SIGNALS = (
    "pca_notional_capacity_z20",
    "pca_volume_capacity_z20",
    "pca_notional_capacity_trend20",
)
_ELASTICITY_SIGNALS = (
    "pvfe_premium_velocity_beta20",
    "pvfe_conv_value_velocity_beta20",
    "pvfe_yield_velocity_beta20",
)
_ANCHOR_SIGNALS = (
    "piat_conversion_floor_gap_z20",
    "piat_market_anchor_wedge_z20",
    "piat_anchor_gap_return_beta20",
)

_CATALOG = (
    _entries(
        "prior_structural_capacity_utilization",
        _CAPACITY_SIGNALS,
        "Prior-session notional and volume relative to remaining issue size describe exceptional capacity use rather than raw liquidity level.",
    )
    + _entries(
        "prior_valuation_flow_elasticity",
        _ELASTICITY_SIGNALS,
        "Historical normalized-flow response to premium, conversion-value, and yield changes describes valuation-sensitive participation.",
    )
    + _entries(
        "prior_intrinsic_anchor_topology",
        _ANCHOR_SIGNALS,
        "The prior conversion/floor anchor geometry and its historical link to market returns describe intrinsic-value topology.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "prior_structural_capacity_utilization": _CAPACITY_SIGNALS,
    "prior_valuation_flow_elasticity": _ELASTICITY_SIGNALS,
    "prior_intrinsic_anchor_topology": _ANCHOR_SIGNALS,
}

FORMULAS: dict[str, str] = {
    "pca_notional_capacity_z20": "z20(log(amount/remain_size)) on the latest strict-prior session against its preceding 20 sessions",
    "pca_volume_capacity_z20": "z20(log(volume/remain_size)) on the latest strict-prior session against its preceding 20 sessions",
    "pca_notional_capacity_trend20": "OLS time slope of log(amount/remain_size) over the last 20 strict-prior sessions",
    "pvfe_premium_velocity_beta20": "OLS beta in Delta log(amount/remain_size) = alpha + beta * Delta bond_prem_ratio over 20 strict-prior changes",
    "pvfe_conv_value_velocity_beta20": "OLS beta in Delta log(amount/remain_size) = alpha + beta * Delta log(conv_value) over 20 strict-prior changes",
    "pvfe_yield_velocity_beta20": "OLS beta in Delta log(amount/remain_size) = alpha + beta * Delta current_yield over 20 strict-prior changes",
    "piat_conversion_floor_gap_z20": "z20(log(conv_value/pure_redemption_value)) on the latest strict-prior session",
    "piat_market_anchor_wedge_z20": "z20(log(close_price/sqrt(conv_value*pure_redemption_value))) on the latest strict-prior session",
    "piat_anchor_gap_return_beta20": "OLS beta in Delta log(close_price) = alpha + beta * Delta log(conv_value/pure_redemption_value) over 20 strict-prior changes",
}


def daily_price_base_relations_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first strict-T-1 research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_price_base_relations_catalog()


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
    return history.sort_values(["code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    ), price_anchor


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


def _log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > _EPS)
        & (denominator > _EPS)
    )
    out[valid] = np.log(numerator[valid] / denominator[valid])
    return out


def _positive_log(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values) & (values > _EPS)
    out[valid] = np.log(values[valid])
    return out


def _z_last(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    scale = float(np.std(prior, ddof=1))
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / scale)


def _slope(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_count: int = _MIN_REGRESSION_OBSERVATIONS,
) -> float:
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


def _empty_metrics() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _capacity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CAPACITY_SIGNALS}
    values = _complete_tail(frame, ("amount", "volume", "remain_size"), _WINDOW + 1)
    if values is None:
        return out
    amount, volume, remain_size = values
    notional_capacity = _log_ratio(amount, remain_size)
    volume_capacity = _log_ratio(volume, remain_size)
    if not (
        np.isfinite(notional_capacity).all() and np.isfinite(volume_capacity).all()
    ):
        return out
    out["pca_notional_capacity_z20"] = _z_last(notional_capacity)
    out["pca_volume_capacity_z20"] = _z_last(volume_capacity)
    out["pca_notional_capacity_trend20"] = _slope(
        np.arange(_WINDOW, dtype="float64"),
        notional_capacity[-_WINDOW:],
    )
    return out


def _valuation_flow_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ELASTICITY_SIGNALS}
    values = _complete_tail(
        frame,
        ("amount", "remain_size", "bond_prem_ratio", "conv_value", "current_yield"),
        _WINDOW + 1,
    )
    if values is None:
        return out
    amount, remain_size, premium, conv_value, current_yield = values
    velocity_change = np.diff(_log_ratio(amount, remain_size))
    conversion_change = np.diff(_positive_log(conv_value))
    if not (
        np.isfinite(velocity_change).all() and np.isfinite(conversion_change).all()
    ):
        return out
    out["pvfe_premium_velocity_beta20"] = _slope(np.diff(premium), velocity_change)
    out["pvfe_conv_value_velocity_beta20"] = _slope(conversion_change, velocity_change)
    out["pvfe_yield_velocity_beta20"] = _slope(np.diff(current_yield), velocity_change)
    return out


def _intrinsic_anchor_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ANCHOR_SIGNALS}
    values = _complete_tail(
        frame,
        ("close_price", "conv_value", "pure_redemption_value"),
        _WINDOW + 1,
    )
    if values is None:
        return out
    close_price, conv_value, pure_redemption_value = values
    conversion_floor_gap = _log_ratio(conv_value, pure_redemption_value)
    log_close = _positive_log(close_price)
    log_conversion = _positive_log(conv_value)
    log_floor = _positive_log(pure_redemption_value)
    market_anchor_wedge = log_close - 0.5 * (log_conversion + log_floor)
    if not (
        np.isfinite(conversion_floor_gap).all()
        and np.isfinite(market_anchor_wedge).all()
    ):
        return out
    out["piat_conversion_floor_gap_z20"] = _z_last(conversion_floor_gap)
    out["piat_market_anchor_wedge_z20"] = _z_last(market_anchor_wedge)
    out["piat_anchor_gap_return_beta20"] = _slope(
        np.diff(conversion_floor_gap),
        np.diff(log_close),
    )
    return out


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics()
    out.update(_capacity_metrics(frame))
    out.update(_valuation_flow_metrics(frame))
    out.update(_intrinsic_anchor_metrics(frame))
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
        )
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyPriceBaseRelationsV1(Factor):
    """Research-only dynamic strict-T-1 daily price/base relation kernel."""

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
    "FactorMiningDailyPriceBaseRelationsV1",
    "daily_price_base_relations_catalog",
    "factor_mining_catalog",
]
