"""Research-only strict-T-1 asymmetric bond/stock beta catalogue.

This catalogue measures how a convertible bond has participated in *prior*
positive and negative moves of its mapped underlying stock.  It deliberately
uses the stock prices carried by the already declared ``daily_base`` row,
rather than loading an equity file, inferring a mapping, or accessing a
future score-day value.

The two concrete signals are one economic family, not sign or window aliases:

* upside beta estimates the historical equity sensitivity conditional on a
  positive underlying return; and
* downside beta estimates the same sensitivity conditional on a negative
  underlying return.

Every daily source row at or after the score date is discarded before the
sources are joined.  A security must also have a row at the latest common
strict-prior source session; otherwise all outputs fail closed to ``NaN``.
This module is research-only and has no live import, configuration, file I/O,
database I/O, model, mask, or scheduler dependency.
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


KERNEL_NAME = "factor_mining_daily_asymmetric_equity_beta_v1"
CATALOG_VERSION = "20260803_daily_asymmetric_equity_beta_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_FINITE_OBSERVATIONS = 40
_MIN_CONDITIONAL_OBSERVATIONS = 10
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
_PRICE_FIELDS = ("prev_close_price", "close_price")
_BASE_FIELDS = ("stk_prev_close_price", "stk_close_price")


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only signal specification."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str,
    signals: Iterable[str],
    hypothesis: str,
) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_ASYMMETRIC_BETA_SIGNALS = (
    "bsab_downside_beta60",
    "bsab_upside_beta60",
)
_CATALOG = _entries(
    "prior_asymmetric_equity_beta",
    _ASYMMETRIC_BETA_SIGNALS,
    "Strict-prior conditional bond/stock beta separates downside participation from upside participation; it is not a static conversion value, premium, or unconditional tracking-error level.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "bsab_downside_beta60": "OLS beta of log(close/prev_close) on log(stk_close/stk_prev_close) across the latest up-to-60 strict-prior sessions with negative stock returns; at least 40 finite joint sessions and 10 negative-stock observations are required.",
    "bsab_upside_beta60": "OLS beta of log(close/prev_close) on log(stk_close/stk_prev_close) across the latest up-to-60 strict-prior sessions with positive stock returns; at least 40 finite joint sessions and 10 positive-stock observations are required.",
}


def daily_asymmetric_equity_beta_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first strict-prior research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research expansion-runner entrypoint."""

    return daily_asymmetric_equity_beta_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series) -> pd.Series:
    """Normalize only daily codes with an explicit or valid suffix."""

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
        [_one(value, exchange) for value, exchange in zip(values, exchanges, strict=False)],
        index=values.index,
        dtype="string",
    )


def _panel_code(value: object) -> str:
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
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique_dates = pd.Index(dates[dates.notna()]).unique()
    if len(unique_dates) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique_dates[0]).normalize()


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
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
    """Return unique normalized source rows strictly before the score date."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), source=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & frame["code"].ne("")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _aligned_history(
    ctx: FactorComputeContext,
    *,
    score_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Join price/base history only when their latest prior sessions agree."""

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
    columns = ["trade_date", "code", *_PRICE_FIELDS, *_BASE_FIELDS]
    if price.empty or base.empty:
        return pd.DataFrame(columns=columns), None
    price_anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    base_anchor = pd.Timestamp(base["trade_date"].max()).normalize()
    if price_anchor != base_anchor:
        return pd.DataFrame(columns=columns), None
    history = price.loc[:, ["trade_date", "code", *_PRICE_FIELDS]].merge(
        base.loc[:, ["trade_date", "code", *_BASE_FIELDS]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    )
    return (
        history.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True),
        price_anchor,
    )


def _returns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bond_prev = pd.to_numeric(frame["prev_close_price"], errors="coerce").to_numpy(dtype="float64")
    bond_close = pd.to_numeric(frame["close_price"], errors="coerce").to_numpy(dtype="float64")
    stock_prev = pd.to_numeric(frame["stk_prev_close_price"], errors="coerce").to_numpy(dtype="float64")
    stock_close = pd.to_numeric(frame["stk_close_price"], errors="coerce").to_numpy(dtype="float64")
    valid = (
        np.isfinite(bond_prev)
        & np.isfinite(bond_close)
        & np.isfinite(stock_prev)
        & np.isfinite(stock_close)
        & (bond_prev > _EPS)
        & (bond_close > _EPS)
        & (stock_prev > _EPS)
        & (stock_close > _EPS)
    )
    bond_return = np.full(len(frame), np.nan, dtype="float64")
    stock_return = np.full(len(frame), np.nan, dtype="float64")
    bond_return[valid] = np.log(bond_close[valid] / bond_prev[valid])
    stock_return[valid] = np.log(stock_close[valid] / stock_prev[valid])
    return bond_return, stock_return


def _conditional_beta(stock_returns: np.ndarray, bond_returns: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < _MIN_CONDITIONAL_OBSERVATIONS:
        return float("nan")
    x = stock_returns[mask]
    y = bond_returns[mask]
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    value = float(np.dot(centered, y - float(np.mean(y))) / denominator)
    return value if np.isfinite(value) else float("nan")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ASYMMETRIC_BETA_SIGNALS}
    bond_returns, stock_returns = _returns(frame.tail(_WINDOW))
    # The latest strict-prior session is the factor's state anchor.  Do not
    # silently discard a malformed anchor merely because an older subset of
    # observations would still satisfy the regression minimum.
    if (
        len(bond_returns) == 0
        or not np.isfinite(bond_returns[-1])
        or not np.isfinite(stock_returns[-1])
    ):
        return out
    finite = np.isfinite(bond_returns) & np.isfinite(stock_returns)
    if int(finite.sum()) < _MIN_FINITE_OBSERVATIONS:
        return out
    bond_returns = bond_returns[finite]
    stock_returns = stock_returns[finite]
    out["bsab_downside_beta60"] = _conditional_beta(
        stock_returns,
        bond_returns,
        stock_returns < 0.0,
    )
    out["bsab_upside_beta60"] = _conditional_beta(
        stock_returns,
        bond_returns,
        stock_returns > 0.0,
    )
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache both strict-prior signals for one T-day context."""

    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = _output_index(ctx, score_date)
    if score_date is None or out_index.empty:
        built = pd.DataFrame(index=out_index, columns=_ASYMMETRIC_BETA_SIGNALS, dtype="float64")
    else:
        history, anchor = _aligned_history(ctx, score_date=score_date)
        groups = {
            str(code): group
            for code, group in history.groupby("code", sort=False)
            if anchor is not None and pd.Timestamp(group["trade_date"].max()).normalize() == anchor
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in out_index:
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ASYMMETRIC_BETA_SIGNALS})
            group = groups.get(_panel_code(raw_code))
            if group is not None:
                row.update(_metrics(group))
            rows.append(row)
        built = (
            pd.DataFrame(rows)
            .set_index(["dt", "code"])[list(_ASYMMETRIC_BETA_SIGNALS)]
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
class FactorMiningDailyAsymmetricEquityBetaV1(Factor):
    """Research-only strict-prior conditional bond/stock beta kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
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
    "FactorMiningDailyAsymmetricEquityBetaV1",
    "daily_asymmetric_equity_beta_catalog",
    "factor_mining_catalog",
]
