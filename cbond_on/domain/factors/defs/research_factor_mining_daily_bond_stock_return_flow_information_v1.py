"""Research-only strict-T-1 bond-stock return/flow information factor.

The single pre-screened signal measures whether the sign of a mapped
underlying stock's completed daily return carries information about the sign
of the convertible bond's completed daily amount change.  It is a temporal
cross-asset information channel, not a level, beta, parity, or current-day
flow feature.

Every input row is supplied by the factor context and is strictly earlier than
the score day.  Missing source rows remain missing through a full source-date
reindex, so an absent session can never be compressed into a synthetic
adjacent amount change.  This module is research-only: it has no file, DB,
label, PnL, scheduler, model, live-config, or production-FactorStore access.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling_v1,
)


KERNEL_NAME = "factor_mining_daily_bond_stock_return_flow_information_v1"
CATALOG_VERSION = "20260803_daily_bond_stock_return_flow_information_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price", "amount")
_BASE_FIELDS = ("stk_prev_close_price", "stk_close_price")


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only candidate in one information family."""

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


_SIGNALS = ("bsfst_stock_return_bond_flow_mutual_information60",)
_CATALOG = _entries(
    "prior_bond_stock_return_flow_information_dependence",
    _SIGNALS,
    "Strict-prior information dependence between an underlying stock's return "
    "direction and its convertible bond's amount-change direction can capture a "
    "cross-asset participation channel distinct from beta, tracking error, or "
    "same-asset return/liquidity dependence.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "bsfst_stock_return_bond_flow_mutual_information60": (
        "Mutual information divided by log(3) between sign(log(stk_close/"
        "stk_prev_close)) and sign(delta log(bond amount)) across the latest "
        "up-to-60 strict-prior source sessions, using a 3x3 Jeffreys-pseudocount "
        "table. It requires a finite terminal state and at least 45 valid joint "
        "source sessions."
    )
}


def daily_bond_stock_return_flow_information_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only family catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_bond_stock_return_flow_information_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _strict_history_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Normalize one supplied daily source and remove score/future rows."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    required = ("trade_date", "code", "exchange_code", *fields)
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")
    frame = raw.loc[:, list(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = rank_coupling_v1._canonical_market_code(
        frame["code"], frame["exchange_code"]
    )
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
) -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.Timestamp | None]:
    """Align source rows while preserving every source calendar gap."""

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
        return pd.DataFrame(), pd.DatetimeIndex([]), None
    price_anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    base_anchor = pd.Timestamp(base["trade_date"].max()).normalize()
    if price_anchor != base_anchor:
        return pd.DataFrame(), pd.DatetimeIndex([]), None
    sessions = pd.DatetimeIndex(
        sorted(set(price["trade_date"]).union(set(base["trade_date"])))
    )
    history = price.loc[:, ["trade_date", "code", *_PRICE_FIELDS]].merge(
        base.loc[:, ["trade_date", "code", *_BASE_FIELDS]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    )
    return (
        history.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True),
        sessions,
        price_anchor,
    )


def _log_return(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(len(numerator), np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > _EPS)
        & (denominator > _EPS)
    )
    result[valid] = np.log(numerator[valid] / denominator[valid])
    return result


def _amount_change(amount: np.ndarray) -> np.ndarray:
    """Return one adjacent-source-session log-amount change per source date."""

    log_amount = np.full(len(amount), np.nan, dtype="float64")
    positive = np.isfinite(amount) & (amount > _EPS)
    log_amount[positive] = np.log(amount[positive])
    result = np.full(len(amount), np.nan, dtype="float64")
    valid = np.isfinite(log_amount[1:]) & np.isfinite(log_amount[:-1])
    result[1:][valid] = log_amount[1:][valid] - log_amount[:-1][valid]
    return result


def _sign3(values: np.ndarray) -> np.ndarray:
    """Encode finite negative/zero/positive values, retaining missing as -1."""

    result = np.full(len(values), -1, dtype=np.int8)
    finite = np.isfinite(values)
    result[finite & (values < -_EPS)] = 0
    result[finite & (np.abs(values) <= _EPS)] = 1
    result[finite & (values > _EPS)] = 2
    return result


def _normalized_mutual_information(stock_return: np.ndarray, amount_change: np.ndarray) -> float:
    """Estimate the fixed 3x3 sign-state MI without missing-value imputation."""

    stock_sign = _sign3(stock_return)
    flow_sign = _sign3(amount_change)
    if stock_sign[-1] < 0 or flow_sign[-1] < 0:
        return float("nan")
    valid = (stock_sign >= 0) & (flow_sign >= 0)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    counts = np.full((3, 3), 0.5, dtype="float64")
    np.add.at(counts, (stock_sign[valid], flow_sign[valid]), 1.0)
    probability = counts / float(counts.sum())
    stock_probability = probability.sum(axis=1, keepdims=True)
    flow_probability = probability.sum(axis=0, keepdims=True)
    value = float(
        np.sum(probability * np.log(probability / (stock_probability * flow_probability)))
        / math.log(3.0)
    )
    return value if np.isfinite(value) else float("nan")


def _signal_value(frame: pd.DataFrame, *, sessions: pd.DatetimeIndex, anchor: pd.Timestamp) -> float:
    """Calculate the pre-registered channel on the latest strict-prior window."""

    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return float("nan")
    recent = frame.set_index("trade_date").reindex(sessions).tail(_WINDOW)
    stock_return = _log_return(
        pd.to_numeric(recent["stk_close_price"], errors="coerce").to_numpy(dtype="float64"),
        pd.to_numeric(recent["stk_prev_close_price"], errors="coerce").to_numpy(dtype="float64"),
    )
    amount_change = _amount_change(
        pd.to_numeric(recent["amount"], errors="coerce").to_numpy(dtype="float64")
    )
    return _normalized_mutual_information(stock_return, amount_change)


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build and cache all family values for one strictly prior score context."""

    score_date = rank_coupling_v1._score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    output_index = rank_coupling_v1._output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_SIGNALS, dtype="float64")
    else:
        history, sessions, anchor = _aligned_history(ctx, score_date=score_date)
        if anchor is None:
            built = pd.DataFrame(index=output_index, columns=_SIGNALS, dtype="float64")
        else:
            groups = {
                str(code): group
                for code, group in history.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in output_index:
                value = float("nan")
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    value = _signal_value(group, sessions=sessions, anchor=anchor)
                rows.append({"dt": dt, "code": raw_code, _SIGNALS[0]: value})
            built = (
                pd.DataFrame(rows)
                .set_index(["dt", "code"])[list(_SIGNALS)]
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
class FactorMiningDailyBondStockReturnFlowInformationV1(Factor):
    """Research-only strict-prior cross-asset return/flow information kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement(
                "market_cbond.daily_price", ("exchange_code", *_PRICE_FIELDS), _LOOKBACK_DAYS
            ),
            DailyFactorRequirement(
                "market_cbond.daily_base", ("exchange_code", *_BASE_FIELDS), _LOOKBACK_DAYS
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
    "FactorMiningDailyBondStockReturnFlowInformationV1",
    "daily_bond_stock_return_flow_information_catalog",
    "factor_mining_catalog",
]
