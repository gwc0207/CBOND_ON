"""Research-only strict-T-1 bond/stock empirical-copula tail dependence.

This family measures whether a convertible bond has historically joined the
upper or lower *tail* of its mapped underlying stock's completed daily-return
distribution.  It deliberately differs from linear beta, tracking residual,
or a static parity level: each output is an empirical conditional tail
probability over a security's own latest strict-prior history.

Only daily rows strictly before the score date are permitted.  A missing,
non-finite, or stale terminal price/base state fails closed to ``NaN`` rather
than carrying an older relation forward.  This module is research-only: it has
no filesystem, database, label, mask, model, PnL, scheduler, live-config, or
production-FactorStore dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling_v1,
)


KERNEL_NAME = "factor_mining_daily_bond_stock_copula_tail_dependence_v1"
CATALOG_VERSION = "20260803_daily_bond_stock_copula_tail_dependence_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_FINITE_OBSERVATIONS = 45
_MIN_TAIL_OBSERVATIONS = 8
_LOWER_TAIL = 0.25
_UPPER_TAIL = 0.75
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price")
_BASE_FIELDS = ("stk_prev_close_price", "stk_close_price")


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit, family-first research candidate."""

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


_SIGNALS = (
    "bsct_upper_tail_dependence60",
    "bsct_lower_tail_dependence60",
    "bsct_tail_dependence_asymmetry60",
)
_CATALOG = _entries(
    "prior_bond_stock_copula_tail_dependence",
    _SIGNALS,
    "Strict-prior empirical tail co-movement of a bond and its underlying stock "
    "can distinguish upside participation, downside participation, and their "
    "asymmetry beyond an unconditional linear beta.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "bsct_upper_tail_dependence60": (
        "Across the latest up-to-60 strict-prior joint bond/stock return pairs, "
        "empirical P(bond return rank >= 75th percentile | stock return rank >= "
        "75th percentile), requiring 45 finite pairs and at least eight stock "
        "upper-tail observations."
    ),
    "bsct_lower_tail_dependence60": (
        "Across the latest up-to-60 strict-prior joint bond/stock return pairs, "
        "empirical P(bond return rank <= 25th percentile | stock return rank <= "
        "25th percentile), requiring 45 finite pairs and at least eight stock "
        "lower-tail observations."
    ),
    "bsct_tail_dependence_asymmetry60": (
        "Strict-prior upper-tail conditional dependence minus lower-tail "
        "conditional dependence on the same empirical-copula return window."
    ),
}


def daily_bond_stock_copula_tail_dependence_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only family catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_bond_stock_copula_tail_dependence_catalog()


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
    """Read one source only from supplied context, strictly before score day."""

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
            frame.duplicated(["trade_date", "code"], keep=False),
            ["trade_date", "code"],
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
    """Join daily price/base only when the latest strict-prior sessions agree."""

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


def _joint_log_returns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    bond_previous = pd.to_numeric(frame["prev_close_price"], errors="coerce").to_numpy(
        dtype="float64"
    )
    bond_close = pd.to_numeric(frame["close_price"], errors="coerce").to_numpy(
        dtype="float64"
    )
    stock_previous = pd.to_numeric(frame["stk_prev_close_price"], errors="coerce").to_numpy(
        dtype="float64"
    )
    stock_close = pd.to_numeric(frame["stk_close_price"], errors="coerce").to_numpy(
        dtype="float64"
    )
    valid = (
        np.isfinite(bond_previous)
        & np.isfinite(bond_close)
        & np.isfinite(stock_previous)
        & np.isfinite(stock_close)
        & (bond_previous > _EPS)
        & (bond_close > _EPS)
        & (stock_previous > _EPS)
        & (stock_close > _EPS)
    )
    bond_return = np.full(len(frame), np.nan, dtype="float64")
    stock_return = np.full(len(frame), np.nan, dtype="float64")
    bond_return[valid] = np.log(bond_close[valid] / bond_previous[valid])
    stock_return[valid] = np.log(stock_close[valid] / stock_previous[valid])
    return bond_return, stock_return


def _tail_dependence_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Compute conditional empirical-copula tails, fail-closed on bad terminal state."""

    out = {signal: float("nan") for signal in _SIGNALS}
    bond_return, stock_return = _joint_log_returns(frame.tail(_WINDOW))
    if (
        len(bond_return) == 0
        or not np.isfinite(bond_return[-1])
        or not np.isfinite(stock_return[-1])
    ):
        return out
    finite = np.isfinite(bond_return) & np.isfinite(stock_return)
    if int(finite.sum()) < _MIN_FINITE_OBSERVATIONS:
        return out
    bond = bond_return[finite]
    stock = stock_return[finite]
    if float(np.ptp(bond)) <= _EPS or float(np.ptp(stock)) <= _EPS:
        return out
    denominator = float(len(bond) + 1)
    bond_rank = rankdata(bond, method="average") / denominator
    stock_rank = rankdata(stock, method="average") / denominator
    upper_stock = stock_rank >= _UPPER_TAIL
    lower_stock = stock_rank <= _LOWER_TAIL
    if int(upper_stock.sum()) < _MIN_TAIL_OBSERVATIONS or int(lower_stock.sum()) < _MIN_TAIL_OBSERVATIONS:
        return out
    upper = float(np.mean(bond_rank[upper_stock] >= _UPPER_TAIL))
    lower = float(np.mean(bond_rank[lower_stock] <= _LOWER_TAIL))
    if not np.isfinite(upper) or not np.isfinite(lower):
        return out
    out["bsct_upper_tail_dependence60"] = upper
    out["bsct_lower_tail_dependence60"] = lower
    out["bsct_tail_dependence_asymmetry60"] = upper - lower
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache all signals from one strict-prior score-date context."""

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
        history, anchor = _aligned_history(ctx, score_date=score_date)
        groups = {
            str(code): group
            for code, group in history.groupby("code", sort=False)
            if anchor is not None and pd.Timestamp(group["trade_date"].max()).normalize() == anchor
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _SIGNALS})
            group = groups.get(rank_coupling_v1._panel_code(raw_code))
            if group is not None:
                row.update(_tail_dependence_metrics(group))
            rows.append(row)
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
class FactorMiningDailyBondStockCopulaTailDependenceV1(Factor):
    """Research-only strict-prior empirical-copula tail-dependence kernel."""

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
    "FactorMiningDailyBondStockCopulaTailDependenceV1",
    "daily_bond_stock_copula_tail_dependence_catalog",
    "factor_mining_catalog",
]
