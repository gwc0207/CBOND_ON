"""Research-only strict-T-1 bond/stock cross-sectional-rank concordance.

For every completed historical session, this family first ranks convertible
bond returns in the full bond cross-section and ranks returns of the distinct
mapped underlying stocks in their own cross-section.  It then measures each
bond's historical rank concordance with its mapped stock.  This is neither a
raw return gap nor a linear beta: the inputs are date-local relative standings
and all temporal aggregation is strictly before the score date.

The module consumes only the supplied daily context.  It never reads files,
uses labels/PnL, accesses a database, or touches live/model/scheduler paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling_v1,
)


KERNEL_NAME = "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
CATALOG_VERSION = "20260803_daily_bond_stock_cross_sectional_rank_concordance_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_JOINT_OBSERVATIONS = 45
_MIN_TAIL_OBSERVATIONS = 8
_LOWER_TAIL = 0.25
_UPPER_TAIL = 0.75
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price")
_BASE_FIELDS = ("stock_code", "stk_prev_close_price", "stk_close_price")


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only candidate in a genuine information family."""

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
    "bssrc_upper_rank_tail_alignment60",
    "bssrc_bond_stock_rank_correlation60",
    "bssrc_lower_rank_tail_alignment60",
)
_CATALOG = _entries(
    "prior_bond_stock_cross_sectional_rank_concordance",
    _SIGNALS,
    "Historical concordance between a convertible bond's full-market return "
    "standing and its mapped underlying stock's distinct-underlying return "
    "standing captures relative cross-asset participation beyond a raw gap, "
    "linear beta, or static contract attribute.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "bssrc_upper_rank_tail_alignment60": (
        "Across the latest up-to-60 strict-prior sessions, empirical "
        "P(bond full-bond-market return rank >= q75 | mapped-stock distinct-stock "
        "return rank >= q75), requiring 45 joint observations and at least eight "
        "stock upper-tail observations."
    ),
    "bssrc_bond_stock_rank_correlation60": (
        "Pearson correlation across the latest up-to-60 strict-prior sessions "
        "of bond full-bond-market return percentile rank and mapped-stock "
        "distinct-underlying return percentile rank, requiring 45 finite pairs."
    ),
    "bssrc_lower_rank_tail_alignment60": (
        "Across the latest up-to-60 strict-prior sessions, empirical "
        "P(bond full-bond-market return rank <= q25 | mapped-stock distinct-stock "
        "return rank <= q25), requiring 45 joint observations and at least eight "
        "stock lower-tail observations."
    ),
}


def daily_bond_stock_cross_sectional_rank_concordance_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_bond_stock_cross_sectional_rank_concordance_catalog()


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
    """Return normalized source rows strictly before score date, fail-closed."""

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
    return frame.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)


def _strict_histories(
    ctx: FactorComputeContext,
    *,
    score_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp | None]:
    """Load both strict-prior histories only when their global anchors agree."""

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
        return price, base, None
    price_anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    base_anchor = pd.Timestamp(base["trade_date"].max()).normalize()
    if price_anchor != base_anchor:
        return price.iloc[0:0].copy(), base.iloc[0:0].copy(), None
    return price, base, price_anchor


def _bond_rank_history(price: pd.DataFrame) -> pd.DataFrame:
    """Add one full-market bond-return rank per historical source day."""

    out = price.loc[:, ["trade_date", "code", *_PRICE_FIELDS]].copy()
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    out["bond_return"] = np.nan
    valid = previous.gt(_EPS) & close.gt(_EPS)
    out.loc[valid, "bond_return"] = np.log(close.loc[valid] / previous.loc[valid])
    out["bond_rank"] = out.groupby("trade_date", sort=False)["bond_return"].rank(
        method="average", pct=True
    )
    return out.loc[:, ["trade_date", "code", "bond_rank"]]


def _stock_rank_history(base: pd.DataFrame) -> pd.DataFrame:
    """Map each bond to a rank among distinct mapped underlying stocks per day."""

    out = base.loc[:, ["trade_date", "code", "exchange_code", *_BASE_FIELDS]].copy()
    out["stock_code"] = rank_coupling_v1._canonical_market_code(
        out["stock_code"], out["exchange_code"]
    )
    previous = pd.to_numeric(out["stk_prev_close_price"], errors="coerce")
    close = pd.to_numeric(out["stk_close_price"], errors="coerce")
    out["stock_return"] = np.nan
    valid = previous.gt(_EPS) & close.gt(_EPS)
    out.loc[valid, "stock_return"] = np.log(close.loc[valid] / previous.loc[valid])
    out = out.loc[out["stock_code"].notna() & out["stock_code"].ne("")].copy()
    if out.empty:
        return pd.DataFrame(columns=["trade_date", "code", "stock_rank"])

    stock_rows: list[dict[str, object]] = []
    for (trade_date, stock_code), group in out.groupby(["trade_date", "stock_code"], sort=False):
        values = pd.to_numeric(group["stock_return"], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        # Shared underlying rows must represent one source value.  If any row
        # is invalid or two finite values disagree, fail that stock/day closed
        # rather than choosing an arbitrary convertible's copy.
        if len(finite) != len(values):
            value = float("nan")
        elif len(finite) and float(np.ptp(finite)) <= _EPS:
            value = float(finite[0])
        elif len(finite):
            raise ValueError(
                f"{KERNEL_NAME} inconsistent strict-prior underlying return "
                f"for {stock_code} on {pd.Timestamp(trade_date).date().isoformat()}"
            )
        else:
            value = float("nan")
        stock_rows.append(
            {"trade_date": trade_date, "stock_code": stock_code, "stock_return": value}
        )
    stock = pd.DataFrame(stock_rows)
    stock["stock_rank"] = stock.groupby("trade_date", sort=False)["stock_return"].rank(
        method="average", pct=True
    )
    return out.loc[:, ["trade_date", "code", "stock_code"]].merge(
        stock.loc[:, ["trade_date", "stock_code", "stock_rank"]],
        on=["trade_date", "stock_code"],
        how="left",
        validate="many_to_one",
    ).loc[:, ["trade_date", "code", "stock_rank"]]


def _rank_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate the three relative-rank mechanisms over a terminal 60-day path."""

    out = {signal: float("nan") for signal in _SIGNALS}
    recent = frame.tail(_WINDOW)
    bond = pd.to_numeric(recent["bond_rank"], errors="coerce").to_numpy(dtype="float64")
    stock = pd.to_numeric(recent["stock_rank"], errors="coerce").to_numpy(dtype="float64")
    if len(bond) == 0 or not (np.isfinite(bond[-1]) and np.isfinite(stock[-1])):
        return out
    valid = np.isfinite(bond) & np.isfinite(stock)
    if int(valid.sum()) < _MIN_JOINT_OBSERVATIONS:
        return out
    bond = bond[valid]
    stock = stock[valid]
    centered_bond = bond - float(bond.mean())
    centered_stock = stock - float(stock.mean())
    denominator = float(
        np.sqrt(np.dot(centered_bond, centered_bond) * np.dot(centered_stock, centered_stock))
    )
    if not np.isfinite(denominator) or denominator <= _EPS:
        return out
    upper_stock = stock >= _UPPER_TAIL
    lower_stock = stock <= _LOWER_TAIL
    if (
        int(upper_stock.sum()) < _MIN_TAIL_OBSERVATIONS
        or int(lower_stock.sum()) < _MIN_TAIL_OBSERVATIONS
    ):
        return out
    correlation = float(np.dot(centered_bond, centered_stock) / denominator)
    upper_alignment = float(np.mean(bond[upper_stock] >= _UPPER_TAIL))
    lower_alignment = float(np.mean(bond[lower_stock] <= _LOWER_TAIL))
    if not all(np.isfinite(value) for value in (correlation, upper_alignment, lower_alignment)):
        return out
    out.update(
        {
            "bssrc_upper_rank_tail_alignment60": upper_alignment,
            "bssrc_bond_stock_rank_correlation60": correlation,
            "bssrc_lower_rank_tail_alignment60": lower_alignment,
        }
    )
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache every family member for a strict score-date context."""

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
        price, base, anchor = _strict_histories(ctx, score_date=score_date)
        if anchor is None:
            built = pd.DataFrame(index=output_index, columns=_SIGNALS, dtype="float64")
        else:
            ranked = _bond_rank_history(price).merge(
                _stock_rank_history(base),
                on=["trade_date", "code"],
                how="inner",
                validate="one_to_one",
            ).sort_values(["code", "trade_date"], kind="mergesort")
            groups = {
                str(code): group
                for code, group in ranked.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in output_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code}
                row.update({signal: float("nan") for signal in _SIGNALS})
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    row.update(_rank_metrics(group))
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
class FactorMiningDailyBondStockCrossSectionalRankConcordanceV1(Factor):
    """Research-only strict-prior bond/stock relative-rank kernel."""

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
    "FactorMiningDailyBondStockCrossSectionalRankConcordanceV1",
    "daily_bond_stock_cross_sectional_rank_concordance_catalog",
    "factor_mining_catalog",
]
