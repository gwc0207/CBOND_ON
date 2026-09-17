"""Research-only strict-prior capacity-normalized return-rank coupling.

The signal measures whether a bond's completed return rank has historically
co-moved with its amount per remaining issue size rank.  It is distinct from
raw amount, turnover level, or static float capacity: both inputs are ranked
over the full date-local, valid daily-price/daily-base intersection before the
per-security temporal coupling is formed.
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


KERNEL_NAME = "factor_mining_daily_capacity_rank_coupling_v1"
CATALOG_VERSION = "20260803_daily_capacity_rank_coupling_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_JOINT_OBSERVATIONS = 45
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price", "amount")
_BASE_FIELDS = ("remain_size",)
_SIGNAL = "prcn_return_capacity_rank_corr60"


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CATALOG = _entries(
    "prior_capacity_normalized_return_rank_coupling",
    (_SIGNAL,),
    "Historical coupling of completed return standing and amount per remaining"
    " issue-size standing captures relative capital participation rather than"
    " raw amount, float, turnover, or an unnormalized return-flow relation.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS = {
    _SIGNAL: (
        "Pearson correlation across the latest up-to-60 strict-prior sessions"
        " between daily full-market percentile rank(log(close/prev_close)) and"
        " percentile rank(log(amount/remain_size)), requiring 45 finite joint"
        " observations and a finite terminal aligned price/base state."
    )
}


def daily_capacity_rank_coupling_catalog() -> tuple[CatalogEntry, ...]:
    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    return daily_capacity_rank_coupling_catalog()


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
        frame["code"],
        frame["exchange_code"],
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


def _with_daily_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce")
    remain_size = pd.to_numeric(out["remain_size"], errors="coerce")
    out["return_value"] = np.nan
    out["capacity_value"] = np.nan
    valid_return = previous.gt(_EPS) & close.gt(_EPS)
    valid_capacity = amount.gt(_EPS) & remain_size.gt(_EPS)
    out.loc[valid_return, "return_value"] = np.log(
        close.loc[valid_return] / previous.loc[valid_return]
    )
    out.loc[valid_capacity, "capacity_value"] = np.log(
        amount.loc[valid_capacity] / remain_size.loc[valid_capacity]
    )
    out["return_rank"] = out.groupby("trade_date", sort=False)["return_value"].rank(
        method="average",
        pct=True,
    )
    out["capacity_rank"] = out.groupby("trade_date", sort=False)["capacity_value"].rank(
        method="average",
        pct=True,
    )
    return out


def _rank_coupling(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    positions: dict[pd.Timestamp, int],
) -> float:
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return float("nan")
    session_positions = frame["trade_date"].map(positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(session_positions, positions[anchor] - _WINDOW + 1, side="left"))
    recent = frame.iloc[first:]
    left = pd.to_numeric(recent["return_rank"], errors="coerce").to_numpy(dtype="float64")
    right = pd.to_numeric(recent["capacity_rank"], errors="coerce").to_numpy(dtype="float64")
    if not len(left) or not np.isfinite(left[-1]) or not np.isfinite(right[-1]):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < _MIN_JOINT_OBSERVATIONS:
        return float("nan")
    left = left[valid] - float(left[valid].mean())
    right = right[valid] - float(right[valid].mean())
    denominator = float(np.sqrt(np.dot(left, left) * np.dot(right, right)))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    value = float(np.dot(left, right) / denominator)
    return value if np.isfinite(value) else float("nan")


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    score_date = rank_coupling_v1._score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = rank_coupling_v1._output_index(ctx, score_date)
    if score_date is None or out_index.empty:
        built = pd.DataFrame(index=out_index, columns=[_SIGNAL], dtype="float64")
    else:
        history, anchor = _aligned_history(ctx, score_date=score_date)
        if history.empty or anchor is None:
            built = pd.DataFrame(index=out_index, columns=[_SIGNAL], dtype="float64")
        else:
            ranked = _with_daily_ranks(history)
            dates = sorted(pd.Timestamp(day).normalize() for day in ranked["trade_date"].unique())
            positions = {day: number for number, day in enumerate(dates)}
            groups = {
                str(code): group
                for code, group in ranked.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in out_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code, _SIGNAL: float("nan")}
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    row[_SIGNAL] = _rank_coupling(group, anchor=anchor, positions=positions)
                rows.append(row)
            built = (
                pd.DataFrame(rows)
                .set_index(["dt", "code"])[[_SIGNAL]]
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
class FactorMiningDailyCapacityRankCouplingV1(Factor):
    """Research-only strict-prior capacity-normalized rank-coupling kernel."""

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
    "FactorMiningDailyCapacityRankCouplingV1",
    "daily_capacity_rank_coupling_catalog",
    "factor_mining_catalog",
]
