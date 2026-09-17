"""Research-only strict-T-1 market-breadth regime relation factor.

The factor asks a temporal, market-relative question: over the latest 60
completed daily sessions, how does a bond's *cross-sectional daily-return
rank* differ between the low- and high-breadth market regimes defined inside
that same window?  It is neither a raw return, a static market breadth value,
nor a pool-filtered statistic.

Every daily-price row is filtered strictly before the score date.  Daily
return ranks and market breadth are calculated over the full valid convertible
bond source cross-section.  The later fixed screen alone applies ``o_0005``.
Absent history remains absent after a full-session reindex, and a security
without a valid terminal strict-prior rank fails closed to ``NaN``.
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


KERNEL_NAME = "factor_mining_daily_breadth_regime_relation_v1"
CATALOG_VERSION = "20260803_daily_breadth_regime_relation_v1"
_LOOKBACK_DAYS = 65
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_MIN_REGIME_OBSERVATIONS = 10
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price")
_SIGNAL = "brr_rank_low_high_spread60"


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only candidate declaration."""

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


_CATALOG = _entries(
    "prior_market_breadth_regime_relation",
    (_SIGNAL,),
    "A security's historical relative-return-rank response to endogenous low "
    "versus high full-market breadth regimes captures conditional market "
    "leadership rather than a raw price return or a breadth level.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS = {
    _SIGNAL: (
        "mean(rank_cs(log(close/prev_close)) | breadth <= q25(breadth)) minus "
        "mean(rank_cs(log(close/prev_close)) | breadth >= q75(breadth)) across "
        "the latest up-to-60 strict-prior source sessions, where breadth is the "
        "full-market fraction of valid daily returns above zero."
    )
}


def daily_breadth_regime_relation_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only one-factor family catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_breadth_regime_relation_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return normalized unique daily-price rows strictly before score date."""

    source = "market_cbond.daily_price"
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    required = ("trade_date", "code", "exchange_code", *_PRICE_FIELDS)
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
            f"{KERNEL_NAME} daily_price has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)


def _rank_and_breadth(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Compute full-market daily return ranks and valid-return breadth."""

    out = frame.copy()
    close = pd.to_numeric(out["close_price"], errors="coerce")
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    valid = close.gt(_EPS) & previous.gt(_EPS)
    out["return_value"] = np.nan
    out.loc[valid, "return_value"] = np.log(close.loc[valid] / previous.loc[valid])
    out["return_rank"] = out.groupby("trade_date", sort=False)["return_value"].rank(
        method="average",
        pct=True,
    )
    valid_return = out["return_value"].notna()
    daily = pd.DataFrame(
        {
            "valid": valid_return.groupby(out["trade_date"], sort=False).sum(),
            "positive": (
                valid_return & out["return_value"].gt(0.0)
            ).groupby(out["trade_date"], sort=False).sum(),
        }
    )
    breadth = daily["positive"].div(daily["valid"].replace(0, np.nan))
    return out, breadth.astype("float64")


def _regime_spread(rank_values: np.ndarray, breadth_values: np.ndarray) -> float:
    """Return low-breadth minus high-breadth conditional mean rank."""

    if not len(rank_values) or not (
        np.isfinite(rank_values[-1]) and np.isfinite(breadth_values[-1])
    ):
        return float("nan")
    valid = np.isfinite(rank_values) & np.isfinite(breadth_values)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    breadth = breadth_values[valid]
    rank = rank_values[valid]
    low_cut, high_cut = np.quantile(breadth, [0.25, 0.75])
    low = breadth <= low_cut
    high = breadth >= high_cut
    if int(low.sum()) < _MIN_REGIME_OBSERVATIONS or int(high.sum()) < _MIN_REGIME_OBSERVATIONS:
        return float("nan")
    value = float(rank[low].mean() - rank[high].mean())
    return value if np.isfinite(value) else float("nan")


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache the full-market breadth relationship for one score date."""

    score_date = rank_coupling_v1._score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    output_index = rank_coupling_v1._output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=[_SIGNAL], dtype="float64")
    else:
        history = _strict_history(ctx, score_date=score_date)
        if history.empty:
            built = pd.DataFrame(index=output_index, columns=[_SIGNAL], dtype="float64")
        else:
            ranked, breadth = _rank_and_breadth(history)
            sessions = pd.DatetimeIndex(sorted(ranked["trade_date"].unique())[-_WINDOW:])
            rank_frame = ranked.pivot(
                index="trade_date",
                columns="code",
                values="return_rank",
            ).reindex(sessions)
            breadth_values = breadth.reindex(sessions).to_numpy(dtype="float64")
            rows: list[dict[str, object]] = []
            for dt, raw_code in output_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code, _SIGNAL: float("nan")}
                code = rank_coupling_v1._panel_code(raw_code)
                if code in rank_frame.columns:
                    values = rank_frame[code].to_numpy(dtype="float64")
                    row[_SIGNAL] = _regime_spread(values, breadth_values)
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
class FactorMiningDailyBreadthRegimeRelationV1(Factor):
    """Research-only strict-prior market-breadth regime relationship kernel."""

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
            )
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
    "FactorMiningDailyBreadthRegimeRelationV1",
    "daily_breadth_regime_relation_catalog",
    "factor_mining_catalog",
]
