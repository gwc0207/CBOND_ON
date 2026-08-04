"""Research-only strict-T-1 relative rank-tail contradiction catalogue.

The signal counts a bond's completed historical occurrences of a weak
cross-sectional return standing together with an exceptionally high
cross-sectional amount standing.  It is a nonlinear tail-state frequency,
not a linear return/amount correlation or a raw activity measure.  Every daily
rank is formed across the full valid daily convertible-bond market observed on
that historical date; the ``o_0005`` universe is never an input to the factor.

The implementation reuses only the strict-prior all-market daily-price helpers
from the independently tested rank-coupling module.  It has no file I/O,
database I/O, labels, masks, model, PnL, scheduler, configuration, or
production FactorStore dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling,
)


KERNEL_NAME = "factor_mining_daily_relative_rank_tail_contradiction_v1"
CATALOG_VERSION = "20260803_daily_relative_rank_tail_contradiction_v1"
_WINDOW = 60
_MIN_JOINT_OBSERVATIONS = 45
_LOW_RETURN_RANK = 0.20
_HIGH_AMOUNT_RANK = 0.80
_INDEPENDENCE_BASELINE = _LOW_RETURN_RANK * (1.0 - _HIGH_AMOUNT_RANK)
_SIGNAL = "drrq_return_amount_opposite_tail_excess60"


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only tail-state candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


_CATALOG = (
    CatalogEntry(
        family="prior_relative_return_flow_tail_contradiction",
        signal=_SIGNAL,
        kernel=KERNEL_NAME,
        hypothesis=(
            "The completed frequency of a bottom-return-rank / top-amount-rank"
            " contradiction is a nonlinear adverse-flow state rather than a raw"
            " return, amount, volatility, or linear return-flow coupling."
        ),
    ),
)

FORMULAS: dict[str, str] = {
    _SIGNAL: (
        "On the latest up-to-60 strict-prior daily-price source sessions, mean"
        " I(return cross-sectional percentile rank <= 0.20 and amount"
        " cross-sectional percentile rank >= 0.80) minus 0.04, requiring at"
        " least 45 jointly finite sessions and finite terminal ranks.  Historical"
        " ranks use all valid date-local convertible-bond daily-price rows."
    ),
}


def daily_relative_rank_tail_contradiction_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research expansion-runner entrypoint."""

    return daily_relative_rank_tail_contradiction_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    if signal != _SIGNAL:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return _CATALOG[0]


def _opposite_tail_excess(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    date_positions: dict[pd.Timestamp, int],
) -> float:
    """Return a terminal-anchored rare-opposition frequency excess."""

    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return float("nan")
    anchor_position = date_positions[anchor]
    positions = frame["trade_date"].map(date_positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(positions, anchor_position - _WINDOW + 1, side="left"))
    recent = frame.iloc[first:]
    if recent.empty:
        return float("nan")
    returns = pd.to_numeric(recent["return_rank"], errors="coerce").to_numpy(dtype="float64")
    amounts = pd.to_numeric(recent["amount_rank"], errors="coerce").to_numpy(dtype="float64")
    if not (np.isfinite(returns[-1]) and np.isfinite(amounts[-1])):
        return float("nan")
    valid = np.isfinite(returns) & np.isfinite(amounts)
    if int(valid.sum()) < _MIN_JOINT_OBSERVATIONS:
        return float("nan")
    value = float(
        np.mean((returns[valid] <= _LOW_RETURN_RANK) & (amounts[valid] >= _HIGH_AMOUNT_RANK))
        - _INDEPENDENCE_BASELINE
    )
    return value if np.isfinite(value) else float("nan")


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache the tail-state feature for the current score day."""

    score_date = rank_coupling._score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = rank_coupling._output_index(ctx, score_date)
    if score_date is None or out_index.empty:
        built = pd.DataFrame(index=out_index, columns=[_SIGNAL], dtype="float64")
    else:
        history = rank_coupling._strict_history(ctx, score_date=score_date)
        if history.empty:
            built = pd.DataFrame(index=out_index, columns=[_SIGNAL], dtype="float64")
        else:
            ranked = rank_coupling._with_historical_cross_sectional_ranks(history)
            anchor = pd.Timestamp(ranked["trade_date"].max()).normalize()
            dates = sorted(pd.Timestamp(day).normalize() for day in ranked["trade_date"].unique())
            date_positions = {day: position for position, day in enumerate(dates)}
            groups = {
                str(code): group
                for code, group in ranked.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in out_index:
                value = float("nan")
                group = groups.get(rank_coupling._panel_code(raw_code))
                if group is not None:
                    value = _opposite_tail_excess(
                        group,
                        anchor=anchor,
                        date_positions=date_positions,
                    )
                rows.append({"dt": dt, "code": raw_code, _SIGNAL: value})
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
class FactorMiningDailyRelativeRankTailContradictionV1(Factor):
    """Research-only strict-prior nonlinear rank-tail state kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement(
                "market_cbond.daily_price",
                ("exchange_code", "prev_close_price", "close_price", "amount"),
                75,
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
    "FactorMiningDailyRelativeRankTailContradictionV1",
    "daily_relative_rank_tail_contradiction_catalog",
    "factor_mining_catalog",
]
