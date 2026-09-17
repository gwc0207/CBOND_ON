"""Research-only strict-T-1 relative return-flow rank-coupling catalogue.

This v2 catalogue supersedes the one-signal v1 only for future research
builds.  It keeps total-notional and average-trade-size coupling in one genuine
family, so their redundancy is evaluated at the within-family 0.80 gate.
Every historical rank uses the full valid date-local daily convertible-bond
market; ``o_0005`` is never a factor input and is applied only by the screen.
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


KERNEL_NAME = "factor_mining_daily_relative_rank_flow_coupling_v2"
CATALOG_VERSION = "20260803_daily_relative_rank_flow_coupling_v2"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_JOINT_OBSERVATIONS = 45
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price", "amount", "deal")
_SIGNALS = (
    "drrc_return_amount_rank_spearman60",
    "drrc_return_trade_size_rank_spearman60",
)
_SIGNAL_TO_RANK = {
    "drrc_return_amount_rank_spearman60": "amount_rank",
    "drrc_return_trade_size_rank_spearman60": "trade_size_rank",
}


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CATALOG = _entries(
    "prior_relative_return_flow_rank_coupling",
    _SIGNALS,
    "Completed return-standing coupling with distinct flow standings is a"
    " relative flow-participation state, not a raw return, amount, turnover,"
    " deal count, or volatility level.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "drrc_return_amount_rank_spearman60": (
        "Pearson correlation of daily cross-sectional percentile ranks of"
        " log(close/prev_close) and log(amount) across the latest up-to-60"
        " strict-prior sessions, with 45 finite joint observations and a finite"
        " terminal state required."
    ),
    "drrc_return_trade_size_rank_spearman60": (
        "Pearson correlation of daily cross-sectional percentile ranks of"
        " log(close/prev_close) and log(amount/deal) across the latest up-to-60"
        " strict-prior sessions, with 45 finite joint observations and a finite"
        " terminal state required."
    ),
}


def daily_relative_rank_flow_coupling_catalog() -> tuple[CatalogEntry, ...]:
    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    return daily_relative_rank_flow_coupling_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    source = "market_cbond.daily_price"
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    required = ("trade_date", "code", "exchange_code", *_PRICE_FIELDS)
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} daily_price missing required columns: {missing}")
    frame = raw.loc[:, list(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = rank_coupling_v1._canonical_market_code(frame["code"], frame["exchange_code"])
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
            f"{KERNEL_NAME} daily_price has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["trade_date", "code"], kind="mergesort").reset_index(drop=True)


def _with_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Form all ranks over each full historical market cross-section."""

    out = frame.copy()
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce")
    deal = pd.to_numeric(out["deal"], errors="coerce")
    out["log_return"] = np.nan
    valid_return = previous.gt(_EPS) & close.gt(_EPS)
    out.loc[valid_return, "log_return"] = np.log(
        close.loc[valid_return] / previous.loc[valid_return]
    )
    out["log_amount"] = np.nan
    valid_amount = amount.gt(_EPS)
    out.loc[valid_amount, "log_amount"] = np.log(amount.loc[valid_amount])
    out["log_trade_size"] = np.nan
    valid_trade_size = valid_amount & deal.gt(_EPS)
    out.loc[valid_trade_size, "log_trade_size"] = np.log(
        amount.loc[valid_trade_size] / deal.loc[valid_trade_size]
    )
    for source, target in (
        ("log_return", "return_rank"),
        ("log_amount", "amount_rank"),
        ("log_trade_size", "trade_size_rank"),
    ):
        out[target] = out.groupby("trade_date", sort=False)[source].rank(method="average", pct=True)
    return out


def _coupling(
    frame: pd.DataFrame,
    *,
    rank_column: str,
    anchor: pd.Timestamp,
    positions: dict[pd.Timestamp, int],
) -> float:
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return float("nan")
    session_positions = frame["trade_date"].map(positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(session_positions, positions[anchor] - _WINDOW + 1, side="left"))
    recent = frame.iloc[first:]
    if recent.empty:
        return float("nan")
    left = pd.to_numeric(recent["return_rank"], errors="coerce").to_numpy(dtype="float64")
    right = pd.to_numeric(recent[rank_column], errors="coerce").to_numpy(dtype="float64")
    if not (np.isfinite(left[-1]) and np.isfinite(right[-1])):
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
        built = pd.DataFrame(index=out_index, columns=_SIGNALS, dtype="float64")
    else:
        history = _strict_history(ctx, score_date=score_date)
        if history.empty:
            built = pd.DataFrame(index=out_index, columns=_SIGNALS, dtype="float64")
        else:
            ranked = _with_ranks(history)
            anchor = pd.Timestamp(ranked["trade_date"].max()).normalize()
            dates = sorted(pd.Timestamp(day).normalize() for day in ranked["trade_date"].unique())
            positions = {day: number for number, day in enumerate(dates)}
            groups = {
                str(code): group
                for code, group in ranked.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in out_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code}
                row.update({signal: float("nan") for signal in _SIGNALS})
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    for signal, rank_column in _SIGNAL_TO_RANK.items():
                        row[signal] = _coupling(
                            group,
                            rank_column=rank_column,
                            anchor=anchor,
                            positions=positions,
                        )
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
class FactorMiningDailyRelativeRankFlowCouplingV2(Factor):
    """Research-only strict-prior daily return-flow rank-coupling kernel."""

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
    "FactorMiningDailyRelativeRankFlowCouplingV2",
    "daily_relative_rank_flow_coupling_catalog",
    "factor_mining_catalog",
]
