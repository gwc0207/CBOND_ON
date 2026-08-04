"""Research-only strict-T-1 observable-market-seasoning factors.

This family measures whether an instrument has been continuously observable
and actively traded in the recent completed daily-price history.  It is not a
static issuance-age or maturity proxy: missing source sessions and inactive
completed days are retained as information.  The module consumes only
``market_cbond.daily_price`` rows strictly before the score date and is not
imported into a live profile.

For every output security, the latest available strict-prior market session
must contain a valid close.  This prevents an old daily row from being carried
forward as if it were a current observable state.  Within the 60-session
window, an absent row remains missing after reindexing and therefore lowers
the density / terminates the activity streak rather than being filled.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling_v1,
)


KERNEL_NAME = "factor_mining_daily_observable_seasoning_v1"
CATALOG_VERSION = "20260803_daily_observable_seasoning_v1"
_LOOKBACK_DAYS = 65
_WINDOW = 60
_EPS = 1e-12
_PRICE_FIELDS = ("close_price", "amount")


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only factor declaration."""

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
    "osa_terminal_amount_streak60",
    "osa_observation_density60",
)
_CATALOG = _entries(
    "prior_observable_market_seasoning",
    _SIGNALS,
    "The continuity of completed observable and actively traded sessions is a "
    "market-seasoning state that is distinct from a price, premium, duration, "
    "or raw liquidity-level transform.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "osa_terminal_amount_streak60": (
        "Number of consecutive strict-prior source sessions, capped at 60, with "
        "positive completed amount ending at the latest strict-prior valid-close "
        "session; an observed but zero/nonpositive terminal amount has value zero."
    ),
    "osa_observation_density60": (
        "Fraction of the latest up-to-60 strict-prior full market sessions with a "
        "valid positive close for the security, conditional on a valid latest "
        "strict-prior close."
    ),
}


def daily_observable_seasoning_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_observable_seasoning_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return unique normalized daily-price rows strictly before ``score_date``."""

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
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _valid_close(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > _EPS)


def _positive_amount(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > _EPS)


def _terminal_streak(active: np.ndarray) -> float:
    """Return the terminal run length, capped by the supplied 60-session window."""

    length = 0
    for value in active[::-1]:
        if not bool(value):
            break
        length += 1
    return float(length)


def _signal_values(
    frame: pd.DataFrame,
    *,
    sessions: pd.DatetimeIndex,
    anchor: pd.Timestamp,
) -> dict[str, float]:
    """Compute both members on a full-session, strict-prior window."""

    out = {signal: float("nan") for signal in _SIGNALS}
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return out
    recent = frame.set_index("trade_date").reindex(sessions).tail(_WINDOW)
    close = pd.to_numeric(recent["close_price"], errors="coerce").to_numpy(dtype="float64")
    amount = pd.to_numeric(recent["amount"], errors="coerce").to_numpy(dtype="float64")
    observed = _valid_close(close)
    if not len(observed) or not bool(observed[-1]):
        return out
    active = _positive_amount(amount)
    out["osa_terminal_amount_streak60"] = _terminal_streak(active)
    out["osa_observation_density60"] = float(np.mean(observed))
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache all family outputs for one strict score-date context."""

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
        history = _strict_history(ctx, score_date=score_date)
        if history.empty:
            built = pd.DataFrame(index=output_index, columns=_SIGNALS, dtype="float64")
        else:
            sessions = pd.DatetimeIndex(sorted(history["trade_date"].unique()))
            anchor = pd.Timestamp(sessions.max()).normalize()
            groups = {
                str(code): group
                for code, group in history.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in output_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code}
                row.update({signal: float("nan") for signal in _SIGNALS})
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    row.update(_signal_values(group, sessions=sessions, anchor=anchor))
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
class FactorMiningDailyObservableSeasoningV1(Factor):
    """Research-only strict-prior observable market-seasoning kernel."""

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
    "FactorMiningDailyObservableSeasoningV1",
    "daily_observable_seasoning_catalog",
    "factor_mining_catalog",
]
