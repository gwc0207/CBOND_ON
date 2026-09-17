"""Research-only strict-T-1 daily liquidity-channel composition factors.

This family describes the *joint organisation* of completed convertible-bond
volume, notional, and trade-count changes.  It deliberately uses categorical
information and state entropy rather than another price, premium, volatility,
or raw-liquidity level transform.

Only context-supplied daily-price rows strictly before the score date are
read.  Each security is reindexed to the full supplied source-session calendar
before changes or transitions are calculated, so a missing day cannot be
compressed into a synthetic adjacent observation.  The module is
research-only: no file, database, label, PnL, model, live-config, scheduler,
or production-FactorStore dependency is present.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_relative_rank_coupling_v1 as rank_coupling_v1,
)


KERNEL_NAME = "factor_mining_daily_liquidity_channel_composition_v1"
CATALOG_VERSION = "20260803_daily_liquidity_channel_composition_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_MIN_TRANSITIONS = 40
_EPS = 1e-12
_PRICE_FIELDS = ("volume", "amount", "deal")


@dataclass(frozen=True)
class CatalogEntry:
    """One explicitly registered research candidate."""

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
    "lcc_volume_deal_information60",
    "lcc_size_frequency_coupling60",
    "lcc_three_channel_state_entropy60",
    "lcc_three_channel_transition_entropy60",
    "lcc_amount_trade_size_information60",
)
_CATALOG = _entries(
    "prior_liquidity_channel_composition",
    _SIGNALS,
    "Strict-prior co-organisation of volume, notional, trade count, and average "
    "trade size represents an information/market-participation composition state, "
    "rather than a level of liquidity or a price-path transform.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "lcc_volume_deal_information60": (
        "Normalized mutual information between sign(delta log(volume)) and "
        "sign(delta log(deal)) on the latest up-to-60 strict-prior source sessions, "
        "using a 3x3 Jeffreys-pseudocount table."
    ),
    "lcc_size_frequency_coupling60": (
        "Pearson correlation between delta log(amount/deal) and delta log(deal) "
        "on the latest up-to-60 strict-prior source sessions."
    ),
    "lcc_three_channel_state_entropy60": (
        "Shannon entropy divided by log(27) of the 27 joint signs of delta log "
        "amount, volume, and deal on the latest up-to-60 strict-prior sessions."
    ),
    "lcc_three_channel_transition_entropy60": (
        "Shannon entropy divided by log(729) of adjacent transitions between the "
        "27 joint amount/volume/deal sign states on strict-prior sessions."
    ),
    "lcc_amount_trade_size_information60": (
        "Normalized mutual information between sign(delta log(amount)) and sign(" 
        "delta log(amount/deal)) on the latest up-to-60 strict-prior sessions."
    ),
}


def daily_liquidity_channel_composition_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only family catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return daily_liquidity_channel_composition_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return unique, normalized daily-price rows strictly before score day."""

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
            f"{KERNEL_NAME} daily_price has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _log_positive(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values) & (values > _EPS)
    result[valid] = np.log(values[valid])
    return result


def _change(values: np.ndarray) -> np.ndarray:
    """Use only exact adjacent source sessions; gaps remain NaN."""

    result = np.full(len(values), np.nan, dtype="float64")
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    result[1:][valid] = values[1:][valid] - values[:-1][valid]
    return result


def _sign3(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int8)
    finite = np.isfinite(values)
    result[finite & (values < -_EPS)] = 0
    result[finite & (np.abs(values) <= _EPS)] = 1
    result[finite & (values > _EPS)] = 2
    return result


def _normalized_mutual_information(left: np.ndarray, right: np.ndarray) -> float:
    left_state = _sign3(left)
    right_state = _sign3(right)
    if left_state[-1] < 0 or right_state[-1] < 0:
        return float("nan")
    valid = (left_state >= 0) & (right_state >= 0)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    counts = np.full((3, 3), 0.5, dtype="float64")
    np.add.at(counts, (left_state[valid], right_state[valid]), 1.0)
    probability = counts / float(counts.sum())
    left_probability = probability.sum(axis=1, keepdims=True)
    right_probability = probability.sum(axis=0, keepdims=True)
    value = float(
        np.sum(probability * np.log(probability / (left_probability * right_probability)))
        / math.log(3.0)
    )
    return value if np.isfinite(value) else float("nan")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if not (np.isfinite(left[-1]) and np.isfinite(right[-1])):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    x = left[valid]
    y = right[valid]
    x_centered = x - float(x.mean())
    y_centered = y - float(y.mean())
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    value = float(np.dot(x_centered, y_centered) / denominator)
    return value if np.isfinite(value) else float("nan")


def _joint_states(
    amount_change: np.ndarray,
    volume_change: np.ndarray,
    deal_change: np.ndarray,
) -> np.ndarray:
    amount_state = _sign3(amount_change)
    volume_state = _sign3(volume_change)
    deal_state = _sign3(deal_change)
    return np.where(
        (amount_state >= 0) & (volume_state >= 0) & (deal_state >= 0),
        9 * amount_state + 3 * volume_state + deal_state,
        -1,
    )


def _state_entropy(states: np.ndarray) -> float:
    if states[-1] < 0:
        return float("nan")
    valid = states >= 0
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    counts = np.full(27, 0.5, dtype="float64")
    np.add.at(counts, states[valid], 1.0)
    probability = counts / float(counts.sum())
    value = float(-np.sum(probability * np.log(probability)) / math.log(27.0))
    return value if np.isfinite(value) else float("nan")


def _transition_entropy(states: np.ndarray) -> float:
    if states[-1] < 0:
        return float("nan")
    valid = (states[1:] >= 0) & (states[:-1] >= 0)
    if int(valid.sum()) < _MIN_TRANSITIONS:
        return float("nan")
    counts = np.full((27, 27), 0.5, dtype="float64")
    np.add.at(counts, (states[:-1][valid], states[1:][valid]), 1.0)
    probability = counts / float(counts.sum())
    value = float(-np.sum(probability * np.log(probability)) / math.log(729.0))
    return value if np.isfinite(value) else float("nan")


def _signal_values(frame: pd.DataFrame, *, sessions: pd.DatetimeIndex, anchor: pd.Timestamp) -> dict[str, float]:
    """Calculate all pre-registered composition signals on one source window."""

    out = {signal: float("nan") for signal in _SIGNALS}
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return out
    recent = frame.set_index("trade_date").reindex(sessions).tail(_WINDOW)
    log_amount = _log_positive(
        pd.to_numeric(recent["amount"], errors="coerce").to_numpy(dtype="float64")
    )
    log_volume = _log_positive(
        pd.to_numeric(recent["volume"], errors="coerce").to_numpy(dtype="float64")
    )
    log_deal = _log_positive(
        pd.to_numeric(recent["deal"], errors="coerce").to_numpy(dtype="float64")
    )
    amount_change = _change(log_amount)
    volume_change = _change(log_volume)
    deal_change = _change(log_deal)
    size_change = _change(log_amount - log_deal)
    states = _joint_states(amount_change, volume_change, deal_change)
    out.update(
        {
            "lcc_volume_deal_information60": _normalized_mutual_information(
                volume_change, deal_change
            ),
            "lcc_size_frequency_coupling60": _correlation(size_change, deal_change),
            "lcc_three_channel_state_entropy60": _state_entropy(states),
            "lcc_three_channel_transition_entropy60": _transition_entropy(states),
            "lcc_amount_trade_size_information60": _normalized_mutual_information(
                amount_change, size_change
            ),
        }
    )
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache all family members for a single strict score-date context."""

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
class FactorMiningDailyLiquidityChannelCompositionV1(Factor):
    """Research-only strict-prior liquidity-channel composition kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement(
                "market_cbond.daily_price", ("exchange_code", *_PRICE_FIELDS), _LOOKBACK_DAYS
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
    "FactorMiningDailyLiquidityChannelCompositionV1",
    "daily_liquidity_channel_composition_catalog",
    "factor_mining_catalog",
]
