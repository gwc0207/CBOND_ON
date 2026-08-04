"""Research-only strict-prior daily OHLC wick/path-asymmetry factors.

The catalogue is deliberately separate from the live factor registry import.
It uses only completed daily-price rows strictly before the T1430 score date;
the fixed T-1 ``o_0005`` universe and same-day 14:42 label are applied later
by the research screen, never by this factor implementation.
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


KERNEL_NAME = "factor_mining_daily_ohlc_wick_path_asymmetry_v1"
CATALOG_VERSION = "20260803_daily_ohlc_wick_path_asymmetry_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_MIN_DIRECTIONAL_OBSERVATIONS = 8
_EPS = 1e-12
_PRICE_FIELDS = (
    "prev_close_price",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
)
_SIGNALS = (
    "dohw_mean_wick_asymmetry60",
    "dohw_intraday_sign_range_asymmetry60",
)


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CATALOG = _entries(
    "prior_daily_ohlc_wick_path_asymmetry",
    _SIGNALS,
    "The historical imbalance between upper and lower daily wicks, and the"
    " range asymmetry conditional on completed intraday direction, describe"
    " prior price-auction shape rather than a level, raw return, or volume"
    " factor.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "dohw_mean_wick_asymmetry60": (
        "Mean over the latest up-to-60 strict-prior daily sessions of"
        " (high-max(open,close)-[min(open,close)-low])/(high-low), requiring"
        " 45 finite observations and a finite terminal daily OHLC state."
    ),
    "dohw_intraday_sign_range_asymmetry60": (
        "Over the latest up-to-60 strict-prior daily sessions, mean"
        " log(high/low) on positive log(close/open) days minus its mean on"
        " negative log(close/open) days, requiring 8 observations per"
        " direction, 45 finite OHLC states, and a finite terminal state."
    ),
}


def daily_ohlc_wick_path_asymmetry_catalog() -> tuple[CatalogEntry, ...]:
    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    return daily_ohlc_wick_path_asymmetry_catalog()


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


def _with_ohlc_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    opened = pd.to_numeric(out["open_price"], errors="coerce")
    high = pd.to_numeric(out["high_price"], errors="coerce")
    low = pd.to_numeric(out["low_price"], errors="coerce")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    width = high - low
    valid = (
        opened.gt(_EPS)
        & high.gt(_EPS)
        & low.gt(_EPS)
        & close.gt(_EPS)
        & previous.gt(_EPS)
        & width.gt(_EPS)
        & high.ge(opened)
        & high.ge(close)
        & low.le(opened)
        & low.le(close)
    )
    out["wick_asymmetry"] = np.nan
    out["intraday_return"] = np.nan
    out["intraday_range"] = np.nan
    if bool(valid.any()):
        top = np.maximum(opened.loc[valid], close.loc[valid])
        bottom = np.minimum(opened.loc[valid], close.loc[valid])
        valid_width = width.loc[valid]
        out.loc[valid, "wick_asymmetry"] = (
            (high.loc[valid] - top - (bottom - low.loc[valid])) / valid_width
        )
        out.loc[valid, "intraday_return"] = np.log(close.loc[valid] / opened.loc[valid])
        out.loc[valid, "intraday_range"] = np.log(high.loc[valid] / low.loc[valid])
    return out


def _recent_history(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    positions: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return frame.iloc[0:0].copy()
    session_positions = frame["trade_date"].map(positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(session_positions, positions[anchor] - _WINDOW + 1, side="left"))
    return frame.iloc[first:]


def _mean_wick_asymmetry(recent: pd.DataFrame) -> float:
    values = pd.to_numeric(recent["wick_asymmetry"], errors="coerce").to_numpy(dtype="float64")
    if not len(values) or not np.isfinite(values[-1]):
        return float("nan")
    valid = np.isfinite(values)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    value = float(values[valid].mean())
    return value if np.isfinite(value) else float("nan")


def _intraday_sign_range_asymmetry(recent: pd.DataFrame) -> float:
    intraday_return = pd.to_numeric(
        recent["intraday_return"], errors="coerce"
    ).to_numpy(dtype="float64")
    intraday_range = pd.to_numeric(
        recent["intraday_range"], errors="coerce"
    ).to_numpy(dtype="float64")
    if (
        not len(intraday_return)
        or not np.isfinite(intraday_return[-1])
        or not np.isfinite(intraday_range[-1])
    ):
        return float("nan")
    valid = np.isfinite(intraday_return) & np.isfinite(intraday_range)
    if int(valid.sum()) < _MIN_OBSERVATIONS:
        return float("nan")
    positive = valid & (intraday_return > 0.0)
    negative = valid & (intraday_return < 0.0)
    if (
        int(positive.sum()) < _MIN_DIRECTIONAL_OBSERVATIONS
        or int(negative.sum()) < _MIN_DIRECTIONAL_OBSERVATIONS
    ):
        return float("nan")
    value = float(intraday_range[positive].mean() - intraday_range[negative].mean())
    return value if np.isfinite(value) else float("nan")


def _metrics(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    positions: dict[pd.Timestamp, int],
) -> dict[str, float]:
    recent = _recent_history(frame, anchor=anchor, positions=positions)
    return {
        "dohw_mean_wick_asymmetry60": _mean_wick_asymmetry(recent),
        "dohw_intraday_sign_range_asymmetry60": _intraday_sign_range_asymmetry(recent),
    }


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
            featured = _with_ohlc_features(history)
            anchor = pd.Timestamp(featured["trade_date"].max()).normalize()
            dates = sorted(pd.Timestamp(day).normalize() for day in featured["trade_date"].unique())
            positions = {day: number for number, day in enumerate(dates)}
            groups = {
                str(code): group
                for code, group in featured.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in out_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code}
                row.update({signal: float("nan") for signal in _SIGNALS})
                group = groups.get(rank_coupling_v1._panel_code(raw_code))
                if group is not None:
                    row.update(_metrics(group, anchor=anchor, positions=positions))
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
class FactorMiningDailyOhlcWickPathAsymmetryV1(Factor):
    """Research-only strict-prior daily OHLC price-auction shape kernel."""

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
    "FactorMiningDailyOhlcWickPathAsymmetryV1",
    "daily_ohlc_wick_path_asymmetry_catalog",
    "factor_mining_catalog",
]
