"""Research-only strict-T-1 daily relative-rank coupling catalogue.

The catalogue deliberately measures a bond's *historical relation* between its
own daily cross-sectional return rank and cross-sectional amount rank.  Each
historical rank is formed from every valid convertible-bond daily-price row
visible on that historical date; the ``o_0005`` universe is never an input to
the factor.  The later research screen alone applies its strict T-1 universe.

Only rows strictly before the score date can affect the output.  A bond needs
an observed, finite row at the latest strict-prior source session and at least
45 jointly finite observations from the most recent 60 source sessions.  A
missing or invalid terminal row therefore fails closed rather than carrying a
stale relation forward.  This module is research-only: it has no file I/O,
database I/O, label, mask, model, PnL, scheduler, live configuration, or
production FactorStore dependency.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    DailyFactorRequirement,
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)


KERNEL_NAME = "factor_mining_daily_relative_rank_coupling_v1"
CATALOG_VERSION = "20260803_daily_relative_rank_coupling_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_JOINT_OBSERVATIONS = 45
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price", "amount")
_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


@dataclass(frozen=True)
class CatalogEntry:
    """One explicit research-only factor candidate."""

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


_SIGNALS = ("drrc_return_amount_rank_spearman60",)
_CATALOG = _entries(
    "prior_relative_return_amount_rank_coupling",
    _SIGNALS,
    "A bond's strict-prior coupling between its return standing and its amount"
    " standing in the observable convertible-bond market is a relative flow"
    " participation state, not a raw return, amount, turnover, or volatility level.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "drrc_return_amount_rank_spearman60": (
        "Pearson correlation (equivalently Spearman-style rank coupling) between"
        " daily cross-sectional percentile ranks of log(close/prev_close) and"
        " log(amount), on the latest up-to-60 strict-prior daily-price source"
        " sessions; every historical rank uses the full valid daily convertible-bond"
        " cross-section, and at least 45 joint observations plus a finite latest"
        " strict-prior row are required."
    ),
}


def daily_relative_rank_coupling_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research expansion-runner entrypoint."""

    return daily_relative_rank_coupling_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series) -> pd.Series:
    """Normalize a daily source code only with an explicit valid exchange."""

    def _one(value: object, exchange: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().upper()
        if not text or text == "NAN":
            return ""
        if text.endswith(".0"):
            text = text[:-2]
        if "." in text:
            bare, suffix = text.rsplit(".", 1)
            suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
            if bare and suffix in _MARKET_EXCHANGES:
                return f"{bare}.{suffix}"
        raw_exchange = "" if pd.isna(exchange) else str(exchange).strip().upper()
        suffix = _EXCHANGE_ALIASES.get(raw_exchange, raw_exchange)
        return f"{text}.{suffix}" if suffix in _MARKET_EXCHANGES else ""

    return pd.Series(
        [_one(value, exchange) for value, exchange in zip(values, exchanges, strict=False)],
        index=values.index,
        dtype="string",
    )


def _panel_code(value: object) -> str:
    return str(_canonical_market_code(pd.Series([value]), pd.Series([""])).iloc[0])


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        score_date = pd.Timestamp(raw_day)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique_dates = pd.Index(dates[dates.notna()]).unique()
    if len(unique_dates) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique_dates[0]).normalize()


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"])
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return unique all-market daily rows strictly prior to the score day."""

    source = "market_cbond.daily_price"
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *_PRICE_FIELDS), source=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *_PRICE_FIELDS]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
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


def _with_historical_cross_sectional_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    """Add date-local ranks before any output-code filtering occurs."""

    out = frame.copy()
    previous = pd.to_numeric(out["prev_close_price"], errors="coerce")
    close = pd.to_numeric(out["close_price"], errors="coerce")
    amount = pd.to_numeric(out["amount"], errors="coerce")
    out["log_return"] = np.nan
    valid_return = previous.gt(_EPS) & close.gt(_EPS)
    out.loc[valid_return, "log_return"] = np.log(
        close.loc[valid_return] / previous.loc[valid_return]
    )
    out["log_amount"] = np.nan
    valid_amount = amount.gt(_EPS)
    out.loc[valid_amount, "log_amount"] = np.log(amount.loc[valid_amount])
    out["return_rank"] = out.groupby("trade_date", sort=False)["log_return"].rank(
        method="average", pct=True
    )
    out["amount_rank"] = out.groupby("trade_date", sort=False)["log_amount"].rank(
        method="average", pct=True
    )
    return out


def _rank_correlation(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    date_positions: dict[pd.Timestamp, int],
) -> float:
    """Measure rank coupling in a fixed latest-60-source-session window."""

    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return float("nan")
    anchor_position = date_positions[anchor]
    session_positions = frame["trade_date"].map(date_positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(session_positions, anchor_position - _WINDOW + 1, side="left"))
    recent = frame.iloc[first:]
    if recent.empty:
        return float("nan")
    x = pd.to_numeric(recent["return_rank"], errors="coerce").to_numpy(dtype="float64")
    y = pd.to_numeric(recent["amount_rank"], errors="coerce").to_numpy(dtype="float64")
    if not (np.isfinite(x[-1]) and np.isfinite(y[-1])):
        return float("nan")
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < _MIN_JOINT_OBSERVATIONS:
        return float("nan")
    x = x[valid]
    y = y[valid]
    x_centered = x - float(x.mean())
    y_centered = y - float(y.mean())
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    value = float(np.dot(x_centered, y_centered) / denominator)
    return value if np.isfinite(value) else float("nan")


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache all catalogue values for the score-day context."""

    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = _output_index(ctx, score_date)
    if score_date is None or out_index.empty:
        built = pd.DataFrame(index=out_index, columns=_SIGNALS, dtype="float64")
    else:
        history = _strict_history(ctx, score_date=score_date)
        if history.empty:
            built = pd.DataFrame(index=out_index, columns=_SIGNALS, dtype="float64")
        else:
            ranked = _with_historical_cross_sectional_ranks(history)
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
                group = groups.get(_panel_code(raw_code))
                if group is not None:
                    value = _rank_correlation(group, anchor=anchor, date_positions=date_positions)
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
class FactorMiningDailyRelativeRankCouplingV1(Factor):
    """Research-only strict-prior daily cross-sectional rank-coupling kernel."""

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
    "FactorMiningDailyRelativeRankCouplingV1",
    "daily_relative_rank_coupling_catalog",
    "factor_mining_catalog",
]
