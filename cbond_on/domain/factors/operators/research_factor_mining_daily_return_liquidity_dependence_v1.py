"""Research-only strict-T-1 return/liquidity information-dependence factor.

The catalogue measures the mutual information between a convertible bond's
completed daily return sign and the sign of its completed daily amount change.
It deliberately retains the daily session sequence: a missing session is never
compressed into a synthetic adjacent observation.  Every source row on the
score date or later is discarded, and a code missing the newest strict-prior
daily source row fails closed.

This is a research-only catalogue.  It has no live/model registration,
configuration, file I/O, database I/O, mask, label, score, PnL, or scheduler
dependency.
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


KERNEL_NAME = "factor_mining_daily_return_liquidity_dependence_v1"
CATALOG_VERSION = "20260803_daily_return_liquidity_dependence_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_MIN_ADJACENT_PAIRS = 40
_PSEUDOCOUNT = 0.5
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
    """One auditable research-only factor specification."""

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


_SIGNALS = ("rlmi_return_liquidity_sign_mutual_information60",)
_CATALOG = _entries(
    "prior_return_liquidity_information_dependence",
    _SIGNALS,
    "The completed daily return-sign/amount-change-sign dependence can describe a"
    " prior information-absorption regime without using a raw amount level,"
    " a return level, or a future session.",
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "rlmi_return_liquidity_sign_mutual_information60": (
        "Mutual information divided by log(3) between sign(log(close/prev_close)) "
        "and sign(delta log(amount)) on the latest up-to-60 strict-prior daily "
        "sessions, using a 3x3 Jeffreys-pseudocount table. Only consecutive "
        "source sessions are pairs; at least 45 source rows, 40 valid adjacent "
        "pairs, and a valid latest strict-prior pair are required."
    ),
}


def daily_return_liquidity_dependence_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research expansion-runner entrypoint."""

    return daily_return_liquidity_dependence_catalog()


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
    """Normalize daily source codes only when an explicit exchange is available."""

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
    keys = (
        keys.loc[dates == score_date]
        .drop_duplicates()
        .sort_values(["dt", "code"], kind="mergesort")
    )
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _strict_history(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return uniquely keyed source rows strictly before the score date."""

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
            frame.duplicated(["trade_date", "code"], keep=False),
            ["trade_date", "code"],
        ].head(3)
        raise ValueError(
            f"{KERNEL_NAME} daily_price has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _log_return_and_amount(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    previous = pd.to_numeric(frame["prev_close_price"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(frame["close_price"], errors="coerce").to_numpy(dtype="float64")
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype="float64")
    returns = np.full(len(frame), np.nan, dtype="float64")
    log_amount = np.full(len(frame), np.nan, dtype="float64")
    valid_price = (
        np.isfinite(previous)
        & np.isfinite(close)
        & (previous > _EPS)
        & (close > _EPS)
    )
    valid_amount = np.isfinite(amount) & (amount > _EPS)
    returns[valid_price] = np.log(close[valid_price] / previous[valid_price])
    log_amount[valid_amount] = np.log(amount[valid_amount])
    return returns, log_amount


def _return_liquidity_sign_mutual_information(
    returns: np.ndarray,
    log_amount: np.ndarray,
    session_positions: np.ndarray,
) -> float:
    """Compute a fixed-table mutual information without inventing adjacency."""

    if (
        len(returns) < _MIN_OBSERVATIONS
        or len(log_amount) != len(returns)
        or len(session_positions) != len(returns)
    ):
        return float("nan")
    return_current = returns[1:]
    amount_change = np.diff(log_amount)
    consecutive = np.diff(session_positions) == 1
    valid = np.isfinite(return_current) & np.isfinite(amount_change) & consecutive
    if int(valid.sum()) < _MIN_ADJACENT_PAIRS or not bool(valid[-1]):
        return float("nan")
    return_state = np.sign(return_current[valid]).astype("int64") + 1
    amount_state = np.sign(amount_change[valid]).astype("int64") + 1
    counts = np.full((3, 3), _PSEUDOCOUNT, dtype="float64")
    np.add.at(counts, (return_state, amount_state), 1.0)
    joint = counts / float(counts.sum())
    return_marginal = joint.sum(axis=1, keepdims=True)
    amount_marginal = joint.sum(axis=0, keepdims=True)
    value = float(
        np.sum(joint * np.log(joint / (return_marginal * amount_marginal)))
        / np.log(3.0)
    )
    return value if np.isfinite(value) else float("nan")


def _metrics(
    frame: pd.DataFrame,
    *,
    anchor: pd.Timestamp,
    date_positions: dict[pd.Timestamp, int],
) -> dict[str, float]:
    out = {signal: float("nan") for signal in _SIGNALS}
    if frame.empty or pd.Timestamp(frame["trade_date"].max()).normalize() != anchor:
        return out
    anchor_position = date_positions[anchor]
    sessions = frame["trade_date"].map(date_positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(sessions, anchor_position - _WINDOW + 1, side="left"))
    returns, log_amount = _log_return_and_amount(frame.iloc[first:])
    out["rlmi_return_liquidity_sign_mutual_information60"] = (
        _return_liquidity_sign_mutual_information(returns, log_amount, sessions[first:])
    )
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Build/cache the research factor once for one score-day context."""

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
            anchor = pd.Timestamp(history["trade_date"].max()).normalize()
            dates = sorted(pd.Timestamp(day).normalize() for day in history["trade_date"].unique())
            date_positions = {day: position for position, day in enumerate(dates)}
            groups = {
                str(code): group
                for code, group in history.groupby("code", sort=False)
                if pd.Timestamp(group["trade_date"].max()).normalize() == anchor
            }
            rows: list[dict[str, object]] = []
            for dt, raw_code in out_index:
                row: dict[str, object] = {"dt": dt, "code": raw_code}
                row.update({signal: float("nan") for signal in _SIGNALS})
                group = groups.get(_panel_code(raw_code))
                if group is not None:
                    row.update(_metrics(group, anchor=anchor, date_positions=date_positions))
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
class FactorMiningDailyReturnLiquidityDependenceV1(Factor):
    """Research-only strict-prior return/liquidity-dependence kernel."""

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
    "FactorMiningDailyReturnLiquidityDependenceV1",
    "daily_return_liquidity_dependence_catalog",
    "factor_mining_catalog",
]
