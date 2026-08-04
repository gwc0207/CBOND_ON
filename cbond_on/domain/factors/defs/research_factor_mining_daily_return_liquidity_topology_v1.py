"""Research-only strict-T-1 daily return/liquidity topology catalogue.

This catalogue exposes three pre-registered, prior-only daily states:

* return-sign versus trade-count-change-sign mutual information;
* return-sign versus average-trade-size-change-sign mutual information; and
* transition entropy of the joint return-sign/amount-change-sign state.

All state changes require consecutive raw daily-source sessions.  Thus a
missing source session, suspension, or stale row is never compressed into an
invented adjacent observation.  Score-date and future rows are excluded
before any calculation.  This module remains research-only: no live/model
configuration, file I/O, database I/O, mask, label, score, PnL, or scheduler
dependency exists here.
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


KERNEL_NAME = "factor_mining_daily_return_liquidity_topology_v1"
CATALOG_VERSION = "20260803_daily_return_liquidity_topology_v1"
_LOOKBACK_DAYS = 75
_WINDOW = 60
_MIN_OBSERVATIONS = 45
_MIN_ADJACENT_PAIRS = 40
_PSEUDOCOUNT = 0.5
_EPS = 1e-12
_PRICE_FIELDS = ("prev_close_price", "close_price", "amount", "deal")
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


_INFORMATION_SIGNALS = (
    "rlmi_return_deal_sign_mutual_information60",
    "rlmi_return_trade_size_sign_mutual_information60",
)
_TOPOLOGY_SIGNALS = ("rjst_amount_joint_transition_entropy60",)
_SIGNALS = (*_INFORMATION_SIGNALS, *_TOPOLOGY_SIGNALS)
_CATALOG = (
    _entries(
        "prior_return_liquidity_information_dependence",
        _INFORMATION_SIGNALS,
        "Completed return-sign dependence on trading-count and average-trade-size"
        " changes is a nonlinear information-absorption state rather than a raw"
        " liquidity level or a lagged linear impact beta.",
    )
    + _entries(
        "prior_joint_return_liquidity_state_topology",
        _TOPOLOGY_SIGNALS,
        "The entropy of completed transitions across joint return-sign and"
        " amount-change-sign states measures state-path topology rather than a"
        " marginal return, amount, or volatility level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}

FORMULAS: dict[str, str] = {
    "rlmi_return_deal_sign_mutual_information60": (
        "Mutual information divided by log(3) between sign(log(close/prev_close)) "
        "and sign(delta log(deal)) on the latest up-to-60 strict-prior sessions, "
        "using a 3x3 Jeffreys-pseudocount table."
    ),
    "rlmi_return_trade_size_sign_mutual_information60": (
        "Mutual information divided by log(3) between sign(log(close/prev_close)) "
        "and sign(delta log(amount/deal)) on the latest up-to-60 strict-prior "
        "sessions, using a 3x3 Jeffreys-pseudocount table."
    ),
    "rjst_amount_joint_transition_entropy60": (
        "Shannon entropy divided by log(81) of transitions between the nine joint "
        "(sign(log(close/prev_close)), sign(delta log(amount))) states on the "
        "latest up-to-60 strict-prior sessions."
    ),
}


def daily_return_liquidity_topology_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research expansion-runner entrypoint."""

    return daily_return_liquidity_topology_catalog()


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
    """Normalize a daily code only with its explicit daily exchange."""

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
    """Load only unique normalized daily rows strictly before the score day."""

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


def _history_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous = pd.to_numeric(frame["prev_close_price"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(frame["close_price"], errors="coerce").to_numpy(dtype="float64")
    amount = pd.to_numeric(frame["amount"], errors="coerce").to_numpy(dtype="float64")
    deal = pd.to_numeric(frame["deal"], errors="coerce").to_numpy(dtype="float64")
    returns = np.full(len(frame), np.nan, dtype="float64")
    log_amount = np.full(len(frame), np.nan, dtype="float64")
    log_deal = np.full(len(frame), np.nan, dtype="float64")
    log_trade_size = np.full(len(frame), np.nan, dtype="float64")
    valid_price = (
        np.isfinite(previous)
        & np.isfinite(close)
        & (previous > _EPS)
        & (close > _EPS)
    )
    valid_amount = np.isfinite(amount) & (amount > _EPS)
    valid_deal = np.isfinite(deal) & (deal > _EPS)
    valid_trade_size = valid_amount & valid_deal
    returns[valid_price] = np.log(close[valid_price] / previous[valid_price])
    log_amount[valid_amount] = np.log(amount[valid_amount])
    log_deal[valid_deal] = np.log(deal[valid_deal])
    log_trade_size[valid_trade_size] = np.log(amount[valid_trade_size] / deal[valid_trade_size])
    return returns, log_amount, log_deal, log_trade_size


def _state_arrays(
    returns: np.ndarray,
    log_liquidity: np.ndarray,
    session_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_return = returns[1:]
    liquidity_change = np.diff(log_liquidity)
    current_positions = session_positions[1:]
    valid = (
        np.isfinite(current_return)
        & np.isfinite(liquidity_change)
        & (np.diff(session_positions) == 1)
    )
    return current_return, liquidity_change, current_positions, valid


def _sign_mutual_information(
    returns: np.ndarray,
    log_liquidity: np.ndarray,
    session_positions: np.ndarray,
) -> float:
    if len(returns) < _MIN_OBSERVATIONS:
        return float("nan")
    current_return, liquidity_change, _, valid = _state_arrays(
        returns,
        log_liquidity,
        session_positions,
    )
    if int(valid.sum()) < _MIN_ADJACENT_PAIRS or not bool(valid[-1]):
        return float("nan")
    return_state = np.sign(current_return[valid]).astype("int64") + 1
    liquidity_state = np.sign(liquidity_change[valid]).astype("int64") + 1
    counts = np.full((3, 3), _PSEUDOCOUNT, dtype="float64")
    np.add.at(counts, (return_state, liquidity_state), 1.0)
    joint = counts / float(counts.sum())
    return_marginal = joint.sum(axis=1, keepdims=True)
    liquidity_marginal = joint.sum(axis=0, keepdims=True)
    value = float(
        np.sum(joint * np.log(joint / (return_marginal * liquidity_marginal)))
        / np.log(3.0)
    )
    return value if np.isfinite(value) else float("nan")


def _joint_transition_entropy(
    returns: np.ndarray,
    log_amount: np.ndarray,
    session_positions: np.ndarray,
) -> float:
    if len(returns) < _MIN_OBSERVATIONS:
        return float("nan")
    current_return, amount_change, current_positions, valid = _state_arrays(
        returns,
        log_amount,
        session_positions,
    )
    adjacent = valid[1:] & valid[:-1] & (np.diff(current_positions) == 1)
    if int(adjacent.sum()) < _MIN_ADJACENT_PAIRS or not bool(adjacent[-1]):
        return float("nan")
    joint_state = np.full(len(current_return), -1, dtype="int64")
    joint_state[valid] = (
        (np.sign(current_return[valid]).astype("int64") + 1) * 3
        + np.sign(amount_change[valid]).astype("int64")
        + 1
    )
    counts = np.zeros((9, 9), dtype="float64")
    np.add.at(
        counts,
        (joint_state[:-1][adjacent], joint_state[1:][adjacent]),
        1.0,
    )
    values = counts[counts > 0.0]
    if not len(values):
        return float("nan")
    probabilities = values / float(values.sum())
    value = float(-np.sum(probabilities * np.log(probabilities)) / np.log(81.0))
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
    session_positions = frame["trade_date"].map(date_positions).to_numpy(dtype="int64")
    first = int(np.searchsorted(session_positions, anchor_position - _WINDOW + 1, side="left"))
    returns, log_amount, log_deal, log_trade_size = _history_arrays(frame.iloc[first:])
    sessions = session_positions[first:]
    out["rlmi_return_deal_sign_mutual_information60"] = _sign_mutual_information(
        returns,
        log_deal,
        sessions,
    )
    out["rlmi_return_trade_size_sign_mutual_information60"] = _sign_mutual_information(
        returns,
        log_trade_size,
        sessions,
    )
    out["rjst_amount_joint_transition_entropy60"] = _joint_transition_entropy(
        returns,
        log_amount,
        sessions,
    )
    return out


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
class FactorMiningDailyReturnLiquidityTopologyV1(Factor):
    """Research-only daily return/liquidity state-topology kernel."""

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
    "FactorMiningDailyReturnLiquidityTopologyV1",
    "daily_return_liquidity_topology_catalog",
    "factor_mining_catalog",
]
