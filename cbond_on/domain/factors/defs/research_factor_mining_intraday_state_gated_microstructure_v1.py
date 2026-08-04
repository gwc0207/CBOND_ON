"""Research-only strict-T-1 structural-gated intraday microstructure catalogue.

The module uses an exact completed daily-price/base anchor to construct four
predeclared *bounded gates* from duration, credit quality, pure-debt/premium,
and stock-volatility states.  It then applies those gates to T-day quote-event,
spread-spell, passive-depth-refresh, and trade/quote-clock *topologies*.

It deliberately does not publish a static state, a price/return/path level, a
raw current-flow times raw-state product, or a cross-sectional residual.  The
current-day measures are also distinct from endpoint spread/imbalance changes:
they use event-clock allocation, own-path high-spread spell topology, passive
depth refresh while the quote is unchanged, and trade/quote clock decoupling.

Only FactorComputeContext inputs are consumed.  Daily rows are restricted to
``trade_date < score_date``; the latest valid daily-price session is the
independent T-1 anchor and every code must have a valid base row on that exact
session.  T-day panel rows must be physical continuous-session observations no
later than 14:29:00.  Missing sources/fields, duplicate strict-prior daily
rows, stale base state, malformed paths/books, counter resets, or insufficient
state cross-section fail closed to ``NaN``.  This file is import-only and
research-only: no I/O, label, PnL, registry import, config, DB, or live change.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


KERNEL_NAME = "factor_mining_intraday_state_gated_microstructure_v1"
CATALOG_VERSION = "20260803_intraday_state_gated_microstructure_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_STATE_CROSS_SECTION = 30
_MIN_ENDPOINTS = 12
_MIN_QUOTE_EVENTS = 4
_MIN_SIDE_EVENTS = 3
_MIN_TRADE_EVENTS = 6
_RELATIVE_QUOTE_TOL = 1e-10
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_CONTINUOUS_SECONDS = 210.0 * 60.0

_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})
_RATING_ORDINAL = {
    "AAA": 18.0,
    "AA+": 17.0,
    "AA": 16.0,
    "AA-": 15.0,
    "A+": 14.0,
    "A": 13.0,
    "A-": 12.0,
    "BBB+": 11.0,
    "BBB": 10.0,
    "BBB-": 9.0,
    "BB+": 8.0,
    "BB": 7.0,
    "BB-": 6.0,
    "B+": 5.0,
    "B": 4.0,
    "B-": 3.0,
    "CCC": 2.0,
    "CC": 1.0,
}

_BASE_FIELDS = (
    "duration",
    "rating",
    "debt_puredebt_ratio",
    "bond_prem_ratio",
    "stock_volatility",
)
_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "last",
    "amount",
    "num_trades",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
)


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable state-gated microstructure candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_DURATION_QUOTE_CLOCK_SIGNALS = (
    "isgm_duration_quote_revision_concentration",
    "isgm_duration_quote_revision_late_tilt",
    "isgm_duration_bid_ask_revision_clock_gap",
)
_CREDIT_SPREAD_SPELL_SIGNALS = (
    "isgm_credit_wide_spread_longest_spell_share",
    "isgm_credit_wide_spread_entry_clock",
    "isgm_credit_wide_spread_recovery_share",
)
_FLOOR_PREMIUM_DEPTH_SIGNALS = (
    "isgm_floorpremium_passive_depth_refresh_share",
    "isgm_floorpremium_passive_depth_side_balance",
    "isgm_floorpremium_passive_depth_late_tilt",
)
_STOCKVOL_TRADE_QUOTE_SIGNALS = (
    "isgm_stockvol_trade_quote_clock_center_gap",
    "isgm_stockvol_trade_quote_profile_cosine",
    "isgm_stockvol_silent_trade_share",
)

_CATALOG = (
    _entries(
        "duration_gated_quote_revision_calendar",
        _DURATION_QUOTE_CLOCK_SIGNALS,
        "A smooth T-1 duration gate modulates quote-revision calendar shape, not a return, spread level, or raw flow-state interaction.",
    )
    + _entries(
        "credit_gated_spread_spell_topology",
        _CREDIT_SPREAD_SPELL_SIGNALS,
        "A predeclared high-credit quantile gate modulates own-path wide-spread spell topology, distinct from spread level or early-late spread shift.",
    )
    + _entries(
        "floor_premium_gated_passive_depth_refresh",
        _FLOOR_PREMIUM_DEPTH_SIGNALS,
        "A bounded pure-debt/high-discount gate modulates quote-stable passive L1 depth refresh geometry rather than unconditional depth or a residual.",
    )
    + _entries(
        "stockvol_gated_trade_quote_clock_decoupling",
        _STOCKVOL_TRADE_QUOTE_SIGNALS,
        "A smooth T-1 stock-volatility gate modulates trade-versus-quote clock geometry, never cumulative flow level times raw state.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# All state gates are bounded and predeclared. r_x is a same-anchor cross-
# sectional percentile rank and S(r)=3r^2-2r^3. q_t is an L1-midpoint revision
# event, W_t is the own-path upper-quartile relative-spread state, D_t is L1
# total displayed depth, and n_t is an adjacent positive trade-count increment.
FORMULAS: dict[str, str] = {
    "isgm_duration_quote_revision_concentration": "S(r_duration)*sum_j(p_quote,j^2)",
    "isgm_duration_quote_revision_late_tilt": "S(r_duration)*(E[u|q=1]-0.5)",
    "isgm_duration_bid_ask_revision_clock_gap": "S(r_duration)*(E[u|bid_revision]-E[u|ask_revision])",
    "isgm_credit_wide_spread_longest_spell_share": "1[r_credit>=2/3]*max_run(W)/T",
    "isgm_credit_wide_spread_entry_clock": "1[r_credit>=2/3]*(E[u|wide-spread spell entry]-0.5)",
    "isgm_credit_wide_spread_recovery_share": "1[r_credit>=2/3]*mean(1[W_t and not W_t+1])",
    "isgm_floorpremium_passive_depth_refresh_share": "S(r_puredebt)*S(1-r_premium)*mean(1[large abs(delta log D) and q=0])",
    "isgm_floorpremium_passive_depth_side_balance": "S(r_puredebt)*S(1-r_premium)*(positive bid refresh-positive ask refresh)/(sum positive refresh)",
    "isgm_floorpremium_passive_depth_late_tilt": "S(r_puredebt)*S(1-r_premium)*(E[u|passive abs(delta log D)]-0.5)",
    "isgm_stockvol_trade_quote_clock_center_gap": "S(r_stockvol)*(E[u;n_t]-E[u;q_t])",
    "isgm_stockvol_trade_quote_profile_cosine": "S(r_stockvol)*cosine(normalized n-clock mass, normalized q-clock mass)",
    "isgm_stockvol_silent_trade_share": "S(r_stockvol)*sum(n_t*1[q_t=0])/sum(n_t)",
}


def intraday_state_gated_microstructure_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable four-family research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic scratch expansion runners."""

    return intraday_state_gated_microstructure_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirements() -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), 5),
        DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", *_BASE_FIELDS), 5),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    exchange_values = exchanges if exchanges is not None else pd.Series("", index=values.index)

    def _one(value: object, exchange: object) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip().upper()
        if not text or text in {"NAN", "NONE", "<NA>"}:
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
        [_one(value, exchange) for value, exchange in zip(values, exchange_values, strict=False)],
        index=values.index,
        dtype="string",
    )


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing columns: {missing}")


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
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), owner=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna() & (frame["trade_date"] < score_date) & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].drop_duplicates()
    return pd.MultiIndex.from_frame(keys.sort_values(["dt", "code"], kind="mergesort"), names=("dt", "code"))


def _score_date_from_index(index: pd.MultiIndex) -> pd.Timestamp:
    values = pd.to_datetime(index.get_level_values("dt"), errors="coerce").normalize().unique()
    parsed = [pd.Timestamp(value) for value in values if not pd.isna(value)]
    if len(parsed) != 1:
        raise ValueError(f"{KERNEL_NAME} requires one valid score date per context")
    return parsed[0]


def _rank01(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    valid = numeric.notna() & np.isfinite(numeric)
    out = pd.Series(np.nan, index=numeric.index, dtype="float64")
    count = int(valid.sum())
    if count < _MIN_STATE_CROSS_SECTION:
        return out
    selected = numeric.loc[valid]
    if float(selected.std(ddof=0)) <= _EPS:
        return out
    out.loc[valid] = (selected.rank(method="average") - 1.0) / (count - 1.0)
    return out


def _smoothstep(values: pd.Series) -> pd.Series:
    clipped = values.clip(lower=0.0, upper=1.0)
    return (3.0 * clipped.pow(2) - 2.0 * clipped.pow(3)).where(values.notna())


def _rating_ordinal(values: pd.Series) -> pd.Series:
    clean = values.astype("string").str.strip().str.upper()
    return clean.map(_RATING_ORDINAL).astype("float64")


def _prior_state_frame(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    price = _strict_history_source(
        ctx, source="market_cbond.daily_price", fields=("close_price",), score_date=score_date
    )
    base = _strict_history_source(
        ctx, source="market_cbond.daily_base", fields=_BASE_FIELDS, score_date=score_date
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    if price.empty:
        return pd.DataFrame(columns=("duration_gate", "credit_gate", "floor_premium_gate", "stockvol_gate"))
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    price_anchor = price.loc[price["trade_date"] == anchor, ["code"]].copy()
    base_anchor = base.loc[base["trade_date"] == anchor, ["code", *_BASE_FIELDS]].copy()
    snapshot = price_anchor.merge(base_anchor, on="code", how="inner", validate="one_to_one")
    if snapshot.empty:
        return pd.DataFrame(columns=("duration_gate", "credit_gate", "floor_premium_gate", "stockvol_gate"))
    duration_rank = _rank01(snapshot["duration"])
    credit_rank = _rank01(_rating_ordinal(snapshot["rating"]))
    pure_debt_rank = _rank01(snapshot["debt_puredebt_ratio"])
    premium_rank = _rank01(snapshot["bond_prem_ratio"])
    stockvol_rank = _rank01(snapshot["stock_volatility"])
    states = pd.DataFrame(
        {
            "code": snapshot["code"].astype(str),
            "duration_gate": _smoothstep(duration_rank),
            "credit_gate": (credit_rank >= (2.0 / 3.0)).astype("float64").where(credit_rank.notna()),
            "floor_premium_gate": _smoothstep(pure_debt_rank) * _smoothstep(1.0 - premium_rank),
            "stockvol_gate": _smoothstep(stockvol_rank),
        }
    )
    return states.set_index("code").replace([np.inf, -np.inf], np.nan)


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or any(column not in panel.columns for column in _REQUIRED_PANEL_COLUMNS):
        return pd.DataFrame(columns=["code", "seq", *_REQUIRED_PANEL_COLUMNS])
    checked = ensure_panel_index(panel)
    try:
        frame = checked.reset_index().loc[:, ["dt", "code", "seq", *_REQUIRED_PANEL_COLUMNS]].copy()
    except KeyError:
        return pd.DataFrame(columns=["code", "seq", *_REQUIRED_PANEL_COLUMNS])
    labels = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    keep = (
        labels.notna()
        & (labels.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    out["canonical_code"] = _canonical_market_code(out["code"])
    out = out.loc[out["canonical_code"] != ""].copy()
    for column in _REQUIRED_PANEL_COLUMNS:
        if column != "trade_time":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["canonical_code", "trade_time", "seq"], kind="mergesort")


def _endpoint_frame(group: pd.DataFrame) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    frame = group.copy()
    frame["event_time"] = pd.to_datetime(frame["trade_time"], errors="coerce").dt.floor(_EVENT_BIN)
    frame = frame.loc[frame["event_time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame()
    endpoint = (
        frame.sort_values(["event_time", "trade_time", "seq"], kind="mergesort")
        .groupby("event_time", sort=True)
        .last()
    )
    endpoint.index = pd.DatetimeIndex(endpoint.index, name="event_time")
    return endpoint.sort_index()


def _adjacent(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_series().diff().eq(_EVENT_BIN).to_numpy(dtype=bool)


def _clock_position(index: pd.DatetimeIndex) -> np.ndarray:
    values: list[float] = []
    for stamp in index:
        clock = stamp.time()
        seconds = clock.hour * 3600 + clock.minute * 60 + clock.second
        if _MORNING_START <= clock <= _MORNING_END:
            start = _MORNING_START.hour * 3600 + _MORNING_START.minute * 60
            values.append((seconds - start) / _CONTINUOUS_SECONDS)
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            start = _AFTERNOON_START.hour * 3600 + _AFTERNOON_START.minute * 60
            values.append((120.0 * 60.0 + seconds - start) / _CONTINUOUS_SECONDS)
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype="float64")


def _quote_state(endpoint: pd.DataFrame) -> pd.DataFrame:
    columns = ("ask_price1", "bid_price1")
    if endpoint.empty or any(column not in endpoint.columns for column in columns):
        return pd.DataFrame(columns=("mid", "spread", "quote_revision", "bid_revision", "ask_revision", "clock"))
    ask = pd.to_numeric(endpoint["ask_price1"], errors="coerce")
    bid = pd.to_numeric(endpoint["bid_price1"], errors="coerce")
    valid = np.isfinite(ask) & np.isfinite(bid) & (ask > bid) & (bid > _EPS)
    quote = pd.DataFrame({"ask": ask.loc[valid], "bid": bid.loc[valid]}).sort_index()
    if len(quote) < _MIN_ENDPOINTS:
        return pd.DataFrame(columns=("mid", "spread", "quote_revision", "bid_revision", "ask_revision", "clock"))
    quote["mid"] = (quote["ask"] + quote["bid"]) / 2.0
    quote["spread"] = (quote["ask"] - quote["bid"]) / quote["mid"]
    adjacent = _adjacent(pd.DatetimeIndex(quote.index))
    log_bid = np.log(quote["bid"])
    log_ask = np.log(quote["ask"])
    bid_revision = np.abs(log_bid.diff().to_numpy(dtype="float64")) > _RELATIVE_QUOTE_TOL
    ask_revision = np.abs(log_ask.diff().to_numpy(dtype="float64")) > _RELATIVE_QUOTE_TOL
    quote["bid_revision"] = adjacent & bid_revision
    quote["ask_revision"] = adjacent & ask_revision
    quote["quote_revision"] = quote["bid_revision"] | quote["ask_revision"]
    quote["clock"] = _clock_position(pd.DatetimeIndex(quote.index))
    return quote


def _clock_bins(clock: np.ndarray, weights: np.ndarray, *, bins: int = 6) -> np.ndarray | None:
    valid = np.isfinite(clock) & np.isfinite(weights) & (weights >= 0.0)
    if int(valid.sum()) == 0 or float(weights[valid].sum()) <= _EPS:
        return None
    locations = np.minimum((clock[valid] * bins).astype(int), bins - 1)
    out = np.zeros(bins, dtype="float64")
    for location, weight in zip(locations, weights[valid], strict=False):
        out[location] += weight
    total = float(out.sum())
    return out / total if total > _EPS else None


def _quote_calendar_metrics(quote: pd.DataFrame) -> dict[str, float]:
    names = _DURATION_QUOTE_CLOCK_SIGNALS
    out = {name: float("nan") for name in names}
    if quote.empty:
        return out
    revision = quote["quote_revision"].to_numpy(dtype=bool)
    clock = quote["clock"].to_numpy(dtype="float64")
    if int(revision.sum()) >= _MIN_QUOTE_EVENTS:
        profile = _clock_bins(clock, revision.astype("float64"))
        if profile is not None:
            out[names[0]] = float(np.square(profile).sum())
        out[names[1]] = float(np.mean(clock[revision]) - 0.5)
    bid = quote["bid_revision"].to_numpy(dtype=bool)
    ask = quote["ask_revision"].to_numpy(dtype=bool)
    if int(bid.sum()) >= _MIN_SIDE_EVENTS and int(ask.sum()) >= _MIN_SIDE_EVENTS:
        out[names[2]] = float(np.mean(clock[bid]) - np.mean(clock[ask]))
    return out


def _longest_true_run(values: np.ndarray, adjacent: np.ndarray) -> int:
    longest = 0
    current = 0
    for index, value in enumerate(values):
        if bool(value):
            current = current + 1 if index > 0 and bool(adjacent[index]) else 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _spread_spell_metrics(quote: pd.DataFrame) -> dict[str, float]:
    names = _CREDIT_SPREAD_SPELL_SIGNALS
    out = {name: float("nan") for name in names}
    if len(quote) < _MIN_ENDPOINTS:
        return out
    spread = quote["spread"].to_numpy(dtype="float64")
    threshold = float(np.quantile(spread, 0.75))
    if not np.isfinite(threshold) or float(np.std(spread)) <= _EPS:
        return out
    wide = spread >= threshold
    if not wide.any() or wide.all():
        return out
    adjacent = _adjacent(pd.DatetimeIndex(quote.index))
    out[names[0]] = float(_longest_true_run(wide, adjacent) / len(wide))
    preceding_wide = np.concatenate(([False], wide[:-1]))
    entry = wide & (~preceding_wide | ~adjacent)
    clock = quote["clock"].to_numpy(dtype="float64")
    if int(entry.sum()) >= 1:
        out[names[1]] = float(np.mean(clock[entry]) - 0.5)
    following_wide = np.concatenate((wide[1:], [False]))
    following_adjacent = np.concatenate((adjacent[1:], [False]))
    eligible = wide & following_adjacent
    if int(eligible.sum()) >= _MIN_QUOTE_EVENTS:
        out[names[2]] = float(np.mean(~following_wide[eligible]))
    return out


def _passive_depth_metrics(endpoint: pd.DataFrame, quote: pd.DataFrame) -> dict[str, float]:
    names = _FLOOR_PREMIUM_DEPTH_SIGNALS
    out = {name: float("nan") for name in names}
    columns = ("bid_volume1", "ask_volume1")
    if endpoint.empty or quote.empty or any(column not in endpoint.columns for column in columns):
        return out
    depth = endpoint.loc[quote.index, list(columns)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(depth.to_numpy(dtype="float64")).all(axis=1) & (depth >= 0.0).all(axis=1)
    depth = depth.loc[valid]
    quote = quote.loc[depth.index]
    if len(depth) < _MIN_ENDPOINTS:
        return out
    bid = depth["bid_volume1"].to_numpy(dtype="float64")
    ask = depth["ask_volume1"].to_numpy(dtype="float64")
    total = bid + ask
    valid_total = total > _EPS
    if int(valid_total.sum()) < _MIN_ENDPOINTS:
        return out
    bid, ask, total = bid[valid_total], ask[valid_total], total[valid_total]
    quote = quote.iloc[np.flatnonzero(valid_total)]
    adjacent = _adjacent(pd.DatetimeIndex(quote.index))
    log_total = np.log(total)
    total_delta = np.diff(log_total, prepend=np.nan)
    bid_delta = np.diff(np.log(bid + _EPS), prepend=np.nan)
    ask_delta = np.diff(np.log(ask + _EPS), prepend=np.nan)
    passive = adjacent & ~quote["quote_revision"].to_numpy(dtype=bool) & np.isfinite(total_delta)
    if int(passive.sum()) < _MIN_QUOTE_EVENTS:
        return out
    scale = float(np.quantile(np.abs(total_delta[passive]), 0.75))
    if not np.isfinite(scale) or scale <= _EPS:
        return out
    large_passive = passive & (np.abs(total_delta) >= scale)
    out[names[0]] = float(np.mean(large_passive[passive]))
    positive_bid = float(np.maximum(bid_delta[passive], 0.0).sum())
    positive_ask = float(np.maximum(ask_delta[passive], 0.0).sum())
    denominator = positive_bid + positive_ask
    if denominator > _EPS:
        out[names[1]] = float((positive_bid - positive_ask) / denominator)
    weights = np.where(passive & np.isfinite(total_delta), np.abs(total_delta), 0.0)
    if float(weights.sum()) > _EPS:
        out[names[2]] = float(np.average(quote["clock"].to_numpy(dtype="float64"), weights=weights) - 0.5)
    return out


def _trade_quote_metrics(endpoint: pd.DataFrame, quote: pd.DataFrame) -> dict[str, float]:
    names = _STOCKVOL_TRADE_QUOTE_SIGNALS
    out = {name: float("nan") for name in names}
    if endpoint.empty or quote.empty or "num_trades" not in endpoint.columns:
        return out
    counts = pd.to_numeric(endpoint.loc[quote.index, "num_trades"], errors="coerce")
    valid = np.isfinite(counts.to_numpy(dtype="float64"))
    counts = counts.loc[valid]
    quote = quote.loc[counts.index]
    if len(counts) < _MIN_ENDPOINTS:
        return out
    adjacent = _adjacent(pd.DatetimeIndex(counts.index))
    increments = counts.diff().to_numpy(dtype="float64")
    usable = adjacent & np.isfinite(increments)
    if int(usable.sum()) < _MIN_TRADE_EVENTS or (increments[usable] < -_EPS).any():
        return out
    increments = np.where(usable, np.maximum(increments, 0.0), 0.0)
    positive = increments > _EPS
    if int(positive.sum()) < _MIN_TRADE_EVENTS:
        return out
    clock = quote["clock"].to_numpy(dtype="float64")
    revisions = quote["quote_revision"].to_numpy(dtype=bool)
    if int(revisions.sum()) >= _MIN_QUOTE_EVENTS:
        out[names[0]] = float(np.average(clock, weights=increments) - np.mean(clock[revisions]))
        trade_profile = _clock_bins(clock, increments)
        quote_profile = _clock_bins(clock, revisions.astype("float64"))
        if trade_profile is not None and quote_profile is not None:
            denominator = float(np.linalg.norm(trade_profile) * np.linalg.norm(quote_profile))
            if denominator > _EPS:
                out[names[1]] = float(np.dot(trade_profile, quote_profile) / denominator)
    total = float(increments.sum())
    if total > _EPS:
        out[names[2]] = float(increments[~revisions].sum() / total)
    return out


def _raw_microstructure_metrics(group: pd.DataFrame) -> dict[str, float]:
    endpoint = _endpoint_frame(group)
    quote = _quote_state(endpoint)
    metrics: dict[str, float] = {}
    metrics.update(_quote_calendar_metrics(quote))
    metrics.update(_spread_spell_metrics(quote))
    metrics.update(_passive_depth_metrics(endpoint, quote))
    metrics.update(_trade_quote_metrics(endpoint, quote))
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in metrics.items()}


def _apply_gates(metrics: dict[str, float], states: pd.Series) -> dict[str, float]:
    gates = {
        "duration": pd.to_numeric(states.get("duration_gate"), errors="coerce"),
        "credit": pd.to_numeric(states.get("credit_gate"), errors="coerce"),
        "floorpremium": pd.to_numeric(states.get("floor_premium_gate"), errors="coerce"),
        "stockvol": pd.to_numeric(states.get("stockvol_gate"), errors="coerce"),
    }
    families = (
        ("duration", _DURATION_QUOTE_CLOCK_SIGNALS),
        ("credit", _CREDIT_SPREAD_SPELL_SIGNALS),
        ("floorpremium", _FLOOR_PREMIUM_DEPTH_SIGNALS),
        ("stockvol", _STOCKVOL_TRADE_QUOTE_SIGNALS),
    )
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    for gate_name, signals in families:
        gate = gates[gate_name]
        if not np.isfinite(gate) or gate < 0.0 or gate > 1.0:
            continue
        for signal in signals:
            value = metrics.get(signal, float("nan"))
            if np.isfinite(value):
                out[signal] = float(gate * value)
    return out


def _build_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    output_index = _output_index(ctx)
    out = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    if out.empty:
        return out
    score_date = _score_date_from_index(output_index)
    states = _prior_state_frame(ctx, score_date=score_date)
    physical = _strict_physical_frame(ctx.panel, score_date=score_date)
    groups = (
        {str(code): group for code, group in physical.groupby("canonical_code", sort=False)}
        if "canonical_code" in physical.columns
        else {}
    )
    for dt, raw_code in output_index:
        code = _canonical_market_code(pd.Series([raw_code])).iloc[0]
        group = groups.get(str(code))
        if not code or code not in states.index or group is None:
            continue
        metrics = _raw_microstructure_metrics(group)
        gated = _apply_gates(metrics, states.loc[code])
        for signal, value in gated.items():
            if np.isfinite(value):
                out.at[(dt, raw_code), signal] = float(value)
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    output_index = _output_index(ctx)
    score_date = _score_date_from_index(output_index) if not output_index.empty else pd.NaT
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _build_feature_frame(ctx)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayStateGatedMicrostructureV1(Factor):
    """Emit one strict-T-1 structural-gated T1430 microstructure signal."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict[str, object] | None = None) -> list[DailyFactorRequirement]:
        if params and params.get("signal"):
            _requested_entry(params)
        return _requirements()

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
    "FactorMiningIntradayStateGatedMicrostructureV1",
    "factor_mining_catalog",
    "intraday_state_gated_microstructure_catalog",
]
