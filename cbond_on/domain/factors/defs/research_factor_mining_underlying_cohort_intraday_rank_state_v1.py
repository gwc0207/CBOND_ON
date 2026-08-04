"""Research-only strict-PIT mapped-stock intraday rank-state catalogue.

Each convertible bond inherits the *cross-sectional state* of its mapped
underlying stock, computed over the complete valid stock T1430 panel on the
score day.  The output is therefore a percentile rank of an intraday
price-path, activity, or displayed-depth *shape*; it is never a raw stock
return, a bond-minus-stock gap, or a bond/stock synchrony statistic.

This fills a different information dimension from the T-1 underlying-cohort
distribution catalogue and from the mapped-pair transmission catalogues.  The
former describes historical mapped-peer distributions, while the latter
compare two assets' paths.  Here, no bond path value enters the calculation:
``ctx.panel`` supplies only the candidate bond index and score day, and every
state value is inherited from ``ctx.stock_panel`` through the supplied mapping.

Both the stock panel and its timestamps are treated as physical point-in-time
evidence.  A row must be on score day, in a continuous auction session, and no
later than 14:29:00.  Future-dated or duplicate mappings, invalid paths,
counter resets, incomplete books, insufficient cross-sectional support, and
missing inputs fail closed to ``NaN``.  The module has no direct I/O and is
intentionally import-only: it is not added to ``defs.__init__``, factor
contracts, configurations, model inputs, databases, or live scheduling.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index


KERNEL_NAME = "factor_mining_underlying_cohort_intraday_rank_state_v1"
CATALOG_VERSION = "20260803_underlying_cohort_intraday_rank_state_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_PRICE_RETURNS = 12
_MIN_PHASE_RETURNS = 4
_MIN_LIQUIDITY_BINS = 12
_MIN_PHASE_ACTIVITY_BINS = 4
_MIN_POSITIVE_ACTIVITY_BINS = 4
_MIN_DEPTH_BINS = 12
_MIN_DEPTH_PERSISTENCE_BINS = 8
_MIN_CROSS_SECTION = 30

_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)

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
    """One auditable research-only inherited stock-state candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_PRICE_SHAPE_SIGNALS = (
    "ucris_stock_price_path_efficiency_rank",
    "ucris_stock_price_tail_variation_rank",
    "ucris_stock_price_phase_reversal_rank",
)
_LIQUIDITY_SHAPE_SIGNALS = (
    "ucris_stock_liquidity_late_acceleration_rank",
    "ucris_stock_liquidity_trade_size_shift_rank",
    "ucris_stock_liquidity_event_concentration_rank",
)
_DEPTH_SHAPE_SIGNALS = (
    "ucris_stock_depth_touch_share_rank",
    "ucris_stock_depth_imbalance_persistence_rank",
    "ucris_stock_depth_late_rebuild_rank",
)

_CATALOG = (
    _entries(
        "underlying_cross_sectional_price_path_state",
        _PRICE_SHAPE_SIGNALS,
        "A bond inherits where its mapped stock lies in the contemporaneous equity cross-section of trend efficiency, tail variation allocation, and early-to-late reversal shape, rather than inheriting a raw stock return.",
    )
    + _entries(
        "underlying_cross_sectional_liquidity_shape_state",
        _LIQUIDITY_SHAPE_SIGNALS,
        "A bond inherits the mapped stock's cross-sectional state of late trade acceleration, trade-size migration, and event concentration; no bond activity or bond-stock activity gap is used.",
    )
    + _entries(
        "underlying_cross_sectional_depth_shape_state",
        _DEPTH_SHAPE_SIGNALS,
        "A bond inherits the mapped stock's cross-sectional displayed-depth curve, imbalance persistence, and intraday rebuild state, not a cross-asset book geometry comparison.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# r_t is a log return over an exactly adjacent five-minute stock endpoint, V is
# sum(abs(r_t)), and rank_cs is the average-tie percentile rank among complete
# physical T-day stock paths.  N/A are cumulative trade-count/notional
# increments and D/I/T are L1--L5 total depth, L1 imbalance, and L1 touch-depth
# share.  The factor publishes rank_cs(...) only; it never publishes r_t or a
# raw mapped-stock state directly.
FORMULAS: dict[str, str] = {
    "ucris_stock_price_path_efficiency_rank": "rank_cs(abs(sum(r_t))/sum(abs(r_t)))",
    "ucris_stock_price_tail_variation_rank": "rank_cs(sum_{t in tail}(abs(r_t))/sum_t(abs(r_t)))",
    "ucris_stock_price_phase_reversal_rank": "rank_cs(-(sum_early(r_t)*sum_late(r_t))/sum_t(abs(r_t))^2)",
    "ucris_stock_liquidity_late_acceleration_rank": "rank_cs(log((sum_late(deltaN)/n_late)/(sum_early(deltaN)/n_early)))",
    "ucris_stock_liquidity_trade_size_shift_rank": "rank_cs(log((sum_late(deltaA)/sum_late(deltaN))/(sum_early(deltaA)/sum_early(deltaN))))",
    "ucris_stock_liquidity_event_concentration_rank": "rank_cs(sum_t((deltaN_t/sum(deltaN))^2))",
    "ucris_stock_depth_touch_share_rank": "rank_cs(mean_t((bidVol1_t+askVol1_t)/D_t))",
    "ucris_stock_depth_imbalance_persistence_rank": "rank_cs(mean(1[sign(I_t)=sign(I_t-1)]) over adjacent nonzero I)",
    "ucris_stock_depth_late_rebuild_rank": "rank_cs(log(mean_late(D_t)/mean_early(D_t)))",
}


def factor_mining_underlying_cohort_intraday_rank_state_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable three-family research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic scratch expansion runners."""

    return factor_mining_underlying_cohort_intraday_rank_state_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _depth_volume_columns() -> tuple[str, ...]:
    return tuple(
        column
        for level in range(1, 6)
        for column in (f"ask_volume{level}", f"bid_volume{level}")
    )


_REQUIRED_STOCK_COLUMNS = (
    "trade_time",
    "last",
    "amount",
    "num_trades",
    *_depth_volume_columns(),
)


def _canonical_market_code(value: object) -> str:
    """Accept only explicit exchange-qualified codes from context data."""

    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if "." not in text:
        return ""
    bare, suffix = text.rsplit(".", 1)
    suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
    return f"{bare}.{suffix}" if bare and suffix in _MARKET_EXCHANGES else ""


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        day = pd.Timestamp(raw_day)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
    if panel.empty:
        return None
    labels = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(labels[labels.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    labels = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[labels == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _empty_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["canonical_stock_code", "seq", *_REQUIRED_STOCK_COLUMNS])


def _strict_stock_frame(stock_panel: object, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only physical score-day stock observations visible by 14:29."""

    if not isinstance(stock_panel, pd.DataFrame) or any(
        column not in stock_panel.columns for column in _REQUIRED_STOCK_COLUMNS
    ):
        return _empty_stock_frame()
    checked = ensure_panel_index(stock_panel)
    try:
        frame = checked.reset_index().loc[:, ["dt", "code", "seq", *_REQUIRED_STOCK_COLUMNS]].copy()
    except KeyError:
        return _empty_stock_frame()
    labels = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
    keep = (
        labels.notna()
        & (labels.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    if out.empty:
        return _empty_stock_frame()
    out["trade_time"] = timestamps.loc[keep]
    out["canonical_stock_code"] = out["code"].map(_canonical_market_code)
    out = out.loc[out["canonical_stock_code"] != ""].copy()
    for column in _REQUIRED_STOCK_COLUMNS:
        if column != "trade_time":
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.sort_values(["canonical_stock_code", "trade_time", "seq"], kind="mergesort")


def _context_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Use only unambiguous, non-forward rows from ``ctx.bond_stock_map``."""

    mapping = ctx.bond_stock_map
    if (
        not isinstance(mapping, pd.DataFrame)
        or mapping.empty
        or "code" not in mapping.columns
        or "stock_code" not in mapping.columns
    ):
        return {}
    columns = ["code", "stock_code"]
    has_trade_date = "trade_date" in mapping.columns
    if has_trade_date:
        columns.append("trade_date")
    frame = mapping.loc[:, columns].copy()
    if has_trade_date:
        mapping_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.loc[mapping_dates.notna() & (mapping_dates <= score_date)].copy()
    if frame.empty:
        return {}
    frame["bond_code"] = frame["code"].map(_canonical_market_code)
    frame["mapped_stock_code"] = frame["stock_code"].map(_canonical_market_code)
    frame = frame.loc[(frame["bond_code"] != "") & (frame["mapped_stock_code"] != "")].copy()
    if frame.empty:
        return {}
    # A duplicate bond mapping is an unresolved PIT ambiguity.  It is never
    # resolved by taking the latest context row or by selecting a panel match.
    frame = frame.loc[~frame["bond_code"].duplicated(keep=False)]
    return dict(zip(frame["bond_code"], frame["mapped_stock_code"], strict=False))


def _endpoint_frame(group: pd.DataFrame) -> pd.DataFrame:
    """Use the final valid snapshot in each observed five-minute endpoint bin."""

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


def _adjacent_mask(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_series().diff().eq(_EVENT_BIN).to_numpy(dtype=bool)


def _price_shape_metrics(endpoint: pd.DataFrame) -> dict[str, float]:
    """Return unranked stock price-path shapes, never raw stock returns."""

    out = {signal: float("nan") for signal in _PRICE_SHAPE_SIGNALS}
    if endpoint.empty or "last" not in endpoint.columns:
        return out
    price = pd.to_numeric(endpoint["last"], errors="coerce")
    price = price.loc[np.isfinite(price) & (price > _EPS)]
    if price.empty:
        return out
    returns = np.log(price).diff()
    returns.loc[~_adjacent_mask(price.index)] = np.nan
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < _MIN_PRICE_RETURNS:
        return out
    variation = float(np.abs(returns.to_numpy(dtype="float64")).sum())
    if not np.isfinite(variation) or variation <= _EPS:
        return out
    out["ucris_stock_price_path_efficiency_rank"] = float(abs(returns.sum()) / variation)

    tail = returns.loc[returns.index.time >= _LATE_START]
    if len(tail) >= _MIN_PHASE_RETURNS:
        out["ucris_stock_price_tail_variation_rank"] = float(
            np.abs(tail.to_numpy(dtype="float64")).sum() / variation
        )

    early = returns.loc[returns.index.time <= _EARLY_END]
    if len(early) >= _MIN_PHASE_RETURNS and len(tail) >= _MIN_PHASE_RETURNS:
        out["ucris_stock_price_phase_reversal_rank"] = float(
            -(early.sum() * tail.sum()) / (variation * variation)
        )
    return out


def _counter_increments(endpoint: pd.DataFrame) -> pd.DataFrame:
    """Return adjacent valid cumulative-count/notional increments or no path."""

    required = ("num_trades", "amount")
    if endpoint.empty or any(column not in endpoint.columns for column in required):
        return pd.DataFrame(columns=["trade_increment", "amount_increment"])
    counters = endpoint.loc[:, list(required)].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(
        np.isfinite(counters.to_numpy(dtype="float64")).all(axis=1),
        index=counters.index,
        dtype=bool,
    )
    if int(valid.sum()) < _MIN_LIQUIDITY_BINS + 1:
        return pd.DataFrame(columns=["trade_increment", "amount_increment"])
    increments = counters.diff().rename(
        columns={"num_trades": "trade_increment", "amount": "amount_increment"}
    )
    adjacent = _adjacent_mask(counters.index)
    valid_current = valid.to_numpy(dtype=bool)
    valid_previous = np.concatenate(([False], valid_current[:-1]))
    usable = adjacent & valid_current & valid_previous
    increments = increments.loc[usable].copy()
    if increments.empty:
        return increments
    numeric = increments.to_numpy(dtype="float64")
    if not np.isfinite(numeric).all() or (numeric < -_EPS).any():
        return pd.DataFrame(columns=["trade_increment", "amount_increment"])
    return increments.clip(lower=0.0)


def _liquidity_shape_metrics(endpoint: pd.DataFrame) -> dict[str, float]:
    """Return unranked intraday activity-shape states from stock counters."""

    out = {signal: float("nan") for signal in _LIQUIDITY_SHAPE_SIGNALS}
    increments = _counter_increments(endpoint)
    if len(increments) < _MIN_LIQUIDITY_BINS:
        return out
    early = increments.loc[increments.index.time <= _EARLY_END]
    late = increments.loc[increments.index.time >= _LATE_START]
    if len(early) >= _MIN_PHASE_ACTIVITY_BINS and len(late) >= _MIN_PHASE_ACTIVITY_BINS:
        early_trades = float(early["trade_increment"].sum())
        late_trades = float(late["trade_increment"].sum())
        if early_trades > _EPS and late_trades > _EPS:
            early_rate = early_trades / len(early)
            late_rate = late_trades / len(late)
            if early_rate > _EPS and late_rate > _EPS:
                out["ucris_stock_liquidity_late_acceleration_rank"] = float(np.log(late_rate / early_rate))

            early_amount = float(early["amount_increment"].sum())
            late_amount = float(late["amount_increment"].sum())
            if early_amount > _EPS and late_amount > _EPS:
                early_size = early_amount / early_trades
                late_size = late_amount / late_trades
                if early_size > _EPS and late_size > _EPS:
                    out["ucris_stock_liquidity_trade_size_shift_rank"] = float(
                        np.log(late_size / early_size)
                    )

    positive = increments.loc[increments["trade_increment"] > _EPS, "trade_increment"]
    if len(positive) >= _MIN_POSITIVE_ACTIVITY_BINS:
        total = float(positive.sum())
        if total > _EPS:
            shares = positive.to_numpy(dtype="float64") / total
            out["ucris_stock_liquidity_event_concentration_rank"] = float(np.square(shares).sum())
    return out


def _depth_shape_metrics(endpoint: pd.DataFrame) -> dict[str, float]:
    """Return unranked L1--L5 displayed-depth shape states for one stock."""

    out = {signal: float("nan") for signal in _DEPTH_SHAPE_SIGNALS}
    columns = _depth_volume_columns()
    if endpoint.empty or any(column not in endpoint.columns for column in columns):
        return out
    values = endpoint.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(
        np.isfinite(values.to_numpy(dtype="float64")).all(axis=1),
        index=values.index,
        dtype=bool,
    ) & (values >= 0.0).all(axis=1)
    book = values.loc[valid].copy()
    if len(book) < _MIN_DEPTH_BINS:
        return out
    bid_touch = book["bid_volume1"]
    ask_touch = book["ask_volume1"]
    total = book.sum(axis=1)
    touch_total = bid_touch + ask_touch
    valid_total = total > _EPS
    if int(valid_total.sum()) < _MIN_DEPTH_BINS:
        return out
    book = book.loc[valid_total]
    total = total.loc[valid_total]
    touch_total = touch_total.loc[valid_total]
    out["ucris_stock_depth_touch_share_rank"] = float((touch_total / total).mean())

    imbalance_valid = touch_total > _EPS
    imbalance = ((bid_touch - ask_touch) / touch_total).loc[imbalance_valid]
    if len(imbalance) >= _MIN_DEPTH_PERSISTENCE_BINS:
        signs = np.sign(imbalance.to_numpy(dtype="float64"))
        adjacent = _adjacent_mask(pd.DatetimeIndex(imbalance.index))
        nonzero = np.abs(signs) > 0.0
        prior_nonzero = np.concatenate(([False], nonzero[:-1]))
        usable = adjacent & nonzero & prior_nonzero
        if int(usable.sum()) >= _MIN_DEPTH_PERSISTENCE_BINS - 1:
            out["ucris_stock_depth_imbalance_persistence_rank"] = float(
                np.mean(signs[usable] == np.roll(signs, 1)[usable])
            )

    early = total.loc[total.index.time <= _EARLY_END]
    late = total.loc[total.index.time >= _LATE_START]
    if len(early) >= _MIN_PHASE_ACTIVITY_BINS and len(late) >= _MIN_PHASE_ACTIVITY_BINS:
        early_depth = float(early.mean())
        late_depth = float(late.mean())
        if early_depth > _EPS and late_depth > _EPS:
            out["ucris_stock_depth_late_rebuild_rank"] = float(np.log(late_depth / early_depth))
    return out


def _stock_metrics(group: pd.DataFrame) -> dict[str, float]:
    endpoint = _endpoint_frame(group)
    metrics = {signal: float("nan") for signal in _ALL_SIGNALS}
    metrics.update(_price_shape_metrics(endpoint))
    metrics.update(_liquidity_shape_metrics(endpoint))
    metrics.update(_depth_shape_metrics(endpoint))
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in metrics.items()}


def _cross_sectional_ranks(raw: pd.DataFrame) -> pd.DataFrame:
    """Rank every complete stock state independently; sparse states fail closed."""

    ranked = pd.DataFrame(index=raw.index, columns=_ALL_SIGNALS, dtype="float64")
    for signal in _ALL_SIGNALS:
        values = pd.to_numeric(raw[signal], errors="coerce")
        valid = values.notna() & np.isfinite(values)
        count = int(valid.sum())
        if count < _MIN_CROSS_SECTION:
            continue
        selected = values.loc[valid]
        if float(selected.std(ddof=0)) <= _EPS:
            continue
        ordinal = selected.rank(method="average")
        ranked.loc[valid, signal] = (ordinal - 1.0) / (count - 1.0)
    return ranked.replace([np.inf, -np.inf], np.nan)


def _stock_rank_states(stock_frame: pd.DataFrame) -> pd.DataFrame:
    if stock_frame.empty:
        return pd.DataFrame(columns=_ALL_SIGNALS, dtype="float64").rename_axis("canonical_stock_code")
    rows: list[dict[str, object]] = []
    for code, group in stock_frame.groupby("canonical_stock_code", sort=True):
        if not code:
            continue
        rows.append({"canonical_stock_code": str(code), **_stock_metrics(group)})
    if not rows:
        return pd.DataFrame(columns=_ALL_SIGNALS, dtype="float64").rename_axis("canonical_stock_code")
    raw = pd.DataFrame(rows).set_index("canonical_stock_code").sort_index()
    return _cross_sectional_ranks(raw)


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        stock_frame = _strict_stock_frame(ctx.stock_panel, score_date=score_date)
        stock_states = _stock_rank_states(stock_frame)
        mapping = _context_mapping(ctx, score_date=score_date)
        rows: list[dict[str, object]] = []
        for dt, raw_bond_code in output_index:
            row: dict[str, object] = {"dt": dt, "code": raw_bond_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            mapped_stock = mapping.get(_canonical_market_code(raw_bond_code), "")
            if mapped_stock and mapped_stock in stock_states.index:
                row.update(stock_states.loc[mapped_stock, list(_ALL_SIGNALS)].to_dict())
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].reindex(output_index)
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningUnderlyingCohortIntradayRankStateV1(Factor):
    """Emit one strict-PIT inherited mapped-stock cross-sectional rank state."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = True

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out


__all__ = [
    "CATALOG_VERSION",
    "FORMULAS",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningUnderlyingCohortIntradayRankStateV1",
    "factor_mining_catalog",
    "factor_mining_underlying_cohort_intraday_rank_state_catalog",
]
