"""Research-only strict-T-1 underlying-state cohort distribution factors.

This catalogue is intentionally complementary to structural-neighborhood v1:
it never emits a peer return mean.  Instead it measures the dispersion and
shape of the distribution among bonds whose *mapped underlying stocks* share
the same observable T-1 state, together with the distribution of bond-stock
tracking and liquidity mismatches in that cohort.

``daily_price`` is the independent market-calendar anchor.  The module first
selects its latest row strictly before the score date, then permits
``daily_base`` data only through an exact same-day/same-code join.  Missing,
stale, duplicate, or mismatched mappings are fail-closed and produce ``NaN``
rather than a carried-forward state.  No labels, pools, PnL, scores, files,
databases, Redis, backtests, or live artifacts are accessed.
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


KERNEL_NAME = "factor_mining_underlying_cohort_distribution_v1"
CATALOG_VERSION = "20260803_underlying_cohort_distribution_v1"
_LOOKBACK_DAYS = 66
_NEIGHBOR_COUNT = 7
_MIN_NEIGHBORS = 3
_EPS = 1e-12

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
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_UNDERLYING_DISTRIBUTION_SIGNALS = (
    "ucd_peer_stock_return_dispersion1",
    "ucd_peer_stock_return_skew1",
    "ucd_peer_stock_liquidity_shock_dispersion20",
)
_TRACKING_DISTRIBUTION_SIGNALS = (
    "utd_peer_tracking_error_dispersion1",
    "utd_peer_tracking_error_skew1",
    "utd_peer_flow_tracking_dispersion20",
)

_CATALOG = (
    _entries(
        "underlying_state_distribution",
        _UNDERLYING_DISTRIBUTION_SIGNALS,
        "The T-1 mapped-stock state neighborhood is represented by peer return and liquidity-shock dispersion/shape rather than a directional peer average.",
    )
    + _entries(
        "bond_stock_tracking_distribution",
        _TRACKING_DISTRIBUTION_SIGNALS,
        "The T-1 distribution of bond-stock tracking errors and relative liquidity shocks captures cohort disagreement without fitting a residual or publishing a peer mean.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "underlying_state_distribution": _UNDERLYING_DISTRIBUTION_SIGNALS,
    "bond_stock_tracking_distribution": _TRACKING_DISTRIBUTION_SIGNALS,
}

_PRICE_FIELDS = ("prev_close_price", "close_price", "amount")
_BASE_FIELDS = ("stock_code", "stock_close_price", "stock_volatility", "stk_amount")


def underlying_cohort_distribution_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return underlying_cohort_distribution_catalog()


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
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", *_PRICE_FIELDS), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", *_BASE_FIELDS), _LOOKBACK_DAYS),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    exchange_values = exchanges if exchanges is not None else pd.Series("", index=values.index)

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
        [_one(value, exchange) for value, exchange in zip(values, exchange_values)],
        index=values.index,
        dtype="string",
    )


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        day = pd.Timestamp(raw)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:output_index"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached.index

    panel = ensure_panel_index(ctx.panel)
    score_date = _score_date_from_panel(panel)
    if score_date is None:
        index = _empty_index()
    else:
        dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
        selected = panel.loc[dates == score_date]
        if not panel.empty and selected.empty:
            raise ValueError(f"{KERNEL_NAME} has no indexed rows for score date {score_date.date().isoformat()}")
        keys = selected.index.droplevel("seq").unique()
        index = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"]).sort_values()

    frame = pd.DataFrame(index=index)
    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior.index
        ctx.cache[cache_key] = frame
    return index


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


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
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), source=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _complete_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count:
        return None
    tail = frame.tail(count)
    positions = pd.to_numeric(tail["__price_session_index"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return None
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    if not bool(np.array_equal(positions, expected)):
        return None
    values = np.column_stack(
        [pd.to_numeric(tail[column], errors="coerce").to_numpy(dtype="float64") for column in columns]
    )
    return values if np.isfinite(values).all() else None


def _safe_return(current: float, previous: float) -> float:
    if not (np.isfinite(current) and np.isfinite(previous) and current > _EPS and previous > _EPS):
        return float("nan")
    return float(current / previous - 1.0)


def _z_last_log(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all() or (values <= _EPS).any():
        return float("nan")
    logged = np.log(values)
    prior = logged[:-1]
    scale = float(np.std(prior, ddof=1))
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return float((logged[-1] - float(np.mean(prior))) / scale)


def _bond_state(price: pd.DataFrame) -> dict[str, float]:
    last = _complete_tail(price, ("prev_close_price", "close_price"), 1)
    ret1 = _safe_return(float(last[-1, 1]), float(last[-1, 0])) if last is not None else float("nan")
    amount_last = _complete_tail(price, ("amount",), 1)
    amount = float(amount_last[-1, 0]) if amount_last is not None else float("nan")
    amount_tail = _complete_tail(price, ("amount",), 21)
    liquidity_z = _z_last_log(amount_tail[:, 0]) if amount_tail is not None else float("nan")
    return {
        "__bond_return1": ret1,
        "__bond_amount": amount,
        "__bond_liquidity_z20": liquidity_z,
    }


def _stock_state(base: pd.DataFrame) -> dict[str, float] | None:
    mapping_raw = base["stock_code"].iloc[-1]
    mapping = "" if pd.isna(mapping_raw) else str(mapping_raw).strip().upper()
    if not mapping or mapping == "NAN":
        return None
    vol_last = _complete_tail(base, ("stock_volatility",), 1)
    close_tail = _complete_tail(base, ("stock_close_price",), 2)
    amount_tail = _complete_tail(base, ("stk_amount",), 21)
    amount_last = _complete_tail(base, ("stk_amount",), 1)
    if vol_last is None or close_tail is None or amount_tail is None or amount_last is None:
        return None
    stock_volatility = float(vol_last[-1, 0])
    stock_return1 = _safe_return(float(close_tail[-1, 0]), float(close_tail[-2, 0]))
    stock_liquidity_z20 = _z_last_log(amount_tail[:, 0])
    stock_amount = float(amount_last[-1, 0])
    if not (
        np.isfinite(stock_volatility)
        and stock_volatility > _EPS
        and np.isfinite(stock_return1)
        and np.isfinite(stock_liquidity_z20)
        and np.isfinite(stock_amount)
        and stock_amount > _EPS
    ):
        return None
    return {
        "__stock_return1": stock_return1,
        "__stock_volatility": stock_volatility,
        "__stock_liquidity_z20": stock_liquidity_z20,
        "__stock_amount": stock_amount,
    }


@dataclass
class _SourceState:
    out_index: pd.MultiIndex
    anchor: pd.Timestamp | None
    anchor_codes: set[str]
    price_groups: dict[str, pd.DataFrame]
    base_groups: dict[str, pd.DataFrame]


def _source_state(ctx: FactorComputeContext) -> _SourceState:
    out_index = _output_index(ctx)
    score_date = _score_date_from_panel(ctx.panel)
    if out_index.empty or score_date is None:
        return _SourceState(out_index, None, set(), {}, {})
    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=_PRICE_FIELDS,
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_BASE_FIELDS,
        score_date=score_date,
    )
    if price.empty or base.empty:
        return _SourceState(out_index, None, set(), {}, {})

    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    calendar = sorted(pd.Timestamp(day).normalize() for day in price["trade_date"].unique())
    session_index = {day: position for position, day in enumerate(calendar)}
    price = price.copy()
    price["__price_session_index"] = price["trade_date"].map(session_index)
    base = base.merge(
        price.loc[:, ["trade_date", "code"]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort")
    base["__price_session_index"] = base["trade_date"].map(session_index)
    return _SourceState(
        out_index=out_index,
        anchor=anchor,
        anchor_codes=anchor_codes,
        price_groups={str(code): group for code, group in price.groupby("code", sort=False)},
        base_groups={str(code): group for code, group in base.groupby("code", sort=False)},
    )


def _snapshot(ctx: FactorComputeContext) -> tuple[_SourceState, pd.DataFrame]:
    state = _source_state(ctx)
    records: list[dict[str, object]] = []
    if state.anchor is None:
        return state, pd.DataFrame(index=pd.Index([], name="code"))
    for _, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        if not code or code not in state.anchor_codes:
            continue
        price = state.price_groups.get(str(code))
        base = state.base_groups.get(str(code))
        if price is None or base is None or price.empty or base.empty:
            continue
        if pd.Timestamp(price["trade_date"].iloc[-1]).normalize() != state.anchor:
            continue
        # Do not carry an old base mapping into the price-anchor date.
        if pd.Timestamp(base["trade_date"].iloc[-1]).normalize() != state.anchor:
            continue
        stock = _stock_state(base)
        if stock is None:
            continue
        bond = _bond_state(price)
        tracking_error = bond["__bond_return1"] - stock["__stock_return1"]
        flow_tracking = bond["__bond_liquidity_z20"] - stock["__stock_liquidity_z20"]
        record: dict[str, object] = {"code": str(code)}
        record.update(stock)
        record.update(bond)
        record["__tracking_error1"] = tracking_error if np.isfinite(tracking_error) else float("nan")
        record["__flow_tracking_gap20"] = flow_tracking if np.isfinite(flow_tracking) else float("nan")
        records.append(record)
    if not records:
        return state, pd.DataFrame(index=pd.Index([], name="code"))
    return state, pd.DataFrame(records).set_index("code").sort_index()


def _neighbor_map(snapshot: pd.DataFrame) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Build deterministic T-1 mapped-stock-state neighborhoods only."""

    if snapshot.empty:
        return {}
    columns = ("__stock_return1", "__stock_volatility", "__stock_liquidity_z20")
    values = snapshot.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    finite_counts = np.isfinite(values).sum(axis=0)
    if (finite_counts < _MIN_NEIGHBORS + 1).any():
        return {}
    centers = np.nanmedian(values, axis=0)
    scales = np.nanstd(values, axis=0, ddof=1)
    if not (np.isfinite(centers).all() and np.isfinite(scales).all() and (scales > _EPS).all()):
        return {}
    standardized = (values - centers) / scales
    valid = np.isfinite(standardized).all(axis=1)
    positions = np.flatnonzero(valid)
    if len(positions) < _MIN_NEIGHBORS + 1:
        return {}
    codes = snapshot.index.astype(str).to_numpy()
    out: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for position in positions:
        candidates = positions[positions != position]
        distances = np.sqrt(np.mean((standardized[candidates] - standardized[position]) ** 2, axis=1))
        if len(candidates) < _MIN_NEIGHBORS or not np.isfinite(distances).all():
            continue
        order = sorted(range(len(candidates)), key=lambda item: (float(distances[item]), str(codes[candidates[item]])))
        chosen_local = np.asarray(order[: min(_NEIGHBOR_COUNT, len(order))], dtype="int64")
        chosen = candidates[chosen_local]
        weights = 1.0 / (1.0 + distances[chosen_local])
        if len(chosen) < _MIN_NEIGHBORS or not np.isfinite(weights).all() or float(weights.sum()) <= _EPS:
            continue
        out[str(codes[position])] = (tuple(str(codes[item]) for item in chosen), weights)
    return out


def _peer_values(snapshot: pd.DataFrame, peer_codes: tuple[str, ...], weights: np.ndarray, column: str) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(snapshot.reindex(list(peer_codes))[column], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) < _MIN_NEIGHBORS:
        return np.asarray([], dtype="float64"), np.asarray([], dtype="float64")
    selected_values = values[valid]
    selected_weights = weights[valid]
    if float(selected_weights.sum()) <= _EPS:
        return np.asarray([], dtype="float64"), np.asarray([], dtype="float64")
    return selected_values, selected_weights


def _weighted_std(snapshot: pd.DataFrame, peer_codes: tuple[str, ...], weights: np.ndarray, column: str) -> float:
    values, selected_weights = _peer_values(snapshot, peer_codes, weights, column)
    if len(values) < _MIN_NEIGHBORS:
        return float("nan")
    total = float(selected_weights.sum())
    center = float(np.dot(selected_weights, values) / total)
    variance = float(np.dot(selected_weights, (values - center) ** 2) / total)
    return float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0.0 else float("nan")


def _weighted_skew(snapshot: pd.DataFrame, peer_codes: tuple[str, ...], weights: np.ndarray, column: str) -> float:
    values, selected_weights = _peer_values(snapshot, peer_codes, weights, column)
    if len(values) < _MIN_NEIGHBORS + 1:
        return float("nan")
    total = float(selected_weights.sum())
    center = float(np.dot(selected_weights, values) / total)
    variance = float(np.dot(selected_weights, (values - center) ** 2) / total)
    if not np.isfinite(variance) or variance <= _EPS:
        return float("nan")
    third_moment = float(np.dot(selected_weights, (values - center) ** 3) / total)
    value = third_moment / variance ** 1.5
    return float(value) if np.isfinite(value) else float("nan")


def _empty_family(state: _SourceState, family: str) -> pd.DataFrame:
    return pd.DataFrame(index=state.out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")


def _build_underlying_distribution_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "underlying_state_distribution"
    state, snapshot = _snapshot(ctx)
    out = _empty_family(state, family)
    neighbors = _neighbor_map(snapshot)
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        peer_state = neighbors.get(str(code))
        if peer_state is None:
            continue
        peer_codes, weights = peer_state
        out.at[(dt, raw_code), "ucd_peer_stock_return_dispersion1"] = _weighted_std(
            snapshot, peer_codes, weights, "__stock_return1"
        )
        out.at[(dt, raw_code), "ucd_peer_stock_return_skew1"] = _weighted_skew(
            snapshot, peer_codes, weights, "__stock_return1"
        )
        out.at[(dt, raw_code), "ucd_peer_stock_liquidity_shock_dispersion20"] = _weighted_std(
            snapshot, peer_codes, weights, "__stock_liquidity_z20"
        )
    return out.replace([np.inf, -np.inf], np.nan)


def _build_tracking_distribution_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "bond_stock_tracking_distribution"
    state, snapshot = _snapshot(ctx)
    out = _empty_family(state, family)
    neighbors = _neighbor_map(snapshot)
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        peer_state = neighbors.get(str(code))
        if peer_state is None:
            continue
        peer_codes, weights = peer_state
        out.at[(dt, raw_code), "utd_peer_tracking_error_dispersion1"] = _weighted_std(
            snapshot, peer_codes, weights, "__tracking_error1"
        )
        out.at[(dt, raw_code), "utd_peer_tracking_error_skew1"] = _weighted_skew(
            snapshot, peer_codes, weights, "__tracking_error1"
        )
        out.at[(dt, raw_code), "utd_peer_flow_tracking_dispersion20"] = _weighted_std(
            snapshot, peer_codes, weights, "__flow_tracking_gap20"
        )
    return out.replace([np.inf, -np.inf], np.nan)


_FAMILY_BUILDERS = {
    "underlying_state_distribution": _build_underlying_distribution_frame,
    "bond_stock_tracking_distribution": _build_tracking_distribution_frame,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_SIGNALS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _FAMILY_BUILDERS[family](ctx)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningUnderlyingCohortDistributionV1(Factor):
    """Research-only strict-T-1 underlying-state distribution kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        if params and str(params.get("signal", "")).strip():
            _requested_entry(params)
        return _requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningUnderlyingCohortDistributionV1",
    "factor_mining_catalog",
    "underlying_cohort_distribution_catalog",
]
