"""Research-only strict-T-1 structural-neighborhood factor catalogue.

The catalogue treats the latest *completed* ``daily_price`` row before the
score date as the only market-calendar anchor.  ``daily_base`` is then joined
on the same ``(trade_date, code)`` key.  A stale, missing, duplicate, or
otherwise mismatched base record never becomes the current structural state:
that instrument simply receives ``NaN`` for the affected family.

The four families deliberately use local peer geometry and conditional
cohort states rather than a cross-sectional rank or a fitted residual:

* structurally similar peers' prior return/risk regime;
* the density and heterogeneity of that structural neighborhood;
* same-rating / same-term cohort state gaps; and
* peer transmission among bonds whose mapped underlying stocks share a
  recent observable state.

This module is import-only and research-only.  It reads only declared
``FactorComputeContext.daily_data`` tables and has no access to labels, pool,
scores, PnL, files, databases, Redis, backtests, or live artifacts.
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


KERNEL_NAME = "factor_mining_structural_neighborhood_v1"
CATALOG_VERSION = "20260803_structural_neighborhood_v1"
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


_PEER_TRANSMISSION_SIGNALS = (
    "spt_peer_return1_mean",
    "spt_peer_return5_mean",
    "spt_peer_volatility20_mean",
)
_NEIGHBORHOOD_GEOMETRY_SIGNALS = (
    "sng_inverse_distance_density",
    "sng_peer_return_dispersion1",
    "sng_peer_liquidity_dispersion20",
)
_RATING_TERM_COHORT_SIGNALS = (
    "rtc_premium_gap_median",
    "rtc_yield_change5_gap_median",
    "rtc_liquidity_shock_gap_median",
)
_STOCK_MAPPING_COHORT_SIGNALS = (
    "smc_peer_bond_return1_mean",
    "smc_peer_stock_return1_mean",
    "smc_peer_liquidity_wedge_mean",
)

_CATALOG = (
    _entries(
        "structural_peer_state_transmission",
        _PEER_TRANSMISSION_SIGNALS,
        "Bonds with similar T-1 maturity, duration, premium, yield, and float can share a prior-day return or risk regime without fitting a cross-sectional residual.",
    )
    + _entries(
        "structural_neighborhood_geometry",
        _NEIGHBORHOOD_GEOMETRY_SIGNALS,
        "The local density and state dispersion around an instrument's T-1 structural neighborhood measure crowding and uncertainty, not the instrument's raw level or rank.",
    )
    + _entries(
        "rating_term_cohort_deviation",
        _RATING_TERM_COHORT_SIGNALS,
        "A bond's T-1 premium, yield transition, and liquidity surprise can be evaluated against peers in the same reported rating and remaining-term cohort.",
    )
    + _entries(
        "underlying_stock_state_cohort",
        _STOCK_MAPPING_COHORT_SIGNALS,
        "Mapped-underlying stock states define a T-1 peer cohort whose bond return and liquidity transmission may differ from a pure bond-structure neighborhood.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "structural_peer_state_transmission": _PEER_TRANSMISSION_SIGNALS,
    "structural_neighborhood_geometry": _NEIGHBORHOOD_GEOMETRY_SIGNALS,
    "rating_term_cohort_deviation": _RATING_TERM_COHORT_SIGNALS,
    "underlying_stock_state_cohort": _STOCK_MAPPING_COHORT_SIGNALS,
}

# Every family uses daily_price as its independent strict-prior calendar
# anchor.  daily_base is only allowed to participate after exact date-key
# matching to that calendar.
_FAMILY_PRICE_FIELDS: dict[str, tuple[str, ...]] = {
    "structural_peer_state_transmission": ("prev_close_price", "close_price"),
    "structural_neighborhood_geometry": ("prev_close_price", "close_price", "amount"),
    "rating_term_cohort_deviation": ("prev_close_price", "close_price", "amount"),
    "underlying_stock_state_cohort": ("prev_close_price", "close_price", "amount"),
}
_FAMILY_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "structural_peer_state_transmission": (
        "year_to_mat",
        "duration",
        "bond_prem_ratio",
        "ytm",
        "remain_size",
    ),
    "structural_neighborhood_geometry": (
        "year_to_mat",
        "duration",
        "bond_prem_ratio",
        "ytm",
        "remain_size",
    ),
    "rating_term_cohort_deviation": ("rating", "year_to_mat", "bond_prem_ratio", "ytm"),
    "underlying_stock_state_cohort": (
        "stock_code",
        "stock_close_price",
        "stock_volatility",
        "stk_amount",
    ),
}


def structural_neighborhood_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable, family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return structural_neighborhood_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirements_for_family(family: str) -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code", *_FAMILY_PRICE_FIELDS[family]),
            _LOOKBACK_DAYS,
        ),
        DailyFactorRequirement(
            "market_cbond.daily_base",
            ("exchange_code", *_FAMILY_BASE_FIELDS[family]),
            _LOOKBACK_DAYS,
        ),
    ]


def _all_requirements() -> list[DailyFactorRequirement]:
    price_fields: set[str] = {"exchange_code"}
    base_fields: set[str] = {"exchange_code"}
    for fields in _FAMILY_PRICE_FIELDS.values():
        price_fields.update(fields)
    for fields in _FAMILY_BASE_FIELDS.values():
        base_fields.update(fields)
    return [
        DailyFactorRequirement("market_cbond.daily_price", tuple(sorted(price_fields)), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_base", tuple(sorted(base_fields)), _LOOKBACK_DAYS),
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
    """Derive keys from the panel only; never consume panel values as inputs."""

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
        keep = dates == score_date
        if not panel.empty and not bool(np.any(keep)):
            raise ValueError(f"{KERNEL_NAME} has no indexed rows for score date {score_date.date().isoformat()}")
        selected = panel.loc[keep]
        keys = selected.index.droplevel("seq").unique()
        index = pd.MultiIndex.from_tuples(keys.tolist(), names=["dt", "code"]).sort_values()

    frame = pd.DataFrame(index=index)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing.index
        ctx.cache[cache_key] = frame
    return index


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required columns: {missing}")


def _strict_history_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return only declared strict-prior rows, rejecting duplicate history."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), owner=source)
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
    """Return a finite run of adjacent daily-price sessions or fail closed."""

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


def _safe_log(value: float) -> float:
    if not np.isfinite(value) or value <= _EPS:
        return float("nan")
    return float(np.log(value))


def _last_price_metrics(price: pd.DataFrame, *, include_amount: bool) -> dict[str, float]:
    out = {
        "__bond_return1": float("nan"),
        "__bond_return5": float("nan"),
        "__bond_volatility20": float("nan"),
        "__bond_liquidity_z20": float("nan"),
        "__bond_amount": float("nan"),
    }
    last = _complete_tail(price, ("prev_close_price", "close_price"), 1)
    if last is not None:
        out["__bond_return1"] = _safe_return(float(last[-1, 1]), float(last[-1, 0]))

    tail6 = _complete_tail(price, ("close_price",), 6)
    if tail6 is not None:
        out["__bond_return5"] = _safe_return(float(tail6[-1, 0]), float(tail6[0, 0]))

    tail20 = _complete_tail(price, ("prev_close_price", "close_price"), 20)
    if tail20 is not None:
        returns = tail20[:, 1] / tail20[:, 0] - 1.0
        if np.isfinite(returns).all() and (tail20 > _EPS).all():
            volatility = float(np.std(returns, ddof=1))
            if np.isfinite(volatility) and volatility > _EPS:
                out["__bond_volatility20"] = volatility

    if include_amount:
        amount_last = _complete_tail(price, ("amount",), 1)
        if amount_last is not None:
            out["__bond_amount"] = float(amount_last[-1, 0])
        amount_tail = _complete_tail(price, ("amount",), 21)
        if amount_tail is not None and (amount_tail > _EPS).all():
            log_amount = np.log(amount_tail[:, 0])
            prior = log_amount[:-1]
            scale = float(np.std(prior, ddof=1))
            if np.isfinite(scale) and scale > _EPS:
                out["__bond_liquidity_z20"] = float((log_amount[-1] - float(np.mean(prior))) / scale)
    return out


def _structural_coordinates(base: pd.DataFrame) -> dict[str, float] | None:
    fields = ("year_to_mat", "duration", "bond_prem_ratio", "ytm", "remain_size")
    latest = _complete_tail(base, fields, 1)
    if latest is None:
        return None
    maturity, duration, premium, ytm, remain_size = (float(value) for value in latest[-1])
    log_remain = _safe_log(remain_size)
    if not (np.isfinite(maturity) and maturity >= 0.0 and np.isfinite(duration) and duration >= 0.0):
        return None
    if not (np.isfinite(premium) and np.isfinite(ytm) and np.isfinite(log_remain)):
        return None
    return {
        "__year_to_mat": maturity,
        "__duration": duration,
        "__bond_prem_ratio": premium,
        "__ytm": ytm,
        "__log_remain_size": log_remain,
    }


def _rating_term_state(base: pd.DataFrame) -> dict[str, object] | None:
    latest = _complete_tail(base, ("year_to_mat", "bond_prem_ratio", "ytm"), 1)
    if latest is None:
        return None
    maturity, premium, ytm = (float(value) for value in latest[-1])
    rating_raw = base["rating"].iloc[-1]
    rating = "" if pd.isna(rating_raw) else str(rating_raw).strip().upper().replace(" ", "")
    if not rating or not (np.isfinite(maturity) and maturity >= 0.0 and np.isfinite(premium) and np.isfinite(ytm)):
        return None
    ytm_tail = _complete_tail(base, ("ytm",), 6)
    ytm_change5 = float("nan")
    if ytm_tail is not None:
        ytm_change5 = float(ytm_tail[-1, 0] - ytm_tail[0, 0])
    return {
        "__rating": rating,
        "__term_bucket": int(np.floor(maturity)),
        "__cohort_premium": premium,
        "__cohort_ytm_change5": ytm_change5,
    }


def _stock_mapping_state(base: pd.DataFrame) -> dict[str, float] | None:
    mapping_raw = base["stock_code"].iloc[-1]
    mapping = "" if pd.isna(mapping_raw) else str(mapping_raw).strip().upper()
    if not mapping or mapping == "NAN":
        return None
    latest = _complete_tail(base, ("stock_volatility",), 1)
    if latest is None:
        return None
    stock_volatility = float(latest[-1, 0])
    if not np.isfinite(stock_volatility) or stock_volatility <= _EPS:
        return None

    stock_tail2 = _complete_tail(base, ("stock_close_price",), 2)
    stock_return1 = float("nan")
    if stock_tail2 is not None:
        stock_return1 = _safe_return(float(stock_tail2[-1, 0]), float(stock_tail2[-2, 0]))

    amount_tail = _complete_tail(base, ("stk_amount",), 21)
    stock_liquidity_z20 = float("nan")
    if amount_tail is not None and (amount_tail > _EPS).all():
        log_amount = np.log(amount_tail[:, 0])
        prior = log_amount[:-1]
        scale = float(np.std(prior, ddof=1))
        if np.isfinite(scale) and scale > _EPS:
            stock_liquidity_z20 = float((log_amount[-1] - float(np.mean(prior))) / scale)

    stock_amount_latest = _complete_tail(base, ("stk_amount",), 1)
    stock_amount = float(stock_amount_latest[-1, 0]) if stock_amount_latest is not None else float("nan")
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


def _source_state(ctx: FactorComputeContext, family: str) -> _SourceState:
    out_index = _output_index(ctx)
    score_date = _score_date_from_panel(ctx.panel)
    if out_index.empty or score_date is None:
        return _SourceState(out_index, None, set(), {}, {})

    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=_FAMILY_PRICE_FIELDS[family],
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_FAMILY_BASE_FIELDS[family],
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

    # Exact inner join is intentional: base rows that are not present on the
    # independently anchored price calendar cannot define a historical state.
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


def _iter_anchored_groups(state: _SourceState) -> Iterable[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield only codes with exact price and base observations on the price anchor."""

    if state.anchor is None:
        return
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
        # This is the anti-staleness / anti-mismatch boundary required before
        # any structural metric is emitted.
        if pd.Timestamp(base["trade_date"].iloc[-1]).normalize() != state.anchor:
            continue
        yield str(code), price, base


def _snapshot_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(index=pd.Index([], name="code"))
    return pd.DataFrame(records).set_index("code").sort_index()


def _structural_snapshot(ctx: FactorComputeContext, family: str, *, include_amount: bool) -> tuple[_SourceState, pd.DataFrame]:
    state = _source_state(ctx, family)
    records: list[dict[str, object]] = []
    for code, price, base in _iter_anchored_groups(state):
        structure = _structural_coordinates(base)
        if structure is None:
            continue
        record: dict[str, object] = {"code": code}
        record.update(structure)
        record.update(_last_price_metrics(price, include_amount=include_amount))
        records.append(record)
    return state, _snapshot_from_records(records)


def _rating_term_snapshot(ctx: FactorComputeContext) -> tuple[_SourceState, pd.DataFrame]:
    state = _source_state(ctx, "rating_term_cohort_deviation")
    records: list[dict[str, object]] = []
    for code, price, base in _iter_anchored_groups(state):
        cohort = _rating_term_state(base)
        if cohort is None:
            continue
        record: dict[str, object] = {"code": code}
        record.update(cohort)
        record.update(_last_price_metrics(price, include_amount=True))
        records.append(record)
    return state, _snapshot_from_records(records)


def _stock_mapping_snapshot(ctx: FactorComputeContext) -> tuple[_SourceState, pd.DataFrame]:
    state = _source_state(ctx, "underlying_stock_state_cohort")
    records: list[dict[str, object]] = []
    for code, price, base in _iter_anchored_groups(state):
        stock = _stock_mapping_state(base)
        if stock is None:
            continue
        bond = _last_price_metrics(price, include_amount=True)
        wedge = float("nan")
        if (
            np.isfinite(bond["__bond_amount"])
            and np.isfinite(stock["__stock_amount"])
            and float(bond["__bond_amount"]) > _EPS
            and float(stock["__stock_amount"]) > _EPS
        ):
            wedge = _safe_log(float(bond["__bond_amount"]) / float(stock["__stock_amount"]))
        record: dict[str, object] = {"code": code}
        record.update(stock)
        record.update(bond)
        record["__bond_stock_liquidity_wedge"] = wedge
        records.append(record)
    return state, _snapshot_from_records(records)


def _neighbor_map(snapshot: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, tuple[tuple[str, ...], np.ndarray, float]]:
    """Create deterministic inverse-distance local neighborhoods.

    Coordinates are centered/scaled only to make dimensions commensurate; the
    output remains a local peer aggregation, never a rank or fitted residual.
    A missing/non-varying coordinate makes the neighborhood undefined rather
    than silently substituting a different factor definition.
    """

    if snapshot.empty:
        return {}
    values = snapshot.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    numeric = values.to_numpy(dtype="float64")
    centers = np.nanmedian(numeric, axis=0)
    scales = np.nanstd(numeric, axis=0, ddof=1)
    if not (np.isfinite(centers).all() and np.isfinite(scales).all() and (scales > _EPS).all()):
        return {}
    standardized = (numeric - centers) / scales
    valid = np.isfinite(standardized).all(axis=1)
    codes = snapshot.index.astype(str).to_numpy()
    available = np.flatnonzero(valid)
    if len(available) < _MIN_NEIGHBORS + 1:
        return {}

    out: dict[str, tuple[tuple[str, ...], np.ndarray, float]] = {}
    for position in available:
        candidates = available[available != position]
        if len(candidates) < _MIN_NEIGHBORS:
            continue
        distances = np.sqrt(np.mean((standardized[candidates] - standardized[position]) ** 2, axis=1))
        if not np.isfinite(distances).all():
            continue
        order = sorted(range(len(candidates)), key=lambda item: (float(distances[item]), str(codes[candidates[item]])))
        picked_local = np.asarray(order[: min(_NEIGHBOR_COUNT, len(order))], dtype="int64")
        picked = candidates[picked_local]
        picked_distances = distances[picked_local]
        weights = 1.0 / (1.0 + picked_distances)
        if len(picked) < _MIN_NEIGHBORS or not (np.isfinite(weights).all() and float(weights.sum()) > _EPS):
            continue
        mean_distance = float(np.dot(weights, picked_distances) / float(weights.sum()))
        out[str(codes[position])] = (tuple(str(codes[item]) for item in picked), weights, mean_distance)
    return out


def _weighted_mean(snapshot: pd.DataFrame, peer_codes: tuple[str, ...], weights: np.ndarray, column: str) -> float:
    values = pd.to_numeric(snapshot.reindex(list(peer_codes))[column], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) < _MIN_NEIGHBORS:
        return float("nan")
    selected_weights = weights[valid]
    total = float(selected_weights.sum())
    if total <= _EPS:
        return float("nan")
    return float(np.dot(selected_weights, values[valid]) / total)


def _weighted_std(snapshot: pd.DataFrame, peer_codes: tuple[str, ...], weights: np.ndarray, column: str) -> float:
    values = pd.to_numeric(snapshot.reindex(list(peer_codes))[column], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) < _MIN_NEIGHBORS:
        return float("nan")
    selected_values = values[valid]
    selected_weights = weights[valid]
    total = float(selected_weights.sum())
    if total <= _EPS:
        return float("nan")
    mean = float(np.dot(selected_weights, selected_values) / total)
    variance = float(np.dot(selected_weights, (selected_values - mean) ** 2) / total)
    return float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0.0 else float("nan")


def _empty_family(state: _SourceState, family: str) -> pd.DataFrame:
    return pd.DataFrame(index=state.out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")


def _build_peer_transmission_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "structural_peer_state_transmission"
    state, snapshot = _structural_snapshot(ctx, family, include_amount=False)
    out = _empty_family(state, family)
    neighbors = _neighbor_map(
        snapshot,
        ("__year_to_mat", "__duration", "__bond_prem_ratio", "__ytm", "__log_remain_size"),
    )
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        peer_state = neighbors.get(str(code))
        if peer_state is None:
            continue
        peer_codes, weights, _ = peer_state
        out.at[(dt, raw_code), "spt_peer_return1_mean"] = _weighted_mean(snapshot, peer_codes, weights, "__bond_return1")
        out.at[(dt, raw_code), "spt_peer_return5_mean"] = _weighted_mean(snapshot, peer_codes, weights, "__bond_return5")
        out.at[(dt, raw_code), "spt_peer_volatility20_mean"] = _weighted_mean(
            snapshot, peer_codes, weights, "__bond_volatility20"
        )
    return out.replace([np.inf, -np.inf], np.nan)


def _build_neighborhood_geometry_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "structural_neighborhood_geometry"
    state, snapshot = _structural_snapshot(ctx, family, include_amount=True)
    out = _empty_family(state, family)
    neighbors = _neighbor_map(
        snapshot,
        ("__year_to_mat", "__duration", "__bond_prem_ratio", "__ytm", "__log_remain_size"),
    )
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        peer_state = neighbors.get(str(code))
        if peer_state is None:
            continue
        peer_codes, weights, mean_distance = peer_state
        if np.isfinite(mean_distance) and mean_distance >= 0.0:
            out.at[(dt, raw_code), "sng_inverse_distance_density"] = 1.0 / (1.0 + mean_distance)
        out.at[(dt, raw_code), "sng_peer_return_dispersion1"] = _weighted_std(
            snapshot, peer_codes, weights, "__bond_return1"
        )
        out.at[(dt, raw_code), "sng_peer_liquidity_dispersion20"] = _weighted_std(
            snapshot, peer_codes, weights, "__bond_liquidity_z20"
        )
    return out.replace([np.inf, -np.inf], np.nan)


def _cohort_peer_median(snapshot: pd.DataFrame, code: str, column: str) -> float:
    if code not in snapshot.index:
        return float("nan")
    record = snapshot.loc[code]
    rating = record.get("__rating")
    term_bucket = record.get("__term_bucket")
    if pd.isna(rating) or pd.isna(term_bucket):
        return float("nan")
    peers = snapshot.loc[(snapshot["__rating"] == rating) & (snapshot["__term_bucket"] == term_bucket)]
    peers = peers.drop(index=code, errors="ignore")
    values = pd.to_numeric(peers[column], errors="coerce").dropna().to_numpy(dtype="float64")
    if len(values) < _MIN_NEIGHBORS:
        return float("nan")
    value = float(np.median(values))
    return value if np.isfinite(value) else float("nan")


def _build_rating_term_cohort_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "rating_term_cohort_deviation"
    state, snapshot = _rating_term_snapshot(ctx)
    out = _empty_family(state, family)
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        if code not in snapshot.index:
            continue
        record = snapshot.loc[code]
        for signal, column in (
            ("rtc_premium_gap_median", "__cohort_premium"),
            ("rtc_yield_change5_gap_median", "__cohort_ytm_change5"),
            ("rtc_liquidity_shock_gap_median", "__bond_liquidity_z20"),
        ):
            own = pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]
            peer_median = _cohort_peer_median(snapshot, str(code), column)
            if np.isfinite(own) and np.isfinite(peer_median):
                out.at[(dt, raw_code), signal] = float(own - peer_median)
    return out.replace([np.inf, -np.inf], np.nan)


def _build_stock_mapping_cohort_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    family = "underlying_stock_state_cohort"
    state, snapshot = _stock_mapping_snapshot(ctx)
    out = _empty_family(state, family)
    neighbors = _neighbor_map(
        snapshot,
        ("__stock_return1", "__stock_volatility", "__stock_liquidity_z20"),
    )
    for dt, raw_code in state.out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        peer_state = neighbors.get(str(code))
        if peer_state is None:
            continue
        peer_codes, weights, _ = peer_state
        out.at[(dt, raw_code), "smc_peer_bond_return1_mean"] = _weighted_mean(
            snapshot, peer_codes, weights, "__bond_return1"
        )
        out.at[(dt, raw_code), "smc_peer_stock_return1_mean"] = _weighted_mean(
            snapshot, peer_codes, weights, "__stock_return1"
        )
        out.at[(dt, raw_code), "smc_peer_liquidity_wedge_mean"] = _weighted_mean(
            snapshot, peer_codes, weights, "__bond_stock_liquidity_wedge"
        )
    return out.replace([np.inf, -np.inf], np.nan)


_FAMILY_BUILDERS = {
    "structural_peer_state_transmission": _build_peer_transmission_frame,
    "structural_neighborhood_geometry": _build_neighborhood_geometry_frame,
    "rating_term_cohort_deviation": _build_rating_term_cohort_frame,
    "underlying_stock_state_cohort": _build_stock_mapping_cohort_frame,
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
class FactorMiningStructuralNeighborhoodV1(Factor):
    """Research-only strict-T-1 structural-neighborhood kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        values = dict(params or {})
        if not str(values.get("signal", "")).strip():
            return _all_requirements()
        return _requirements_for_family(_requested_entry(values).family)

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
    "FactorMiningStructuralNeighborhoodV1",
    "factor_mining_catalog",
    "structural_neighborhood_catalog",
]
