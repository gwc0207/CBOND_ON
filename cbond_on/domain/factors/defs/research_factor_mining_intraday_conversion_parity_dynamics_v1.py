"""Research-only T1430 conversion-parity wedge dynamics.

This catalogue starts from the latest completed, strict-prior conversion
contract and observes the mapped bond and stock only during the physical score
day's continuous auction through 14:29.  Its subject is not static conversion
moneyness or an unscaled stock/bond return gap: it asks whether the *distance*
between the executable bond price and the T-1 contract-implied parity value
converges, when that happens, and whether stock shocks trigger immediate or
deferred re-centering.

The module is import-only research code.  It consumes only the factor context,
does not read files or external state, and is intentionally not imported by
``defs.__init__`` or wired into configs, contracts, models, or live paths.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    DailyFactorRequirement,
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)


KERNEL_NAME = "factor_mining_intraday_conversion_parity_dynamics_v1"
CATALOG_VERSION = "20260803_intraday_conversion_parity_dynamics_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_PATH_POINTS = 12
_MIN_SEGMENT_POINTS = 4
_MIN_PHASE_POINTS = 3
_MIN_SHOCKS = 4
_MIN_SIDE_SHOCKS = 2
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)
_TAIL_START = dt_time(14, 0)
_TERMINAL_BIN = dt_time(14, 25)
_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})
_REQUIRED_PANEL_COLUMNS = ("trade_time", "last")


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable conversion-parity dynamics research candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class PriorContract:
    """One exact T-1 conversion contract and its verified mapped stock code."""

    conv_price: float
    stock_code: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_TRAJECTORY_SIGNALS = (
    "icpd_wedge_segment_convergence",
    "icpd_wedge_segment_path_efficiency",
    "icpd_wedge_segment_range",
)
_TIMING_SIGNALS = (
    "icpd_wedge_closest_parity_time",
    "icpd_wedge_early_late_distance_shift",
    "icpd_wedge_tail_reclaim",
)
_STOCK_CONDITIONED_SIGNALS = (
    "icpd_stock_shock_samebin_wedge_convergence",
    "icpd_stock_shock_nextbin_wedge_convergence",
    "icpd_stock_up_down_wedge_convergence_asymmetry",
)

_CATALOG = (
    _entries(
        "intraday_parity_wedge_trajectory",
        _TRAJECTORY_SIGNALS,
        "Within each uninterrupted five-minute score-day segment, a T-1-contract parity wedge can converge, travel efficiently, or remain broadly dispersed.",
    )
    + _entries(
        "intraday_parity_wedge_timing",
        _TIMING_SIGNALS,
        "The time and phase at which the dynamic contract-parity distance is smallest distinguish early discovery from late re-centering.",
    )
    + _entries(
        "stock_conditioned_parity_adjustment",
        _STOCK_CONDITIONED_SIGNALS,
        "Large mapped-stock moves can be followed by immediate or one-bin-delayed convergence of the conversion-parity distance, rather than a raw bond-stock return response.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "intraday_parity_wedge_trajectory": _TRAJECTORY_SIGNALS,
    "intraday_parity_wedge_timing": _TIMING_SIGNALS,
    "stock_conditioned_parity_adjustment": _STOCK_CONDITIONED_SIGNALS,
}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

FORMULAS = {
    "icpd_wedge_segment_convergence": "weighted mean over uninterrupted segments of |wedge_start|-|wedge_end|, where wedge=log(bond_last/(100*stock_last/cb_conv_price_T-1)).",
    "icpd_wedge_segment_path_efficiency": "weighted mean over uninterrupted segments of |wedge_end-wedge_start| / sum(|adjacent wedge changes|).",
    "icpd_wedge_segment_range": "weighted mean over uninterrupted segments of max(wedge)-min(wedge).",
    "icpd_wedge_closest_parity_time": "continuous-session-normalized five-minute time of min(|wedge|) on the physical score-day path.",
    "icpd_wedge_early_late_distance_shift": "mean(|wedge|, late>=13:30)-mean(|wedge|, early<=10:30).",
    "icpd_wedge_tail_reclaim": "max(|wedge|, 14:00-14:29)-|wedge_terminal|, requiring a 14:25 endpoint.",
    "icpd_stock_shock_samebin_wedge_convergence": "mean(|wedge_t-1|-|wedge_t| | |stock_log_return_t|>=q75) across contiguous five-minute edges.",
    "icpd_stock_shock_nextbin_wedge_convergence": "mean(|wedge_t|-|wedge_t+1| | stock shock at t, contiguous t+1).",
    "icpd_stock_up_down_wedge_convergence_asymmetry": "same-bin wedge convergence after positive stock shocks minus that after negative stock shocks.",
}


def intraday_conversion_parity_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return intraday_conversion_parity_dynamics_catalog()


def _requested_entry(params: dict | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _canonical_market_code(value: object, exchange: object | None = None) -> str:
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
        return f"{bare}.{suffix}" if bare and suffix in _MARKET_EXCHANGES else ""
    raw_exchange = "" if exchange is None or pd.isna(exchange) else str(exchange).strip().upper()
    suffix = _EXCHANGE_ALIASES.get(raw_exchange, raw_exchange)
    return f"{text}.{suffix}" if suffix in _MARKET_EXCHANGES else ""


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        score_date = pd.Timestamp(raw)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if panel.empty:
        return None
    days = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(days[days.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return _empty_index()
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    labels = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[labels == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"]) if not keys.empty else _empty_index()


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(
    panel: pd.DataFrame | None,
    *,
    score_date: pd.Timestamp,
    owner: str,
) -> pd.DataFrame:
    """Return causal, physical, valid-price score-day snapshots only."""

    columns = ["dt", "code", "seq", "trade_time", "last"]
    if panel is None or panel.empty:
        return pd.DataFrame(columns=columns)
    missing = [column for column in _REQUIRED_PANEL_COLUMNS if column not in panel.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required columns: {missing}")
    checked = ensure_panel_index(panel)
    frame = checked.reset_index()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required index columns: {missing}")
    frame = frame.loc[:, columns].copy()
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
    out["trade_time"] = timestamps.loc[keep]
    out["last"] = pd.to_numeric(out["last"], errors="coerce")
    return out.loc[np.isfinite(out["last"]) & (out["last"] > _EPS)].sort_values(
        ["code", "trade_time", "seq"], kind="mergesort"
    )


def _context_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Use only unambiguous, non-forward mappings supplied by the pipeline."""

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
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.loc[dates.notna() & (dates <= score_date)].copy()
    if frame.empty:
        return {}
    frame["bond_code"] = frame["code"].map(_canonical_market_code)
    frame["mapped_stock_code"] = frame["stock_code"].map(_canonical_market_code)
    frame = frame.loc[(frame["bond_code"] != "") & (frame["mapped_stock_code"] != "")].copy()
    if frame.empty:
        return {}
    ambiguous = frame["bond_code"].duplicated(keep=False)
    frame = frame.loc[~ambiguous]
    return dict(zip(frame["bond_code"], frame["mapped_stock_code"], strict=False))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _strict_daily_source(
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
    frame["code"] = [
        _canonical_market_code(code, exchange)
        for code, exchange in zip(frame["code"], frame["exchange_code"], strict=False)
    ]
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & (frame["code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]
        ].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _prior_contracts(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, PriorContract]:
    """Return only contracts present on the exact T-1 daily-price anchor."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:contracts:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, dict):
            return cached

    price = _strict_daily_source(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    base = _strict_daily_source(
        ctx,
        source="market_cbond.daily_base",
        fields=("cb_conv_price", "stock_code"),
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    contracts: dict[str, PriorContract] = {}
    if not price.empty:
        anchor = pd.Timestamp(price["trade_date"].max()).normalize()
        anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
        snapshot = base.loc[(base["trade_date"] == anchor) & base["code"].isin(anchor_codes)].copy()
        snapshot["cb_conv_price"] = pd.to_numeric(snapshot["cb_conv_price"], errors="coerce")
        snapshot["prior_stock_code"] = snapshot["stock_code"].map(_canonical_market_code)
        snapshot = snapshot.loc[
            np.isfinite(snapshot["cb_conv_price"])
            & (snapshot["cb_conv_price"] > _EPS)
            & (snapshot["prior_stock_code"] != "")
        ]
        contracts = {
            str(row.code): PriorContract(float(row.cb_conv_price), str(row.prior_stock_code))
            for row in snapshot.itertuples(index=False)
        }

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, dict):
            return existing
        ctx.cache[cache_key] = contracts
    return contracts


def _endpoint_prices(frame: pd.DataFrame) -> pd.Series:
    """Use the last physical tick in each five-minute bin; never manufacture a bin."""

    if frame.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="event_time"))
    data = frame.loc[:, ["trade_time", "seq", "last"]].copy()
    data["event_time"] = pd.to_datetime(data["trade_time"], errors="coerce").dt.floor(_EVENT_BIN)
    data = data.loc[data["event_time"].notna() & np.isfinite(data["last"]) & (data["last"] > _EPS)]
    if data.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="event_time"))
    out = (
        data.sort_values(["event_time", "trade_time", "seq"], kind="mergesort")
        .groupby("event_time", sort=True)["last"]
        .last()
        .astype("float64")
    )
    out.index.name = "event_time"
    return out


def _parity_path(
    bond_frame: pd.DataFrame,
    stock_frame: pd.DataFrame,
    *,
    conv_price: float,
) -> pd.DataFrame:
    bond = _endpoint_prices(bond_frame).rename("bond_last")
    stock = _endpoint_prices(stock_frame).rename("stock_last")
    joined = bond.to_frame().join(stock, how="inner").dropna().sort_index()
    if len(joined) < _MIN_PATH_POINTS or not np.isfinite(conv_price) or conv_price <= _EPS:
        return pd.DataFrame()
    bond_values = joined["bond_last"].to_numpy(dtype="float64")
    stock_values = joined["stock_last"].to_numpy(dtype="float64")
    valid = (
        np.isfinite(bond_values)
        & np.isfinite(stock_values)
        & (bond_values > _EPS)
        & (stock_values > _EPS)
    )
    joined = joined.loc[valid].copy()
    if len(joined) < _MIN_PATH_POINTS:
        return pd.DataFrame()
    parity = 100.0 * joined["stock_last"].to_numpy(dtype="float64") / float(conv_price)
    if not np.isfinite(parity).all() or (parity <= _EPS).any():
        return pd.DataFrame()
    joined["wedge"] = np.log(joined["bond_last"].to_numpy(dtype="float64") / parity)
    if not np.isfinite(joined["wedge"].to_numpy(dtype="float64")).all():
        return pd.DataFrame()
    contiguous = joined.index.to_series().diff().eq(_EVENT_BIN).to_numpy()
    joined["__contiguous"] = contiguous
    joined["__segment"] = (~joined["__contiguous"]).cumsum().astype("int64")
    return joined


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return float("nan")
    array = np.asarray(values, dtype="float64")
    mass = np.asarray(weights, dtype="float64")
    valid = np.isfinite(array) & np.isfinite(mass) & (mass > 0.0)
    if not valid.any():
        return float("nan")
    return float(np.average(array[valid], weights=mass[valid]))


def _trajectory_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TRAJECTORY_SIGNALS}
    if path.empty:
        return out
    convergence: list[float] = []
    efficiency: list[float] = []
    ranges: list[float] = []
    weights: list[float] = []
    for _, segment in path.groupby("__segment", sort=False):
        if len(segment) < _MIN_SEGMENT_POINTS:
            continue
        wedge = segment["wedge"].to_numpy(dtype="float64")
        changes = np.diff(wedge)
        variation = float(np.abs(changes).sum())
        if not np.isfinite(wedge).all() or not np.isfinite(variation) or variation <= _EPS:
            continue
        mass = float(len(segment) - 1)
        convergence.append(float(abs(wedge[0]) - abs(wedge[-1])))
        efficiency.append(float(abs(wedge[-1] - wedge[0]) / variation))
        ranges.append(float(np.max(wedge) - np.min(wedge)))
        weights.append(mass)
    out.update(
        {
            "icpd_wedge_segment_convergence": _weighted_mean(convergence, weights),
            "icpd_wedge_segment_path_efficiency": _weighted_mean(efficiency, weights),
            "icpd_wedge_segment_range": _weighted_mean(ranges, weights),
        }
    )
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _continuous_coordinate(timestamp: pd.Timestamp) -> float:
    clock = timestamp.time()
    minutes = clock.hour * 60 + clock.minute
    if _MORNING_START <= clock <= _MORNING_END:
        elapsed = minutes - (_MORNING_START.hour * 60 + _MORNING_START.minute)
    elif _AFTERNOON_START <= clock <= _CUTOFF:
        elapsed = 120 + minutes - (_AFTERNOON_START.hour * 60 + _AFTERNOON_START.minute)
    else:
        return float("nan")
    total = 120 + ((_CUTOFF.hour * 60 + _CUTOFF.minute) - (_AFTERNOON_START.hour * 60 + _AFTERNOON_START.minute))
    return float(elapsed / total) if total > 0 else float("nan")


def _timing_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TIMING_SIGNALS}
    if path.empty or len(path) < _MIN_PATH_POINTS:
        return out
    clocks = pd.Series(path.index.time, index=path.index)
    terminal_clock = path.index[-1].time()
    if terminal_clock < _TERMINAL_BIN:
        return out
    absolute_wedge = path["wedge"].abs()
    closest_time = _continuous_coordinate(pd.Timestamp(absolute_wedge.idxmin()))
    early = absolute_wedge.loc[clocks <= _EARLY_END]
    late = absolute_wedge.loc[clocks >= _LATE_START]
    tail = absolute_wedge.loc[clocks >= _TAIL_START]
    if len(early) >= _MIN_PHASE_POINTS and len(late) >= _MIN_PHASE_POINTS:
        out["icpd_wedge_early_late_distance_shift"] = float(late.mean() - early.mean())
    if len(tail) >= _MIN_PHASE_POINTS:
        out["icpd_wedge_tail_reclaim"] = float(tail.max() - absolute_wedge.iloc[-1])
    out["icpd_wedge_closest_parity_time"] = closest_time
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _stock_conditioned_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _STOCK_CONDITIONED_SIGNALS}
    if path.empty or len(path) < _MIN_PATH_POINTS:
        return out
    stock_log_return = np.log(path["stock_last"]).diff()
    closure = path["wedge"].abs().shift(1) - path["wedge"].abs()
    edge = pd.DataFrame({"stock_return": stock_log_return, "closure": closure}, index=path.index)
    edge = edge.loc[path["__contiguous"].to_numpy()].dropna().sort_index()
    if len(edge) < _MIN_SHOCKS:
        return out
    absolute_stock_return = edge["stock_return"].abs().to_numpy(dtype="float64")
    threshold = float(np.quantile(absolute_stock_return, 0.75))
    if not np.isfinite(threshold) or threshold <= _EPS:
        return out
    shocks = edge.loc[absolute_stock_return >= threshold].copy()
    if len(shocks) < _MIN_SHOCKS:
        return out
    out["icpd_stock_shock_samebin_wedge_convergence"] = float(shocks["closure"].mean())
    up = shocks.loc[shocks["stock_return"] > 0.0, "closure"]
    down = shocks.loc[shocks["stock_return"] < 0.0, "closure"]
    if len(up) >= _MIN_SIDE_SHOCKS and len(down) >= _MIN_SIDE_SHOCKS:
        out["icpd_stock_up_down_wedge_convergence_asymmetry"] = float(up.mean() - down.mean())

    next_index = pd.DatetimeIndex(shocks.index) + _EVENT_BIN
    next_closure = edge["closure"].reindex(next_index)
    if int(next_closure.notna().sum()) >= _MIN_SHOCKS:
        out["icpd_stock_shock_nextbin_wedge_convergence"] = float(next_closure.dropna().mean())
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _metrics(path: pd.DataFrame, family: str) -> dict[str, float]:
    if family == "intraday_parity_wedge_trajectory":
        return _trajectory_metrics(path)
    if family == "intraday_parity_wedge_timing":
        return _timing_metrics(path)
    if family == "stock_conditioned_parity_adjustment":
        return _stock_conditioned_metrics(path)
    raise KeyError(f"{KERNEL_NAME} unknown family: {family}")


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_SIGNALS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
    else:
        contracts = _prior_contracts(ctx, score_date=score_date)
        mapping = _context_mapping(ctx, score_date=score_date)
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date, owner="panel")
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date, owner="stock_panel")
        bond_groups = {
            canonical: group
            for code, group in bond_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        stock_groups = {
            canonical: group
            for code, group in stock_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        rows: list[dict[str, object]] = []
        for dt, raw_bond_code in output_index:
            canonical_bond = _canonical_market_code(raw_bond_code)
            row: dict[str, object] = {"dt": dt, "code": raw_bond_code}
            row.update({signal: float("nan") for signal in _FAMILY_SIGNALS[family]})
            contract = contracts.get(canonical_bond)
            mapped_stock = mapping.get(canonical_bond, "")
            if (
                contract is not None
                and mapped_stock == contract.stock_code
                and canonical_bond in bond_groups
                and mapped_stock in stock_groups
            ):
                path = _parity_path(
                    bond_groups[canonical_bond],
                    stock_groups[mapped_stock],
                    conv_price=contract.conv_price,
                )
                row.update(_metrics(path, family))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_FAMILY_SIGNALS[family])]
        built = built.reindex(output_index).replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayConversionParityDynamicsV1(Factor):
    """Research-only dynamic T-1-contract conversion-parity factor kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement(
                "market_cbond.daily_price",
                ("exchange_code", "close_price"),
                lookback_days=2,
            ),
            DailyFactorRequirement(
                "market_cbond.daily_base",
                ("exchange_code", "cb_conv_price", "stock_code"),
                lookback_days=2,
            ),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)
