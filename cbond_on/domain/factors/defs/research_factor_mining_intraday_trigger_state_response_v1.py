"""Research-only dynamic intraday response under strict T-1 trigger state.

The existing trigger and redemption catalogues describe completed daily
contract state.  This module deliberately does something narrower and
different: it observes the physical T-day mapped bond/stock path through
14:29 and asks how price discovery behaves while conditioned on the already
known T-1 revised trigger, call, put, and active-redemption state.

No raw trigger-progress level, static barrier distance, factor score, label,
PnL, file, database, or live artifact enters a calculation.  The module is
import-only research code and intentionally stays out of ``defs.__init__``.
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


KERNEL_NAME = "factor_mining_intraday_trigger_state_response_v1"
CATALOG_VERSION = "20260803_intraday_trigger_state_response_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_PATH_POINTS = 12
_MIN_SEGMENT_POINTS = 4
_MIN_PHASE_POINTS = 3
_MIN_APPROACH_EVENTS = 4
_NEAR_COMPLETION_DAYS = 3.0
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
    """One auditable trigger-state response candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class PriorTriggerState:
    """Exact T-1 trigger/call/put state joined to a verified underlying."""

    stock_code: str
    trigger_price: float
    call_price: float
    put_price: float
    active: bool
    days_remaining: float


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_TRIGGER_TRAJECTORY_SIGNALS = (
    "itsr_trigger_distance_segment_convergence",
    "itsr_trigger_closest_distance_time",
    "itsr_trigger_tail_distance_reclaim",
)
_APPROACH_RESPONSE_SIGNALS = (
    "itsr_bond_samebin_trigger_approach_response",
    "itsr_bond_nextbin_trigger_approach_response",
    "itsr_bond_trigger_approach_latency_gap",
)
_CORRIDOR_SIGNALS = (
    "itsr_call_distance_segment_convergence",
    "itsr_put_distance_segment_convergence",
    "itsr_call_put_corridor_turn_rate",
)
_NEAR_COMPLETION_SIGNALS = (
    "itsr_nearcompletion_samebin_trigger_approach_response",
    "itsr_nearcompletion_nextbin_trigger_approach_response",
    "itsr_nearcompletion_call_distance_tail_reclaim",
)

_CATALOG = (
    _entries(
        "trigger_barrier_distance_trajectory",
        _TRIGGER_TRAJECTORY_SIGNALS,
        "A T-1 revised trigger defines a dynamic stock-distance path; convergence and timing are not a static trigger-distance level.",
    )
    + _entries(
        "mapped_bond_trigger_approach_response",
        _APPROACH_RESPONSE_SIGNALS,
        "Mapped bond returns are measured only around large same-day stock moves toward the T-1 trigger, including a strictly adjacent next-bin response.",
    )
    + _entries(
        "call_put_redemption_corridor_dynamics",
        _CORRIDOR_SIGNALS,
        "The T-1 call-put corridor conditions dynamic bond-distance convergence and turning, rather than returning a barrier position or level.",
    )
    + _entries(
        "near_completion_trigger_response",
        _NEAR_COMPLETION_SIGNALS,
        "Only active T-1 trigger states with at most three revised required days remaining are allowed to emit their dynamic event response.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "trigger_barrier_distance_trajectory": _TRIGGER_TRAJECTORY_SIGNALS,
    "mapped_bond_trigger_approach_response": _APPROACH_RESPONSE_SIGNALS,
    "call_put_redemption_corridor_dynamics": _CORRIDOR_SIGNALS,
    "near_completion_trigger_response": _NEAR_COMPLETION_SIGNALS,
}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

FORMULAS = {
    "itsr_trigger_distance_segment_convergence": "weighted mean over uninterrupted five-minute segments of |log(stock/trigger_revise_T-1)|_start-|...|_end.",
    "itsr_trigger_closest_distance_time": "continuous-session-normalized time of min(|log(stock/trigger_revise_T-1)|).",
    "itsr_trigger_tail_distance_reclaim": "max(|log(stock/trigger_revise_T-1)|,14:00-14:29)-terminal absolute distance.",
    "itsr_bond_samebin_trigger_approach_response": "mean(bond_log_return_t | stock distance to trigger decreases by its upper-quartile positive amount on contiguous t).",
    "itsr_bond_nextbin_trigger_approach_response": "mean(bond_log_return_t+1 | trigger-approach event at t and exactly adjacent t+1).",
    "itsr_bond_trigger_approach_latency_gap": "next-bin trigger-approach response minus same-bin trigger-approach response.",
    "itsr_call_distance_segment_convergence": "weighted mean of |log(bond/call_T-1)|_start-|...|_end over uninterrupted segments.",
    "itsr_put_distance_segment_convergence": "weighted mean of |log(bond/put_T-1)|_start-|...|_end over uninterrupted segments.",
    "itsr_call_put_corridor_turn_rate": "weighted fraction of sign changes in contiguous five-minute changes of (bond-put)/(call-put).",
    "itsr_nearcompletion_samebin_trigger_approach_response": "same-bin trigger-approach bond response, emitted only when T-1 state is active with 0<=revised days remaining<=3.",
    "itsr_nearcompletion_nextbin_trigger_approach_response": "next-bin trigger-approach bond response, emitted only in the strict T-1 near-completion state.",
    "itsr_nearcompletion_call_distance_tail_reclaim": "14:00-14:29 call-distance reclaim, emitted only in the strict T-1 near-completion state.",
}


def intraday_trigger_state_response_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return intraday_trigger_state_response_catalog()


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
        day = pd.Timestamp(raw)
        if pd.isna(day):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return day.normalize()
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
    """Keep only causal physical score-day ticks with a finite positive last."""

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
    """Use only unambiguous context mappings that are not explicitly future dated."""

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
    frame = frame.loc[~frame["bond_code"].duplicated(keep=False)]
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


def _prior_states(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, PriorTriggerState]:
    """Load strict, exact-T-1 trigger state; stale rows are never carried forward."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:states:{score_date.date().isoformat()}"
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
        fields=(
            "stock_code",
            "trigger_price_revise",
            "cb_call_price",
            "cb_put_price",
            "in_trigger_process",
            "trigger_cum_days_revise",
            "trigger_reach_days_revise",
        ),
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    states: dict[str, PriorTriggerState] = {}
    if not price.empty:
        anchor = pd.Timestamp(price["trade_date"].max()).normalize()
        anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
        snapshot = base.loc[(base["trade_date"] == anchor) & base["code"].isin(anchor_codes)].copy()
        numeric = (
            "trigger_price_revise",
            "cb_call_price",
            "cb_put_price",
            "in_trigger_process",
            "trigger_cum_days_revise",
            "trigger_reach_days_revise",
        )
        for column in numeric:
            snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
        snapshot["prior_stock_code"] = snapshot["stock_code"].map(_canonical_market_code)
        valid = (
            (snapshot["prior_stock_code"] != "")
            & np.isfinite(snapshot["trigger_price_revise"])
            & (snapshot["trigger_price_revise"] > _EPS)
            & np.isfinite(snapshot["cb_put_price"])
            & np.isfinite(snapshot["cb_call_price"])
            & (snapshot["cb_put_price"] > _EPS)
            & (snapshot["cb_call_price"] > snapshot["cb_put_price"] + _EPS)
            & np.isfinite(snapshot["in_trigger_process"])
            & np.isfinite(snapshot["trigger_cum_days_revise"])
            & np.isfinite(snapshot["trigger_reach_days_revise"])
            & (snapshot["trigger_reach_days_revise"] > _EPS)
        )
        snapshot = snapshot.loc[valid].copy()
        states = {
            str(row.code): PriorTriggerState(
                stock_code=str(row.prior_stock_code),
                trigger_price=float(row.trigger_price_revise),
                call_price=float(row.cb_call_price),
                put_price=float(row.cb_put_price),
                active=bool(float(row.in_trigger_process) > 0.0),
                days_remaining=float(row.trigger_reach_days_revise - row.trigger_cum_days_revise),
            )
            for row in snapshot.itertuples(index=False)
        }

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, dict):
            return existing
        ctx.cache[cache_key] = states
    return states


def _endpoint_prices(frame: pd.DataFrame) -> pd.Series:
    """Use only observed physical five-minute endpoints; do not interpolate gaps."""

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


def _trigger_path(
    bond_frame: pd.DataFrame,
    stock_frame: pd.DataFrame,
    *,
    state: PriorTriggerState,
) -> pd.DataFrame:
    bond = _endpoint_prices(bond_frame).rename("bond_last")
    stock = _endpoint_prices(stock_frame).rename("stock_last")
    path = bond.to_frame().join(stock, how="inner").dropna().sort_index()
    if len(path) < _MIN_PATH_POINTS:
        return pd.DataFrame()
    values = path[["bond_last", "stock_last"]].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= _EPS).any():
        return pd.DataFrame()
    path["trigger_distance"] = np.log(path["stock_last"].to_numpy(dtype="float64") / state.trigger_price)
    path["call_distance"] = np.log(path["bond_last"].to_numpy(dtype="float64") / state.call_price)
    path["put_distance"] = np.log(path["bond_last"].to_numpy(dtype="float64") / state.put_price)
    path["corridor_position"] = (
        path["bond_last"].to_numpy(dtype="float64") - state.put_price
    ) / (state.call_price - state.put_price)
    derived = path[["trigger_distance", "call_distance", "put_distance", "corridor_position"]].to_numpy(dtype="float64")
    if not np.isfinite(derived).all():
        return pd.DataFrame()
    contiguous = path.index.to_series().diff().eq(_EVENT_BIN).to_numpy()
    path["__contiguous"] = contiguous
    path["__segment"] = (~path["__contiguous"]).cumsum().astype("int64")
    return path


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights):
        return float("nan")
    values_array = np.asarray(values, dtype="float64")
    weights_array = np.asarray(weights, dtype="float64")
    valid = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0.0)
    return float(np.average(values_array[valid], weights=weights_array[valid])) if valid.any() else float("nan")


def _segment_distance_convergence(path: pd.DataFrame, column: str) -> float:
    values: list[float] = []
    weights: list[float] = []
    for _, segment in path.groupby("__segment", sort=False):
        if len(segment) < _MIN_SEGMENT_POINTS:
            continue
        distance = segment[column].to_numpy(dtype="float64")
        if not np.isfinite(distance).all():
            continue
        values.append(float(abs(distance[0]) - abs(distance[-1])))
        weights.append(float(len(segment) - 1))
    return _weighted_mean(values, weights)


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


def _trigger_trajectory_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TRIGGER_TRAJECTORY_SIGNALS}
    if path.empty or path.index[-1].time() < _TERMINAL_BIN:
        return out
    absolute = path["trigger_distance"].abs()
    clocks = pd.Series(path.index.time, index=path.index)
    tail = absolute.loc[clocks >= _TAIL_START]
    out["itsr_trigger_distance_segment_convergence"] = _segment_distance_convergence(path, "trigger_distance")
    out["itsr_trigger_closest_distance_time"] = _continuous_coordinate(pd.Timestamp(absolute.idxmin()))
    if len(tail) >= _MIN_PHASE_POINTS:
        out["itsr_trigger_tail_distance_reclaim"] = float(tail.max() - absolute.iloc[-1])
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _approach_edges(path: pd.DataFrame) -> pd.DataFrame:
    if path.empty:
        return pd.DataFrame(columns=["approach", "bond_return"])
    edges = pd.DataFrame(
        {
            "approach": path["trigger_distance"].abs().shift(1) - path["trigger_distance"].abs(),
            "bond_return": np.log(path["bond_last"]).diff(),
        },
        index=path.index,
    )
    return edges.loc[path["__contiguous"].to_numpy()].dropna().sort_index()


def _response_values(path: pd.DataFrame) -> tuple[float, float]:
    """Return immediate and next-bin response to large positive trigger approach."""

    edges = _approach_edges(path)
    positive = edges.loc[edges["approach"] > _EPS].copy()
    if len(positive) < _MIN_APPROACH_EVENTS:
        return float("nan"), float("nan")
    threshold = float(np.quantile(positive["approach"].to_numpy(dtype="float64"), 0.75))
    if not np.isfinite(threshold) or threshold <= _EPS:
        return float("nan"), float("nan")
    events = positive.loc[positive["approach"] >= threshold]
    if len(events) < _MIN_APPROACH_EVENTS:
        return float("nan"), float("nan")
    immediate = float(events["bond_return"].mean())
    next_returns = edges["bond_return"].reindex(pd.DatetimeIndex(events.index) + _EVENT_BIN).dropna()
    deferred = float(next_returns.mean()) if len(next_returns) >= _MIN_APPROACH_EVENTS else float("nan")
    return immediate, deferred


def _approach_response_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _APPROACH_RESPONSE_SIGNALS}
    immediate, deferred = _response_values(path)
    out["itsr_bond_samebin_trigger_approach_response"] = immediate
    out["itsr_bond_nextbin_trigger_approach_response"] = deferred
    if np.isfinite(immediate) and np.isfinite(deferred):
        out["itsr_bond_trigger_approach_latency_gap"] = float(deferred - immediate)
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _corridor_turn_rate(path: pd.DataFrame) -> float:
    values: list[float] = []
    weights: list[float] = []
    for _, segment in path.groupby("__segment", sort=False):
        if len(segment) < _MIN_SEGMENT_POINTS:
            continue
        changes = np.diff(segment["corridor_position"].to_numpy(dtype="float64"))
        signs = np.sign(changes[np.abs(changes) > _EPS])
        if len(signs) < 3:
            continue
        values.append(float(np.mean(signs[1:] != signs[:-1])))
        weights.append(float(len(signs) - 1))
    return _weighted_mean(values, weights)


def _corridor_metrics(path: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CORRIDOR_SIGNALS}
    if path.empty:
        return out
    out["itsr_call_distance_segment_convergence"] = _segment_distance_convergence(path, "call_distance")
    out["itsr_put_distance_segment_convergence"] = _segment_distance_convergence(path, "put_distance")
    out["itsr_call_put_corridor_turn_rate"] = _corridor_turn_rate(path)
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _near_completion_metrics(path: pd.DataFrame, state: PriorTriggerState) -> dict[str, float]:
    out = {signal: float("nan") for signal in _NEAR_COMPLETION_SIGNALS}
    near_completion = state.active and 0.0 <= state.days_remaining <= _NEAR_COMPLETION_DAYS
    if path.empty or not near_completion or path.index[-1].time() < _TERMINAL_BIN:
        return out
    immediate, deferred = _response_values(path)
    out["itsr_nearcompletion_samebin_trigger_approach_response"] = immediate
    out["itsr_nearcompletion_nextbin_trigger_approach_response"] = deferred
    clocks = pd.Series(path.index.time, index=path.index)
    tail = path.loc[clocks >= _TAIL_START, "call_distance"].abs()
    if len(tail) >= _MIN_PHASE_POINTS:
        out["itsr_nearcompletion_call_distance_tail_reclaim"] = float(tail.max() - tail.iloc[-1])
    return {key: value if np.isfinite(value) else float("nan") for key, value in out.items()}


def _metrics(path: pd.DataFrame, state: PriorTriggerState, family: str) -> dict[str, float]:
    if family == "trigger_barrier_distance_trajectory":
        return _trigger_trajectory_metrics(path)
    if family == "mapped_bond_trigger_approach_response":
        return _approach_response_metrics(path)
    if family == "call_put_redemption_corridor_dynamics":
        return _corridor_metrics(path)
    if family == "near_completion_trigger_response":
        return _near_completion_metrics(path, state)
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
        states = _prior_states(ctx, score_date=score_date)
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
            state = states.get(canonical_bond)
            mapped_stock = mapping.get(canonical_bond, "")
            if (
                state is not None
                and mapped_stock == state.stock_code
                and canonical_bond in bond_groups
                and mapped_stock in stock_groups
            ):
                path = _trigger_path(bond_groups[canonical_bond], stock_groups[mapped_stock], state=state)
                row.update(_metrics(path, state, family))
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
class FactorMiningIntradayTriggerStateResponseV1(Factor):
    """Research-only strict-PIT intraday response conditional on trigger state."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = True

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), 2),
            DailyFactorRequirement(
                "market_cbond.daily_base",
                (
                    "exchange_code",
                    "stock_code",
                    "trigger_price_revise",
                    "cb_call_price",
                    "cb_put_price",
                    "in_trigger_process",
                    "trigger_cum_days_revise",
                    "trigger_reach_days_revise",
                ),
                2,
            ),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)
