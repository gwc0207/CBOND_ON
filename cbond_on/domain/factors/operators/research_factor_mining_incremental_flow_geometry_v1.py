"""Research-only strict-PIT incremental intraday flow geometry factors.

The catalogue uses only adjacent physical snapshots from the score-day T1430
panel through 14:29. Amount, volume, and num_trades are treated as cumulative
counters and every activity measure is based on their non-negative increments.
Lunch is a hard session boundary. Invalid counters, timestamps, or required
core observations fail closed to NaN; there is no direct file, database,
label, pool, score, PnL, model, or live-system access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_incremental_flow_geometry_v1"
CATALOG_VERSION = "20260804_incremental_flow_geometry_v1"
_EPS = 1e-12
_PRICE_TOL = 1e-8
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_MAX_GAP_SECONDS = 20.0 * 60.0
_MIN_POINTS = 5
_MIN_INTERVALS = 12
_MIN_ACTIVE = 8
_TRADING_MINUTES = 209.0
_CORE_COLUMNS = ("trade_time", "last", "amount", "volume", "num_trades")
_L1_COLUMNS = ("ask_price1", "bid_price1", "ask_volume1", "bid_volume1")
_L5_COLUMNS = tuple(
    column
    for level in range(1, 6)
    for column in (
        f"ask_price{level}",
        f"bid_price{level}",
        f"ask_volume{level}",
        f"bid_volume{level}",
    )
)


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class _Path:
    session: np.ndarray
    clock: np.ndarray
    amount: np.ndarray
    volume: np.ndarray
    trades: np.ndarray
    amount_rate: np.ndarray
    volume_rate: np.ndarray
    trade_rate: np.ndarray
    ret: np.ndarray
    abs_ret: np.ndarray
    vwap_gap: np.ndarray
    imbalance1: np.ndarray
    imbalance5: np.ndarray
    imbalance1_change: np.ndarray
    depth1_change: np.ndarray
    depth5_change: np.ndarray
    depth_ratio: np.ndarray
    spread: np.ndarray
    spread_change: np.ndarray
    micro_gap: np.ndarray
    micro_gap_change: np.ndarray
    mid_ret: np.ndarray


_FAMILY_SPECS = (
    (
        "incremental_amount_clock_allocation",
        (
            "ifg_amount_increment_entropy",
            "ifg_amount_increment_hhi",
            "ifg_amount_increment_clock_center",
            "ifg_amount_increment_clock_dispersion",
            "ifg_amount_increment_afternoon_share",
        ),
        "Incremental notional can be concentrated, dispersed, early, late, or session-skewed without using a cumulative amount level.",
    ),
    (
        "incremental_volume_clock_allocation",
        (
            "ifg_volume_increment_entropy",
            "ifg_volume_increment_hhi",
            "ifg_volume_increment_clock_center",
            "ifg_volume_increment_clock_dispersion",
            "ifg_volume_increment_afternoon_share",
        ),
        "Executed share volume can have a temporal organisation that differs from notional when prices and ticket sizes vary.",
    ),
    (
        "incremental_trade_clock_allocation",
        (
            "ifg_trade_increment_entropy",
            "ifg_trade_increment_hhi",
            "ifg_trade_increment_clock_center",
            "ifg_trade_increment_clock_dispersion",
            "ifg_trade_increment_afternoon_share",
        ),
        "Trade-message activity can have a distinct intraday clock from volume and notional increments.",
    ),
    (
        "incremental_execution_size_state",
        (
            "ifg_trade_size_log_median",
            "ifg_trade_size_log_dispersion",
            "ifg_trade_size_upper_tail_share",
            "ifg_trade_size_afternoon_shift",
            "ifg_trade_size_clock_correlation",
        ),
        "Actual incremental ticket-size distribution and timing are not a relabelled total-flow statistic.",
    ),
    (
        "incremental_execution_price_state",
        (
            "ifg_interval_vwap_last_gap_abs",
            "ifg_interval_vwap_last_gap_signed",
            "ifg_interval_vwap_last_gap_dispersion",
            "ifg_interval_vwap_last_gap_amount_correlation",
            "ifg_interval_vwap_last_gap_afternoon_shift",
        ),
        "Incremental execution pricing compares the price paid within an interval with its endpoint mark rather than a cumulative VWAP level.",
    ),
    (
        "incremental_amount_return_coupling",
        (
            "ifg_amount_abs_return_correlation",
            "ifg_amount_signed_return_correlation",
            "ifg_amount_highflow_abs_return_excess",
            "ifg_amount_return_directional_asymmetry",
            "ifg_amount_impact_tail_mass_share",
        ),
        "Incremental notional can couple differently to discovery magnitude, direction, and tails.",
    ),
    (
        "incremental_volume_return_coupling",
        (
            "ifg_volume_abs_return_correlation",
            "ifg_volume_signed_return_correlation",
            "ifg_volume_highflow_abs_return_excess",
            "ifg_volume_return_directional_asymmetry",
            "ifg_volume_impact_tail_mass_share",
        ),
        "Executed share flow can expose a distinct price-discovery mechanism from notional flow.",
    ),
    (
        "incremental_trade_return_coupling",
        (
            "ifg_trade_abs_return_correlation",
            "ifg_trade_signed_return_correlation",
            "ifg_trade_highflow_abs_return_excess",
            "ifg_trade_return_directional_asymmetry",
            "ifg_trade_impact_tail_mass_share",
        ),
        "Ticket frequency can couple to return events separately from amount and volume increments.",
    ),
    (
        "incremental_directional_flow_persistence",
        (
            "ifg_signed_amount_autocorrelation",
            "ifg_signed_amount_reversal_rate",
            "ifg_signed_amount_run_concentration",
            "ifg_signed_amount_run_length_entropy",
            "ifg_signed_amount_terminal_alignment",
        ),
        "Sign-adjusted incremental flow has persistence and run topology beyond endpoint momentum.",
    ),
    (
        "incremental_amount_burst_topology",
        (
            "ifg_amount_burst_frequency",
            "ifg_amount_burst_cluster_concentration",
            "ifg_amount_burst_terminal_recency",
            "ifg_amount_burst_next_return_excess",
            "ifg_amount_burst_return_alignment",
        ),
        "Burst clustering and its local response use event topology rather than a parallel short-window flow statistic.",
    ),
    (
        "incremental_flow_book_alignment",
        (
            "ifg_amount_imbalance1_correlation",
            "ifg_amount_imbalance5_correlation",
            "ifg_amount_imbalance_direction_agreement",
            "ifg_amount_imbalance_change_correlation",
            "ifg_highflow_imbalance_dispersion",
        ),
        "Realised incremental activity can align with book pressure and its changes in ways a static imbalance cannot express.",
    ),
    (
        "incremental_depth_consumption_response",
        (
            "ifg_amount_depth1_change_correlation",
            "ifg_amount_depth5_change_correlation",
            "ifg_highflow_depth_contraction_share",
            "ifg_depth_layer_flow_correlation",
            "ifg_amount_depth_recovery_after_burst",
        ),
        "Depth consumption and recovery conditional on realised activity differs from raw depth or simple imbalance.",
    ),
    (
        "incremental_spread_resilience_flow",
        (
            "ifg_amount_spread_change_correlation",
            "ifg_highflow_spread_widening",
            "ifg_spread_recovery_after_burst",
            "ifg_spread_activity_dispersion",
            "ifg_afternoon_spread_flow_shift",
        ),
        "Spread resilience is defined by widening and recovery after actual activity, not a static-spread alias.",
    ),
    (
        "incremental_microprice_execution_convergence",
        (
            "ifg_amount_microprice_gap_correlation",
            "ifg_microprice_gap_closure_share",
            "ifg_highflow_microprice_alignment",
            "ifg_microprice_gap_return_correlation",
            "ifg_microprice_gap_afternoon_shift",
        ),
        "The path from book-implied microprice to realised incremental execution is distinct from a terminal microprice deviation.",
    ),
    (
        "incremental_quote_refresh_activity",
        (
            "ifg_amount_quote_refresh_correlation",
            "ifg_quote_refresh_trade_share",
            "ifg_quote_stasis_amount_share",
            "ifg_quote_refresh_return_excess",
            "ifg_quote_refresh_terminal_share",
        ),
        "Quote refresh conditional on actual execution measures update responsiveness rather than quote-change count alone.",
    ),
    (
        "incremental_cross_session_flow_handoff",
        (
            "ifg_session_amount_rate_log_ratio",
            "ifg_session_volume_rate_log_ratio",
            "ifg_session_trade_rate_log_ratio",
            "ifg_session_amount_impact_gap",
            "ifg_session_book_imbalance_gap",
        ),
        "The morning-to-afternoon handoff can reveal a separate execution regime without bridging lunch.",
    ),
)

_CATALOG = tuple(
    CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis)
    for family, signals, hypothesis in _FAMILY_SPECS
    for signal in signals
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {family: signals for family, signals, _hypothesis in _FAMILY_SPECS}
FORMULAS = {
    signal: f"{family}: member {position + 1} of the strict incremental-flow family"
    for family, signals, _hypothesis in _FAMILY_SPECS
    for position, signal in enumerate(signals)
}


def factor_mining_incremental_flow_geometry_catalog() -> tuple[CatalogEntry, ...]:
    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    return factor_mining_incremental_flow_geometry_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    family = ctx.params.get("family")
    if family is not None and str(family).strip() not in {"", entry.family}:
        raise ValueError(f"{KERNEL_NAME} signal/family mismatch for {signal}")
    return entry


def _score_date(panel: pd.DataFrame) -> pd.Timestamp | None:
    checked = ensure_panel_index(panel)
    raw = checked.attrs.get("__build_day__")
    if raw is not None:
        value = pd.Timestamp(raw)
        if pd.isna(value):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return value.normalize()
    if checked.empty:
        return None
    dates = pd.to_datetime(checked.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[dates.eq(score_date)].drop_duplicates().sort_values(["dt", "code"])
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _session(clock: dt_time) -> int | None:
    if _MORNING_START <= clock <= _MORNING_END:
        return 0
    if _AFTERNOON_START <= clock <= _CUTOFF:
        return 1
    return None


def _clock_position(timestamp: pd.Timestamp) -> float:
    clock = timestamp.time()
    seconds = clock.hour * 3600 + clock.minute * 60 + clock.second + clock.microsecond / 1_000_000
    if _MORNING_START <= clock <= _MORNING_END:
        return (seconds - (9 * 3600 + 30 * 60)) / 60.0
    if _AFTERNOON_START <= clock <= _CUTOFF:
        return 120.0 + (seconds - 13 * 3600) / 60.0
    return float("nan")


def _strict_frame(panel: pd.DataFrame, score_date: pd.Timestamp) -> pd.DataFrame | None:
    checked = ensure_panel_index(panel)
    if any(column not in checked.columns for column in _CORE_COLUMNS):
        return None
    frame = ensure_trade_time(checked).reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    sessions = clocks.map(lambda value: _session(value) if pd.notna(value) else None)
    keep = (
        indexed.notna()
        & indexed.dt.normalize().eq(score_date)
        & timestamps.notna()
        & timestamps.dt.normalize().eq(score_date)
        & sessions.notna()
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    out["__ifg_session"] = sessions.loc[keep].astype("int8")
    return out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")


def _num(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _nan_record() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _entropy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype="float64")
    positive = values[np.isfinite(values) & (values > _EPS)]
    if len(positive) <= 1:
        return float("nan")
    probability = positive / float(positive.sum())
    result = -float(np.dot(probability, np.log(probability))) / float(np.log(len(probability)))
    return result if np.isfinite(result) else float("nan")


def _corr(left: np.ndarray, right: np.ndarray, minimum: int = _MIN_ACTIVE) -> float:
    pairs = np.column_stack((left, right)).astype("float64", copy=False)
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if len(pairs) < minimum:
        return float("nan")
    lhs, rhs = pairs[:, 0], pairs[:, 1]
    lhs_std, rhs_std = float(lhs.std(ddof=0)), float(rhs.std(ddof=0))
    if lhs_std <= _EPS or rhs_std <= _EPS:
        return float("nan")
    result = float(np.mean((lhs - lhs.mean()) * (rhs - rhs.mean())) / (lhs_std * rhs_std))
    return result if np.isfinite(result) else float("nan")


def _wmean(values: np.ndarray, weights: np.ndarray, minimum: int = _MIN_ACTIVE) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if int(mask.sum()) < minimum:
        return float("nan")
    local_values, local_weights = values[mask], weights[mask]
    total = float(local_weights.sum())
    if total <= _EPS:
        return float("nan")
    result = float(np.dot(local_values, local_weights) / total)
    return result if np.isfinite(result) else float("nan")


def _wstd(values: np.ndarray, weights: np.ndarray, minimum: int = _MIN_ACTIVE) -> float:
    mean = _wmean(values, weights, minimum)
    if not np.isfinite(mean):
        return float("nan")
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    local_values, local_weights = values[mask], weights[mask]
    total = float(local_weights.sum())
    if total <= _EPS:
        return float("nan")
    result = float(np.sqrt(np.dot(np.square(local_values - mean), local_weights) / total))
    return result if np.isfinite(result) else float("nan")


def _true_runs(values: np.ndarray, sessions: np.ndarray) -> np.ndarray:
    lengths: list[int] = []
    for session in (0, 1):
        local = values[sessions == session]
        run = 0
        for value in local:
            if bool(value):
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return np.asarray(lengths, dtype="float64")


def _next_same_session(path: _Path, values: np.ndarray, event: np.ndarray) -> np.ndarray:
    output: list[float] = []
    for index in np.flatnonzero(event):
        next_index = int(index) + 1
        if next_index < len(values) and path.session[next_index] == path.session[index]:
            value = float(values[next_index])
            if np.isfinite(value):
                output.append(value)
    return np.asarray(output, dtype="float64")


def _book(sampled: pd.DataFrame) -> dict[str, np.ndarray]:
    size = len(sampled)
    nan = np.full(size, np.nan, dtype="float64")
    out = {
        "imbalance1": nan.copy(),
        "imbalance5": nan.copy(),
        "depth1": nan.copy(),
        "depth5": nan.copy(),
        "spread": nan.copy(),
        "micro_gap": nan.copy(),
        "mid": nan.copy(),
    }
    if any(column not in sampled.columns for column in _L1_COLUMNS):
        return out
    ask, bid = _num(sampled, "ask_price1"), _num(sampled, "bid_price1")
    askv, bidv = _num(sampled, "ask_volume1"), _num(sampled, "bid_volume1")
    depth, mid = askv + bidv, (ask + bid) / 2.0
    valid = (
        np.isfinite(ask)
        & np.isfinite(bid)
        & np.isfinite(askv)
        & np.isfinite(bidv)
        & (ask > 0.0)
        & (bid > 0.0)
        & (ask >= bid)
        & (askv >= 0.0)
        & (bidv >= 0.0)
        & (depth > _EPS)
        & (mid > _EPS)
    )
    out["depth1"][valid] = depth[valid]
    out["mid"][valid] = mid[valid]
    out["imbalance1"][valid] = (bidv[valid] - askv[valid]) / depth[valid]
    out["spread"][valid] = (ask[valid] - bid[valid]) / mid[valid]
    micro = (bid * askv + ask * bidv) / depth
    out["micro_gap"][valid] = (micro[valid] - mid[valid]) / mid[valid]
    if any(column not in sampled.columns for column in _L5_COLUMNS):
        return out
    ask_depth, bid_depth = np.zeros(size), np.zeros(size)
    valid5 = np.ones(size, dtype=bool)
    for level in range(1, 6):
        ask_price, bid_price = _num(sampled, f"ask_price{level}"), _num(sampled, f"bid_price{level}")
        ask_volume, bid_volume = _num(sampled, f"ask_volume{level}"), _num(sampled, f"bid_volume{level}")
        valid5 &= (
            np.isfinite(ask_price)
            & np.isfinite(bid_price)
            & np.isfinite(ask_volume)
            & np.isfinite(bid_volume)
            & (ask_price > 0.0)
            & (bid_price > 0.0)
            & (ask_price >= bid_price)
            & (ask_volume >= 0.0)
            & (bid_volume >= 0.0)
        )
        ask_depth += ask_volume
        bid_depth += bid_volume
    total = ask_depth + bid_depth
    valid5 &= total > _EPS
    out["depth5"][valid5] = total[valid5]
    out["imbalance5"][valid5] = (bid_depth[valid5] - ask_depth[valid5]) / total[valid5]
    return out


def _change(values: np.ndarray, log: bool = False) -> np.ndarray:
    result = np.full(len(values) - 1, np.nan, dtype="float64")
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    if log:
        valid &= (values[1:] > _EPS) & (values[:-1] > _EPS)
        result[valid] = np.log(values[1:][valid] / values[:-1][valid])
    else:
        result[valid] = values[1:][valid] - values[:-1][valid]
    return result


def _path(group: pd.DataFrame) -> _Path | None:
    if len(group) < 2 * _MIN_POINTS:
        return None
    for column in ("amount", "volume", "num_trades"):
        values = _num(group, column)
        if not np.isfinite(values).all() or (values < 0.0).any() or (np.diff(values) < 0.0).any():
            return None
    frame = group.copy(deep=False)
    times = pd.to_datetime(frame["trade_time"], errors="coerce")
    if times.isna().any():
        return None
    bins: list[int] = []
    for value in times:
        session = _session(value.time())
        if session is None:
            return None
        minute = value.hour * 60 + value.minute
        bins.append((minute - (9 * 60 + 30)) // 5 if session == 0 else (minute - 13 * 60) // 5)
    frame = frame.assign(__ifg_bin=bins)
    sampled = (
        frame.groupby(["__ifg_session", "__ifg_bin"], sort=True, as_index=False)
        .tail(1)
        .sort_values(["__ifg_session", "trade_time", "seq"], kind="mergesort")
    )
    chunks: dict[str, list[np.ndarray]] = {field: [] for field in _Path.__dataclass_fields__}
    for session in (0, 1):
        local = sampled.loc[sampled["__ifg_session"].eq(session)].copy()
        if len(local) < _MIN_POINTS:
            return None
        local_times = pd.to_datetime(local["trade_time"], errors="coerce")
        gap = local_times.diff().dropna().dt.total_seconds().to_numpy(dtype="float64")
        if local_times.isna().any() or local_times.duplicated().any() or not np.isfinite(gap).all() or (gap <= 0.0).any() or (gap > _MAX_GAP_SECONDS).any():
            return None
        last, amount, volume, trades = (_num(local, name) for name in ("last", "amount", "volume", "num_trades"))
        if not np.isfinite(last).all() or (last <= 0.0).any():
            return None
        da, dv, dn = np.diff(amount), np.diff(volume), np.diff(trades)
        if (da < 0.0).any() or (dv < 0.0).any() or (dn < 0.0).any():
            return None
        book = _book(local)
        interval_vwap = np.full(len(da), np.nan)
        vwap_valid = dv > _EPS
        interval_vwap[vwap_valid] = da[vwap_valid] / dv[vwap_valid]
        vwap_gap = np.full(len(da), np.nan)
        vwap_valid &= np.isfinite(interval_vwap) & (interval_vwap > _EPS)
        vwap_gap[vwap_valid] = (interval_vwap[vwap_valid] - last[1:][vwap_valid]) / last[1:][vwap_valid]
        ratio = np.full(len(da), np.nan)
        ratio_valid = np.isfinite(book["depth1"][1:]) & np.isfinite(book["depth5"][1:]) & (book["depth1"][1:] > _EPS) & (book["depth5"][1:] > _EPS)
        ratio[ratio_valid] = book["depth1"][1:][ratio_valid] / book["depth5"][1:][ratio_valid]
        chunks["session"].append(np.full(len(da), session, dtype="int8"))
        chunks["clock"].append(np.asarray([_clock_position(value) for value in local_times.iloc[1:]], dtype="float64"))
        chunks["amount"].append(da)
        chunks["volume"].append(dv)
        chunks["trades"].append(dn)
        chunks["amount_rate"].append(da / gap)
        chunks["volume_rate"].append(dv / gap)
        chunks["trade_rate"].append(dn / gap)
        chunks["ret"].append(np.log(last[1:] / last[:-1]))
        chunks["abs_ret"].append(np.abs(np.log(last[1:] / last[:-1])))
        chunks["vwap_gap"].append(vwap_gap)
        chunks["imbalance1"].append(book["imbalance1"][1:])
        chunks["imbalance5"].append(book["imbalance5"][1:])
        chunks["imbalance1_change"].append(_change(book["imbalance1"]))
        chunks["depth1_change"].append(_change(book["depth1"], log=True))
        chunks["depth5_change"].append(_change(book["depth5"], log=True))
        chunks["depth_ratio"].append(ratio)
        chunks["spread"].append(book["spread"][1:])
        chunks["spread_change"].append(_change(book["spread"]))
        chunks["micro_gap"].append(book["micro_gap"][1:])
        chunks["micro_gap_change"].append(_change(book["micro_gap"]))
        chunks["mid_ret"].append(_change(book["mid"], log=True))
    if not chunks["amount"]:
        return None
    joined = {key: np.concatenate(values) for key, values in chunks.items()}
    active = (joined["amount"] > _EPS) & (joined["volume"] > _EPS) & (joined["trades"] > _EPS)
    if len(joined["amount"]) < _MIN_INTERVALS or int(active.sum()) < _MIN_ACTIVE:
        return None
    return _Path(**joined)


def _clock_metrics(path: _Path, mass: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    out = {name: float("nan") for name in names}
    valid = np.isfinite(mass) & (mass > _EPS) & np.isfinite(path.clock)
    if int(valid.sum()) < _MIN_ACTIVE:
        return out
    weights, clock, session = mass[valid], path.clock[valid], path.session[valid]
    if int((session == 0).sum()) < 3 or int((session == 1).sum()) < 3:
        return out
    total = float(weights.sum())
    probability = weights / total
    center = float(np.dot(clock, probability))
    out.update(
        {
            names[0]: _entropy(weights),
            names[1]: float(np.dot(probability, probability)),
            names[2]: center / _TRADING_MINUTES,
            names[3]: float(np.sqrt(np.dot(np.square(clock - center), probability)) / _TRADING_MINUTES),
            names[4]: float(weights[session == 1].sum() / total),
        }
    )
    return out


def _mass_return(path: _Path, mass: np.ndarray, names: tuple[str, ...]) -> dict[str, float]:
    out = {name: float("nan") for name in names}
    valid = np.isfinite(mass) & (mass > _EPS) & np.isfinite(path.ret) & np.isfinite(path.abs_ret)
    if int(valid.sum()) < _MIN_ACTIVE:
        return out
    local, ret, absolute = mass[valid], path.ret[valid], path.abs_ret[valid]
    high, low = local >= np.quantile(local, 0.75), local <= np.quantile(local, 0.25)
    tail = absolute >= np.quantile(absolute, 0.75)
    if int(high.sum()) < 3 or int(low.sum()) < 3 or int(tail.sum()) < 3:
        return out
    total = float(local.sum())
    out.update(
        {
            names[0]: _corr(np.log1p(local), absolute),
            names[1]: _corr(np.log1p(local), ret),
            names[2]: float(absolute[high].mean() - absolute[low].mean()),
            names[3]: float((local[ret > _EPS].sum() - local[ret < -_EPS].sum()) / total),
            names[4]: float(local[tail].sum() / total),
        }
    )
    return out


def _burst(path: _Path) -> np.ndarray | None:
    valid = np.isfinite(path.amount_rate) & (path.amount_rate > _EPS)
    if int(valid.sum()) < _MIN_ACTIVE:
        return None
    return valid & (path.amount_rate >= np.quantile(path.amount_rate[valid], 0.80))


def _metrics(path: _Path) -> dict[str, float]:
    out = _nan_record()
    out.update(_clock_metrics(path, path.amount, _FAMILY_SIGNALS["incremental_amount_clock_allocation"]))
    out.update(_clock_metrics(path, path.volume, _FAMILY_SIGNALS["incremental_volume_clock_allocation"]))
    out.update(_clock_metrics(path, path.trades, _FAMILY_SIGNALS["incremental_trade_clock_allocation"]))
    out.update(_mass_return(path, path.amount, _FAMILY_SIGNALS["incremental_amount_return_coupling"]))
    out.update(_mass_return(path, path.volume, _FAMILY_SIGNALS["incremental_volume_return_coupling"]))
    out.update(_mass_return(path, path.trades, _FAMILY_SIGNALS["incremental_trade_return_coupling"]))
    size_names = _FAMILY_SIGNALS["incremental_execution_size_state"]
    active_size = (path.amount > _EPS) & (path.trades > _EPS) & np.isfinite(path.clock)
    if int(active_size.sum()) >= _MIN_ACTIVE:
        values, amount, session = np.log(path.amount[active_size] / path.trades[active_size]), path.amount[active_size], path.session[active_size]
        high = values >= np.quantile(values, 0.75)
        morning, afternoon = values[session == 0], values[session == 1]
        if int(high.sum()) >= 3 and len(morning) >= 4 and len(afternoon) >= 4:
            out.update(
                {
                    size_names[0]: float(np.median(values)),
                    size_names[1]: float(values.std(ddof=0)),
                    size_names[2]: float(amount[high].sum() / amount.sum()),
                    size_names[3]: float(np.median(afternoon) - np.median(morning)),
                    size_names[4]: _corr(path.clock[active_size], values),
                }
            )
    vwap_names = _FAMILY_SIGNALS["incremental_execution_price_state"]
    valid_vwap = np.isfinite(path.vwap_gap) & (path.amount > _EPS) & np.isfinite(path.amount_rate)
    if int(valid_vwap.sum()) >= _MIN_ACTIVE:
        gap, amount, session = path.vwap_gap[valid_vwap], path.amount[valid_vwap], path.session[valid_vwap]
        morning, afternoon = gap[session == 0], gap[session == 1]
        if len(morning) >= 4 and len(afternoon) >= 4:
            out.update(
                {
                    vwap_names[0]: _wmean(np.abs(gap), amount),
                    vwap_names[1]: _wmean(gap, amount),
                    vwap_names[2]: float(gap.std(ddof=0)),
                    vwap_names[3]: _corr(np.log1p(path.amount_rate[valid_vwap]), gap),
                    vwap_names[4]: float(afternoon.mean() - morning.mean()),
                }
            )
    persistence_names = _FAMILY_SIGNALS["incremental_directional_flow_persistence"]
    directional = (path.amount > _EPS) & np.isfinite(path.ret) & (np.abs(path.ret) > _EPS)
    if int(directional.sum()) >= _MIN_ACTIVE:
        values = path.amount[directional] * np.sign(path.ret[directional])
        signs, sessions = np.sign(values).astype("int8"), path.session[directional]
        autocorrelation: list[float] = []
        transitions, reversals, lengths = 0, 0, []
        for session in (0, 1):
            local_values, local_signs = values[sessions == session], signs[sessions == session]
            if len(local_values) >= 3:
                value = _corr(local_values[:-1], local_values[1:], minimum=3)
                if np.isfinite(value):
                    autocorrelation.append(value)
            if len(local_signs) >= 2:
                transitions += len(local_signs) - 1
                reversals += int(np.sum(local_signs[1:] != local_signs[:-1]))
                starts = np.r_[True, local_signs[1:] != local_signs[:-1]]
                lengths.extend(np.diff(np.r_[np.flatnonzero(starts), len(local_signs)]).tolist())
        terminal = values[-3:]
        if autocorrelation and transitions and len(lengths) >= 3 and len(terminal) == 3 and float(np.abs(terminal).sum()) > _EPS:
            run = np.asarray(lengths, dtype="float64")
            weight = run / run.sum()
            counts = np.asarray(list(Counter(run.astype("int64")).values()), dtype="float64")
            out.update(
                {
                    persistence_names[0]: float(np.mean(autocorrelation)),
                    persistence_names[1]: float(reversals / transitions),
                    persistence_names[2]: float(np.dot(weight, weight)),
                    persistence_names[3]: _entropy(counts),
                    persistence_names[4]: float(terminal.sum() / np.abs(terminal).sum()),
                }
            )
    burst_names = _FAMILY_SIGNALS["incremental_amount_burst_topology"]
    burst = _burst(path)
    valid = (path.amount > _EPS) & np.isfinite(path.ret) & np.isfinite(path.abs_ret)
    if burst is not None and int((burst & valid).sum()) >= 3:
        runs = _true_runs(burst, path.session)
        next_return = _next_same_session(path, path.abs_ret, burst & valid)
        normal = valid & ~burst
        high = burst & valid
        if len(runs) and len(next_return) >= 3 and int(normal.sum()) >= 3:
            weight = runs / runs.sum()
            amount = path.amount[high]
            out.update(
                {
                    burst_names[0]: float(np.mean(burst[valid])),
                    burst_names[1]: float(np.dot(weight, weight)),
                    burst_names[2]: float(path.clock[np.flatnonzero(high)[-1]] / _TRADING_MINUTES),
                    burst_names[3]: float(next_return.mean() - path.abs_ret[normal].mean()),
                    burst_names[4]: float(np.dot(amount, np.sign(path.ret[high])) / amount.sum()),
                }
            )
    book_names = _FAMILY_SIGNALS["incremental_flow_book_alignment"]
    activity = np.log1p(path.amount_rate)
    book = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.imbalance1) & np.isfinite(path.imbalance5)
    book_change = book & np.isfinite(path.imbalance1_change)
    if int(book.sum()) >= _MIN_ACTIVE and int(book_change.sum()) >= _MIN_ACTIVE:
        high = book & (path.amount_rate >= np.quantile(path.amount_rate[book], 0.75))
        directional_book = book & np.isfinite(path.ret) & (np.abs(path.ret) > _EPS)
        if int(high.sum()) >= 3 and int(directional_book.sum()) >= _MIN_ACTIVE:
            weight = path.amount[directional_book]
            agreement = np.sign(path.ret[directional_book]) * np.sign(path.imbalance1[directional_book])
            out.update(
                {
                    book_names[0]: _corr(activity[book], path.imbalance1[book]),
                    book_names[1]: _corr(activity[book], path.imbalance5[book]),
                    book_names[2]: float(np.dot(weight, agreement) / weight.sum()),
                    book_names[3]: _corr(activity[book_change], path.imbalance1_change[book_change]),
                    book_names[4]: float(path.imbalance1[high].std(ddof=0)),
                }
            )
    depth_names = _FAMILY_SIGNALS["incremental_depth_consumption_response"]
    depth1 = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.depth1_change)
    depth5 = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.depth5_change)
    layer = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.depth_ratio)
    if burst is not None and int(depth1.sum()) >= _MIN_ACTIVE and int(depth5.sum()) >= _MIN_ACTIVE and int(layer.sum()) >= _MIN_ACTIVE:
        high = depth1 & (path.amount_rate >= np.quantile(path.amount_rate[depth1], 0.75))
        next_depth = _next_same_session(path, path.depth1_change, burst & depth1)
        if int(high.sum()) >= 3 and len(next_depth) >= 3:
            out.update(
                {
                    depth_names[0]: _corr(activity[depth1], path.depth1_change[depth1]),
                    depth_names[1]: _corr(activity[depth5], path.depth5_change[depth5]),
                    depth_names[2]: float(np.mean(path.depth1_change[high] < 0.0)),
                    depth_names[3]: _corr(activity[layer], path.depth_ratio[layer]),
                    depth_names[4]: float(next_depth.mean()),
                }
            )
    spread_names = _FAMILY_SIGNALS["incremental_spread_resilience_flow"]
    spread = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.spread) & np.isfinite(path.spread_change)
    if burst is not None and int(spread.sum()) >= _MIN_ACTIVE:
        high = spread & (path.amount_rate >= np.quantile(path.amount_rate[spread], 0.75))
        next_spread = _next_same_session(path, path.spread_change, burst & spread)
        morning, afternoon = spread & (path.session == 0), spread & (path.session == 1)
        if int(high.sum()) >= 3 and len(next_spread) >= 3 and int(morning.sum()) >= 4 and int(afternoon.sum()) >= 4:
            out.update(
                {
                    spread_names[0]: _corr(activity[spread], path.spread_change[spread]),
                    spread_names[1]: float(path.spread_change[high].mean()),
                    spread_names[2]: float(-next_spread.mean()),
                    spread_names[3]: float(path.spread[high].std(ddof=0)),
                    spread_names[4]: float(_wmean(path.spread[afternoon], path.amount[afternoon], 4) - _wmean(path.spread[morning], path.amount[morning], 4)),
                }
            )
    micro_names = _FAMILY_SIGNALS["incremental_microprice_execution_convergence"]
    micro = np.isfinite(activity) & (path.amount > _EPS) & np.isfinite(path.micro_gap) & np.isfinite(path.micro_gap_change)
    directional_micro = micro & np.isfinite(path.ret) & (np.abs(path.ret) > _EPS)
    if int(micro.sum()) >= _MIN_ACTIVE and int(directional_micro.sum()) >= _MIN_ACTIVE:
        high = directional_micro & (path.amount_rate >= np.quantile(path.amount_rate[micro], 0.75))
        morning, afternoon = micro & (path.session == 0), micro & (path.session == 1)
        if int(high.sum()) >= 3 and int(morning.sum()) >= 4 and int(afternoon.sum()) >= 4:
            weight = path.amount[high]
            alignment = np.sign(path.ret[high]) * np.sign(path.micro_gap[high])
            out.update(
                {
                    micro_names[0]: _corr(activity[micro], path.micro_gap[micro]),
                    micro_names[1]: float(np.mean(np.abs(path.micro_gap_change[micro]) < np.abs(path.micro_gap[micro]))),
                    micro_names[2]: float(np.dot(weight, alignment) / weight.sum()),
                    micro_names[3]: _corr(path.micro_gap[directional_micro], path.ret[directional_micro]),
                    micro_names[4]: float(_wmean(path.micro_gap[afternoon], path.amount[afternoon], 4) - _wmean(path.micro_gap[morning], path.amount[morning], 4)),
                }
            )
    quote_names = _FAMILY_SIGNALS["incremental_quote_refresh_activity"]
    quote = np.isfinite(activity) & (path.amount > _EPS) & (path.trades > _EPS) & np.isfinite(path.mid_ret) & np.isfinite(path.spread_change) & np.isfinite(path.imbalance1_change)
    if int(quote.sum()) >= _MIN_ACTIVE:
        refresh = (np.abs(path.mid_ret[quote]) > _PRICE_TOL) | (np.abs(path.spread_change[quote]) > _EPS) | (np.abs(path.imbalance1_change[quote]) > _PRICE_TOL)
        amount, trades, absolute, clock = path.amount[quote], path.trades[quote], path.abs_ret[quote], path.clock[quote]
        terminal = clock >= 0.75 * _TRADING_MINUTES
        if int(refresh.sum()) >= 3 and int((~refresh).sum()) >= 3 and int(terminal.sum()) >= 3:
            out.update(
                {
                    quote_names[0]: _corr(activity[quote], refresh.astype("float64")),
                    quote_names[1]: float(trades[refresh].sum() / trades.sum()),
                    quote_names[2]: float(amount[~refresh].sum() / amount.sum()),
                    quote_names[3]: float(absolute[refresh].mean() - absolute[~refresh].mean()),
                    quote_names[4]: _wmean(refresh[terminal].astype("float64"), amount[terminal], 3),
                }
            )
    handoff_names = _FAMILY_SIGNALS["incremental_cross_session_flow_handoff"]
    morning, afternoon = path.session == 0, path.session == 1
    def session_rate(values: np.ndarray, mask: np.ndarray) -> float:
        local = values[mask]
        local = local[np.isfinite(local) & (local > _EPS)]
        return float(local.mean()) if len(local) >= 4 else float("nan")
    am, aa = session_rate(path.amount_rate, morning), session_rate(path.amount_rate, afternoon)
    vm, va = session_rate(path.volume_rate, morning), session_rate(path.volume_rate, afternoon)
    tm, ta = session_rate(path.trade_rate, morning), session_rate(path.trade_rate, afternoon)
    im, ia = _wmean(path.abs_ret[morning], path.amount[morning], 4), _wmean(path.abs_ret[afternoon], path.amount[afternoon], 4)
    bm, ba = _wmean(path.imbalance1[morning], path.amount[morning], 4), _wmean(path.imbalance1[afternoon], path.amount[afternoon], 4)
    values = np.asarray((am, aa, vm, va, tm, ta, im, ia, bm, ba), dtype="float64")
    if np.isfinite(values).all() and min(am, vm, tm) > _EPS:
        out.update(
            {
                handoff_names[0]: float(np.log(aa / am)),
                handoff_names[1]: float(np.log(va / vm)),
                handoff_names[2]: float(np.log(ta / tm)),
                handoff_names[3]: float(ia - im),
                handoff_names[4]: float(ba - bm),
            }
        )
    return {name: float(value) if np.isfinite(value) else float("nan") for name, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    score_date = _score_date(ctx.panel)
    output_index = _output_index(ctx, score_date)
    strict = None if score_date is None else _strict_frame(ctx.panel, score_date)
    if score_date is None or output_index.empty or strict is None or strict.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        for (dt, code), group in strict.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": code}
            path = _path(group)
            row.update(_metrics(path) if path is not None else _nan_record())
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan).reindex(output_index)
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIncrementalFlowGeometryV1(Factor):
    """Research-only incremental execution and book-response candidates."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = False
    requires_bond_stock_map = False

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        output = _feature_frame(ctx)[entry.signal].copy()
        output.name = self.output_name(entry.signal)
        return output


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningIncrementalFlowGeometryV1",
    "FORMULAS",
    "factor_mining_catalog",
    "factor_mining_incremental_flow_geometry_catalog",
]
