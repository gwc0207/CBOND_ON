"""Research-only strict-PIT intraday × prior-state interaction catalogue.

This module deliberately creates information *interactions*, rather than
renaming the existing intraday, daily, hybrid, or cross-asset primitives.  It
uses only the FactorComputeContext supplied by the regular factor builder:

* current-day bond (and, for one family, mapped-stock) T1430 observations are
  physically constrained to the score date and to 14:29:00 or earlier;
* all daily state is strictly before the score date, with ``daily_price`` as a
  common completed-session anchor; and
* a stale ``daily_base`` row, score-day mapping, missing field, counter reset,
  or malformed book produces an explicit ``NaN`` for the affected candidate.

There is no direct file, database, Redis, pool, FactorStore, label, score, or
backtest access in this factor implementation.  It remains import-only and
research-only: it is intentionally not added to ``defs.__init__``, factor
contracts, live configuration, or a production factor pack.
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
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_state_interaction_v1"
CATALOG_VERSION = "20260803_intraday_state_interaction_v1"

_LOOKBACK_DAYS = 66
_EPS = 1e-12
_MIN_PATH_ROWS = 8
_AMOUNT_CORRECTION_ABS = 100.0
_AMOUNT_CORRECTION_RATIO = 1e-5

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

_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "pre_close",
    "open",
    "last",
    "volume",
    "amount",
    "num_trades",
    "ask_price1",
    "ask_volume1",
    "bid_price1",
    "bid_volume1",
)

_PRICE_FIELDS = (
    "close_price",
    "prev_close_price",
    "act_prev_close_price",
    "amount",
)
_BASE_FIELDS = (
    "pure_redemption_value",
    "redemption_prem_ratio",
    "convexity",
    "duration",
    "year_to_mat",
    "bond_prem_ratio",
    "ytm",
    "remain_size",
    "cb_amount",
    "in_trigger_process",
    "trigger_cum_days_revise",
    "trigger_reach_days_revise",
    "cb_conv_price",
    "cb_put_price",
    "cb_call_price",
    "stock_close_price",
    "stk_prev_close_price",
    "stk_act_prev_close_price",
    "turnover_rate",
    "stock_code",
    "stk_amount",
    "stk_deal",
    "cb_deal",
    "stock_volatility",
    "trigger_price_revise",
)
_TWAP_FIELDS = (
    "twap_0930_1000",
    "twap_1400_1430",
    "twap_1442_1457",
)


@dataclass(frozen=True)
class CatalogEntry:
    """One immutable research candidate and its economic family."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_DEFENSIVE_PATH_SIGNALS = (
    "isi_def_tail_eff_x_floor_change5",
    "isi_def_tail_jump_x_redemption_prem_z20",
    "isi_def_gapabsorb_x_convexity_duration",
    "isi_def_terminal_loc_x_maturity_duration_gap",
    "isi_def_noise_x_premium_yield_innovation",
    "isi_def_reclaim_x_floor_vol20",
)
_SUPPLY_RESPONSE_SIGNALS = (
    "isi_supply_flowaccel_x_shrink1",
    "isi_supply_tail_impact_x_recency60",
    "isi_supply_depthrec_x_eventrate20",
    "isi_supply_gapabsorb_x_cumchange20",
    "isi_supply_midlast_x_shrink_liq_impulse",
    "isi_supply_lullrelease_x_change1",
)
_CALL_BOOK_SIGNALS = (
    "isi_call_imbalance_x_active_age",
    "isi_call_spread_shift_x_progress_velocity",
    "isi_call_depthrec_x_required_days",
    "isi_call_midlead_x_barrier_asym",
    "isi_call_gapabsorb_x_contract_recency",
    "isi_call_midlead_x_active_age",
)
_ADJUSTMENT_REPRICING_SIGNALS = (
    "isi_adj_open_gap_x_net_shift",
    "isi_adj_gapabsorb_x_cross_gap",
    "isi_adj_midlast_x_absimbalance",
    "isi_adj_tailflow_x_magnitude",
    "isi_adj_rotation_x_recency",
    "isi_adj_quotephase_x_eventrate",
)
_LIQUIDITY_EXECUTION_SIGNALS = (
    "isi_liq_tailvwap_x_lagflowbeta",
    "isi_liq_flowlead_x_highflow_response",
    "isi_liq_depthimpact_x_impactmemory",
    "isi_liq_gapresid_x_amount_persistence",
    "isi_liq_eventimpact_x_amihud_z",
    "isi_liq_lull_x_twap_transition",
)
_CROSS_REGIME_SIGNALS = (
    "isi_cross_phase_rotation_x_beta_shift",
    "isi_cross_tailcojump_x_residvol",
    "isi_cross_bookcoherence_x_volforecast",
    "isi_cross_stockshockbook_x_trackingresid",
    "isi_cross_tailrange_x_liqsharediv",
    "isi_cross_quotechannel_x_barrierdist",
)

_CATALOG = (
    _entries(
        "isi_defensive_path_state",
        _DEFENSIVE_PATH_SIGNALS,
        "Current path quality is gated by T-1 defensive-value changes and risk geometry, not current flow times a static floor or duration.",
    )
    + _entries(
        "isi_supply_event_response",
        _SUPPLY_RESPONSE_SIGNALS,
        "Current flow and book response is conditioned on completed outstanding-supply transitions, recency, and event rate rather than static remaining size.",
    )
    + _entries(
        "isi_call_contract_book_gate",
        _CALL_BOOK_SIGNALS,
        "Current L1 book and path mechanics are gated by the prior completed call lifecycle and contract-revision state, not return times call state.",
    )
    + _entries(
        "isi_adjustment_discontinuity_repricing",
        _ADJUSTMENT_REPRICING_SIGNALS,
        "Current opening, tail, and quote repricing is conditioned on strictly prior cross-asset adjusted-close discontinuities.",
    )
    + _entries(
        "isi_liquidity_execution_memory",
        _LIQUIDITY_EXECUTION_SIGNALS,
        "Current execution style is gated by completed daily liquidity and TWAP-transition memory, avoiding existing current-flow or range hybrids.",
    )
    + _entries(
        "isi_cross_asset_daily_regime_gate",
        _CROSS_REGIME_SIGNALS,
        "Current mapped stock/bond joint state is gated by strict T-1 tracking, liquidity, volatility, and barrier regimes rather than prior premium level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "isi_defensive_path_state": _DEFENSIVE_PATH_SIGNALS,
    "isi_supply_event_response": _SUPPLY_RESPONSE_SIGNALS,
    "isi_call_contract_book_gate": _CALL_BOOK_SIGNALS,
    "isi_adjustment_discontinuity_repricing": _ADJUSTMENT_REPRICING_SIGNALS,
    "isi_liquidity_execution_memory": _LIQUIDITY_EXECUTION_SIGNALS,
    "isi_cross_asset_daily_regime_gate": _CROSS_REGIME_SIGNALS,
}


def factor_mining_intraday_state_interaction_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable six-family, 36-signal research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return factor_mining_intraday_state_interaction_catalog()


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
        DailyFactorRequirement("market_cbond.daily_twap", ("exchange_code", *_TWAP_FIELDS), _LOOKBACK_DAYS),
    ]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required columns: {missing}")


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
        [_one(value, exchange) for value, exchange in zip(values, exchange_values, strict=False)],
        index=values.index,
        dtype="string",
    )


def _canonical_stock_code(value: object) -> str:
    """Normalize the explicitly T-1 daily-base mapping; never infer a code."""

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
    digits = text.zfill(6) if text.isdigit() else text
    if len(digits) != 6 or not digits.isdigit():
        return ""
    if digits[0] == "6":
        return f"{digits}.SH"
    if digits[0] in {"0", "3"}:
        return f"{digits}.SZ"
    if digits[0] in {"4", "8"}:
        return f"{digits}.BJ"
    return ""


def _panel_code(value: object) -> str:
    return str(_canonical_market_code(pd.Series([value])).iloc[0])


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
    if panel.empty or score_date is None:
        index = _empty_index()
    else:
        keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
        dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
        keys = keys.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
        index = pd.MultiIndex.from_frame(keys, names=("dt", "code"))

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing.index
        ctx.cache[cache_key] = pd.DataFrame(index=index)
    return index


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp, owner: str) -> pd.DataFrame:
    """Keep only genuine score-day intraday observations visible at 14:29."""

    checked = ensure_trade_time(panel)
    _require_columns(checked, _REQUIRED_PANEL_COLUMNS, owner=owner)
    frame = checked.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(lambda value: _continuous_session(value) if pd.notna(value) else False)
    keep = (
        indexed.notna()
        & (indexed.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    return out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")


def _strict_prior_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read only context-backed, strictly pre-score-day daily observations."""

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


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator) and numerator > _EPS and denominator > _EPS):
        return float("nan")
    return float(np.log(numerator / denominator))


def _safe_corr(left: np.ndarray, right: np.ndarray, *, min_count: int = 4) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < min_count:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if float(np.std(x)) <= _EPS or float(np.std(y)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _last_n_are_consecutive(frame: pd.DataFrame, count: int) -> bool:
    if len(frame) < count:
        return False
    positions = pd.to_numeric(frame.tail(count)["__price_session_index"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _complete_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count or not _last_n_are_consecutive(frame, count):
        return None
    values = np.column_stack([_numeric(frame.tail(count), column) for column in columns])
    return values if np.isfinite(values).all() else None


def _z_last_against_prior(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    deviation = float(np.std(prior, ddof=1))
    if not np.isfinite(deviation) or deviation <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / deviation)


def _beta(x: np.ndarray, y: np.ndarray, *, min_count: int) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < min_count:
        return float("nan")
    centered = x - float(np.mean(x))
    denominator = float(np.dot(centered, centered))
    if denominator <= _EPS or not np.isfinite(denominator):
        return float("nan")
    return float(np.dot(centered, y - float(np.mean(y))) / denominator)


def _event_recency(events: np.ndarray) -> float:
    if not len(events) or not np.isfinite(events.astype("float64")).all():
        return float("nan")
    locations = np.flatnonzero(events.astype(bool))
    return float(1.0 / (1.0 + (len(events) - 1 - int(locations[-1])))) if len(locations) else 0.0


def _counter_increments(frame: pd.DataFrame, column: str, *, allow_bounded_amount_correction: bool = False) -> np.ndarray | None:
    """Return visible counter increments without clipping or fill-zero repair."""

    values = _numeric(frame, column)
    if values.size < _MIN_PATH_ROWS or not (np.isfinite(values).all() and (values >= 0.0).all()):
        return None
    increments = np.diff(values)
    if not np.isfinite(increments).all():
        return None
    negative = increments < 0.0
    if negative.any():
        if column != "amount" or not allow_bounded_amount_correction:
            return None
        correction = -increments[negative]
        prior = np.abs(values[:-1][negative])
        allowed = (correction <= _AMOUNT_CORRECTION_ABS) & (correction <= prior * _AMOUNT_CORRECTION_RATIO)
        if not bool(allowed.all()):
            return None
    return increments


def _phase_mask(times: pd.Series, start: dt_time, end: dt_time, *, include_end: bool) -> np.ndarray:
    clocks = pd.to_datetime(times, errors="coerce").dt.time
    if include_end:
        return ((clocks >= start) & (clocks <= end)).to_numpy(dtype=bool)
    return ((clocks >= start) & (clocks < end)).to_numpy(dtype=bool)


def _log_return(values: np.ndarray) -> float:
    if len(values) < 2 or not (np.isfinite(values).all() and (values > _EPS).all()):
        return float("nan")
    return float(np.log(values[-1] / values[0]))


def _quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ask = _numeric(frame, "ask_price1")
    bid = _numeric(frame, "bid_price1")
    ask_size = _numeric(frame, "ask_volume1")
    bid_size = _numeric(frame, "bid_volume1")
    last = _numeric(frame, "last")
    total = ask_size + bid_size
    valid = (
        np.isfinite(ask)
        & np.isfinite(bid)
        & np.isfinite(ask_size)
        & np.isfinite(bid_size)
        & np.isfinite(last)
        & (ask > _EPS)
        & (bid > _EPS)
        & (ask >= bid)
        & (ask_size >= 0.0)
        & (bid_size >= 0.0)
        & (total > _EPS)
        & (last > _EPS)
    )
    out = frame.loc[valid].copy()
    if out.empty:
        for column in ("mid", "spread", "imbalance", "depth", "micro_bias"):
            out[column] = pd.Series(dtype="float64")
        return out
    mid = (ask[valid] + bid[valid]) / 2.0
    micro = (ask[valid] * bid_size[valid] + bid[valid] * ask_size[valid]) / total[valid]
    out["mid"] = mid
    out["spread"] = (ask[valid] - bid[valid]) / mid
    out["imbalance"] = (bid_size[valid] - ask_size[valid]) / total[valid]
    out["depth"] = total[valid]
    out["micro_bias"] = (micro - mid) / mid
    return out


def _empty_intraday_metrics() -> dict[str, float]:
    return {
        "gap": float("nan"),
        "gap_absorb": float("nan"),
        "gap_residual": float("nan"),
        "early_return": float("nan"),
        "late_return": float("nan"),
        "tail_efficiency": float("nan"),
        "tail_jump_share": float("nan"),
        "tail_drawdown": float("nan"),
        "tail_range": float("nan"),
        "terminal_location": float("nan"),
        "bipower_jump_share": float("nan"),
        "early_highwater_reclaim": float("nan"),
        "flow_accel": float("nan"),
        "tail_impact": float("nan"),
        "lull_release_ratio": float("nan"),
        "imbalance_last": float("nan"),
        "spread_shift": float("nan"),
        "depth_recovery": float("nan"),
        "mid_lead": float("nan"),
        "midlast": float("nan"),
        "quote_phase_gap": float("nan"),
        "tail_vwap_shift": float("nan"),
        "flow_lead": float("nan"),
        "depth_impact": float("nan"),
        "event_impact_shift": float("nan"),
        "longest_lull_share": float("nan"),
        "imbalance_shift": float("nan"),
        "micro_shift": float("nan"),
        "book_transition_vector_imbalance": float("nan"),
        "book_transition_vector_spread": float("nan"),
        "book_transition_vector_depth": float("nan"),
        "book_transition_vector_micro": float("nan"),
        "mid_early_return": float("nan"),
    }


def _intraday_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Summarise one physically visible path, with independent failure domains."""

    out = _empty_intraday_metrics()
    if len(frame) < _MIN_PATH_ROWS:
        return out
    if frame["seq"].duplicated(keep=False).any() or frame["trade_time"].duplicated(keep=False).any():
        return out
    times = pd.to_datetime(frame["trade_time"], errors="coerce")
    if times.isna().any() or not times.is_monotonic_increasing:
        return out
    time_ns = times.to_numpy(dtype="datetime64[ns]").astype("int64")
    if (np.diff(time_ns) <= 0).any():
        return out

    prices = _numeric(frame, "last")
    pre_close = _numeric(frame, "pre_close")
    if not (
        np.isfinite(prices).all()
        and np.isfinite(pre_close).all()
        and (prices > _EPS).all()
        and (pre_close > _EPS).all()
    ):
        return out
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < _MIN_PATH_ROWS - 1 or not np.isfinite(log_returns).all():
        return out

    out["gap"] = _safe_log_ratio(float(prices[0]), float(pre_close[0]))
    full_return = _log_return(prices)
    out["gap_absorb"] = -float(np.sign(out["gap"])) * full_return if np.isfinite(out["gap"]) else float("nan")
    early_rows = _phase_mask(times, _MORNING_START, _EARLY_END, include_end=True)
    late_rows = _phase_mask(times, _LATE_START, _CUTOFF, include_end=True)
    early = _log_return(prices[early_rows])
    late = _log_return(prices[late_rows])
    out["early_return"] = early
    out["late_return"] = late
    out["gap_residual"] = out["gap"] + early if np.isfinite(out["gap"]) and np.isfinite(early) else float("nan")

    high = float(np.max(prices))
    low = float(np.min(prices))
    path_range = high - low
    out["terminal_location"] = _safe_div(float(prices[-1] - low), path_range)
    out["tail_range"] = _safe_div(float(np.max(prices[late_rows]) - np.min(prices[late_rows])), float(prices[late_rows][0])) if int(late_rows.sum()) >= 2 else float("nan")
    out["bipower_jump_share"] = _safe_div(float(np.max(np.abs(log_returns))), float(np.abs(log_returns).sum()))
    if int(early_rows.sum()) >= 2:
        early_high = float(np.max(prices[early_rows]))
        out["early_highwater_reclaim"] = _safe_log_ratio(float(prices[-1]), early_high)
    if int(late_rows.sum()) >= 3:
        tail_prices = prices[late_rows]
        tail_returns = np.diff(np.log(tail_prices))
        out["tail_efficiency"] = _safe_div(float(tail_returns.sum()), float(np.abs(tail_returns).sum()))
        out["tail_jump_share"] = _safe_div(float(np.max(np.abs(tail_returns))), float(np.abs(tail_returns).sum()))
        wealth = tail_prices / tail_prices[0]
        out["tail_drawdown"] = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))

    quotes = _quote_frame(frame)
    if not quotes.empty:
        quote_times = pd.to_datetime(quotes["trade_time"], errors="coerce")
        q_early = _phase_mask(quote_times, _MORNING_START, _EARLY_END, include_end=True)
        q_late = _phase_mask(quote_times, _LATE_START, _CUTOFF, include_end=True)
        q_mid = _numeric(quotes, "mid")
        q_last = _numeric(quotes, "last")
        q_imb = _numeric(quotes, "imbalance")
        q_spread = _numeric(quotes, "spread")
        q_depth = _numeric(quotes, "depth")
        q_micro = _numeric(quotes, "micro_bias")
        if len(q_mid) >= 2:
            out["midlast"] = _safe_log_ratio(float(q_last[-1]), float(q_mid[-1]))
            mid_returns = np.diff(np.log(q_mid))
            last_returns = np.diff(np.log(q_last))
            out["mid_lead"] = _safe_corr(mid_returns[:-1], last_returns[1:], min_count=3)
        if int(q_early.sum()) >= 2:
            out["mid_early_return"] = _log_return(q_mid[q_early])
        if int(q_late.sum()) >= 2:
            mid_late = _log_return(q_mid[q_late])
            last_late = _log_return(q_last[q_late])
            mid_early = _log_return(q_mid[q_early]) if int(q_early.sum()) >= 2 else float("nan")
            last_early = _log_return(q_last[q_early]) if int(q_early.sum()) >= 2 else float("nan")
            if np.isfinite(mid_late) and np.isfinite(last_late) and np.isfinite(mid_early) and np.isfinite(last_early):
                out["quote_phase_gap"] = (mid_late - last_late) - (mid_early - last_early)
        if int(q_early.sum()) >= 2 and int(q_late.sum()) >= 2:
            imb_early, imb_late = float(np.mean(q_imb[q_early])), float(np.mean(q_imb[q_late]))
            spread_early, spread_late = float(np.mean(q_spread[q_early])), float(np.mean(q_spread[q_late]))
            depth_early, depth_late = float(np.mean(q_depth[q_early])), float(np.mean(q_depth[q_late]))
            micro_early, micro_late = float(np.mean(q_micro[q_early])), float(np.mean(q_micro[q_late]))
            out["imbalance_last"] = float(q_imb[-1])
            out["imbalance_shift"] = imb_late - imb_early
            out["spread_shift"] = spread_late - spread_early
            out["depth_recovery"] = _safe_log_ratio(depth_late, depth_early)
            out["micro_shift"] = micro_late - micro_early
            out["book_transition_vector_imbalance"] = out["imbalance_shift"]
            out["book_transition_vector_spread"] = out["spread_shift"]
            out["book_transition_vector_depth"] = out["depth_recovery"]
            out["book_transition_vector_micro"] = out["micro_shift"]
        if len(q_depth) >= 5:
            out["depth_impact"] = _safe_corr(np.diff(np.log(q_depth)), np.diff(np.log(q_last)), min_count=4)

    amount = _counter_increments(frame, "amount", allow_bounded_amount_correction=True)
    volume = _counter_increments(frame, "volume")
    trades = _counter_increments(frame, "num_trades")
    interval_late = late_rows[1:]
    interval_early = early_rows[1:]
    if amount is not None and int(interval_late.sum()) >= 2 and int(interval_early.sum()) >= 2:
        late_amount = amount[interval_late]
        early_amount = amount[interval_early]
        late_sum = float(late_amount.sum())
        early_sum = float(early_amount.sum())
        out["flow_accel"] = _safe_log_ratio(float(np.mean(late_amount)), float(np.mean(early_amount)))
        tail_abs = float(np.abs(log_returns[interval_late]).sum())
        out["tail_impact"] = _safe_div(tail_abs, float(np.log1p(abs(late_sum))))
        if late_sum > _EPS and early_sum > _EPS:
            late_vwap = float(np.dot(prices[1:][interval_late], late_amount) / late_sum)
            early_vwap = float(np.dot(prices[1:][interval_early], early_amount) / early_sum)
            out["tail_vwap_shift"] = _safe_log_ratio(late_vwap, early_vwap)
    if volume is not None and len(volume) >= 5:
        out["flow_lead"] = _safe_corr(np.log1p(volume[:-1]), log_returns[1:], min_count=4)
    if trades is not None and len(trades) == len(log_returns):
        zero = trades <= 0.0
        out["longest_lull_share"] = _longest_run_share(zero)
        release = np.abs(log_returns[1:][zero[:-1]]) if len(zero) >= 2 else np.array([], dtype="float64")
        baseline = np.abs(log_returns)
        if len(release) and float(np.mean(baseline)) > _EPS:
            out["lull_release_ratio"] = _safe_div(float(np.mean(release)), float(np.mean(baseline)))
        if int(interval_late.sum()) >= 2 and int(interval_early.sum()) >= 2:
            early_impact = _safe_div(float(np.mean(np.abs(log_returns[interval_early]))), float(np.mean(trades[interval_early])))
            late_impact = _safe_div(float(np.mean(np.abs(log_returns[interval_late]))), float(np.mean(trades[interval_late])))
            out["event_impact_shift"] = late_impact - early_impact if np.isfinite(late_impact) and np.isfinite(early_impact) else float("nan")
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in out.items()}


def _longest_run_share(values: np.ndarray) -> float:
    if not len(values):
        return float("nan")
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest / len(values))


def _daily_state_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Build only strict-prior state values from one price-anchored history."""

    names = (
        "floor_change5", "redemption_prem_z20", "convexity_duration", "maturity_duration_gap",
        "premium_yield_innovation", "floor_vol20", "supply_shrink1", "supply_event_recency60",
        "supply_event_rate20", "supply_cumchange20", "supply_shrink_liq_impulse", "supply_change1",
        "call_age", "call_progress_velocity5", "call_required_days", "barrier_asym", "contract_recency60",
        "adj_net", "adj_cross", "adj_abs_imbalance", "adj_magnitude", "adj_recency60",
        "adj_event_rate20", "lagflow_beta20", "highflow_return_spread20", "impact_memory20",
        "amount_persistence20", "amihud_z20", "twap_transition_surprise", "bstk_beta_shift10_40",
        "bstk_residual_vol20", "svfe_error_z20", "bstk_residual_last20", "lqr_amount_deal_divergence",
        "barrier_trigger_distance",
    )
    out = {name: float("nan") for name in names}

    redemption6 = _complete_tail(frame, ("pure_redemption_value",), 6)
    if redemption6 is not None:
        out["floor_change5"] = _safe_log_ratio(float(redemption6[-1, 0]), float(redemption6[0, 0]))
    redemption21 = _complete_tail(frame, ("pure_redemption_value", "redemption_prem_ratio"), 21)
    if redemption21 is not None:
        out["redemption_prem_z20"] = _z_last_against_prior(redemption21[:, 1])
        out["floor_vol20"] = float(np.std(redemption21[:, 0], ddof=1))
    defensive = _complete_tail(frame, ("convexity", "duration", "year_to_mat"), 1)
    if defensive is not None:
        convexity, duration, maturity = defensive[-1]
        out["convexity_duration"] = _safe_div(float(convexity), float(duration))
        out["maturity_duration_gap"] = float(maturity - duration)
    premium_yield = _complete_tail(frame, ("bond_prem_ratio", "ytm"), 61)
    if premium_yield is not None:
        premium_delta = np.diff(premium_yield[:, 0])
        yield_delta = np.diff(premium_yield[:, 1])
        valid = np.isfinite(premium_delta) & np.isfinite(yield_delta)
        if int(valid.sum()) >= 40 and float(np.std(yield_delta[valid])) > _EPS:
            design = np.column_stack([np.ones(int(valid.sum())), yield_delta[valid]])
            intercept, slope = np.linalg.lstsq(design, premium_delta[valid], rcond=None)[0]
            out["premium_yield_innovation"] = float(premium_delta[-1] - (intercept + slope * yield_delta[-1]))

    supply2 = _complete_tail(frame, ("remain_size", "cb_amount"), 2)
    if supply2 is not None:
        change = _safe_log_ratio(float(supply2[-1, 0]), float(supply2[0, 0]))
        out["supply_change1"] = change
        out["supply_shrink1"] = max(-change, 0.0) if np.isfinite(change) else float("nan")
        if np.isfinite(out["supply_shrink1"]) and supply2[-1, 0] > _EPS and supply2[-1, 1] >= 0.0:
            out["supply_shrink_liq_impulse"] = float(out["supply_shrink1"] * np.log1p(supply2[-1, 1] / supply2[-1, 0]))
    supply21 = _complete_tail(frame, ("remain_size",), 21)
    if supply21 is not None and (supply21[:, 0] > _EPS).all():
        changes = np.diff(np.log(supply21[:, 0]))
        out["supply_event_rate20"] = float(np.mean(np.abs(changes) > _EPS))
        out["supply_cumchange20"] = float(np.log(supply21[-1, 0] / supply21[0, 0]))
    supply61 = _complete_tail(frame, ("remain_size",), 61)
    if supply61 is not None and (supply61[:, 0] > _EPS).all():
        out["supply_event_recency60"] = _event_recency(np.abs(np.diff(np.log(supply61[:, 0]))) > _EPS)

    active = _complete_tail(frame, ("in_trigger_process",), 1)
    if active is not None:
        is_active = bool(active[-1, 0] > 0.0)
        out["call_age"] = 0.0
        out["call_progress_velocity5"] = 0.0
        out["call_required_days"] = 0.0
        if is_active:
            state = _numeric(frame, "in_trigger_process")
            positions = _numeric(frame, "__price_session_index")
            age = 0
            previous: float | None = None
            for index in range(len(state) - 1, -1, -1):
                if not np.isfinite(state[index]) or state[index] <= 0.0:
                    break
                if previous is not None and previous - positions[index] != 1.0:
                    break
                previous = positions[index]
                age += 1
            out["call_age"] = float(age)
            revised = _complete_tail(frame, ("trigger_cum_days_revise", "trigger_reach_days_revise"), 1)
            if revised is not None and revised[-1, 1] > _EPS:
                out["call_required_days"] = float(revised[-1, 1])
            revised6 = _complete_tail(frame, ("in_trigger_process", "trigger_cum_days_revise", "trigger_reach_days_revise"), 6)
            if revised6 is not None and (revised6[:, 0] > 0.0).all() and (revised6[:, 2] > _EPS).all():
                progress = revised6[:, 1] / revised6[:, 2]
                out["call_progress_velocity5"] = float(progress[-1] - progress[0])
    barrier = _complete_tail(frame, ("cb_call_price", "cb_put_price", "stock_close_price"), 1)
    if barrier is not None:
        call, put, stock = barrier[-1]
        out["barrier_asym"] = _safe_div(float(call + put - 2.0 * stock), float(call - put))
    contract61 = _complete_tail(frame, ("cb_conv_price", "cb_put_price", "cb_call_price"), 61)
    if contract61 is not None:
        out["contract_recency60"] = _event_recency((np.abs(np.diff(contract61, axis=0)) > _EPS).any(axis=1))

    adjustment = _complete_tail(
        frame,
        ("prev_close_price", "act_prev_close_price", "stk_prev_close_price", "stk_act_prev_close_price"),
        1,
    )
    if adjustment is not None:
        cb = _safe_log_ratio(float(adjustment[-1, 1]), float(adjustment[-1, 0]))
        stock = _safe_log_ratio(float(adjustment[-1, 3]), float(adjustment[-1, 2]))
        if np.isfinite(cb) and np.isfinite(stock):
            out["adj_net"] = cb + stock
            out["adj_cross"] = cb - stock
            out["adj_abs_imbalance"] = abs(cb) - abs(stock)
            out["adj_magnitude"] = abs(cb) + abs(stock)
    adjustment21 = _complete_tail(
        frame,
        ("prev_close_price", "act_prev_close_price", "stk_prev_close_price", "stk_act_prev_close_price"),
        21,
    )
    if adjustment21 is not None:
        cb = np.log(adjustment21[:, 1] / adjustment21[:, 0])
        stock = np.log(adjustment21[:, 3] / adjustment21[:, 2])
        out["adj_event_rate20"] = float(np.mean((np.abs(cb[1:]) + np.abs(stock[1:])) > _EPS))
    adjustment61 = _complete_tail(
        frame,
        ("prev_close_price", "act_prev_close_price", "stk_prev_close_price", "stk_act_prev_close_price"),
        61,
    )
    if adjustment61 is not None:
        cb = np.log(adjustment61[:, 1] / adjustment61[:, 0])
        stock = np.log(adjustment61[:, 3] / adjustment61[:, 2])
        out["adj_recency60"] = _event_recency((np.abs(cb[1:]) + np.abs(stock[1:])) > _EPS)

    liquidity21 = _complete_tail(
        frame,
        ("close_price", "prev_close_price", "amount", "turnover_rate", "twap_0930_1000", "twap_1400_1430", "twap_1442_1457"),
        21,
    )
    if liquidity21 is not None:
        close, previous, amount, turnover, morning, late_twap, execution = liquidity21.T
        returns = np.log(close / previous)
        log_amount = np.log1p(amount)
        flow_delta = np.diff(log_amount)
        out["lagflow_beta20"] = _beta(flow_delta[:-1], returns[2:], min_count=12) if len(flow_delta) >= 3 else float("nan")
        # Flow change from session t-1 to t is paired only with the next
        # completed daily return t+1; this keeps the two arrays aligned.
        next_returns = returns[2:]
        flow_for_next = flow_delta[:-1]
        if len(flow_for_next) >= 8:
            median = float(np.median(flow_for_next))
            high = next_returns[flow_for_next >= median]
            low = next_returns[flow_for_next < median]
            if len(high) >= 3 and len(low) >= 3:
                out["highflow_return_spread20"] = float(np.mean(high) - np.mean(low))
        impact = np.abs(returns) / log_amount
        out["impact_memory20"] = float(np.mean(impact)) if np.isfinite(impact).all() else float("nan")
        out["amount_persistence20"] = _safe_corr(log_amount[:-1], log_amount[1:], min_count=12)
        out["amihud_z20"] = _z_last_against_prior(impact)
        transition = np.log(late_twap / morning) + np.log(execution / late_twap)
        out["twap_transition_surprise"] = _z_last_against_prior(transition)

    tracking41 = _complete_tail(frame, ("close_price", "prev_close_price", "stock_close_price", "stk_prev_close_price"), 41)
    if tracking41 is not None:
        bond_return = np.log(tracking41[:, 0] / tracking41[:, 1])
        stock_return = np.log(tracking41[:, 2] / tracking41[:, 3])
        beta_short = _beta(stock_return[-10:], bond_return[-10:], min_count=8)
        beta_long = _beta(stock_return[-40:], bond_return[-40:], min_count=30)
        out["bstk_beta_shift10_40"] = beta_short - beta_long if np.isfinite(beta_short) and np.isfinite(beta_long) else float("nan")
        beta20 = _beta(stock_return[-20:], bond_return[-20:], min_count=15)
        if np.isfinite(beta20):
            residual = bond_return[-20:] - beta20 * stock_return[-20:]
            out["bstk_residual_vol20"] = float(np.std(residual, ddof=1))
            out["bstk_residual_last20"] = float(residual[-1])
    volatility21 = _complete_tail(frame, ("stock_volatility", "stock_close_price", "stk_prev_close_price"), 21)
    if volatility21 is not None:
        stated, stock_close, stock_previous = volatility21.T
        stock_return = np.log(stock_close / stock_previous)
        errors: list[float] = []
        for end in range(5, len(stock_return) + 1):
            realized = float(np.std(stock_return[end - 5 : end], ddof=1))
            errors.append(float(stated[end - 1] - realized))
        out["svfe_error_z20"] = _z_last_against_prior(np.asarray(errors, dtype="float64"))
    liquidity_share = _complete_tail(frame, ("cb_amount", "stk_amount", "cb_deal", "stk_deal"), 1)
    if liquidity_share is not None:
        cb_amount, stock_amount, cb_deal, stock_deal = liquidity_share[-1]
        amount_ratio = _safe_log_ratio(float(cb_amount), float(stock_amount))
        deal_ratio = _safe_log_ratio(float(cb_deal), float(stock_deal))
        out["lqr_amount_deal_divergence"] = amount_ratio - deal_ratio if np.isfinite(amount_ratio) and np.isfinite(deal_ratio) else float("nan")
    trigger_distance = _complete_tail(frame, ("trigger_price_revise", "stock_close_price"), 1)
    if trigger_distance is not None:
        trigger, stock = trigger_distance[-1]
        out["barrier_trigger_distance"] = _safe_div(abs(float(trigger - stock)), float(stock))
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in out.items()}


def _prior_state_frame(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:prior_state:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    price = _strict_prior_source(ctx, source="market_cbond.daily_price", fields=_PRICE_FIELDS, score_date=score_date)
    base = _strict_prior_source(ctx, source="market_cbond.daily_base", fields=_BASE_FIELDS, score_date=score_date)
    twap = _strict_prior_source(ctx, source="market_cbond.daily_twap", fields=_TWAP_FIELDS, score_date=score_date)
    if price.empty or base.empty:
        built = pd.DataFrame(columns=["code", "__stock_code"])
    else:
        anchor = pd.Timestamp(price["trade_date"].max()).normalize()
        sessions = {pd.Timestamp(day).normalize(): position for position, day in enumerate(sorted(price["trade_date"].unique()))}
        anchor_base_codes = set(base.loc[base["trade_date"] == anchor, "code"].astype(str))
        base_values = base.drop(columns=["exchange_code"])
        twap_values = twap.drop(columns=["exchange_code"])
        rows: list[dict[str, object]] = []
        for code, group in price.groupby("code", sort=False):
            ordered = group.sort_values("trade_date", kind="mergesort").copy()
            if (
                ordered.empty
                or pd.Timestamp(ordered["trade_date"].iloc[-1]).normalize() != anchor
                or str(code) not in anchor_base_codes
            ):
                continue
            merged = ordered.merge(base_values, on=["code", "trade_date"], how="left", validate="one_to_one")
            merged = merged.merge(twap_values, on=["code", "trade_date"], how="left", validate="one_to_one")
            merged["__price_session_index"] = merged["trade_date"].map(sessions).astype("float64")
            state = _daily_state_metrics(merged)
            state["code"] = str(code)
            state["__stock_code"] = _canonical_stock_code(merged.loc[merged.index[-1], "stock_code"])
            rows.append(state)
        built = pd.DataFrame(rows)
        if not built.empty:
            built = built.set_index("code", drop=False).sort_index()
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _value(values: dict[str, float], key: str) -> float:
    value = values.get(key)
    return float(value) if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else float("nan")


def _product(left: float, right: float) -> float:
    return float(left * right) if np.isfinite(left) and np.isfinite(right) else float("nan")


def _cross_book_coherence(bond: dict[str, float], stock: dict[str, float]) -> float:
    fields = (
        "book_transition_vector_imbalance",
        "book_transition_vector_spread",
        "book_transition_vector_depth",
        "book_transition_vector_micro",
    )
    left = np.asarray([_value(bond, field) for field in fields], dtype="float64")
    right = np.asarray([_value(stock, field) for field in fields], dtype="float64")
    return _safe_corr(left, right, min_count=3)


def _output_metrics(bond: dict[str, float], state: dict[str, float], stock: dict[str, float] | None) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}

    def b(name: str) -> float:
        return _value(bond, name)

    def s(name: str) -> float:
        return _value(state, name)

    out.update(
        {
            "isi_def_tail_eff_x_floor_change5": _product(b("tail_efficiency"), s("floor_change5")),
            "isi_def_tail_jump_x_redemption_prem_z20": _product(b("tail_jump_share"), s("redemption_prem_z20")),
            "isi_def_gapabsorb_x_convexity_duration": _product(b("gap_absorb"), s("convexity_duration")),
            "isi_def_terminal_loc_x_maturity_duration_gap": _product(b("terminal_location"), s("maturity_duration_gap")),
            "isi_def_noise_x_premium_yield_innovation": _product(b("bipower_jump_share"), s("premium_yield_innovation")),
            "isi_def_reclaim_x_floor_vol20": _product(b("early_highwater_reclaim"), s("floor_vol20")),
            "isi_supply_flowaccel_x_shrink1": _product(b("flow_accel"), s("supply_shrink1")),
            "isi_supply_tail_impact_x_recency60": _product(b("tail_impact"), s("supply_event_recency60")),
            "isi_supply_depthrec_x_eventrate20": _product(b("depth_recovery"), s("supply_event_rate20")),
            "isi_supply_gapabsorb_x_cumchange20": _product(b("gap_absorb"), s("supply_cumchange20")),
            "isi_supply_midlast_x_shrink_liq_impulse": _product(b("midlast"), s("supply_shrink_liq_impulse")),
            "isi_supply_lullrelease_x_change1": _product(b("lull_release_ratio"), s("supply_change1")),
            "isi_call_imbalance_x_active_age": _product(b("imbalance_last"), s("call_age")),
            "isi_call_spread_shift_x_progress_velocity": _product(b("spread_shift"), s("call_progress_velocity5")),
            "isi_call_depthrec_x_required_days": _product(b("depth_recovery"), s("call_required_days")),
            "isi_call_midlead_x_barrier_asym": _product(b("mid_lead"), s("barrier_asym")),
            "isi_call_gapabsorb_x_contract_recency": _product(b("gap_absorb"), s("contract_recency60")),
            "isi_call_midlead_x_active_age": _product(b("mid_lead"), s("call_age")),
            "isi_adj_open_gap_x_net_shift": _product(b("gap"), s("adj_net")),
            "isi_adj_gapabsorb_x_cross_gap": _product(b("gap_absorb"), s("adj_cross")),
            "isi_adj_midlast_x_absimbalance": _product(b("midlast"), s("adj_abs_imbalance")),
            "isi_adj_tailflow_x_magnitude": _product(b("flow_accel"), s("adj_magnitude")),
            "isi_adj_rotation_x_recency": _product(b("late_return") - b("early_return"), s("adj_recency60")),
            "isi_adj_quotephase_x_eventrate": _product(b("quote_phase_gap"), s("adj_event_rate20")),
            "isi_liq_tailvwap_x_lagflowbeta": _product(b("tail_vwap_shift"), s("lagflow_beta20")),
            "isi_liq_flowlead_x_highflow_response": _product(b("flow_lead"), s("highflow_return_spread20")),
            "isi_liq_depthimpact_x_impactmemory": _product(b("depth_impact"), s("impact_memory20")),
            "isi_liq_gapresid_x_amount_persistence": _product(b("gap_residual"), s("amount_persistence20")),
            "isi_liq_eventimpact_x_amihud_z": _product(b("event_impact_shift"), s("amihud_z20")),
            "isi_liq_lull_x_twap_transition": _product(b("longest_lull_share"), s("twap_transition_surprise")),
        }
    )
    if stock is not None:
        def st(name: str) -> float:
            return _value(stock, name)

        phase_rotation = (b("late_return") - b("early_return")) - (st("late_return") - st("early_return"))
        tail_cojump = b("late_return") * st("late_return")
        stock_shock_book = st("early_return") * b("imbalance_last")
        tail_range = b("tail_range") * st("tail_range")
        quote_channel = st("mid_early_return") * b("late_return")
        out.update(
            {
                "isi_cross_phase_rotation_x_beta_shift": _product(phase_rotation, s("bstk_beta_shift10_40")),
                "isi_cross_tailcojump_x_residvol": _product(tail_cojump, s("bstk_residual_vol20")),
                "isi_cross_bookcoherence_x_volforecast": _product(_cross_book_coherence(bond, stock), s("svfe_error_z20")),
                "isi_cross_stockshockbook_x_trackingresid": _product(stock_shock_book, s("bstk_residual_last20")),
                "isi_cross_tailrange_x_liqsharediv": _product(tail_range, s("lqr_amount_deal_divergence")),
                "isi_cross_quotechannel_x_barrierdist": _product(quote_channel, s("barrier_trigger_distance")),
            }
        )
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        if ctx.stock_panel is None or ctx.stock_panel.empty:
            raise RuntimeError(f"{KERNEL_NAME} requires a non-empty stock_panel")
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date, owner="bond_panel")
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date, owner="stock_panel")
        states = _prior_state_frame(ctx, score_date=score_date)
        bond_groups = {_panel_code(code): group for code, group in bond_frame.groupby("code", sort=False) if _panel_code(code)}
        stock_groups = {_panel_code(code): group for code, group in stock_frame.groupby("code", sort=False) if _panel_code(code)}
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            code = _panel_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            if not code or code not in states.index or code not in bond_groups:
                row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            else:
                state_record = states.loc[code].to_dict()
                bond = _intraday_metrics(bond_groups[code])
                stock_code = str(state_record.get("__stock_code", ""))
                stock = _intraday_metrics(stock_groups[stock_code]) if stock_code in stock_groups else None
                row.update(_output_metrics(bond, state_record, stock))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayStateInteractionV1(Factor):
    """Research-only strict-T1430 and strict-T-1 interaction catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    # The only mapping comes from the exact T-1 daily-base anchor in this file.
    requires_bond_stock_map = False

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return _requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningIntradayStateInteractionV1",
    "factor_mining_catalog",
    "factor_mining_intraday_state_interaction_catalog",
]
