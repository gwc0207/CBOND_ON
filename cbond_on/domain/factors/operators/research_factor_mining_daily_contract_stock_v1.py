"""Research-only strict-T-1 contract and bond/stock daily-state catalogue.

This module is deliberately import-only.  It is not included from
``defs.__init__`` and is not part of any factor, model, or live configuration.
Every signal uses only ``trade_date < score_date`` daily DataHub rows.  In
addition, every output code must have both a daily-price row and a *latest*
daily-base row on the independent daily-price T-1 anchor; a stale base row
therefore produces ``NaN`` rather than silently carrying a state forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


KERNEL_NAME = "factor_mining_daily_contract_stock_v1"
CATALOG_VERSION = "20260803_daily_contract_stock_v1"
_LOOKBACK_DAYS = 66
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CONTRACT_SIGNALS = (
    "ctr_conv_term_change1",
    "ctr_put_term_change1",
    "ctr_call_term_change1",
    "ctr_term_event_count20",
    "ctr_term_corevision_rate20",
    "ctr_term_directional_coherence1",
    "ctr_term_revision_recency60",
)
_BARRIER_SIGNALS = (
    "barrier_center_position",
    "barrier_width_to_price",
    "barrier_nearest_edge_norm",
    "barrier_side_asymmetry",
    "barrier_revision_trigger_position",
    "barrier_revision_trigger_distance",
)
_TRACKING_SIGNALS = (
    "bstk_residual_last20",
    "bstk_residual_vol20",
    "bstk_residual_autocorr20",
    "bstk_beta_shift10_40",
    "bstk_tail_cocrash_residual20",
    "bstk_tracking_drawdown20",
)
_LIQUIDITY_SIGNALS = (
    "lqr_amount_share_delta1",
    "lqr_deal_share_delta1",
    "lqr_amount_share_z20",
    "lqr_deal_share_z20",
    "lqr_amount_deal_divergence",
    "lqr_flow_return_wedge",
    "lqr_amount_share_persistence20",
)
_VOLATILITY_SIGNALS = (
    "svfe_level_ratio20",
    "svfe_error_z20",
    "svfe_implied_change_minus_realized_change",
    "svfe_abs_return_scale",
    "svfe_realized_short_long_gap",
    "svfe_error_autocorr20",
)
_TRIGGER_SIGNALS = (
    "trw_mode_wedge",
    "trw_required_days_wedge",
    "trw_progress_wedge",
    "trw_progress_velocity_wedge5",
    "trw_revision_recency60",
    "trw_revision_event_rate20",
    "trw_post_revision_activation",
)
_RATING_SIGNALS = (
    "rating_current_ordinal",
    "rating_change1",
    "rating_signed_decay20",
    "rating_upgrade_decay20",
    "rating_downgrade_decay20",
    "rating_event_recency60",
    "rating_migration_ytm_wedge",
)

_CATALOG = (
    _entries(
        "contract_term_revision_vector",
        _CONTRACT_SIGNALS,
        "Changes and co-revisions in conversion, put, and call contract terms are distinct from their static moneyness.",
    )
    + _entries(
        "put_call_barrier_geometry",
        _BARRIER_SIGNALS,
        "The joint geometry of put, call, and revised-trigger barriers is distinct from a one-sided call distance.",
    )
    + _entries(
        "daily_bond_stock_tracking_error",
        _TRACKING_SIGNALS,
        "Completed daily bond-versus-stock residual behaviour captures tracking-error regime rather than intraday beta.",
    )
    + _entries(
        "bond_stock_liquidity_reallocation",
        _LIQUIDITY_SIGNALS,
        "Changes in the bond share of paired bond/stock activity describe reallocation rather than a static liquidity ratio.",
    )
    + _entries(
        "stock_volatility_forecast_error",
        _VOLATILITY_SIGNALS,
        "Reported stock volatility versus trailing realized stock volatility describes a forecast-error state.",
    )
    + _entries(
        "trigger_revision_phase_wedge",
        _TRIGGER_SIGNALS,
        "Revised trigger mode, required-day, and progress states describe a revision phase rather than raw redemption progress.",
    )
    + _entries(
        "rating_migration_decay",
        _RATING_SIGNALS,
        "Explicit ordinal rating migrations retain a decaying historical state that is distinct from a raw yield level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "contract_term_revision_vector": _CONTRACT_SIGNALS,
    "put_call_barrier_geometry": _BARRIER_SIGNALS,
    "daily_bond_stock_tracking_error": _TRACKING_SIGNALS,
    "bond_stock_liquidity_reallocation": _LIQUIDITY_SIGNALS,
    "stock_volatility_forecast_error": _VOLATILITY_SIGNALS,
    "trigger_revision_phase_wedge": _TRIGGER_SIGNALS,
    "rating_migration_decay": _RATING_SIGNALS,
}

# ``daily_price`` is present for every family solely to establish a source
# independent T-1 session anchor.  Contract barrier prices are in underlying
# stock-price units, so barrier geometry uses ``daily_base.stock_close_price``
# rather than the convertible-bond close from ``daily_price``.
_FAMILY_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "contract_term_revision_vector": ("cb_conv_price", "cb_put_price", "cb_call_price"),
    "put_call_barrier_geometry": (
        "cb_put_price",
        "cb_call_price",
        "trigger_price_revise",
        "stock_close_price",
    ),
    "daily_bond_stock_tracking_error": (
        "cb_prev_close_price",
        "cb_close_price",
        "stk_prev_close_price",
        "stk_close_price",
    ),
    "bond_stock_liquidity_reallocation": (
        "cb_amount",
        "stk_amount",
        "cb_deal",
        "stk_deal",
        "cb_prev_close_price",
        "cb_close_price",
        "stk_prev_close_price",
        "stk_close_price",
    ),
    "stock_volatility_forecast_error": (
        "stock_volatility",
        "stk_prev_close_price",
        "stk_close_price",
    ),
    "trigger_revision_phase_wedge": (
        "trigger_price_revise",
        "trigger_is_price",
        "trigger_is_price_revise",
        "trigger_cum_days",
        "trigger_reach_days",
        "trigger_cum_days_revise",
        "trigger_reach_days_revise",
        "in_trigger_process",
    ),
    "rating_migration_decay": ("rating", "ytm"),
}

_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})

# Higher is stronger credit quality.  The mapping is intentionally explicit:
# alphabetical sorting would put AA+ / AA / AA- in the wrong economic order.
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


def daily_contract_stock_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only family-first catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research-launcher compatibility alias."""

    return daily_contract_stock_catalog()


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
            ("exchange_code", "close_price"),
            _LOOKBACK_DAYS,
        ),
        DailyFactorRequirement(
            "market_cbond.daily_base",
            ("exchange_code", *_FAMILY_BASE_FIELDS[family]),
            _LOOKBACK_DAYS,
        ),
    ]


def _all_requirements() -> list[DailyFactorRequirement]:
    base_fields: set[str] = {"exchange_code"}
    for fields in _FAMILY_BASE_FIELDS.values():
        base_fields.update(fields)
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_base", tuple(sorted(base_fields)), _LOOKBACK_DAYS),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize DataHub codes while preserving exchange disambiguation."""

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
        normalized_exchange = _EXCHANGE_ALIASES.get(raw_exchange, raw_exchange)
        return f"{text}.{normalized_exchange}" if normalized_exchange in _MARKET_EXCHANGES else ""

    return pd.Series(
        [_one(value, exchange) for value, exchange in zip(values, exchange_values)],
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
    """Read a declared source from context and remove score-day/future rows."""

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
            frame.duplicated(["trade_date", "code"], keep=False),
            ["trade_date", "code"],
        ].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    key_frame = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].drop_duplicates()
    key_frame = key_frame.sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(key_frame, names=("dt", "code"))


def _score_date_from_index(index: pd.MultiIndex) -> pd.Timestamp:
    dates = pd.to_datetime(index.get_level_values("dt"), errors="coerce").normalize().unique()
    dates = [pd.Timestamp(value) for value in dates if not pd.isna(value)]
    if len(dates) != 1:
        raise ValueError(f"{KERNEL_NAME} requires one valid score date per context")
    return dates[0]


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _finite_scalar(value: float | int | np.floating | None) -> float:
    if value is None:
        return float("nan")
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("nan")


def _pct_change(current: float, previous: float) -> float:
    if not (np.isfinite(current) and np.isfinite(previous)) or abs(previous) <= _EPS:
        return float("nan")
    return float(current / previous - 1.0)


def _complete_numeric_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count:
        return None
    values = np.column_stack([_numeric(frame.tail(count), column) for column in columns])
    if not np.isfinite(values).all():
        return None
    if not _last_n_are_consecutive(frame, count):
        return None
    return values


def _last_n_are_consecutive(frame: pd.DataFrame, count: int) -> bool:
    """Require completed observations to span adjacent price-calendar days.

    Direct helper tests do not carry the internal ordinal and retain their
    pure-formula behaviour.  Production family frames always carry it after
    their date-key join to the independent daily-price calendar.
    """

    if len(frame) < count:
        return False
    if "__price_session_index" not in frame.columns:
        return True
    positions = pd.to_numeric(frame.tail(count)["__price_session_index"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _z_last_against_prior(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    std = float(np.std(prior, ddof=1))
    if not np.isfinite(std) or std <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / std)


def _autocorr(values: np.ndarray) -> float:
    if len(values) < 4 or not np.isfinite(values).all():
        return float("nan")
    left = values[:-1]
    right = values[1:]
    if np.std(left) <= _EPS or np.std(right) <= _EPS:
        return float("nan")
    return _finite_scalar(np.corrcoef(left, right)[0, 1])


def _safe_beta(stock_returns: np.ndarray, bond_returns: np.ndarray) -> float:
    if len(stock_returns) < 5 or len(stock_returns) != len(bond_returns):
        return float("nan")
    if not (np.isfinite(stock_returns).all() and np.isfinite(bond_returns).all()):
        return float("nan")
    centered_stock = stock_returns - float(np.mean(stock_returns))
    denominator = float(np.dot(centered_stock, centered_stock))
    if not np.isfinite(denominator) or denominator <= _EPS:
        return float("nan")
    centered_bond = bond_returns - float(np.mean(bond_returns))
    return _finite_scalar(float(np.dot(centered_stock, centered_bond) / denominator))


def _history_returns(frame: pd.DataFrame, *, bond_prefix: str = "cb", stock_prefix: str = "stk") -> tuple[np.ndarray, np.ndarray] | None:
    columns = (
        f"{bond_prefix}_prev_close_price",
        f"{bond_prefix}_close_price",
        f"{stock_prefix}_prev_close_price",
        f"{stock_prefix}_close_price",
    )
    values = _complete_numeric_tail(frame, columns, min(len(frame), _LOOKBACK_DAYS))
    if values is None:
        return None
    bond_previous, bond_close, stock_previous, stock_close = values.T
    if (
        (bond_previous <= _EPS).any()
        or (bond_close <= _EPS).any()
        or (stock_previous <= _EPS).any()
        or (stock_close <= _EPS).any()
    ):
        return None
    return bond_close / bond_previous - 1.0, stock_close / stock_previous - 1.0


def _empty_metrics(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _contract_term_revision_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_CONTRACT_SIGNALS)
    fields = ("cb_conv_price", "cb_put_price", "cb_call_price")
    names = _CONTRACT_SIGNALS[:3]
    for field, signal in zip(fields, names):
        values = _numeric(frame, field)
        if len(values) >= 2 and _last_n_are_consecutive(frame, 2):
            out[signal] = _pct_change(float(values[-1]), float(values[-2]))

    tail20 = _complete_numeric_tail(frame, fields, 21)
    if tail20 is not None and (tail20 > _EPS).all():
        changes = tail20[1:] / tail20[:-1] - 1.0
        events = np.abs(changes) > _EPS
        out["ctr_term_event_count20"] = float(np.mean(events.sum(axis=1)))
        out["ctr_term_corevision_rate20"] = float(np.mean(events.sum(axis=1) >= 2))
        current = changes[-1]
        moved = np.abs(current) > _EPS
        out["ctr_term_directional_coherence1"] = float(np.mean(np.sign(current[moved]))) if moved.any() else 0.0

    tail60 = _complete_numeric_tail(frame, fields, 61)
    if tail60 is not None and (tail60 > _EPS).all():
        events = np.any(np.abs(tail60[1:] / tail60[:-1] - 1.0) > _EPS, axis=1)
        event_positions = np.flatnonzero(events)
        out["ctr_term_revision_recency60"] = (
            float(1.0 / (1.0 + (len(events) - 1 - int(event_positions[-1]))))
            if len(event_positions)
            else 0.0
        )
    return out


def _barrier_geometry_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_BARRIER_SIGNALS)
    stock_close = _finite_scalar(_numeric(frame, "stock_close_price")[-1])
    put = _finite_scalar(_numeric(frame, "cb_put_price")[-1])
    call = _finite_scalar(_numeric(frame, "cb_call_price")[-1])
    if not (np.isfinite(stock_close) and stock_close > _EPS and np.isfinite(put) and put > _EPS and np.isfinite(call) and call > put):
        return out
    width = call - put
    lower_distance = stock_close - put
    upper_distance = call - stock_close
    out["barrier_center_position"] = float(lower_distance / width)
    out["barrier_width_to_price"] = float(width / stock_close)
    out["barrier_nearest_edge_norm"] = float(min(abs(lower_distance), abs(upper_distance)) / width)
    out["barrier_side_asymmetry"] = float((upper_distance - lower_distance) / width)
    trigger = _finite_scalar(_numeric(frame, "trigger_price_revise")[-1])
    if np.isfinite(trigger) and trigger > _EPS:
        out["barrier_revision_trigger_position"] = float((trigger - put) / width)
        out["barrier_revision_trigger_distance"] = float((trigger - stock_close) / width)
    return out


def _tracking_error_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_TRACKING_SIGNALS)
    columns = (
        "cb_prev_close_price",
        "cb_close_price",
        "stk_prev_close_price",
        "stk_close_price",
    )
    values = _complete_numeric_tail(frame, columns, 41)
    if values is None:
        return out
    cb_prev, cb_close, stk_prev, stk_close = values.T
    if (values <= _EPS).any():
        return out
    bond_returns = cb_close / cb_prev - 1.0
    stock_returns = stk_close / stk_prev - 1.0
    beta20 = _safe_beta(stock_returns[-21:-1], bond_returns[-21:-1])
    beta10 = _safe_beta(stock_returns[-10:], bond_returns[-10:])
    beta40 = _safe_beta(stock_returns[-40:], bond_returns[-40:])
    if not np.isfinite(beta20):
        return out
    residual = bond_returns[-20:] - beta20 * stock_returns[-20:]
    out["bstk_residual_last20"] = float(residual[-1])
    out["bstk_residual_vol20"] = _finite_scalar(np.std(residual, ddof=1))
    out["bstk_residual_autocorr20"] = _autocorr(residual)
    if np.isfinite(beta10) and np.isfinite(beta40):
        out["bstk_beta_shift10_40"] = float(beta10 - beta40)
    tail_stock = stock_returns[-20:]
    cocrash = tail_stock <= float(np.quantile(tail_stock, 0.2))
    if int(cocrash.sum()) >= 4:
        out["bstk_tail_cocrash_residual20"] = float(np.mean(residual[cocrash]))
    path = np.r_[0.0, np.cumsum(residual)]
    out["bstk_tracking_drawdown20"] = float(path[-1] - np.max(path))
    return out


def _liquidity_reallocation_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_LIQUIDITY_SIGNALS)
    columns = (
        "cb_amount",
        "stk_amount",
        "cb_deal",
        "stk_deal",
        "cb_prev_close_price",
        "cb_close_price",
        "stk_prev_close_price",
        "stk_close_price",
    )
    values = _complete_numeric_tail(frame, columns, 21)
    if values is None:
        return out
    cb_amount, stk_amount, cb_deal, stk_deal, cb_prev, cb_close, stk_prev, stk_close = values.T
    if (
        (cb_amount <= _EPS).any()
        or (stk_amount <= _EPS).any()
        or (cb_deal <= _EPS).any()
        or (stk_deal <= _EPS).any()
        or (cb_prev <= _EPS).any()
        or (cb_close <= _EPS).any()
        or (stk_prev <= _EPS).any()
        or (stk_close <= _EPS).any()
    ):
        return out
    amount_share = cb_amount / (cb_amount + stk_amount)
    deal_share = cb_deal / (cb_deal + stk_deal)
    out["lqr_amount_share_delta1"] = float(amount_share[-1] - amount_share[-2])
    out["lqr_deal_share_delta1"] = float(deal_share[-1] - deal_share[-2])
    out["lqr_amount_share_z20"] = _z_last_against_prior(amount_share)
    out["lqr_deal_share_z20"] = _z_last_against_prior(deal_share)
    out["lqr_amount_deal_divergence"] = float(amount_share[-1] - deal_share[-1])
    bond_return = cb_close[-1] / cb_prev[-1] - 1.0
    stock_return = stk_close[-1] / stk_prev[-1] - 1.0
    out["lqr_flow_return_wedge"] = float((amount_share[-1] - amount_share[-2]) * (bond_return - stock_return))
    out["lqr_amount_share_persistence20"] = _autocorr(amount_share[-20:])
    return out


def _rolling_realized_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(returns), np.nan, dtype="float64")
    for stop in range(window, len(returns) + 1):
        tail = returns[stop - window : stop]
        if np.isfinite(tail).all():
            value = float(np.std(tail, ddof=1))
            if np.isfinite(value) and value > _EPS:
                out[stop - 1] = value
    return out


def _stock_volatility_forecast_error_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_VOLATILITY_SIGNALS)
    columns = ("stock_volatility", "stk_prev_close_price", "stk_close_price")
    values = _complete_numeric_tail(frame, columns, 65)
    if values is None:
        return out
    implied, previous, close = values.T
    if (implied <= _EPS).any() or (previous <= _EPS).any() or (close <= _EPS).any():
        return out
    returns = close / previous - 1.0
    realized10 = _rolling_realized_volatility(returns, 10)
    realized20 = _rolling_realized_volatility(returns, 20)
    realized40 = _rolling_realized_volatility(returns, 40)
    if not (np.isfinite(realized10[-1]) and np.isfinite(realized20[-1]) and np.isfinite(realized40[-1])):
        return out
    error = implied / realized20
    error_tail = error[-21:]
    out["svfe_level_ratio20"] = float(error[-1])
    out["svfe_error_z20"] = _z_last_against_prior(error_tail)
    if np.isfinite(realized20[-2]) and realized20[-2] > _EPS:
        out["svfe_implied_change_minus_realized_change"] = float(
            (implied[-1] / implied[-2] - 1.0) - (realized20[-1] / realized20[-2] - 1.0)
        )
    out["svfe_abs_return_scale"] = float(abs(returns[-1]) / implied[-1])
    out["svfe_realized_short_long_gap"] = float(realized10[-1] / realized40[-1] - 1.0)
    out["svfe_error_autocorr20"] = _autocorr(error_tail[-20:])
    return out


def _binary_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
    accepted = np.isclose(values, 0.0, atol=_EPS) | np.isclose(values, 1.0, atol=_EPS)
    return np.where(accepted, values, np.nan)


def _progress(cumulative: np.ndarray, required: np.ndarray) -> np.ndarray:
    out = np.full(len(cumulative), np.nan, dtype="float64")
    valid = np.isfinite(cumulative) & np.isfinite(required) & (required > _EPS)
    out[valid] = cumulative[valid] / required[valid]
    return out


def _trigger_events(price: np.ndarray, revised_mode: np.ndarray) -> np.ndarray | None:
    if len(price) < 2 or not np.isfinite(price).all() or not np.isfinite(revised_mode).all() or (price <= _EPS).any():
        return None
    price_change = np.abs(price[1:] / price[:-1] - 1.0) > _EPS
    mode_change = np.abs(revised_mode[1:] - revised_mode[:-1]) > _EPS
    return price_change | mode_change


def _trigger_revision_phase_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_TRIGGER_SIGNALS)
    original_mode = _binary_array(frame, "trigger_is_price")
    revised_mode = _binary_array(frame, "trigger_is_price_revise")
    original_cum = _numeric(frame, "trigger_cum_days")
    original_required = _numeric(frame, "trigger_reach_days")
    revised_cum = _numeric(frame, "trigger_cum_days_revise")
    revised_required = _numeric(frame, "trigger_reach_days_revise")
    revised_price = _numeric(frame, "trigger_price_revise")
    activation = _numeric(frame, "in_trigger_process")
    original_progress = _progress(original_cum, original_required)
    revised_progress = _progress(revised_cum, revised_required)

    if np.isfinite(original_mode[-1]) and np.isfinite(revised_mode[-1]):
        out["trw_mode_wedge"] = float(revised_mode[-1] - original_mode[-1])
    if np.isfinite(original_required[-1]) and original_required[-1] > _EPS and np.isfinite(revised_required[-1]):
        out["trw_required_days_wedge"] = float(revised_required[-1] / original_required[-1] - 1.0)
    if np.isfinite(original_progress[-1]) and np.isfinite(revised_progress[-1]):
        out["trw_progress_wedge"] = float(revised_progress[-1] - original_progress[-1])
    if len(original_progress) >= 6 and _last_n_are_consecutive(frame, 6):
        old_tail = original_progress[-6:]
        new_tail = revised_progress[-6:]
        if np.isfinite(old_tail).all() and np.isfinite(new_tail).all():
            out["trw_progress_velocity_wedge5"] = float((new_tail[-1] - new_tail[0]) - (old_tail[-1] - old_tail[0]))

    if len(revised_price) >= 21 and _last_n_are_consecutive(frame, 21):
        events20 = _trigger_events(revised_price[-21:], revised_mode[-21:])
        if events20 is not None:
            out["trw_revision_event_rate20"] = float(np.mean(events20))
    recency = float("nan")
    if len(revised_price) >= 61 and _last_n_are_consecutive(frame, 61):
        events60 = _trigger_events(revised_price[-61:], revised_mode[-61:])
        if events60 is not None:
            positions = np.flatnonzero(events60)
            recency = float(1.0 / (1.0 + (len(events60) - 1 - int(positions[-1])))) if len(positions) else 0.0
            out["trw_revision_recency60"] = recency
    if np.isfinite(recency) and np.isfinite(activation[-1]):
        out["trw_post_revision_activation"] = float(float(activation[-1] > 0.0) * recency)
    return out


def _rating_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().upper().replace(" ", "")
    return _RATING_ORDINAL.get(text, float("nan"))


def _rating_array(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([_rating_value(value) for value in frame["rating"]], dtype="float64")


def _rating_migration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_RATING_SIGNALS)
    ratings = _rating_array(frame)
    if len(ratings) == 0 or not np.isfinite(ratings[-1]):
        return out
    out["rating_current_ordinal"] = float(ratings[-1])
    if len(ratings) >= 2 and np.isfinite(ratings[-2]) and _last_n_are_consecutive(frame, 2):
        out["rating_change1"] = float(ratings[-1] - ratings[-2])

    if len(ratings) >= 21 and np.isfinite(ratings[-21:]).all() and _last_n_are_consecutive(frame, 21):
        changes = np.diff(ratings[-21:])
        weights = np.exp(-np.arange(len(changes) - 1, -1, -1, dtype="float64") / 10.0)
        denominator = float(weights.sum())
        signed_decay = float(np.dot(changes, weights) / denominator)
        out["rating_signed_decay20"] = signed_decay
        out["rating_upgrade_decay20"] = float(np.dot(np.maximum(changes, 0.0), weights) / denominator)
        out["rating_downgrade_decay20"] = float(np.dot(np.maximum(-changes, 0.0), weights) / denominator)
        ytm = _numeric(frame, "ytm")
        if len(ytm) >= 2 and np.isfinite(ytm[-1]) and np.isfinite(ytm[-2]):
            out["rating_migration_ytm_wedge"] = float(signed_decay * (ytm[-1] - ytm[-2]))

    if len(ratings) >= 61 and np.isfinite(ratings[-61:]).all() and _last_n_are_consecutive(frame, 61):
        changes = np.diff(ratings[-61:])
        positions = np.flatnonzero(np.abs(changes) > _EPS)
        out["rating_event_recency60"] = (
            float(1.0 / (1.0 + (len(changes) - 1 - int(positions[-1]))))
            if len(positions)
            else 0.0
        )
    return out


_FAMILY_CALCULATORS: dict[str, Callable[..., dict[str, float]]] = {
    "contract_term_revision_vector": _contract_term_revision_metrics,
    "put_call_barrier_geometry": _barrier_geometry_metrics,
    "daily_bond_stock_tracking_error": _tracking_error_metrics,
    "bond_stock_liquidity_reallocation": _liquidity_reallocation_metrics,
    "stock_volatility_forecast_error": _stock_volatility_forecast_error_metrics,
    "trigger_revision_phase_wedge": _trigger_revision_phase_metrics,
    "rating_migration_decay": _rating_migration_metrics,
}


def _build_family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    out_index = _output_index(ctx)
    signals = _FAMILY_SIGNALS[family]
    out = pd.DataFrame(index=out_index, columns=signals, dtype="float64")
    if out_index.empty:
        return out
    score_date = _score_date_from_index(out_index)
    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_FAMILY_BASE_FIELDS[family],
        score_date=score_date,
    )
    if price.empty:
        return out
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    price_calendar = sorted(pd.Timestamp(day).normalize() for day in price["trade_date"].unique())
    price_session_index = {day: position for position, day in enumerate(price_calendar)}
    # Historical state is deliberately date-key inner joined to the
    # independently sourced price calendar.  This prevents a sparse base
    # sequence from measuring an event across an unknown number of sessions.
    base = base.merge(
        price.loc[:, ["trade_date", "code"]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
    base["__price_session_index"] = base["trade_date"].map(price_session_index)
    base_groups = {str(code): group for code, group in base.groupby("code", sort=False)}
    calculator = _FAMILY_CALCULATORS[family]

    for dt, raw_code in out_index:
        code = _canonical_market_code(pd.Series([raw_code])).iloc[0]
        if not code or code not in anchor_codes:
            continue
        history = base_groups.get(str(code))
        # The date comparison is the anti-staleness guard.  A row from an
        # older base partition must never be carried into the current score.
        if history is None or history.empty or pd.Timestamp(history["trade_date"].iloc[-1]).normalize() != anchor:
            continue
        metrics = calculator(history)
        for signal in signals:
            value = _finite_scalar(metrics.get(signal))
            if np.isfinite(value):
                out.at[(dt, raw_code), signal] = value
    return out.replace([np.inf, -np.inf], np.nan)


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    out_index = _output_index(ctx)
    score_date = _score_date_from_index(out_index) if not out_index.empty else pd.NaT
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _build_family_feature_frame(ctx, family)
    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyContractStockV1(Factor):
    """Research-only strict-prior daily contract/stock state kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        if params and params.get("signal"):
            return _requirements_for_family(_requested_entry(params).family)
        return _all_requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningDailyContractStockV1",
    "daily_contract_stock_catalog",
    "factor_mining_catalog",
]
