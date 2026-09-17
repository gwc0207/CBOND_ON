"""Research-only strict-PIT stock/bond joint-state factor catalogue.

This module deliberately remains import-only: it is not imported from
``defs.__init__`` and is not part of a factor, model, or live configuration.
It is a second-wave research catalogue for the IC mining workflow.

Every current-day input comes from the supplied T1430 bond/stock panels.  A
clean-direct panel can label a prior physical snapshot with the score-day
partition, so both panels are independently restricted to physical score-day
continuous-session observations at or before 14:29:00.  Stock mapping is derived
only from ``market_cbond.daily_base`` on the latest completed daily-price
session strictly before the score date.  The context-provided ``bond_stock_map``
is deliberately not used: its loader is permitted to resolve a score-day map,
which is not a sufficient T-1 mapping certificate for this research contract.

The families below are intentionally joint mechanisms, rather than a simple
bond-minus-stock return, scalar rescaling, or a parallel-window rewrite of an
existing single-asset factor.  Missing inputs remain NaN; no formula substitutes
zeros, reads files, opens labels, accesses a pool, or performs I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_joint_state_v1"
CATALOG_VERSION = "20260803_intraday_joint_state_v1"
_EPS = 1e-12
_CUTOFF = dt_time(14, 29)
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_PHASES: tuple[tuple[str, dt_time, dt_time, bool], ...] = (
    ("opening", dt_time(9, 30), dt_time(10, 0), False),
    ("morning", dt_time(10, 0), dt_time(11, 30), False),
    ("afternoon", dt_time(13, 0), dt_time(13, 30), False),
    ("tail", dt_time(13, 30), _CUTOFF, True),
)
_EARLY_START = dt_time(9, 30)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only signal instance."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_OPENING_GAP_SIGNALS = (
    "joint_gap_signed_alignment",
    "joint_stock_gap_bond_early_response",
    "joint_bond_gap_stock_early_response",
    "joint_gap_relative_absorption",
    "joint_gap_phase_pass_through_asymmetry",
    "joint_gap_tail_resolution_alignment",
)
_PHASE_VECTOR_SIGNALS = (
    "joint_phase_return_cosine",
    "joint_phase_centered_return_corr",
    "joint_phase_direction_agreement_share",
    "joint_phase_transition_agreement",
    "joint_phase_mismatch_energy",
    "joint_phase_tail_rotation_gap",
)
_QUOTE_TRADE_SIGNALS = (
    "joint_stock_mid_bond_last_tail_response",
    "joint_stock_last_bond_mid_tail_response",
    "joint_bond_mid_stock_last_tail_response",
    "joint_bond_last_stock_mid_tail_response",
    "joint_cross_quote_trade_dynamic_gap",
    "joint_cross_quote_trade_phase_shift_gap",
)
_STOCK_SHOCK_BOOK_SIGNALS = (
    "joint_stock_early_x_bond_imbalance",
    "joint_stock_early_x_bond_spread",
    "joint_stock_late_x_bond_imbalance_shift",
    "joint_stock_late_x_bond_spread_change",
    "joint_stock_full_x_bond_depth_recovery",
    "joint_stock_tail_x_bond_microprice_bias",
)
_BOOK_ELASTICITY_SIGNALS = (
    "joint_book_delta_imbalance_return_coupling_gap",
    "joint_book_delta_spread_return_coupling_gap",
    "joint_book_delta_depth_return_coupling_gap",
    "joint_book_delta_microprice_return_coupling_gap",
    "joint_book_tail_imbalance_coupling_gap",
    "joint_book_phase_coupling_shift_gap",
)
_BOOK_SYNCHRONY_SIGNALS = (
    "joint_book_imbalance_transition_alignment",
    "joint_book_spread_transition_alignment",
    "joint_book_depth_transition_alignment",
    "joint_book_microprice_transition_alignment",
    "joint_book_state_transition_coherence",
    "joint_book_transition_x_tail_return_gap",
)
_TAIL_COJUMP_SIGNALS = (
    "joint_tail_signed_cojump",
    "joint_tail_range_coexpansion",
    "joint_tail_efficiency_alignment",
    "joint_tail_jump_intensity_alignment",
    "joint_tail_terminal_location_coshock",
    "joint_tail_drawdown_containment",
)


_CATALOG = (
    _entries(
        "joint_opening_gap_transmission",
        _OPENING_GAP_SIGNALS,
        "Opening discontinuities and their early/tail resolution encode cross-asset transmission rather than a contemporaneous full-session return gap.",
    )
    + _entries(
        "joint_phase_vector_coherence",
        _PHASE_VECTOR_SIGNALS,
        "The agreement, rotation, and mismatch of four fixed intraday return phases describe joint price discovery beyond extrema timing or a single beta.",
    )
    + _entries(
        "joint_quote_trade_channel_transmission",
        _QUOTE_TRADE_SIGNALS,
        "Cross-asset transmission can differ when the initiating asset moves in midpoint quotes versus traded last prices.",
    )
    + _entries(
        "joint_stock_shock_bond_book_gate",
        _STOCK_SHOCK_BOOK_SIGNALS,
        "A stock shock has different bond-price-discovery implications under the bond's own displayed liquidity and pressure state.",
    )
    + _entries(
        "joint_cross_book_price_elasticity",
        _BOOK_ELASTICITY_SIGNALS,
        "The two assets can differ in how same-tick changes in book state couple to their own price changes, distinct from a static book-level gap.",
    )
    + _entries(
        "joint_cross_book_state_synchrony",
        _BOOK_SYNCHRONY_SIGNALS,
        "Joint early-to-late relocation of imbalance, spread, depth, and microprice measures shared versus disconnected liquidity regimes.",
    )
    + _entries(
        "joint_tail_cojump_containment",
        _TAIL_COJUMP_SIGNALS,
        "Tail co-jump intensity, path containment, and range co-expansion measure shared tail-risk structure rather than a tail beta residual.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "joint_opening_gap_transmission": _OPENING_GAP_SIGNALS,
    "joint_phase_vector_coherence": _PHASE_VECTOR_SIGNALS,
    "joint_quote_trade_channel_transmission": _QUOTE_TRADE_SIGNALS,
    "joint_stock_shock_bond_book_gate": _STOCK_SHOCK_BOOK_SIGNALS,
    "joint_cross_book_price_elasticity": _BOOK_ELASTICITY_SIGNALS,
    "joint_cross_book_state_synchrony": _BOOK_SYNCHRONY_SIGNALS,
    "joint_tail_cojump_containment": _TAIL_COJUMP_SIGNALS,
}


# This is deliberately explicit evidence for later research review.  The
# symbols b and s denote bond and T-1-mapped stock summaries; phase p has four
# fixed clock buckets; q denotes midpoint-quote and l denotes last-price.
FORMULAS: dict[str, str] = {
    "joint_gap_signed_alignment": "gap_b * gap_s",
    "joint_stock_gap_bond_early_response": "gap_s * early_return_b",
    "joint_bond_gap_stock_early_response": "gap_b * early_return_s",
    "joint_gap_relative_absorption": "(gap_b-gap_s) * (late_return_b-late_return_s)",
    "joint_gap_phase_pass_through_asymmetry": "gap_s*early_return_b - gap_b*early_return_s",
    "joint_gap_tail_resolution_alignment": "(gap_b*gap_s) * (late_return_b*late_return_s)",
    "joint_phase_return_cosine": "dot(p_b,p_s)/(||p_b||*||p_s||)",
    "joint_phase_centered_return_corr": "corr(p_b-mean(p_b), p_s-mean(p_s))",
    "joint_phase_direction_agreement_share": "mean(sign(p_b)==sign(p_s)) over nonzero phase pairs",
    "joint_phase_transition_agreement": "mean(sign(diff(p_b))*sign(diff(p_s))) over nonzero transitions",
    "joint_phase_mismatch_energy": "mean((p_b/||p_b|| - p_s/||p_s||)^2)",
    "joint_phase_tail_rotation_gap": "(p_b_tail-mean(p_b_pre_tail))-(p_s_tail-mean(p_s_pre_tail))",
    "joint_stock_mid_bond_last_tail_response": "q_s_opening * l_b_tail",
    "joint_stock_last_bond_mid_tail_response": "l_s_opening * q_b_tail",
    "joint_bond_mid_stock_last_tail_response": "q_b_opening * l_s_tail",
    "joint_bond_last_stock_mid_tail_response": "l_b_opening * q_s_tail",
    "joint_cross_quote_trade_dynamic_gap": "(q_b_full-l_b_full)-(q_s_full-l_s_full)",
    "joint_cross_quote_trade_phase_shift_gap": "((q_b_tail-l_b_tail)-(q_b_opening-l_b_opening))-((q_s_tail-l_s_tail)-(q_s_opening-l_s_opening))",
    "joint_stock_early_x_bond_imbalance": "early_return_s * mean_early(imbalance_b)",
    "joint_stock_early_x_bond_spread": "early_return_s * mean_early(spread_b)",
    "joint_stock_late_x_bond_imbalance_shift": "late_return_s * (mean_late(imbalance_b)-mean_early(imbalance_b))",
    "joint_stock_late_x_bond_spread_change": "late_return_s * (mean_late(spread_b)-mean_early(spread_b))",
    "joint_stock_full_x_bond_depth_recovery": "full_return_s * log(mean_late(depth_b)/mean_early(depth_b))",
    "joint_stock_tail_x_bond_microprice_bias": "late_return_s * mean_late(microprice_bias_b)",
    "joint_book_delta_imbalance_return_coupling_gap": "corr(delta imbalance_b, logret_b)-corr(delta imbalance_s, logret_s)",
    "joint_book_delta_spread_return_coupling_gap": "corr(delta spread_b, logret_b)-corr(delta spread_s, logret_s)",
    "joint_book_delta_depth_return_coupling_gap": "corr(delta log depth_b, logret_b)-corr(delta log depth_s, logret_s)",
    "joint_book_delta_microprice_return_coupling_gap": "corr(delta microprice_bias_b, logret_b)-corr(delta microprice_bias_s, logret_s)",
    "joint_book_tail_imbalance_coupling_gap": "corr_tail(delta imbalance_b,logret_b)-corr_tail(delta imbalance_s,logret_s)",
    "joint_book_phase_coupling_shift_gap": "(tail_minus_early_imbalance_coupling_b)-(tail_minus_early_imbalance_coupling_s)",
    "joint_book_imbalance_transition_alignment": "delta_early_to_late(imbalance_b)*delta_early_to_late(imbalance_s)",
    "joint_book_spread_transition_alignment": "delta_early_to_late(spread_b)*delta_early_to_late(spread_s)",
    "joint_book_depth_transition_alignment": "delta_early_to_late(log depth_b)*delta_early_to_late(log depth_s)",
    "joint_book_microprice_transition_alignment": "delta_early_to_late(microprice_bias_b)*delta_early_to_late(microprice_bias_s)",
    "joint_book_state_transition_coherence": "corr([delta imbalance,delta spread,delta logdepth,delta microbias]_b, same_s)",
    "joint_book_transition_x_tail_return_gap": "(delta imbalance_b-delta imbalance_s)*(late_return_b-late_return_s)",
    "joint_tail_signed_cojump": "late_return_b * late_return_s",
    "joint_tail_range_coexpansion": "tail_range_b * tail_range_s",
    "joint_tail_efficiency_alignment": "tail_efficiency_b * tail_efficiency_s",
    "joint_tail_jump_intensity_alignment": "tail_jump_share_b * tail_jump_share_s",
    "joint_tail_terminal_location_coshock": "(tail_location_b-tail_location_s)*(late_return_b*late_return_s)",
    "joint_tail_drawdown_containment": "tail_drawdown_b * tail_drawdown_s",
}


_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "pre_close",
    "open",
    "last",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
)
_EXCHANGE_ALIASES = {"XSHG": "SH", "SHSE": "SH", "XSHE": "SZ", "SZSE": "SZ", "BSE": "BJ", "BJSE": "BJ"}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def factor_mining_intraday_joint_state_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research catalogue in family-first order."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for the generic scratch expansion runner."""

    return factor_mining_intraday_joint_state_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing columns: {missing}")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)) or abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator) and numerator > 0.0 and denominator > 0.0):
        return float("nan")
    return float(np.log(numerator / denominator))


def _safe_corr(left: np.ndarray, right: np.ndarray, *, min_count: int = 5) -> float:
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


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize a bond code while preserving a validated exchange suffix."""

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
    """Normalize a daily-base stock mapping without guessing from panel rows."""

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
    """Panel codes are expected to be exchange-qualified; reject ambiguity."""

    return _canonical_market_code(pd.Series([value])).iloc[0]


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


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp, owner: str) -> pd.DataFrame:
    """Filter an indexed panel to physical score-day rows through 14:29:00."""

    checked = ensure_trade_time(panel)
    _require_columns(checked, _REQUIRED_PANEL_COLUMNS, owner=owner)
    frame = checked.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
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


def _strict_prior_daily(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read only declared prior rows from daily context; no score-day fallback."""

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
        examples = frame.loc[frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _tminus1_stock_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Return only mappings backed by the common latest completed price day.

    The global daily-price anchor is a market-session certificate.  A bond is
    usable only when both its daily-price row and daily-base mapping row exist
    on that exact prior session; an older mapping is intentionally missing.
    """

    price = _strict_prior_daily(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    base = _strict_prior_daily(
        ctx,
        source="market_cbond.daily_base",
        fields=("stock_code",),
        score_date=score_date,
    )
    if price.empty or base.empty:
        return {}
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    current = base.loc[(base["trade_date"] == anchor) & base["code"].astype(str).isin(anchor_codes), ["code", "stock_code"]].copy()
    if current.empty:
        return {}
    current["stock_code"] = current["stock_code"].map(_canonical_stock_code)
    current = current.loc[current["stock_code"] != ""].drop_duplicates(subset=["code"], keep="last")
    return {str(row.code): str(row.stock_code) for row in current.itertuples(index=False)}


def _phase_mask(frame: pd.DataFrame, start: dt_time, end: dt_time, *, include_end: bool) -> np.ndarray:
    clocks = pd.to_datetime(frame["trade_time"], errors="coerce").dt.time
    if include_end:
        return ((clocks >= start) & (clocks <= end)).to_numpy(dtype=bool)
    return ((clocks >= start) & (clocks < end)).to_numpy(dtype=bool)


def _window_mean(frame: pd.DataFrame, column: str, *, start: dt_time, end: dt_time, include_end: bool) -> float:
    mask = _phase_mask(frame, start, end, include_end=include_end)
    values = _numeric(frame.loc[mask], column)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) >= 2 else float("nan")


def _window_return(frame: pd.DataFrame, column: str, *, start: dt_time, end: dt_time, include_end: bool) -> float:
    mask = _phase_mask(frame, start, end, include_end=include_end)
    values = _numeric(frame.loc[mask], column)
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values) < 2:
        return float("nan")
    return _safe_div(float(values[-1] - values[0]), float(values[0]))


def _first_valid(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column)
    values = values[np.isfinite(values)]
    return float(values[0]) if len(values) else float("nan")


def _tail_metrics(frame: pd.DataFrame) -> dict[str, float]:
    mask = _phase_mask(frame, _LATE_START, _CUTOFF, include_end=True)
    values = _numeric(frame.loc[mask], "last")
    values = values[np.isfinite(values) & (values > 0.0)]
    out = {
        "tail_range": float("nan"),
        "tail_efficiency": float("nan"),
        "tail_jump_share": float("nan"),
        "tail_location": float("nan"),
        "tail_drawdown": float("nan"),
    }
    if len(values) < 3:
        return out
    low = float(np.min(values))
    high = float(np.max(values))
    path_range = high - low
    out["tail_range"] = _safe_div(path_range, float(values[0]))
    deltas = np.diff(values)
    path_length = float(np.abs(deltas).sum())
    out["tail_efficiency"] = _safe_div(abs(float(values[-1] - values[0])), path_length)
    logret = np.diff(np.log(values))
    abs_sum = float(np.abs(logret).sum())
    out["tail_jump_share"] = _safe_div(float(np.max(np.abs(logret))), abs_sum)
    out["tail_location"] = _safe_div(float(values[-1] - low), path_range)
    wealth = values / values[0]
    out["tail_drawdown"] = float(np.min(wealth / np.maximum.accumulate(wealth) - 1.0))
    return out


def _quote_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only internally valid L1 quote observations; never impute them."""

    ask = _numeric(frame, "ask_price1")
    bid = _numeric(frame, "bid_price1")
    ask_volume = _numeric(frame, "ask_volume1")
    bid_volume = _numeric(frame, "bid_volume1")
    last = _numeric(frame, "last")
    denom = ask_volume + bid_volume
    valid = (
        np.isfinite(ask)
        & np.isfinite(bid)
        & np.isfinite(ask_volume)
        & np.isfinite(bid_volume)
        & np.isfinite(last)
        & (ask > 0.0)
        & (bid > 0.0)
        & (ask >= bid)
        & (ask_volume >= 0.0)
        & (bid_volume >= 0.0)
        & (denom > _EPS)
        & (last > 0.0)
    )
    out = frame.loc[valid].copy()
    if out.empty:
        for column in ("mid", "spread", "imbalance", "depth", "microprice_bias"):
            out[column] = pd.Series(dtype="float64")
        return out
    ask = ask[valid]
    bid = bid[valid]
    ask_volume = ask_volume[valid]
    bid_volume = bid_volume[valid]
    denom = denom[valid]
    mid = (ask + bid) / 2.0
    microprice = (ask * bid_volume + bid * ask_volume) / denom
    out["mid"] = mid
    out["spread"] = (ask - bid) / mid
    out["imbalance"] = (bid_volume - ask_volume) / denom
    out["depth"] = denom
    out["microprice_bias"] = (microprice - mid) / mid
    return out


def _book_couplings(frame: pd.DataFrame, *, start: dt_time | None = None, end: dt_time | None = None) -> dict[str, float]:
    if start is not None and end is not None:
        frame = frame.loc[_phase_mask(frame, start, end, include_end=end == _CUTOFF)].copy()
    out = {"imb": float("nan"), "spread": float("nan"), "depth": float("nan"), "micro": float("nan")}
    if len(frame) < 6:
        return out
    last = _numeric(frame, "last")
    imbalance = _numeric(frame, "imbalance")
    spread = _numeric(frame, "spread")
    depth = _numeric(frame, "depth")
    micro = _numeric(frame, "microprice_bias")
    if not (np.isfinite(last).all() and (last > 0.0).all() and np.isfinite(depth).all() and (depth > 0.0).all()):
        return out
    returns = np.diff(np.log(last))
    out["imb"] = _safe_corr(np.diff(imbalance), returns)
    out["spread"] = _safe_corr(np.diff(spread), returns)
    out["depth"] = _safe_corr(np.diff(np.log(depth)), returns)
    out["micro"] = _safe_corr(np.diff(micro), returns)
    return out


def _asset_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Derive all allowed current-day state from one physical asset path."""

    out: dict[str, object] = {
        "gap": float("nan"),
        "early_return": float("nan"),
        "late_return": float("nan"),
        "full_return": float("nan"),
        "phase_returns": np.full(len(_PHASES), np.nan, dtype="float64"),
        "mid_full_return": float("nan"),
        "mid_phase_returns": np.full(len(_PHASES), np.nan, dtype="float64"),
        "imbalance_early": float("nan"),
        "imbalance_late": float("nan"),
        "spread_early": float("nan"),
        "spread_late": float("nan"),
        "depth_early": float("nan"),
        "depth_late": float("nan"),
        "micro_early": float("nan"),
        "micro_late": float("nan"),
        "coupling": {"imb": float("nan"), "spread": float("nan"), "depth": float("nan"), "micro": float("nan")},
        "early_coupling": {"imb": float("nan"), "spread": float("nan"), "depth": float("nan"), "micro": float("nan")},
        "tail_coupling": {"imb": float("nan"), "spread": float("nan"), "depth": float("nan"), "micro": float("nan")},
        "tail_range": float("nan"),
        "tail_efficiency": float("nan"),
        "tail_jump_share": float("nan"),
        "tail_location": float("nan"),
        "tail_drawdown": float("nan"),
    }
    if frame.empty:
        return out
    valid_last = np.isfinite(_numeric(frame, "last")) & (_numeric(frame, "last") > 0.0)
    last_frame = frame.loc[valid_last].copy()
    if len(last_frame) < 3:
        return out
    pre_close = _first_valid(last_frame, "pre_close")
    open_price = _first_valid(last_frame, "open")
    out["gap"] = _safe_div(open_price - pre_close, pre_close)
    last = _numeric(last_frame, "last")
    out["full_return"] = _safe_div(float(last[-1] - last[0]), float(last[0]))
    out["early_return"] = _window_return(last_frame, "last", start=_EARLY_START, end=_EARLY_END, include_end=False)
    out["late_return"] = _window_return(last_frame, "last", start=_LATE_START, end=_CUTOFF, include_end=True)
    out["phase_returns"] = np.asarray(
        [_window_return(last_frame, "last", start=start, end=end, include_end=include_end) for _, start, end, include_end in _PHASES],
        dtype="float64",
    )
    out.update(_tail_metrics(last_frame))

    quotes = _quote_frame(last_frame)
    if quotes.empty:
        return out
    out["mid_full_return"] = _window_return(quotes, "mid", start=_MORNING_START, end=_CUTOFF, include_end=True)
    out["mid_phase_returns"] = np.asarray(
        [_window_return(quotes, "mid", start=start, end=end, include_end=include_end) for _, start, end, include_end in _PHASES],
        dtype="float64",
    )
    out["imbalance_early"] = _window_mean(quotes, "imbalance", start=_EARLY_START, end=_EARLY_END, include_end=False)
    out["imbalance_late"] = _window_mean(quotes, "imbalance", start=_LATE_START, end=_CUTOFF, include_end=True)
    out["spread_early"] = _window_mean(quotes, "spread", start=_EARLY_START, end=_EARLY_END, include_end=False)
    out["spread_late"] = _window_mean(quotes, "spread", start=_LATE_START, end=_CUTOFF, include_end=True)
    out["depth_early"] = _window_mean(quotes, "depth", start=_EARLY_START, end=_EARLY_END, include_end=False)
    out["depth_late"] = _window_mean(quotes, "depth", start=_LATE_START, end=_CUTOFF, include_end=True)
    out["micro_early"] = _window_mean(quotes, "microprice_bias", start=_EARLY_START, end=_EARLY_END, include_end=False)
    out["micro_late"] = _window_mean(quotes, "microprice_bias", start=_LATE_START, end=_CUTOFF, include_end=True)
    out["coupling"] = _book_couplings(quotes)
    out["early_coupling"] = _book_couplings(quotes, start=_EARLY_START, end=_EARLY_END)
    out["tail_coupling"] = _book_couplings(quotes, start=_LATE_START, end=_CUTOFF)
    return out


def _number(state: dict[str, object], key: str) -> float:
    value = state.get(key)
    return float(value) if isinstance(value, (int, float, np.floating)) else float("nan")


def _array(state: dict[str, object], key: str) -> np.ndarray:
    value = state.get(key)
    return np.asarray(value, dtype="float64") if value is not None else np.array([], dtype="float64")


def _coupling(state: dict[str, object], key: str, field: str) -> float:
    raw = state.get(key)
    value = raw.get(field) if isinstance(raw, dict) else None
    return float(value) if isinstance(value, (int, float, np.floating)) else float("nan")


def _joint_metrics(bond: dict[str, object], stock: dict[str, object]) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    b_gap, s_gap = _number(bond, "gap"), _number(stock, "gap")
    b_early, s_early = _number(bond, "early_return"), _number(stock, "early_return")
    b_late, s_late = _number(bond, "late_return"), _number(stock, "late_return")
    b_full, s_full = _number(bond, "full_return"), _number(stock, "full_return")

    out.update(
        {
            "joint_gap_signed_alignment": b_gap * s_gap,
            "joint_stock_gap_bond_early_response": s_gap * b_early,
            "joint_bond_gap_stock_early_response": b_gap * s_early,
            "joint_gap_relative_absorption": (b_gap - s_gap) * (b_late - s_late),
            "joint_gap_phase_pass_through_asymmetry": s_gap * b_early - b_gap * s_early,
            "joint_gap_tail_resolution_alignment": (b_gap * s_gap) * (b_late * s_late),
        }
    )

    b_phase, s_phase = _array(bond, "phase_returns"), _array(stock, "phase_returns")
    if len(b_phase) == len(_PHASES) and len(s_phase) == len(_PHASES) and np.isfinite(b_phase).all() and np.isfinite(s_phase).all():
        b_norm = float(np.linalg.norm(b_phase))
        s_norm = float(np.linalg.norm(s_phase))
        out["joint_phase_return_cosine"] = _safe_div(float(np.dot(b_phase, s_phase)), b_norm * s_norm)
        out["joint_phase_centered_return_corr"] = _safe_corr(b_phase - float(np.mean(b_phase)), s_phase - float(np.mean(s_phase)), min_count=3)
        nonzero = (np.sign(b_phase) != 0.0) & (np.sign(s_phase) != 0.0)
        if int(nonzero.sum()) >= 3:
            out["joint_phase_direction_agreement_share"] = float(np.mean(np.sign(b_phase[nonzero]) == np.sign(s_phase[nonzero])))
        b_delta, s_delta = np.diff(b_phase), np.diff(s_phase)
        nonzero_delta = (np.sign(b_delta) != 0.0) & (np.sign(s_delta) != 0.0)
        if int(nonzero_delta.sum()) >= 2:
            out["joint_phase_transition_agreement"] = float(np.mean(np.sign(b_delta[nonzero_delta]) * np.sign(s_delta[nonzero_delta])))
        if b_norm > _EPS and s_norm > _EPS:
            out["joint_phase_mismatch_energy"] = float(np.mean(np.square(b_phase / b_norm - s_phase / s_norm)))
        out["joint_phase_tail_rotation_gap"] = float((b_phase[-1] - np.mean(b_phase[:-1])) - (s_phase[-1] - np.mean(s_phase[:-1])))

    b_mid_phase, s_mid_phase = _array(bond, "mid_phase_returns"), _array(stock, "mid_phase_returns")
    if len(b_mid_phase) == len(_PHASES) and len(s_mid_phase) == len(_PHASES):
        b_mid_full, s_mid_full = _number(bond, "mid_full_return"), _number(stock, "mid_full_return")
        out.update(
            {
                "joint_stock_mid_bond_last_tail_response": s_mid_phase[0] * b_phase[-1] if len(b_phase) else float("nan"),
                "joint_stock_last_bond_mid_tail_response": s_phase[0] * b_mid_phase[-1] if len(s_phase) else float("nan"),
                "joint_bond_mid_stock_last_tail_response": b_mid_phase[0] * s_phase[-1] if len(s_phase) else float("nan"),
                "joint_bond_last_stock_mid_tail_response": b_phase[0] * s_mid_phase[-1] if len(b_phase) else float("nan"),
                "joint_cross_quote_trade_dynamic_gap": (b_mid_full - b_full) - (s_mid_full - s_full),
                "joint_cross_quote_trade_phase_shift_gap": ((b_mid_phase[-1] - b_phase[-1]) - (b_mid_phase[0] - b_phase[0])) - ((s_mid_phase[-1] - s_phase[-1]) - (s_mid_phase[0] - s_phase[0])),
            }
        )

    b_imb_early, b_imb_late = _number(bond, "imbalance_early"), _number(bond, "imbalance_late")
    b_spread_early, b_spread_late = _number(bond, "spread_early"), _number(bond, "spread_late")
    b_depth_early, b_depth_late = _number(bond, "depth_early"), _number(bond, "depth_late")
    b_micro_late = _number(bond, "micro_late")
    out.update(
        {
            "joint_stock_early_x_bond_imbalance": s_early * b_imb_early,
            "joint_stock_early_x_bond_spread": s_early * b_spread_early,
            "joint_stock_late_x_bond_imbalance_shift": s_late * (b_imb_late - b_imb_early),
            "joint_stock_late_x_bond_spread_change": s_late * (b_spread_late - b_spread_early),
            "joint_stock_full_x_bond_depth_recovery": s_full * _safe_log_ratio(b_depth_late, b_depth_early),
            "joint_stock_tail_x_bond_microprice_bias": s_late * b_micro_late,
        }
    )

    for signal, field in (
        ("joint_book_delta_imbalance_return_coupling_gap", "imb"),
        ("joint_book_delta_spread_return_coupling_gap", "spread"),
        ("joint_book_delta_depth_return_coupling_gap", "depth"),
        ("joint_book_delta_microprice_return_coupling_gap", "micro"),
    ):
        out[signal] = _coupling(bond, "coupling", field) - _coupling(stock, "coupling", field)
    out["joint_book_tail_imbalance_coupling_gap"] = _coupling(bond, "tail_coupling", "imb") - _coupling(stock, "tail_coupling", "imb")
    out["joint_book_phase_coupling_shift_gap"] = (
        _coupling(bond, "tail_coupling", "imb")
        - _coupling(bond, "early_coupling", "imb")
        - _coupling(stock, "tail_coupling", "imb")
        + _coupling(stock, "early_coupling", "imb")
    )

    b_micro_early = _number(bond, "micro_early")
    s_imb_early, s_imb_late = _number(stock, "imbalance_early"), _number(stock, "imbalance_late")
    s_spread_early, s_spread_late = _number(stock, "spread_early"), _number(stock, "spread_late")
    s_depth_early, s_depth_late = _number(stock, "depth_early"), _number(stock, "depth_late")
    s_micro_early, s_micro_late = _number(stock, "micro_early"), _number(stock, "micro_late")
    b_transition = np.array(
        [
            b_imb_late - b_imb_early,
            b_spread_late - b_spread_early,
            _safe_log_ratio(b_depth_late, b_depth_early),
            b_micro_late - b_micro_early,
        ],
        dtype="float64",
    )
    s_transition = np.array(
        [
            s_imb_late - s_imb_early,
            s_spread_late - s_spread_early,
            _safe_log_ratio(s_depth_late, s_depth_early),
            s_micro_late - s_micro_early,
        ],
        dtype="float64",
    )
    out.update(
        {
            "joint_book_imbalance_transition_alignment": b_transition[0] * s_transition[0],
            "joint_book_spread_transition_alignment": b_transition[1] * s_transition[1],
            "joint_book_depth_transition_alignment": b_transition[2] * s_transition[2],
            "joint_book_microprice_transition_alignment": b_transition[3] * s_transition[3],
            "joint_book_state_transition_coherence": _safe_corr(b_transition, s_transition, min_count=3),
            "joint_book_transition_x_tail_return_gap": (b_transition[0] - s_transition[0]) * (b_late - s_late),
        }
    )

    out.update(
        {
            "joint_tail_signed_cojump": b_late * s_late,
            "joint_tail_range_coexpansion": _number(bond, "tail_range") * _number(stock, "tail_range"),
            "joint_tail_efficiency_alignment": _number(bond, "tail_efficiency") * _number(stock, "tail_efficiency"),
            "joint_tail_jump_intensity_alignment": _number(bond, "tail_jump_share") * _number(stock, "tail_jump_share"),
            "joint_tail_terminal_location_coshock": (_number(bond, "tail_location") - _number(stock, "tail_location")) * (b_late * s_late),
            "joint_tail_drawdown_containment": _number(bond, "tail_drawdown") * _number(stock, "tail_drawdown"),
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
    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        if ctx.stock_panel is None or ctx.stock_panel.empty:
            raise RuntimeError(f"{KERNEL_NAME} requires a non-empty stock_panel")
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date, owner="bond_panel")
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date, owner="stock_panel")
        mapping = _tminus1_stock_mapping(ctx, score_date=score_date)
        bond_groups = {
            _panel_code(code): group
            for code, group in bond_frame.groupby("code", sort=False)
            if _panel_code(code)
        }
        stock_groups = {
            _panel_code(code): group
            for code, group in stock_frame.groupby("code", sort=False)
            if _panel_code(code)
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            code = _panel_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            stock_code = mapping.get(code, "")
            bond_group = bond_groups.get(code)
            stock_group = stock_groups.get(stock_code)
            if bond_group is not None and stock_group is not None:
                row.update(_joint_metrics(_asset_summary(bond_group), _asset_summary(stock_group)))
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
class FactorMiningIntradayJointStateV1(Factor):
    """Research-only T1430 stock/bond joint price-discovery catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    # Mapping uses the explicitly strict T-1 daily-base path above.  Do not
    # request ctx.bond_stock_map, which may be resolved on the score date.
    requires_bond_stock_map = False

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), 10),
            DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", "stock_code"), 10),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
