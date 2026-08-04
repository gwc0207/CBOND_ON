"""Research-only strict-T-1 daily state-transition factor catalogue.

This second-wave catalogue intentionally works with *changes of state* that
were not covered by the existing daily price, valuation, liquidity, contract
revision, or intraday-path catalogues:

* changes in reported convertible-bond outstanding supply;
* the lifecycle after a bond enters the redemption-trigger process;
* prior-close adjustment / corporate-action discontinuities; and
* abnormal bond-versus-stock quantity activity after each leg is normalised
  by its own history.

It is deliberately import-only and research-only.  The kernel consumes only
declared ``FactorComputeContext.daily_data`` tables, reads no files or data
stores itself, and cannot access labels, pools, scores, backtests, or live
artifacts.  Each source is strictly filtered to ``trade_date < score_date``.
``daily_price`` is retained as an independent T-1 calendar anchor: a stale
``daily_base`` row is never carried forward to manufacture a state value.

The source audit that led to this module rejected ``trigger_process`` and
``trigger_process_revise``: both serialisations are exactly equivalent to the
already available cumulative-days / required-days pairs.  It also rejected
``trigger_type`` and trigger-date fields because they do not have enough
historical cross-sectional coverage for the fixed 381-day screen.  Neither
appears in this module's requirements.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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


KERNEL_NAME = "factor_mining_daily_state_events_v1"
CATALOG_VERSION = "20260803_daily_state_events_v1_r2"
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


_SUPPLY_SIGNALS = (
    "supply_log_change1",
    "supply_shrink_log1",
    "supply_event_recency60",
    "supply_event_rate20",
    "supply_cumulative_log_change20",
    "supply_shrink_liquidity_impulse1",
)
_ACTIVATION_SIGNALS = (
    "call_activation_flag",
    "call_active_age",
    "call_activation_recency60",
    "call_active_revised_progress",
    "call_active_revised_progress_velocity5",
    "call_active_revised_required_days",
)
_ADJUSTMENT_SIGNALS = (
    "adjustment_net_log_shift",
    "adjustment_abs_imbalance",
    "adjustment_cross_asset_gap",
    "adjustment_any_magnitude",
    "adjustment_event_recency60",
    "adjustment_event_rate20",
)
_QUANTITY_SIGNALS = (
    "quantity_relative_volume_surprise20",
    "quantity_relative_trade_size_surprise20",
    "quantity_relative_notional_per_unit_surprise20",
    "quantity_volume_coupling20",
    "quantity_volume_lead_lag20",
    "quantity_imbalance_persistence20",
)


_CATALOG = (
    _entries(
        "capital_supply_transition",
        _SUPPLY_SIGNALS,
        "T-1 reported outstanding-size changes identify discrete supply retirement or expansion events, distinct from a static float level.",
    )
    + _entries(
        "call_activation_lifecycle",
        _ACTIVATION_SIGNALS,
        "The age, recency, and revised-contract progress after activation are distinct from a raw trigger-revision wedge.",
    )
    + _entries(
        "prior_close_adjustment_discontinuity",
        _ADJUSTMENT_SIGNALS,
        "T-1 cross-asset adjustment events expose already-published corporate-action discontinuities rather than an intraday return path.",
    )
    + _entries(
        "bond_stock_quantity_regime",
        _QUANTITY_SIGNALS,
        "Bond and stock quantity activity is normalised within each leg before comparison, so it is distinct from raw amount/deal-share reallocation.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "capital_supply_transition": _SUPPLY_SIGNALS,
    "call_activation_lifecycle": _ACTIVATION_SIGNALS,
    "prior_close_adjustment_discontinuity": _ADJUSTMENT_SIGNALS,
    "bond_stock_quantity_regime": _QUANTITY_SIGNALS,
}

# ``daily_price`` is retained for every family as the independent T-1 market
# calendar anchor.  It is a source requirement, not an implicit file read.
_FAMILY_PRICE_FIELDS: dict[str, tuple[str, ...]] = {
    "capital_supply_transition": ("close_price",),
    "call_activation_lifecycle": ("close_price",),
    "prior_close_adjustment_discontinuity": (
        "close_price",
        "prev_close_price",
        "act_prev_close_price",
    ),
    "bond_stock_quantity_regime": ("close_price",),
}
_FAMILY_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "capital_supply_transition": ("remain_size", "cb_amount"),
    "call_activation_lifecycle": (
        "in_trigger_process",
        "trigger_cum_days_revise",
        "trigger_reach_days_revise",
    ),
    "prior_close_adjustment_discontinuity": (
        "stk_prev_close_price",
        "stk_act_prev_close_price",
    ),
    "bond_stock_quantity_regime": (
        "cb_volume",
        "stk_volume",
        "cb_amount",
        "stk_amount",
        "cb_deal",
        "stk_deal",
    ),
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


def daily_state_events_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable, family-first research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return daily_state_events_catalog()


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
    exchanges = exchanges if exchanges is not None else pd.Series("", index=values.index)

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
        [_one(value, exchange) for value, exchange in zip(values, exchanges)],
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
    """Use only output keys from the panel, never its data values."""

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
    """Read only declared context rows and reject score-day/future observations."""

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


def _last_n_are_consecutive(frame: pd.DataFrame, count: int) -> bool:
    if len(frame) < count:
        return False
    if "__price_session_index" not in frame.columns:
        return True
    positions = pd.to_numeric(frame.tail(count)["__price_session_index"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _complete_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count or not _last_n_are_consecutive(frame, count):
        return None
    values = np.column_stack(
        [pd.to_numeric(frame.tail(count)[column], errors="coerce").to_numpy(dtype="float64") for column in columns]
    )
    return values if np.isfinite(values).all() else None


def _safe_log_ratio(current: float, prior: float) -> float:
    if not (np.isfinite(current) and np.isfinite(prior) and current > _EPS and prior > _EPS):
        return float("nan")
    return float(np.log(current / prior))


def _z_last_against_prior(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    std = float(np.std(prior, ddof=1))
    if not np.isfinite(std) or std <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / std)


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(left) != len(right) or not (np.isfinite(left).all() and np.isfinite(right).all()):
        return float("nan")
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _empty_metrics(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _supply_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_SUPPLY_SIGNALS)
    last2 = _complete_tail(frame, ("remain_size", "cb_amount"), 2)
    if last2 is not None:
        prior_size, latest_size = last2[:, 0]
        change = _safe_log_ratio(float(latest_size), float(prior_size))
        if np.isfinite(change):
            out["supply_log_change1"] = change
            out["supply_shrink_log1"] = float(max(-change, 0.0))
            amount = float(last2[-1, 1])
            if amount > _EPS:
                # Zero denotes a genuine non-shrink state, not imputed input.
                out["supply_shrink_liquidity_impulse1"] = float(max(-change, 0.0) * np.log1p(amount / latest_size))

    tail20 = _complete_tail(frame, ("remain_size",), 21)
    if tail20 is not None and (tail20[:, 0] > _EPS).all():
        changes20 = np.diff(np.log(tail20[:, 0]))
        events20 = np.abs(changes20) > _EPS
        out["supply_event_rate20"] = float(np.mean(events20))
        out["supply_cumulative_log_change20"] = float(np.log(tail20[-1, 0] / tail20[0, 0]))

    tail60 = _complete_tail(frame, ("remain_size",), 61)
    if tail60 is not None and (tail60[:, 0] > _EPS).all():
        events60 = np.abs(np.diff(np.log(tail60[:, 0]))) > _EPS
        positions = np.flatnonzero(events60)
        out["supply_event_recency60"] = (
            float(1.0 / (1.0 + (len(events60) - 1 - int(positions[-1])))) if len(positions) else 0.0
        )
    return out


def _activation_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_ACTIVATION_SIGNALS)
    state = pd.to_numeric(frame["in_trigger_process"], errors="coerce").to_numpy(dtype="float64")
    if not len(state) or not np.isfinite(state[-1]):
        return out
    active = np.where(np.isfinite(state), (state > 0.0).astype("float64"), np.nan)
    is_active = bool(active[-1] > 0.0)
    # These zeros encode the observed inactive state, never a missing input.
    out["call_activation_flag"] = float(is_active)
    if not is_active:
        out["call_active_age"] = 0.0
        out["call_active_revised_progress"] = 0.0
        out["call_active_revised_progress_velocity5"] = 0.0
        out["call_active_revised_required_days"] = 0.0
    else:
        age = 0
        positions = pd.to_numeric(frame["__price_session_index"], errors="coerce").to_numpy(dtype="float64") if "__price_session_index" in frame.columns else None
        prior_position: float | None = None
        for reverse_index, value in enumerate(active[::-1]):
            if not np.isfinite(value) or value <= 0.0:
                break
            if positions is not None:
                position = positions[len(positions) - 1 - reverse_index]
                if not np.isfinite(position) or (prior_position is not None and prior_position - position != 1.0):
                    break
                prior_position = position
            age += 1
        out["call_active_age"] = float(age)
        last = _complete_tail(frame, ("trigger_cum_days_revise", "trigger_reach_days_revise"), 1)
        if last is not None and last[-1, 1] > _EPS:
            out["call_active_revised_progress"] = float(last[-1, 0] / last[-1, 1])
            out["call_active_revised_required_days"] = float(last[-1, 1])

        tail6 = _complete_tail(
            frame,
            ("in_trigger_process", "trigger_cum_days_revise", "trigger_reach_days_revise"),
            6,
        )
        if tail6 is not None and np.all(tail6[:, 0] > 0.0) and np.all(tail6[:, 2] > _EPS):
            progress = tail6[:, 1] / tail6[:, 2]
            out["call_active_revised_progress_velocity5"] = float(progress[-1] - progress[0])

    tail60 = _complete_tail(frame, ("in_trigger_process",), 61)
    if tail60 is not None:
        active60 = tail60[:, 0] > 0.0
        starts = active60[1:] & ~active60[:-1]
        if active60[0]:
            starts = np.r_[True, starts]
        else:
            starts = np.r_[False, starts]
        positions = np.flatnonzero(starts)
        out["call_activation_recency60"] = (
            float(1.0 / (1.0 + (len(active60) - 1 - int(positions[-1])))) if len(positions) else 0.0
        )
    return out


def _adjustment_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_ADJUSTMENT_SIGNALS)
    latest = _complete_tail(
        frame,
        ("prev_close_price", "act_prev_close_price", "stk_prev_close_price", "stk_act_prev_close_price"),
        1,
    )
    if latest is not None:
        cb = _safe_log_ratio(float(latest[-1, 1]), float(latest[-1, 0]))
        stock = _safe_log_ratio(float(latest[-1, 3]), float(latest[-1, 2]))
        if np.isfinite(cb) and np.isfinite(stock):
            # The individual bond- and stock-leg events are too sparse for the
            # 250-day IC-validity gate.  These cross-asset combinations remain
            # non-constant whenever either observed leg has an event.
            out["adjustment_net_log_shift"] = float(cb + stock)
            out["adjustment_abs_imbalance"] = float(abs(cb) - abs(stock))
            out["adjustment_cross_asset_gap"] = float(cb - stock)
            out["adjustment_any_magnitude"] = float(abs(cb) + abs(stock))

    fields = ("prev_close_price", "act_prev_close_price", "stk_prev_close_price", "stk_act_prev_close_price")
    tail20 = _complete_tail(frame, fields, 21)
    if tail20 is not None and (tail20 > _EPS).all():
        cb20 = np.log(tail20[:, 1] / tail20[:, 0])
        stock20 = np.log(tail20[:, 3] / tail20[:, 2])
        out["adjustment_event_rate20"] = float(np.mean((np.abs(cb20[1:]) + np.abs(stock20[1:])) > _EPS))

    tail60 = _complete_tail(frame, fields, 61)
    if tail60 is not None and (tail60 > _EPS).all():
        cb60 = np.log(tail60[:, 1] / tail60[:, 0])
        stock60 = np.log(tail60[:, 3] / tail60[:, 2])
        events60 = (np.abs(cb60[1:]) + np.abs(stock60[1:])) > _EPS
        positions = np.flatnonzero(events60)
        out["adjustment_event_recency60"] = (
            float(1.0 / (1.0 + (len(events60) - 1 - int(positions[-1])))) if len(positions) else 0.0
        )
    return out


def _quantity_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _empty_metrics(_QUANTITY_SIGNALS)
    fields = ("cb_volume", "stk_volume", "cb_amount", "stk_amount", "cb_deal", "stk_deal")
    tail = _complete_tail(frame, fields, 21)
    if tail is None or (tail <= _EPS).any():
        return out
    cb_volume, stock_volume, cb_amount, stock_amount, cb_deal, stock_deal = tail.T
    log_cb_volume = np.log(cb_volume)
    log_stock_volume = np.log(stock_volume)
    log_cb_trade_size = np.log(cb_volume / cb_deal)
    log_stock_trade_size = np.log(stock_volume / stock_deal)
    log_cb_notional_unit = np.log(cb_amount / cb_volume)
    log_stock_notional_unit = np.log(stock_amount / stock_volume)
    cb_volume_z = _z_last_against_prior(log_cb_volume)
    stock_volume_z = _z_last_against_prior(log_stock_volume)
    cb_trade_size_z = _z_last_against_prior(log_cb_trade_size)
    stock_trade_size_z = _z_last_against_prior(log_stock_trade_size)
    cb_notional_z = _z_last_against_prior(log_cb_notional_unit)
    stock_notional_z = _z_last_against_prior(log_stock_notional_unit)
    if np.isfinite(cb_volume_z) and np.isfinite(stock_volume_z):
        out["quantity_relative_volume_surprise20"] = float(cb_volume_z - stock_volume_z)
    if np.isfinite(cb_trade_size_z) and np.isfinite(stock_trade_size_z):
        out["quantity_relative_trade_size_surprise20"] = float(cb_trade_size_z - stock_trade_size_z)
    if np.isfinite(cb_notional_z) and np.isfinite(stock_notional_z):
        out["quantity_relative_notional_per_unit_surprise20"] = float(cb_notional_z - stock_notional_z)
    cb_flow = np.diff(log_cb_volume)
    stock_flow = np.diff(log_stock_volume)
    out["quantity_volume_coupling20"] = _safe_corr(cb_flow, stock_flow)
    out["quantity_volume_lead_lag20"] = _safe_corr(cb_flow[1:], stock_flow[:-1])
    imbalance = log_cb_volume - log_stock_volume
    out["quantity_imbalance_persistence20"] = _safe_corr(imbalance[:-1], imbalance[1:])
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "capital_supply_transition": _supply_metrics,
    "call_activation_lifecycle": _activation_metrics,
    "prior_close_adjustment_discontinuity": _adjustment_metrics,
    "bond_stock_quantity_regime": _quantity_metrics,
}


def _build_family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    out_index = _output_index(ctx)
    signals = _FAMILY_SIGNALS[family]
    out = pd.DataFrame(index=out_index, columns=signals, dtype="float64")
    if out_index.empty:
        return out
    score_date = _score_date_from_panel(ctx.panel)
    if score_date is None:
        return out
    price_fields = _FAMILY_PRICE_FIELDS[family]
    base_fields = _FAMILY_BASE_FIELDS[family]
    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=price_fields,
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=base_fields,
        score_date=score_date,
    )
    if price.empty or base.empty:
        return out
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    calendar = sorted(pd.Timestamp(day).normalize() for day in price["trade_date"].unique())
    session_index = {day: position for position, day in enumerate(calendar)}
    price_values = price.loc[:, ["trade_date", "code", *price_fields]]
    history = base.merge(
        price_values,
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
    history["__price_session_index"] = history["trade_date"].map(session_index)
    groups = {str(code): group for code, group in history.groupby("code", sort=False)}
    calculator = _FAMILY_CALCULATORS[family]
    for dt, raw_code in out_index:
        code = _canonical_market_code(pd.Series([raw_code]), pd.Series([""])).iloc[0]
        if not code or code not in anchor_codes:
            continue
        group = groups.get(str(code))
        # A prior-day base row may never be treated as the current T-1 state.
        if group is None or group.empty or pd.Timestamp(group["trade_date"].iloc[-1]).normalize() != anchor:
            continue
        metrics = calculator(group)
        for signal in signals:
            value = metrics.get(signal)
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                out.at[(dt, raw_code), signal] = numeric
    return out.replace([np.inf, -np.inf], np.nan)


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_SIGNALS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _build_family_feature_frame(ctx, family)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyStateEventsV1(Factor):
    """Research-only strict-T-1 state-event candidate kernel."""

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
    "FactorMiningDailyStateEventsV1",
    "daily_state_events_catalog",
    "factor_mining_catalog",
]
