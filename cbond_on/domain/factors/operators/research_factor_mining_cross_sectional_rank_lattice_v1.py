"""Research-only strict-T-1 cross-sectional rank-lattice factors.

Each candidate is a non-linear interaction of visible, completed T-1 daily
state ranks.  The module never sees a label, pool, mask, score, PnL, file, or
database.  ``daily_price`` supplies the independent trading-calendar anchor;
``daily_base`` states must match that exact latest strict-prior date, otherwise
the affected instrument fails closed to ``NaN``.

The catalogue deliberately uses rank *interactions* rather than a monotonic
re-expression of an existing raw state.  That gives the later fixed-universe
screen a genuine opportunity to find low-redundancy economic states, while
the final correlation gates remain the sole authority for admission.
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


KERNEL_NAME = "factor_mining_cross_sectional_rank_lattice_v1"
CATALOG_VERSION = "20260803_cross_sectional_rank_lattice_v1"
_LOOKBACK_DAYS = 8
_MIN_CROSS_SECTION = 30
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
    """One auditable research-only cross-sectional interaction candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_OPTIONALITY_SIGNALS = (
    "csl_option_moneyness_stockvol_lattice",
    "csl_option_premium_duration_lattice",
    "csl_option_moneyness_premium_vol_wedge",
)
_FLOOR_SIGNALS = (
    "csl_floor_debtgap_yield_lattice",
    "csl_floor_redemption_distance_lattice",
    "csl_floor_credit_redemption_wedge",
)
_LIQUIDITY_SIGNALS = (
    "csl_liquidity_turnover_size_lattice",
    "csl_liquidity_reallocation_stockvol_lattice",
    "csl_liquidity_trade_size_flow_wedge",
)
_BARRIER_SIGNALS = (
    "csl_barrier_call_put_curvature_lattice",
    "csl_barrier_trigger_premium_wedge",
    "csl_barrier_call_put_moneyness_skew",
)

_CATALOG = (
    _entries(
        "cross_sectional_optionality_rank_lattice",
        _OPTIONALITY_SIGNALS,
        "The joint T-1 rank state of conversion moneyness, premium, duration, and stock volatility can be nonlinear even when each marginal state is already known.",
    )
    + _entries(
        "cross_sectional_floor_credit_rank_lattice",
        _FLOOR_SIGNALS,
        "The joint T-1 rank state of debt-premium gap, redemption distance, yield, and redemption premium can distinguish floor-dominated credit configurations.",
    )
    + _entries(
        "cross_sectional_liquidity_capacity_rank_lattice",
        _LIQUIDITY_SIGNALS,
        "Turnover, float, paired bond-stock flow, trade-size, and stock-volatility ranks define non-linear capacity states rather than a raw liquidity ratio.",
    )
    + _entries(
        "cross_sectional_barrier_geometry_rank_lattice",
        _BARRIER_SIGNALS,
        "The relative ordering of T-1 conversion, put, call, and revised-trigger barriers can encode option geometry not present in a one-sided distance.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "cross_sectional_optionality_rank_lattice": _OPTIONALITY_SIGNALS,
    "cross_sectional_floor_credit_rank_lattice": _FLOOR_SIGNALS,
    "cross_sectional_liquidity_capacity_rank_lattice": _LIQUIDITY_SIGNALS,
    "cross_sectional_barrier_geometry_rank_lattice": _BARRIER_SIGNALS,
}

_FAMILY_BASE_FIELDS = {
    "cross_sectional_optionality_rank_lattice": (
        "bond_prem_ratio",
        "conv_value",
        "stock_volatility",
        "duration",
    ),
    "cross_sectional_floor_credit_rank_lattice": (
        "debt_puredebt_ratio",
        "puredebt_prem_ratio",
        "pure_redemption_value",
        "redemption_prem_ratio",
        "ytm",
    ),
    "cross_sectional_liquidity_capacity_rank_lattice": (
        "turnover_rate",
        "remain_size",
        "cb_amount",
        "stk_amount",
        "cb_deal",
        "stock_volatility",
    ),
    "cross_sectional_barrier_geometry_rank_lattice": (
        "cb_conv_price",
        "cb_put_price",
        "cb_call_price",
        "trigger_price_revise",
        "stock_close_price",
        "bond_prem_ratio",
    ),
}

FORMULAS = {
    "csl_option_moneyness_stockvol_lattice": "rank(log(conv_value / bond_close))_c * rank(log(stock_volatility))_c.",
    "csl_option_premium_duration_lattice": "rank(bond_prem_ratio)_c * rank(duration)_c.",
    "csl_option_moneyness_premium_vol_wedge": "rank(log(conv_value / bond_close))_c * (rank(bond_prem_ratio)_c - rank(log(stock_volatility))_c).",
    "csl_floor_debtgap_yield_lattice": "rank(debt_puredebt_ratio - puredebt_prem_ratio)_c * rank(ytm)_c.",
    "csl_floor_redemption_distance_lattice": "rank(log(pure_redemption_value / bond_close))_c * rank(redemption_prem_ratio)_c.",
    "csl_floor_credit_redemption_wedge": "(rank(debt_gap)_c - rank(redemption_distance)_c) * rank(redemption_prem_ratio)_c.",
    "csl_liquidity_turnover_size_lattice": "rank(log(turnover_rate))_c * -rank(log(remain_size))_c.",
    "csl_liquidity_reallocation_stockvol_lattice": "rank(log(cb_amount / stk_amount))_c * rank(log(stock_volatility))_c.",
    "csl_liquidity_trade_size_flow_wedge": "rank(log(cb_amount / cb_deal))_c * (rank(log(turnover_rate))_c - rank(log(cb_amount / stk_amount))_c).",
    "csl_barrier_call_put_curvature_lattice": "rank(log(stock_close / call_price))_c * rank(log(stock_close / put_price))_c.",
    "csl_barrier_trigger_premium_wedge": "(rank(log(stock_close / trigger_price_revise))_c - rank(log(stock_close / conv_price))_c) * rank(bond_prem_ratio)_c.",
    "csl_barrier_call_put_moneyness_skew": "(rank(call_distance)_c - rank(put_distance)_c) * rank(conv_distance)_c.",
}


def cross_sectional_rank_lattice_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic expansion runner."""

    return cross_sectional_rank_lattice_catalog()


def _requested_entry(params: dict | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirements_for_family(family: str) -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement(
            "market_cbond.daily_base",
            ("exchange_code", *_FAMILY_BASE_FIELDS[family]),
            _LOOKBACK_DAYS,
        ),
    ]


def _all_requirements() -> list[DailyFactorRequirement]:
    fields = tuple(sorted({field for values in _FAMILY_BASE_FIELDS.values() for field in values}))
    return [
        DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), _LOOKBACK_DAYS),
        DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", *fields), _LOOKBACK_DAYS),
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
        frame = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
        dates = pd.to_datetime(frame["dt"], errors="coerce").dt.normalize()
        frame = frame.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
        index = pd.MultiIndex.from_frame(frame, names=["dt", "code"]) if not frame.empty else _empty_index()

    holder = pd.DataFrame(index=index)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing.index
        ctx.cache[cache_key] = holder
    return index


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, source: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing required columns: {missing}")


def _strict_source(
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


def _snapshot_for_family(ctx: FactorComputeContext, family: str, score_date: pd.Timestamp) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:snapshot:{family}:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    price = _strict_source(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    base_fields = _FAMILY_BASE_FIELDS[family]
    base = _strict_source(
        ctx,
        source="market_cbond.daily_base",
        fields=base_fields,
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    if price.empty:
        built = pd.DataFrame(columns=["code", "close_price", *base_fields])
    else:
        anchor = pd.Timestamp(price["trade_date"].max()).normalize()
        anchored_price = price.loc[price["trade_date"] == anchor, ["trade_date", "code", "close_price"]]
        anchored_base = base.loc[base["trade_date"] == anchor, ["trade_date", "code", *base_fields]]
        built = anchored_price.merge(
            anchored_base,
            on=["trade_date", "code"],
            how="inner",
            validate="one_to_one",
        ).sort_values("code", kind="mergesort")

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    left = pd.to_numeric(numerator, errors="coerce")
    right = pd.to_numeric(denominator, errors="coerce")
    out = pd.Series(np.nan, index=left.index, dtype="float64")
    valid = np.isfinite(left) & np.isfinite(right) & (left > _EPS) & (right > _EPS)
    out.loc[valid] = np.log(left.loc[valid] / right.loc[valid])
    return out


def _centered_rank_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Rank finite state coordinates, refusing a small or degenerate universe."""

    out = pd.DataFrame(np.nan, index=frame.index, columns=columns, dtype="float64")
    if frame.empty:
        return out
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    work = numeric.loc[valid]
    if len(work) < _MIN_CROSS_SECTION or any(work[column].nunique(dropna=True) < 2 for column in columns):
        return out
    out.loc[work.index] = work.rank(method="average", pct=True) - 0.5
    return out


def _optionality_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=snapshot.index)
    frame["moneyness"] = _safe_log_ratio(snapshot["conv_value"], snapshot["close_price"])
    frame["premium"] = pd.to_numeric(snapshot["bond_prem_ratio"], errors="coerce")
    frame["stockvol"] = _safe_log_ratio(snapshot["stock_volatility"], pd.Series(1.0, index=snapshot.index))
    frame["duration"] = pd.to_numeric(snapshot["duration"], errors="coerce")
    ranks = _centered_rank_frame(frame, ("moneyness", "premium", "stockvol", "duration"))
    out = pd.DataFrame(index=snapshot.index)
    out["csl_option_moneyness_stockvol_lattice"] = ranks["moneyness"] * ranks["stockvol"]
    out["csl_option_premium_duration_lattice"] = ranks["premium"] * ranks["duration"]
    out["csl_option_moneyness_premium_vol_wedge"] = ranks["moneyness"] * (ranks["premium"] - ranks["stockvol"])
    return out


def _floor_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=snapshot.index)
    frame["debt_gap"] = pd.to_numeric(snapshot["debt_puredebt_ratio"], errors="coerce") - pd.to_numeric(
        snapshot["puredebt_prem_ratio"], errors="coerce"
    )
    frame["redemption_distance"] = _safe_log_ratio(snapshot["pure_redemption_value"], snapshot["close_price"])
    frame["redemption_premium"] = pd.to_numeric(snapshot["redemption_prem_ratio"], errors="coerce")
    frame["ytm"] = pd.to_numeric(snapshot["ytm"], errors="coerce")
    ranks = _centered_rank_frame(frame, ("debt_gap", "redemption_distance", "redemption_premium", "ytm"))
    out = pd.DataFrame(index=snapshot.index)
    out["csl_floor_debtgap_yield_lattice"] = ranks["debt_gap"] * ranks["ytm"]
    out["csl_floor_redemption_distance_lattice"] = ranks["redemption_distance"] * ranks["redemption_premium"]
    out["csl_floor_credit_redemption_wedge"] = (
        ranks["debt_gap"] - ranks["redemption_distance"]
    ) * ranks["redemption_premium"]
    return out


def _liquidity_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=snapshot.index)
    frame["turnover"] = _safe_log_ratio(snapshot["turnover_rate"], pd.Series(1.0, index=snapshot.index))
    frame["size"] = _safe_log_ratio(snapshot["remain_size"], pd.Series(1.0, index=snapshot.index))
    frame["flow"] = _safe_log_ratio(snapshot["cb_amount"], snapshot["stk_amount"])
    frame["trade_size"] = _safe_log_ratio(snapshot["cb_amount"], snapshot["cb_deal"])
    frame["stockvol"] = _safe_log_ratio(snapshot["stock_volatility"], pd.Series(1.0, index=snapshot.index))
    ranks = _centered_rank_frame(frame, ("turnover", "size", "flow", "trade_size", "stockvol"))
    out = pd.DataFrame(index=snapshot.index)
    out["csl_liquidity_turnover_size_lattice"] = -ranks["turnover"] * ranks["size"]
    out["csl_liquidity_reallocation_stockvol_lattice"] = ranks["flow"] * ranks["stockvol"]
    out["csl_liquidity_trade_size_flow_wedge"] = ranks["trade_size"] * (ranks["turnover"] - ranks["flow"])
    return out


def _barrier_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=snapshot.index)
    frame["conv"] = _safe_log_ratio(snapshot["stock_close_price"], snapshot["cb_conv_price"])
    frame["put"] = _safe_log_ratio(snapshot["stock_close_price"], snapshot["cb_put_price"])
    frame["call"] = _safe_log_ratio(snapshot["stock_close_price"], snapshot["cb_call_price"])
    frame["trigger"] = _safe_log_ratio(snapshot["stock_close_price"], snapshot["trigger_price_revise"])
    frame["premium"] = pd.to_numeric(snapshot["bond_prem_ratio"], errors="coerce")
    ranks = _centered_rank_frame(frame, ("conv", "put", "call", "trigger", "premium"))
    out = pd.DataFrame(index=snapshot.index)
    out["csl_barrier_call_put_curvature_lattice"] = ranks["call"] * ranks["put"]
    out["csl_barrier_trigger_premium_wedge"] = (ranks["trigger"] - ranks["conv"]) * ranks["premium"]
    out["csl_barrier_call_put_moneyness_skew"] = (ranks["call"] - ranks["put"]) * ranks["conv"]
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "cross_sectional_optionality_rank_lattice": _optionality_features,
    "cross_sectional_floor_credit_rank_lattice": _floor_features,
    "cross_sectional_liquidity_capacity_rank_lattice": _liquidity_features,
    "cross_sectional_barrier_geometry_rank_lattice": _barrier_features,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    out_index = _output_index(ctx)
    score_date = _score_date_from_panel(ctx.panel)
    if out_index.empty or score_date is None:
        built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
    else:
        snapshot = _snapshot_for_family(ctx, family, score_date)
        calculated = _FAMILY_CALCULATORS[family](snapshot)
        if snapshot.empty:
            built = pd.DataFrame(index=out_index, columns=_FAMILY_SIGNALS[family], dtype="float64")
        else:
            calculated = calculated.assign(code=snapshot["code"].astype(str).to_numpy()).set_index("code")
            codes = [
                str(_canonical_market_code(pd.Series([code]), pd.Series([""])).iloc[0])
                for _, code in out_index
            ]
            values = calculated.reindex(codes).loc[:, list(_FAMILY_SIGNALS[family])]
            built = values.copy()
            built.index = out_index
            built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningCrossSectionalRankLatticeV1(Factor):
    """Research-only non-linear strict-T-1 cross-sectional state kernel."""

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
