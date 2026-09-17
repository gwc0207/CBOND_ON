"""Research-only strict-PIT conditional T1430-response residual catalogue.

This module is deliberately narrower than the general intraday and daily
catalogues.  It asks a cross-sectional question on each score day: after the
completed T-1 price/structural state explains an instrument's visible T1430
response, is the remaining response unusual?  The fitted Ridge is a
cross-sectional de-scaling device, never a label, return, pool, or model fit.

Every input has an explicit availability boundary:

* ``market_cbond.daily_price`` and ``market_cbond.daily_base`` are filtered to
  ``trade_date < score_date``.  The latest daily-price date is the independent
  T-1 market anchor; a base row must exist on that exact date, so stale base
  state is not carried forward.
* T-day panel rows must have both an index date and a physical ``trade_time``
  on the score date, within a continuous market session, and no later than
  14:29.

The implementation consumes only :class:`~cbond_on.domain.factors.base.FactorComputeContext`.
It performs no direct I/O and is intentionally import-only: it is not added to
``defs.__init__``, factor contracts, production configurations, or live packs.
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
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_conditional_response_residual_v1"
CATALOG_VERSION = "20260803_conditional_response_residual_v1"

_EPS = 1e-12
_LOOKBACK_DAYS = 5
_MIN_PATH_ROWS = 6
_MIN_CROSS_SECTION = 30
_RIDGE_ALPHA = 5.0

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

_PRICE_FIELDS = (
    "close_price",
    "prev_close_price",
    "high_price",
    "low_price",
    "amount",
    "deal",
)
_BASE_FIELDS = (
    "bond_prem_ratio",
    "duration",
    "stock_volatility",
    "turnover_rate",
    "remain_size",
)
_BASIC_PANEL_COLUMNS = ("trade_time", "pre_close", "last", "amount")
_QUOTE_PANEL_COLUMNS = ("ask_price1", "bid_price1", "ask_volume1", "bid_volume1")


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable, research-only signal and its economic family."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class ResidualDefinition:
    """A visible T1430 target and strictly-PIT conditioning variables."""

    signal: str
    target: str
    covariates: tuple[str, ...]


def _definition(signal: str, target: str, *covariates: str) -> ResidualDefinition:
    return ResidualDefinition(signal=signal, target=target, covariates=tuple(covariates))


# Targets are deliberately different response channels.  Each fitted model is
# recomputed only across the contemporaneous score-day cross-section, using no
# label or later panel observation.
_FAMILY_DEFINITIONS: dict[str, tuple[ResidualDefinition, ...]] = {
    "conditional_price_response_residual": (
        _definition(
            "ccr_late_reversal_prior_state_residual",
            "late_reversal",
            "early_return",
            "prior_return",
            "prior_range",
            "prior_premium",
            "prior_stock_volatility",
        ),
        _definition(
            "ccr_path_efficiency_prior_state_residual",
            "path_efficiency",
            "early_return",
            "prior_return",
            "prior_range",
            "prior_turnover",
            "prior_duration",
        ),
    ),
    "conditional_liquidity_response_residual": (
        _definition(
            "ccr_flow_acceleration_prior_liquidity_residual",
            "flow_acceleration",
            "early_return",
            "prior_log_amount",
            "prior_log_deal",
            "prior_turnover",
            "prior_log_remain_size",
        ),
        _definition(
            "ccr_late_impact_prior_liquidity_residual",
            "late_impact",
            "late_return",
            "prior_log_amount",
            "prior_log_deal",
            "prior_range",
            "prior_log_remain_size",
        ),
    ),
    "conditional_book_response_residual": (
        _definition(
            "ccr_spread_shift_prior_state_residual",
            "spread_shift",
            "early_return",
            "prior_premium",
            "prior_stock_volatility",
            "prior_turnover",
            "prior_duration",
        ),
        _definition(
            "ccr_imbalance_shift_prior_state_residual",
            "imbalance_shift",
            "early_return",
            "prior_premium",
            "prior_stock_volatility",
            "prior_log_amount",
            "prior_duration",
        ),
    ),
    "conditional_state_interaction_residual": (
        _definition(
            "ccr_premium_tail_interaction_residual",
            "premium_tail_interaction",
            "late_return",
            "prior_premium",
            "early_return",
            "prior_return",
            "prior_log_remain_size",
        ),
        _definition(
            "ccr_duration_flow_interaction_residual",
            "duration_flow_interaction",
            "flow_acceleration",
            "prior_duration",
            "early_return",
            "prior_log_amount",
            "prior_turnover",
        ),
    ),
}

_FAMILY_HYPOTHESES = {
    "conditional_price_response_residual": (
        "Late reversal and path efficiency after removing the visible T-1 return, range, valuation, and risk state."
    ),
    "conditional_liquidity_response_residual": (
        "T1430 flow acceleration and price impact after removing the completed T-1 liquidity and float state."
    ),
    "conditional_book_response_residual": (
        "Intraday repricing of displayed spread and imbalance after removing known T-1 valuation, risk, and liquidity state."
    ),
    "conditional_state_interaction_residual": (
        "Abnormal T1430 response of prior premium and duration regimes after both constituent variables are explained cross-sectionally."
    ),
}

_CATALOG = tuple(
    CatalogEntry(
        family=family,
        signal=definition.signal,
        kernel=KERNEL_NAME,
        hypothesis=_FAMILY_HYPOTHESES[family],
    )
    for family, definitions in _FAMILY_DEFINITIONS.items()
    for definition in definitions
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_DEFINITION_BY_SIGNAL = {
    definition.signal: definition
    for definitions in _FAMILY_DEFINITIONS.values()
    for definition in definitions
}
_FAMILY_SIGNALS = {
    family: tuple(definition.signal for definition in definitions)
    for family, definitions in _FAMILY_DEFINITIONS.items()
}

# This mapping is intentionally kept next to the executable definitions so a
# screen report can explain a residual without reverse-engineering a feature
# matrix.  ``z_cs`` is a same-score-day, non-label cross-sectional standardisation.
FORMULAS = {
    "ccr_late_reversal_prior_state_residual": (
        "ridge_resid(z_cs(late_return-early_return) | early_return, prior_return, prior_range, prior_premium, prior_stock_volatility)"
    ),
    "ccr_path_efficiency_prior_state_residual": (
        "ridge_resid(z_cs(full_log_return/sum(abs(intraday_log_returns))) | early_return, prior_return, prior_range, prior_turnover, prior_duration)"
    ),
    "ccr_flow_acceleration_prior_liquidity_residual": (
        "ridge_resid(z_cs(log(mean(late_amount_increment)/mean(early_amount_increment))) | early_return, prior_log_amount, prior_log_deal, prior_turnover, prior_log_remain_size)"
    ),
    "ccr_late_impact_prior_liquidity_residual": (
        "ridge_resid(z_cs(abs(late_return)/log1p(late_amount)) | late_return, prior_log_amount, prior_log_deal, prior_range, prior_log_remain_size)"
    ),
    "ccr_spread_shift_prior_state_residual": (
        "ridge_resid(z_cs(mean_late(relative_spread)-mean_early(relative_spread)) | early_return, prior_premium, prior_stock_volatility, prior_turnover, prior_duration)"
    ),
    "ccr_imbalance_shift_prior_state_residual": (
        "ridge_resid(z_cs(mean_late(L1_imbalance)-mean_early(L1_imbalance)) | early_return, prior_premium, prior_stock_volatility, prior_log_amount, prior_duration)"
    ),
    "ccr_premium_tail_interaction_residual": (
        "ridge_resid(z_cs(late_return*z_cs(prior_premium)) | late_return, prior_premium, early_return, prior_return, prior_log_remain_size)"
    ),
    "ccr_duration_flow_interaction_residual": (
        "ridge_resid(z_cs(flow_acceleration*z_cs(prior_duration)) | flow_acceleration, prior_duration, early_return, prior_log_amount, prior_turnover)"
    ),
}


def conditional_response_residual_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable four-family, eight-signal research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic expansion-runner compatibility entrypoint."""

    return conditional_response_residual_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing required columns: {missing}")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalise a market code without guessing an exchange when it is absent."""

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
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _output_keys(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return every requested score-day output key plus its canonical join key."""

    panel = ensure_panel_index(ctx.panel)
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[dates == score_date].drop_duplicates().copy()
    keys["code"] = keys["code"].astype(str)
    keys["canonical_code"] = _canonical_market_code(keys["code"])
    return keys.sort_values(["dt", "code"], kind="mergesort").reset_index(drop=True)


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only observations physically visible on score day through 14:29."""

    panel = ensure_trade_time(ctx.panel)
    _require_columns(panel, _BASIC_PANEL_COLUMNS, owner="panel")
    frame = panel.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(
        lambda clock: _continuous_session(clock) if pd.notna(clock) else False
    )
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
    out["canonical_code"] = _canonical_market_code(out["code"])
    return out.sort_values(["canonical_code", "trade_time", "seq"], kind="mergesort")


def _strict_prior_source(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    """Read only declared, pre-score-day observations from daily context."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), owner=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["canonical_code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    frame = frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["canonical_code"].notna()
        & (frame["canonical_code"] != "")
    ].copy()
    if frame.duplicated(["trade_date", "canonical_code"], keep=False).any():
        examples = frame.loc[
            frame.duplicated(["trade_date", "canonical_code"], keep=False),
            ["trade_date", "canonical_code"],
        ].head(3)
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}"
        )
    return frame.sort_values(["canonical_code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if numerator <= _EPS or denominator <= _EPS:
        return float("nan")
    return float(np.log(numerator / denominator))


def _safe_log1p(value: float) -> float:
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    return float(np.log1p(value))


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _prior_state_frame(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return exact T-1 price/base state; never carry a stale base row forward."""

    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:prior_state:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    price = _strict_prior_source(
        ctx,
        source="market_cbond.daily_price",
        fields=_PRICE_FIELDS,
        score_date=score_date,
    )
    base = _strict_prior_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_BASE_FIELDS,
        score_date=score_date,
    )
    if price.empty:
        state = pd.DataFrame(columns=("canonical_code", *_state_columns())).set_index("canonical_code")
    else:
        anchor = pd.Timestamp(price["trade_date"].max()).normalize()
        price_t1 = price.loc[price["trade_date"] == anchor].copy()
        base_t1 = base.loc[base["trade_date"] == anchor].copy()
        # The inner join is deliberate: a code with an older base row gets no
        # state at all instead of an implicit stale carry.
        joined = price_t1.merge(
            base_t1.loc[:, ["canonical_code", *_BASE_FIELDS]],
            on="canonical_code",
            how="inner",
            validate="one_to_one",
        )
        state = pd.DataFrame({"canonical_code": joined["canonical_code"].astype(str)})
        close = _numeric(joined, "close_price")
        prev_close = _numeric(joined, "prev_close_price")
        high = _numeric(joined, "high_price")
        low = _numeric(joined, "low_price")
        amount = _numeric(joined, "amount")
        deal = _numeric(joined, "deal")
        remain_size = _numeric(joined, "remain_size")
        state["prior_return"] = [
            _safe_log_ratio(float(current), float(previous))
            for current, previous in zip(close, prev_close, strict=False)
        ]
        state["prior_range"] = [
            _safe_log_ratio(float(current_high), float(current_low))
            for current_high, current_low in zip(high, low, strict=False)
        ]
        state["prior_log_amount"] = [_safe_log1p(float(value)) for value in amount]
        state["prior_log_deal"] = [_safe_log1p(float(value)) for value in deal]
        state["prior_log_remain_size"] = [
            _safe_log_ratio(float(value), 1.0) for value in remain_size
        ]
        for column, output in (
            ("bond_prem_ratio", "prior_premium"),
            ("duration", "prior_duration"),
            ("stock_volatility", "prior_stock_volatility"),
            ("turnover_rate", "prior_turnover"),
        ):
            state[output] = pd.to_numeric(joined[column], errors="coerce").astype("float64")
        state = state.set_index("canonical_code").sort_index()

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = state
    return state


def _state_columns() -> tuple[str, ...]:
    return (
        "prior_return",
        "prior_range",
        "prior_log_amount",
        "prior_log_deal",
        "prior_log_remain_size",
        "prior_premium",
        "prior_duration",
        "prior_stock_volatility",
        "prior_turnover",
    )


def _empty_response_metrics() -> dict[str, float]:
    return {
        "early_return": float("nan"),
        "late_return": float("nan"),
        "late_reversal": float("nan"),
        "path_efficiency": float("nan"),
        "flow_acceleration": float("nan"),
        "late_impact": float("nan"),
        "spread_shift": float("nan"),
        "imbalance_shift": float("nan"),
    }


def _phase_mask(times: pd.Series, start: dt_time, end: dt_time) -> np.ndarray:
    clocks = pd.to_datetime(times, errors="coerce").dt.time
    return ((clocks >= start) & (clocks <= end)).to_numpy(dtype=bool)


def _response_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Summarise one score-day path, keeping failure domains explicit."""

    out = _empty_response_metrics()
    if len(frame) < _MIN_PATH_ROWS:
        return out
    if frame["seq"].duplicated(keep=False).any() or frame["trade_time"].duplicated(keep=False).any():
        return out
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        return out
    ticks = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
    if (np.diff(ticks) <= 0).any():
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
    returns = np.diff(np.log(prices))
    if len(returns) < _MIN_PATH_ROWS - 1 or not np.isfinite(returns).all():
        return out
    early = _phase_mask(timestamps, _MORNING_START, _EARLY_END)
    late = _phase_mask(timestamps, _LATE_START, _CUTOFF)
    if int(early.sum()) >= 2:
        out["early_return"] = _safe_log_ratio(float(prices[early][-1]), float(prices[early][0]))
    if int(late.sum()) >= 2:
        out["late_return"] = _safe_log_ratio(float(prices[late][-1]), float(prices[late][0]))
    if np.isfinite(out["late_return"]) and np.isfinite(out["early_return"]):
        out["late_reversal"] = float(out["late_return"] - out["early_return"])
    denominator = float(np.abs(returns).sum())
    full_return = float(np.log(prices[-1] / prices[0]))
    if denominator > _EPS and np.isfinite(full_return):
        out["path_efficiency"] = float(full_return / denominator)

    amount = _numeric(frame, "amount")
    if np.isfinite(amount).all() and (amount >= 0.0).all():
        increments = np.diff(amount)
        # Any reset is a failed flow observation.  Unlike a label or score,
        # it must not be imputed or clipped into a usable amount response.
        if np.isfinite(increments).all() and (increments >= 0.0).all():
            interval_times = timestamps.iloc[1:]
            early_intervals = _phase_mask(interval_times, _MORNING_START, _EARLY_END)
            late_intervals = _phase_mask(interval_times, _LATE_START, _CUTOFF)
            if int(early_intervals.sum()) >= 2 and int(late_intervals.sum()) >= 2:
                early_mean = float(np.mean(increments[early_intervals]))
                late_mean = float(np.mean(increments[late_intervals]))
                out["flow_acceleration"] = _safe_log_ratio(late_mean, early_mean)
                late_amount = float(increments[late_intervals].sum())
                if np.isfinite(out["late_return"]) and late_amount > _EPS:
                    out["late_impact"] = float(
                        abs(out["late_return"]) / np.log1p(late_amount)
                    )

    if set(_QUOTE_PANEL_COLUMNS).issubset(frame.columns):
        ask = _numeric(frame, "ask_price1")
        bid = _numeric(frame, "bid_price1")
        ask_size = _numeric(frame, "ask_volume1")
        bid_size = _numeric(frame, "bid_volume1")
        depth = ask_size + bid_size
        valid_book = (
            np.isfinite(ask)
            & np.isfinite(bid)
            & np.isfinite(ask_size)
            & np.isfinite(bid_size)
            & (ask > _EPS)
            & (bid > _EPS)
            & (ask >= bid)
            & (ask_size >= 0.0)
            & (bid_size >= 0.0)
            & (depth > _EPS)
        )
        if bool(valid_book.all()) and int(early.sum()) >= 2 and int(late.sum()) >= 2:
            midpoint = (ask + bid) / 2.0
            spread = (ask - bid) / midpoint
            imbalance = (bid_size - ask_size) / depth
            out["spread_shift"] = float(np.mean(spread[late]) - np.mean(spread[early]))
            out["imbalance_shift"] = float(
                np.mean(imbalance[late]) - np.mean(imbalance[early])
            )
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in out.items()}


def _response_frame(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:response:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    physical = _strict_physical_frame(ctx, score_date=score_date)
    rows: list[dict[str, object]] = []
    for canonical_code, group in physical.groupby("canonical_code", sort=True):
        if not canonical_code:
            continue
        metrics = _response_metrics(group)
        rows.append({"canonical_code": str(canonical_code), **metrics})
    response = (
        pd.DataFrame(rows).set_index("canonical_code").sort_index()
        if rows
        else pd.DataFrame(columns=_empty_response_metrics()).rename_axis("canonical_code")
    )
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = response
    return response


def _cross_section_z(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = numeric.notna() & np.isfinite(numeric)
    out = pd.Series(np.nan, index=numeric.index, dtype="float64")
    if int(finite.sum()) < _MIN_CROSS_SECTION:
        return out
    selected = numeric.loc[finite]
    scale = float(selected.std(ddof=0))
    if not np.isfinite(scale) or scale <= _EPS:
        return out
    out.loc[finite] = (selected - float(selected.mean())) / scale
    return out


def _ridge_residual(frame: pd.DataFrame, definition: ResidualDefinition) -> pd.Series:
    """Fit a same-day, non-label cross-sectional Ridge and return z residuals."""

    selected = frame.loc[:, [definition.target, *definition.covariates]].apply(
        pd.to_numeric, errors="coerce"
    )
    finite = np.isfinite(selected.to_numpy(dtype="float64")).all(axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    if int(finite.sum()) < _MIN_CROSS_SECTION:
        return out
    values = selected.loc[finite]
    target = values[definition.target].to_numpy(dtype="float64")
    target_scale = float(np.std(target))
    if not np.isfinite(target_scale) or target_scale <= _EPS:
        return out
    x_parts: list[np.ndarray] = []
    for column in definition.covariates:
        value = values[column].to_numpy(dtype="float64")
        scale = float(np.std(value))
        if not np.isfinite(scale) or scale <= _EPS:
            return out
        x_parts.append((value - float(np.mean(value))) / scale)
    design = np.column_stack([np.ones(len(values), dtype="float64"), *x_parts])
    target_z = (target - float(np.mean(target))) / target_scale
    penalty = np.eye(design.shape[1], dtype="float64") * _RIDGE_ALPHA
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target_z)
    except np.linalg.LinAlgError:
        return out
    residual = target_z - design @ coefficients
    if not np.isfinite(residual).all():
        return out
    out.loc[values.index] = residual
    return out


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    score_date = _score_date_from_panel(ctx.panel)
    if score_date is None:
        return pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=["dt", "code"]))
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date.date().isoformat()}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    keys = _output_keys(ctx, score_date=score_date)
    state = _prior_state_frame(ctx, score_date=score_date)
    response = _response_frame(ctx, score_date=score_date)
    work = keys.merge(
        response.reset_index(), on="canonical_code", how="left", validate="many_to_one"
    ).merge(
        state.reset_index(), on="canonical_code", how="left", validate="many_to_one"
    )
    work["premium_tail_interaction"] = work["late_return"] * _cross_section_z(
        work["prior_premium"]
    )
    work["duration_flow_interaction"] = work["flow_acceleration"] * _cross_section_z(
        work["prior_duration"]
    )
    result = pd.DataFrame(index=pd.MultiIndex.from_frame(work[["dt", "code"]], names=("dt", "code")))
    for definition in _DEFINITION_BY_SIGNAL.values():
        result[definition.signal] = _ridge_residual(work, definition).to_numpy(dtype="float64")

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = result
    return result


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningConditionalResponseResidualV1(Factor):
    """Emit one strict-PIT conditional-response residual selected by ``signal``."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(
        cls, params: dict[str, object] | None = None
    ) -> list[DailyFactorRequirement]:
        _requested_entry(params)
        return [
            DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", *_PRICE_FIELDS), _LOOKBACK_DAYS),
            DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", *_BASE_FIELDS), _LOOKBACK_DAYS),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        if entry.family == "conditional_book_response_residual":
            _require_columns(ctx.panel, _QUOTE_PANEL_COLUMNS, owner="panel")
        frame = _feature_frame(ctx)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
