"""Research-only strict-T-1 credit, duration, term, and supply dynamics.

This catalogue studies *changes and prior-only residuals* among a convertible
bond's completed credit-yield proxy, duration/convexity, remaining term, issue
supply, and realised trading capacity.  It deliberately does not expose a
raw yield, duration, convexity, maturity, supply, premium, debt-floor, or
intrinsic-value level.  In particular, the known coverage-fractured
``debt_puredebt_ratio`` and ``puredebt_prem_ratio`` fields are not requested.

Only declared ``market_cbond.daily_price`` and ``market_cbond.daily_base``
context tables are consumed.  ``daily_price.close_price`` establishes an
independent latest completed T-1 session anchor.  Every source is separately
restricted to ``trade_date < score_date`` before the date-key join, so a
score-date or future daily row cannot influence any output.  Missing or
invalid histories stay ``NaN``; this module never substitutes zero or reads
files, databases, labels, pools, scores, PnL, or live artifacts.

The module is import-only research code.  It is intentionally absent from
``defs.__init__``, aggregates, factor contracts, configurations, and all live
paths.
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


KERNEL_NAME = "factor_mining_daily_credit_duration_term_supply_dynamics_v1"
CATALOG_VERSION = "20260803_daily_credit_duration_term_supply_dynamics_v1"
_LOOKBACK_DAYS = 66
_WINDOW = 20
_EPS = 1e-12
_MIN_REGRESSION_OBSERVATIONS = 12

_PRICE_FIELDS = ("close_price", "volume", "amount")
_BASE_FIELDS = (
    "current_yield",
    "duration",
    "convexity",
    "year_to_mat",
    "remain_size",
    "bond_prem_ratio",
    "conv_value",
    "pure_redemption_value",
)
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
    """One auditable research-only credit/term dynamics candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str,
    signals: Iterable[str],
    hypothesis: str,
) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CREDIT_MOTION_SIGNALS = (
    "cdcm_yield_duration_delta_beta20",
    "cdcm_yield_convexity_delta_beta20",
    "cdcm_duration_convexity_delta_corr20",
)
_TERM_SUPPLY_SIGNALS = (
    "tsrr_supply_term_level_beta20",
    "tsrr_duration_term_ratio_delta5",
    "tsrr_supply_term_oos_residual20",
)
_FLOW_ABSORPTION_SIGNALS = (
    "cfad_yield_amount_capacity_delta_beta20",
    "cfad_yield_volume_capacity_delta_beta20",
    "cfad_yield_price_supply_oos_residual20",
)
_CONVEXITY_ANCHOR_SIGNALS = (
    "cat_convexity_duration_delta_beta20",
    "cat_convexity_anchor_delta_beta20",
    "cat_convexity_state_oos_residual20",
)

_CATALOG = (
    _entries(
        "prior_credit_duration_convexity_motion",
        _CREDIT_MOTION_SIGNALS,
        "Completed changes in yield, duration, and convexity describe a sensitivity-state motion rather than any one prior-day risk level.",
    )
    + _entries(
        "prior_term_supply_roll_rebalancing",
        _TERM_SUPPLY_SIGNALS,
        "The historical relation between declining term, duration-to-term shape, and changing remaining issue supply describes roll-down rebalancing rather than a static maturity or supply level.",
    )
    + _entries(
        "prior_credit_flow_absorption_dynamics",
        _FLOW_ABSORPTION_SIGNALS,
        "Completed yield changes conditional on amount, volume, price, and supply changes describe credit-sensitive flow absorption rather than raw liquidity or yield.",
    )
    + _entries(
        "prior_convexity_anchor_transition",
        _CONVEXITY_ANCHOR_SIGNALS,
        "Convexity changes and prior-only state residuals relative to duration and conversion-anchor geometry describe an anchor transition rather than static option value.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "prior_credit_duration_convexity_motion": _CREDIT_MOTION_SIGNALS,
    "prior_term_supply_roll_rebalancing": _TERM_SUPPLY_SIGNALS,
    "prior_credit_flow_absorption_dynamics": _FLOW_ABSORPTION_SIGNALS,
    "prior_convexity_anchor_transition": _CONVEXITY_ANCHOR_SIGNALS,
}

# All lower-case ``t`` terms below are completed daily sessions.  The latest
# eligible session A is the exact maximum daily-price date strictly before the
# score date T.  No formula consumes any row on T or later.
FORMULAS: dict[str, str] = {
    "cdcm_yield_duration_delta_beta20": (
        "OLS beta of delta current_yield on delta duration over the latest 20 completed strict-prior changes"
    ),
    "cdcm_yield_convexity_delta_beta20": (
        "OLS beta of delta current_yield on delta convexity over the latest 20 completed strict-prior changes"
    ),
    "cdcm_duration_convexity_delta_corr20": (
        "Pearson correlation of delta duration and delta convexity over the latest 20 completed strict-prior changes"
    ),
    "tsrr_supply_term_level_beta20": (
        "OLS beta of log(remain_size) on year_to_mat over the latest 20 completed strict-prior sessions"
    ),
    "tsrr_duration_term_ratio_delta5": (
        "log(duration/year_to_mat)_A - log(duration/year_to_mat)_(A-5) on completed strict-prior sessions"
    ),
    "tsrr_supply_term_oos_residual20": (
        "OOS residual of log(remain_size)_A from an OLS trained only on A-20..A-1 with year_to_mat and duration"
    ),
    "cfad_yield_amount_capacity_delta_beta20": (
        "OLS beta of delta current_yield on delta log1p(amount/remain_size) over 20 completed strict-prior changes"
    ),
    "cfad_yield_volume_capacity_delta_beta20": (
        "OLS beta of delta current_yield on delta log1p(volume/remain_size) over 20 completed strict-prior changes"
    ),
    "cfad_yield_price_supply_oos_residual20": (
        "OOS residual of latest delta current_yield after prior-20 OLS on delta log(close_price) and delta log(remain_size)"
    ),
    "cat_convexity_duration_delta_beta20": (
        "OLS beta of delta convexity on delta duration over the latest 20 completed strict-prior changes"
    ),
    "cat_convexity_anchor_delta_beta20": (
        "OLS beta of delta convexity on delta log(conv_value/pure_redemption_value) over 20 completed strict-prior changes"
    ),
    "cat_convexity_state_oos_residual20": (
        "OOS residual of convexity_A from prior-20 OLS on duration, current_yield, bond_prem_ratio, and log(conv_value/pure_redemption_value)"
    ),
}


def daily_credit_duration_term_supply_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first strict-T-1 research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_credit_duration_term_supply_dynamics_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _all_requirements() -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code", *_PRICE_FIELDS),
            _LOOKBACK_DAYS,
        ),
        DailyFactorRequirement(
            "market_cbond.daily_base",
            ("exchange_code", *_BASE_FIELDS),
            _LOOKBACK_DAYS,
        ),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize source codes while retaining exchange disambiguation."""

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
    """Read declared daily context and reject score-date/future rows."""

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
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].drop_duplicates()
    keys = keys.sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=("dt", "code"))


def _score_date_from_index(index: pd.MultiIndex) -> pd.Timestamp:
    dates = pd.to_datetime(index.get_level_values("dt"), errors="coerce").normalize().unique()
    parsed = [pd.Timestamp(day) for day in dates if not pd.isna(day)]
    if len(parsed) != 1:
        raise ValueError(f"{KERNEL_NAME} requires one valid score date per context")
    return parsed[0]


def _last_n_are_consecutive(frame: pd.DataFrame, count: int) -> bool:
    """Require the tail to occupy adjacent independent daily-price sessions."""

    if len(frame) < count or "__price_session_index" not in frame.columns:
        return False
    positions = pd.to_numeric(
        frame.tail(count)["__price_session_index"], errors="coerce"
    ).to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _numeric_tail(
    frame: pd.DataFrame,
    fields: tuple[str, ...],
    count: int,
) -> np.ndarray | None:
    """Return a complete finite, consecutive tail in the requested field order."""

    if not _last_n_are_consecutive(frame, count):
        return None
    values = np.column_stack(
        [
            pd.to_numeric(frame.tail(count)[field], errors="coerce").to_numpy(dtype="float64")
            for field in fields
        ]
    )
    return values if np.isfinite(values).all() else None


def _safe_beta(y: np.ndarray, x: np.ndarray) -> float:
    """Return an intercept-inclusive OLS slope only for variable finite series."""

    if len(y) != len(x) or len(y) < _MIN_REGRESSION_OBSERVATIONS:
        return float("nan")
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        return float("nan")
    if float(np.std(y)) <= _EPS or float(np.std(x)) <= _EPS:
        return float("nan")
    design = np.column_stack((np.ones(len(x), dtype="float64"), x))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    value = float(beta[1])
    return value if np.isfinite(value) else float("nan")


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation only for variable finite paired series."""

    if len(left) != len(right) or len(left) < _MIN_REGRESSION_OBSERVATIONS:
        return float("nan")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return float("nan")
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _safe_log(values: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(values).all() or (values <= _EPS).any():
        return None
    return np.log(values)


def _safe_log1p_capacity(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray | None:
    if not (
        np.isfinite(numerator).all()
        and np.isfinite(denominator).all()
        and (numerator >= 0.0).all()
        and (denominator > _EPS).all()
    ):
        return None
    values = np.log1p(numerator / denominator)
    return values if np.isfinite(values).all() else None


def _safe_anchor_log(conv_value: np.ndarray, redemption_value: np.ndarray) -> np.ndarray | None:
    if not (
        np.isfinite(conv_value).all()
        and np.isfinite(redemption_value).all()
        and (conv_value > _EPS).all()
        and (redemption_value > _EPS).all()
    ):
        return None
    values = np.log(conv_value / redemption_value)
    return values if np.isfinite(values).all() else None


def _one_oos_residual(training: np.ndarray, current: np.ndarray) -> float:
    """Fit a standardized OLS on completed predecessors and score one held-out row."""

    if training.ndim != 2 or current.ndim != 1:
        return float("nan")
    if training.shape[0] != _WINDOW or training.shape[1] != len(current) or len(current) < 2:
        return float("nan")
    if not (np.isfinite(training).all() and np.isfinite(current).all()):
        return float("nan")
    target = training[:, 0]
    predictors = training[:, 1:]
    if float(np.std(target, ddof=1)) <= _EPS:
        return float("nan")
    mean = predictors.mean(axis=0)
    scale = predictors.std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= _EPS).any():
        return float("nan")
    design = np.column_stack((np.ones(len(training)), (predictors - mean) / scale))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    current_design = np.r_[1.0, (current[1:] - mean) / scale]
    residual = float(current[0] - np.dot(current_design, beta))
    return residual if np.isfinite(residual) else float("nan")


def _credit_motion_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CREDIT_MOTION_SIGNALS}
    values = _numeric_tail(frame, ("current_yield", "duration", "convexity"), _WINDOW + 1)
    if values is None:
        return out
    yield_changes = np.diff(values[:, 0])
    duration_changes = np.diff(values[:, 1])
    convexity_changes = np.diff(values[:, 2])
    out["cdcm_yield_duration_delta_beta20"] = _safe_beta(yield_changes, duration_changes)
    out["cdcm_yield_convexity_delta_beta20"] = _safe_beta(yield_changes, convexity_changes)
    out["cdcm_duration_convexity_delta_corr20"] = _safe_corr(duration_changes, convexity_changes)
    return out


def _term_supply_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TERM_SUPPLY_SIGNALS}

    level_values = _numeric_tail(frame, ("remain_size", "year_to_mat"), _WINDOW)
    if level_values is not None:
        supply_log = _safe_log(level_values[:, 0])
        if supply_log is not None:
            out["tsrr_supply_term_level_beta20"] = _safe_beta(supply_log, level_values[:, 1])

    ratio_values = _numeric_tail(frame, ("duration", "year_to_mat"), 6)
    if ratio_values is not None and (ratio_values > _EPS).all():
        ratio = np.log(ratio_values[:, 0] / ratio_values[:, 1])
        if np.isfinite(ratio).all():
            out["tsrr_duration_term_ratio_delta5"] = float(ratio[-1] - ratio[0])

    residual_values = _numeric_tail(frame, ("remain_size", "year_to_mat", "duration"), _WINDOW + 1)
    if residual_values is not None:
        supply_log = _safe_log(residual_values[:, 0])
        if supply_log is not None:
            design = np.column_stack((supply_log, residual_values[:, 1], residual_values[:, 2]))
            out["tsrr_supply_term_oos_residual20"] = _one_oos_residual(design[:-1], design[-1])
    return out


def _credit_flow_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _FLOW_ABSORPTION_SIGNALS}

    capacity_values = _numeric_tail(
        frame,
        ("current_yield", "amount", "volume", "remain_size"),
        _WINDOW + 1,
    )
    if capacity_values is not None:
        yield_changes = np.diff(capacity_values[:, 0])
        amount_capacity = _safe_log1p_capacity(capacity_values[:, 1], capacity_values[:, 3])
        volume_capacity = _safe_log1p_capacity(capacity_values[:, 2], capacity_values[:, 3])
        if amount_capacity is not None:
            out["cfad_yield_amount_capacity_delta_beta20"] = _safe_beta(
                yield_changes,
                np.diff(amount_capacity),
            )
        if volume_capacity is not None:
            out["cfad_yield_volume_capacity_delta_beta20"] = _safe_beta(
                yield_changes,
                np.diff(volume_capacity),
            )

    residual_values = _numeric_tail(
        frame,
        ("current_yield", "close_price", "remain_size"),
        _WINDOW + 2,
    )
    if residual_values is not None:
        close_log = _safe_log(residual_values[:, 1])
        supply_log = _safe_log(residual_values[:, 2])
        if close_log is not None and supply_log is not None:
            changes = np.column_stack(
                (
                    np.diff(residual_values[:, 0]),
                    np.diff(close_log),
                    np.diff(supply_log),
                )
            )
            out["cfad_yield_price_supply_oos_residual20"] = _one_oos_residual(
                changes[:-1],
                changes[-1],
            )
    return out


def _convexity_anchor_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CONVEXITY_ANCHOR_SIGNALS}

    duration_values = _numeric_tail(frame, ("convexity", "duration"), _WINDOW + 1)
    if duration_values is not None:
        out["cat_convexity_duration_delta_beta20"] = _safe_beta(
            np.diff(duration_values[:, 0]),
            np.diff(duration_values[:, 1]),
        )

    anchor_values = _numeric_tail(
        frame,
        ("convexity", "conv_value", "pure_redemption_value"),
        _WINDOW + 1,
    )
    if anchor_values is not None:
        anchor_log = _safe_anchor_log(anchor_values[:, 1], anchor_values[:, 2])
        if anchor_log is not None:
            out["cat_convexity_anchor_delta_beta20"] = _safe_beta(
                np.diff(anchor_values[:, 0]),
                np.diff(anchor_log),
            )

    residual_values = _numeric_tail(
        frame,
        (
            "convexity",
            "duration",
            "current_yield",
            "bond_prem_ratio",
            "conv_value",
            "pure_redemption_value",
        ),
        _WINDOW + 1,
    )
    if residual_values is not None:
        anchor_log = _safe_anchor_log(residual_values[:, 4], residual_values[:, 5])
        if anchor_log is not None:
            design = np.column_stack(
                (
                    residual_values[:, 0],
                    residual_values[:, 1],
                    residual_values[:, 2],
                    residual_values[:, 3],
                    anchor_log,
                )
            )
            out["cat_convexity_state_oos_residual20"] = _one_oos_residual(design[:-1], design[-1])
    return out


def _all_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_credit_motion_metrics(frame))
    out.update(_term_supply_metrics(frame))
    out.update(_credit_flow_metrics(frame))
    out.update(_convexity_anchor_metrics(frame))
    return out


def _build_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    output_index = _output_index(ctx)
    out = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    if out.empty:
        return out
    score_date = _score_date_from_index(output_index)
    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=_PRICE_FIELDS,
        score_date=score_date,
    )
    base = _strict_history_source(
        ctx,
        source="market_cbond.daily_base",
        fields=_BASE_FIELDS,
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    if price.empty:
        return out
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    price_calendar = sorted(pd.Timestamp(day).normalize() for day in price["trade_date"].unique())
    price_session_index = {day: position for position, day in enumerate(price_calendar)}
    history = base.merge(
        price.loc[:, ["trade_date", "code", *_PRICE_FIELDS]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort")
    history["__price_session_index"] = history["trade_date"].map(price_session_index)
    groups = {
        str(code): group.reset_index(drop=True)
        for code, group in history.groupby("code", sort=False)
    }
    for dt, raw_code in output_index:
        code = _canonical_market_code(pd.Series([raw_code])).iloc[0]
        if not code or code not in anchor_codes:
            continue
        instrument_history = groups.get(str(code))
        if (
            instrument_history is None
            or instrument_history.empty
            or pd.Timestamp(instrument_history["trade_date"].iloc[-1]).normalize() != anchor
        ):
            continue
        metrics = _all_metrics(instrument_history)
        for signal, value in metrics.items():
            if np.isfinite(value):
                out.at[(dt, raw_code), signal] = float(value)
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    output_index = _output_index(ctx)
    score_date = _score_date_from_index(output_index) if not output_index.empty else pd.NaT
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
    built = _build_feature_frame(ctx)
    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyCreditDurationTermSupplyDynamicsV1(Factor):
    """Research-only strict-T-1 credit/duration/term/supply dynamics kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        if params and params.get("signal"):
            _requested_entry(params)
        return _all_requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        out = _feature_frame(ctx)[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
