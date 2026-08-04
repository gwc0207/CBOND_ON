"""Research-only strict-T-1 robust credit, term, and supply dynamics v2.

This is an independent successor to the v1 research candidate.  It retains
the historical credit-yield, duration, convexity, term, supply, and completed
flow hypotheses, but never emits a dimensional raw OLS slope.  Every
single-predictor relationship is a standardized correlation with an explicit
minimum effective scale.  Every multi-predictor innovation is a standardized
out-of-sample residual and fails closed when the standardized design is poorly
conditioned.

Only declared ``market_cbond.daily_price`` and ``market_cbond.daily_base``
context tables are consumed.  Every source is independently restricted to
``trade_date < score_date``.  ``daily_price.close_price`` establishes the
latest strict-prior anchor, and the base table is date-key joined to that
independent calendar.  Missing, stale, low-scale, or ill-conditioned histories
remain ``NaN``; this code does not fill, clip, read external files, connect to
a database, access labels/PnL, or open live artifacts.

The module is research-only and intentionally absent from ``defs.__init__``,
aggregates, configurations, contracts, and live paths.
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


KERNEL_NAME = "factor_mining_daily_credit_duration_term_supply_dynamics_v2"
CATALOG_VERSION = "20260803_daily_credit_duration_term_supply_dynamics_v2"
_LOOKBACK_DAYS = 66
_WINDOW = 20
_EPS = 1e-12
_MIN_REGRESSION_OBSERVATIONS = 12
# This is deliberately much stronger than numerical epsilon: a relationship
# with less variation cannot provide an economically interpretable standardised
# coefficient.  It is a fail-closed validity gate, never a rescaling or clip.
_MIN_EFFECTIVE_SCALE = 1e-6
_MAX_STANDARDIZED_CONDITION_NUMBER = 1e4

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
    """One auditable research-only robust dynamics candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(
    family: str,
    signals: Iterable[str],
    hypothesis: str,
) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals
    )


_CREDIT_MOTION_SIGNALS = (
    "cdmr_yield_duration_delta_corr20",
    "cdmr_yield_convexity_delta_corr20",
    "cdmr_duration_convexity_delta_corr20",
)
_TERM_SUPPLY_SIGNALS = (
    "tsrr2_supply_term_level_corr20",
    "tsrr2_duration_term_ratio_delta5",
    "tsrr2_supply_term_standardized_oos20",
)
_FLOW_ABSORPTION_SIGNALS = (
    "cfar_yield_amount_capacity_delta_corr20",
    "cfar_yield_volume_capacity_delta_corr20",
    "cfar_yield_price_supply_standardized_oos20",
)
_CONVEXITY_ANCHOR_SIGNALS = (
    "catr_convexity_duration_delta_corr20",
    "catr_convexity_anchor_delta_corr20",
    "catr_convexity_state_standardized_oos20",
)

_CATALOG = (
    _entries(
        "prior_credit_duration_convexity_motion_robust",
        _CREDIT_MOTION_SIGNALS,
        "Standardized completed change co-movements of yield, duration, and convexity describe sensitivity-state motion without unit-dependent slopes.",
    )
    + _entries(
        "prior_term_supply_roll_rebalancing_robust",
        _TERM_SUPPLY_SIGNALS,
        "Term, duration-to-term shape, and changing remaining supply describe prior roll-down rebalancing only where the completed supply history has effective variation.",
    )
    + _entries(
        "prior_credit_flow_absorption_dynamics_robust",
        _FLOW_ABSORPTION_SIGNALS,
        "Standardized yield/flow co-movements and conditioned innovations describe credit-sensitive flow absorption without a dimensional liquidity slope.",
    )
    + _entries(
        "prior_convexity_anchor_transition_robust",
        _CONVEXITY_ANCHOR_SIGNALS,
        "Bounded convexity-change co-movements and a condition-checked standardized anchor-state innovation describe a convexity transition without beta small-denominator explosions.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "prior_credit_duration_convexity_motion_robust": _CREDIT_MOTION_SIGNALS,
    "prior_term_supply_roll_rebalancing_robust": _TERM_SUPPLY_SIGNALS,
    "prior_credit_flow_absorption_dynamics_robust": _FLOW_ABSORPTION_SIGNALS,
    "prior_convexity_anchor_transition_robust": _CONVEXITY_ANCHOR_SIGNALS,
}

# A is the latest independent daily-price session strictly before score date T.
# Every sequence below ends at A and every innovation model trains only on
# predecessors of its held-out completed row.
FORMULAS: dict[str, str] = {
    "cdmr_yield_duration_delta_corr20": (
        "Pearson corr(delta current_yield, delta duration) over 20 strict-prior changes; each series requires std > 1e-6"
    ),
    "cdmr_yield_convexity_delta_corr20": (
        "Pearson corr(delta current_yield, delta convexity) over 20 strict-prior changes; each series requires std > 1e-6"
    ),
    "cdmr_duration_convexity_delta_corr20": (
        "Pearson corr(delta duration, delta convexity) over 20 strict-prior changes; each series requires std > 1e-6"
    ),
    "tsrr2_supply_term_level_corr20": (
        "Pearson corr(log(remain_size), year_to_mat) over 20 completed sessions; both series require std > 1e-6"
    ),
    "tsrr2_duration_term_ratio_delta5": (
        "log(duration/year_to_mat)_A - log(duration/year_to_mat)_(A-5) on completed strict-prior sessions"
    ),
    "tsrr2_supply_term_standardized_oos20": (
        "Standardized OOS residual of log(remain_size)_A after prior-20 OLS on year_to_mat and duration; fail closed for any scale <=1e-6 or condition number >1e4"
    ),
    "cfar_yield_amount_capacity_delta_corr20": (
        "Pearson corr(delta current_yield, delta log1p(amount/remain_size)) over 20 strict-prior changes; each series requires std > 1e-6"
    ),
    "cfar_yield_volume_capacity_delta_corr20": (
        "Pearson corr(delta current_yield, delta log1p(volume/remain_size)) over 20 strict-prior changes; each series requires std > 1e-6"
    ),
    "cfar_yield_price_supply_standardized_oos20": (
        "Standardized OOS residual of latest delta current_yield after prior-20 OLS on delta log(close_price) and delta log(remain_size), subject to scale/condition gates"
    ),
    "catr_convexity_duration_delta_corr20": (
        "Pearson corr(delta convexity, delta duration) over 20 strict-prior changes; bounded standardized sensitivity coupling"
    ),
    "catr_convexity_anchor_delta_corr20": (
        "Pearson corr(delta convexity, delta log(conv_value/pure_redemption_value)) over 20 strict-prior changes; bounded anchor coupling"
    ),
    "catr_convexity_state_standardized_oos20": (
        "Standardized OOS convexity innovation at A after prior-20 OLS on duration,current_yield,bond_prem_ratio,anchor log; fail closed for low scale or condition number >1e4"
    ),
}


def daily_credit_duration_term_supply_dynamics_v2_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first robust strict-T-1 catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_credit_duration_term_supply_dynamics_v2_catalog()


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


def _canonical_market_code(
    values: pd.Series, exchanges: pd.Series | None = None
) -> pd.Series:
    """Normalize source codes while retaining exchange disambiguation."""

    exchange_values = (
        exchanges if exchanges is not None else pd.Series("", index=values.index)
    )

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
        [
            _one(value, exchange)
            for value, exchange in zip(values, exchange_values, strict=False)
        ],
        index=values.index,
        dtype="string",
    )


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, owner: str
) -> None:
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
    """Read declared daily context and independently reject score/future rows."""

    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(
        raw, ("trade_date", "code", "exchange_code", *fields), owner=source
    )
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"], errors="coerce"
    ).dt.normalize()
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
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].drop_duplicates()
    keys = keys.sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=("dt", "code"))


def _score_date_from_index(index: pd.MultiIndex) -> pd.Timestamp:
    dates = (
        pd.to_datetime(index.get_level_values("dt"), errors="coerce")
        .normalize()
        .unique()
    )
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
    """Return a complete finite consecutive tail in requested field order."""

    if not _last_n_are_consecutive(frame, count):
        return None
    values = np.column_stack(
        [
            pd.to_numeric(frame.tail(count)[field], errors="coerce").to_numpy(
                dtype="float64"
            )
            for field in fields
        ]
    )
    return values if np.isfinite(values).all() else None


def _effective_scale(values: np.ndarray) -> float | None:
    if len(values) < _MIN_REGRESSION_OBSERVATIONS or not np.isfinite(values).all():
        return None
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= _MIN_EFFECTIVE_SCALE:
        return None
    return scale


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return a bounded continuous correlation only above both scale gates."""

    if len(left) != len(right) or len(left) < _MIN_REGRESSION_OBSERVATIONS:
        return float("nan")
    if _effective_scale(left) is None or _effective_scale(right) is None:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _safe_log(values: np.ndarray) -> np.ndarray | None:
    if not np.isfinite(values).all() or (values <= _EPS).any():
        return None
    result = np.log(values)
    return result if np.isfinite(result).all() else None


def _safe_log1p_capacity(
    numerator: np.ndarray, denominator: np.ndarray
) -> np.ndarray | None:
    if not (
        np.isfinite(numerator).all()
        and np.isfinite(denominator).all()
        and (numerator >= 0.0).all()
        and (denominator > _EPS).all()
    ):
        return None
    result = np.log1p(numerator / denominator)
    return result if np.isfinite(result).all() else None


def _safe_anchor_log(
    conv_value: np.ndarray, redemption_value: np.ndarray
) -> np.ndarray | None:
    if not (
        np.isfinite(conv_value).all()
        and np.isfinite(redemption_value).all()
        and (conv_value > _EPS).all()
        and (redemption_value > _EPS).all()
    ):
        return None
    result = np.log(conv_value / redemption_value)
    return result if np.isfinite(result).all() else None


def _one_standardized_oos_residual(training: np.ndarray, current: np.ndarray) -> float:
    """Return a condition-checked OOS residual in prior-target standard deviations.

    ``training`` includes only completed predecessor rows and has first column
    as the target.  A low-scale target/predictor or a collinear standardized
    design returns ``NaN`` rather than an unstable raw coefficient or residual.
    """

    if training.ndim != 2 or current.ndim != 1:
        return float("nan")
    if (
        training.shape[0] != _WINDOW
        or training.shape[1] != len(current)
        or len(current) < 2
    ):
        return float("nan")
    if not (np.isfinite(training).all() and np.isfinite(current).all()):
        return float("nan")
    target = training[:, 0]
    predictors = training[:, 1:]
    target_scale = _effective_scale(target)
    if target_scale is None:
        return float("nan")
    predictor_mean = predictors.mean(axis=0)
    predictor_scale = predictors.std(axis=0, ddof=1)
    if (
        not np.isfinite(predictor_scale).all()
        or (predictor_scale <= _MIN_EFFECTIVE_SCALE).any()
    ):
        return float("nan")
    target_mean = float(target.mean())
    normalized_target = (target - target_mean) / target_scale
    normalized_predictors = (predictors - predictor_mean) / predictor_scale
    design = np.column_stack((np.ones(len(training)), normalized_predictors))
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition) or condition > _MAX_STANDARDIZED_CONDITION_NUMBER:
        return float("nan")
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    coefficients, _, _, _ = np.linalg.lstsq(design, normalized_target, rcond=None)
    current_design = np.r_[1.0, (current[1:] - predictor_mean) / predictor_scale]
    standardized_current = (current[0] - target_mean) / target_scale
    residual = float(standardized_current - np.dot(current_design, coefficients))
    return residual if np.isfinite(residual) else float("nan")


def _credit_motion_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CREDIT_MOTION_SIGNALS}
    values = _numeric_tail(
        frame, ("current_yield", "duration", "convexity"), _WINDOW + 1
    )
    if values is None:
        return out
    yield_changes = np.diff(values[:, 0])
    duration_changes = np.diff(values[:, 1])
    convexity_changes = np.diff(values[:, 2])
    out["cdmr_yield_duration_delta_corr20"] = _safe_corr(
        yield_changes, duration_changes
    )
    out["cdmr_yield_convexity_delta_corr20"] = _safe_corr(
        yield_changes, convexity_changes
    )
    out["cdmr_duration_convexity_delta_corr20"] = _safe_corr(
        duration_changes, convexity_changes
    )
    return out


def _term_supply_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TERM_SUPPLY_SIGNALS}

    level_values = _numeric_tail(frame, ("remain_size", "year_to_mat"), _WINDOW)
    if level_values is not None:
        supply_log = _safe_log(level_values[:, 0])
        if supply_log is not None:
            out["tsrr2_supply_term_level_corr20"] = _safe_corr(
                supply_log, level_values[:, 1]
            )

    ratio_values = _numeric_tail(frame, ("duration", "year_to_mat"), 6)
    if ratio_values is not None and (ratio_values > _EPS).all():
        ratio = np.log(ratio_values[:, 0] / ratio_values[:, 1])
        if np.isfinite(ratio).all():
            out["tsrr2_duration_term_ratio_delta5"] = float(ratio[-1] - ratio[0])

    residual_values = _numeric_tail(
        frame, ("remain_size", "year_to_mat", "duration"), _WINDOW + 1
    )
    if residual_values is not None:
        supply_log = _safe_log(residual_values[:, 0])
        if supply_log is not None:
            design = np.column_stack(
                (supply_log, residual_values[:, 1], residual_values[:, 2])
            )
            out["tsrr2_supply_term_standardized_oos20"] = (
                _one_standardized_oos_residual(
                    design[:-1],
                    design[-1],
                )
            )
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
        amount_capacity = _safe_log1p_capacity(
            capacity_values[:, 1], capacity_values[:, 3]
        )
        volume_capacity = _safe_log1p_capacity(
            capacity_values[:, 2], capacity_values[:, 3]
        )
        if amount_capacity is not None:
            out["cfar_yield_amount_capacity_delta_corr20"] = _safe_corr(
                yield_changes,
                np.diff(amount_capacity),
            )
        if volume_capacity is not None:
            out["cfar_yield_volume_capacity_delta_corr20"] = _safe_corr(
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
            out["cfar_yield_price_supply_standardized_oos20"] = (
                _one_standardized_oos_residual(
                    changes[:-1],
                    changes[-1],
                )
            )
    return out


def _convexity_anchor_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CONVEXITY_ANCHOR_SIGNALS}

    duration_values = _numeric_tail(frame, ("convexity", "duration"), _WINDOW + 1)
    if duration_values is not None:
        out["catr_convexity_duration_delta_corr20"] = _safe_corr(
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
            out["catr_convexity_anchor_delta_corr20"] = _safe_corr(
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
            out["catr_convexity_state_standardized_oos20"] = (
                _one_standardized_oos_residual(
                    design[:-1],
                    design[-1],
                )
            )
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
    price = price.loc[
        np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)
    ].copy()
    if price.empty:
        return out
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    price_calendar = sorted(
        pd.Timestamp(day).normalize() for day in price["trade_date"].unique()
    )
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
            or pd.Timestamp(instrument_history["trade_date"].iloc[-1]).normalize()
            != anchor
        ):
            continue
        metrics = _all_metrics(instrument_history)
        for signal, value in metrics.items():
            if np.isfinite(value):
                out.at[(dt, raw_code), signal] = float(value)
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    output_index = _output_index(ctx)
    score_date = (
        _score_date_from_index(output_index) if not output_index.empty else pd.NaT
    )
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
class FactorMiningDailyCreditDurationTermSupplyDynamicsV2(Factor):
    """Research-only strict-T-1 robust credit/term/supply dynamics kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(
        cls, params: dict | None = None
    ) -> list[DailyFactorRequirement]:
        if params and params.get("signal"):
            _requested_entry(params)
        return _all_requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        out = _feature_frame(ctx)[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
