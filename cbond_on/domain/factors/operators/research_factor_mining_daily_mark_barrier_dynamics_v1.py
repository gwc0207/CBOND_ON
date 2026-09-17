"""Research-only strict-T-1 daily final-mark and barrier-path candidates.

This module deliberately joins every state to the independently sourced
``daily_price`` calendar.  The latest strict-prior price session is the only
anchor: a TWAP/base row that is stale, missing, duplicated, or separated by a
missing price-calendar session produces ``NaN`` instead of a carried state.

The three families are intentionally distinct from the existing daily-TWAP
curve catalogues and static barrier geometry:

* official close versus the completed execution TWAP;
* the out-of-sample final-mark residual after completed intraday phases; and
* the history of occupying, leaving, and revisiting put/call barrier regions.

No files, labels, pools, scores, model state, databases, or live artefacts are
opened here.  Every source is explicitly restricted to ``trade_date <
score_date`` before it is used.
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


KERNEL_NAME = "factor_mining_daily_mark_barrier_dynamics_v1"
CATALOG_VERSION = "20260803_daily_mark_barrier_dynamics_v1"
_LOOKBACK_DAYS = 66
_MARK_WINDOW = 20
_PHASE_TRAINING_DAYS = 40
_RESIDUAL_WINDOW = 20
_BARRIER_WINDOW = 20
_RIDGE_ALPHA = 1e-3
_LOWER_EDGE = 0.20
_UPPER_EDGE = 0.80
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
    """One auditable research-only signal declaration."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_FINAL_MARK_SIGNALS = (
    "fmed_mark_execution_log_basis",
    "fmed_mark_execution_basis_z20",
    "fmed_mark_execution_sign_persistence20",
)
_PHASE_FINALIZATION_SIGNALS = (
    "pcf_final_mark_oos_residual40",
    "pcf_final_mark_residual_z20",
    "pcf_final_mark_residual_autocorr20",
)
_BARRIER_OCCUPANCY_SIGNALS = (
    "boh_edge_state_flip_rate20",
    "boh_outer_band_dwell20",
    "boh_location_innovation20",
)

_CATALOG = (
    _entries(
        "final_mark_execution_dislocation",
        _FINAL_MARK_SIGNALS,
        "The final official daily close can differ from the completed 14:42-14:57 execution TWAP, a state not represented by a TWAP-only curve.",
    )
    + _entries(
        "phase_conditioned_finalization",
        _PHASE_FINALIZATION_SIGNALS,
        "A final official-mark residual is evaluated out of sample after prior-only lunch, late, and execution phase information, rather than using a same-window fitted residual.",
    )
    + _entries(
        "barrier_occupancy_hysteresis",
        _BARRIER_OCCUPANCY_SIGNALS,
        "The recent path through put/middle/call barrier regions is distinct from a latest-day barrier location or a contract-term revision level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "final_mark_execution_dislocation": _FINAL_MARK_SIGNALS,
    "phase_conditioned_finalization": _PHASE_FINALIZATION_SIGNALS,
    "barrier_occupancy_hysteresis": _BARRIER_OCCUPANCY_SIGNALS,
}

_FAMILY_TWAP_FIELDS: dict[str, tuple[str, ...]] = {
    "final_mark_execution_dislocation": (
        "twap_1430_1442",
        "twap_1442_1457",
    ),
    "phase_conditioned_finalization": (
        "twap_1100_1130",
        "twap_1300_1330",
        "twap_1400_1430",
        "twap_1430_1442",
        "twap_1442_1457",
    ),
    "barrier_occupancy_hysteresis": (),
}
_FAMILY_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "final_mark_execution_dislocation": (),
    "phase_conditioned_finalization": (),
    "barrier_occupancy_hysteresis": (
        "cb_put_price",
        "cb_call_price",
        "stock_close_price",
    ),
}

FORMULAS: dict[str, str] = {
    "fmed_mark_execution_log_basis": "log(close_price / twap_1442_1457) on the latest strict-prior exact price/TWAP session.",
    "fmed_mark_execution_basis_z20": "latest final-mark/execution-TWAP log basis standardized against the preceding 20 exact completed sessions.",
    "fmed_mark_execution_sign_persistence20": "absolute mean sign of the last 20 exact completed final-mark/execution-TWAP log bases.",
    "pcf_final_mark_oos_residual40": "B_A - Bhat_A where fixed ridge B~(lunch_gap, late_move, execution_move) trains only on the preceding 40 sessions A-40..A-1.",
    "pcf_final_mark_residual_z20": "latest sequential prior-window phase-conditioned residual standardized against the preceding 20 sequential OOS residuals.",
    "pcf_final_mark_residual_autocorr20": "lag-one autocorrelation of the latest 20 sequential prior-window phase-conditioned OOS residuals.",
    "boh_edge_state_flip_rate20": "fraction of 20 adjacent transitions through put-edge, middle, and call-edge barrier states over the last 21 exact sessions.",
    "boh_outer_band_dwell20": "fraction of the last 20 exact sessions whose normalized stock barrier location lies in an outer put/call band.",
    "boh_location_innovation20": "latest normalized stock barrier location standardized against the preceding 20 exact sessions.",
}


def daily_mark_barrier_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_mark_barrier_dynamics_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _requirements_for_family(family: str) -> list[DailyFactorRequirement]:
    requirements = [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code", "close_price"),
            _LOOKBACK_DAYS,
        )
    ]
    twap_fields = _FAMILY_TWAP_FIELDS[family]
    if twap_fields:
        requirements.append(
            DailyFactorRequirement(
                "market_cbond.daily_twap",
                ("exchange_code", *twap_fields),
                _LOOKBACK_DAYS,
            )
        )
    base_fields = _FAMILY_BASE_FIELDS[family]
    if base_fields:
        requirements.append(
            DailyFactorRequirement(
                "market_cbond.daily_base",
                ("exchange_code", *base_fields),
                _LOOKBACK_DAYS,
            )
        )
    return requirements


def _all_requirements() -> list[DailyFactorRequirement]:
    twap_fields = tuple(sorted({field for fields in _FAMILY_TWAP_FIELDS.values() for field in fields}))
    base_fields = tuple(sorted({field for fields in _FAMILY_BASE_FIELDS.values() for field in fields}))
    requirements = [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code", "close_price"),
            _LOOKBACK_DAYS,
        )
    ]
    if twap_fields:
        requirements.append(
            DailyFactorRequirement(
                "market_cbond.daily_twap",
                ("exchange_code", *twap_fields),
                _LOOKBACK_DAYS,
            )
        )
    if base_fields:
        requirements.append(
            DailyFactorRequirement(
                "market_cbond.daily_base",
                ("exchange_code", *base_fields),
                _LOOKBACK_DAYS,
            )
        )
    return requirements


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
    """Read one declared source and exclude score-day/future observations."""

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
    days = pd.to_datetime(index.get_level_values("dt"), errors="coerce").normalize().unique()
    parsed = [pd.Timestamp(day) for day in days if not pd.isna(day)]
    if len(parsed) != 1:
        raise ValueError(f"{KERNEL_NAME} requires one valid score date per context")
    return parsed[0]


def _last_n_are_consecutive(frame: pd.DataFrame, count: int) -> bool:
    """Require a complete sequence on the independent daily-price calendar."""

    if len(frame) < count:
        return False
    if "__price_session_index" not in frame.columns:
        return True
    positions = pd.to_numeric(frame.tail(count)["__price_session_index"], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _complete_numeric_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count or not _last_n_are_consecutive(frame, count):
        return None
    values = np.column_stack(
        [pd.to_numeric(frame.tail(count)[column], errors="coerce").to_numpy(dtype="float64") for column in columns]
    )
    return values if np.isfinite(values).all() else None


def _safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(len(numerator), np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > _EPS)
        & (denominator > _EPS)
    )
    out[valid] = np.log(numerator[valid] / denominator[valid])
    return out


def _z_last_against_prior(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    scale = float(np.std(prior, ddof=1))
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / scale)


def _safe_autocorr(values: np.ndarray) -> float:
    if len(values) < 4 or not np.isfinite(values).all():
        return float("nan")
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _mark_basis(values: np.ndarray) -> np.ndarray:
    """Return B=log(official close / completed execution TWAP)."""

    return _safe_log_ratio(values[:, 0], values[:, 2])


def _final_mark_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _FINAL_MARK_SIGNALS}
    fields = ("close_price", "twap_1430_1442", "twap_1442_1457")
    latest = _complete_numeric_tail(frame, fields, 1)
    if latest is not None:
        basis = _mark_basis(latest)
        if np.isfinite(basis[0]):
            out["fmed_mark_execution_log_basis"] = float(basis[0])

    recent = _complete_numeric_tail(frame, fields, _MARK_WINDOW + 1)
    if recent is not None:
        basis = _mark_basis(recent)
        out["fmed_mark_execution_basis_z20"] = _z_last_against_prior(basis)

    persistence = _complete_numeric_tail(frame, fields, _MARK_WINDOW)
    if persistence is not None:
        basis = _mark_basis(persistence)
        if np.isfinite(basis).all():
            out["fmed_mark_execution_sign_persistence20"] = float(np.abs(np.mean(np.sign(basis))))
    return out


def _phase_design(values: np.ndarray) -> np.ndarray:
    """Return [final-mark basis, lunch gap, late move, execution move]."""

    close, lunch_end, reopen, afternoon_end, late, execution = values.T
    basis = _safe_log_ratio(close, execution)
    lunch_gap = _safe_log_ratio(reopen, lunch_end)
    late_move = _safe_log_ratio(late, afternoon_end)
    execution_move = _safe_log_ratio(execution, late)
    return np.column_stack((basis, lunch_gap, late_move, execution_move))


def _one_phase_oos_residual(training: np.ndarray, current: np.ndarray) -> float:
    """Fit fixed ridge on prior rows only, then evaluate one held-out row."""

    if training.shape != (_PHASE_TRAINING_DAYS, 4) or current.shape != (4,):
        return float("nan")
    if not (np.isfinite(training).all() and np.isfinite(current).all()):
        return float("nan")
    target = training[:, 0]
    predictors = training[:, 1:]
    mean = predictors.mean(axis=0)
    scale = predictors.std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= _EPS).any():
        return float("nan")
    design = np.column_stack((np.ones(len(training)), (predictors - mean) / scale))
    penalty = np.diag(np.r_[0.0, np.full(design.shape[1] - 1, _RIDGE_ALPHA)])
    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return float("nan")
    current_design = np.r_[1.0, (current[1:] - mean) / scale]
    residual = float(current[0] - np.dot(current_design, beta))
    return residual if np.isfinite(residual) else float("nan")


def _phase_oos_residual_sequence(values: np.ndarray) -> np.ndarray:
    """Generate sequential held-out residuals without fitting on their targets."""

    if len(values) <= _PHASE_TRAINING_DAYS:
        return np.array([], dtype="float64")
    design = _phase_design(values)
    return np.asarray(
        [
            _one_phase_oos_residual(
                design[position - _PHASE_TRAINING_DAYS : position],
                design[position],
            )
            for position in range(_PHASE_TRAINING_DAYS, len(design))
        ],
        dtype="float64",
    )


def _phase_finalization_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _PHASE_FINALIZATION_SIGNALS}
    fields = (
        "close_price",
        "twap_1100_1130",
        "twap_1300_1330",
        "twap_1400_1430",
        "twap_1430_1442",
        "twap_1442_1457",
    )
    latest_values = _complete_numeric_tail(frame, fields, _PHASE_TRAINING_DAYS + 1)
    if latest_values is not None:
        residuals = _phase_oos_residual_sequence(latest_values)
        if len(residuals) == 1 and np.isfinite(residuals[0]):
            out["pcf_final_mark_oos_residual40"] = float(residuals[0])

    z_values = _complete_numeric_tail(
        frame,
        fields,
        _PHASE_TRAINING_DAYS + _RESIDUAL_WINDOW + 1,
    )
    if z_values is not None:
        residuals = _phase_oos_residual_sequence(z_values)
        if len(residuals) == _RESIDUAL_WINDOW + 1 and np.isfinite(residuals).all():
            out["pcf_final_mark_residual_z20"] = _z_last_against_prior(residuals)

    autocorr_values = _complete_numeric_tail(
        frame,
        fields,
        _PHASE_TRAINING_DAYS + _RESIDUAL_WINDOW,
    )
    if autocorr_values is not None:
        residuals = _phase_oos_residual_sequence(autocorr_values)
        if len(residuals) == _RESIDUAL_WINDOW and np.isfinite(residuals).all():
            out["pcf_final_mark_residual_autocorr20"] = _safe_autocorr(residuals)
    return out


def _barrier_location(values: np.ndarray) -> np.ndarray:
    put, call, stock = values.T
    out = np.full(len(values), np.nan, dtype="float64")
    width = call - put
    valid = (
        np.isfinite(put)
        & np.isfinite(call)
        & np.isfinite(stock)
        & (put > _EPS)
        & (call > _EPS)
        & (stock > _EPS)
        & (width > _EPS)
    )
    out[valid] = (stock[valid] - put[valid]) / width[valid]
    return out


def _barrier_state(location: np.ndarray) -> np.ndarray:
    state = np.full(len(location), -1, dtype="int64")
    finite = np.isfinite(location)
    state[finite & (location <= _LOWER_EDGE)] = 0
    state[finite & (location > _LOWER_EDGE) & (location < _UPPER_EDGE)] = 1
    state[finite & (location >= _UPPER_EDGE)] = 2
    return state


def _barrier_occupancy_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _BARRIER_OCCUPANCY_SIGNALS}
    fields = ("cb_put_price", "cb_call_price", "stock_close_price")
    transitions = _complete_numeric_tail(frame, fields, _BARRIER_WINDOW + 1)
    if transitions is not None:
        location = _barrier_location(transitions)
        if np.isfinite(location).all():
            state = _barrier_state(location)
            out["boh_edge_state_flip_rate20"] = float(np.mean(state[1:] != state[:-1]))
            out["boh_location_innovation20"] = _z_last_against_prior(location)

    dwell = _complete_numeric_tail(frame, fields, _BARRIER_WINDOW)
    if dwell is not None:
        location = _barrier_location(dwell)
        if np.isfinite(location).all():
            outer = (location <= _LOWER_EDGE) | (location >= _UPPER_EDGE)
            out["boh_outer_band_dwell20"] = float(np.mean(outer))
    return out


_FAMILY_METRICS = {
    "final_mark_execution_dislocation": _final_mark_metrics,
    "phase_conditioned_finalization": _phase_finalization_metrics,
    "barrier_occupancy_hysteresis": _barrier_occupancy_metrics,
}


def _build_family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    out_index = _output_index(ctx)
    signals = _FAMILY_SIGNALS[family]
    out = pd.DataFrame(index=out_index, columns=signals, dtype="float64")
    if out.empty:
        return out

    score_date = _score_date_from_index(out_index)
    price = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    price["close_price"] = pd.to_numeric(price["close_price"], errors="coerce")
    price = price.loc[np.isfinite(price["close_price"]) & (price["close_price"] > _EPS)].copy()
    if price.empty:
        return out

    twap_fields = _FAMILY_TWAP_FIELDS[family]
    base_fields = _FAMILY_BASE_FIELDS[family]
    if twap_fields:
        source = _strict_history_source(
            ctx,
            source="market_cbond.daily_twap",
            fields=twap_fields,
            score_date=score_date,
        )
    elif base_fields:
        source = _strict_history_source(
            ctx,
            source="market_cbond.daily_base",
            fields=base_fields,
            score_date=score_date,
        )
    else:
        raise ValueError(f"{KERNEL_NAME} family has no declared daily source: {family}")

    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    price_calendar = sorted(pd.Timestamp(day).normalize() for day in price["trade_date"].unique())
    session_index = {day: position for position, day in enumerate(price_calendar)}
    # Exact same-date join establishes both the independent price anchor and
    # missing-session propagation for every non-price source.
    history = source.merge(
        price.loc[:, ["trade_date", "code", "close_price"]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort")
    history["__price_session_index"] = history["trade_date"].map(session_index)
    groups = {str(code): group.reset_index(drop=True) for code, group in history.groupby("code", sort=False)}
    calculator = _FAMILY_METRICS[family]

    for dt, raw_code in out_index:
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
        metrics = calculator(instrument_history)
        for signal in signals:
            value = metrics.get(signal, float("nan"))
            if np.isfinite(value):
                out.at[(dt, raw_code), signal] = float(value)
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
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningDailyMarkBarrierDynamicsV1(Factor):
    """Research-only strict-T-1 final-mark and barrier-path kernel."""

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
    "FactorMiningDailyMarkBarrierDynamicsV1",
    "daily_mark_barrier_dynamics_catalog",
    "factor_mining_catalog",
]
