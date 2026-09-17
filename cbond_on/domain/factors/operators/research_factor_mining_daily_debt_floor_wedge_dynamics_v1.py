"""Research-only strict-T-1 debt-floor wedge *dynamics* catalogue.

The existing research pool already contains the raw T-1
``debt_puredebt_ratio - puredebt_prem_ratio`` level, a cross-sectional rank
interaction of that level with YTM, and pure-redemption-value floor measures.
This module deliberately emits none of those levels.  It instead studies how
the debt-floor wedge changes, accelerates, persists, and deviates from its own
historical YTM/duration-conditioned relation.

Only declared ``market_cbond.daily_price`` and ``market_cbond.daily_base``
context tables are consumed.  ``daily_price`` establishes the independent
latest completed T-1 session anchor; a base history is date-key joined to that
calendar, and an output is eligible only if it has a base row on the exact
same anchor session.  The context can contain score-date data, so every source
is independently restricted to ``trade_date < score_date`` before any
calculation.  No files, database, labels, PnL, scores, pool/mask state, or
live artefacts are opened.  This import-only candidate is intentionally absent
from ``defs.__init__``, factor contracts, configurations, and live paths.
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


KERNEL_NAME = "factor_mining_daily_debt_floor_wedge_dynamics_v1"
CATALOG_VERSION = "20260803_daily_debt_floor_wedge_dynamics_v1"
_LOOKBACK_DAYS = 66
_RESIDUAL_TRAINING_DAYS = 20
_PERSISTENCE_DAYS = 20
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only debt-floor dynamics candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_CHANGE_SIGNALS = (
    "dfw_wedge_delta1",
    "dfw_wedge_delta5",
    "dfw_wedge_acceleration5",
)
_PERSISTENCE_SIGNALS = (
    "dfw_wedge_change_autocorrelation20",
    "dfw_wedge_change_sign_agreement20",
    "dfw_wedge_directional_run_share20",
)
_INNOVATION_SIGNALS = (
    "dfw_ytm_duration_oos_residual20",
    "dfw_ytm_duration_residual_z20",
    "dfw_ytm_duration_residual_acceleration5",
)

_CATALOG = (
    _entries(
        "debt_floor_wedge_change_acceleration",
        _CHANGE_SIGNALS,
        "One- and five-session changes plus current-versus-prior change acceleration describe debt-floor wedge repricing, not its raw T-1 gap.",
    )
    + _entries(
        "debt_floor_wedge_change_persistence",
        _PERSISTENCE_SIGNALS,
        "Serial correlation, directional agreement, and run concentration of completed wedge changes distinguish persistent repricing from a static floor or premium level.",
    )
    + _entries(
        "debt_floor_wedge_ytm_duration_innovation",
        _INNOVATION_SIGNALS,
        "The latest wedge innovation is an out-of-sample residual from its own prior 20-session YTM/duration relation; residual dynamics are distinct from raw YTM, duration, or a cross-sectional lattice.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "debt_floor_wedge_change_acceleration": _CHANGE_SIGNALS,
    "debt_floor_wedge_change_persistence": _PERSISTENCE_SIGNALS,
    "debt_floor_wedge_ytm_duration_innovation": _INNOVATION_SIGNALS,
}

# W_t=debt_puredebt_ratio_t-puredebt_prem_ratio_t.  All t below denote
# completed daily-base sessions ending at exact price-calendar T-1.  For the
# innovation family epsilon_t is an OOS residual: beta_t is fit on t-20..t-1
# in W=a+b*standardize(ytm)+c*standardize(duration), then epsilon_t=W_t-X_t beta_t.
FORMULAS: dict[str, str] = {
    "dfw_wedge_delta1": "W_t-W_t-1",
    "dfw_wedge_delta5": "W_t-W_t-5",
    "dfw_wedge_acceleration5": "(W_t-W_t-1)-mean(W_i-W_i-1 for i=t-4..t-1)",
    "dfw_wedge_change_autocorrelation20": "corr(delta W_i,delta W_i-1) over the latest 20 completed changes",
    "dfw_wedge_change_sign_agreement20": "mean(sign(delta W_i)*sign(delta W_i-1)) over nonzero adjacent latest-20 changes",
    "dfw_wedge_directional_run_share20": "longest contiguous nonzero same-sign delta-W run / 20",
    "dfw_ytm_duration_oos_residual20": "epsilon_t from prior-20-session standardized OLS(W | ytm,duration)",
    "dfw_ytm_duration_residual_z20": "z(epsilon_t versus preceding 20 sequential OOS residuals)",
    "dfw_ytm_duration_residual_acceleration5": "epsilon_t-mean(epsilon_i for i=t-5..t-1)",
}

_BASE_FIELDS = (
    "debt_puredebt_ratio",
    "puredebt_prem_ratio",
    "ytm",
    "duration",
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


def daily_debt_floor_wedge_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return daily_debt_floor_wedge_dynamics_catalog()


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
            ("exchange_code", "close_price"),
            _LOOKBACK_DAYS,
        ),
        DailyFactorRequirement(
            "market_cbond.daily_base",
            ("exchange_code", *_BASE_FIELDS),
            _LOOKBACK_DAYS,
        ),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize DataHub codes while retaining exchange disambiguation."""

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
    """Read declared daily context only and reject score-date/future rows."""

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
    """Require history to occupy consecutive daily-price sessions, not base rows."""

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
    if not np.isfinite(values).all() or (values[:, 3] <= _EPS).any():
        return None
    return values


def _wedge(values: np.ndarray) -> np.ndarray:
    """Return the unreported internal wedge W, never a catalogue output itself."""

    return values[:, 0] - values[:, 1]


def _safe_autocorr(values: np.ndarray) -> float:
    if len(values) < 4 or not np.isfinite(values).all():
        return float("nan")
    left = values[:-1]
    right = values[1:]
    if float(np.std(left)) <= _EPS or float(np.std(right)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _longest_same_sign_run_share(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values).all():
        return float("nan")
    signs = np.sign(values)
    longest = 0
    run = 0
    prior = 0.0
    for sign in signs:
        if sign == 0.0:
            run = 0
            prior = 0.0
        elif sign == prior:
            run += 1
        else:
            run = 1
            prior = sign
        longest = max(longest, run)
    return float(longest / len(values)) if longest else 0.0


def _change_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CHANGE_SIGNALS}
    values = _complete_numeric_tail(frame, _BASE_FIELDS, 6)
    if values is None:
        return out
    wedge = _wedge(values)
    changes = np.diff(wedge)
    out["dfw_wedge_delta1"] = float(changes[-1])
    out["dfw_wedge_delta5"] = float(wedge[-1] - wedge[0])
    out["dfw_wedge_acceleration5"] = float(changes[-1] - np.mean(changes[:-1]))
    return out


def _persistence_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _PERSISTENCE_SIGNALS}
    values = _complete_numeric_tail(frame, _BASE_FIELDS, _PERSISTENCE_DAYS + 1)
    if values is None:
        return out
    changes = np.diff(_wedge(values))
    out["dfw_wedge_change_autocorrelation20"] = _safe_autocorr(changes)
    left = np.sign(changes[:-1])
    right = np.sign(changes[1:])
    nonzero = (left != 0.0) & (right != 0.0)
    if int(nonzero.sum()) >= 12:
        out["dfw_wedge_change_sign_agreement20"] = float(np.mean(left[nonzero] * right[nonzero]))
    out["dfw_wedge_directional_run_share20"] = _longest_same_sign_run_share(changes)
    return out


def _one_oos_residual(training: np.ndarray, current: np.ndarray) -> float:
    """Fit standardized OLS only on past rows and evaluate one held-out row."""

    if training.shape != (_RESIDUAL_TRAINING_DAYS, 3) or current.shape != (3,):
        return float("nan")
    if not (np.isfinite(training).all() and np.isfinite(current).all()) or (training[:, 2] <= _EPS).any() or current[2] <= _EPS:
        return float("nan")
    y = training[:, 0]
    predictors = training[:, 1:]
    mean = predictors.mean(axis=0)
    scale = predictors.std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= _EPS).any():
        return float("nan")
    design = np.column_stack((np.ones(len(training)), (predictors - mean) / scale))
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return float("nan")
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    current_design = np.r_[1.0, (current[1:] - mean) / scale]
    residual = float(current[0] - np.dot(current_design, beta))
    return residual if np.isfinite(residual) else float("nan")


def _oos_residual_sequence(values: np.ndarray) -> np.ndarray:
    """Return sequential prior-window residuals for a complete local history."""

    if len(values) <= _RESIDUAL_TRAINING_DAYS:
        return np.array([], dtype="float64")
    wedge = _wedge(values)
    design_values = np.column_stack((wedge, values[:, 2], values[:, 3]))
    residuals = [
        _one_oos_residual(
            design_values[position - _RESIDUAL_TRAINING_DAYS : position],
            design_values[position],
        )
        for position in range(_RESIDUAL_TRAINING_DAYS, len(design_values))
    ]
    return np.asarray(residuals, dtype="float64")


def _innovation_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _INNOVATION_SIGNALS}

    latest_values = _complete_numeric_tail(frame, _BASE_FIELDS, _RESIDUAL_TRAINING_DAYS + 1)
    if latest_values is not None:
        latest = _oos_residual_sequence(latest_values)
        if len(latest) == 1 and np.isfinite(latest[0]):
            out["dfw_ytm_duration_oos_residual20"] = float(latest[0])

    acceleration_values = _complete_numeric_tail(frame, _BASE_FIELDS, _RESIDUAL_TRAINING_DAYS + 6)
    if acceleration_values is not None:
        residuals = _oos_residual_sequence(acceleration_values)
        if len(residuals) == 6 and np.isfinite(residuals).all():
            out["dfw_ytm_duration_residual_acceleration5"] = float(residuals[-1] - np.mean(residuals[:-1]))

    z_values = _complete_numeric_tail(frame, _BASE_FIELDS, 2 * _RESIDUAL_TRAINING_DAYS + 1)
    if z_values is not None:
        residuals = _oos_residual_sequence(z_values)
        if len(residuals) == _RESIDUAL_TRAINING_DAYS + 1 and np.isfinite(residuals).all():
            prior = residuals[:-1]
            scale = float(np.std(prior, ddof=1))
            if np.isfinite(scale) and scale > _EPS:
                out["dfw_ytm_duration_residual_z20"] = float((residuals[-1] - np.mean(prior)) / scale)
    return out


def _all_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_change_metrics(frame))
    out.update(_persistence_metrics(frame))
    out.update(_innovation_metrics(frame))
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
        fields=("close_price",),
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
        price.loc[:, ["trade_date", "code"]],
        on=["trade_date", "code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["code", "trade_date"], kind="mergesort")
    history["__price_session_index"] = history["trade_date"].map(price_session_index)
    groups = {str(code): group.reset_index(drop=True) for code, group in history.groupby("code", sort=False)}
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
    out_index = _output_index(ctx)
    score_date = _score_date_from_index(out_index) if not out_index.empty else pd.NaT
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
class FactorMiningDailyDebtFloorWedgeDynamicsV1(Factor):
    """Research-only strict-T-1 debt-floor wedge dynamics kernel."""

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
