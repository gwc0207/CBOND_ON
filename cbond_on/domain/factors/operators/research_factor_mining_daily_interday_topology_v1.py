"""Research-only strict-T-1 interday and contract-payoff candidates.

Every daily state is anchored to the latest exact completed ``daily_price``
session before the score date.  Other sources must have an exact same-session
row for that code; stale source rows, duplicate rows, an interior missing
price-calendar session, and score/future observations fail closed to ``NaN``.

The three families deliberately address distinct hypotheses:

* whether a completed official-final-mark dislocation transmits to the next
  session's opening execution window;
* whether that final-mark dislocation is unusual after the completed stock
  return and stock-volatility state; and
* whether nonlinear call/put-floor payoff topology is unusual after its
  simpler state variables.

No files, database, labels, pools, model state, scores, PnL, live artefacts,
or production FactorStore are opened here.  All daily context is explicitly
restricted to ``trade_date < score_date`` before use.
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


KERNEL_NAME = "factor_mining_daily_interday_topology_v1"
CATALOG_VERSION = "20260803_daily_interday_topology_v1"
_LOOKBACK_DAYS = 66
_TRAINING_DAYS = 40
_SHORT_TRAINING_DAYS = 20
_RESIDUAL_WINDOW = 20
_RIDGE_ALPHA = 1e-3
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
    """One auditable research-only candidate declaration."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_INTERDAY_SIGNALS = (
    "ift_centered_basis_forecast40",
    "ift_transmission_beta40",
    "ift_direction_hit_rate40",
)
_STOCK_RESIDUAL_SIGNALS = (
    "scmr_mark_stock_oos_residual40",
    "scmr_residual_z20",
    "scmr_stock_beta_shift20_40",
)
_CONTRACT_TOPOLOGY_SIGNALS = (
    "cft_call_floor_oos_residual40",
    "cft_put_floor_oos_residual40",
    "cft_side_topology_residual_z20",
)

_CATALOG = (
    _entries(
        "interday_finalization_transmission",
        _INTERDAY_SIGNALS,
        "A completed official-final-mark versus execution-TWAP basis can transmit into the following session's opening execution return, separately from an intraday curve shape.",
    )
    + _entries(
        "stock_conditioned_mark_residual",
        _STOCK_RESIDUAL_SIGNALS,
        "The official-final-mark versus execution-TWAP basis can be surprising after strictly completed stock return and stock-volatility information, unlike a bond-close stock tracking error.",
    )
    + _entries(
        "contract_floor_payoff_topology",
        _CONTRACT_TOPOLOGY_SIGNALS,
        "Nonlinear payoff topology around call and put barriers can depart from a prior-only linear floor, moneyness, maturity, and volatility relation rather than restating a static floor level.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "interday_finalization_transmission": _INTERDAY_SIGNALS,
    "stock_conditioned_mark_residual": _STOCK_RESIDUAL_SIGNALS,
    "contract_floor_payoff_topology": _CONTRACT_TOPOLOGY_SIGNALS,
}

_FAMILY_PRICE_FIELDS: dict[str, tuple[str, ...]] = {
    "interday_finalization_transmission": (
        "close_price",
        "prev_close_price",
        "act_prev_close_price",
    ),
    "stock_conditioned_mark_residual": ("close_price",),
    "contract_floor_payoff_topology": ("close_price",),
}
_FAMILY_TWAP_FIELDS: dict[str, tuple[str, ...]] = {
    "interday_finalization_transmission": (
        "twap_0930_0935",
        "twap_1442_1457",
    ),
    "stock_conditioned_mark_residual": ("twap_1442_1457",),
    "contract_floor_payoff_topology": (),
}
_FAMILY_BASE_FIELDS: dict[str, tuple[str, ...]] = {
    "interday_finalization_transmission": (),
    "stock_conditioned_mark_residual": (
        "stk_prev_close_price",
        "stk_act_prev_close_price",
        "stk_close_price",
        "stock_volatility",
    ),
    "contract_floor_payoff_topology": (
        "cb_close_price",
        "pure_redemption_value",
        "cb_put_price",
        "cb_call_price",
        "stock_close_price",
        "year_to_mat",
        "stock_volatility",
    ),
}

FORMULAS: dict[str, str] = {
    "ift_centered_basis_forecast40": (
        "E[log(twap_0930_0935[t+1]/P*_t+1)|B_t] - mean(open_return) from a fixed ridge on the 40 adjacent strict-prior pairs ending B_(A-1)->open_A, evaluated at B_A; B_A->open_T is never observed. P*=act_prev_close_price when positive, otherwise prev_close_price when positive."
    ),
    "ift_transmission_beta40": (
        "Standardized fixed-ridge coefficient of B_t=log(close_price_t/twap_1442_1457_t) in the same 40 adjacent B_t->open_return_(t+1) pairs."
    ),
    "ift_direction_hit_rate40": (
        "Fraction of the same 40 completed adjacent pairs with matching signs after separately centering B_t and next-session opening return."
    ),
    "scmr_mark_stock_oos_residual40": (
        "B_A-Bhat_A where fixed ridge B~(strict completed stock return,stock_volatility) trains only on A-40..A-1."
    ),
    "scmr_residual_z20": (
        "Current stock-conditioned OOS mark residual standardized against the preceding 20 sequential prior-window OOS residuals."
    ),
    "scmr_stock_beta_shift20_40": (
        "Difference between the standardized stock-return ridge coefficients fit on A-20..A-1 and A-40..A-1; neither fit contains B_A."
    ),
    "cft_call_floor_oos_residual40": (
        "OOS residual of F_A*call_proximity(x_A) after a prior-40 fixed ridge on F,x,year_to_mat,stock_volatility, where F=log(cb_close_price/pure_redemption_value) and x=(stock_close_price-cb_put_price)/(cb_call_price-cb_put_price)."
    ),
    "cft_put_floor_oos_residual40": (
        "OOS residual of F_A*put_proximity(x_A) after the same prior-only linear state model."
    ),
    "cft_side_topology_residual_z20": (
        "OOS residual z-score for F*(call_proximity(x)-put_proximity(x)), against the preceding 20 sequential prior-window OOS residuals."
    ),
}


def daily_interday_topology_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research-only catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic research expansion runner."""

    return daily_interday_topology_catalog()


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
            ("exchange_code", *_FAMILY_PRICE_FIELDS[family]),
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
    price_fields = tuple(
        dict.fromkeys(field for fields in _FAMILY_PRICE_FIELDS.values() for field in fields)
    )
    twap_fields = tuple(
        dict.fromkeys(field for fields in _FAMILY_TWAP_FIELDS.values() for field in fields)
    )
    base_fields = tuple(
        dict.fromkeys(field for fields in _FAMILY_BASE_FIELDS.values() for field in fields)
    )
    requirements = [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code", *price_fields),
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
        raise ValueError(
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
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
    """Require complete source history on the independent daily-price calendar."""

    if len(frame) < count:
        return False
    positions = pd.to_numeric(
        frame.tail(count)["__price_session_index"], errors="coerce"
    ).to_numpy(dtype="float64")
    if not np.isfinite(positions).all():
        return False
    expected = np.arange(positions[-1] - count + 1, positions[-1] + 1, dtype="float64")
    return bool(np.array_equal(positions, expected))


def _numeric_tail(frame: pd.DataFrame, columns: tuple[str, ...], count: int) -> np.ndarray | None:
    if len(frame) < count or not _last_n_are_consecutive(frame, count):
        return None
    return np.column_stack(
        [pd.to_numeric(frame.tail(count)[column], errors="coerce").to_numpy(dtype="float64") for column in columns]
    )


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


def _adjusted_prior_close(preferred: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Use adjusted prior close when valid, otherwise the explicitly defined raw prior."""

    out = np.full(len(preferred), np.nan, dtype="float64")
    preferred_valid = np.isfinite(preferred) & (preferred > _EPS)
    fallback_valid = np.isfinite(fallback) & (fallback > _EPS)
    out[preferred_valid] = preferred[preferred_valid]
    use_fallback = ~preferred_valid & fallback_valid
    out[use_fallback] = fallback[use_fallback]
    return out


def _z_last_against_prior(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    prior = values[:-1]
    scale = float(np.std(prior, ddof=1))
    if not np.isfinite(scale) or scale <= _EPS:
        return float("nan")
    return float((values[-1] - float(np.mean(prior))) / scale)


def _fit_ridge(
    training: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float, np.ndarray] | None:
    """Fit a fixed standardized ridge on historical rows and evaluate current once."""

    if training.ndim != 2 or current.ndim != 1 or training.shape[1] != len(current):
        return None
    if len(training) < 3 or training.shape[1] < 2:
        return None
    if not (np.isfinite(training).all() and np.isfinite(current).all()):
        return None
    target = training[:, 0]
    predictors = training[:, 1:]
    mean = predictors.mean(axis=0)
    scale = predictors.std(axis=0, ddof=1)
    if not np.isfinite(scale).all() or (scale <= _EPS).any():
        return None
    design = np.column_stack((np.ones(len(training)), (predictors - mean) / scale))
    penalty = np.diag(np.r_[0.0, np.full(design.shape[1] - 1, _RIDGE_ALPHA)])
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        return None
    current_design = np.r_[1.0, (current[1:] - mean) / scale]
    prediction = float(np.dot(current_design, coefficients))
    residual = float(current[0] - prediction)
    standardized_beta = coefficients[1:].astype("float64", copy=False)
    if not (np.isfinite(prediction) and np.isfinite(residual) and np.isfinite(standardized_beta).all()):
        return None
    return prediction, residual, standardized_beta


def _one_oos_residual(training: np.ndarray, current: np.ndarray) -> float:
    """Return a held-out residual; the current target never enters fitting."""

    fitted = _fit_ridge(training, current)
    return float("nan") if fitted is None else float(fitted[1])


def _oos_residual_sequence(design_values: np.ndarray) -> np.ndarray:
    """Generate sequential fixed-window OOS residuals for a complete history."""

    if len(design_values) <= _TRAINING_DAYS:
        return np.array([], dtype="float64")
    return np.asarray(
        [
            _one_oos_residual(
                design_values[position - _TRAINING_DAYS : position],
                design_values[position],
            )
            for position in range(_TRAINING_DAYS, len(design_values))
        ],
        dtype="float64",
    )


def _interday_design(values: np.ndarray) -> np.ndarray:
    """Return [final-mark basis, completed-session opening return]."""

    close, prior_close, adjusted_prior_close, opening_twap, execution_twap = values.T
    basis = _safe_log_ratio(close, execution_twap)
    opening_return = _safe_log_ratio(
        opening_twap,
        _adjusted_prior_close(adjusted_prior_close, prior_close),
    )
    return np.column_stack((basis, opening_return))


def _interday_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _INTERDAY_SIGNALS}
    fields = (
        "close_price",
        "prev_close_price",
        "act_prev_close_price",
        "twap_0930_0935",
        "twap_1442_1457",
    )
    values = _numeric_tail(frame, fields, _TRAINING_DAYS + 1)
    if values is None:
        return out
    design = _interday_design(values)
    if not np.isfinite(design).all():
        return out
    targets = design[1:, 1]
    predictors = design[:-1, 0]
    training = np.column_stack((targets, predictors))
    current = np.array([0.0, design[-1, 0]], dtype="float64")
    fitted = _fit_ridge(training, current)
    if fitted is None:
        return out
    prediction, _, beta = fitted
    out["ift_centered_basis_forecast40"] = float(prediction - np.mean(targets))
    out["ift_transmission_beta40"] = float(beta[0])
    centered_predictor = predictors - np.mean(predictors)
    centered_target = targets - np.mean(targets)
    nonzero = (np.abs(centered_predictor) > _EPS) & (np.abs(centered_target) > _EPS)
    if int(nonzero.sum()) >= _TRAINING_DAYS // 2:
        out["ift_direction_hit_rate40"] = float(
            np.mean(np.sign(centered_predictor[nonzero]) == np.sign(centered_target[nonzero]))
        )
    return out


def _stock_mark_design(values: np.ndarray) -> np.ndarray:
    """Return [mark basis, strict completed stock return, stock volatility]."""

    close, execution_twap, stk_prior, stk_adjusted_prior, stk_close, stock_volatility = values.T
    basis = _safe_log_ratio(close, execution_twap)
    stock_return = _safe_log_ratio(stk_close, _adjusted_prior_close(stk_adjusted_prior, stk_prior))
    return np.column_stack((basis, stock_return, stock_volatility))


def _stock_residual_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _STOCK_RESIDUAL_SIGNALS}
    fields = (
        "close_price",
        "twap_1442_1457",
        "stk_prev_close_price",
        "stk_act_prev_close_price",
        "stk_close_price",
        "stock_volatility",
    )
    latest_values = _numeric_tail(frame, fields, _TRAINING_DAYS + 1)
    if latest_values is not None:
        latest_design = _stock_mark_design(latest_values)
        latest = _fit_ridge(latest_design[:-1], latest_design[-1])
        if latest is not None:
            _, residual, long_beta = latest
            out["scmr_mark_stock_oos_residual40"] = float(residual)
            short = _fit_ridge(
                latest_design[-1 - _SHORT_TRAINING_DAYS : -1],
                latest_design[-1],
            )
            if short is not None:
                out["scmr_stock_beta_shift20_40"] = float(short[2][0] - long_beta[0])

    z_values = _numeric_tail(frame, fields, _TRAINING_DAYS + _RESIDUAL_WINDOW + 1)
    if z_values is not None:
        residuals = _oos_residual_sequence(_stock_mark_design(z_values))
        if len(residuals) == _RESIDUAL_WINDOW + 1 and np.isfinite(residuals).all():
            out["scmr_residual_z20"] = _z_last_against_prior(residuals)
    return out


def _topology_design(values: np.ndarray) -> np.ndarray:
    """Return nonlinear payoff targets followed by linear-state predictors."""

    cb_close, floor, put, call, stock, year_to_mat, stock_volatility = values.T
    floor_log = _safe_log_ratio(cb_close, floor)
    location = np.full(len(values), np.nan, dtype="float64")
    width = call - put
    valid_location = (
        np.isfinite(put)
        & np.isfinite(call)
        & np.isfinite(stock)
        & (put > _EPS)
        & (call > _EPS)
        & (stock > _EPS)
        & (width > _EPS)
    )
    location[valid_location] = (stock[valid_location] - put[valid_location]) / width[valid_location]
    call_proximity = 1.0 / (1.0 + np.abs(location - 1.0))
    put_proximity = 1.0 / (1.0 + np.abs(location))
    call_target = floor_log * call_proximity
    put_target = floor_log * put_proximity
    side_target = floor_log * (call_proximity - put_proximity)
    return np.column_stack(
        (
            call_target,
            put_target,
            side_target,
            floor_log,
            location,
            year_to_mat,
            stock_volatility,
        )
    )


def _topology_residual_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _CONTRACT_TOPOLOGY_SIGNALS}
    fields = (
        "cb_close_price",
        "pure_redemption_value",
        "cb_put_price",
        "cb_call_price",
        "stock_close_price",
        "year_to_mat",
        "stock_volatility",
    )
    latest_values = _numeric_tail(frame, fields, _TRAINING_DAYS + 1)
    if latest_values is not None:
        latest_design = _topology_design(latest_values)
        call_design = np.column_stack((latest_design[:, 0], latest_design[:, 3:]))
        put_design = np.column_stack((latest_design[:, 1], latest_design[:, 3:]))
        call_residual = _one_oos_residual(call_design[:-1], call_design[-1])
        put_residual = _one_oos_residual(put_design[:-1], put_design[-1])
        if np.isfinite(call_residual):
            out["cft_call_floor_oos_residual40"] = float(call_residual)
        if np.isfinite(put_residual):
            out["cft_put_floor_oos_residual40"] = float(put_residual)

    z_values = _numeric_tail(frame, fields, _TRAINING_DAYS + _RESIDUAL_WINDOW + 1)
    if z_values is not None:
        z_design = _topology_design(z_values)
        side_design = np.column_stack((z_design[:, 2], z_design[:, 3:]))
        residuals = _oos_residual_sequence(side_design)
        if len(residuals) == _RESIDUAL_WINDOW + 1 and np.isfinite(residuals).all():
            out["cft_side_topology_residual_z20"] = _z_last_against_prior(residuals)
    return out


_FAMILY_METRICS = {
    "interday_finalization_transmission": _interday_metrics,
    "stock_conditioned_mark_residual": _stock_residual_metrics,
    "contract_floor_payoff_topology": _topology_residual_metrics,
}


def _build_family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    out_index = _output_index(ctx)
    signals = _FAMILY_SIGNALS[family]
    out = pd.DataFrame(index=out_index, columns=signals, dtype="float64")
    if out.empty:
        return out

    score_date = _score_date_from_index(out_index)
    price_fields = _FAMILY_PRICE_FIELDS[family]
    price_all = _strict_history_source(
        ctx,
        source="market_cbond.daily_price",
        fields=price_fields,
        score_date=score_date,
    )
    if price_all.empty:
        return out
    price_calendar = sorted(pd.Timestamp(day).normalize() for day in price_all["trade_date"].unique())
    anchor = price_calendar[-1]
    price_all["close_price"] = pd.to_numeric(price_all["close_price"], errors="coerce")
    price = price_all.loc[
        np.isfinite(price_all["close_price"]) & (price_all["close_price"] > _EPS)
    ].copy()
    if price.empty:
        return out
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    if not anchor_codes:
        return out
    price_session_index = {day: position for position, day in enumerate(price_calendar)}
    history = price.loc[:, ["trade_date", "code", *price_fields]].copy()

    twap_fields = _FAMILY_TWAP_FIELDS[family]
    if twap_fields:
        twap = _strict_history_source(
            ctx,
            source="market_cbond.daily_twap",
            fields=twap_fields,
            score_date=score_date,
        )
        history = history.merge(
            twap.loc[:, ["trade_date", "code", *twap_fields]],
            on=["trade_date", "code"],
            how="inner",
            validate="one_to_one",
        )

    base_fields = _FAMILY_BASE_FIELDS[family]
    if base_fields:
        base = _strict_history_source(
            ctx,
            source="market_cbond.daily_base",
            fields=base_fields,
            score_date=score_date,
        )
        history = history.merge(
            base.loc[:, ["trade_date", "code", *base_fields]],
            on=["trade_date", "code"],
            how="inner",
            validate="one_to_one",
        )
    if history.empty:
        return out
    history = history.sort_values(["code", "trade_date"], kind="mergesort")
    history["__price_session_index"] = history["trade_date"].map(price_session_index)
    groups = {
        str(code): group.reset_index(drop=True)
        for code, group in history.groupby("code", sort=False)
    }
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
class FactorMiningDailyInterdayTopologyV1(Factor):
    """Research-only strict-T-1 interday and contract-payoff kernel."""

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
    "FactorMiningDailyInterdayTopologyV1",
    "daily_interday_topology_catalog",
    "factor_mining_catalog",
]
