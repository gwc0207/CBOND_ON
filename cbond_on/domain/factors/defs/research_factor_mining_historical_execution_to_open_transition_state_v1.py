"""Research-only historical execution-to-open transition-state candidates.

This catalogue uses completed daily-TWAP triplets only.  For an historical
start date ``D`` the triplet is
``twap_1430_1442[D] -> twap_1442_1457[D] -> twap_0930_1000[D+1]``.  At score
date ``T`` a triplet is eligible only when its end session satisfies
``D + 1 < T``.  Thus neither a score-day observation nor a not-yet-completed
opening transition can enter a value.

``market_cbond.daily_price`` is deliberately used only as the independent
trading-calendar anchor.  It contributes no price, return, liquidity, or
contract value.  All numerical state inputs come from declared
``market_cbond.daily_twap`` columns.  The output T1430 panel supplies keys
only.

The three families use discrete transition tables, distributional regime
change, and magnitude-state coupling.  They are intentionally not aliases for
the existing daily TWAP slope, curvature, session-memory, or intraday phase
formulas.  The module opens no files, database, network, labels, models,
scores, PnL, live artefacts, or production FactorStore.
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


KERNEL_NAME = "factor_mining_historical_execution_to_open_transition_state_v1"
CATALOG_VERSION = "20260803_historical_execution_to_open_transition_state_v1"
_PAIR_WINDOW = 60
_RECENT_WINDOW = 20
_BASE_WINDOW = _PAIR_WINDOW - _RECENT_WINDOW
_LOOKBACK_DAYS = _PAIR_WINDOW + 4
_PSEUDOCOUNT = 0.5
_EPS = 1e-12
_TWAP_FIELDS = ("twap_1430_1442", "twap_1442_1457", "twap_0930_1000")
_STATE_VALUES = np.asarray((-1.0, 0.0, 1.0), dtype="float64")

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
    """One explicit research-only factor candidate."""

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


_TERMINAL_STATE_SIGNALS = (
    "heot_terminal_joint_surprisal60",
    "heot_terminal_conditional_information60",
    "heot_terminal_open_state_residual60",
)
_REGIME_STATE_SIGNALS = (
    "heot_joint_state_js_divergence20_60",
    "heot_joint_state_concentration_shift20_60",
    "heot_conditional_open_distribution_drift20_60",
)
_MAGNITUDE_STATE_SIGNALS = (
    "heot_extreme_magnitude_lift60",
    "heot_quiet_tail_open_escape60",
    "heot_magnitude_state_information60",
)

_CATALOG = (
    _entries(
        "historical_execution_open_terminal_transition",
        _TERMINAL_STATE_SIGNALS,
        "The newest fully completed execution-to-next-morning state can be unusual under its own prior-only ternary transition table, without using the score day.",
    )
    + _entries(
        "historical_execution_open_transition_regime",
        _REGIME_STATE_SIGNALS,
        "The recent mix of completed close-to-next-morning states can change relative to its earlier history, independently of a raw return slope or session average.",
    )
    + _entries(
        "historical_execution_open_magnitude_coupling",
        _MAGNITUDE_STATE_SIGNALS,
        "Completed execution-tail magnitude can map nonlinearly into next-morning magnitude states, separately from signed direction and linear response measures.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS: dict[str, tuple[str, ...]] = {
    "historical_execution_open_terminal_transition": _TERMINAL_STATE_SIGNALS,
    "historical_execution_open_transition_regime": _REGIME_STATE_SIGNALS,
    "historical_execution_open_magnitude_coupling": _MAGNITUDE_STATE_SIGNALS,
}

FORMULAS: dict[str, str] = {
    "heot_terminal_joint_surprisal60": (
        "-log P(S_exec,S_open) for the terminal completed D->D+1 ternary state, "
        "where the probability table is estimated from the preceding 59 eligible pairs only "
        "with Jeffreys pseudocount 0.5."
    ),
    "heot_terminal_conditional_information60": (
        "log[P(S_open|S_exec)/P(S_open)] for the terminal completed D->D+1 "
        "ternary state using the preceding 59 eligible pairs only."
    ),
    "heot_terminal_open_state_residual60": (
        "S_open - E[S_open|S_exec] for the terminal completed D->D+1 ternary "
        "state under the preceding 59 eligible-pair transition table."
    ),
    "heot_joint_state_js_divergence20_60": (
        "Jensen-Shannon divergence between 3x3 execution-to-open ternary state "
        "distributions of the latest 20 and preceding 40 eligible pairs."
    ),
    "heot_joint_state_concentration_shift20_60": (
        "Latest-20 minus preceding-40 Herfindahl concentration of the 3x3 "
        "completed execution-to-open ternary state distribution."
    ),
    "heot_conditional_open_distribution_drift20_60": (
        "Tail-state-marginal-weighted total-variation drift between latest-20 and "
        "preceding-40 conditional next-morning ternary distributions."
    ),
    "heot_extreme_magnitude_lift60": (
        "log[P(M_open=high|M_exec=high)/P(M_open=high)] from a 3x3 magnitude "
        "state table over the latest 60 eligible pairs."
    ),
    "heot_quiet_tail_open_escape60": (
        "P(M_open is low or high|M_exec=middle) - P(M_open is low or high) from "
        "the latest 60 eligible magnitude-state pairs."
    ),
    "heot_magnitude_state_information60": (
        "Mutual information, normalized by log(3), of the execution-tail and "
        "next-morning 3-state magnitude table over the latest 60 eligible pairs."
    ),
}


def historical_execution_to_open_transition_state_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the isolated factor-mining runner."""

    return historical_execution_to_open_transition_state_catalog()


def _requested_entry(params: dict[str, object] | None) -> CatalogEntry:
    signal = str((params or {}).get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _daily_requirements() -> list[DailyFactorRequirement]:
    return [
        DailyFactorRequirement(
            "market_cbond.daily_price",
            ("exchange_code",),
            _LOOKBACK_DAYS,
        ),
        DailyFactorRequirement(
            "market_cbond.daily_twap",
            ("exchange_code", *_TWAP_FIELDS),
            _LOOKBACK_DAYS,
        ),
    ]


def _canonical_market_code(values: pd.Series, exchanges: pd.Series) -> pd.Series:
    """Normalize only codes with an explicit or already-valid market suffix."""

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
            for value, exchange in zip(values, exchanges, strict=False)
        ],
        index=values.index,
        dtype="string",
    )


def _panel_code(value: object) -> str:
    return str(_canonical_market_code(pd.Series([value]), pd.Series([""])).iloc[0])


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, source: str
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {source} missing columns: {missing}")


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    indexed = ensure_panel_index(panel)
    raw_day = indexed.attrs.get("__build_day__")
    if raw_day is not None:
        score_date = pd.Timestamp(raw_day)
        if pd.isna(score_date):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return score_date.normalize()
    if indexed.empty:
        return None
    days = pd.to_datetime(
        indexed.index.get_level_values("dt"), errors="coerce"
    ).normalize()
    unique = pd.Index(days[days.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(
            f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel"
        )
    return pd.Timestamp(unique[0]).normalize()


def _output_index(ctx: FactorComputeContext) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].drop_duplicates()
    keys = keys.sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=("dt", "code"))


def _strict_price_calendar(
    ctx: FactorComputeContext, score_date: pd.Timestamp
) -> pd.DatetimeIndex:
    """Return a strict-prior market calendar without consuming price values."""

    source = "market_cbond.daily_price"
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date",), source=source)
    dates = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    dates = dates.loc[dates.notna() & (dates < score_date)]
    if dates.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    return pd.DatetimeIndex(
        sorted(pd.Timestamp(day).normalize() for day in dates.unique())
    )


def _strict_twap_history(
    ctx: FactorComputeContext, score_date: pd.Timestamp
) -> pd.DataFrame:
    source = "market_cbond.daily_twap"
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(
        raw, ("trade_date", "code", "exchange_code", *_TWAP_FIELDS), source=source
    )
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *_TWAP_FIELDS]].copy()
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
            f"{KERNEL_NAME} {source} has duplicate strict-prior rows: "
            f"{examples.to_dict('records')}"
        )
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(
        drop=True
    )


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (
        np.isfinite(numerator)
        and np.isfinite(denominator)
        and numerator > _EPS
        and denominator > _EPS
    ):
        return float("nan")
    return float(np.log(numerator / denominator))


def _completed_transition_pairs(
    history: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    score_date: pd.Timestamp,
) -> np.ndarray | None:
    """Return the latest contiguous eligible ``[tail_return, open_return]`` pairs."""

    if len(calendar) < _PAIR_WINDOW + 1 or history.empty:
        return None
    by_day = history.set_index("trade_date")
    if not by_day.index.is_unique:
        raise ValueError(f"{KERNEL_NAME} code history has duplicate calendar rows")
    rows: list[tuple[int, float, float]] = []
    for start_index in range(len(calendar) - 1):
        start_day = pd.Timestamp(calendar[start_index]).normalize()
        end_day = pd.Timestamp(calendar[start_index + 1]).normalize()
        if (
            end_day >= score_date
            or start_day not in by_day.index
            or end_day not in by_day.index
        ):
            continue
        start = by_day.loc[start_day]
        end = by_day.loc[end_day]
        pre_execution = float(pd.to_numeric(start["twap_1430_1442"], errors="coerce"))
        execution = float(pd.to_numeric(start["twap_1442_1457"], errors="coerce"))
        morning = float(pd.to_numeric(end["twap_0930_1000"], errors="coerce"))
        rows.append(
            (
                start_index,
                _safe_log_ratio(execution, pre_execution),
                _safe_log_ratio(morning, execution),
            )
        )
    if len(rows) < _PAIR_WINDOW:
        return None
    latest = rows[-_PAIR_WINDOW:]
    positions = np.asarray([row[0] for row in latest], dtype="int64")
    expected = np.arange(
        len(calendar) - _PAIR_WINDOW - 1, len(calendar) - 1, dtype="int64"
    )
    if not np.array_equal(positions, expected):
        return None
    values = np.asarray([[row[1], row[2]] for row in latest], dtype="float64")
    return values if np.isfinite(values).all() else None


def _tercile_edges(values: np.ndarray) -> tuple[float, float] | None:
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        return None
    low, high = np.quantile(values, (1.0 / 3.0, 2.0 / 3.0))
    if not (np.isfinite(low) and np.isfinite(high) and high - low > _EPS):
        return None
    return float(low), float(high)


def _ternary_states(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    low, high = edges
    state = np.zeros(len(values), dtype="int64")
    state[values < low] = -1
    state[values > high] = 1
    return state


def _joint_counts(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if len(left) != len(right) or len(left) == 0:
        raise ValueError(f"{KERNEL_NAME} requires nonempty aligned state arrays")
    if not (np.isin(left, (-1, 0, 1)).all() and np.isin(right, (-1, 0, 1)).all()):
        raise ValueError(f"{KERNEL_NAME} encountered an invalid ternary state")
    counts = np.zeros((3, 3), dtype="float64")
    np.add.at(counts, (left + 1, right + 1), 1.0)
    return counts


def _smoothed_joint(counts: np.ndarray) -> np.ndarray:
    smoothed = counts.astype("float64", copy=False) + _PSEUDOCOUNT
    total = float(smoothed.sum())
    if not np.isfinite(total) or total <= _EPS:
        raise ValueError(f"{KERNEL_NAME} has an invalid transition table")
    return smoothed / total


def _conditional_distribution(joint: np.ndarray) -> np.ndarray:
    row_sum = joint.sum(axis=1, keepdims=True)
    if not np.isfinite(row_sum).all() or (row_sum <= _EPS).any():
        raise ValueError(f"{KERNEL_NAME} has an invalid conditional transition table")
    return joint / row_sum


def _terminal_state_metrics(values: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _TERMINAL_STATE_SIGNALS}
    training = values[:-1]
    tail_edges = _tercile_edges(training[:, 0])
    open_edges = _tercile_edges(training[:, 1])
    if tail_edges is None or open_edges is None:
        return out
    tail_train = _ternary_states(training[:, 0], tail_edges)
    open_train = _ternary_states(training[:, 1], open_edges)
    terminal_tail = int(_ternary_states(values[-1:, 0], tail_edges)[0])
    terminal_open = int(_ternary_states(values[-1:, 1], open_edges)[0])
    joint = _smoothed_joint(_joint_counts(tail_train, open_train))
    conditional = _conditional_distribution(joint)
    open_marginal = joint.sum(axis=0)
    row = terminal_tail + 1
    column = terminal_open + 1
    expected_open = float(np.dot(_STATE_VALUES, conditional[row]))
    joint_probability = float(joint[row, column])
    conditional_probability = float(conditional[row, column])
    marginal_probability = float(open_marginal[column])
    if not (
        np.isfinite(expected_open)
        and joint_probability > _EPS
        and conditional_probability > _EPS
        and marginal_probability > _EPS
    ):
        return out
    out.update(
        {
            "heot_terminal_joint_surprisal60": float(-np.log(joint_probability)),
            "heot_terminal_conditional_information60": float(
                np.log(conditional_probability / marginal_probability)
            ),
            "heot_terminal_open_state_residual60": float(terminal_open - expected_open),
        }
    )
    return out


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)
    if not (
        np.isfinite(midpoint).all() and (left > _EPS).all() and (right > _EPS).all()
    ):
        return float("nan")
    return float(
        0.5 * np.sum(left * np.log(left / midpoint))
        + 0.5 * np.sum(right * np.log(right / midpoint))
    )


def _transition_regime_metrics(values: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _REGIME_STATE_SIGNALS}
    tail_edges = _tercile_edges(values[:, 0])
    open_edges = _tercile_edges(values[:, 1])
    if tail_edges is None or open_edges is None:
        return out
    tail_state = _ternary_states(values[:, 0], tail_edges)
    open_state = _ternary_states(values[:, 1], open_edges)
    base_counts = _joint_counts(tail_state[:_BASE_WINDOW], open_state[:_BASE_WINDOW])
    recent_counts = _joint_counts(tail_state[_BASE_WINDOW:], open_state[_BASE_WINDOW:])
    base_joint = _smoothed_joint(base_counts)
    recent_joint = _smoothed_joint(recent_counts)
    base_conditional = _conditional_distribution(base_joint)
    recent_conditional = _conditional_distribution(recent_joint)
    tail_weight = 0.5 * (base_joint.sum(axis=1) + recent_joint.sum(axis=1))
    conditional_drift = float(
        np.sum(
            tail_weight
            * 0.5
            * np.sum(np.abs(recent_conditional - base_conditional), axis=1)
        )
    )
    values_out = {
        "heot_joint_state_js_divergence20_60": _js_divergence(
            base_joint.ravel(), recent_joint.ravel()
        ),
        "heot_joint_state_concentration_shift20_60": float(
            np.sum(recent_joint * recent_joint) - np.sum(base_joint * base_joint)
        ),
        "heot_conditional_open_distribution_drift20_60": conditional_drift,
    }
    out.update({key: value for key, value in values_out.items() if np.isfinite(value)})
    return out


def _magnitude_state_metrics(values: np.ndarray) -> dict[str, float]:
    out = {signal: float("nan") for signal in _MAGNITUDE_STATE_SIGNALS}
    tail_edges = _tercile_edges(np.abs(values[:, 0]))
    open_edges = _tercile_edges(np.abs(values[:, 1]))
    if tail_edges is None or open_edges is None:
        return out
    tail_state = _ternary_states(np.abs(values[:, 0]), tail_edges)
    open_state = _ternary_states(np.abs(values[:, 1]), open_edges)
    joint = _smoothed_joint(_joint_counts(tail_state, open_state))
    conditional = _conditional_distribution(joint)
    tail_marginal = joint.sum(axis=1)
    open_marginal = joint.sum(axis=0)
    high = 2
    middle = 1
    extreme_open_probability = float(open_marginal[0] + open_marginal[2])
    high_conditional = float(conditional[high, high])
    high_marginal = float(open_marginal[high])
    denominator = tail_marginal[:, None] * open_marginal[None, :]
    if not (
        high_conditional > _EPS
        and high_marginal > _EPS
        and np.isfinite(denominator).all()
        and (denominator > _EPS).all()
    ):
        return out
    information = float(np.sum(joint * np.log(joint / denominator)) / np.log(3.0))
    values_out = {
        "heot_extreme_magnitude_lift60": float(
            np.log(high_conditional / high_marginal)
        ),
        "heot_quiet_tail_open_escape60": float(
            conditional[middle, 0] + conditional[middle, 2] - extreme_open_probability
        ),
        "heot_magnitude_state_information60": information,
    }
    out.update({key: value for key, value in values_out.items() if np.isfinite(value)})
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[np.ndarray], dict[str, float]]] = {
    "historical_execution_open_terminal_transition": _terminal_state_metrics,
    "historical_execution_open_transition_regime": _transition_regime_metrics,
    "historical_execution_open_magnitude_coupling": _magnitude_state_metrics,
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
    calendar = _strict_price_calendar(ctx, score_date)
    twap = _strict_twap_history(ctx, score_date)
    if len(calendar) < _PAIR_WINDOW + 1 or twap.empty:
        return out
    groups = {str(code): frame for code, frame in twap.groupby("code", sort=False)}
    calculator = _FAMILY_CALCULATORS[family]
    for dt, raw_code in out_index:
        code = _panel_code(raw_code)
        group = groups.get(code)
        if not code or group is None:
            continue
        pairs = _completed_transition_pairs(group, calendar, score_date)
        if pairs is None:
            continue
        metrics = calculator(pairs)
        for signal in signals:
            value = metrics.get(signal)
            if value is not None and np.isfinite(value):
                out.at[(dt, raw_code), signal] = float(value)
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
class FactorMiningHistoricalExecutionToOpenTransitionStateV1(Factor):
    """Research-only completed historical execution-to-open state kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    @classmethod
    def daily_requirements(
        cls, params: dict | None = None
    ) -> list[DailyFactorRequirement]:
        del params
        return _daily_requirements()

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx.params)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "CATALOG_VERSION",
    "FORMULAS",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningHistoricalExecutionToOpenTransitionStateV1",
    "factor_mining_catalog",
    "historical_execution_to_open_transition_state_catalog",
]
