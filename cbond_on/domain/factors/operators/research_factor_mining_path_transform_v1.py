"""Research-only strict-PIT price-path transform factor catalogue.

The catalogue is intentionally not imported by :mod:`defs.__init__`.  It is a
research candidate pool for the IC-mining workflow, rather than a production
factor pack.  It consumes only the current ``FactorComputeContext.panel`` and
only the ``last`` / ``pre_close`` price fields for signal construction.

``clean_direct`` can carry previous physical snapshots under the current
partition label.  Therefore every calculation independently verifies that a
row belongs to the physical score date and that it was observed no later than
14:30.  Missing or malformed price paths stay missing: this module never
substitutes zero for unavailable data.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_path_transform_v1"
PATH_TRANSFORM_VERSION = "20260803_path_transform_v1"
_CUTOFF = dt_time(14, 30)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)
_EPS = 1e-12


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research candidate in the path-transform catalogue."""

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
        CatalogEntry(
            family=family,
            signal=signal,
            kernel=KERNEL_NAME,
            hypothesis=hypothesis,
        )
        for signal in signals
    )


_PHASE_TRANSITION_SIGNALS = (
    "path_phase_slope_acceleration",
    "path_phase_slope_reversal_intensity",
    "path_phase_drawdown_repair",
    "path_phase_early_highwater_reclaim",
    "path_phase_location_shift",
    "path_phase_fit_quality_shift",
    "path_phase_median_crossing_shift",
    "path_phase_anchor_curvature",
)
_EXTREMUM_SEQUENCE_SIGNALS = (
    "path_extreme_high_low_order",
    "path_extreme_order_separation",
    "path_extreme_terminal_since_latest",
    "path_extreme_near_high_revisit_share",
    "path_extreme_near_low_revisit_share",
    "path_extreme_latest_low_recovery",
    "path_extreme_latest_high_fade",
    "path_extreme_terminal_last_distance",
)
_ALPHA_PATH_TRANSFORM_SIGNALS = (
    "path_alpha_signed_power_vol",
    "path_alpha_tail_rank_spread",
    "path_alpha_delayed_momentum_product",
    "path_alpha_terminal_delta_regime",
    "path_alpha_momentum_sign_persistence",
    "path_alpha_terminal_volatility_breakout",
    "path_alpha_ewm_momentum",
    "path_alpha_power_rank_trend",
)


_CATALOG = (
    _entries(
        "intraday_phase_transition",
        _PHASE_TRANSITION_SIGNALS,
        "The state transition between early, middle, and late price paths can differ even when the full-session return is similar.",
    )
    + _entries(
        "extremum_sequence",
        _EXTREMUM_SEQUENCE_SIGNALS,
        "The ordering, revisiting, and terminal resolution of intraday extrema encode path structure beyond a path endpoint or drawdown level.",
    )
    + _entries(
        "alpha_path_transform_port",
        _ALPHA_PATH_TRANSFORM_SIGNALS,
        "Nonlinear price-path transforms adapt classic alpha ideas using only a strict-PIT convertible-bond last-price path.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "intraday_phase_transition": _PHASE_TRANSITION_SIGNALS,
    "extremum_sequence": _EXTREMUM_SEQUENCE_SIGNALS,
    "alpha_path_transform_port": _ALPHA_PATH_TRANSFORM_SIGNALS,
}
_FAMILY_REQUIRED_COLUMNS = {
    "intraday_phase_transition": ("last", "pre_close"),
    "extremum_sequence": ("last",),
    "alpha_path_transform_port": ("last", "pre_close"),
}


def factor_mining_path_transform_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable research-only catalogue in family-first order."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility alias for generic research catalogue loaders."""

    return factor_mining_path_transform_catalog()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _signal_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    """Resolve one score date without treating rolling index labels as proof."""

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


def _score_day_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Keep only physical score-day observations through the 14:30 cutoff."""

    cache_key = f"{KERNEL_NAME}:{PATH_TRANSFORM_VERSION}:score_day_frame"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    panel = ensure_trade_time(ctx.panel)
    frame = panel.reset_index().copy(deep=False)
    target = _signal_date_from_panel(panel)
    if target is None:
        out = frame.iloc[0:0].copy()
    else:
        labels = pd.to_datetime(frame["dt"], errors="coerce")
        if not frame.empty and not bool((labels.dt.normalize() == target).any()):
            raise ValueError(
                f"{KERNEL_NAME} has no indexed panel rows for signal day {target.date().isoformat()}"
            )
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        keep = (
            labels.notna()
            & (labels.dt.normalize() == target)
            & timestamps.notna()
            & (timestamps.dt.normalize() == target)
            & (timestamps.dt.time <= _CUTOFF)
        )
        out = frame.loc[keep].copy()
        out["trade_time"] = timestamps.loc[keep]
        if not out.empty:
            out = out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = out
    return out


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _safe_div(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if abs(denominator) <= _EPS:
        return float("nan")
    return float(numerator / denominator)


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _numeric(g: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(g[column], errors="coerce").to_numpy(dtype="float64")


def _prices(g: pd.DataFrame, *, min_count: int = 2) -> np.ndarray | None:
    values = _numeric(g, "last")
    if values.size < min_count:
        return None
    if not (np.isfinite(values).all() and (values > 0.0).all()):
        return None
    return values.astype("float64", copy=False)


def _pre_close(g: pd.DataFrame) -> float:
    values = _numeric(g, "pre_close")
    if values.size == 0 or not np.isfinite(values[0]) or values[0] <= 0.0:
        return float("nan")
    return float(values[0])


def _time_positions(g: pd.DataFrame) -> np.ndarray | None:
    timestamps = pd.to_datetime(g["trade_time"], errors="coerce")
    if len(timestamps) < 2 or timestamps.isna().any():
        return None
    elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype="float64")
    total = float(elapsed[-1])
    if not np.isfinite(elapsed).all() or total <= _EPS:
        return None
    return elapsed / total


def _phase_masks(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    timestamps = pd.to_datetime(g["trade_time"], errors="coerce")
    if timestamps.isna().any():
        return None
    clocks = timestamps.dt.time.to_numpy()
    early = clocks <= _EARLY_END
    late = clocks >= _LATE_START
    middle = ~(early | late)
    return early, middle, late


def _linear_slope_r2(values: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
    if (
        len(values) < 3
        or len(values) != len(positions)
        or not (np.isfinite(values).all() and np.isfinite(positions).all())
    ):
        return float("nan"), float("nan")
    x = positions - float(np.mean(positions))
    y = values - float(np.mean(values))
    xx = float(np.dot(x, x))
    yy = float(np.dot(y, y))
    if xx <= _EPS or yy <= _EPS:
        return float("nan"), float("nan")
    slope = float(np.dot(x, y) / xx)
    fitted = slope * x
    residual = y - fitted
    r2 = 1.0 - _safe_div(float(np.dot(residual, residual)), yy)
    return slope, r2 if np.isfinite(r2) else float("nan")


def _terminal_location(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return float("nan")
    low = float(np.min(values))
    high = float(np.max(values))
    return _safe_div(float(values[-1] - low), high - low)


def _rank_percentile(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    ranked = pd.Series(values, dtype="float64").rank(pct=True, method="average")
    value = float(ranked.iloc[-1])
    return value if np.isfinite(value) else float("nan")


def _robust_scale(values: np.ndarray) -> float:
    if len(values) < 3 or not np.isfinite(values).all():
        return float("nan")
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad > _EPS:
        return float(1.4826 * mad)
    mean_abs = float(np.mean(np.abs(values)))
    return mean_abs if mean_abs > _EPS else float("nan")


def _phase_transition_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_PHASE_TRANSITION_SIGNALS)
    prices = _prices(g, min_count=9)
    positions = _time_positions(g)
    masks = _phase_masks(g)
    if prices is None or positions is None or masks is None:
        return out
    early_mask, middle_mask, late_mask = masks
    if min(int(early_mask.sum()), int(middle_mask.sum()), int(late_mask.sum())) < 3:
        return out

    log_prices = np.log(prices)
    early = prices[early_mask]
    middle = prices[middle_mask]
    late = prices[late_mask]
    early_slope, early_r2 = _linear_slope_r2(log_prices[early_mask], positions[early_mask])
    late_slope, late_r2 = _linear_slope_r2(log_prices[late_mask], positions[late_mask])
    out["path_phase_slope_acceleration"] = late_slope - early_slope
    out["path_phase_slope_reversal_intensity"] = _safe_div(
        -early_slope * late_slope,
        early_slope * early_slope + late_slope * late_slope,
    )

    early_peak = float(np.max(early))
    early_trough = float(np.min(early))
    early_drawdown = _safe_div(early_peak - early_trough, early_peak)
    late_recovery = _safe_div(float(late[-1] - early_trough), early_trough)
    out["path_phase_drawdown_repair"] = _safe_div(late_recovery, early_drawdown)
    out["path_phase_early_highwater_reclaim"] = _safe_div(float(late[-1] - early_peak), early_peak)
    out["path_phase_location_shift"] = _terminal_location(late) - _terminal_location(early)
    out["path_phase_fit_quality_shift"] = late_r2 - early_r2

    early_median = float(np.median(early))
    early_side = np.sign(early - early_median)
    late_side = np.sign(late - early_median)
    out["path_phase_median_crossing_shift"] = float(np.mean(late_side) - np.mean(early_side))

    anchor = _pre_close(g)
    if np.isfinite(anchor):
        early_anchor = float(np.mean(early / anchor - 1.0))
        middle_anchor = float(np.mean(middle / anchor - 1.0))
        late_anchor = float(np.mean(late / anchor - 1.0))
        out["path_phase_anchor_curvature"] = late_anchor - 2.0 * middle_anchor + early_anchor
    return out


def _extremum_sequence_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_EXTREMUM_SEQUENCE_SIGNALS)
    prices = _prices(g, min_count=4)
    positions = _time_positions(g)
    if prices is None or positions is None:
        return out
    low = float(np.min(prices))
    high = float(np.max(prices))
    path_range = high - low
    if path_range <= _EPS:
        return out

    first_high = int(np.argmax(prices))
    first_low = int(np.argmin(prices))
    last_high = int(np.flatnonzero(prices == high)[-1])
    last_low = int(np.flatnonzero(prices == low)[-1])
    latest = max(last_high, last_low)
    out["path_extreme_high_low_order"] = float(positions[first_high] - positions[first_low])
    out["path_extreme_order_separation"] = float(abs(positions[first_high] - positions[first_low]))
    out["path_extreme_terminal_since_latest"] = float(1.0 - positions[latest])

    band = 0.10 * path_range
    high_revisits = max(0, int((prices >= high - band).sum()) - 1)
    low_revisits = max(0, int((prices <= low + band).sum()) - 1)
    out["path_extreme_near_high_revisit_share"] = _safe_div(float(high_revisits), float(len(prices) - 1))
    out["path_extreme_near_low_revisit_share"] = _safe_div(float(low_revisits), float(len(prices) - 1))
    out["path_extreme_latest_low_recovery"] = _safe_div(float(prices[-1] - prices[last_low]), path_range)
    out["path_extreme_latest_high_fade"] = _safe_div(float(prices[-1] - prices[last_high]), path_range)
    out["path_extreme_terminal_last_distance"] = _safe_div(float(prices[-1] - prices[latest]), path_range)
    return out


def _alpha_path_transform_metrics(g: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_ALPHA_PATH_TRANSFORM_SIGNALS)
    prices = _prices(g, min_count=12)
    positions = _time_positions(g)
    if prices is None or positions is None:
        return out
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < 10 or not np.isfinite(log_returns).all():
        return out
    return_scale = _robust_scale(log_returns)

    anchor = _pre_close(g)
    if np.isfinite(anchor) and np.isfinite(return_scale):
        anchor_return = float(np.log(prices[-1] / anchor))
        signed_power = float(np.sign(anchor_return) * abs(anchor_return) ** 1.5)
        out["path_alpha_signed_power_vol"] = _safe_div(signed_power, return_scale)

    full_rank = _rank_percentile(prices)
    tail_rank = _rank_percentile(prices[-7:])
    out["path_alpha_tail_rank_spread"] = tail_rank - full_rank

    delayed_product = float(np.sum(log_returns[-3:]) * np.sum(log_returns[-9:-6]))
    if np.isfinite(return_scale):
        transformed_product = float(np.sign(delayed_product) * np.sqrt(abs(delayed_product)))
        out["path_alpha_delayed_momentum_product"] = _safe_div(transformed_product, return_scale)

    price_deltas = np.diff(prices)
    delta_scale = _robust_scale(price_deltas)
    recent_regime = float(np.sign(np.sum(np.sign(price_deltas[-5:-1]))))
    if np.isfinite(delta_scale) and recent_regime != 0.0:
        out["path_alpha_terminal_delta_regime"] = _safe_div(
            recent_regime * float(price_deltas[-1]),
            delta_scale,
        )

    weights = np.exp(np.linspace(-2.0, 0.0, num=len(log_returns), dtype="float64"))
    weight_sum = float(weights.sum())
    if weight_sum > _EPS:
        out["path_alpha_momentum_sign_persistence"] = float(
            np.dot(weights, np.sign(log_returns)) / weight_sum
        )
        if np.isfinite(return_scale):
            out["path_alpha_ewm_momentum"] = _safe_div(
                float(np.dot(weights, log_returns) / weight_sum),
                return_scale,
            )

    prior_abs = np.abs(log_returns[-6:-1])
    prior_median = float(np.median(prior_abs))
    latest_return = float(log_returns[-1])
    if prior_median > _EPS and np.isfinite(latest_return):
        ratio = abs(latest_return) / prior_median
        out["path_alpha_terminal_volatility_breakout"] = float(np.sign(latest_return) * np.log1p(ratio))

    ranks = pd.Series(prices, dtype="float64").rank(pct=True, method="average").to_numpy()
    powered_ranks = np.sign(2.0 * ranks - 1.0) * np.abs(2.0 * ranks - 1.0) ** 3.0
    centered_time = positions - float(np.mean(positions))
    denominator = float(np.sqrt(np.dot(powered_ranks, powered_ranks) * np.dot(centered_time, centered_time)))
    if denominator > _EPS:
        out["path_alpha_power_rank_trend"] = float(np.dot(powered_ranks, centered_time) / denominator)
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "intraday_phase_transition": _phase_transition_metrics,
    "extremum_sequence": _extremum_sequence_metrics,
    "alpha_path_transform_port": _alpha_path_transform_metrics,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    """Compute and cache one whole family for all eight of its signal specs."""

    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{PATH_TRANSFORM_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    frame = _score_day_frame(ctx)
    signals = _FAMILY_SIGNALS[family]
    _require_columns(frame, ("trade_time", *_FAMILY_REQUIRED_COLUMNS[family]))
    if frame.empty:
        built = pd.DataFrame(index=_empty_index(), columns=signals, dtype="float64")
    else:
        rows: list[dict[str, object]] = []
        calculator = _FAMILY_CALCULATORS[family]
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            values = calculator(group)
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            row.update({signal: float(values.get(signal, np.nan)) for signal in signals})
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(signals)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires an explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningPathTransformV1(Factor):
    """Research-only price-path transformation kernel for the T1430 panel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
