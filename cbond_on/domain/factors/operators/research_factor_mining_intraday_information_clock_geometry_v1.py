"""Research-only strict-PIT intraday information-clock geometry factors.

This catalogue treats the completed T-day quote/trade path as a geometric
object on an *information clock*.  Its four families measure trajectory
curvature, channel-simplex shape, price--quote mass transport, and the joint
book/discovery manifold.  They deliberately do not repeat the existing
return-slope, static-book, lead-lag, profile-cosine, or direct depth-centroid
relocation factors.

Every value comes only from physical score-day cbond observations in the
continuous sessions through exactly 14:29:00.  Price, quote, and cumulative
counter defects fail closed to NaN.  In particular, a ``num_trades`` reset
invalidates only the two families whose information-clock mass needs that
counter; it never causes a fabricated zero or silently changes the formulas
of the quote/book-only families.  No file, database, label, pool, PnL, stock
panel, or future-day input is accessed here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import (
    Factor,
    FactorComputeContext,
    ensure_panel_index,
)
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_information_clock_geometry_v1"
CATALOG_VERSION = "20260803_intraday_information_clock_geometry_v1"
_EPS = 1e-12
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_BIN_MINUTES = 5
_MORNING_BIN_COUNT = 24
_AFTERNOON_BIN_COUNT = 18
_TOTAL_BIN_COUNT = _MORNING_BIN_COUNT + _AFTERNOON_BIN_COUNT
_MIN_BINS = 6
_MIN_ACTIVE_MASS_BINS = 3


@dataclass(frozen=True)
class CatalogEntry:
    """One family-first, auditable research-only signal registration."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class _L1Bins:
    bins: np.ndarray
    last: np.ndarray
    mid: np.ndarray
    trade_mass: np.ndarray | None


@dataclass(frozen=True)
class _BookBins:
    bins: np.ndarray
    last: np.ndarray
    mid: np.ndarray
    imbalance: np.ndarray
    centroid_dislocation: np.ndarray


def _entries(
    family: str, signals: Iterable[str], hypothesis: str
) -> tuple[CatalogEntry, ...]:
    return tuple(
        CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals
    )


_CURVE_TRAJECTORY_SIGNALS = (
    "icg_curve_arc_chord_excess",
    "icg_curve_turning_energy",
    "icg_curve_torsion_intensity",
)
_CHANNEL_SIMPLEX_SIGNALS = (
    "icg_simplex_covariance_volume",
    "icg_simplex_planarity_ratio",
    "icg_simplex_axis_anisotropy",
)
_MASS_TRANSPORT_SIGNALS = (
    "icg_pq_transport_distance",
    "icg_pq_transport_dispersion_gap",
    "icg_pq_transport_skewness_gap",
)
_BOOK_MANIFOLD_SIGNALS = (
    "icg_book_discovery_loop_area",
    "icg_book_discovery_pressure_turning",
    "icg_book_discovery_centroid_torsion",
)

_CATALOG = (
    _entries(
        "information_clock_curve_trajectory",
        _CURVE_TRAJECTORY_SIGNALS,
        "The three-channel five-minute path of signed last discovery, signed midpoint repricing, and completed trade mass has curvature and torsion that a scalar return or flow statistic cannot represent.",
    )
    + _entries(
        "information_clock_channel_simplex",
        _CHANNEL_SIMPLEX_SIGNALS,
        "The spectral geometry of the normalized last-change, midpoint-change, and trade-mass profiles measures whether price discovery occupies one, two, or three independent information-clock channels.",
    )
    + _entries(
        "price_quote_mass_transport",
        _MASS_TRANSPORT_SIGNALS,
        "The transport between the time distributions of last-price discovery and valid quote repricing measures temporal displacement, spread of discovery, and asymmetry without a lead-lag regression or cosine profile rewrite.",
    )
    + _entries(
        "book_discovery_manifold",
        _BOOK_MANIFOLD_SIGNALS,
        "The path geometry jointly formed by last-versus-mid discovery, five-level book pressure, and depth-weighted centroid dislocation captures loops, turns, and torsion rather than a static book state.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "information_clock_curve_trajectory": _CURVE_TRAJECTORY_SIGNALS,
    "information_clock_channel_simplex": _CHANNEL_SIMPLEX_SIGNALS,
    "price_quote_mass_transport": _MASS_TRANSPORT_SIGNALS,
    "book_discovery_manifold": _BOOK_MANIFOLD_SIGNALS,
}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

FORMULAS: dict[str, str] = {
    "icg_curve_arc_chord_excess": "sum_t ||x_t|| / ||sum_t x_t|| - 1, x_t=[scaled Δlog(last), scaled Δlog(mid), scaled completed_trade_mass] on observed five-minute bins.",
    "icg_curve_turning_energy": "mean_t ||unit(x_t)-unit(x_(t-1))||^2 for the same three-channel information-clock increments.",
    "icg_curve_torsion_intensity": "mean_t |det(unit(x_(t-2)), unit(x_(t-1)), unit(x_t))| for the same three-channel information-clock increments.",
    "icg_simplex_covariance_volume": "cuberoot(det(Cov([p_last(t), p_mid(t), p_trade(t)]))), where each p is its own normalized observed-bin mass profile.",
    "icg_simplex_planarity_ratio": "lambda_2/lambda_3 of the ascending eigenvalues of the three-channel normalized-profile covariance matrix.",
    "icg_simplex_axis_anisotropy": "(lambda_3-lambda_1)/(lambda_1+lambda_2+lambda_3) for the same covariance spectrum.",
    "icg_pq_transport_distance": "mean_t |CDF_abs_delta_last(t)-CDF_abs_delta_mid(t)| over observed information-clock bins.",
    "icg_pq_transport_dispersion_gap": "Var_u[p_abs_delta_last(u)]-Var_u[p_abs_delta_mid(u)] using observed-bin centers u.",
    "icg_pq_transport_skewness_gap": "Skew_u[p_abs_delta_last(u)]-Skew_u[p_abs_delta_mid(u)] using observed-bin centers u.",
    "icg_book_discovery_loop_area": "0.5*||sum_t cross(z_t,z_(t+1))||/(n-1), z=[last-minus-mid discovery, L1-L5 imbalance shift, centroid-dislocation shift] after local increment scaling.",
    "icg_book_discovery_pressure_turning": "mean_t ||cross(Δz_(t-1), Δz_t)|| for the scaled three-dimensional book/discovery path.",
    "icg_book_discovery_centroid_torsion": "mean_t |det(Δz_(t-2), Δz_(t-1), Δz_t)| for the scaled three-dimensional book/discovery path.",
}

_L1_FIELDS = ("last", "ask_price1", "bid_price1", "ask_volume1", "bid_volume1")
_COUNTER_FIELDS = (*_L1_FIELDS, "num_trades")
_BOOK_FIELDS = (
    "last",
    *tuple(
        field
        for level in range(1, 6)
        for field in (
            f"ask_price{level}",
            f"bid_price{level}",
            f"ask_volume{level}",
            f"bid_volume{level}",
        )
    ),
)
_REQUIRED_FIELDS = {
    "information_clock_curve_trajectory": _COUNTER_FIELDS,
    "information_clock_channel_simplex": _COUNTER_FIELDS,
    "price_quote_mass_transport": _L1_FIELDS,
    "book_discovery_manifold": _BOOK_FIELDS,
}


def factor_mining_intraday_information_clock_geometry_v1_catalog() -> tuple[
    CatalogEntry, ...
]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic scratch-expansion runner compatibility entrypoint."""

    return factor_mining_intraday_information_clock_geometry_v1_catalog()


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
    dates = pd.to_datetime(
        panel.index.get_level_values("dt"), errors="coerce"
    ).normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(
            f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel"
        )
    return pd.Timestamp(unique[0]).normalize()


def _output_index(
    ctx: FactorComputeContext, score_date: pd.Timestamp | None
) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if score_date is None or panel.empty:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    frame = panel.reset_index()
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    selected = frame.loc[
        indexed.notna() & (indexed.dt.normalize() == score_date), ["dt", "code"]
    ]
    selected = selected.drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(selected, names=["dt", "code"])


def _session_bin(clock: dt_time) -> int | None:
    """Map a valid continuous-session clock to an observed five-minute bin."""

    if _MORNING_START <= clock <= _MORNING_END:
        minutes = (clock.hour * 60 + clock.minute) - (
            _MORNING_START.hour * 60 + _MORNING_START.minute
        )
        return min(int(minutes // _BIN_MINUTES), _MORNING_BIN_COUNT - 1)
    if _AFTERNOON_START <= clock <= _CUTOFF:
        minutes = (clock.hour * 60 + clock.minute) - (
            _AFTERNOON_START.hour * 60 + _AFTERNOON_START.minute
        )
        return _MORNING_BIN_COUNT + min(
            int(minutes // _BIN_MINUTES), _AFTERNOON_BIN_COUNT - 1
        )
    return None


def _strict_score_day_frame(
    ctx: FactorComputeContext, score_date: pd.Timestamp | None
) -> pd.DataFrame:
    """Keep only physical score-day rows through the strict 14:29 cutoff."""

    panel = ensure_trade_time(ctx.panel)
    frame = panel.reset_index().copy(deep=False)
    if score_date is None:
        return frame.iloc[0:0].copy()

    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    if not frame.empty and not bool((indexed.dt.normalize() == score_date).any()):
        raise ValueError(
            f"{KERNEL_NAME} has no indexed rows for signal day {score_date.date().isoformat()}"
        )
    bins = timestamps.dt.time.map(
        lambda value: _session_bin(value) if pd.notna(value) else None
    )
    keep = (
        indexed.notna()
        & timestamps.notna()
        & (indexed.dt.normalize() == score_date)
        & (timestamps.dt.normalize() == score_date)
        & bins.notna()
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    out["__icg_bin"] = bins.loc[keep].astype("int64")
    return out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}"
        )


def _finite_number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _nan_values(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


def _last_observation_per_bin(group: pd.DataFrame) -> pd.DataFrame:
    """Use each observed bin's final physical snapshot without synthetic bins."""

    ordered = group.sort_values(["trade_time", "seq"], kind="mergesort")
    return (
        ordered.groupby("__icg_bin", sort=True, as_index=False)
        .tail(1)
        .sort_values("__icg_bin", kind="mergesort")
    )


def _l1_bins(group: pd.DataFrame, *, need_trade_mass: bool) -> _L1Bins | None:
    if len(group) < _MIN_BINS:
        return None
    required = _COUNTER_FIELDS if need_trade_mass else _L1_FIELDS
    values = {
        field: pd.to_numeric(group[field], errors="coerce").to_numpy(dtype="float64")
        for field in required
    }
    last = values["last"]
    ask = values["ask_price1"]
    bid = values["bid_price1"]
    ask_depth = values["ask_volume1"]
    bid_depth = values["bid_volume1"]
    depth = ask_depth + bid_depth
    valid = (
        np.isfinite(last)
        & np.isfinite(ask)
        & np.isfinite(bid)
        & np.isfinite(ask_depth)
        & np.isfinite(bid_depth)
        & (last > 0.0)
        & (ask > 0.0)
        & (bid > 0.0)
        & (ask >= bid)
        & (ask_depth >= 0.0)
        & (bid_depth >= 0.0)
        & (depth > _EPS)
    )
    if need_trade_mass:
        counter = values["num_trades"]
        valid = valid & np.isfinite(counter) & (counter >= 0.0)
        if len(counter) > 1 and bool((np.diff(counter) < -_EPS).any()):
            return None
    if not bool(valid.all()):
        return None

    sampled = _last_observation_per_bin(group)
    if len(sampled) < _MIN_BINS:
        return None
    bins = pd.to_numeric(sampled["__icg_bin"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_last = pd.to_numeric(sampled["last"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_ask = pd.to_numeric(sampled["ask_price1"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_bid = pd.to_numeric(sampled["bid_price1"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_ask_depth = pd.to_numeric(sampled["ask_volume1"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_bid_depth = pd.to_numeric(sampled["bid_volume1"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_mid = (sampled_ask + sampled_bid) / 2.0
    sampled_depth = sampled_ask_depth + sampled_bid_depth
    sampled_valid = (
        np.isfinite(bins)
        & np.isfinite(sampled_last)
        & np.isfinite(sampled_mid)
        & np.isfinite(sampled_depth)
        & (sampled_last > 0.0)
        & (sampled_mid > 0.0)
        & (sampled_depth > _EPS)
    )
    if not bool(sampled_valid.all()):
        return None

    mass: np.ndarray | None = None
    if need_trade_mass:
        sampled_counter = pd.to_numeric(
            sampled["num_trades"], errors="coerce"
        ).to_numpy(dtype="float64")
        if not (np.isfinite(sampled_counter).all() and (sampled_counter >= 0.0).all()):
            return None
        mass = np.diff(sampled_counter, prepend=sampled_counter[0])
        mass[0] = sampled_counter[0]
        if not (np.isfinite(mass).all() and (mass >= -_EPS).all()):
            return None
        mass = np.maximum(mass, 0.0)
        if (
            int((mass > _EPS).sum()) < _MIN_ACTIVE_MASS_BINS
            or float(mass.sum()) <= _EPS
        ):
            return None
    return _L1Bins(
        bins=bins.astype("int64"), last=sampled_last, mid=sampled_mid, trade_mass=mass
    )


def _price_mid_changes(bins: _L1Bins) -> tuple[np.ndarray, np.ndarray] | None:
    if len(bins.last) < _MIN_BINS or len(bins.mid) != len(bins.last):
        return None
    last_change = np.diff(np.log(bins.last))
    mid_change = np.diff(np.log(bins.mid))
    if len(last_change) < _MIN_BINS - 1 or not (
        np.isfinite(last_change).all() and np.isfinite(mid_change).all()
    ):
        return None
    return last_change, mid_change


def _scaled_information_increments(bins: _L1Bins) -> np.ndarray | None:
    changes = _price_mid_changes(bins)
    mass = bins.trade_mass
    if changes is None or mass is None or len(mass) != len(bins.last):
        return None
    last_change, mid_change = changes
    clock_mass = mass[1:]
    if len(clock_mass) != len(last_change) or not (
        np.isfinite(clock_mass).all() and (clock_mass >= 0.0).all()
    ):
        return None
    if (
        float(clock_mass.sum()) <= _EPS
        or int((clock_mass > _EPS).sum()) < _MIN_ACTIVE_MASS_BINS
    ):
        return None
    last_scale = float(np.sqrt(np.mean(np.square(last_change))))
    mid_scale = float(np.sqrt(np.mean(np.square(mid_change))))
    mass_scale = float(np.mean(clock_mass))
    if not (last_scale > _EPS and mid_scale > _EPS and mass_scale > _EPS):
        return None
    out = np.column_stack(
        (last_change / last_scale, mid_change / mid_scale, clock_mass / mass_scale)
    )
    return out if np.isfinite(out).all() else None


def _curve_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_values(_CURVE_TRAJECTORY_SIGNALS)
    bins = _l1_bins(group, need_trade_mass=True)
    if bins is None:
        return out
    increments = _scaled_information_increments(bins)
    if increments is None or len(increments) < 4:
        return out
    step_norm = np.linalg.norm(increments, axis=1)
    chord = float(np.linalg.norm(increments.sum(axis=0)))
    if not (np.isfinite(step_norm).all() and chord > _EPS) or bool(
        (step_norm <= _EPS).any()
    ):
        return out
    unit = increments / step_norm[:, None]
    if not np.isfinite(unit).all():
        return out
    out["icg_curve_arc_chord_excess"] = float(step_norm.sum() / chord - 1.0)
    turns = np.diff(unit, axis=0)
    out["icg_curve_turning_energy"] = (
        float(np.mean(np.square(turns).sum(axis=1))) if len(turns) else float("nan")
    )
    triple = np.einsum(
        "ij,ij->i",
        np.cross(unit[:-2], unit[1:-1]),
        unit[2:],
    )
    out["icg_curve_torsion_intensity"] = (
        float(np.mean(np.abs(triple))) if len(triple) else float("nan")
    )
    return {signal: _finite_number(value) for signal, value in out.items()}


def _simplex_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_values(_CHANNEL_SIMPLEX_SIGNALS)
    bins = _l1_bins(group, need_trade_mass=True)
    if bins is None or bins.trade_mass is None:
        return out
    changes = _price_mid_changes(bins)
    if changes is None:
        return out
    last_change, mid_change = changes
    mass = bins.trade_mass[1:]
    channels = np.column_stack((np.abs(last_change), np.abs(mid_change), mass))
    if (
        len(channels) < _MIN_BINS - 1
        or not np.isfinite(channels).all()
        or bool((channels < 0.0).any())
    ):
        return out
    totals = channels.sum(axis=0)
    if not bool((totals > _EPS).all()):
        return out
    profile = channels / totals[None, :]
    covariance = np.cov(profile, rowvar=False, ddof=0)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return out
    eigenvalues = np.linalg.eigvalsh(covariance)
    if (
        len(eigenvalues) != 3
        or not np.isfinite(eigenvalues).all()
        or eigenvalues[-1] <= _EPS
    ):
        return out
    eigenvalues = np.maximum(eigenvalues, 0.0)
    determinant = float(np.prod(eigenvalues))
    total = float(eigenvalues.sum())
    if total <= _EPS:
        return out
    out["icg_simplex_covariance_volume"] = float(np.cbrt(determinant))
    out["icg_simplex_planarity_ratio"] = float(eigenvalues[1] / eigenvalues[2])
    out["icg_simplex_axis_anisotropy"] = float(
        (eigenvalues[2] - eigenvalues[0]) / total
    )
    return {signal: _finite_number(value) for signal, value in out.items()}


def _weighted_moments(
    location: np.ndarray, weight: np.ndarray
) -> tuple[float, float] | None:
    if len(location) != len(weight) or len(location) < 3:
        return None
    if not (
        np.isfinite(location).all()
        and np.isfinite(weight).all()
        and (weight >= 0.0).all()
    ):
        return None
    total = float(weight.sum())
    if total <= _EPS:
        return None
    mean = float(np.dot(location, weight) / total)
    centered = location - mean
    variance = float(np.dot(np.square(centered), weight) / total)
    if variance <= _EPS:
        return variance, float("nan")
    skew = float(np.dot(np.power(centered, 3), weight) / total / (variance**1.5))
    return variance, skew if np.isfinite(skew) else float("nan")


def _transport_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_values(_MASS_TRANSPORT_SIGNALS)
    bins = _l1_bins(group, need_trade_mass=False)
    if bins is None:
        return out
    changes = _price_mid_changes(bins)
    if changes is None:
        return out
    last_change, mid_change = changes
    last_mass = np.abs(last_change)
    mid_mass = np.abs(mid_change)
    last_total, mid_total = float(last_mass.sum()), float(mid_mass.sum())
    if last_total <= _EPS or mid_total <= _EPS:
        return out
    last_profile = last_mass / last_total
    mid_profile = mid_mass / mid_total
    if not (np.isfinite(last_profile).all() and np.isfinite(mid_profile).all()):
        return out
    location = (bins.bins[1:].astype("float64") + 0.5) / float(_TOTAL_BIN_COUNT)
    last_moments = _weighted_moments(location, last_profile)
    mid_moments = _weighted_moments(location, mid_profile)
    if last_moments is None or mid_moments is None:
        return out
    last_variance, last_skew = last_moments
    mid_variance, mid_skew = mid_moments
    if not (np.isfinite(last_skew) and np.isfinite(mid_skew)):
        return out
    out["icg_pq_transport_distance"] = float(
        np.mean(np.abs(np.cumsum(last_profile) - np.cumsum(mid_profile)))
    )
    out["icg_pq_transport_dispersion_gap"] = float(last_variance - mid_variance)
    out["icg_pq_transport_skewness_gap"] = float(last_skew - mid_skew)
    return {signal: _finite_number(value) for signal, value in out.items()}


def _book_bins(group: pd.DataFrame) -> _BookBins | None:
    if len(group) < _MIN_BINS:
        return None
    ask_price = np.column_stack(
        [
            pd.to_numeric(group[f"ask_price{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    bid_price = np.column_stack(
        [
            pd.to_numeric(group[f"bid_price{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    ask_depth = np.column_stack(
        [
            pd.to_numeric(group[f"ask_volume{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    bid_depth = np.column_stack(
        [
            pd.to_numeric(group[f"bid_volume{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    last = pd.to_numeric(group["last"], errors="coerce").to_numpy(dtype="float64")
    total_depth = ask_depth.sum(axis=1) + bid_depth.sum(axis=1)
    valid = (
        np.isfinite(last)
        & np.isfinite(ask_price).all(axis=1)
        & np.isfinite(bid_price).all(axis=1)
        & np.isfinite(ask_depth).all(axis=1)
        & np.isfinite(bid_depth).all(axis=1)
        & (last > 0.0)
        & (ask_price > 0.0).all(axis=1)
        & (bid_price > 0.0).all(axis=1)
        & (ask_depth >= 0.0).all(axis=1)
        & (bid_depth >= 0.0).all(axis=1)
        & (ask_price[:, 0] >= bid_price[:, 0])
        & (np.diff(ask_price, axis=1) >= 0.0).all(axis=1)
        & (np.diff(bid_price, axis=1) <= 0.0).all(axis=1)
        & (total_depth > _EPS)
    )
    if not bool(valid.all()):
        return None
    sampled = _last_observation_per_bin(group)
    if len(sampled) < _MIN_BINS:
        return None
    ask_price = np.column_stack(
        [
            pd.to_numeric(sampled[f"ask_price{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    bid_price = np.column_stack(
        [
            pd.to_numeric(sampled[f"bid_price{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    ask_depth = np.column_stack(
        [
            pd.to_numeric(sampled[f"ask_volume{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    bid_depth = np.column_stack(
        [
            pd.to_numeric(sampled[f"bid_volume{level}"], errors="coerce").to_numpy(
                dtype="float64"
            )
            for level in range(1, 6)
        ]
    )
    sampled_last = pd.to_numeric(sampled["last"], errors="coerce").to_numpy(
        dtype="float64"
    )
    ask_total = ask_depth.sum(axis=1)
    bid_total = bid_depth.sum(axis=1)
    total = ask_total + bid_total
    midpoint = (ask_price[:, 0] + bid_price[:, 0]) / 2.0
    centroid = (
        np.sum(ask_price * ask_depth, axis=1) + np.sum(bid_price * bid_depth, axis=1)
    ) / total
    imbalance = (bid_total - ask_total) / total
    centroid_dislocation = np.log(centroid / midpoint)
    bins = pd.to_numeric(sampled["__icg_bin"], errors="coerce").to_numpy(
        dtype="float64"
    )
    sampled_valid = (
        np.isfinite(bins)
        & np.isfinite(sampled_last)
        & np.isfinite(midpoint)
        & np.isfinite(imbalance)
        & np.isfinite(centroid_dislocation)
        & (sampled_last > 0.0)
        & (midpoint > 0.0)
    )
    if not bool(sampled_valid.all()):
        return None
    return _BookBins(
        bins=bins.astype("int64"),
        last=sampled_last,
        mid=midpoint,
        imbalance=imbalance,
        centroid_dislocation=centroid_dislocation,
    )


def _scaled_book_discovery_path(bins: _BookBins) -> np.ndarray | None:
    if len(bins.last) < _MIN_BINS:
        return None
    last_state = np.log(bins.last / bins.last[0])
    mid_state = np.log(bins.mid / bins.mid[0])
    points = np.column_stack(
        (
            last_state - mid_state,
            bins.imbalance - bins.imbalance[0],
            bins.centroid_dislocation - bins.centroid_dislocation[0],
        )
    )
    deltas = np.diff(points, axis=0)
    if len(deltas) < 3 or not (np.isfinite(points).all() and np.isfinite(deltas).all()):
        return None
    scales = np.sqrt(np.mean(np.square(deltas), axis=0))
    if not bool((scales > _EPS).all()):
        return None
    out = points / scales[None, :]
    return out if np.isfinite(out).all() else None


def _book_manifold_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_values(_BOOK_MANIFOLD_SIGNALS)
    bins = _book_bins(group)
    if bins is None:
        return out
    path = _scaled_book_discovery_path(bins)
    if path is None or len(path) < _MIN_BINS:
        return out
    loop_vector = np.cross(path[:-1], path[1:]).sum(axis=0)
    increments = np.diff(path, axis=0)
    turns = np.cross(increments[:-1], increments[1:])
    torsion = np.einsum(
        "ij,ij->i",
        np.cross(increments[:-2], increments[1:-1]),
        increments[2:],
    )
    out["icg_book_discovery_loop_area"] = float(
        0.5 * np.linalg.norm(loop_vector) / max(1, len(path) - 1)
    )
    out["icg_book_discovery_pressure_turning"] = (
        float(np.mean(np.linalg.norm(turns, axis=1))) if len(turns) else float("nan")
    )
    out["icg_book_discovery_centroid_torsion"] = (
        float(np.mean(np.abs(torsion))) if len(torsion) else float("nan")
    )
    return {signal: _finite_number(value) for signal, value in out.items()}


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "information_clock_curve_trajectory": _curve_metrics,
    "information_clock_channel_simplex": _simplex_metrics,
    "price_quote_mass_transport": _transport_metrics,
    "book_discovery_manifold": _book_manifold_metrics,
}


def _family_feature_frame(ctx: FactorComputeContext, family: str) -> pd.DataFrame:
    if family not in _FAMILY_CALCULATORS:
        raise KeyError(f"{KERNEL_NAME} unknown family: {family}")
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:family:{family}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    strict_frame = _strict_score_day_frame(ctx, score_date)
    _require_columns(strict_frame, _REQUIRED_FIELDS[family])
    signals = _FAMILY_SIGNALS[family]
    groups = {
        str(code): group for code, group in strict_frame.groupby("code", sort=False)
    }
    calculator = _FAMILY_CALCULATORS[family]
    rows: list[dict[str, object]] = []
    for dt, raw_code in output_index:
        values: dict[str, object] = {"dt": dt, "code": raw_code}
        group = groups.get(str(raw_code))
        metrics = calculator(group) if group is not None else _nan_values(signals)
        values.update(
            {signal: _finite_number(metrics.get(signal)) for signal in signals}
        )
        rows.append(values)
    if rows:
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(signals)].sort_index()
    else:
        built = pd.DataFrame(index=output_index, columns=signals, dtype="float64")
    built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        prior = ctx.cache.get(cache_key)
        if isinstance(prior, pd.DataFrame):
            return prior
        ctx.cache[cache_key] = built
    return built


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayInformationClockGeometryV1(Factor):
    """Research-only strict-T1430 cbond information-clock geometry kernel."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _family_feature_frame(ctx, entry.family)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out.replace([np.inf, -np.inf], np.nan)


__all__ = [
    "CATALOG_VERSION",
    "KERNEL_NAME",
    "CatalogEntry",
    "FactorMiningIntradayInformationClockGeometryV1",
    "FORMULAS",
    "factor_mining_catalog",
    "factor_mining_intraday_information_clock_geometry_v1_catalog",
]
