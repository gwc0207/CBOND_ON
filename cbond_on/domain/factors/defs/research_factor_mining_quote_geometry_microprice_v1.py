"""Research-only strict-PIT L1--L5 quote-geometry and microprice catalogue.

The existing research book candidates already cover quote/last lead-lag,
queue events, update cascades, depth-centroid relocation, and ladder-reprice
directions.  This import-only module deliberately uses a separate layer of
information: the *shape* of each five-level depth distribution, curvature of
the displayed price ladder, and the alignment of the L1 microprice with the
traded last price.

Only ``FactorComputeContext.panel`` is consumed.  Each retained observation
must be physically on the score date, in a continuous session, and no later
than 14:29.  Any malformed timestamp/order, crossed book, invalid ladder, or
negative depth fails closed for that instrument; no input is imputed.  It does
not read files, databases, labels, PnL, factor outputs, scores, or live state,
and intentionally remains outside ``defs.__init__`` and all configurations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_quote_geometry_microprice_v1"
CATALOG_VERSION = "20260803_quote_geometry_microprice_v1"

_EPS = 1e-12
_MIN_ROWS = 8
_MIN_PAIR_COUNT = 4
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EARLY_END = dt_time(10, 30)
_LATE_START = dt_time(13, 30)


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable signal with a distinct quote-geometry hypothesis."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_DEPTH_INFORMATION_SIGNALS = (
    "qgeo_depth_entropy_asymmetry_mean",
    "qgeo_outer_depth_asymmetry_phase_shift",
)
_LADDER_CURVATURE_SIGNALS = (
    "qgeo_ladder_convexity_asymmetry_mean",
    "qgeo_ladder_convexity_phase_shift",
)
_MICRO_LAST_ALIGNMENT_SIGNALS = (
    "qgeo_micro_last_next_return_sign_alignment",
    "qgeo_micro_last_dislocation_persistence",
)
_BID_ASK_GEOMETRY_COUPLING_SIGNALS = (
    "qgeo_bidask_depth_shape_coherence",
    "qgeo_price_depth_geometry_coupling",
)

_CATALOG = (
    _entries(
        "five_level_depth_information_geometry",
        _DEPTH_INFORMATION_SIGNALS,
        "Five-level depth entropy and outer-layer mass distinguish the distribution of displayed liquidity from total depth, L1 concentration, or reprice-conditioned migration.",
    )
    + _entries(
        "price_ladder_spacing_curvature",
        _LADDER_CURVATURE_SIGNALS,
        "Relative outer-versus-inner spacing of L1--L5 quotes is ladder curvature, not the endpoint book slope or a reprice direction.",
    )
    + _entries(
        "microprice_last_execution_alignment",
        _MICRO_LAST_ALIGNMENT_SIGNALS,
        "The L1 microprice can align with the next visible last-price move or persistently disagree with last without being a raw microprice bias or return correlation.",
    )
    + _entries(
        "bid_ask_geometry_coupling",
        _BID_ASK_GEOMETRY_COUPLING_SIGNALS,
        "Coupling between bid/ask depth shapes and between depth asymmetry and ladder curvature is distinct from same-side queue updates or centroid relocation.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_FAMILY_SIGNALS = {
    "five_level_depth_information_geometry": _DEPTH_INFORMATION_SIGNALS,
    "price_ladder_spacing_curvature": _LADDER_CURVATURE_SIGNALS,
    "microprice_last_execution_alignment": _MICRO_LAST_ALIGNMENT_SIGNALS,
    "bid_ask_geometry_coupling": _BID_ASK_GEOMETRY_COUPLING_SIGNALS,
}

FORMULAS = {
    "qgeo_depth_entropy_asymmetry_mean": "mean(H_bid,t-H_ask,t), H_side=-sum_{k=1..5} w_side,k,t*log(w_side,k,t)/log(5), w=depth/sum(depth).",
    "qgeo_outer_depth_asymmetry_phase_shift": "mean_late((w_bid,4+w_bid,5)-(w_ask,4+w_ask,5)) - mean_early(same).",
    "qgeo_ladder_convexity_asymmetry_mean": "mean(C_bid,t-C_ask,t), C=(mean(outer two log gaps)-mean(inner two log gaps))/(mean(outer)+mean(inner)).",
    "qgeo_ladder_convexity_phase_shift": "mean_late((C_bid+C_ask)/2)-mean_early((C_bid+C_ask)/2).",
    "qgeo_micro_last_next_return_sign_alignment": "mean(sign(z_t)*sign(log(last_{t+1}/last_t))) over valid adjacent same-session intervals, z_t=(microprice_t-last_t)/(ask1_t-bid1_t).",
    "qgeo_micro_last_dislocation_persistence": "corr(z_t,z_{t+1}) over valid adjacent same-session intervals with the same z_t definition.",
    "qgeo_bidask_depth_shape_coherence": "corr(H_bid,t,H_ask,t) across valid score-day snapshots.",
    "qgeo_price_depth_geometry_coupling": "corr(C_bid,t-C_ask,t, H_bid,t-H_ask,t) across valid score-day snapshots.",
}


def quote_geometry_microprice_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable four-family, eight-signal research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic research catalogue compatibility entrypoint."""

    return quote_geometry_microprice_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _book_columns() -> tuple[str, ...]:
    return tuple(
        column
        for level in range(1, 6)
        for column in (
            f"ask_price{level}",
            f"bid_price{level}",
            f"ask_volume{level}",
            f"bid_volume{level}",
        )
    )


_REQUIRED_COLUMNS = ("trade_time", "last", *_book_columns())


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing required panel column(s): {', '.join(missing)}")


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw = panel.attrs.get("__build_day__")
    if raw is not None:
        value = pd.Timestamp(raw)
        if pd.isna(value):
            raise ValueError(f"{KERNEL_NAME} has invalid panel __build_day__")
        return value.normalize()
    if panel.empty:
        return None
    dates = pd.to_datetime(panel.index.get_level_values("dt"), errors="coerce").normalize()
    unique = pd.Index(dates[dates.notna()]).unique()
    if len(unique) != 1:
        raise ValueError(f"{KERNEL_NAME} requires panel __build_day__ for a multi-date panel")
    return pd.Timestamp(unique[0]).normalize()


def _empty_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([], names=["dt", "code"])


def _output_index(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.MultiIndex:
    panel = ensure_panel_index(panel)
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    selected = keys.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(selected, names=("dt", "code"))


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _session_id(clock: dt_time) -> int:
    return 0 if _MORNING_START <= clock <= _MORNING_END else 1


def _score_day_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    """Filter a rolling panel to physically observed score-day T1430 data."""

    target = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:score_day_frame:{target}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    panel = ensure_trade_time(ctx.panel)
    frame = panel.reset_index().copy(deep=False)
    if target is None:
        out = frame.iloc[0:0].copy()
    else:
        indexed = pd.to_datetime(frame["dt"], errors="coerce")
        timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
        clocks = timestamps.dt.time
        continuous = timestamps.notna() & clocks.map(
            lambda clock: _continuous_session(clock) if pd.notna(clock) else False
        )
        keep = (
            indexed.notna()
            & (indexed.dt.normalize() == target)
            & timestamps.notna()
            & (timestamps.dt.normalize() == target)
            & continuous
            & (clocks <= _CUTOFF)
        )
        out = frame.loc[keep].copy()
        out["trade_time"] = timestamps.loc[keep]
        out["__session"] = [
            _session_id(clock) for clock in clocks.loc[keep]
        ]
        out = out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = out
    return out


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < _MIN_PAIR_COUNT:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if float(np.std(x)) <= _EPS or float(np.std(y)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _nan_record(signals: tuple[str, ...]) -> dict[str, float]:
    return {signal: float("nan") for signal in signals}


@dataclass(frozen=True)
class _BookGeometry:
    last: np.ndarray
    sessions: np.ndarray
    entropy_bid: np.ndarray
    entropy_ask: np.ndarray
    outer_bid: np.ndarray
    outer_ask: np.ndarray
    convexity_bid: np.ndarray
    convexity_ask: np.ndarray
    micro_last_dislocation: np.ndarray


def _depth_entropy(depth: np.ndarray) -> np.ndarray | None:
    totals = depth.sum(axis=1)
    if not (np.isfinite(totals).all() and (totals > _EPS).all()):
        return None
    weights = depth / totals[:, None]
    if not (np.isfinite(weights).all() and (weights >= 0.0).all()):
        return None
    term = np.where(weights > 0.0, weights * np.log(weights), 0.0)
    entropy = -term.sum(axis=1) / np.log(5.0)
    return entropy if np.isfinite(entropy).all() else None


def _ladder_convexity(prices: np.ndarray, *, side: str) -> np.ndarray | None:
    if side == "ask":
        gaps = np.log(prices[:, 1:] / prices[:, :-1])
    elif side == "bid":
        gaps = np.log(prices[:, :-1] / prices[:, 1:])
    else:
        raise ValueError(f"{KERNEL_NAME} unknown ladder side: {side}")
    if not np.isfinite(gaps).all() or (gaps < 0.0).any():
        return None
    inner = np.mean(gaps[:, :2], axis=1)
    outer = np.mean(gaps[:, 2:], axis=1)
    denominator = outer + inner
    valid = np.isfinite(denominator) & (denominator > _EPS)
    if not bool(valid.all()):
        return None
    return (outer - inner) / denominator


def _book_geometry(frame: pd.DataFrame) -> _BookGeometry | None:
    if len(frame) < _MIN_ROWS:
        return None
    if frame["seq"].duplicated(keep=False).any() or frame["trade_time"].duplicated(keep=False).any():
        return None
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        return None
    time_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype("int64")
    if (np.diff(time_ns) <= 0).any():
        return None

    ask_price = np.column_stack([_numeric(frame, f"ask_price{level}") for level in range(1, 6)])
    bid_price = np.column_stack([_numeric(frame, f"bid_price{level}") for level in range(1, 6)])
    ask_depth = np.column_stack([_numeric(frame, f"ask_volume{level}") for level in range(1, 6)])
    bid_depth = np.column_stack([_numeric(frame, f"bid_volume{level}") for level in range(1, 6)])
    last = _numeric(frame, "last")
    sessions = _numeric(frame, "__session")
    valid = (
        np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and np.isfinite(last).all()
        and np.isfinite(sessions).all()
        and (ask_price > _EPS).all()
        and (bid_price > _EPS).all()
        and (last > _EPS).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (ask_price[:, 0] >= bid_price[:, 0]).all()
        and (np.diff(ask_price, axis=1) >= 0.0).all()
        and (np.diff(bid_price, axis=1) <= 0.0).all()
    )
    if not valid:
        return None
    entropy_bid = _depth_entropy(bid_depth)
    entropy_ask = _depth_entropy(ask_depth)
    convexity_bid = _ladder_convexity(bid_price, side="bid")
    convexity_ask = _ladder_convexity(ask_price, side="ask")
    if any(value is None for value in (entropy_bid, entropy_ask, convexity_bid, convexity_ask)):
        return None
    touch_spread = ask_price[:, 0] - bid_price[:, 0]
    depth_touch = ask_depth[:, 0] + bid_depth[:, 0]
    if not ((touch_spread > _EPS).all() and (depth_touch > _EPS).all()):
        return None
    microprice = (
        ask_price[:, 0] * bid_depth[:, 0] + bid_price[:, 0] * ask_depth[:, 0]
    ) / depth_touch
    micro_last_dislocation = (microprice - last) / touch_spread
    if not np.isfinite(micro_last_dislocation).all():
        return None
    return _BookGeometry(
        last=last,
        sessions=sessions.astype("int64"),
        entropy_bid=entropy_bid,
        entropy_ask=entropy_ask,
        outer_bid=bid_depth[:, 3:].sum(axis=1) / bid_depth.sum(axis=1),
        outer_ask=ask_depth[:, 3:].sum(axis=1) / ask_depth.sum(axis=1),
        convexity_bid=convexity_bid,
        convexity_ask=convexity_ask,
        micro_last_dislocation=micro_last_dislocation,
    )


def _early_late_masks(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    clocks = pd.to_datetime(frame["trade_time"], errors="coerce").dt.time
    if clocks.isna().any():
        return None
    early = ((clocks >= _MORNING_START) & (clocks <= _EARLY_END)).to_numpy(dtype=bool)
    late = ((clocks >= _LATE_START) & (clocks <= _CUTOFF)).to_numpy(dtype=bool)
    if min(int(early.sum()), int(late.sum())) < 3:
        return None
    return early, late


def _depth_information_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_DEPTH_INFORMATION_SIGNALS)
    geometry = _book_geometry(frame)
    masks = _early_late_masks(frame)
    if geometry is None or masks is None:
        return out
    early, late = masks
    entropy_asymmetry = geometry.entropy_bid - geometry.entropy_ask
    outer_asymmetry = geometry.outer_bid - geometry.outer_ask
    out["qgeo_depth_entropy_asymmetry_mean"] = float(np.mean(entropy_asymmetry))
    out["qgeo_outer_depth_asymmetry_phase_shift"] = float(
        np.mean(outer_asymmetry[late]) - np.mean(outer_asymmetry[early])
    )
    return out


def _ladder_curvature_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_LADDER_CURVATURE_SIGNALS)
    geometry = _book_geometry(frame)
    masks = _early_late_masks(frame)
    if geometry is None or masks is None:
        return out
    early, late = masks
    asymmetry = geometry.convexity_bid - geometry.convexity_ask
    joint = (geometry.convexity_bid + geometry.convexity_ask) / 2.0
    out["qgeo_ladder_convexity_asymmetry_mean"] = float(np.mean(asymmetry))
    out["qgeo_ladder_convexity_phase_shift"] = float(np.mean(joint[late]) - np.mean(joint[early]))
    return out


def _micro_last_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_MICRO_LAST_ALIGNMENT_SIGNALS)
    geometry = _book_geometry(frame)
    if geometry is None:
        return out
    returns = np.diff(np.log(geometry.last))
    same_session = geometry.sessions[:-1] == geometry.sessions[1:]
    displacement = geometry.micro_last_dislocation[:-1]
    valid = same_session & np.isfinite(displacement) & np.isfinite(returns)
    nonzero = valid & (np.abs(displacement) > _EPS) & (np.abs(returns) > _EPS)
    if int(nonzero.sum()) >= _MIN_PAIR_COUNT:
        out["qgeo_micro_last_next_return_sign_alignment"] = float(
            np.mean(np.sign(displacement[nonzero]) * np.sign(returns[nonzero]))
        )
    persistence = same_session & np.isfinite(geometry.micro_last_dislocation[:-1]) & np.isfinite(
        geometry.micro_last_dislocation[1:]
    )
    if int(persistence.sum()) >= _MIN_PAIR_COUNT:
        out["qgeo_micro_last_dislocation_persistence"] = _safe_corr(
            geometry.micro_last_dislocation[:-1][persistence],
            geometry.micro_last_dislocation[1:][persistence],
        )
    return out


def _geometry_coupling_metrics(frame: pd.DataFrame) -> dict[str, float]:
    out = _nan_record(_BID_ASK_GEOMETRY_COUPLING_SIGNALS)
    geometry = _book_geometry(frame)
    if geometry is None:
        return out
    out["qgeo_bidask_depth_shape_coherence"] = _safe_corr(
        geometry.entropy_bid, geometry.entropy_ask
    )
    out["qgeo_price_depth_geometry_coupling"] = _safe_corr(
        geometry.convexity_bid - geometry.convexity_ask,
        geometry.entropy_bid - geometry.entropy_ask,
    )
    return out


_FAMILY_CALCULATORS: dict[str, Callable[[pd.DataFrame], dict[str, float]]] = {
    "five_level_depth_information_geometry": _depth_information_metrics,
    "price_ladder_spacing_curvature": _ladder_curvature_metrics,
    "microprice_last_execution_alignment": _micro_last_metrics,
    "bid_ask_geometry_coupling": _geometry_coupling_metrics,
}


def _full_feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    score_date = _score_date_from_panel(ctx.panel)
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features:{score_date}"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    all_signals = tuple(entry.signal for entry in _CATALOG)
    if score_date is None:
        built = pd.DataFrame(index=_empty_index(), columns=all_signals, dtype="float64")
    else:
        frame = _score_day_frame(ctx)
        _require_columns(frame)
        output_index = _output_index(ctx.panel, score_date=score_date)
        rows: list[dict[str, object]] = []
        for (dt, code), group in frame.groupby(["dt", "code"], sort=False):
            row: dict[str, object] = {"dt": dt, "code": str(code)}
            for calculator in _FAMILY_CALCULATORS.values():
                row.update(calculator(group))
            rows.append(row)
        if rows:
            built = pd.DataFrame(rows).set_index(["dt", "code"])[list(all_signals)].sort_index()
            built = built.reindex(output_index)
        else:
            built = pd.DataFrame(index=output_index, columns=all_signals, dtype="float64")
    built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningQuoteGeometryMicropriceV1(Factor):
    """Emit one L1--L5 quote-geometry signal selected by ``params.signal``."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        frame = _full_feature_frame(ctx)
        out = frame[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
