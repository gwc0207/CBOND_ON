"""Research-only strict-PIT mapped bond/stock order-book geometry catalogue.

The candidate set compares *displayed liquidity geometry* of a convertible
bond with its context-mapped underlying stock.  It intentionally excludes
prices, returns, trades, labels, PnL, and all daily/structural inputs.

This is disjoint from the existing cross-asset endpoint spread/imbalance gaps
and from the single-asset quote-geometry catalogue.  Rather than reuse a
terminal L1 statistic, it measures matched five-minute paths of L1 imbalance,
touch-depth share, relative-spread state, and the complete L1--L5 depth
distribution.  In particular, Hellinger distances compare the *two assets'*
displayed depth curves; they are not an intra-book entropy, price-ladder
curvature, price response, or reprice event.

Only ``ctx.panel``, ``ctx.stock_panel``, and ``ctx.bond_stock_map`` are used.
Both panels are independently restricted to physical T-day continuous-auction
snapshots through 14:29:00.  The supplied mapping is treated as the upstream
point-in-time context contract; rows explicitly dated after the score day,
duplicate mappings, malformed mappings, incomplete books, and data gaps fail
closed to ``NaN``.  No files, databases, labels, scores, pool/mask state,
daily data, or live configuration are accessed.  The module is import-only
and research-only and is deliberately absent from ``defs.__init__``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index


KERNEL_NAME = "factor_mining_cross_asset_book_geometry_v1"
CATALOG_VERSION = "20260803_cross_asset_book_geometry_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_COMMON_BINS = 12
_MIN_PHASE_BINS = 4
_MIN_CORRELATION_BINS = 8
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


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only cross-asset book candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_L1_STATE_SIGNALS = (
    "cabg_l1_imbalance_directional_agreement",
    "cabg_touch_depth_share_gap_mean",
    "cabg_relative_spread_gap_iqr",
)
_DEPTH_CURVE_SIGNALS = (
    "cabg_joint_depth_curve_hellinger_mean",
    "cabg_bid_ask_curve_divergence_asymmetry",
    "cabg_joint_curve_divergence_phase_shift",
)
_CO_MOVEMENT_SIGNALS = (
    "cabg_l1_imbalance_time_correlation",
    "cabg_relative_spread_time_correlation",
    "cabg_touch_depth_migration_time_correlation",
)

_CATALOG = (
    _entries(
        "cross_asset_l1_liquidity_state",
        _L1_STATE_SIGNALS,
        "Matched L1 imbalance, L1-versus-L1--L5 touch share, and the path dispersion of relative-spread gaps are liquidity states rather than terminal spread/imbalance aliases.",
    )
    + _entries(
        "cross_asset_depth_curve_geometry",
        _DEPTH_CURVE_SIGNALS,
        "Hellinger geometry compares mapped bond and stock L1--L5 displayed-depth distributions, including side asymmetry and clock evolution, without using within-asset entropy or price-ladder curvature.",
    )
    + _entries(
        "cross_asset_book_state_comovement",
        _CO_MOVEMENT_SIGNALS,
        "Time-path synchrony of imbalance, relative spread, and touch-depth migration captures joint liquidity-state movement, rather than any return, event-count, or early-late scalar interaction.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# At matched five-minute endpoints, I=(bid_volume1-ask_volume1)/(bid_volume1+ask_volume1),
# T=(bid_volume1+ask_volume1)/sum_{k=1..5}(bid_volumek+ask_volumek),
# R=(ask_price1-bid_price1)/((ask_price1+bid_price1)/2), and H(p,q) is the
# Hellinger distance of non-negative distributions p and q.  D is the 10-level
# bid+ask depth distribution; B/A are the corresponding five-level side paths.
FORMULAS: dict[str, str] = {
    "cabg_l1_imbalance_directional_agreement": "mean(sign(I_b,t)*sign(I_s,t)) over matched valid endpoints",
    "cabg_touch_depth_share_gap_mean": "mean(T_b,t-T_s,t) over matched valid endpoints",
    "cabg_relative_spread_gap_iqr": "q75(log(R_b,t)-log(R_s,t))-q25(log(R_b,t)-log(R_s,t))",
    "cabg_joint_depth_curve_hellinger_mean": "mean(H(D_b,t,D_s,t)) over matched valid endpoints",
    "cabg_bid_ask_curve_divergence_asymmetry": "mean(H(B_b,t,B_s,t)-H(A_b,t,A_s,t))",
    "cabg_joint_curve_divergence_phase_shift": "mean_late(H(D_b,D_s))-mean_early(H(D_b,D_s))",
    "cabg_l1_imbalance_time_correlation": "corr(I_b,t,I_s,t) over matched valid endpoints",
    "cabg_relative_spread_time_correlation": "corr(log(R_b,t),log(R_s,t)) over matched valid endpoints",
    "cabg_touch_depth_migration_time_correlation": "corr(delta T_b,t,delta T_s,t) over exactly adjacent five-minute endpoints",
}


def factor_mining_cross_asset_book_geometry_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return factor_mining_cross_asset_book_geometry_catalog()


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


_REQUIRED_PANEL_COLUMNS = ("trade_time", *_book_columns())
_PRICE_COLUMNS = tuple(column for column in _book_columns() if "price" in column)
_VOLUME_COLUMNS = tuple(column for column in _book_columns() if "volume" in column)


def _canonical_market_code(value: object) -> str:
    """Accept only an exchange-qualified context code; never infer a map row."""

    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if "." not in text:
        return ""
    bare, suffix = text.rsplit(".", 1)
    suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
    return f"{bare}.{suffix}" if bare and suffix in _MARKET_EXCHANGES else ""


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
    panel = ensure_panel_index(panel)
    raw_day = panel.attrs.get("__build_day__")
    if raw_day is not None:
        score_date = pd.Timestamp(raw_day)
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


def _output_index(ctx: FactorComputeContext, score_date: pd.Timestamp | None) -> pd.MultiIndex:
    panel = ensure_panel_index(ctx.panel)
    if panel.empty or score_date is None:
        return pd.MultiIndex.from_tuples([], names=["dt", "code"])
    keys = panel.index.to_frame(index=False).loc[:, ["dt", "code"]].copy()
    dates = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[dates == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(panel: pd.DataFrame | None, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only physical score-day book snapshots through the 14:29 cutoff."""

    if (
        not isinstance(panel, pd.DataFrame)
        or any(column not in panel.columns for column in _REQUIRED_PANEL_COLUMNS)
    ):
        return pd.DataFrame(columns=["dt", "code", "seq", *_REQUIRED_PANEL_COLUMNS])
    checked = ensure_panel_index(panel)
    frame = checked.reset_index().loc[:, ["dt", "code", "seq", *_REQUIRED_PANEL_COLUMNS]].copy()
    labels = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
    keep = (
        labels.notna()
        & (labels.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    return out.sort_values(["code", "trade_time", "seq"], kind="mergesort")


def _context_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Use only unambiguous context mappings that are not explicitly future-dated."""

    mapping = ctx.bond_stock_map
    if (
        not isinstance(mapping, pd.DataFrame)
        or mapping.empty
        or "code" not in mapping.columns
        or "stock_code" not in mapping.columns
    ):
        return {}
    columns = ["code", "stock_code"]
    has_trade_date = "trade_date" in mapping.columns
    if has_trade_date:
        columns.append("trade_date")
    frame = mapping.loc[:, columns].copy()
    if has_trade_date:
        dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.loc[dates.notna() & (dates <= score_date)].copy()
    frame["bond_code"] = frame["code"].map(_canonical_market_code)
    frame["mapped_stock_code"] = frame["stock_code"].map(_canonical_market_code)
    frame = frame.loc[(frame["bond_code"] != "") & (frame["mapped_stock_code"] != "")].copy()
    if frame.empty:
        return {}
    # Do not make an arbitrary choice if the upstream mapping has duplicate bonds.
    frame = frame.loc[~frame["bond_code"].duplicated(keep=False)]
    return dict(zip(frame["bond_code"], frame["mapped_stock_code"], strict=False))


def _empty_path() -> pd.DataFrame:
    columns = ["imbalance", "relative_spread", "touch_share", "phase"]
    columns += [f"bid_weight{level}" for level in range(1, 6)]
    columns += [f"ask_weight{level}" for level in range(1, 6)]
    columns += [f"joint_weight{level}" for level in range(1, 11)]
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="event_time"))


def _asset_book_path(frame: pd.DataFrame) -> pd.DataFrame:
    """Build exact five-minute endpoint states; an invalid book invalidates its day."""

    if frame.empty or frame["trade_time"].duplicated(keep=False).any():
        return _empty_path()
    data = frame.loc[:, ["trade_time", "seq", *_PRICE_COLUMNS, *_VOLUME_COLUMNS]].copy()
    for column in (*_PRICE_COLUMNS, *_VOLUME_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    ask_prices = data[[f"ask_price{level}" for level in range(1, 6)]].to_numpy(dtype="float64")
    bid_prices = data[[f"bid_price{level}" for level in range(1, 6)]].to_numpy(dtype="float64")
    ask_volume = data[[f"ask_volume{level}" for level in range(1, 6)]].to_numpy(dtype="float64")
    bid_volume = data[[f"bid_volume{level}" for level in range(1, 6)]].to_numpy(dtype="float64")
    finite = np.isfinite(ask_prices).all(axis=1) & np.isfinite(bid_prices).all(axis=1)
    finite &= np.isfinite(ask_volume).all(axis=1) & np.isfinite(bid_volume).all(axis=1)
    positive_prices = (ask_prices > _EPS).all(axis=1) & (bid_prices > _EPS).all(axis=1)
    nonnegative_volume = (ask_volume >= 0.0).all(axis=1) & (bid_volume >= 0.0).all(axis=1)
    ordered = (np.diff(ask_prices, axis=1) > _EPS).all(axis=1)
    ordered &= (np.diff(bid_prices, axis=1) < -_EPS).all(axis=1)
    crossed = ask_prices[:, 0] > bid_prices[:, 0] + _EPS
    l1_total = ask_volume[:, 0] + bid_volume[:, 0]
    full_total = ask_volume.sum(axis=1) + bid_volume.sum(axis=1)
    valid = finite & positive_prices & nonnegative_volume & ordered & crossed
    valid &= l1_total > _EPS
    valid &= full_total > _EPS
    if not bool(valid.all()):
        return _empty_path()

    midpoint = (ask_prices[:, 0] + bid_prices[:, 0]) / 2.0
    relative_spread = (ask_prices[:, 0] - bid_prices[:, 0]) / midpoint
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    state: dict[str, object] = {
        "event_time": timestamps.dt.floor(_EVENT_BIN),
        "seq": data["seq"].to_numpy(),
        "imbalance": (bid_volume[:, 0] - ask_volume[:, 0]) / l1_total,
        "relative_spread": relative_spread,
        "touch_share": l1_total / full_total,
        "phase": np.where(timestamps.dt.time <= _EARLY_END, "early", np.where(timestamps.dt.time >= _LATE_START, "late", "middle")),
    }
    for level in range(1, 6):
        state[f"bid_weight{level}"] = bid_volume[:, level - 1] / bid_volume.sum(axis=1)
        state[f"ask_weight{level}"] = ask_volume[:, level - 1] / ask_volume.sum(axis=1)
    joint = np.concatenate((bid_volume, ask_volume), axis=1) / full_total[:, None]
    for level in range(1, 11):
        state[f"joint_weight{level}"] = joint[:, level - 1]
    out = pd.DataFrame(state)
    if out["event_time"].isna().any():
        return _empty_path()
    out = out.sort_values(["event_time", "seq"], kind="mergesort").groupby("event_time", sort=True).last()
    out.index.name = "event_time"
    return out.drop(columns=["seq"])


def _safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < _MIN_CORRELATION_BINS:
        return float("nan")
    lhs = left[valid]
    rhs = right[valid]
    if float(np.std(lhs)) <= _EPS or float(np.std(rhs)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(lhs, rhs)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _hellinger(frame: pd.DataFrame, *, kind: str) -> np.ndarray:
    if kind == "joint":
        columns = [f"joint_weight{level}" for level in range(1, 11)]
    else:
        columns = [f"{kind}_weight{level}" for level in range(1, 6)]
    bond = frame[[f"{column}_bond" for column in columns]].to_numpy(dtype="float64")
    stock = frame[[f"{column}_stock" for column in columns]].to_numpy(dtype="float64")
    valid = np.isfinite(bond).all(axis=1) & np.isfinite(stock).all(axis=1)
    valid &= (bond >= 0.0).all(axis=1) & (stock >= 0.0).all(axis=1)
    out = np.full(len(frame), np.nan, dtype="float64")
    out[valid] = np.sqrt(0.5 * np.square(np.sqrt(bond[valid]) - np.sqrt(stock[valid])).sum(axis=1))
    return out


def _adjacent_delta(values: pd.Series) -> pd.Series:
    delta = values.astype("float64").diff()
    contiguous = values.index.to_series().diff().eq(_EVENT_BIN).to_numpy()
    delta.loc[~contiguous] = np.nan
    return delta


def _book_metrics(aligned: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    if len(aligned) < _MIN_COMMON_BINS:
        return out
    imbalance_bond = aligned["imbalance_bond"].to_numpy(dtype="float64")
    imbalance_stock = aligned["imbalance_stock"].to_numpy(dtype="float64")
    spread_bond = aligned["relative_spread_bond"].to_numpy(dtype="float64")
    spread_stock = aligned["relative_spread_stock"].to_numpy(dtype="float64")
    touch_bond = aligned["touch_share_bond"].to_numpy(dtype="float64")
    touch_stock = aligned["touch_share_stock"].to_numpy(dtype="float64")
    primitive_valid = (
        np.isfinite(imbalance_bond)
        & np.isfinite(imbalance_stock)
        & np.isfinite(spread_bond)
        & np.isfinite(spread_stock)
        & np.isfinite(touch_bond)
        & np.isfinite(touch_stock)
        & (spread_bond > _EPS)
        & (spread_stock > _EPS)
    )
    if int(primitive_valid.sum()) < _MIN_COMMON_BINS:
        return out
    idx = np.flatnonzero(primitive_valid)
    imbalance_bond = imbalance_bond[idx]
    imbalance_stock = imbalance_stock[idx]
    spread_bond = spread_bond[idx]
    spread_stock = spread_stock[idx]
    touch_bond = touch_bond[idx]
    touch_stock = touch_stock[idx]
    log_spread_gap = np.log(spread_bond) - np.log(spread_stock)
    out["cabg_l1_imbalance_directional_agreement"] = float(
        np.mean(np.sign(imbalance_bond) * np.sign(imbalance_stock))
    )
    out["cabg_touch_depth_share_gap_mean"] = float(np.mean(touch_bond - touch_stock))
    out["cabg_relative_spread_gap_iqr"] = float(np.quantile(log_spread_gap, 0.75) - np.quantile(log_spread_gap, 0.25))

    joint_distance = _hellinger(aligned, kind="joint")
    bid_distance = _hellinger(aligned, kind="bid")
    ask_distance = _hellinger(aligned, kind="ask")
    geometry_valid = np.isfinite(joint_distance) & np.isfinite(bid_distance) & np.isfinite(ask_distance)
    if int(geometry_valid.sum()) >= _MIN_COMMON_BINS:
        out["cabg_joint_depth_curve_hellinger_mean"] = float(np.mean(joint_distance[geometry_valid]))
        out["cabg_bid_ask_curve_divergence_asymmetry"] = float(
            np.mean(bid_distance[geometry_valid] - ask_distance[geometry_valid])
        )
        phases = aligned["phase_bond"].to_numpy()
        early = geometry_valid & (phases == "early")
        late = geometry_valid & (phases == "late")
        if int(early.sum()) >= _MIN_PHASE_BINS and int(late.sum()) >= _MIN_PHASE_BINS:
            out["cabg_joint_curve_divergence_phase_shift"] = float(
                np.mean(joint_distance[late]) - np.mean(joint_distance[early])
            )

    out["cabg_l1_imbalance_time_correlation"] = _safe_corr(imbalance_bond, imbalance_stock)
    out["cabg_relative_spread_time_correlation"] = _safe_corr(np.log(spread_bond), np.log(spread_stock))
    migration_bond = _adjacent_delta(aligned["touch_share_bond"])
    migration_stock = _adjacent_delta(aligned["touch_share_stock"])
    out["cabg_touch_depth_migration_time_correlation"] = _safe_corr(
        migration_bond.to_numpy(dtype="float64"), migration_stock.to_numpy(dtype="float64")
    )
    return {key: (float(value) if np.isfinite(value) else float("nan")) for key, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    if score_date is None or output_index.empty:
        built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
    else:
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date)
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date)
        mapping = _context_mapping(ctx, score_date=score_date)
        bond_paths = {
            canonical: _asset_book_path(group)
            for code, group in bond_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        stock_paths = {
            canonical: _asset_book_path(group)
            for code, group in stock_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            canonical_bond = _canonical_market_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            bond_path = bond_paths.get(canonical_bond)
            stock_path = stock_paths.get(mapping.get(canonical_bond, ""))
            if bond_path is not None and stock_path is not None and not bond_path.empty and not stock_path.empty:
                aligned = bond_path.join(stock_path, how="inner", lsuffix="_bond", rsuffix="_stock").sort_index()
                row.update(_book_metrics(aligned))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].reindex(output_index)
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningCrossAssetBookGeometryV1(Factor):
    """Research-only mapped bond/stock liquidity-and-geometry catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = True

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
