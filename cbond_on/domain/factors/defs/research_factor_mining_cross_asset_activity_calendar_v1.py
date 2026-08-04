"""Research-only strict-PIT cross-asset activity-calendar factors.

This module intentionally measures *when* completed transactions occur during
the T-day trading session.  It does not use prices, returns, labels, scores,
masks, PnL, files, databases, or the context bond-stock map.

The mapping is certified strictly from the latest completed daily-price
session before the score date and a daily-base stock-code row on that exact
same session.  Both bond and stock panels are physically restricted to
continuous T-day observations through 14:29:00.  Invalid/missing inputs,
counter resets, insufficient active bins, and uncertified mappings produce
NaN rather than an inferred value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_cross_asset_activity_calendar_v1"
CATALOG_VERSION = "20260803_cross_asset_activity_calendar_v1"
_EPS = 1e-12
_CUTOFF = dt_time(14, 29)
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_ACTIVE_BINS = 6
_SESSION_MINUTES = 210.0
_AFTERNOON_START_MINUTE = 120.0
_LATE_TAIL_START_MINUTE = 180.0


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable family-first research candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis) for signal in signals)


_GEOMETRY_SIGNALS = (
    "xca_activity_calendar_center_gap",
    "xca_activity_calendar_dispersion_gap",
    "xca_activity_calendar_late_tail_share_gap",
)
_BURST_ASYMMETRY_SIGNALS = (
    "xca_activity_calendar_burst_centroid_gap",
    "xca_activity_calendar_skewness_gap",
    "xca_activity_calendar_afternoon_mass_gap",
)

_CATALOG = (
    _entries(
        "cross_asset_activity_calendar_geometry",
        _GEOMETRY_SIGNALS,
        "The bond-stock gap in the activity-time center, dispersion, and final-half-hour mass captures calendar shape without a price input.",
    )
    + _entries(
        "cross_asset_activity_calendar_burst_asymmetry",
        _BURST_ASYMMETRY_SIGNALS,
        "The gap in high-activity timing, temporal skewness, and afternoon mass captures burst placement and distribution asymmetry without cross-asset correlation.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# Let u be continuous-session minute / 210 and w be a positive five-minute
# num_trades increment.  Every reported gap is bond minus mapped-stock.
FORMULAS: dict[str, str] = {
    "xca_activity_calendar_center_gap": "E_b[u; w]-E_s[u; w]",
    "xca_activity_calendar_dispersion_gap": "sd_b[u; w]-sd_s[u; w]",
    "xca_activity_calendar_late_tail_share_gap": "P_b(u>=180min; w)-P_s(u>=180min; w)",
    "xca_activity_calendar_burst_centroid_gap": "E_b[u; w >= q75(w)]-E_s[u; w >= q75(w)]",
    "xca_activity_calendar_skewness_gap": "skew_b[u; w]-skew_s[u; w]",
    "xca_activity_calendar_afternoon_mass_gap": "P_b(u>=120min; w)-P_s(u>=120min; w)",
}

_REQUIRED_PANEL_COLUMNS = ("trade_time", "num_trades")
_EXCHANGE_ALIASES = {"XSHG": "SH", "SHSE": "SH", "XSHE": "SZ", "SZSE": "SZ", "BSE": "BJ", "BJSE": "BJ"}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def factor_mining_cross_asset_activity_calendar_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return factor_mining_cross_asset_activity_calendar_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, owner: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} {owner} missing columns: {missing}")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize only codes whose market suffix is explicit and valid."""

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


def _canonical_stock_code(value: object) -> str:
    """Normalize a daily-base stock mapping without looking at panel content."""

    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if "." in text:
        bare, suffix = text.rsplit(".", 1)
        suffix = _EXCHANGE_ALIASES.get(suffix, suffix)
        return f"{bare}.{suffix}" if bare and suffix in _MARKET_EXCHANGES else ""
    digits = text.zfill(6) if text.isdigit() else text
    if len(digits) != 6 or not digits.isdigit():
        return ""
    if digits[0] == "6":
        return f"{digits}.SH"
    if digits[0] in {"0", "3"}:
        return f"{digits}.SZ"
    if digits[0] in {"4", "8"}:
        return f"{digits}.BJ"
    return ""


def _panel_code(value: object) -> str:
    """Panel codes must already carry an exchange suffix."""

    return _canonical_market_code(pd.Series([value])).iloc[0]


def _score_date_from_panel(panel: pd.DataFrame) -> pd.Timestamp | None:
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


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp, owner: str) -> pd.DataFrame:
    """Keep physical T-day observations in continuous session through 14:29."""

    checked = ensure_trade_time(panel)
    _require_columns(checked, _REQUIRED_PANEL_COLUMNS, owner=owner)
    frame = checked.reset_index().copy(deep=False)
    indexed = pd.to_datetime(frame["dt"], errors="coerce")
    timestamps = pd.to_datetime(frame["trade_time"], errors="coerce")
    clocks = timestamps.dt.time
    continuous = timestamps.notna() & clocks.map(lambda clock: _continuous_session(clock) if pd.notna(clock) else False)
    keep = (
        indexed.notna()
        & (indexed.dt.normalize() == score_date)
        & timestamps.notna()
        & (timestamps.dt.normalize() == score_date)
        & continuous
        & (clocks <= _CUTOFF)
    )
    out = frame.loc[keep].copy()
    out["trade_time"] = timestamps.loc[keep]
    return out.sort_values(["dt", "code", "trade_time", "seq"], kind="mergesort")


def _strict_prior_daily(
    ctx: FactorComputeContext,
    *,
    source: str,
    fields: tuple[str, ...],
    score_date: pd.Timestamp,
) -> pd.DataFrame:
    raw = ctx.daily_data.get(source)
    if raw is None:
        raise KeyError(f"{KERNEL_NAME} missing daily source: {source}")
    _require_columns(raw, ("trade_date", "code", "exchange_code", *fields), owner=source)
    frame = raw.loc[:, ["trade_date", "code", "exchange_code", *fields]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = _canonical_market_code(frame["code"], frame["exchange_code"])
    return frame.loc[
        frame["trade_date"].notna()
        & (frame["trade_date"] < score_date)
        & frame["code"].notna()
        & (frame["code"] != "")
    ].sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _tminus1_stock_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Return only unambiguous mappings certified on the price anchor date."""

    price = _strict_prior_daily(
        ctx,
        source="market_cbond.daily_price",
        fields=("close_price",),
        score_date=score_date,
    )
    base = _strict_prior_daily(
        ctx,
        source="market_cbond.daily_base",
        fields=("stock_code",),
        score_date=score_date,
    )
    if price.empty or base.empty:
        return {}
    anchor = pd.Timestamp(price["trade_date"].max()).normalize()
    anchor_price = price.loc[price["trade_date"] == anchor, ["code"]].copy()
    anchor_price = anchor_price.loc[~anchor_price.duplicated("code", keep=False)]
    if anchor_price.empty:
        return {}
    anchor_codes = set(anchor_price["code"].astype(str))
    current = base.loc[
        (base["trade_date"] == anchor) & base["code"].astype(str).isin(anchor_codes),
        ["code", "stock_code"],
    ].copy()
    current["stock_code"] = current["stock_code"].map(_canonical_stock_code)
    current = current.loc[current["stock_code"] != ""]
    current = current.loc[~current.duplicated("code", keep=False)]
    return {str(row.code): str(row.stock_code) for row in current.itertuples(index=False)}


def _empty_activity_bins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_minute": pd.Series(dtype="float64"),
            "activity": pd.Series(dtype="float64"),
        },
        index=pd.DatetimeIndex([], name="event_time"),
    )


def _session_minute(event_times: pd.DatetimeIndex) -> np.ndarray:
    """Compress the lunch break before mapping timestamps onto session minutes."""

    values: list[float] = []
    for stamp in event_times:
        clock = stamp.time()
        if _MORNING_START <= clock <= _MORNING_END:
            values.append((stamp.hour * 60.0 + stamp.minute) - (_MORNING_START.hour * 60.0 + _MORNING_START.minute))
        elif _AFTERNOON_START <= clock <= _CUTOFF:
            values.append(
                _AFTERNOON_START_MINUTE
                + (stamp.hour * 60.0 + stamp.minute)
                - (_AFTERNOON_START.hour * 60.0 + _AFTERNOON_START.minute)
            )
        else:
            values.append(float("nan"))
    return np.asarray(values, dtype="float64")


def _asset_activity_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Return positive observed five-minute num-trades increments only."""

    if frame.empty:
        return _empty_activity_bins()
    data = frame.loc[:, ["trade_time", "num_trades"]].copy()
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    data["num_trades"] = pd.to_numeric(data["num_trades"], errors="coerce")
    increment = data["num_trades"].diff()
    observed = increment.iloc[1:]
    if (observed < -_EPS).any():
        return _empty_activity_bins()
    valid = timestamps.notna() & np.isfinite(increment) & (increment >= 0.0)
    if not bool(valid.any()):
        return _empty_activity_bins()
    base = pd.DataFrame(
        {
            "event_time": timestamps.loc[valid].dt.floor(_EVENT_BIN),
            "activity": increment.loc[valid].to_numpy(dtype="float64"),
        }
    )
    grouped = base.groupby("event_time", sort=True)["activity"].sum(min_count=1)
    grouped = grouped.loc[np.isfinite(grouped) & (grouped > _EPS)]
    if len(grouped) < _MIN_ACTIVE_BINS:
        return _empty_activity_bins()
    minutes = _session_minute(pd.DatetimeIndex(grouped.index))
    valid_minutes = np.isfinite(minutes) & (minutes >= 0.0) & (minutes <= _SESSION_MINUTES)
    if int(valid_minutes.sum()) < _MIN_ACTIVE_BINS:
        return _empty_activity_bins()
    out = pd.DataFrame(
        {
            "session_minute": minutes[valid_minutes],
            "activity": grouped.to_numpy(dtype="float64")[valid_minutes],
        },
        index=pd.DatetimeIndex(grouped.index[valid_minutes], name="event_time"),
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def _activity_shape(frame: pd.DataFrame) -> dict[str, float] | None:
    if len(frame) < _MIN_ACTIVE_BINS:
        return None
    minutes = frame["session_minute"].to_numpy(dtype="float64")
    weights = frame["activity"].to_numpy(dtype="float64")
    valid = np.isfinite(minutes) & np.isfinite(weights) & (weights > _EPS)
    if int(valid.sum()) < _MIN_ACTIVE_BINS:
        return None
    minutes = minutes[valid]
    weights = weights[valid]
    total = float(weights.sum())
    if not np.isfinite(total) or total <= _EPS:
        return None
    normalized = minutes / _SESSION_MINUTES
    center = float(np.dot(weights, normalized) / total)
    variance = float(np.dot(weights, (normalized - center) ** 2) / total)
    if not np.isfinite(variance) or variance <= _EPS:
        return None
    dispersion = float(np.sqrt(variance))
    skewness = float(np.dot(weights, (normalized - center) ** 3) / total / (dispersion**3))
    if not np.isfinite(skewness):
        return None
    late_tail_share = float(weights[minutes >= _LATE_TAIL_START_MINUTE].sum() / total)
    afternoon_mass = float(weights[minutes >= _AFTERNOON_START_MINUTE].sum() / total)
    threshold = float(np.quantile(weights, 0.75))
    burst = weights >= threshold
    if not bool(burst.any()) or not bool(np.any(weights > threshold + _EPS)):
        return None
    burst_weight = weights[burst]
    burst_total = float(burst_weight.sum())
    if burst_total <= _EPS:
        return None
    burst_centroid = float(np.dot(burst_weight, normalized[burst]) / burst_total)
    values = {
        "center": center,
        "dispersion": dispersion,
        "late_tail_share": late_tail_share,
        "burst_centroid": burst_centroid,
        "skewness": skewness,
        "afternoon_mass": afternoon_mass,
    }
    return values if all(np.isfinite(value) for value in values.values()) else None


def _joint_metrics(bond_frame: pd.DataFrame, stock_frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    bond = _activity_shape(_asset_activity_bins(bond_frame))
    stock = _activity_shape(_asset_activity_bins(stock_frame))
    if bond is None or stock is None:
        return out
    out["xca_activity_calendar_center_gap"] = bond["center"] - stock["center"]
    out["xca_activity_calendar_dispersion_gap"] = bond["dispersion"] - stock["dispersion"]
    out["xca_activity_calendar_late_tail_share_gap"] = bond["late_tail_share"] - stock["late_tail_share"]
    out["xca_activity_calendar_burst_centroid_gap"] = bond["burst_centroid"] - stock["burst_centroid"]
    out["xca_activity_calendar_skewness_gap"] = bond["skewness"] - stock["skewness"]
    out["xca_activity_calendar_afternoon_mass_gap"] = bond["afternoon_mass"] - stock["afternoon_mass"]
    return {name: float(value) if np.isfinite(value) else float("nan") for name, value in out.items()}


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
        if ctx.stock_panel is None or ctx.stock_panel.empty:
            raise RuntimeError(f"{KERNEL_NAME} requires a non-empty stock_panel")
        bond_frame = _strict_physical_frame(ctx.panel, score_date=score_date, owner="bond_panel")
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date, owner="stock_panel")
        mapping = _tminus1_stock_mapping(ctx, score_date=score_date)
        bond_groups = {
            _panel_code(code): group
            for code, group in bond_frame.groupby("code", sort=False)
            if _panel_code(code)
        }
        stock_groups = {
            _panel_code(code): group
            for code, group in stock_frame.groupby("code", sort=False)
            if _panel_code(code)
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            code = _panel_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            bond_group = bond_groups.get(code)
            stock_group = stock_groups.get(mapping.get(code, ""))
            if bond_group is not None and stock_group is not None:
                row.update(_joint_metrics(bond_group, stock_group))
            rows.append(row)
        built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
        built = built.replace([np.inf, -np.inf], np.nan)

    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningCrossAssetActivityCalendarV1(Factor):
    """Research-only cross-asset intraday activity-calendar catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    requires_bond_stock_map = False

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        del params
        return [
            DailyFactorRequirement("market_cbond.daily_price", ("exchange_code", "close_price"), 10),
            DailyFactorRequirement("market_cbond.daily_base", ("exchange_code", "stock_code"), 10),
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
