"""Research-only strict-PIT cross-asset execution-clock catalogue.

This import-only module is deliberately excluded from ``defs.__init__`` and
from every production factor/model/live configuration.  It has two families:

* ``cross_asset_event_clock_transmission`` compares *completed intraday
  trading-event clocks* between a bond and its mapped stock; and
* ``cross_asset_signed_execution_transmission`` compares quote-anchored,
  volume-weighted execution direction without using a price return.

Both current-day panels are physically filtered to continuous T-day
observations through 14:29:00.  The stock mapping is certified only by a
``daily_base.stock_code`` row on the latest common completed
``daily_price`` session before T.  ``ctx.bond_stock_map`` is intentionally
ignored because it may contain a score-day mapping.  No factor reads files,
databases, masks, labels, scores, or strategy results.  Missing/invalid inputs
remain ``NaN``; they are never replaced with zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_cross_asset_event_clock_v1"
CATALOG_VERSION = "20260803_cross_asset_event_clock_v1"
_EPS = 1e-12
_CUTOFF = dt_time(14, 29)
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_COMMON_BINS = 6


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only candidate signal."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis) for signal in signals)


_EVENT_CLOCK_SIGNALS = (
    "xca_event_burst_coactivity",
    "xca_stock_leads_bond_event_clock",
    "xca_lull_overlap_excess",
)
_SIGNED_EXECUTION_SIGNALS = (
    "xca_signed_execution_agreement",
    "xca_stock_leads_bond_signed_execution",
    "xca_signed_execution_intensity_gap",
)

_CATALOG = (
    _entries(
        "cross_asset_event_clock_transmission",
        _EVENT_CLOCK_SIGNALS,
        "Common burst timing, directed event-clock lead, and excess simultaneous lulls describe execution transmission without a price-return input.",
    )
    + _entries(
        "cross_asset_signed_execution_transmission",
        _SIGNED_EXECUTION_SIGNALS,
        "Quote-anchored execution direction compares signed volume agreement, lead direction, and imbalance intensity across the T-1-mapped pair.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)
_FAMILY_SIGNALS = {
    "cross_asset_event_clock_transmission": _EVENT_CLOCK_SIGNALS,
    "cross_asset_signed_execution_transmission": _SIGNED_EXECUTION_SIGNALS,
}

# These formula strings are evidence for later research review.  ``n`` is the
# positive cumulative ``num_trades`` increment in one observed five-minute
# event bin.  ``q`` is quote-anchored signed volume divided by gross volume.
FORMULAS: dict[str, str] = {
    "xca_event_burst_coactivity": "corr(log1p(n_b), log1p(n_s)) over common observed event bins",
    "xca_stock_leads_bond_event_clock": "corr(log1p(n_s,t),log1p(n_b,t+1))-corr(log1p(n_b,t),log1p(n_s,t+1))",
    "xca_lull_overlap_excess": "mean(1[n_b=0 and n_s=0])-mean(1[n_b=0])*mean(1[n_s=0])",
    "xca_signed_execution_agreement": "corr(q_b, q_s) over common quote-anchored event bins",
    "xca_stock_leads_bond_signed_execution": "corr(q_s,t,q_b,t+1)-corr(q_b,t,q_s,t+1)",
    "xca_signed_execution_intensity_gap": "mean(abs(q_b))-mean(abs(q_s))",
}

_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "last",
    "ask_price1",
    "bid_price1",
    "volume",
    "num_trades",
)
_EXCHANGE_ALIASES = {"XSHG": "SH", "SHSE": "SH", "XSHE": "SZ", "SZSE": "SZ", "BSE": "BJ", "BJSE": "BJ"}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def factor_mining_cross_asset_event_clock_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return factor_mining_cross_asset_event_clock_catalog()


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


def _safe_corr(left: np.ndarray, right: np.ndarray, *, min_count: int = _MIN_COMMON_BINS) -> float:
    if len(left) != len(right):
        return float("nan")
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < min_count:
        return float("nan")
    lhs = left[valid]
    rhs = right[valid]
    if float(np.std(lhs)) <= _EPS or float(np.std(rhs)) <= _EPS:
        return float("nan")
    value = float(np.corrcoef(lhs, rhs)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _canonical_market_code(values: pd.Series, exchanges: pd.Series | None = None) -> pd.Series:
    """Normalize a code only when a valid exchange suffix is known."""

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
    """Normalize a daily-base mapping without inferring it from the panel."""

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
    """Panel codes must already be exchange-qualified or they are rejected."""

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
    labels = pd.to_datetime(keys["dt"], errors="coerce").dt.normalize()
    keys = keys.loc[labels == score_date].drop_duplicates().sort_values(["dt", "code"], kind="mergesort")
    return pd.MultiIndex.from_frame(keys, names=["dt", "code"])


def _continuous_session(clock: dt_time) -> bool:
    return (_MORNING_START <= clock <= _MORNING_END) or (_AFTERNOON_START <= clock <= _CUTOFF)


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp, owner: str) -> pd.DataFrame:
    """Keep only physical T-day continuous-session rows through 14:29:00."""

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
    """Read only explicitly declared daily rows before the score date."""

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
        examples = frame.loc[frame.duplicated(["trade_date", "code"], keep=False), ["trade_date", "code"]].head(3)
        raise ValueError(f"{KERNEL_NAME} {source} has duplicate strict-prior rows: {examples.to_dict('records')}")
    return frame.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)


def _tminus1_stock_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Return mappings certified by the latest completed daily-price session."""

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
    anchor_codes = set(price.loc[price["trade_date"] == anchor, "code"].astype(str))
    current = base.loc[
        (base["trade_date"] == anchor) & base["code"].astype(str).isin(anchor_codes),
        ["code", "stock_code"],
    ].copy()
    if current.empty:
        return {}
    current["stock_code"] = current["stock_code"].map(_canonical_stock_code)
    current = current.loc[current["stock_code"] != ""].drop_duplicates(subset=["code"], keep="last")
    return {str(row.code): str(row.stock_code) for row in current.itertuples(index=False)}


def _empty_event_bins() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["event_log1p", "signed_fraction", "signed_abs_fraction"],
        index=pd.DatetimeIndex([], name="event_time"),
        dtype="float64",
    )


def _asset_event_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct observed event bins from strictly monotone cumulative counters.

    A negative cumulative ``volume`` or ``num_trades`` delta invalidates the
    entire asset/day for this module.  That deliberate fail-closed contract
    avoids pretending vendor counter resets are neutral execution observations.
    """

    if frame.empty:
        return _empty_event_bins()
    columns = ["trade_time", "last", "bid_price1", "ask_price1", "volume", "num_trades"]
    data = frame.loc[:, columns].copy()
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    for column in columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    volume_delta = data["volume"].diff()
    trade_delta = data["num_trades"].diff()
    observed_delta = pd.DataFrame({"volume": volume_delta.iloc[1:], "num_trades": trade_delta.iloc[1:]})
    if ((observed_delta["volume"] < -_EPS) | (observed_delta["num_trades"] < -_EPS)).any():
        return _empty_event_bins()

    valid_counter = (
        timestamps.notna()
        & np.isfinite(volume_delta)
        & np.isfinite(trade_delta)
        & (volume_delta >= 0.0)
        & (trade_delta >= 0.0)
    )
    if not bool(valid_counter.any()):
        return _empty_event_bins()

    base = pd.DataFrame(
        {
            "event_time": timestamps.loc[valid_counter].dt.floor(_EVENT_BIN),
            "trade_increment": trade_delta.loc[valid_counter].to_numpy(dtype="float64"),
            "volume_increment": volume_delta.loc[valid_counter].to_numpy(dtype="float64"),
        }
    )
    if base.empty:
        return _empty_event_bins()
    event_increment = base.groupby("event_time", sort=True)["trade_increment"].sum(min_count=1)
    out = pd.DataFrame({"event_log1p": np.log1p(event_increment)})
    out.index.name = "event_time"
    out["signed_fraction"] = np.nan
    out["signed_abs_fraction"] = np.nan

    spread = data["ask_price1"] - data["bid_price1"]
    midpoint = (data["ask_price1"] + data["bid_price1"]) / 2.0
    quote_valid = (
        valid_counter
        & (trade_delta > 0.0)
        & (volume_delta > _EPS)
        & np.isfinite(data["last"])
        & np.isfinite(data["bid_price1"])
        & np.isfinite(data["ask_price1"])
        & (spread > _EPS)
    )
    if bool(quote_valid.any()):
        location = ((data.loc[quote_valid, "last"] - midpoint.loc[quote_valid]) / (spread.loc[quote_valid] / 2.0)).clip(-1.0, 1.0)
        gross = volume_delta.loc[quote_valid].to_numpy(dtype="float64")
        signed = gross * location.to_numpy(dtype="float64")
        signed_rows = pd.DataFrame(
            {
                "event_time": timestamps.loc[quote_valid].dt.floor(_EVENT_BIN),
                "gross": gross,
                "signed": signed,
                "signed_abs": np.abs(signed),
            }
        )
        grouped = signed_rows.groupby("event_time", sort=True)[["gross", "signed", "signed_abs"]].sum(min_count=1)
        denominator = grouped["gross"].where(grouped["gross"] > _EPS)
        out.loc[grouped.index, "signed_fraction"] = grouped["signed"] / denominator
        out.loc[grouped.index, "signed_abs_fraction"] = grouped["signed_abs"] / denominator
    return out.replace([np.inf, -np.inf], np.nan).sort_index()


def _common_bins(bond_bins: pd.DataFrame, stock_bins: pd.DataFrame) -> pd.DataFrame:
    bond = bond_bins.add_suffix("_bond")
    stock = stock_bins.add_suffix("_stock")
    return bond.join(stock, how="inner").sort_index()


def _lead_difference(frame: pd.DataFrame, *, bond_column: str, stock_column: str) -> float:
    if len(frame) < _MIN_COMMON_BINS + 1:
        return float("nan")
    event_times = pd.DatetimeIndex(frame.index)
    contiguous = (event_times[1:] - event_times[:-1]) == _EVENT_BIN
    if int(contiguous.sum()) < _MIN_COMMON_BINS:
        return float("nan")
    bond = frame[bond_column].to_numpy(dtype="float64")
    stock = frame[stock_column].to_numpy(dtype="float64")
    stock_leads = _safe_corr(stock[:-1][contiguous], bond[1:][contiguous])
    bond_leads = _safe_corr(bond[:-1][contiguous], stock[1:][contiguous])
    if not (np.isfinite(stock_leads) and np.isfinite(bond_leads)):
        return float("nan")
    return float(stock_leads - bond_leads)


def _lull_overlap_excess(frame: pd.DataFrame) -> float:
    if len(frame) < _MIN_COMMON_BINS:
        return float("nan")
    bond = frame["event_log1p_bond"].to_numpy(dtype="float64")
    stock = frame["event_log1p_stock"].to_numpy(dtype="float64")
    valid = np.isfinite(bond) & np.isfinite(stock)
    if int(valid.sum()) < _MIN_COMMON_BINS:
        return float("nan")
    bond = bond[valid]
    stock = stock[valid]
    if not bool(np.any(bond > _EPS)) or not bool(np.any(stock > _EPS)):
        return float("nan")
    bond_lull = bond <= _EPS
    stock_lull = stock <= _EPS
    return float(np.mean(bond_lull & stock_lull) - np.mean(bond_lull) * np.mean(stock_lull))


def _joint_metrics(bond_frame: pd.DataFrame, stock_frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    common = _common_bins(_asset_event_bins(bond_frame), _asset_event_bins(stock_frame))
    if common.empty:
        return out

    event_bond = common["event_log1p_bond"].to_numpy(dtype="float64")
    event_stock = common["event_log1p_stock"].to_numpy(dtype="float64")
    out["xca_event_burst_coactivity"] = _safe_corr(event_bond, event_stock)
    out["xca_stock_leads_bond_event_clock"] = _lead_difference(
        common,
        bond_column="event_log1p_bond",
        stock_column="event_log1p_stock",
    )
    out["xca_lull_overlap_excess"] = _lull_overlap_excess(common)

    signed = common.dropna(subset=["signed_fraction_bond", "signed_fraction_stock"])
    if not signed.empty:
        signed_bond = signed["signed_fraction_bond"].to_numpy(dtype="float64")
        signed_stock = signed["signed_fraction_stock"].to_numpy(dtype="float64")
        out["xca_signed_execution_agreement"] = _safe_corr(signed_bond, signed_stock)
        out["xca_stock_leads_bond_signed_execution"] = _lead_difference(
            signed,
            bond_column="signed_fraction_bond",
            stock_column="signed_fraction_stock",
        )
    intensity = common.dropna(subset=["signed_abs_fraction_bond", "signed_abs_fraction_stock"])
    if len(intensity) >= _MIN_COMMON_BINS:
        bond_intensity = intensity["signed_abs_fraction_bond"].to_numpy(dtype="float64")
        stock_intensity = intensity["signed_abs_fraction_stock"].to_numpy(dtype="float64")
        out["xca_signed_execution_intensity_gap"] = float(np.mean(bond_intensity) - np.mean(stock_intensity))
    return out


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
        existing = ctx.cache.get(cache_key)
        if isinstance(existing, pd.DataFrame):
            return existing
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningCrossAssetEventClockV1(Factor):
    """Research-only T1430 cross-asset execution-clock catalogue."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = True
    # The only permissible mapping is the strict prior-daily certificate above.
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
