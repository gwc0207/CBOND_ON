"""Research-only strict-PIT cross-asset event-sequence topology factors.

The module measures topology of observed five-minute transaction-count event
bins, not returns, labels, PnL, scores, masks, or files.  It deliberately
does not reuse activity-calendar moments or a same-time correlation.  Each
bond is paired only with the stock mapping certified by daily_price and
daily_base on the latest completed session strictly before T.

Physical panels are restricted to continuous T-day observations through
14:29:00.  A counter reset, missing/invalid data, uncertified mapping, or an
insufficient topology sample remains NaN.  No synthetic time-bin filling is
performed: adjacency is recognized only between actually observed consecutive
five-minute bins.
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


KERNEL_NAME = "factor_mining_cross_asset_event_sequence_topology_v1"
CATALOG_VERSION = "20260803_cross_asset_event_sequence_topology_v1"
_EPS = 1e-12
_CUTOFF = dt_time(14, 29)
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_COMMON_BINS = 8
_MIN_LEADER_STATES = 6
_MIN_ADJACENT_PAIRS = 4
_MIN_LEADER_TRIPLES = 4
_MIN_ACTIVE_BINS = 6
_MIN_BURST_ORIGINS = 3
_MIN_JOINT_BURST_ORIGINS = 2


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable family-first research candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family=family, signal=signal, kernel=KERNEL_NAME, hypothesis=hypothesis) for signal in signals)


_RELATIVE_ORDER_SIGNALS = (
    "xca_relative_activity_leader_switch_rate",
    "xca_relative_activity_leader_entropy",
    "xca_relative_activity_leader_lag2_agreement",
)
_BURST_TOPOLOGY_SIGNALS = (
    "xca_event_bin_entropy_gap",
    "xca_burst_cluster_persistence_gap",
    "xca_joint_burst_cluster_continuity",
)

_CATALOG = (
    _entries(
        "cross_asset_relative_event_order",
        _RELATIVE_ORDER_SIGNALS,
        "The ordered sequence of bond-versus-stock normalized activity leaders captures turn-taking and two-step topology without a correlation or a clock moment.",
    )
    + _entries(
        "cross_asset_burst_cluster_topology",
        _BURST_TOPOLOGY_SIGNALS,
        "Within-asset bin concentration and high-activity cluster persistence, plus continuity of joint burst clusters, capture event topology rather than calendar moments.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# B_t and S_t are observed five-minute positive-or-zero delta-num-trades
# masses. Leader_t is sign(B_t/sum(B)-S_t/sum(S)); all gaps are bond-stock.
FORMULAS: dict[str, str] = {
    "xca_relative_activity_leader_switch_rate": "mean(1[L_t != L_{t-1}]) over adjacent non-tie leader states",
    "xca_relative_activity_leader_entropy": "H(P[L=+1], P[L=-1]) / log(2) over non-tie observed leader states",
    "xca_relative_activity_leader_lag2_agreement": "mean(L_t * L_{t-2}) over three consecutive non-tie leader states",
    "xca_event_bin_entropy_gap": "H(B_t/sum(B))-H(S_t/sum(S)), normalized on the common observed bin count",
    "xca_burst_cluster_persistence_gap": "P(Burst_{t+1}|Burst_t)_bond-P(Burst_{t+1}|Burst_t)_stock, Burst=within-asset q75 positive bin",
    "xca_joint_burst_cluster_continuity": "P(Burst_b,t+1 and Burst_s,t+1 | Burst_b,t and Burst_s,t) on adjacent observed bins",
}

_REQUIRED_PANEL_COLUMNS = ("trade_time", "num_trades")
_EXCHANGE_ALIASES = {"XSHG": "SH", "SHSE": "SH", "XSHE": "SZ", "SZSE": "SZ", "BSE": "BJ", "BJSE": "BJ"}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})


def factor_mining_cross_asset_event_sequence_topology_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic scratch catalogue loaders."""

    return factor_mining_cross_asset_event_sequence_topology_catalog()


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
    """Normalize only codes with an explicit valid market suffix."""

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
    """Normalize only the daily-base stock mapping; never infer panel matches."""

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
    """Panel codes must already be exchange-qualified."""

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
    """Keep only physical T-day continuous-session observations through 14:29."""

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
    required = ("trade_date", "code", "exchange_code", *fields)
    if raw is None or any(column not in raw.columns for column in required):
        return pd.DataFrame(columns=["trade_date", "code", *fields])
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
    """Return only exact-anchor, unambiguous prior-daily stock mappings."""

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


def _empty_event_bins() -> pd.DataFrame:
    return pd.DataFrame(
        {"activity": pd.Series(dtype="float64")},
        index=pd.DatetimeIndex([], name="event_time"),
    )


def _asset_event_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Create observed five-minute non-negative num-trades increment bins."""

    if frame.empty:
        return _empty_event_bins()
    data = frame.loc[:, ["trade_time", "num_trades"]].copy()
    timestamps = pd.to_datetime(data["trade_time"], errors="coerce")
    count = pd.to_numeric(data["num_trades"], errors="coerce")
    if timestamps.isna().any() or not np.isfinite(count.to_numpy(dtype="float64")).all() or (count < 0.0).any():
        return _empty_event_bins()
    increment = count.diff()
    observed = increment.iloc[1:]
    if (observed < -_EPS).any():
        return _empty_event_bins()
    valid = timestamps.notna() & np.isfinite(increment) & (increment >= 0.0)
    if not bool(valid.any()):
        return _empty_event_bins()
    grouped = (
        pd.DataFrame(
            {
                "event_time": timestamps.loc[valid].dt.floor(_EVENT_BIN),
                "activity": increment.loc[valid].to_numpy(dtype="float64"),
            }
        )
        .groupby("event_time", sort=True)["activity"]
        .sum(min_count=1)
    )
    if len(grouped) < _MIN_COMMON_BINS:
        return _empty_event_bins()
    out = grouped.to_frame().replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    out.index.name = "event_time"
    return out


def _adjacent_mask(index: pd.DatetimeIndex) -> np.ndarray:
    if len(index) < 2:
        return np.zeros(0, dtype=bool)
    return np.asarray((index[1:] - index[:-1]) == _EVENT_BIN, dtype=bool)


def _leader_metrics(common: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _RELATIVE_ORDER_SIGNALS}
    bond = common["activity_bond"].to_numpy(dtype="float64")
    stock = common["activity_stock"].to_numpy(dtype="float64")
    if not (np.isfinite(bond).all() and np.isfinite(stock).all() and (bond >= 0.0).all() and (stock >= 0.0).all()):
        return out
    bond_total = float(bond.sum())
    stock_total = float(stock.sum())
    if bond_total <= _EPS or stock_total <= _EPS:
        return out
    difference = bond / bond_total - stock / stock_total
    leader = np.where(difference > _EPS, 1.0, np.where(difference < -_EPS, -1.0, 0.0))
    valid_state = leader != 0.0
    if int(valid_state.sum()) < _MIN_LEADER_STATES:
        return out
    plus_share = float(np.mean(leader[valid_state] > 0.0))
    minus_share = 1.0 - plus_share
    entropy_terms = [share * np.log(share) for share in (plus_share, minus_share) if share > _EPS]
    out["xca_relative_activity_leader_entropy"] = float(-sum(entropy_terms) / np.log(2.0))

    adjacent = _adjacent_mask(pd.DatetimeIndex(common.index))
    pair = adjacent & valid_state[:-1] & valid_state[1:]
    if int(pair.sum()) >= _MIN_ADJACENT_PAIRS:
        out["xca_relative_activity_leader_switch_rate"] = float(np.mean(leader[:-1][pair] != leader[1:][pair]))

    triple = adjacent[:-1] & adjacent[1:] & valid_state[:-2] & valid_state[1:-1] & valid_state[2:]
    if int(triple.sum()) >= _MIN_LEADER_TRIPLES:
        out["xca_relative_activity_leader_lag2_agreement"] = float(np.mean(leader[:-2][triple] * leader[2:][triple]))
    return out


def _normalized_entropy(activity: np.ndarray) -> float:
    if len(activity) < 2 or not (np.isfinite(activity).all() and (activity >= 0.0).all()):
        return float("nan")
    total = float(activity.sum())
    if total <= _EPS:
        return float("nan")
    mass = activity / total
    positive = mass > _EPS
    terms = mass[positive] * np.log(mass[positive])
    return float(-terms.sum() / np.log(float(len(activity))))


def _burst_state(activity: np.ndarray) -> np.ndarray | None:
    if len(activity) < _MIN_COMMON_BINS or not (np.isfinite(activity).all() and (activity >= 0.0).all()):
        return None
    active = activity > _EPS
    if int(active.sum()) < _MIN_ACTIVE_BINS:
        return None
    positive = activity[active]
    if float(np.max(positive) - np.min(positive)) <= _EPS:
        return None
    threshold = float(np.quantile(positive, 0.75))
    return active & (activity >= threshold)


def _cluster_continuity(state: np.ndarray, index: pd.DatetimeIndex, *, min_origins: int) -> float:
    adjacent = _adjacent_mask(index)
    origin = adjacent & state[:-1]
    if int(origin.sum()) < min_origins:
        return float("nan")
    return float(np.mean(state[1:][origin]))


def _burst_topology_metrics(common: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _BURST_TOPOLOGY_SIGNALS}
    bond = common["activity_bond"].to_numpy(dtype="float64")
    stock = common["activity_stock"].to_numpy(dtype="float64")
    bond_entropy = _normalized_entropy(bond)
    stock_entropy = _normalized_entropy(stock)
    if np.isfinite(bond_entropy) and np.isfinite(stock_entropy):
        out["xca_event_bin_entropy_gap"] = float(bond_entropy - stock_entropy)

    bond_burst = _burst_state(bond)
    stock_burst = _burst_state(stock)
    if bond_burst is None or stock_burst is None:
        return out
    index = pd.DatetimeIndex(common.index)
    bond_persistence = _cluster_continuity(bond_burst, index, min_origins=_MIN_BURST_ORIGINS)
    stock_persistence = _cluster_continuity(stock_burst, index, min_origins=_MIN_BURST_ORIGINS)
    if np.isfinite(bond_persistence) and np.isfinite(stock_persistence):
        out["xca_burst_cluster_persistence_gap"] = float(bond_persistence - stock_persistence)

    joint_burst = bond_burst & stock_burst
    out["xca_joint_burst_cluster_continuity"] = _cluster_continuity(
        joint_burst,
        index,
        min_origins=_MIN_JOINT_BURST_ORIGINS,
    )
    return out


def _joint_metrics(bond_frame: pd.DataFrame, stock_frame: pd.DataFrame) -> dict[str, float]:
    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    bond = _asset_event_bins(bond_frame).add_suffix("_bond")
    stock = _asset_event_bins(stock_frame).add_suffix("_stock")
    common = bond.join(stock, how="inner").sort_index()
    if len(common) < _MIN_COMMON_BINS:
        return out
    out.update(_leader_metrics(common))
    out.update(_burst_topology_metrics(common))
    return {signal: float(value) if np.isfinite(value) else float("nan") for signal, value in out.items()}


def _feature_frame(ctx: FactorComputeContext) -> pd.DataFrame:
    cache_key = f"{KERNEL_NAME}:{CATALOG_VERSION}:features"
    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached

    score_date = _score_date_from_panel(ctx.panel)
    output_index = _output_index(ctx, score_date)
    panel_has_required = all(column in ctx.panel.columns for column in _REQUIRED_PANEL_COLUMNS)
    stock_has_required = bool(
        ctx.stock_panel is not None
        and all(column in ctx.stock_panel.columns for column in _REQUIRED_PANEL_COLUMNS)
    )
    if score_date is None or output_index.empty or not panel_has_required or not stock_has_required:
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
class FactorMiningCrossAssetEventSequenceTopologyV1(Factor):
    """Research-only strict-PIT cross-asset event-topology catalogue."""

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
