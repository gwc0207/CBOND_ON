"""Research-only T1430 mapped-stock shock-transmission factor catalogue.

This catalogue is deliberately independent of the existing daily bond-stock
tracking, valuation/premium, beta-residual, and event-*activity* catalogues.
It asks a narrower intraday question: after a mapped underlying stock makes a
large *five-minute price shock*, does the convertible bond respond in the same
direction, at the next observed interval, with a different relative amplitude,
or with a different follow-through shape?

Only the declared :class:`~cbond_on.domain.factors.base.FactorComputeContext`
inputs are consumed: ``ctx.panel``, ``ctx.stock_panel``, and
``ctx.bond_stock_map``.  The factor never reads daily data, labels, scores,
pool/mask membership, PnL, files, databases, or external state.  Both panels
are independently restricted to physical T-day continuous-auction snapshots
through 14:29:00; a rolling clean-direct panel's index label alone is never
accepted as timestamp evidence.  The supplied mapping is treated as the
pipeline's point-in-time context contract: explicit mapping rows dated after
the score day, malformed rows, and duplicate bond mappings are rejected rather
than inferred or carried forward.

Missing map/panel fields, invalid prices, incomplete five-minute paths,
non-varying stock returns, or too few shock observations produce ``NaN`` for
the affected candidate.  This module is import-only and research-only: it is
intentionally not added to ``defs.__init__``, a factor contract, any model
configuration, FactorStore, or live configuration.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index


KERNEL_NAME = "factor_mining_intraday_transmission_response_v1"
CATALOG_VERSION = "20260803_intraday_transmission_response_v1"

_EPS = 1e-12
_EVENT_BIN = pd.Timedelta(minutes=5)
_MIN_COMMON_RETURNS = 12
_MIN_SHOCKS = 4
_MIN_SIDE_SHOCKS = 2
_MIN_FOLLOW_THROUGH = 3
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_EXCHANGE_ALIASES = {
    "XSHG": "SH",
    "SHSE": "SH",
    "XSHE": "SZ",
    "SZSE": "SZ",
    "BSE": "BJ",
    "BJSE": "BJ",
}
_MARKET_EXCHANGES = frozenset({"SH", "SZ", "BJ"})
_REQUIRED_PANEL_COLUMNS = ("trade_time", "last")


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable research-only transmission candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_DIRECTIONAL_SIGNALS = (
    "itr_stock_shock_same_bin_directional_agreement",
    "itr_stock_shock_next_bin_directional_agreement",
    "itr_stock_shock_directional_latency_gap",
)
_AMPLITUDE_SIGNALS = (
    "itr_stock_shock_relative_amplitude_mean",
    "itr_stock_shock_relative_amplitude_dispersion",
    "itr_stock_shock_up_down_amplitude_asymmetry",
)
_KINETIC_SIGNALS = (
    "itr_stock_shock_bond_response_followthrough",
    "itr_stock_shock_bond_response_deferment_share",
    "itr_stock_shock_bond_vs_stock_followthrough_gap",
)

_CATALOG = (
    _entries(
        "intraday_stock_shock_directional_response",
        _DIRECTIONAL_SIGNALS,
        "Large mapped-stock five-minute shocks condition a nonparametric same-bin versus next-bin bond direction response, rather than estimating an all-interval beta or a full-day return gap.",
    )
    + _entries(
        "intraday_stock_shock_relative_amplitude",
        _AMPLITUDE_SIGNALS,
        "Mapped stock shocks condition the bond-versus-stock response magnitude after each asset is normalized by its own observed intraday median absolute return; this is neither daily tracking nor premium state.",
    )
    + _entries(
        "intraday_stock_shock_bond_kinetics",
        _KINETIC_SIGNALS,
        "For stock shock intervals only, bond response persistence and deferment are compared with the underlying stock's own next-bin path; activity-clock factors do not use price-response kinetics.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# r^B_t and r^S_t are log returns between consecutive observed five-minute
# endpoint bins. E={t: |r^S_t| >= q_0.75(|r^S|)} and t+1 must be exactly one
# later observed five-minute bin for the kinetic/delayed terms.
FORMULAS: dict[str, str] = {
    "itr_stock_shock_same_bin_directional_agreement": "mean(sign(rS_t)*sign(rB_t) | E)",
    "itr_stock_shock_next_bin_directional_agreement": "mean(sign(rS_t)*sign(rB_t+1) | E, contiguous t+1)",
    "itr_stock_shock_directional_latency_gap": "next_bin_directional_agreement - same_bin_directional_agreement",
    "itr_stock_shock_relative_amplitude_mean": "mean(log1p(|rB_t|/median|rB|)-log1p(|rS_t|/median|rS|) | E)",
    "itr_stock_shock_relative_amplitude_dispersion": "sd(log1p(|rB_t|/median|rB|)-log1p(|rS_t|/median|rS|) | E)",
    "itr_stock_shock_up_down_amplitude_asymmetry": "mean(relative_amplitude | E,rS_t>0)-mean(relative_amplitude | E,rS_t<0)",
    "itr_stock_shock_bond_response_followthrough": "mean(sign(rB_t)*sign(rB_t+1) | E, contiguous t+1)",
    "itr_stock_shock_bond_response_deferment_share": "mean(|rB_t+1|/(|rB_t|+|rB_t+1|) | E, contiguous t+1)",
    "itr_stock_shock_bond_vs_stock_followthrough_gap": "mean(sign(rB_t)*sign(rB_t+1)-sign(rS_t)*sign(rS_t+1) | E, contiguous t+1)",
}


def factor_mining_intraday_transmission_response_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for the generic scratch expansion runner."""

    return factor_mining_intraday_transmission_response_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


def _canonical_market_code(value: object) -> str:
    """Accept only explicit exchange-qualified codes from a supplied context."""

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


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Return physical T-day continuous-session price snapshots, else an empty frame."""

    if not isinstance(panel, pd.DataFrame) or any(column not in panel.columns for column in _REQUIRED_PANEL_COLUMNS):
        return pd.DataFrame(columns=["dt", "code", "seq", "trade_time", "last"])
    checked = ensure_panel_index(panel)
    frame = checked.reset_index().loc[:, ["dt", "code", "seq", "trade_time", "last"]].copy()
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
    out["last"] = pd.to_numeric(out["last"], errors="coerce")
    return out.loc[np.isfinite(out["last"]) & (out["last"] > _EPS)].sort_values(
        ["code", "trade_time", "seq"], kind="mergesort"
    )


def _context_mapping(ctx: FactorComputeContext, *, score_date: pd.Timestamp) -> dict[str, str]:
    """Use only unambiguous, non-forward rows from ``ctx.bond_stock_map``."""

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
        mapping_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
        frame = frame.loc[mapping_dates.notna() & (mapping_dates <= score_date)].copy()
    if frame.empty:
        return {}
    frame["bond_code"] = frame["code"].map(_canonical_market_code)
    frame["mapped_stock_code"] = frame["stock_code"].map(_canonical_market_code)
    frame = frame.loc[(frame["bond_code"] != "") & (frame["mapped_stock_code"] != "")].copy()
    if frame.empty:
        return {}
    # A duplicate is an unresolved mapping ambiguity, even when one row happens
    # to match a stock panel.  Do not silently select the last row.
    ambiguous = frame["bond_code"].duplicated(keep=False)
    frame = frame.loc[~ambiguous]
    return dict(zip(frame["bond_code"], frame["mapped_stock_code"], strict=False))


def _asset_returns(frame: pd.DataFrame) -> pd.Series:
    """Compute log returns only across exactly adjacent observed five-minute bins."""

    if frame.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="event_time"))
    data = frame.loc[:, ["trade_time", "seq", "last"]].copy()
    data["event_time"] = pd.to_datetime(data["trade_time"], errors="coerce").dt.floor(_EVENT_BIN)
    data = data.loc[data["event_time"].notna() & np.isfinite(data["last"]) & (data["last"] > _EPS)]
    if data.empty:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="event_time"))
    endpoint = (
        data.sort_values(["event_time", "trade_time", "seq"], kind="mergesort")
        .groupby("event_time", sort=True)["last"]
        .last()
        .astype("float64")
    )
    returns = np.log(endpoint).diff()
    contiguous = endpoint.index.to_series().diff().eq(_EVENT_BIN).to_numpy()
    returns.loc[~contiguous] = np.nan
    returns.name = "return"
    return returns.replace([np.inf, -np.inf], np.nan)


def _joint_returns(bond_frame: pd.DataFrame, stock_frame: pd.DataFrame) -> pd.DataFrame:
    bond = _asset_returns(bond_frame).rename("bond_return")
    stock = _asset_returns(stock_frame).rename("stock_return")
    return bond.to_frame().join(stock, how="inner").dropna().sort_index()


def _response_metrics(joint: pd.DataFrame) -> dict[str, float]:
    """Build all shock-conditioned metrics from an already mapped price path."""

    out = {signal: float("nan") for signal in _ALL_SIGNALS}
    if len(joint) < _MIN_COMMON_RETURNS:
        return out
    stock_abs = np.abs(joint["stock_return"].to_numpy(dtype="float64"))
    threshold = float(np.quantile(stock_abs, 0.75))
    if not np.isfinite(threshold) or threshold <= _EPS:
        return out
    shocks = joint.loc[stock_abs >= threshold].copy()
    if len(shocks) < _MIN_SHOCKS:
        return out

    same_direction = np.sign(shocks["stock_return"].to_numpy()) * np.sign(shocks["bond_return"].to_numpy())
    out["itr_stock_shock_same_bin_directional_agreement"] = float(np.mean(same_direction))

    bond_scale = float(np.median(np.abs(joint["bond_return"].to_numpy(dtype="float64"))))
    stock_scale = float(np.median(np.abs(joint["stock_return"].to_numpy(dtype="float64"))))
    if np.isfinite(bond_scale) and np.isfinite(stock_scale) and bond_scale > _EPS and stock_scale > _EPS:
        relative_amplitude = np.log1p(np.abs(shocks["bond_return"]) / bond_scale) - np.log1p(
            np.abs(shocks["stock_return"]) / stock_scale
        )
        relative_values = relative_amplitude.to_numpy(dtype="float64")
        if np.isfinite(relative_values).all():
            out["itr_stock_shock_relative_amplitude_mean"] = float(np.mean(relative_values))
            out["itr_stock_shock_relative_amplitude_dispersion"] = (
                float(np.std(relative_values, ddof=1)) if len(relative_values) >= _MIN_SHOCKS else float("nan")
            )
            up = relative_amplitude.loc[shocks["stock_return"] > 0.0]
            down = relative_amplitude.loc[shocks["stock_return"] < 0.0]
            if len(up) >= _MIN_SIDE_SHOCKS and len(down) >= _MIN_SIDE_SHOCKS:
                out["itr_stock_shock_up_down_amplitude_asymmetry"] = float(up.mean() - down.mean())

    next_index = pd.DatetimeIndex(shocks.index) + _EVENT_BIN
    next_rows = joint.reindex(next_index).copy()
    next_rows.index = shocks.index
    valid_next = next_rows["bond_return"].notna() & next_rows["stock_return"].notna()
    current = shocks.loc[valid_next]
    future = next_rows.loc[valid_next]
    if len(current) < _MIN_FOLLOW_THROUGH:
        return out

    delayed_direction = np.sign(current["stock_return"].to_numpy()) * np.sign(future["bond_return"].to_numpy())
    delayed = float(np.mean(delayed_direction))
    out["itr_stock_shock_next_bin_directional_agreement"] = delayed
    out["itr_stock_shock_directional_latency_gap"] = delayed - out[
        "itr_stock_shock_same_bin_directional_agreement"
    ]

    bond_follow = np.sign(current["bond_return"].to_numpy()) * np.sign(future["bond_return"].to_numpy())
    stock_follow = np.sign(current["stock_return"].to_numpy()) * np.sign(future["stock_return"].to_numpy())
    out["itr_stock_shock_bond_response_followthrough"] = float(np.mean(bond_follow))
    out["itr_stock_shock_bond_vs_stock_followthrough_gap"] = float(np.mean(bond_follow - stock_follow))

    immediate_abs = np.abs(current["bond_return"].to_numpy(dtype="float64"))
    delayed_abs = np.abs(future["bond_return"].to_numpy(dtype="float64"))
    denominator = immediate_abs + delayed_abs
    valid_denom = np.isfinite(denominator) & (denominator > _EPS)
    if int(valid_denom.sum()) >= _MIN_FOLLOW_THROUGH:
        out["itr_stock_shock_bond_response_deferment_share"] = float(
            np.mean(delayed_abs[valid_denom] / denominator[valid_denom])
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
        stock_frame = _strict_physical_frame(ctx.stock_panel, score_date=score_date) if isinstance(ctx.stock_panel, pd.DataFrame) else pd.DataFrame()
        mapping = _context_mapping(ctx, score_date=score_date)
        bond_groups = {
            canonical: group
            for code, group in bond_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        stock_groups = {
            canonical: group
            for code, group in stock_frame.groupby("code", sort=False)
            if (canonical := _canonical_market_code(code))
        }
        rows: list[dict[str, object]] = []
        for dt, raw_code in output_index:
            canonical_bond = _canonical_market_code(raw_code)
            row: dict[str, object] = {"dt": dt, "code": raw_code}
            row.update({signal: float("nan") for signal in _ALL_SIGNALS})
            bond_group = bond_groups.get(canonical_bond)
            stock_group = stock_groups.get(mapping.get(canonical_bond, ""))
            if bond_group is not None and stock_group is not None:
                row.update(_response_metrics(_joint_returns(bond_group, stock_group)))
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
class FactorMiningIntradayTransmissionResponseV1(Factor):
    """Research-only mapped-stock shock response catalogue at T1430."""

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
