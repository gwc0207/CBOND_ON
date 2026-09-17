"""Research-only strict-PIT lunch-break / reopen dynamics catalogue.

This module is deliberately import-only: it is not imported from
``defs.__init__`` and is absent from factor/model/live configuration.  It uses
only ``FactorComputeContext.panel`` and only physical score-day observations
visible by 14:29.  It neither reads labels, scores, PnL, pools, files,
databases, Redis, daily data, nor a stock panel.

The existing cross-sectional-residual catalogue already contains a residual of
the *raw* lunch reopen return.  This module intentionally does not output that
return.  Instead it describes mechanisms around the noon discontinuity: whether
the first post-lunch price path absorbs the jump, whether trade activity and
per-trade impact restart differently from the late morning, and how displayed
L1 depth rebuilds across the interruption.

Missing local pre/post-lunch observations, local timestamp gaps, duplicate or
reversed timestamps, cumulative-counter resets, and invalid price, quote, or
depth inputs fail closed to NaN for the entire bond-day.  The local
``ctx.cache`` only memoizes values derived from ``ctx.panel``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index
from cbond_on.domain.factors.operators._intraday_utils import ensure_trade_time


KERNEL_NAME = "factor_mining_intraday_lunch_reopen_dynamics_v1"
CATALOG_VERSION = "20260803_intraday_lunch_reopen_dynamics_v1"
_EPS = 1e-12
_MORNING_START = dt_time(9, 30)
_MORNING_END = dt_time(11, 30)
_AFTERNOON_START = dt_time(13, 0)
_CUTOFF = dt_time(14, 29)
_PRE_WINDOW_START = dt_time(11, 10)
_POST_WINDOW_END = dt_time(13, 20)
_PRE_LATEST_ALLOWED = dt_time(11, 25)
_POST_EARLIEST_ALLOWED = dt_time(13, 5)
_MAX_LOCAL_GAP_SECONDS = 10.0 * 60.0
_MIN_LOCAL_ROWS = 4
_MIN_TRADE_EVENTS = 2


@dataclass(frozen=True)
class CatalogEntry:
    """One auditable, research-only lunch-reopen candidate."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


def _entries(family: str, signals: Iterable[str], hypothesis: str) -> tuple[CatalogEntry, ...]:
    return tuple(CatalogEntry(family, signal, KERNEL_NAME, hypothesis) for signal in signals)


_PRICE_RESOLUTION_SIGNALS = (
    "lrd_price_gap_absorption_signed",
    "lrd_price_gap_absorption_efficiency",
    "lrd_price_gap_absorption_frontload",
)
_TRADE_RESUMPTION_SIGNALS = (
    "lrd_trade_reopen_intensity_ratio",
    "lrd_trade_reopen_notional_per_trade_ratio",
    "lrd_trade_reopen_impact_per_trade_ratio",
)
_BOOK_REBUILD_SIGNALS = (
    "lrd_depth_reopen_level_gap",
    "lrd_depth_reopen_rebuild_speed",
    "lrd_depth_reopen_imbalance_switch",
)

_CATALOG = (
    _entries(
        "lunch_price_discontinuity_resolution",
        _PRICE_RESOLUTION_SIGNALS,
        "The first post-lunch price path can absorb, efficiently resolve, or front-load an existing noon discontinuity without outputting the raw lunch-gap return.",
    )
    + _entries(
        "lunch_trade_resumption_recovery",
        _TRADE_RESUMPTION_SIGNALS,
        "The immediate post-lunch restart can differ from the late-morning baseline in trade rate, mean traded notional, and price impact per trade rather than a full-session event clock.",
    )
    + _entries(
        "lunch_book_rebuild_dynamics",
        _BOOK_REBUILD_SIGNALS,
        "Displayed L1 liquidity can jump, rebuild, and rotate between bid and ask sides specifically across the lunch interruption, distinct from generic queue lifecycle or ladder repricing events.",
    )
)
_ENTRY_BY_SIGNAL = {entry.signal: entry for entry in _CATALOG}
_ALL_SIGNALS = tuple(entry.signal for entry in _CATALOG)

# Let G=log(P_post,0/P_pre,-1), R=log(P_post,K/P_post,0), r_i be a
# post-lunch local return, N/A be cumulative trade count/amount, D be L1 total
# depth, and I=(bid_depth-ask_depth)/D.  ``pre`` is the final four valid
# snapshots from 11:10--11:30; ``post`` is the first four valid snapshots from
# 13:00--13:20.  Raw G is deliberately only an internal orientation anchor.
FORMULAS: dict[str, str] = {
    "lrd_price_gap_absorption_signed": "-sign(G)*R",
    "lrd_price_gap_absorption_efficiency": "-sign(G)*R/sum_i(abs(r_i))",
    "lrd_price_gap_absorption_frontload": "-sign(G)*(r_1-sum_{i=2..K}(r_i))",
    "lrd_trade_reopen_intensity_ratio": "log((sum(deltaN_post)/elapsed_post)/(sum(deltaN_pre)/elapsed_pre))",
    "lrd_trade_reopen_notional_per_trade_ratio": "log((sum(deltaA_post)/sum(deltaN_post))/(sum(deltaA_pre)/sum(deltaN_pre)))",
    "lrd_trade_reopen_impact_per_trade_ratio": "log((sum(abs(r_post))/sum(deltaN_post))/(sum(abs(r_pre))/sum(deltaN_pre)))",
    "lrd_depth_reopen_level_gap": "log(D_post,0/D_pre,-1)",
    "lrd_depth_reopen_rebuild_speed": "log(D_post,K/D_post,0)",
    "lrd_depth_reopen_imbalance_switch": "I_post,0-I_pre,-1",
}

_REQUIRED_PANEL_COLUMNS = (
    "trade_time",
    "last",
    "num_trades",
    "amount",
    "ask_price1",
    "bid_price1",
    "ask_volume1",
    "bid_volume1",
)


def factor_mining_intraday_lunch_reopen_dynamics_catalog() -> tuple[CatalogEntry, ...]:
    """Return the immutable family-first research catalogue."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Compatibility entrypoint for generic research catalogue loaders."""

    return factor_mining_intraday_lunch_reopen_dynamics_catalog()


def _requested_entry(ctx: FactorComputeContext) -> CatalogEntry:
    signal = str(ctx.params.get("signal", "")).strip()
    if not signal:
        raise ValueError(f"{KERNEL_NAME} requires explicit params.signal")
    entry = _ENTRY_BY_SIGNAL.get(signal)
    if entry is None:
        raise KeyError(f"{KERNEL_NAME} unknown signal: {signal}")
    return entry


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


def _strict_physical_frame(panel: pd.DataFrame, *, score_date: pd.Timestamp) -> pd.DataFrame:
    """Keep only physical score-day continuous-session observations <=14:29."""

    checked = ensure_panel_index(panel)
    missing = [column for column in _REQUIRED_PANEL_COLUMNS if column not in checked.columns]
    if missing:
        raise KeyError(f"{KERNEL_NAME} missing panel columns: {missing}")
    checked = ensure_trade_time(checked)
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


def _nan_record() -> dict[str, float]:
    return {signal: float("nan") for signal in _ALL_SIGNALS}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")


def _valid_whole_day(group: pd.DataFrame) -> bool:
    """Fail closed before taking local lunch windows from the panel."""

    times = pd.to_datetime(group["trade_time"], errors="coerce")
    if times.isna().any() or times.duplicated().any() or (times.diff().dropna() <= pd.Timedelta(0)).any():
        return False
    last = _numeric(group, "last")
    count = _numeric(group, "num_trades")
    amount = _numeric(group, "amount")
    ask_price = _numeric(group, "ask_price1")
    bid_price = _numeric(group, "bid_price1")
    ask_depth = _numeric(group, "ask_volume1")
    bid_depth = _numeric(group, "bid_volume1")
    depth = ask_depth + bid_depth
    if not (
        np.isfinite(last).all()
        and np.isfinite(count).all()
        and np.isfinite(amount).all()
        and np.isfinite(ask_price).all()
        and np.isfinite(bid_price).all()
        and np.isfinite(ask_depth).all()
        and np.isfinite(bid_depth).all()
        and (last > 0.0).all()
        and (count >= 0.0).all()
        and (amount >= 0.0).all()
        and (ask_price > 0.0).all()
        and (bid_price > 0.0).all()
        and (ask_price >= bid_price).all()
        and (ask_depth >= 0.0).all()
        and (bid_depth >= 0.0).all()
        and (depth > _EPS).all()
    ):
        return False
    return bool(
        np.isfinite(np.diff(count)).all()
        and np.isfinite(np.diff(amount)).all()
        and not (np.diff(count) < 0.0).any()
        and not (np.diff(amount) < 0.0).any()
    )


def _local_window(group: pd.DataFrame, *, pre_lunch: bool) -> pd.DataFrame | None:
    times = pd.to_datetime(group["trade_time"], errors="coerce")
    clocks = times.dt.time
    if pre_lunch:
        mask = (clocks >= _PRE_WINDOW_START) & (clocks <= _MORNING_END)
    else:
        mask = (clocks >= _AFTERNOON_START) & (clocks <= _POST_WINDOW_END)
    local = group.loc[mask].copy()
    if len(local) < _MIN_LOCAL_ROWS:
        return None
    local["trade_time"] = pd.to_datetime(local["trade_time"], errors="coerce")
    local = local.sort_values("trade_time", kind="mergesort")
    local_times = local["trade_time"]
    if local_times.isna().any() or local_times.duplicated().any():
        return None
    gaps = local_times.diff().dropna().dt.total_seconds().to_numpy(dtype="float64")
    if not np.isfinite(gaps).all() or (gaps <= 0.0).any() or (gaps > _MAX_LOCAL_GAP_SECONDS).any():
        return None
    local_clocks = local_times.dt.time
    if pre_lunch:
        if local_clocks.iloc[-1] < _PRE_LATEST_ALLOWED:
            return None
        return local.tail(_MIN_LOCAL_ROWS).copy()
    if local_clocks.iloc[0] > _POST_EARLIEST_ALLOWED:
        return None
    return local.head(_MIN_LOCAL_ROWS).copy()


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator) and numerator > _EPS and denominator > _EPS):
        return float("nan")
    value = float(np.log(numerator / denominator))
    return value if np.isfinite(value) else float("nan")


def _elapsed_seconds(frame: pd.DataFrame) -> float:
    times = pd.to_datetime(frame["trade_time"], errors="coerce")
    if times.isna().any() or len(times) < 2:
        return float("nan")
    seconds = float((times.iloc[-1] - times.iloc[0]).total_seconds())
    return seconds if np.isfinite(seconds) and seconds > _EPS else float("nan")


def _group_metrics(group: pd.DataFrame) -> dict[str, float]:
    out = _nan_record()
    if not _valid_whole_day(group):
        return out
    pre = _local_window(group, pre_lunch=True)
    post = _local_window(group, pre_lunch=False)
    if pre is None or post is None:
        return out

    pre_last = _numeric(pre, "last")
    post_last = _numeric(post, "last")
    pre_count = _numeric(pre, "num_trades")
    post_count = _numeric(post, "num_trades")
    pre_amount = _numeric(pre, "amount")
    post_amount = _numeric(post, "amount")
    pre_depth = _numeric(pre, "ask_volume1") + _numeric(pre, "bid_volume1")
    post_depth = _numeric(post, "ask_volume1") + _numeric(post, "bid_volume1")
    pre_imbalance = (_numeric(pre, "bid_volume1") - _numeric(pre, "ask_volume1")) / pre_depth
    post_imbalance = (_numeric(post, "bid_volume1") - _numeric(post, "ask_volume1")) / post_depth
    if not (
        np.isfinite(pre_imbalance).all()
        and np.isfinite(post_imbalance).all()
        and (np.abs(pre_imbalance) <= 1.0 + _EPS).all()
        and (np.abs(post_imbalance) <= 1.0 + _EPS).all()
    ):
        return out

    gap = _safe_log_ratio(float(post_last[0]), float(pre_last[-1]))
    pre_returns = np.diff(np.log(pre_last))
    post_returns = np.diff(np.log(post_last))
    recovery = _safe_log_ratio(float(post_last[-1]), float(post_last[0]))
    total_post_move = float(np.abs(post_returns).sum())
    if np.isfinite(gap) and np.isfinite(recovery) and total_post_move > _EPS:
        orientation = float(np.sign(gap))
        out["lrd_price_gap_absorption_signed"] = float(-orientation * recovery)
        out["lrd_price_gap_absorption_efficiency"] = float(-orientation * recovery / total_post_move)
        out["lrd_price_gap_absorption_frontload"] = float(
            -orientation * (post_returns[0] - float(post_returns[1:].sum()))
        )

    pre_count_inc = np.diff(pre_count)
    post_count_inc = np.diff(post_count)
    pre_amount_inc = np.diff(pre_amount)
    post_amount_inc = np.diff(post_amount)
    pre_trade_total = float(pre_count_inc.sum())
    post_trade_total = float(post_count_inc.sum())
    pre_amount_total = float(pre_amount_inc.sum())
    post_amount_total = float(post_amount_inc.sum())
    pre_events = int((pre_count_inc > 0.0).sum())
    post_events = int((post_count_inc > 0.0).sum())
    pre_elapsed = _elapsed_seconds(pre)
    post_elapsed = _elapsed_seconds(post)
    if (
        pre_events >= _MIN_TRADE_EVENTS
        and post_events >= _MIN_TRADE_EVENTS
        and pre_trade_total > _EPS
        and post_trade_total > _EPS
        and np.isfinite(pre_elapsed)
        and np.isfinite(post_elapsed)
    ):
        intensity_ratio = _safe_log_ratio(post_trade_total / post_elapsed, pre_trade_total / pre_elapsed)
        if np.isfinite(intensity_ratio):
            out["lrd_trade_reopen_intensity_ratio"] = intensity_ratio
        notional_ratio = _safe_log_ratio(
            post_amount_total / post_trade_total,
            pre_amount_total / pre_trade_total,
        )
        if np.isfinite(notional_ratio):
            out["lrd_trade_reopen_notional_per_trade_ratio"] = notional_ratio
        pre_event_move = float(np.abs(pre_returns)[pre_count_inc > 0.0].sum())
        post_event_move = float(np.abs(post_returns)[post_count_inc > 0.0].sum())
        impact_ratio = _safe_log_ratio(
            post_event_move / post_trade_total,
            pre_event_move / pre_trade_total,
        )
        if np.isfinite(impact_ratio):
            out["lrd_trade_reopen_impact_per_trade_ratio"] = impact_ratio

    level_gap = _safe_log_ratio(float(post_depth[0]), float(pre_depth[-1]))
    rebuild_speed = _safe_log_ratio(float(post_depth[-1]), float(post_depth[0]))
    if np.isfinite(level_gap):
        out["lrd_depth_reopen_level_gap"] = level_gap
    if np.isfinite(rebuild_speed):
        out["lrd_depth_reopen_rebuild_speed"] = rebuild_speed
    out["lrd_depth_reopen_imbalance_switch"] = float(post_imbalance[0] - pre_imbalance[-1])
    return {signal: (float(value) if np.isfinite(value) else float("nan")) for signal, value in out.items()}


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
        strict = _strict_physical_frame(ctx.panel, score_date=score_date)
        if strict.empty:
            built = pd.DataFrame(index=output_index, columns=_ALL_SIGNALS, dtype="float64")
        else:
            rows: list[dict[str, object]] = []
            for (dt, code), group in strict.groupby(["dt", "code"], sort=False):
                row: dict[str, object] = {"dt": dt, "code": code}
                row.update(_group_metrics(group))
                rows.append(row)
            built = pd.DataFrame(rows).set_index(["dt", "code"])[list(_ALL_SIGNALS)].sort_index()
            built = built.replace([np.inf, -np.inf], np.nan).reindex(output_index)

    with ctx.cache_lock:
        cached = ctx.cache.get(cache_key)
        if isinstance(cached, pd.DataFrame):
            return cached
        ctx.cache[cache_key] = built
    return built


@FactorRegistry.register(KERNEL_NAME)
class FactorMiningIntradayLunchReopenDynamicsV1(Factor):
    """Research-only T1430 lunch-break discontinuity and reopening dynamics."""

    name = KERNEL_NAME
    kernel_name = KERNEL_NAME
    requires_stock_panel = False
    requires_bond_stock_map = False

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        entry = _requested_entry(ctx)
        features = _feature_frame(ctx)
        out = features[entry.signal].copy()
        out.name = self.output_name(entry.signal)
        return out
