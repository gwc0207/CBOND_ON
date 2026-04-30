from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.infra.backtest.execution import (
    apply_twap_bps,
    split_cycle_return_by_bridge,
)
from cbond_on.infra.io.market_twap import read_price_daily, read_twap_daily
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    apply_pool_filter_to_universe,
    resolve_pool_codes_for_trade_day,
)


@dataclass(frozen=True)
class BenchmarkPoolConfig:
    pool_table: str
    buy_twap_col: str
    sell_twap_col: str
    positive_field: str
    positive_fallback_field: str
    positive_threshold: float
    pool_lag_trading_days: int
    pool_asset: str


@dataclass(frozen=True)
class BenchmarkReturnBreakdown:
    full_cycle_ret_net: float
    buy_leg_ret_net: float
    sell_leg_ret_net: float
    count: int


def load_benchmark_pool_config(cfg: dict | None = None) -> BenchmarkPoolConfig:
    raw = dict(cfg or load_config_file("benchmark"))
    return BenchmarkPoolConfig(
        pool_table=str(raw.get("pool_table", "quant_factor_dev.researcher_xuvb.o_0005")),
        buy_twap_col=str(raw.get("buy_twap_col", "twap_1442_1457")),
        sell_twap_col=str(raw.get("sell_twap_col", "twap_0930_0945")),
        positive_field=str(raw.get("positive_field", "factor_value")),
        positive_fallback_field=str(raw.get("positive_fallback_field", "weight")),
        positive_threshold=float(raw.get("positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(raw.get("pool_lag_trading_days", 1))),
        pool_asset=str(raw.get("pool_asset", "cbond")),
    )


def _normalize_code_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def compute_benchmark_return_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> float:
    breakdown = compute_benchmark_breakdown_for_day(
        raw_data_root=raw_data_root,
        trade_day=trade_day,
        next_day=next_day,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
    )
    return float(breakdown.full_cycle_ret_net)


def compute_benchmark_breakdown_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> BenchmarkReturnBreakdown:
    cfg = pool_cfg or load_benchmark_pool_config()
    upstream_cfg = UpstreamPoolConfig(
        pool_table=cfg.pool_table,
        positive_field=cfg.positive_field,
        positive_fallback_field=cfg.positive_fallback_field,
        positive_threshold=cfg.positive_threshold,
        pool_lag_trading_days=cfg.pool_lag_trading_days,
        pool_asset=cfg.pool_asset,
    )
    pool_codes, pool_info = resolve_pool_codes_for_trade_day(
        raw_data_root=raw_data_root,
        trade_day=trade_day,
        pool_cfg=upstream_cfg,
    )
    if bool(pool_info.get("fallback_no_filter", False)):
        print(
            "[pool_filter] fallback_no_filter",
            f"trade_day={trade_day:%Y-%m-%d}",
            f"expected_pool_day={pool_info.get('pool_day_expected')}",
            f"reason={pool_info.get('fallback_reason')}",
            f"nearest_pool_day={pool_info.get('nearest_pool_day')}",
            f"pool_table={cfg.pool_table}",
        )

    buy_df = read_twap_daily(str(raw_data_root), trade_day)
    sell_df = read_twap_daily(str(raw_data_root), next_day)
    bridge_df = read_price_daily(str(raw_data_root), next_day)
    if buy_df.empty:
        raise RuntimeError(f"benchmark twap missing buy day: {trade_day:%Y-%m-%d}")
    if sell_df.empty:
        raise RuntimeError(f"benchmark twap missing sell day: {next_day:%Y-%m-%d}")
    if bridge_df.empty:
        raise RuntimeError(f"benchmark daily_price missing next day: {next_day:%Y-%m-%d}")
    if cfg.buy_twap_col not in buy_df.columns:
        raise RuntimeError(f"benchmark twap missing buy column: {cfg.buy_twap_col}")
    if cfg.sell_twap_col not in sell_df.columns:
        raise RuntimeError(f"benchmark twap missing sell column: {cfg.sell_twap_col}")
    bridge_col = "prev_close_price"
    if bridge_col not in bridge_df.columns:
        raise RuntimeError(f"benchmark daily_price missing bridge column: {bridge_col}")

    buy = buy_df.copy()
    sell = sell_df.copy()
    bridge = bridge_df.copy()
    if "code" not in buy.columns:
        raise RuntimeError("benchmark buy twap missing code column")
    if "code" not in sell.columns:
        raise RuntimeError("benchmark sell twap missing code column")
    if "code" not in bridge.columns:
        raise RuntimeError("benchmark daily_price missing code column")
    buy["code"] = _normalize_code_series(buy["code"])
    sell["code"] = _normalize_code_series(sell["code"])
    bridge["code"] = _normalize_code_series(bridge["code"])

    merged = (
        buy[["code", cfg.buy_twap_col]]
        .merge(sell[["code", cfg.sell_twap_col]], on="code", how="inner")
        .merge(bridge[["code", bridge_col]], on="code", how="inner")
    )
    merged = apply_pool_filter_to_universe(merged, pool_codes=pool_codes)
    merged = merged[
        merged[cfg.buy_twap_col].notna()
        & merged[cfg.sell_twap_col].notna()
        & merged[bridge_col].notna()
        & (merged[cfg.buy_twap_col] > 0)
        & (merged[cfg.sell_twap_col] > 0)
        & (merged[bridge_col] > 0)
    ]
    if merged.empty:
        pool_tag = (
            f"pool_day={pool_info.get('pool_day_used')}"
            if pool_codes is not None
            else "pool_filter=fallback_no_filter"
        )
        raise RuntimeError(
            "benchmark universe empty after twap merge/filter: "
            f"trade_day={trade_day:%Y-%m-%d} next_day={next_day:%Y-%m-%d} {pool_tag}"
        )
    buy_px = apply_twap_bps(merged[cfg.buy_twap_col], float(buy_bps), side="buy")
    sell_px = apply_twap_bps(merged[cfg.sell_twap_col], float(sell_bps), side="sell")
    buy_leg, sell_leg, full_cycle = split_cycle_return_by_bridge(
        buy_px,
        sell_px,
        pd.to_numeric(merged[bridge_col], errors="coerce"),
    )
    full_out = float(pd.to_numeric(full_cycle, errors="coerce").mean())
    if pd.isna(full_out):
        raise RuntimeError(
            "benchmark return is NaN after computation: "
            f"trade_day={trade_day:%Y-%m-%d} next_day={next_day:%Y-%m-%d}"
        )
    buy_out = float(pd.to_numeric(buy_leg, errors="coerce").mean())
    sell_out = float(pd.to_numeric(sell_leg, errors="coerce").mean())
    return BenchmarkReturnBreakdown(
        full_cycle_ret_net=full_out,
        buy_leg_ret_net=buy_out,
        sell_leg_ret_net=sell_out,
        count=int(len(merged)),
    )


def _build_next_day_map(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
) -> dict[date, date]:
    wanted = sorted(set(trade_days))
    if not wanted:
        return {}
    all_days = list_available_trading_days_from_raw(
        raw_data_root,
        kind="snapshot",
        asset="cbond",
    )
    if not all_days:
        raise RuntimeError("benchmark failed: no trading calendar days in raw_data_root")
    pos_map = {d: i for i, d in enumerate(all_days)}
    out: dict[date, date] = {}
    for day in wanted:
        idx = pos_map.get(day)
        if idx is None:
            raise RuntimeError(f"benchmark trade day not in calendar: {day:%Y-%m-%d}")
        if idx + 1 >= len(all_days):
            raise RuntimeError(f"benchmark next trading day missing for: {day:%Y-%m-%d}")
        out[day] = all_days[idx + 1]
    return out


def compute_benchmark_returns_for_days(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
    skip_failed_days: bool = False,
) -> pd.Series:
    cfg = pool_cfg or load_benchmark_pool_config()
    next_day_map = _build_next_day_map(raw_data_root=raw_data_root, trade_days=trade_days)
    rows: list[tuple[date, float]] = []
    for trade_day in sorted(next_day_map):
        try:
            ret = compute_benchmark_return_for_day(
                raw_data_root=raw_data_root,
                trade_day=trade_day,
                next_day=next_day_map[trade_day],
                buy_bps=buy_bps,
                sell_bps=sell_bps,
                pool_cfg=cfg,
            )
        except Exception:
            if not skip_failed_days:
                raise
            continue
        rows.append((trade_day, ret))
    if not rows:
        return pd.Series(dtype=float)
    idx = [d for d, _ in rows]
    vals = [float(v) for _, v in rows]
    return pd.Series(vals, index=idx, dtype=float).sort_index()
