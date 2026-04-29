from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.infra.backtest.execution import apply_twap_bps
from cbond_on.infra.data.io import read_table_range
from cbond_on.infra.io.market_twap import read_twap_daily


@dataclass(frozen=True)
class BenchmarkPoolConfig:
    pool_table: str
    buy_twap_col: str
    sell_twap_col: str
    positive_field: str
    positive_fallback_field: str
    positive_threshold: float


def load_benchmark_pool_config(cfg: dict | None = None) -> BenchmarkPoolConfig:
    raw = dict(cfg or load_config_file("benchmark"))
    return BenchmarkPoolConfig(
        pool_table=str(raw.get("pool_table", "quant_factor_dev.researcher_xuvb.o_0005")),
        buy_twap_col=str(raw.get("buy_twap_col", "twap_1442_1457")),
        sell_twap_col=str(raw.get("sell_twap_col", "twap_0930_0945")),
        positive_field=str(raw.get("positive_field", "factor_value")),
        positive_fallback_field=str(raw.get("positive_fallback_field", "weight")),
        positive_threshold=float(raw.get("positive_threshold", 0.0)),
    )


def _normalize_code_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    return out


def _normalize_pool_code(df: pd.DataFrame) -> pd.Series:
    if "code" in df.columns:
        return _normalize_code_series(df["code"])
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        left = _normalize_code_series(df["instrument_code"])
        right = _normalize_code_series(df["exchange_code"])
        return left + "." + right
    if "instrument_code" in df.columns:
        return _normalize_code_series(df["instrument_code"])
    raise RuntimeError("benchmark pool missing code columns (need code or instrument_code/exchange_code)")


def _load_benchmark_pool_day_codes(
    *,
    raw_data_root: str | Path,
    day: date,
    pool_cfg: BenchmarkPoolConfig,
) -> pd.Series:
    pool_df = read_table_range(raw_data_root, pool_cfg.pool_table, day, day)
    if pool_df.empty:
        raise RuntimeError(
            f"benchmark pool missing day file: table={pool_cfg.pool_table} day={day:%Y-%m-%d}"
        )
    pool_df = pool_df.copy()
    pool_df["code"] = _normalize_pool_code(pool_df)
    if pool_cfg.positive_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[pool_cfg.positive_field], errors="coerce") > pool_cfg.positive_threshold
    elif pool_cfg.positive_fallback_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[pool_cfg.positive_fallback_field], errors="coerce") > pool_cfg.positive_threshold
    else:
        raise RuntimeError(
            "benchmark pool missing positive-filter columns: "
            f"{pool_cfg.positive_field} / {pool_cfg.positive_fallback_field}"
        )
    codes = pool_df.loc[mask, "code"].dropna()
    codes = _normalize_code_series(codes).drop_duplicates().sort_values(kind="mergesort")
    if codes.empty:
        raise RuntimeError(
            f"benchmark pool empty after positive filter on {day:%Y-%m-%d}: "
            f"{pool_cfg.positive_field}>{pool_cfg.positive_threshold}"
        )
    return codes


def compute_benchmark_return_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> float:
    cfg = pool_cfg or load_benchmark_pool_config()
    pool_codes = _load_benchmark_pool_day_codes(
        raw_data_root=raw_data_root,
        day=trade_day,
        pool_cfg=cfg,
    )

    buy_df = read_twap_daily(str(raw_data_root), trade_day)
    sell_df = read_twap_daily(str(raw_data_root), next_day)
    if buy_df.empty:
        raise RuntimeError(f"benchmark twap missing buy day: {trade_day:%Y-%m-%d}")
    if sell_df.empty:
        raise RuntimeError(f"benchmark twap missing sell day: {next_day:%Y-%m-%d}")
    if cfg.buy_twap_col not in buy_df.columns:
        raise RuntimeError(f"benchmark twap missing buy column: {cfg.buy_twap_col}")
    if cfg.sell_twap_col not in sell_df.columns:
        raise RuntimeError(f"benchmark twap missing sell column: {cfg.sell_twap_col}")

    buy = buy_df.copy()
    sell = sell_df.copy()
    if "code" not in buy.columns:
        raise RuntimeError("benchmark buy twap missing code column")
    if "code" not in sell.columns:
        raise RuntimeError("benchmark sell twap missing code column")
    buy["code"] = _normalize_code_series(buy["code"])
    sell["code"] = _normalize_code_series(sell["code"])

    pool_df = pd.DataFrame({"code": pool_codes.values})
    merged = (
        pool_df.merge(buy[["code", cfg.buy_twap_col]], on="code", how="inner")
        .merge(sell[["code", cfg.sell_twap_col]], on="code", how="inner")
    )
    merged = merged[
        merged[cfg.buy_twap_col].notna()
        & merged[cfg.sell_twap_col].notna()
        & (merged[cfg.buy_twap_col] > 0)
        & (merged[cfg.sell_twap_col] > 0)
    ]
    if merged.empty:
        raise RuntimeError(
            "benchmark universe empty after twap merge/filter: "
            f"trade_day={trade_day:%Y-%m-%d} next_day={next_day:%Y-%m-%d}"
        )
    buy_px = apply_twap_bps(merged[cfg.buy_twap_col], float(buy_bps), side="buy")
    sell_px = apply_twap_bps(merged[cfg.sell_twap_col], float(sell_bps), side="sell")
    ret = (sell_px - buy_px) / buy_px
    out = float(pd.to_numeric(ret, errors="coerce").mean())
    if pd.isna(out):
        raise RuntimeError(
            "benchmark return is NaN after computation: "
            f"trade_day={trade_day:%Y-%m-%d} next_day={next_day:%Y-%m-%d}"
        )
    return out


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
