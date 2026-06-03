from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.core.universe import normalize_price_bound
from cbond_on.infra.backtest.execution import bps_to_rate, split_cycle_return_by_bridge_with_cost
from cbond_on.infra.data.io import read_table_range
from cbond_on.infra.io.market_twap import read_price_daily, read_twap_daily
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    resolve_pool_codes_for_trade_day,
)


PREV_CLOSE_PRICE_CANDIDATES = ("prev_close_price", "prev_close", "pre_close")


@dataclass(frozen=True)
class BenchmarkPoolConfig:
    method: str
    pool_table: str
    buy_twap_col: str
    sell_twap_col: str
    use_window_data: bool
    window_data_root: str
    min_price: float
    max_price: float
    positive_field: str
    positive_fallback_field: str
    positive_threshold: float
    pool_lag_trading_days: int
    pool_asset: str
    use_pool_weight: bool


@dataclass(frozen=True)
class BenchmarkReturnBreakdown:
    full_cycle_ret_net: float
    buy_leg_ret_net: float
    sell_leg_ret_net: float
    count: int
    sell_count: int = 0
    fallback_sell_codes: int = 0
    fallback_sell_weight: float = 0.0


def load_benchmark_pool_config(cfg: dict | None = None) -> BenchmarkPoolConfig:
    raw = dict(cfg or load_config_file("benchmark"))
    return BenchmarkPoolConfig(
        method=str(raw.get("method", "strict_full_overnight_net")),
        pool_table=str(raw.get("pool_table", "quant_factor_dev.researcher_xuvb.o_0005")),
        buy_twap_col=str(raw.get("buy_twap_col", "twap_1442_1457")),
        sell_twap_col=str(raw.get("sell_twap_col", "twap_0930_0939")),
        use_window_data=bool(raw.get("use_window_data", True)),
        window_data_root=str(raw.get("window_data_root", r"\\nfs\10.1.30.100\data\yinhe-data\kline\cbond\window-data")),
        min_price=normalize_price_bound(raw.get("min_price", 0.0)),
        max_price=normalize_price_bound(raw.get("max_price")),
        positive_field=str(raw.get("positive_field", "factor_value")),
        positive_fallback_field=str(raw.get("positive_fallback_field", "weight")),
        positive_threshold=float(raw.get("positive_threshold", 0.0)),
        pool_lag_trading_days=max(0, int(raw.get("pool_lag_trading_days", 1))),
        pool_asset=str(raw.get("pool_asset", "cbond")),
        use_pool_weight=bool(raw.get("use_pool_weight", False)),
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
        right = _normalize_code_series(df["exchange_code"]).str.upper()
        return left + "." + right
    if "instrument_code" in df.columns:
        return _normalize_code_series(df["instrument_code"])
    raise RuntimeError("benchmark pool missing code columns")


def _ensure_code_column(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
    work = df.copy()
    if "code" in work.columns:
        work["code"] = _normalize_code_series(work["code"])
        return work
    if "instrument_code" in work.columns and "exchange_code" in work.columns:
        left = _normalize_code_series(work["instrument_code"])
        right = _normalize_code_series(work["exchange_code"]).str.upper()
        work["code"] = left + "." + right
        return work
    raise RuntimeError(f"{label} missing code columns")


def _normalize_weight_series(weight: pd.Series) -> pd.Series:
    w = pd.to_numeric(weight, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return w / total


def _first_price_series(df: pd.DataFrame, candidates: Iterable[str], *, label: str) -> pd.Series:
    if df.empty:
        raise RuntimeError(f"benchmark daily_price missing for {label}")
    work = _ensure_code_column(df, label=f"{label} daily_price")
    for col in candidates:
        if col in work.columns:
            out = work[["code", col]].copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=["code"])
            return out.groupby("code", sort=False)[col].first()
    raise RuntimeError(f"benchmark daily_price missing {label} columns: {list(candidates)}")


def _twap_col_to_window_file(col: str) -> str:
    text = str(col).strip()
    if text.startswith("twap_"):
        text = text[5:]
    return text.replace("_", "-")


def _read_window_twap_daily(day: date, cfg: BenchmarkPoolConfig) -> pd.DataFrame:
    base = Path(cfg.window_data_root) / f"{day:%Y-%m}" / f"{day:%Y-%m-%d}"
    if not base.exists():
        return pd.DataFrame()

    cols = [cfg.buy_twap_col, cfg.sell_twap_col]
    windows = []
    for col in cols:
        window = _twap_col_to_window_file(col)
        if window and window not in windows:
            windows.append(window)

    merged: pd.DataFrame | None = None
    for window in windows:
        p = base / f"{window}.parquet"
        if not p.exists():
            continue
        one = pd.read_parquet(p)
        one = _ensure_code_column(one, label=f"window-data {window}")
        if "twap_price" not in one.columns:
            continue
        col = f"twap_{window.replace('-', '_')}"
        tmp = one[["code", "twap_price"]].copy()
        tmp[col] = pd.to_numeric(tmp["twap_price"], errors="coerce")
        tmp = tmp.dropna(subset=[col]).groupby("code", as_index=False)[col].mean()
        tmp = tmp[["code", col]]
        merged = tmp if merged is None else merged.merge(tmp, on="code", how="outer")

    return merged if merged is not None else pd.DataFrame()


def _read_benchmark_twap_daily(raw_data_root: str | Path, day: date, cfg: BenchmarkPoolConfig) -> pd.DataFrame:
    if cfg.use_window_data:
        return _read_window_twap_daily(day, cfg)
    return read_twap_daily(str(raw_data_root), day)


def _load_target_pool_for_buy_day(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    cfg: BenchmarkPoolConfig,
) -> pd.DataFrame:
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
        trade_day=buy_day,
        pool_cfg=upstream_cfg,
    )
    if pool_codes is None:
        raise RuntimeError(
            "benchmark target pool unavailable: "
            f"buy_day={buy_day:%Y-%m-%d} reason={pool_info.get('fallback_reason')}"
        )
    pool_day = pool_info.get("pool_day_used")
    if pool_day is None:
        raise RuntimeError(f"benchmark target pool missing used day: buy_day={buy_day:%Y-%m-%d}")

    pool_df = read_table_range(raw_data_root, cfg.pool_table, pool_day, pool_day)
    if pool_df.empty:
        raise RuntimeError(f"benchmark target pool file empty: pool_day={pool_day:%Y-%m-%d}")
    pool_df = pool_df.copy()
    pool_df["code"] = _normalize_pool_code(pool_df)
    if cfg.positive_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[cfg.positive_field], errors="coerce") > cfg.positive_threshold
    elif cfg.positive_fallback_field in pool_df.columns:
        mask = pd.to_numeric(pool_df[cfg.positive_fallback_field], errors="coerce") > cfg.positive_threshold
    else:
        raise RuntimeError(
            "benchmark target pool missing positive fields: "
            f"{cfg.positive_field}/{cfg.positive_fallback_field}"
        )

    keep_cols = ["code"]
    if "weight" in pool_df.columns:
        keep_cols.append("weight")
    target = pool_df.loc[mask, keep_cols].copy()
    target["code"] = _normalize_code_series(target["code"])
    target = target[target["code"].isin(pool_codes)]
    target = target.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")
    if target.empty:
        raise RuntimeError(f"benchmark target pool empty after filter: buy_day={buy_day:%Y-%m-%d}")

    if cfg.use_pool_weight and "weight" in target.columns:
        normalized = _normalize_weight_series(target["weight"])
        if normalized.empty:
            target["target_weight"] = 1.0 / len(target)
        else:
            target["target_weight"] = normalized.reindex(target.index).fillna(0.0)
    else:
        target["target_weight"] = 1.0 / len(target)
    target["pool_day"] = pd.to_datetime(pool_day)
    target["buy_trade_day"] = pd.to_datetime(buy_day)
    return target[["code", "target_weight", "pool_day", "buy_trade_day"]].reset_index(drop=True)


def compute_benchmark_detail_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    target = _load_target_pool_for_buy_day(
        raw_data_root=raw_data_root,
        buy_day=trade_day,
        cfg=cfg,
    )

    buy_df = _read_benchmark_twap_daily(raw_data_root, trade_day, cfg)
    if buy_df.empty or cfg.buy_twap_col not in buy_df.columns:
        raise RuntimeError(f"benchmark buy TWAP missing: {trade_day:%Y-%m-%d} {cfg.buy_twap_col}")
    buy = _ensure_code_column(buy_df, label="benchmark buy TWAP")[["code", cfg.buy_twap_col]].copy()

    sell_df = _read_benchmark_twap_daily(raw_data_root, next_day, cfg)
    if sell_df.empty or cfg.sell_twap_col not in sell_df.columns:
        raise RuntimeError(f"benchmark sell TWAP missing: {next_day:%Y-%m-%d} {cfg.sell_twap_col}")
    sell = _ensure_code_column(sell_df, label="benchmark sell TWAP")[["code", cfg.sell_twap_col]].copy()

    price_df = read_price_daily(str(raw_data_root), next_day)
    bridge = _first_price_series(
        price_df,
        PREV_CLOSE_PRICE_CANDIDATES,
        label="strict_prev_close_price",
    ).rename("bridge_prev_close")

    detail = (
        target.merge(buy, on="code", how="inner")
        .merge(sell, on="code", how="inner")
        .merge(bridge.reset_index(), on="code", how="inner")
    )
    detail[cfg.buy_twap_col] = pd.to_numeric(detail[cfg.buy_twap_col], errors="coerce")
    detail[cfg.sell_twap_col] = pd.to_numeric(detail[cfg.sell_twap_col], errors="coerce")
    detail["bridge_prev_close"] = pd.to_numeric(detail["bridge_prev_close"], errors="coerce")
    detail = detail[
        detail[cfg.buy_twap_col].notna()
        & detail[cfg.sell_twap_col].notna()
        & detail["bridge_prev_close"].notna()
        & (detail[cfg.buy_twap_col] > 0)
        & (detail[cfg.sell_twap_col] > 0)
        & (detail["bridge_prev_close"] > 0)
    ].copy()
    if detail.empty:
        raise RuntimeError(f"benchmark full-cycle detail empty: trade_day={trade_day:%Y-%m-%d}")

    weights = _normalize_weight_series(detail["target_weight"])
    detail["weight"] = weights.reindex(detail.index).fillna(0.0) if not weights.empty else 1.0 / len(detail)
    buy_leg, sell_leg, full = split_cycle_return_by_bridge_with_cost(
        detail[cfg.buy_twap_col],
        detail[cfg.sell_twap_col],
        detail["bridge_prev_close"],
        buy_bps=buy_bps,
        sell_bps=sell_bps,
    )
    detail["trade_date"] = pd.to_datetime(trade_day)
    detail["trade_day"] = pd.to_datetime(trade_day)
    detail["next_day"] = pd.to_datetime(next_day)
    detail["buy_price"] = detail[cfg.buy_twap_col]
    detail["sell_price"] = detail[cfg.sell_twap_col]
    detail["buy_leg_ret_net"] = buy_leg
    detail["sell_leg_ret_net"] = sell_leg
    detail["full_cycle_ret_net"] = full
    detail["return_net"] = full
    detail["weighted_return"] = pd.to_numeric(detail["weight"], errors="coerce") * full
    detail["weighted_buy_leg_ret_net"] = pd.to_numeric(detail["weight"], errors="coerce") * buy_leg
    detail["weighted_sell_leg_ret_net"] = pd.to_numeric(detail["weight"], errors="coerce") * sell_leg
    detail["buy_cost_bps"] = float(buy_bps)
    detail["sell_cost_bps"] = float(sell_bps)
    return detail.sort_values("code", kind="mergesort").reset_index(drop=True)


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


def compute_benchmark_breakdowns_for_days(
    *,
    raw_data_root: str | Path,
    trade_days: Iterable[date],
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
    skip_failed_days: bool = False,
) -> pd.DataFrame:
    cfg = pool_cfg or load_benchmark_pool_config()
    days = sorted(set(trade_days))
    if not days:
        return pd.DataFrame()
    try:
        next_day_map = _build_next_day_map(raw_data_root=raw_data_root, trade_days=days)
    except Exception:
        if not skip_failed_days:
            raise
        return pd.DataFrame()

    rows: list[dict] = []
    nav = 1.0
    for day in days:
        try:
            next_day = next_day_map[day]
            detail = compute_benchmark_detail_for_day(
                raw_data_root=raw_data_root,
                trade_day=day,
                next_day=next_day,
                buy_bps=buy_bps,
                sell_bps=sell_bps,
                pool_cfg=cfg,
            )
        except Exception:
            if not skip_failed_days:
                raise
            continue

        weight = pd.to_numeric(detail["weight"], errors="coerce").fillna(0.0)
        benchmark_return = float(pd.to_numeric(detail["weighted_return"], errors="coerce").sum())
        buy_net_return = float(pd.to_numeric(detail["weighted_buy_leg_ret_net"], errors="coerce").sum())
        sell_net_return = float(pd.to_numeric(detail["weighted_sell_leg_ret_net"], errors="coerce").sum())
        nav *= 1.0 + benchmark_return
        rows.append(
            {
                "trade_date": day,
                "trade_day": day,
                "next_day": next_day,
                "benchmark_return": benchmark_return,
                "full_cycle_ret_net": benchmark_return,
                "buy_leg_ret_net": buy_net_return,
                "sell_leg_ret_net": sell_net_return,
                "strict_buy_leg_net_return": buy_net_return,
                "strict_sell_leg_net_return": sell_net_return,
                "buy_fee_weighted": float(weight.sum() * bps_to_rate(buy_bps)),
                "strict_sell_fee_weighted": float(weight.sum() * bps_to_rate(sell_bps)),
                "count": int(detail["code"].nunique()),
                "buy_count": int(detail["code"].nunique()),
                "sell_count": int(detail["code"].nunique()),
                "strict_sell_count": int(detail["code"].nunique()),
                "fallback_sell_codes": 0,
                "fallback_sell_weight": 0.0,
                "buy_cost_bps_mean": float(buy_bps),
                "sell_cost_bps_mean": float(sell_bps),
                "nav": float(nav),
                "benchmark_method": "strict_full_overnight_net",
                "buy_twap_col": cfg.buy_twap_col,
                "sell_twap_col": cfg.sell_twap_col,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


def compute_benchmark_breakdown_for_day(
    *,
    raw_data_root: str | Path,
    trade_day: date,
    next_day: date,
    buy_bps: float,
    sell_bps: float,
    pool_cfg: BenchmarkPoolConfig | None = None,
) -> BenchmarkReturnBreakdown:
    detail = compute_benchmark_detail_for_day(
        raw_data_root=raw_data_root,
        trade_day=trade_day,
        next_day=next_day,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
    )
    if detail.empty:
        raise RuntimeError(f"benchmark failed for day: {trade_day:%Y-%m-%d}")
    return BenchmarkReturnBreakdown(
        full_cycle_ret_net=float(pd.to_numeric(detail["weighted_return"], errors="coerce").sum()),
        buy_leg_ret_net=float(pd.to_numeric(detail["weighted_buy_leg_ret_net"], errors="coerce").sum()),
        sell_leg_ret_net=float(pd.to_numeric(detail["weighted_sell_leg_ret_net"], errors="coerce").sum()),
        count=int(detail["code"].nunique()),
        sell_count=int(detail["code"].nunique()),
        fallback_sell_codes=0,
        fallback_sell_weight=0.0,
    )


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
    frame = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_data_root,
        trade_days=trade_days,
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=cfg,
        skip_failed_days=skip_failed_days,
    )
    if frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        pd.to_numeric(frame["benchmark_return"], errors="coerce").values,
        index=pd.to_datetime(frame["trade_date"], errors="coerce").dt.date,
        dtype=float,
    ).dropna().sort_index()
