from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.config import load_config_file
from cbond_on.core.trading_days import list_available_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.core.universe import normalize_price_bound
from cbond_on.infra.backtest.execution import bps_to_rate
from cbond_on.infra.data.io import read_table_range
from cbond_on.infra.io.market_twap import read_price_daily, read_twap_daily
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    resolve_pool_codes_for_trade_day,
)


CLOSE_PRICE_CANDIDATES = ("close", "close_price", "close_px", "closePrice")
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
        method=str(raw.get("method", "strict_official_prev_close")),
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


def _build_buy_holdings_for_day(
    *,
    raw_data_root: str | Path,
    buy_day: date,
    buy_bps: float,
    cfg: BenchmarkPoolConfig,
) -> pd.DataFrame:
    target = _load_target_pool_for_buy_day(
        raw_data_root=raw_data_root,
        buy_day=buy_day,
        cfg=cfg,
    )
    buy_df = _read_benchmark_twap_daily(raw_data_root, buy_day, cfg)
    if buy_df.empty:
        raise RuntimeError(f"benchmark buy TWAP missing: {buy_day:%Y-%m-%d}")
    if cfg.buy_twap_col not in buy_df.columns:
        raise RuntimeError(f"benchmark buy TWAP column missing: {cfg.buy_twap_col}")
    buy = _ensure_code_column(buy_df, label="benchmark buy TWAP")

    price_df = read_price_daily(str(raw_data_root), buy_day)
    close = _first_price_series(price_df, CLOSE_PRICE_CANDIDATES, label="close_price").rename("close_price")
    holdings = (
        target.merge(buy[["code", cfg.buy_twap_col]], on="code", how="inner")
        .merge(close.reset_index(), on="code", how="inner")
    )
    holdings[cfg.buy_twap_col] = pd.to_numeric(holdings[cfg.buy_twap_col], errors="coerce")
    holdings["close_price"] = pd.to_numeric(holdings["close_price"], errors="coerce")
    holdings = holdings[
        holdings[cfg.buy_twap_col].notna()
        & holdings["close_price"].notna()
        & (holdings[cfg.buy_twap_col] > 0)
        & (holdings["close_price"] > 0)
    ].copy()
    if holdings.empty:
        raise RuntimeError(f"benchmark buy holdings empty: buy_day={buy_day:%Y-%m-%d}")

    weights = _normalize_weight_series(holdings["target_weight"])
    holdings["weight"] = weights.reindex(holdings.index).fillna(0.0) if not weights.empty else 1.0 / len(holdings)
    holdings["buy_price"] = pd.to_numeric(holdings[cfg.buy_twap_col], errors="coerce")
    holdings["buy_cost_bps"] = float(buy_bps)
    holdings["buy_leg_ret_gross"] = holdings["close_price"] / holdings["buy_price"] - 1.0
    holdings["prev_close_price"] = holdings["close_price"]
    return holdings[
        [
            "pool_day",
            "buy_trade_day",
            "code",
            "weight",
            "target_weight",
            "buy_price",
            "close_price",
            "prev_close_price",
            "buy_cost_bps",
            "buy_leg_ret_gross",
        ]
    ].sort_values("code", kind="mergesort").reset_index(drop=True)


def _to_strict_sell_holdings(buy_holdings: pd.DataFrame) -> pd.DataFrame:
    holdings = buy_holdings.copy()
    close_value = (
        pd.to_numeric(holdings["weight"], errors="coerce")
        * pd.to_numeric(holdings["close_price"], errors="coerce")
        / pd.to_numeric(holdings["buy_price"], errors="coerce")
    )
    weights = _normalize_weight_series(close_value)
    holdings["weight"] = weights.reindex(holdings.index).fillna(0.0) if not weights.empty else 0.0
    holdings["prev_close_price"] = pd.to_numeric(holdings["close_price"], errors="coerce")
    return holdings


def _compute_strict_sell_detail_for_day(
    *,
    raw_data_root: str | Path,
    sell_day: date,
    sell_bps: float,
    prev_holdings: pd.DataFrame,
    cfg: BenchmarkPoolConfig,
) -> pd.DataFrame:
    if prev_holdings is None or prev_holdings.empty:
        raise RuntimeError(f"benchmark previous holdings empty: sell_day={sell_day:%Y-%m-%d}")
    sell_df = _read_benchmark_twap_daily(raw_data_root, sell_day, cfg)
    if sell_df.empty or cfg.sell_twap_col not in sell_df.columns:
        sell = pd.DataFrame({"code": pd.Series(dtype=str), cfg.sell_twap_col: pd.Series(dtype=float)})
    else:
        sell = _ensure_code_column(sell_df, label="benchmark sell TWAP")[["code", cfg.sell_twap_col]].copy()

    price_df = read_price_daily(str(raw_data_root), sell_day)
    official_prev_close = _first_price_series(
        price_df,
        PREV_CLOSE_PRICE_CANDIDATES,
        label="strict_prev_close_price",
    ).rename("strict_prev_close_price")

    detail = (
        prev_holdings.merge(official_prev_close.reset_index(), on="code", how="inner")
        .merge(sell, on="code", how="left")
    )
    detail[cfg.sell_twap_col] = pd.to_numeric(detail[cfg.sell_twap_col], errors="coerce")
    detail["strict_prev_close_price"] = pd.to_numeric(detail["strict_prev_close_price"], errors="coerce")
    detail = detail[detail["strict_prev_close_price"].notna() & (detail["strict_prev_close_price"] > 0)].copy()
    if detail.empty:
        raise RuntimeError(f"benchmark strict sell detail empty: sell_day={sell_day:%Y-%m-%d}")

    sell_valid = detail[cfg.sell_twap_col].notna() & (detail[cfg.sell_twap_col] > 0)
    detail["strict_sell_price"] = detail[cfg.sell_twap_col].where(sell_valid, detail["strict_prev_close_price"])
    detail["sell_missing_fallback"] = ~sell_valid
    detail["sell_trade_day"] = pd.to_datetime(sell_day)
    detail["sell_cost_bps"] = float(sell_bps)
    sell_rate = bps_to_rate(sell_bps)
    detail["strict_sell_leg_gross_ret"] = detail["strict_sell_price"] / detail["strict_prev_close_price"] - 1.0
    detail["strict_sell_leg_net_ret"] = (
        detail["strict_sell_price"] * (1.0 - sell_rate) / detail["strict_prev_close_price"] - 1.0
    )
    detail["strict_sell_weighted_gross"] = (
        pd.to_numeric(detail["weight"], errors="coerce") * pd.to_numeric(detail["strict_sell_leg_gross_ret"], errors="coerce")
    )
    detail["strict_sell_weighted_net"] = (
        pd.to_numeric(detail["weight"], errors="coerce") * pd.to_numeric(detail["strict_sell_leg_net_ret"], errors="coerce")
    )
    detail["strict_sell_fee_weighted"] = pd.to_numeric(detail["weight"], errors="coerce") * sell_rate
    return detail.sort_values("code", kind="mergesort").reset_index(drop=True)


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

    prev_days = prev_trading_days_from_raw(
        raw_data_root,
        days[0],
        1,
        kind="snapshot",
        asset=cfg.pool_asset,
    )
    if not prev_days:
        if not skip_failed_days:
            raise RuntimeError(f"benchmark previous trading day missing for: {days[0]:%Y-%m-%d}")
        all_days = list_available_trading_days_from_raw(raw_data_root, kind="snapshot", asset=cfg.pool_asset)
        day_set = set(days)
        start_idx = next((i for i, day in enumerate(all_days) if day in day_set and i > 0), None)
        if start_idx is None:
            return pd.DataFrame()
        days = [day for day in days if day >= all_days[start_idx]]
        prev_day = all_days[start_idx - 1]
    else:
        prev_day = prev_days[-1]

    try:
        prev_holdings = _to_strict_sell_holdings(
            _build_buy_holdings_for_day(
                raw_data_root=raw_data_root,
                buy_day=prev_day,
                buy_bps=buy_bps,
                cfg=cfg,
            )
        )
    except Exception:
        if not skip_failed_days:
            raise
        prev_holdings = pd.DataFrame()

    rows: list[dict] = []
    nav = 1.0
    for day in days:
        try:
            if prev_holdings.empty:
                prev_day_candidates = prev_trading_days_from_raw(
                    raw_data_root,
                    day,
                    1,
                    kind="snapshot",
                    asset=cfg.pool_asset,
                )
                if not prev_day_candidates:
                    raise RuntimeError(f"benchmark previous trading day missing for: {day:%Y-%m-%d}")
                prev_holdings = _to_strict_sell_holdings(
                    _build_buy_holdings_for_day(
                        raw_data_root=raw_data_root,
                        buy_day=prev_day_candidates[-1],
                        buy_bps=buy_bps,
                        cfg=cfg,
                    )
                )

            sell_detail = _compute_strict_sell_detail_for_day(
                raw_data_root=raw_data_root,
                sell_day=day,
                sell_bps=sell_bps,
                prev_holdings=prev_holdings,
                cfg=cfg,
            )
            buy_holdings = _build_buy_holdings_for_day(
                raw_data_root=raw_data_root,
                buy_day=day,
                buy_bps=buy_bps,
                cfg=cfg,
            )
        except Exception:
            if not skip_failed_days:
                raise
            continue

        buy_weight = pd.to_numeric(buy_holdings["weight"], errors="coerce").fillna(0.0)
        buy_gross = pd.to_numeric(buy_holdings["buy_leg_ret_gross"], errors="coerce")
        buy_cost_rate = bps_to_rate(buy_bps)
        buy_net_by_bond = buy_gross - buy_cost_rate
        buy_net_return = float((buy_weight * buy_net_by_bond).sum())
        buy_gross_return = float((buy_weight * buy_gross).sum())
        buy_fee_weighted = float((buy_weight * buy_cost_rate).sum())

        sell_weight = pd.to_numeric(sell_detail["weight"], errors="coerce").fillna(0.0)
        sell_net_return = float(pd.to_numeric(sell_detail["strict_sell_weighted_net"], errors="coerce").sum())
        sell_gross_return = float(pd.to_numeric(sell_detail["strict_sell_weighted_gross"], errors="coerce").sum())
        sell_fee_weighted = float(pd.to_numeric(sell_detail["strict_sell_fee_weighted"], errors="coerce").sum())
        fallback_mask = sell_detail["sell_missing_fallback"].astype(bool)

        benchmark_return = float(buy_net_return + sell_net_return)
        nav *= 1.0 + benchmark_return
        rows.append(
            {
                "trade_date": day,
                "trade_day": day,
                "benchmark_return": benchmark_return,
                "full_cycle_ret_net": benchmark_return,
                "buy_leg_ret_net": buy_net_return,
                "sell_leg_ret_net": sell_net_return,
                "strict_buy_leg_net_return": buy_net_return,
                "strict_buy_leg_gross_return": buy_gross_return,
                "strict_sell_leg_net_return": sell_net_return,
                "strict_sell_leg_gross_return": sell_gross_return,
                "buy_fee_weighted": buy_fee_weighted,
                "strict_sell_fee_weighted": sell_fee_weighted,
                "count": int(buy_holdings["code"].nunique()),
                "buy_count": int(buy_holdings["code"].nunique()),
                "sell_count": int(sell_detail["code"].nunique()),
                "strict_sell_count": int(sell_detail["code"].nunique()),
                "fallback_sell_codes": int(fallback_mask.sum()),
                "fallback_sell_weight": float(sell_weight.loc[fallback_mask].sum()),
                "buy_cost_bps_mean": float(buy_bps),
                "sell_cost_bps_mean": float(sell_bps),
                "nav": float(nav),
                "benchmark_method": "strict_official_prev_close",
                "buy_twap_col": cfg.buy_twap_col,
                "sell_twap_col": cfg.sell_twap_col,
            }
        )
        prev_holdings = _to_strict_sell_holdings(buy_holdings)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)


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
    frame = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_data_root,
        trade_days=[trade_day],
        buy_bps=buy_bps,
        sell_bps=sell_bps,
        pool_cfg=pool_cfg,
        skip_failed_days=False,
    )
    if frame.empty:
        raise RuntimeError(f"benchmark failed for day: {trade_day:%Y-%m-%d}")
    row = frame.iloc[0]
    return BenchmarkReturnBreakdown(
        full_cycle_ret_net=float(row["benchmark_return"]),
        buy_leg_ret_net=float(row["buy_leg_ret_net"]),
        sell_leg_ret_net=float(row["sell_leg_ret_net"]),
        count=int(row["buy_count"]),
        sell_count=int(row["sell_count"]),
        fallback_sell_codes=int(row["fallback_sell_codes"]),
        fallback_sell_weight=float(row["fallback_sell_weight"]),
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
