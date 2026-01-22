from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_table_range, read_trading_calendar, iter_clean_dates
from cbond_on.models.score_io import load_scores_by_date
from .execution import apply_twap_bps


@dataclass
class BacktestResult:
    days: int = 0
    daily_returns: pd.DataFrame | None = None
    nav_curve: pd.DataFrame | None = None
    positions: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None
    ic_series: pd.DataFrame | None = None
    bin_stats: pd.DataFrame | None = None
    bin_nav: pd.DataFrame | None = None


def _rank_ic(x: pd.Series, y: pd.Series) -> float:
    x_rank = x.rank(method="average")
    y_rank = y.rank(method="average")
    return float(x_rank.corr(y_rank, method="pearson"))


def _load_trading_days(raw_data_root: str) -> list[date]:
    cal = read_trading_calendar(raw_data_root)
    if cal.empty:
        return []
    if "calendar_date" not in cal.columns:
        raise KeyError("trading_calendar missing calendar_date column")
    if "is_open" in cal.columns:
        cal = cal[cal["is_open"].astype(bool)]
    days = pd.to_datetime(cal["calendar_date"]).dt.date.dropna().unique().tolist()
    days.sort()
    return days


def _infer_trading_days(clean_data_root: str) -> list[date]:
    days = iter_clean_dates(clean_data_root)
    days.sort()
    return days


def _read_twap_daily(raw_data_root: str, day: date) -> pd.DataFrame:
    df = read_table_range(raw_data_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = (
            df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
        )
    return df


def run_backtest(
    *,
    raw_data_root: str,
    clean_data_root: str,
    start: date,
    end: date,
    score_path: str,
    buy_twap_col: str,
    sell_twap_col: str,
    twap_bps: float,
    fee_bps: float,
    top_n: int,
    min_count: int,
    max_weight: float,
    filter_tradable_flag: bool,
    min_amount: float,
    min_volume: float,
    ic_bins: int,
) -> BacktestResult:
    result = BacktestResult()
    scores = load_scores_by_date(score_path)
    trading_days = _load_trading_days(raw_data_root)
    if not trading_days:
        trading_days = _infer_trading_days(clean_data_root)
    if not trading_days:
        raise ValueError("no trading days available")

    day_list = [d for d in trading_days if start <= d <= end]
    if len(day_list) < 2:
        raise ValueError("need at least two trading days for overnight backtest")

    daily_records: list[dict] = []
    position_records: list[dict] = []
    diagnostics: list[dict] = []
    ic_records: list[dict] = []
    bin_records: list[dict] = []
    bin_time_records: list[dict] = []
    cost_bps = float(twap_bps) + float(fee_bps)

    for i in range(len(day_list) - 1):
        day = day_list[i]
        next_day = day_list[i + 1]
        score_df = scores.get(day, pd.DataFrame())
        if score_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue

        buy_df = _read_twap_daily(raw_data_root, day)
        sell_df = _read_twap_daily(raw_data_root, next_day)
        if buy_df.empty or sell_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_clean"})
            continue

        merged = buy_df.merge(
            sell_df[["code", sell_twap_col]],
            on="code",
            how="left",
            suffixes=("", "_next"),
        )
        merged = merged.merge(score_df[["code", "score"]], on="code", how="left")

        if filter_tradable_flag:
            merged = filter_tradable(
                merged,
                buy_twap_col=buy_twap_col,
                sell_twap_col=sell_twap_col,
                min_amount=min_amount,
                min_volume=min_volume,
            )
        else:
            required = [buy_twap_col, sell_twap_col]
            missing_cols = [c for c in required if c not in merged.columns]
            if missing_cols:
                raise KeyError(f"missing twap columns: {missing_cols}")

        merged = merged[merged["score"].notna()]
        if merged.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "no_tradable"})
            continue

        buy_all = apply_twap_bps(merged[buy_twap_col], cost_bps, side="buy")
        sell_all = apply_twap_bps(merged[sell_twap_col], cost_bps, side="sell")
        returns_all = (sell_all - buy_all) / buy_all
        benchmark_return = float(returns_all.mean()) if not returns_all.empty else 0.0
        ic_val = merged["score"].corr(returns_all, method="pearson")
        rank_ic_val = _rank_ic(merged["score"], returns_all)
        ic_records.append(
            {
                "trade_date": day,
                "ic": ic_val,
                "rank_ic": rank_ic_val,
                "count": int(len(merged)),
            }
        )

        if ic_bins > 1:
            try:
                bins_cat = pd.qcut(merged["score"], ic_bins, labels=False, duplicates="drop")
            except ValueError:
                bins_cat = None
            if bins_cat is not None:
                bin_df = pd.DataFrame(
                    {"bin": bins_cat, "score": merged["score"].values, "return": returns_all.values}
                ).dropna()
                grouped = bin_df.groupby("bin", dropna=True)
                for bin_id, group in grouped:
                    bin_records.append(
                        {
                            "bin": int(bin_id),
                            "score_mean": float(group["score"].mean()),
                            "return_mean": float(group["return"].mean()),
                            "count": int(len(group)),
                        }
                    )
                    bin_time_records.append(
                        {
                            "trade_date": day,
                            "bin": int(bin_id),
                            "return_mean": float(group["return"].mean()),
                        }
                    )

        picks = merged.sort_values("score", ascending=False).head(int(top_n))
        if len(picks) < min_count:
            diagnostics.append(
                {"trade_date": day, "status": "skip", "reason": "min_count_not_met"}
            )
            continue

        buy_px = apply_twap_bps(picks[buy_twap_col], cost_bps, side="buy")
        sell_px = apply_twap_bps(picks[sell_twap_col], cost_bps, side="sell")
        returns = (sell_px - buy_px) / buy_px
        weight = min(1.0 / len(picks), max_weight)
        day_return = float((returns * weight).sum())
        total_weight = float(weight * len(picks))

        daily_records.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "avg_return": float(returns.mean()),
                "day_return": day_return,
                "benchmark_return": benchmark_return,
                "total_weight": total_weight,
            }
        )
        for idx, row in picks.iterrows():
            position_records.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "score": float(row["score"]),
                    "weight": weight,
                    "buy_price": float(buy_px.loc[idx]),
                    "sell_price": float(sell_px.loc[idx]),
                    "return": float(returns.loc[idx]),
                }
            )
        result.days += 1
        diagnostics.append({"trade_date": day, "status": "ok", "reason": ""})

    if daily_records:
        daily_df = pd.DataFrame(daily_records).sort_values("trade_date")
        nav = (1.0 + daily_df["day_return"]).cumprod()
        nav_df = pd.DataFrame({"trade_date": daily_df["trade_date"], "nav": nav})
        result.daily_returns = daily_df
        result.nav_curve = nav_df
        result.positions = pd.DataFrame(position_records)
    if ic_records:
        ic_df = pd.DataFrame(ic_records).sort_values("trade_date")
        result.ic_series = ic_df
    if bin_records:
        result.bin_stats = pd.DataFrame(bin_records)
    if bin_time_records:
        bin_time_df = pd.DataFrame(bin_time_records)
        if not bin_time_df.empty:
            pivot = bin_time_df.pivot_table(
                index="trade_date", columns="bin", values="return_mean", aggfunc="mean"
            ).sort_index()
            nav_bins = (1.0 + pivot.fillna(0.0)).cumprod()
            result.bin_nav = nav_bins.reset_index()
    if diagnostics:
        result.diagnostics = pd.DataFrame(diagnostics).sort_values("trade_date")
    return result
