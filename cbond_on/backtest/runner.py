from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_clean_daily, read_trading_calendar, iter_clean_dates
from cbond_on.models.score_io import load_scores_by_date
from .execution import apply_twap_bps


@dataclass
class BacktestResult:
    days: int = 0
    daily_returns: pd.DataFrame | None = None
    nav_curve: pd.DataFrame | None = None
    positions: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None


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
    cost_bps = float(twap_bps) + float(fee_bps)

    for i in range(len(day_list) - 1):
        day = day_list[i]
        next_day = day_list[i + 1]
        score_df = scores.get(day, pd.DataFrame())
        if score_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue

        buy_df = read_clean_daily(clean_data_root, day)
        sell_df = read_clean_daily(clean_data_root, next_day)
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
    if diagnostics:
        result.diagnostics = pd.DataFrame(diagnostics).sort_values("trade_date")
    return result
