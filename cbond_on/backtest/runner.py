from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.universe import filter_tradable
from cbond_on.core.utils import progress
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
    bin_dir_acc: pd.DataFrame | None = None


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
    bin_source: str,
    bin_top_k: int,
    bin_lookback_days: int,
    min_count: int,
    max_weight: float,
    filter_tradable_flag: bool,
    min_amount: float,
    min_volume: float,
    ic_bins: int,
    live_bin_source: str | None = None,
    live_bin_top_k: int | None = None,
    live_bin_lookback_days: int | None = None,
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
    bin_dir_records: list[dict] = []
    cost_bps = float(twap_bps) + float(fee_bps)
    bin_source = str(bin_source or "auto").lower()
    bin_top_k = max(1, int(bin_top_k))
    bin_lookback_days = max(1, int(bin_lookback_days))
    live_bin_source = str(live_bin_source or bin_source).lower()
    live_bin_top_k = max(1, int(live_bin_top_k if live_bin_top_k is not None else bin_top_k))
    live_bin_lookback_days = max(
        1, int(live_bin_lookback_days if live_bin_lookback_days is not None else bin_lookback_days)
    )
    bin_history: list[dict[int, float]] = []

    for i in progress(range(len(day_list) - 1), desc="backtest", unit="day", total=len(day_list) - 1):
        day = day_list[i]
        next_day = day_list[i + 1]
        sell_next_col = f"{sell_twap_col}_next"
        score_df = scores.get(day, pd.DataFrame())
        if score_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue

        buy_df = _read_twap_daily(raw_data_root, day)
        sell_df = _read_twap_daily(raw_data_root, next_day)
        if buy_df.empty or sell_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_clean"})
            continue

        # Benchmark universe must be independent of model score availability.
        bench_df = buy_df.merge(
            sell_df[["code", sell_twap_col]].rename(columns={sell_twap_col: sell_next_col}),
            on="code",
            how="inner",
        )
        if filter_tradable_flag:
            bench_df = filter_tradable(
                bench_df,
                buy_twap_col=buy_twap_col,
                sell_twap_col=sell_next_col,
                min_amount=min_amount,
                min_volume=min_volume,
            )
        else:
            required = [buy_twap_col, sell_next_col]
            missing_cols = [c for c in required if c not in bench_df.columns]
            if missing_cols:
                raise KeyError(f"missing twap columns: {missing_cols}")
        if bench_df.empty:
            benchmark_return = 0.0
        else:
            bench_buy = apply_twap_bps(bench_df[buy_twap_col], cost_bps, side="buy")
            bench_sell = apply_twap_bps(bench_df[sell_next_col], cost_bps, side="sell")
            benchmark_return = float(((bench_sell - bench_buy) / bench_buy).mean())

        merged = buy_df.merge(
            sell_df[["code", sell_twap_col]].rename(columns={sell_twap_col: sell_next_col}),
            on="code",
            how="left",
        )
        merged = merged.merge(score_df[["code", "score"]], on="code", how="left")

        if filter_tradable_flag:
            merged = filter_tradable(
                merged,
                buy_twap_col=buy_twap_col,
                sell_twap_col=sell_next_col,
                min_amount=min_amount,
                min_volume=min_volume,
            )
        else:
            required = [buy_twap_col, sell_next_col]
            missing_cols = [c for c in required if c not in merged.columns]
            if missing_cols:
                raise KeyError(f"missing twap columns: {missing_cols}")

        merged = merged[merged["score"].notna()]
        if merged.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "no_tradable"})
            continue

        buy_all = apply_twap_bps(merged[buy_twap_col], cost_bps, side="buy")
        sell_all = apply_twap_bps(merged[sell_next_col], cost_bps, side="sell")
        returns_all = (sell_all - buy_all) / buy_all
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

        bins_cat = None
        day_bin_means: dict[int, float] = {}
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
                    dir_acc = float((group["return"] > 0).mean()) if len(group) else float("nan")
                    bin_records.append(
                        {
                            "bin": int(bin_id),
                            "score_mean": float(group["score"].mean()),
                            "return_mean": float(group["return"].mean()),
                            "dir_acc": dir_acc,
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
                    bin_dir_records.append(
                        {
                            "trade_date": day,
                            "bin": int(bin_id),
                            "dir_acc": dir_acc,
                        }
                    )
                if not bin_df.empty:
                    bin_means = bin_df.groupby("bin")["return"].mean().to_dict()
                    day_bin_means = {int(k): float(v) for k, v in bin_means.items()}

        if bins_cat is None:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "binning_failed"})
            continue
        merged = merged.copy()
        merged["bin"] = bins_cat.values
        available_bins = sorted(merged["bin"].dropna().unique().tolist())
        if not available_bins:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "binning_failed"})
            continue

        # Live-style picks in backtest: use live config bin mode + lookback history.
        if live_bin_source == "auto" and bin_history:
            live_lookback = (
                bin_history[-live_bin_lookback_days:]
                if len(bin_history) > live_bin_lookback_days
                else bin_history
            )
            live_agg: dict[int, list[float]] = {}
            for rec in live_lookback:
                for b, v in rec.items():
                    live_agg.setdefault(int(b), []).append(float(v))
            live_ranked = sorted(
                [(b, float(pd.Series(v).mean())) for b, v in live_agg.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            live_chosen_bins = (
                [b for b, _ in live_ranked][:live_bin_top_k]
                if live_ranked
                else sorted(available_bins, reverse=True)[:live_bin_top_k]
            )
            live_chosen_bins = [b for b in live_chosen_bins if b in set(available_bins)]
            if not live_chosen_bins:
                live_chosen_bins = sorted(available_bins, reverse=True)[:live_bin_top_k]
        else:
            live_chosen_bins = sorted(available_bins, reverse=True)[:live_bin_top_k]
        picks_live = merged[merged["bin"].isin(live_chosen_bins)].copy()
        live_avg_return = float("nan")
        live_day_return = float("nan")
        live_total_weight = float("nan")
        live_count = int(len(picks_live))
        if live_count > 0:
            buy_live = apply_twap_bps(picks_live[buy_twap_col], cost_bps, side="buy")
            sell_live = apply_twap_bps(picks_live[sell_next_col], cost_bps, side="sell")
            returns_live = (sell_live - buy_live) / buy_live
            live_avg_return = float(returns_live.mean()) if not returns_live.empty else float("nan")
            w_live = min(1.0 / live_count, max_weight)
            live_day_return = float((returns_live * w_live).sum())
            live_total_weight = float(w_live * live_count)

        if bin_source == "auto" and bin_history:
            lookback = bin_history[-bin_lookback_days:] if len(bin_history) > bin_lookback_days else bin_history
            agg: dict[int, list[float]] = {}
            for rec in lookback:
                for b, v in rec.items():
                    agg.setdefault(b, []).append(v)
            ranked = sorted(
                [(b, float(pd.Series(v).mean())) for b, v in agg.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            chosen_bins = [b for b, _ in ranked][:bin_top_k] if ranked else sorted(available_bins, reverse=True)[:bin_top_k]
        else:
            chosen_bins = sorted(available_bins, reverse=True)[:bin_top_k]

        # Align historical chosen bins with today's actually available bins.
        effective_bins = [b for b in chosen_bins if b in set(available_bins)]
        if not effective_bins:
            effective_bins = sorted(available_bins, reverse=True)[:bin_top_k]

        picks = merged[merged["bin"].isin(effective_bins)]
        if len(picks) < min_count:
            if day_bin_means:
                bin_history.append(day_bin_means)
            daily_records.append(
                {
                    "trade_date": day,
                    "count": int(len(picks)),
                    "avg_return": float("nan"),
                    "day_return": float("nan"),
                    "benchmark_return": benchmark_return,
                    "total_weight": float("nan"),
                    "live_count": live_count,
                    "live_avg_return": live_avg_return,
                    "live_day_return": live_day_return,
                    "live_total_weight": live_total_weight,
                }
            )
            diagnostics.append(
                {
                    "trade_date": day,
                    "status": "skip",
                    "reason": "min_count_not_met",
                    "bins_actual": int(len(available_bins)),
                    "chosen_bins": ",".join(str(int(x)) for x in effective_bins),
                    "picks_count": int(len(picks)),
                    "min_count": int(min_count),
                }
            )
            continue

        buy_px = apply_twap_bps(picks[buy_twap_col], cost_bps, side="buy")
        sell_px = apply_twap_bps(picks[sell_next_col], cost_bps, side="sell")
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
                "live_count": live_count,
                "live_avg_return": live_avg_return,
                "live_day_return": live_day_return,
                "live_total_weight": live_total_weight,
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
        diagnostics.append(
            {
                "trade_date": day,
                "status": "ok",
                "reason": "",
                "bins_actual": int(len(available_bins)),
                "chosen_bins": ",".join(str(int(x)) for x in effective_bins),
                "picks_count": int(len(picks)),
                "min_count": int(min_count),
            }
        )

        if day_bin_means:
            bin_history.append(day_bin_means)

    if daily_records:
        daily_df = pd.DataFrame(daily_records).sort_values("trade_date")
        bt_nav = (1.0 + daily_df["day_return"].fillna(0.0)).cumprod()
        live_nav = (
            (1.0 + daily_df["live_day_return"].fillna(0.0)).cumprod()
            if "live_day_return" in daily_df.columns
            else pd.Series([1.0] * len(daily_df), index=daily_df.index)
        )
        nav_df = pd.DataFrame(
            {
                "trade_date": daily_df["trade_date"],
                "nav": bt_nav,
                "live_nav": live_nav,
            }
        )
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
    if bin_dir_records:
        bin_dir_df = pd.DataFrame(bin_dir_records)
        if not bin_dir_df.empty:
            result.bin_dir_acc = bin_dir_df.sort_values(["trade_date", "bin"])
    if diagnostics:
        result.diagnostics = pd.DataFrame(diagnostics).sort_values("trade_date")
    return result
