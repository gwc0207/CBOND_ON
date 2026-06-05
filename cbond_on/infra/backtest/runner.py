from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.core.utils import progress
from cbond_on.infra.benchmark.service import (
    compute_benchmark_breakdown_for_day,
    compute_strict_sell_detail_for_holdings,
    load_strict_market_day,
)
from cbond_on.infra.data.io import read_table_range, read_trading_calendar, iter_clean_dates
from cbond_on.infra.universe.pool_filter import (
    apply_pool_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.infra.model.score_io import load_scores_by_date


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


def _build_runner_strict_holdings(picks: pd.DataFrame, *, trade_day: date, weight: float) -> pd.DataFrame:
    if picks.empty:
        return pd.DataFrame()
    out = picks.copy()
    out["weight"] = float(weight)
    out["buy_trade_day"] = pd.to_datetime(trade_day)
    out["close_price"] = pd.to_numeric(out["buy_close_price"], errors="coerce")
    out["prev_close_price"] = out["close_price"]
    out["buy_weight_base"] = pd.to_numeric(out["weight"], errors="coerce")
    out["weighted_buy_leg_ret_net"] = (
        pd.to_numeric(out["weight"], errors="coerce")
        * pd.to_numeric(out["buy_leg_ret_net"], errors="coerce")
    )
    return out


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


def _read_price_daily(raw_data_root: str, day: date) -> pd.DataFrame:
    df = read_table_range(raw_data_root, "market_cbond.daily_price", day, day)
    if df.empty:
        return df
    if "code" not in df.columns and "instrument_code" in df.columns and "exchange_code" in df.columns:
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
    fee_bps: float,
    buy_slippage_bps: float,
    sell_slippage_bps: float,
    bin_source: str,
    bin_top_k: int,
    bin_lookback_days: int,
    min_count: int,
    max_weight: float,
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
    buy_cost_bps = max(0.0, float(fee_bps) - float(buy_slippage_bps))
    sell_cost_bps = max(0.0, float(fee_bps) - float(sell_slippage_bps))
    bin_source = str(bin_source or "auto").lower()
    bin_top_k = max(1, int(bin_top_k))
    bin_lookback_days = max(1, int(bin_lookback_days))
    live_bin_source = str(live_bin_source or bin_source).lower()
    live_bin_top_k = max(1, int(live_bin_top_k if live_bin_top_k is not None else bin_top_k))
    live_bin_lookback_days = max(
        1, int(live_bin_lookback_days if live_bin_lookback_days is not None else bin_lookback_days)
    )
    pool_cfg = load_upstream_pool_config()
    bin_history: list[dict[int, float]] = []
    strategy_prev_holdings = pd.DataFrame()
    live_prev_holdings = pd.DataFrame()

    for i in progress(range(len(day_list) - 1), desc="backtest", unit="day", total=len(day_list) - 1):
        day = day_list[i]
        score_df = scores.get(day, pd.DataFrame())
        if score_df.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue

        try:
            merged = load_strict_market_day(
                raw_data_root=raw_data_root,
                trade_day=day,
                buy_bps=buy_cost_bps,
                sell_bps=sell_cost_bps,
            )
        except Exception as exc:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": f"missing_strict_market:{exc}"})
            continue

        benchmark_breakdown = compute_benchmark_breakdown_for_day(
            raw_data_root=raw_data_root,
            trade_day=day,
            next_day=day,
            buy_bps=buy_cost_bps,
            sell_bps=sell_cost_bps,
        )
        benchmark_return = float(benchmark_breakdown.full_cycle_ret_net)

        merged = merged.merge(score_df[["code", "score"]], on="code", how="left")
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_data_root,
            trade_day=day,
            pool_cfg=pool_cfg,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "[pool_filter] required pool is unavailable; no-filter fallback is disabled: "
                f"trade_day={day:%Y-%m-%d} "
                f"expected_pool_day={pool_info.get('pool_day_expected')} "
                f"reason={pool_info.get('fallback_reason')} "
                f"nearest_pool_day={pool_info.get('nearest_pool_day')}"
            )
        merged = apply_pool_filter_to_universe(merged, pool_codes=pool_codes)
        if merged.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "empty_pool_filtered_universe"})
            continue

        required = ["buy_price", "buy_close_price", "buy_leg_ret_net"]
        missing_cols = [c for c in required if c not in merged.columns]
        if missing_cols:
            raise KeyError(f"missing strict market columns: {missing_cols}")

        merged = merged[merged["score"].notna()]
        merged = merged[
            pd.to_numeric(merged["buy_price"], errors="coerce").notna()
            & pd.to_numeric(merged["buy_close_price"], errors="coerce").notna()
            & (pd.to_numeric(merged["buy_price"], errors="coerce") > 0)
            & (pd.to_numeric(merged["buy_close_price"], errors="coerce") > 0)
        ]
        if merged.empty:
            diagnostics.append({"trade_date": day, "status": "skip", "reason": "no_tradable"})
            continue

        returns_all = pd.to_numeric(merged["buy_leg_ret_net"], errors="coerce")
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
            w_live = min(1.0 / live_count, max_weight)
            live_buy_holdings = _build_runner_strict_holdings(picks_live, trade_day=day, weight=w_live)
            live_avg_return = float(pd.to_numeric(live_buy_holdings["buy_leg_ret_net"], errors="coerce").mean())
            live_day_buy_leg_ret = float(
                pd.to_numeric(live_buy_holdings["weighted_buy_leg_ret_net"], errors="coerce").sum()
            )
            live_day_sell_leg_ret = 0.0
            if not live_prev_holdings.empty:
                live_sell_detail = compute_strict_sell_detail_for_holdings(
                    raw_data_root=raw_data_root,
                    sell_day=day,
                    prev_holdings=live_prev_holdings,
                    sell_bps=sell_cost_bps,
                )
                if not live_sell_detail.empty:
                    live_day_sell_leg_ret = float(
                        pd.to_numeric(live_sell_detail["weighted_sell_leg_ret_net"], errors="coerce").sum()
                    )
            live_day_return = live_day_sell_leg_ret + live_day_buy_leg_ret
            live_total_weight = float(w_live * live_count)
            live_prev_holdings = live_buy_holdings
        else:
            live_day_buy_leg_ret = float("nan")
            live_day_sell_leg_ret = float("nan")

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

        weight = min(1.0 / len(picks), max_weight)
        buy_holdings = _build_runner_strict_holdings(picks, trade_day=day, weight=weight)
        returns = pd.to_numeric(buy_holdings["buy_leg_ret_net"], errors="coerce")
        day_buy_leg_ret = float(pd.to_numeric(buy_holdings["weighted_buy_leg_ret_net"], errors="coerce").sum())
        day_sell_leg_ret = 0.0
        if not strategy_prev_holdings.empty:
            sell_detail = compute_strict_sell_detail_for_holdings(
                raw_data_root=raw_data_root,
                sell_day=day,
                prev_holdings=strategy_prev_holdings,
                sell_bps=sell_cost_bps,
            )
            if not sell_detail.empty:
                day_sell_leg_ret = float(pd.to_numeric(sell_detail["weighted_sell_leg_ret_net"], errors="coerce").sum())
        day_return = day_sell_leg_ret + day_buy_leg_ret
        total_weight = float(weight * len(picks))

        daily_records.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "avg_return": float(returns.mean()),
                "day_return": day_return,
                "full_cycle_ret_net": day_return,
                "buy_leg_ret_net": day_buy_leg_ret,
                "sell_leg_ret_net": day_sell_leg_ret,
                "benchmark_return": float(benchmark_breakdown.full_cycle_ret_net),
                "benchmark_full_cycle_ret_net": float(benchmark_breakdown.full_cycle_ret_net),
                "benchmark_buy_leg_ret_net": float(benchmark_breakdown.buy_leg_ret_net),
                "benchmark_sell_leg_ret_net": float(benchmark_breakdown.sell_leg_ret_net),
                "total_weight": total_weight,
                "live_count": live_count,
                "live_avg_return": live_avg_return,
                "live_day_return": live_day_return,
                "live_buy_leg_ret_net": live_day_buy_leg_ret,
                "live_sell_leg_ret_net": live_day_sell_leg_ret,
                "live_total_weight": live_total_weight,
            }
        )
        for idx, row in buy_holdings.iterrows():
            position_records.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "score": float(row["score"]),
                    "weight": weight,
                    "buy_price": float(row["buy_price"]),
                    "sell_price": float("nan"),
                    "bridge_prev_close": float(row["buy_close_price"]),
                    "buy_leg_ret_net": float(row["buy_leg_ret_net"]),
                    "sell_leg_ret_net": float("nan"),
                    "full_cycle_ret_net": float(row["buy_leg_ret_net"]),
                    "return": float(row["buy_leg_ret_net"]),
                }
            )
        strategy_prev_holdings = buy_holdings
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

