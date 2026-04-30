from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.infra.backtest.execution import (
    apply_twap_bps,
    split_cycle_return_by_bridge,
)
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.core.trading_days import next_trading_days_from_raw
from cbond_on.core.universe import filter_tradable
from cbond_on.infra.benchmark.service import compute_benchmark_breakdown_for_day
from cbond_on.infra.universe.pool_filter import (
    apply_pool_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.domain.portfolio.service import normalize_weights, to_prev_positions
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.infra.backtest.config import load_strategy_config, resolve_score_path
from cbond_on.infra.io.market_twap import iter_open_days, read_price_daily, read_twap_daily
from cbond_on.infra.report.backtest_plot import write_backtest_report_image


@dataclass
class BacktestRunResult:
    out_dir: Path
    days: int

def run(
    *,
    start: date | None = None,
    end: date | None = None,
    cfg: dict | None = None,
) -> BacktestRunResult:
    paths_cfg = load_config_file("paths")
    bt_cfg = dict(cfg or load_config_file("backtest"))

    start_day = parse_date(start or bt_cfg.get("start"))
    end_day = parse_date(end or bt_cfg.get("end"))
    strategy_id = str(bt_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy_cfg = load_strategy_config(
        bt_cfg.get("strategy_config_path"),
        inline=bt_cfg.get("strategy_config"),
    )
    score_path = resolve_score_path(bt_cfg, paths_cfg)
    score_cache = load_scores_by_date(score_path)

    buy_col = str(bt_cfg.get("buy_twap_col", "twap_1442_1457"))
    sell_col = str(bt_cfg.get("sell_twap_col", "twap_0930_0945"))
    buy_cost_bps, sell_cost_bps, fee_source = load_fees_buy_sell_bps()
    print(
        "backtest cost:",
        f"buy_bps={buy_cost_bps:.4f}",
        f"sell_bps={sell_cost_bps:.4f}",
        f"source={fee_source}",
    )
    min_amount = float(bt_cfg.get("min_amount", 0.0))
    min_volume = float(bt_cfg.get("min_volume", 0.0))
    filter_flag = bool(bt_cfg.get("filter_tradable", True))
    pool_cfg = load_upstream_pool_config()

    raw_root = paths_cfg["raw_data_root"]
    days = iter_open_days(raw_root, start_day, end_day)
    tail_day = next_trading_days_from_raw(raw_root, end_day, 1, kind="snapshot", asset="cbond")
    if tail_day:
        days = sorted(set(days + tail_day))
    if len(days) < 2:
        raise ValueError("not enough trading days for backtest")

    daily_rows: list[dict] = []
    pos_rows: list[dict] = []
    diag_rows: list[dict] = []
    ic_rows: list[dict] = []
    prev_positions = pd.DataFrame(columns=["code", "weight"])

    for idx in range(len(days) - 1):
        day = days[idx]
        next_day = days[idx + 1]
        if not (start_day <= day <= end_day):
            continue

        score_df = score_cache.get(day, pd.DataFrame())
        if score_df.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue
        buy_df = read_twap_daily(paths_cfg["raw_data_root"], day)
        sell_df = read_twap_daily(paths_cfg["raw_data_root"], next_day)
        bridge_df = read_price_daily(paths_cfg["raw_data_root"], next_day)
        if buy_df.empty or sell_df.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_twap"})
            continue
        if bridge_df.empty or "prev_close_price" not in bridge_df.columns:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_bridge_price"})
            continue

        merged = buy_df.merge(
            sell_df[["code", sell_col]].rename(columns={sell_col: f"{sell_col}_next"}),
            on="code",
            how="inner",
        )
        merged = merged.merge(
            bridge_df[["code", "prev_close_price"]].rename(columns={"prev_close_price": "bridge_prev_close"}),
            on="code",
            how="inner",
        )
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_root,
            trade_day=day,
            pool_cfg=pool_cfg,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            print(
                "[pool_filter] fallback_no_filter",
                f"trade_day={day:%Y-%m-%d}",
                f"expected_pool_day={pool_info.get('pool_day_expected')}",
                f"reason={pool_info.get('fallback_reason')}",
                f"nearest_pool_day={pool_info.get('nearest_pool_day')}",
            )
        merged = apply_pool_filter_to_universe(merged, pool_codes=pool_codes)
        if merged.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_pool_filtered_universe"})
            continue
        merged = merged.merge(score_df[["code", "score"]], on="code", how="inner")
        if filter_flag:
            merged = filter_tradable(
                merged,
                buy_twap_col=buy_col,
                sell_twap_col=f"{sell_col}_next",
                min_amount=min_amount,
                min_volume=min_volume,
            )
        else:
            merged = merged[
                merged[buy_col].notna()
                & merged[f"{sell_col}_next"].notna()
                & merged["bridge_prev_close"].notna()
                & (merged[buy_col] > 0)
                & (merged[f"{sell_col}_next"] > 0)
                & (merged["bridge_prev_close"] > 0)
            ]
        merged = merged[
            merged["bridge_prev_close"].notna() & (pd.to_numeric(merged["bridge_prev_close"], errors="coerce") > 0)
        ]
        if merged.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_universe"})
            continue

        picks = select_signals(
            SignalSelectionRequest(
                universe=merged[["code", "score"]],
                trade_date=day,
                prev_positions=prev_positions,
                strategy_id=strategy_id,
                strategy_config=strategy_cfg,
            )
        )
        if picks.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_picks"})
            continue

        picks = picks.merge(
            merged[["code", buy_col, f"{sell_col}_next", "bridge_prev_close"]],
            on="code",
            how="left",
        ).dropna(subset=[buy_col, f"{sell_col}_next", "bridge_prev_close"])
        if picks.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_trade_price"})
            continue

        picks = normalize_weights(picks, weight_col="weight")

        buy_px = apply_twap_bps(picks[buy_col], buy_cost_bps, side="buy")
        sell_px = apply_twap_bps(picks[f"{sell_col}_next"], sell_cost_bps, side="sell")
        buy_leg_ret, sell_leg_ret, full_cycle_ret = split_cycle_return_by_bridge(
            buy_px,
            sell_px,
            pd.to_numeric(picks["bridge_prev_close"], errors="coerce"),
        )
        picks["buy_leg_ret_net"] = buy_leg_ret
        picks["sell_leg_ret_net"] = sell_leg_ret
        picks["full_cycle_ret_net"] = full_cycle_ret
        picks["return"] = full_cycle_ret
        day_buy_leg_ret = float((picks["buy_leg_ret_net"] * picks["weight"]).sum())
        day_sell_leg_ret = float((picks["sell_leg_ret_net"] * picks["weight"]).sum())
        day_return = float((picks["full_cycle_ret_net"] * picks["weight"]).sum())

        benchmark_breakdown = compute_benchmark_breakdown_for_day(
            raw_data_root=raw_root,
            trade_day=day,
            next_day=next_day,
            buy_bps=buy_cost_bps,
            sell_bps=sell_cost_bps,
        )
        daily_rows.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "day_return": day_return,
                "full_cycle_ret_net": day_return,
                "buy_leg_ret_net": day_buy_leg_ret,
                "sell_leg_ret_net": day_sell_leg_ret,
                "benchmark_return": float(benchmark_breakdown.full_cycle_ret_net),
                "benchmark_full_cycle_ret_net": float(benchmark_breakdown.full_cycle_ret_net),
                "benchmark_buy_leg_ret_net": float(benchmark_breakdown.buy_leg_ret_net),
                "benchmark_sell_leg_ret_net": float(benchmark_breakdown.sell_leg_ret_net),
                "avg_return": float(picks["return"].mean()),
                "total_weight": float(picks["weight"].sum()),
            }
        )

        full_buy = apply_twap_bps(merged[buy_col], buy_cost_bps, side="buy")
        full_sell = apply_twap_bps(merged[f"{sell_col}_next"], sell_cost_bps, side="sell")
        _, _, full_ret = split_cycle_return_by_bridge(
            full_buy,
            full_sell,
            pd.to_numeric(merged["bridge_prev_close"], errors="coerce"),
        )
        ic_rows.append(
            {
                "trade_date": day,
                "ic": float(merged["score"].corr(full_ret, method="pearson")),
                "rank_ic": float(merged["score"].corr(full_ret, method="spearman")),
                "count": int(len(merged)),
            }
        )
        diag_rows.append({"trade_date": day, "status": "ok", "reason": "", "count": int(len(picks))})

        for _, row in picks.iterrows():
            pos_rows.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "weight": float(row["weight"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "buy_price": float(row[buy_col]),
                    "sell_price": float(row[f"{sell_col}_next"]),
                    "bridge_prev_close": float(row["bridge_prev_close"]),
                    "buy_leg_ret_net": float(row["buy_leg_ret_net"]),
                    "sell_leg_ret_net": float(row["sell_leg_ret_net"]),
                    "full_cycle_ret_net": float(row["full_cycle_ret_net"]),
                    "return": float(row["return"]),
                }
            )
        prev_positions = to_prev_positions(picks)

    if not daily_rows:
        raise RuntimeError("backtest produced no daily returns")

    daily_df = pd.DataFrame(daily_rows).sort_values("trade_date")
    nav_df = daily_df[["trade_date"]].copy()
    nav_df["nav"] = (1.0 + daily_df["day_return"].fillna(0.0)).cumprod()
    nav_df["benchmark_nav"] = (1.0 + pd.to_numeric(daily_df["benchmark_return"], errors="coerce").fillna(0.0)).cumprod()
    positions_df = pd.DataFrame(pos_rows)
    ic_df = pd.DataFrame(ic_rows).sort_values("trade_date") if ic_rows else pd.DataFrame()
    diag_df = pd.DataFrame(diag_rows).sort_values("trade_date") if diag_rows else pd.DataFrame()

    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    batch_id = str(bt_cfg.get("batch_id", "Backtest"))
    out_dir = (
        Path(paths_cfg["results_root"])
        / date_label
        / batch_id
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(out_dir / "daily_returns.csv", index=False)
    nav_df.to_csv(out_dir / "nav_curve.csv", index=False)
    positions_df.to_csv(out_dir / "positions.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    write_backtest_report_image(
        out_dir=out_dir,
        daily_df=daily_df,
        nav_df=nav_df,
        ic_df=ic_df if not ic_df.empty else None,
    )

    return BacktestRunResult(out_dir=out_dir, days=int(len(daily_df)))

