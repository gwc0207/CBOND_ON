from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.core.trading_days import next_trading_days_from_raw
from cbond_on.infra.benchmark.service import (
    build_strict_buy_holdings_from_selection,
    compute_benchmark_breakdowns_for_days,
    compute_strict_cycle_detail_for_holdings,
    load_strict_market_day,
)
from cbond_on.infra.universe.pool_filter import (
    apply_allowlist_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.domain.portfolio.service import normalize_weights, to_prev_positions
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.infra.backtest.config import load_strategy_config, resolve_score_path
from cbond_on.infra.io.market_twap import iter_open_days
from cbond_on.infra.report.backtest_plot import write_backtest_report_image
from cbond_on.common.artifact_keys import BACKTEST


@dataclass
class BacktestRunResult:
    out_dir: Path
    days: int


O005_ALLOWLIST_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"


def _allowlist_diagnostics(pool_info: dict, pre_count: int, post_count: int) -> dict:
    return {
        "allowlist_table": pool_info.get("allowlist_table") or pool_info.get("pool_table"),
        "allowlist_lag_trading_days": pool_info.get("allowlist_lag_trading_days")
        or pool_info.get("pool_lag_trading_days"),
        "allowlist_day_expected": pool_info.get("allowlist_day_expected") or pool_info.get("pool_day_expected"),
        "allowlist_day_used": pool_info.get("allowlist_day_used") or pool_info.get("pool_day_used"),
        "allowlist_applied": bool(pool_info.get("allowlist_applied") or pool_info.get("pool_enabled")),
        "allowlist_codes_count": int(pool_info.get("allowlist_codes_count") or pool_info.get("pool_codes_count") or 0),
        "allowlist_fallback_no_filter": bool(
            pool_info.get("allowlist_fallback_no_filter") or pool_info.get("fallback_no_filter")
        ),
        "allowlist_fallback_reason": pool_info.get("allowlist_fallback_reason") or pool_info.get("fallback_reason", ""),
        "pre_allowlist_count": int(pre_count),
        "post_allowlist_count": int(post_count),
    }


def _assert_o005_only_universe_config(bt_cfg: dict) -> None:
    forbidden_keys = {
        "security_banlist",
        "filter_tradable",
        "min_price",
        "max_price",
        "min_amount",
        "min_volume",
    }
    present = sorted(k for k in forbidden_keys if k in bt_cfg)
    if present:
        raise ValueError(
            "backtest universe filter must only use o_0005 allowlist; remove: "
            + ", ".join(present)
        )

    allowlist_raw = bt_cfg.get("allowlist")
    if not isinstance(allowlist_raw, dict):
        raise ValueError("backtest allowlist must be configured as an o_0005 allowlist block")
    if not bool(allowlist_raw.get("enabled", True)):
        raise ValueError("backtest allowlist must be enabled; o_0005 is the only allowed universe filter")
    table = str(allowlist_raw.get("table", allowlist_raw.get("pool_table", ""))).strip()
    if table != O005_ALLOWLIST_TABLE:
        raise ValueError(
            "backtest allowlist table must be "
            f"{O005_ALLOWLIST_TABLE}; got {table or '<missing>'}"
        )


def _progress_step(total: int) -> int:
    return max(1, min(25, max(1, total // 20)))


def _build_output_dir(results_root: str | Path, date_label: str, batch_id: str) -> Path:
    return (
        Path(results_root)
        / BACKTEST
        / date_label
        / batch_id
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )


def _resolve_output_root(bt_cfg: dict, paths_cfg: dict) -> Path:
    """Allow research configs to keep reports outside the shared backtest root."""
    return resolve_output_path(
        bt_cfg.get("output_root"),
        default_path=paths_cfg["results_root"],
        results_root=paths_cfg["results_root"],
    )


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    cfg: dict | None = None,
) -> BacktestRunResult:
    paths_cfg = load_config_file("paths")
    bt_cfg = dict(cfg or load_config_file("backtest"))
    _assert_o005_only_universe_config(bt_cfg)

    start_day = parse_date(start or bt_cfg.get("start"))
    end_day = parse_date(end or bt_cfg.get("end"))
    strategy_id = str(bt_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy_cfg = load_strategy_config(
        bt_cfg.get("strategy_config_path"),
        inline=bt_cfg.get("strategy_config"),
    )
    score_path = resolve_score_path(bt_cfg, paths_cfg)
    batch_id = str(bt_cfg.get("batch_id", "Backtest"))
    output_root = _resolve_output_root(bt_cfg, paths_cfg)
    print(
        "[backtest] start",
        f"batch={batch_id}",
        f"range={start_day:%Y-%m-%d}..{end_day:%Y-%m-%d}",
        f"strategy={strategy_id}",
        f"score_path={score_path}",
        f"output_root={output_root}",
        flush=True,
    )
    score_cache = load_scores_by_date(score_path)
    if score_cache:
        score_days = sorted(score_cache)
        print(
            "[backtest] scores loaded",
            f"days={len(score_days)}",
            f"range={score_days[0]:%Y-%m-%d}..{score_days[-1]:%Y-%m-%d}",
            flush=True,
        )
    else:
        print("[backtest] scores loaded days=0", flush=True)

    buy_cost_bps, sell_cost_bps, fee_source = load_fees_buy_sell_bps()
    print(
        "backtest cost:",
        f"buy_bps={buy_cost_bps:.4f}",
        f"sell_bps={sell_cost_bps:.4f}",
        f"source={fee_source}",
    )
    allowlist_raw = bt_cfg.get("allowlist", {})
    allowlist_cfg = allowlist_raw if isinstance(allowlist_raw, dict) else {}
    allowlist_enabled = bool(allowlist_cfg.get("enabled", True)) if isinstance(allowlist_raw, dict) else bool(allowlist_raw)
    pool_cfg = load_upstream_pool_config(allowlist_cfg or None)

    raw_root = paths_cfg["raw_data_root"]
    days = iter_open_days(raw_root, start_day, end_day)
    tail_day = next_trading_days_from_raw(raw_root, end_day, 1, kind="snapshot", asset="cbond")
    if tail_day:
        days = sorted(set(days + tail_day))
    if len(days) < 2:
        raise ValueError("not enough trading days for backtest")

    benchmark_days = [day for day in days if start_day <= day <= end_day]
    print(
        "[backtest] benchmark start",
        f"days={len(benchmark_days)}",
        f"range={benchmark_days[0]:%Y-%m-%d}..{benchmark_days[-1]:%Y-%m-%d}"
        if benchmark_days
        else "range=<empty>",
        flush=True,
    )
    benchmark_daily = compute_benchmark_breakdowns_for_days(
        raw_data_root=raw_root,
        trade_days=benchmark_days,
        buy_bps=buy_cost_bps,
        sell_bps=sell_cost_bps,
        skip_failed_days=True,
    )
    print(
        "[backtest] benchmark done",
        f"rows={len(benchmark_daily)}",
        flush=True,
    )
    benchmark_by_day: dict[date, pd.Series] = {}
    if not benchmark_daily.empty:
        benchmark_lookup = benchmark_daily.copy()
        benchmark_lookup["trade_date"] = pd.to_datetime(benchmark_lookup["trade_date"], errors="coerce").dt.date
        benchmark_lookup = benchmark_lookup.dropna(subset=["trade_date"])
        benchmark_by_day = {
            row["trade_date"]: row
            for _, row in benchmark_lookup.iterrows()
        }

    daily_rows: list[dict] = []
    pos_rows: list[dict] = []
    diag_rows: list[dict] = []
    ic_rows: list[dict] = []
    prev_positions = pd.DataFrame(columns=["code", "weight"])
    loop_days = [day for day in days[:-1] if start_day <= day <= end_day]
    loop_total = len(loop_days)
    loop_step = _progress_step(loop_total)
    loop_seen = 0
    print("[backtest] daily loop start", f"days={loop_total}", flush=True)

    for idx in range(len(days) - 1):
        day = days[idx]
        if not (start_day <= day <= end_day):
            continue
        next_day = days[idx + 1]
        loop_seen += 1
        if loop_seen == 1 or loop_seen % loop_step == 0 or loop_seen == loop_total:
            print(
                "[backtest] daily progress",
                f"{loop_seen}/{loop_total}",
                f"day={day:%Y-%m-%d}",
                f"ok={len(daily_rows)}",
                f"diag={len(diag_rows)}",
                flush=True,
            )

        score_df = score_cache.get(day, pd.DataFrame())
        if score_df.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_score"})
            continue
        try:
            merged = load_strict_market_day(
                raw_data_root=raw_root,
                trade_day=day,
                buy_bps=buy_cost_bps,
                sell_bps=sell_cost_bps,
            )
        except Exception as exc:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": f"missing_strict_market:{exc}"})
            continue
        merged = merged[
            pd.to_numeric(merged["buy_price"], errors="coerce").notna()
            & pd.to_numeric(merged["buy_close_price"], errors="coerce").notna()
            & (pd.to_numeric(merged["buy_price"], errors="coerce") > 0)
            & (pd.to_numeric(merged["buy_close_price"], errors="coerce") > 0)
        ].copy()
        if merged.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_strict_buy_market"})
            continue
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_root,
            trade_day=day,
            pool_cfg=pool_cfg,
            enabled=allowlist_enabled,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "[allowlist] required pool is unavailable; backtest does not allow no-filter fallback: "
                f"trade_day={day:%Y-%m-%d} "
                f"expected_pool_day={pool_info.get('pool_day_expected')} "
                f"reason={pool_info.get('fallback_reason')} "
                f"nearest_pool_day={pool_info.get('nearest_pool_day')}"
            )
        pre_allowlist_count = int(len(merged))
        merged = apply_allowlist_filter_to_universe(merged, allowlist_codes=pool_codes)
        allowlist_diag = _allowlist_diagnostics(pool_info, pre_allowlist_count, int(len(merged)))
        if merged.empty:
            diag_rows.append(
                {
                    "trade_date": day,
                    "status": "skip",
                    "reason": "empty_allowlist_filtered_universe",
                    **allowlist_diag,
                }
            )
            continue
        filter_diag = {**allowlist_diag, "universe_filter": "o_0005_only"}
        merged = merged.merge(score_df[["code", "score"]], on="code", how="inner")
        if merged.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_universe", **filter_diag})
            continue

        try:
            ic_cycle = compute_strict_cycle_detail_for_holdings(
                raw_data_root=raw_root,
                buy_day=day,
                sell_day=next_day,
                buy_holdings=merged.assign(weight=1.0),
                sell_bps=sell_cost_bps,
            )
            ic_base = merged[["code", "score"]].merge(ic_cycle[["code", "return_net"]], on="code", how="inner")
            ic_base["return_net"] = pd.to_numeric(ic_base["return_net"], errors="coerce")
            ic_base = ic_base[ic_base["return_net"].notna()]
            if not ic_base.empty:
                ic_rows.append(
                    {
                        "trade_date": day,
                        "ic": float(ic_base["score"].corr(ic_base["return_net"], method="pearson")),
                        "rank_ic": float(ic_base["score"].corr(ic_base["return_net"], method="spearman")),
                        "count": int(len(ic_base)),
                    }
                )
        except Exception:
            pass

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
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "empty_picks", **filter_diag})
            continue

        picks = build_strict_buy_holdings_from_selection(
            raw_data_root=raw_root,
            buy_day=day,
            selection=picks,
            buy_bps=buy_cost_bps,
            normalize=True,
        )
        if picks.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_strict_buy_price", **filter_diag})
            continue

        picks = normalize_weights(picks, weight_col="weight")
        cycle_detail = compute_strict_cycle_detail_for_holdings(
            raw_data_root=raw_root,
            buy_day=day,
            sell_day=next_day,
            buy_holdings=picks,
            sell_bps=sell_cost_bps,
        )
        cycle_detail = cycle_detail[pd.to_numeric(cycle_detail["return_net"], errors="coerce").notna()].copy()
        if cycle_detail.empty:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_strict_cycle_return", **filter_diag})
            continue

        day_return = float(pd.to_numeric(cycle_detail["weighted_return"], errors="coerce").sum())
        day_buy_leg_ret = float(pd.to_numeric(cycle_detail["weighted_buy_leg_ret_net"], errors="coerce").sum())
        day_sell_leg_ret = float(pd.to_numeric(cycle_detail["weighted_sell_leg_ret_net"], errors="coerce").sum())
        sell_count = int(cycle_detail["code"].nunique())
        fallback_mask = cycle_detail["sell_missing_fallback"].astype(bool)
        fallback_sell_codes = int(fallback_mask.sum())
        fallback_sell_weight = float(pd.to_numeric(cycle_detail.loc[fallback_mask, "weight"], errors="coerce").sum())

        benchmark_row = benchmark_by_day.get(day)
        if benchmark_row is None:
            diag_rows.append({"trade_date": day, "status": "skip", "reason": "missing_benchmark", **filter_diag})
            continue
        daily_rows.append(
            {
                "trade_date": day,
                "count": int(len(picks)),
                "day_return": day_return,
                "full_cycle_ret_net": day_return,
                "buy_leg_ret_net": day_buy_leg_ret,
                "sell_leg_ret_net": day_sell_leg_ret,
                "benchmark_return": float(benchmark_row["benchmark_return"]),
                "benchmark_full_cycle_ret_net": float(benchmark_row["benchmark_return"]),
                "benchmark_buy_leg_ret_net": float(benchmark_row["buy_leg_ret_net"]),
                "benchmark_sell_leg_ret_net": float(benchmark_row["sell_leg_ret_net"]),
                "benchmark_buy_count": int(benchmark_row.get("buy_count", benchmark_row.get("count", 0))),
                "benchmark_sell_count": int(benchmark_row.get("sell_count", 0)),
                "benchmark_fallback_sell_codes": int(benchmark_row.get("fallback_sell_codes", 0)),
                "benchmark_fallback_sell_weight": float(benchmark_row.get("fallback_sell_weight", 0.0)),
                "benchmark_method": str(benchmark_row.get("benchmark_method", "strict_official_prev_close_split")),
                "avg_return": float(pd.to_numeric(cycle_detail["return_net"], errors="coerce").mean()),
                "total_weight": float(pd.to_numeric(cycle_detail["weight"], errors="coerce").sum()),
                "sell_count": sell_count,
                "fallback_sell_codes": fallback_sell_codes,
                "fallback_sell_weight": fallback_sell_weight,
            }
        )
        diag_rows.append({"trade_date": day, "status": "ok", "reason": "", "count": int(len(picks)), **filter_diag})

        for _, row in cycle_detail.iterrows():
            pos_rows.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "weight": float(row["weight"]),
                    "score": float(row["score"]),
                    "rank": int(row["rank"]),
                    "buy_price": float(row["buy_price"]),
                    "sell_price": float(row["sell_price"]),
                    "buy_close_price": float(row["buy_close_price"]),
                    "bridge_prev_close": float(row["buy_close_price"]),
                    "buy_leg_ret_net": float(row["buy_leg_ret_net"]),
                    "sell_leg_ret_net": float(row["strict_sell_leg_net_ret"]),
                    "full_cycle_ret_net": float(row["full_cycle_ret_net"]),
                    "return": float(row["return_net"]),
                }
            )
        prev_positions = to_prev_positions(picks)

    print(
        "[backtest] daily loop done",
        f"ok={len(daily_rows)}",
        f"diag={len(diag_rows)}",
        flush=True,
    )
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
    out_dir = _build_output_dir(output_root, date_label, batch_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[backtest] write outputs", f"out_dir={out_dir}", flush=True)
    daily_df.to_csv(out_dir / "daily_returns.csv", index=False)
    nav_df.to_csv(out_dir / "nav_curve.csv", index=False)
    positions_df.to_csv(out_dir / "positions.csv", index=False)
    if not ic_df.empty:
        ic_df.to_csv(out_dir / "ic.csv", index=False)
    if not diag_df.empty:
        diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    write_backtest_report_image(
        out_dir=out_dir,
        daily_df=daily_df,
        nav_df=nav_df,
        ic_df=ic_df if not ic_df.empty else None,
        positions_df=positions_df if not positions_df.empty else None,
        diagnostics_df=diag_df if not diag_df.empty else None,
        configured_start=start_day,
        configured_end=end_day,
    )
    print("[backtest] done", f"out_dir={out_dir}", f"days={len(daily_df)}", flush=True)

    return BacktestRunResult(out_dir=out_dir, days=int(len(daily_df)))

