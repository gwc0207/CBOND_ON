from __future__ import annotations

from datetime import date
from pathlib import Path

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.universe import filter_tradable
from cbond_on.infra.data.io import read_clean_daily
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.app.usecases.label_runtime import run as run_label
from cbond_on.app.usecases.panel_runtime import run as run_panel
from cbond_on.app.usecases.factor_build_runtime import run as run_factor_build
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.infra.live.config import (
    assert_no_date_fields_in_live_config,
    load_live_factor_runtime,
    load_live_model_runtime,
    load_strategy_config,
)
from cbond_on.infra.live.db_writer import write_trades_to_db
from cbond_on.infra.live.holdings import load_previous_holdings
from cbond_on.infra.live.publish_gate import (
    data_hub_runtime_from_live,
    ensure_publish_ready,
    redis_snapshot_enabled,
    resolve_rebuild_window,
    resolve_redis_sync_day,
    today_shanghai,
)
from cbond_on.infra.live.score import resolve_score_df_for_target


def run_once(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    _ = mode
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    backtest_cfg = load_config_file("backtest_pipeline")

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    data_cfg = dict(live_cfg.get("data", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    factor_cfg_key, live_factor_cfg = load_live_factor_runtime(live_cfg)
    model_cfg_key, live_model_score_cfg, model_id = load_live_model_runtime(live_cfg)

    assert_no_date_fields_in_live_config(schedule_cfg, model_cfg)

    today = today_shanghai()
    target_day = parse_date(target) if target is not None else today
    start_day = parse_date(start) if start is not None else target_day

    refresh_data = bool(data_cfg.get("refresh", False))
    overwrite_data = bool(data_cfg.get("overwrite", False))
    lookback_days = max(0, int(data_cfg.get("lookback_days", 0)))
    use_redis_snapshot = redis_snapshot_enabled(data_cfg, live_cfg)

    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"))
    data_hub = data_hub_runtime_from_live(
        live_cfg,
        raw_root=raw_root,
        clean_root=clean_root,
    )

    run_day = resolve_redis_sync_day(data_cfg, target_day) if use_redis_snapshot else target_day
    rebuild_start_day, prev_trade_day = resolve_rebuild_window(
        raw_root=raw_root,
        run_day=run_day,
        lookback_days=lookback_days,
    )

    if use_redis_snapshot:
        print(
            "live run window:",
            f"run_day={run_day}",
            f"target_day={target_day}",
            f"lookback_start={rebuild_start_day}",
            f"prev_trading_day={prev_trade_day}",
        )

    gate_day = run_day if use_redis_snapshot else target_day
    if not bool(data_hub.get("ready_gate_enabled", True)):
        raise ValueError("live_config.data_hub.ready_gate_enabled must be true in consumer-only mode")
    ensure_publish_ready(
        runtime=data_hub,
        trade_day=gate_day,
    )

    if use_redis_snapshot:
        run_panel(
            start=rebuild_start_day,
            end=run_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
        )
        if prev_trade_day >= rebuild_start_day:
            run_label(
                start=rebuild_start_day,
                end=prev_trade_day,
                refresh=refresh_data,
                overwrite=overwrite_data,
            )
        run_factor_build(
            start=rebuild_start_day,
            end=run_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
            cfg=live_factor_cfg,
        )
    else:
        label_end = prev_trade_day
        run_panel(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
        if label_end >= start_day:
            run_label(start=start_day, end=label_end, refresh=refresh_data, overwrite=overwrite_data)
        run_factor_build(
            start=start_day,
            end=target_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
            cfg=live_factor_cfg,
        )

    model_start = start_day
    model_end = target_day
    model_label_cutoff = model_cfg.get("label_cutoff")
    score_day = target_day
    if use_redis_snapshot:
        model_start = run_day
        model_end = run_day
        model_label_cutoff = prev_trade_day
        score_day = run_day
        print(
            "model score window:",
            f"score_day={score_day}",
            f"label_cutoff={model_label_cutoff}",
        )
    print(
        "live config profile:",
        f"factors={factor_cfg_key}",
        f"models={model_cfg_key}",
        f"model_id={model_id}",
    )

    model_result = run_model_score(
        model_id=model_id,
        start=model_start,
        end=model_end,
        label_cutoff=model_label_cutoff,
        cfg=live_model_score_cfg,
    )
    score_path = resolve_output_path(
        model_result.get("score_output"),
        default_path=Path(paths_cfg["results_root"]) / "scores" / model_id,
        results_root=paths_cfg["results_root"],
    )
    score_cache = load_scores_by_date(score_path)
    score_df = resolve_score_df_for_target(score_cache, score_day, score_path)

    clean_daily = read_clean_daily(clean_root, score_day)
    if clean_daily.empty:
        universe = score_df[["code", "score"]].copy()
    else:
        universe = clean_daily.merge(score_df[["code", "score"]], on="code", how="inner")
        if universe.empty:
            raise ValueError("no score matched to clean data")
        buy_col = str(
            backtest_cfg.get("buy_twap_col", output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
        )
        sell_col = str(
            backtest_cfg.get("sell_twap_col", output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_0945")))
        )
        if buy_col in universe.columns and sell_col in universe.columns:
            universe = filter_tradable(
                universe,
                buy_twap_col=buy_col,
                sell_twap_col=sell_col,
                min_amount=float(data_cfg.get("min_amount", 0.0)),
                min_volume=float(data_cfg.get("min_volume", 0.0)),
            )
    if universe.empty:
        raise ValueError("live universe is empty after filters")

    strategy_id = str(strategy_cfg.get("strategy_id", "strategy01_topk_turnover"))
    strategy_config = load_strategy_config(strategy_cfg.get("strategy_config_path"))
    strategy_config = strategy_config or {k: v for k, v in strategy_cfg.items() if k != "strategy_id"}
    prev_positions = load_previous_holdings(Path(paths_cfg["results_root"]) / "live", target_day)
    picks = select_signals(
        SignalSelectionRequest(
            universe=universe[["code", "score"]],
            trade_date=target_day,
            prev_positions=prev_positions,
            strategy_id=strategy_id,
            strategy_config=strategy_config,
        )
    )
    if picks.empty:
        raise ValueError("strategy returned empty picks")

    out_dir = Path(paths_cfg["results_root"]) / "live" / f"{target_day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = picks.copy()
    picks["trade_date"] = target_day
    picks.to_csv(out_dir / "trade_list.csv", index=False)

    if bool(output_cfg.get("db_write", False)):
        if not output_cfg.get("db_table"):
            raise ValueError("live_config.output.db_table is required when db_write=true")
        db_trade_day = prev_trade_day
        db_picks = picks.copy()
        db_picks["trade_date"] = db_trade_day
        try:
            write_trades_to_db(
                trades=db_picks,
                trade_day=db_trade_day,
                table=str(output_cfg["db_table"]),
                mode=str(output_cfg.get("db_mode", "replace_date")),
                backend=output_cfg.get("db_backend"),
            )
        except FileNotFoundError as exc:
            print(f"skip output db write: {exc}")

    return out_dir


def run(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    return run_once(start=start, target=target, mode=mode)

