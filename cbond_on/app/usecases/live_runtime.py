from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.universe import filter_tradable, normalize_price_bound
from cbond_on.infra.data.io import read_clean_daily
from cbond_on.infra.universe.pool_filter import (
    apply_allowlist_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.infra.universe.security_banlist import (
    apply_security_banlist_to_universe,
    load_security_banlist,
)
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


def _security_banlist_diagnostics(ban_info: dict, pre_count: int, post_count: int) -> dict:
    return {
        **ban_info,
        "pre_security_banlist_count": int(pre_count),
        "post_security_banlist_count": int(post_count),
        "security_banlist_removed_count": int(pre_count - post_count),
    }


def run_once(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    _ = mode
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    data_cfg = dict(live_cfg.get("data", {}))
    min_price = normalize_price_bound(data_cfg.get("min_price", 0.0))
    max_price = normalize_price_bound(data_cfg.get("max_price"))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    allowlist_raw = live_cfg.get("allowlist", {})
    allowlist_cfg = allowlist_raw if isinstance(allowlist_raw, dict) else {}
    allowlist_enabled = bool(allowlist_cfg.get("enabled", True)) if isinstance(allowlist_raw, dict) else bool(allowlist_raw)
    security_banlist_cfg = live_cfg.get("security_banlist", {})
    banned_codes, security_banlist_info = load_security_banlist(
        security_banlist_cfg if isinstance(security_banlist_cfg, dict) else {"enabled": bool(security_banlist_cfg)}
    )
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

    buy_col = str(
        output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457"))
    )
    sell_col = str(
        output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_0945"))
    )
    clean_daily = read_clean_daily(clean_root, score_day)
    if clean_daily.empty:
        universe = score_df[["code", "score"]].copy()
    else:
        universe = clean_daily.merge(score_df[["code", "score"]], on="code", how="inner")
        if universe.empty:
            raise ValueError("no score matched to clean data")

    pool_cfg = load_upstream_pool_config(allowlist_cfg or None)
    pool_codes, pool_info = resolve_pool_codes_for_trade_day(
        raw_data_root=raw_root,
        trade_day=score_day,
        pool_cfg=pool_cfg,
        enabled=allowlist_enabled,
    )
    if bool(pool_info.get("fallback_no_filter", False)):
        print(
            "[allowlist] fallback_no_filter",
            f"trade_day={score_day:%Y-%m-%d}",
            f"expected_pool_day={pool_info.get('pool_day_expected')}",
            f"reason={pool_info.get('fallback_reason')}",
            f"nearest_pool_day={pool_info.get('nearest_pool_day')}",
        )
    pre_allowlist_count = int(len(universe))
    universe = apply_allowlist_filter_to_universe(universe, allowlist_codes=pool_codes)
    allowlist_diag = _allowlist_diagnostics(pool_info, pre_allowlist_count, int(len(universe)))
    if universe.empty:
        raise ValueError("live universe is empty after allowlist filter")

    pre_banlist_count = int(len(universe))
    universe = apply_security_banlist_to_universe(universe, banned_codes=banned_codes)
    banlist_diag = _security_banlist_diagnostics(security_banlist_info, pre_banlist_count, int(len(universe)))
    if universe.empty:
        raise ValueError("live universe is empty after security banlist filter")

    if buy_col in universe.columns and sell_col in universe.columns:
        universe = filter_tradable(
            universe,
            buy_twap_col=buy_col,
            sell_twap_col=sell_col,
            min_amount=float(data_cfg.get("min_amount", 0.0)),
            min_volume=float(data_cfg.get("min_volume", 0.0)),
            min_price=min_price,
            max_price=max_price,
        )
    if universe.empty:
        raise ValueError("live universe is empty after tradable filter")

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
    allowlist_summary = {
        **allowlist_diag,
        **banlist_diag,
        "target_day": target_day,
        "score_day": score_day,
        "min_price": min_price,
        "max_price": max_price,
        "post_tradable_count": int(len(universe)),
        "picks_count": int(len(picks)),
    }
    summary_text = json.dumps(allowlist_summary, ensure_ascii=False, indent=2, default=str)
    (out_dir / "allowlist_summary.json").write_text(summary_text, encoding="utf-8")
    (out_dir / "universe_filter_summary.json").write_text(summary_text, encoding="utf-8")

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

