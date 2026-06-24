from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.trading_days import prev_trading_days_from_raw
from cbond_on.infra.data.io import read_clean_daily
from cbond_on.infra.universe.pool_filter import (
    apply_allowlist_filter_to_universe,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.domain.signals.service import SignalSelectionRequest, select_signals
from cbond_on.infra.model.score_io import load_scores_by_date
from cbond_on.app.usecases.factor_build_runtime import run as run_factor_build
from cbond_on.app.usecases.label_runtime import run as run_label_build
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.app.usecases.panel_runtime import run as run_panel_build
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
    today_shanghai,
)
from cbond_on.infra.live.score import resolve_score_df_for_target


O005_ALLOWLIST_TABLE = "quant_factor_dev.researcher_xuvb.o_0005"


def _month_day_path(root: str | Path, day: date, *, filename_root: str = "") -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    base = Path(root)
    return base / filename_root / month / filename if filename_root else base / month / filename


def _panel_day_path(paths_cfg: dict, day: date, *, asset: str, panel_name: str) -> Path:
    return (
        Path(paths_cfg["panel_data_root"])
        / "panels"
        / asset
        / panel_name
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.parquet"
    )


def _factor_day_path(paths_cfg: dict, day: date, *, panel_name: str) -> Path:
    return (
        Path(paths_cfg["factor_data_root"])
        / "factors"
        / panel_name
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day.strftime('%Y%m%d')}.parquet"
    )


def _label_day_path(paths_cfg: dict, day: date) -> Path:
    return _month_day_path(paths_cfg["label_data_root"], day)


def _require_existing(path: Path, *, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{name} missing after live build: {path}")


def _normalize_assets(value: object) -> list[str]:
    if isinstance(value, str):
        return [x.strip().lower() for x in value.replace(";", ",").split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip().lower() for x in value if str(x).strip()]
    return []


def _factor_panel_source_mode(factor_cfg: dict) -> str:
    raw = factor_cfg.get("panel_source")
    if isinstance(raw, dict):
        raw = raw.get("mode", "cached_panel")
    text = str(raw or "cached_panel").strip().lower()
    if text in {"clean", "clean_data", "clean_direct", "on_demand", "on_demand_clean"}:
        return "clean_direct"
    return "cached_panel"


def _missing_days(days: list[date], *, path_builder) -> list[date]:
    return [day for day in days if not path_builder(day).exists()]


def _build_day_span(days: list[date]) -> tuple[date, date]:
    ordered = sorted(set(days))
    return ordered[0], ordered[-1]


def _parse_live_model_window_days(live_model_score_cfg: dict, model_id: str) -> int:
    models_raw = live_model_score_cfg.get("models", {})
    if not isinstance(models_raw, dict):
        return 0
    model_entry_raw = models_raw.get(model_id, {})
    if not isinstance(model_entry_raw, dict):
        return 0
    model_cfg_key = str(model_entry_raw.get("model_config", "")).strip()
    if not model_cfg_key:
        return 0
    model_cfg = load_json_like(resolve_config_path(model_cfg_key))
    rolling_cfg = model_cfg.get("rolling", {})
    if not isinstance(rolling_cfg, dict):
        return 0
    if not bool(rolling_cfg.get("enabled", False)):
        return 0
    return int(rolling_cfg.get("window_days", 0))


def _backfill_live_history(
    *,
    paths_cfg: dict,
    raw_root: str,
    score_day: date,
    prev_trade_day: date,
    panel_cfg: dict,
    label_cfg: dict,
    factor_cfg: dict,
    panel_name: str,
    window_days: int,
    skip_panel_build: bool = False,
) -> None:
    history_days = prev_trading_days_from_raw(
        raw_root,
        score_day,
        max(1, int(window_days)),
        kind="snapshot",
        asset="cbond",
    )
    if not history_days:
        history_days = [prev_trade_day]
    history_days = sorted(set(history_days))

    assets = _normalize_assets(panel_cfg.get("assets", []))
    if not assets:
        assets = ["cbond"]

    missing_panel_by_asset: dict[str, list[date]] = {}
    if skip_panel_build:
        print(
            "live backfill panel skipped:",
            "reason=clean_direct_factor_panel_source",
            f"history_days={len(history_days)}",
        )
    else:
        for asset in assets:
            missing_panel_days = _missing_days(
                history_days,
                path_builder=lambda d, _asset=asset: _panel_day_path(paths_cfg, d, asset=_asset, panel_name=panel_name),
            )
            if missing_panel_days:
                missing_panel_by_asset[asset] = missing_panel_days

    if (not skip_panel_build) and missing_panel_by_asset:
        all_missing_panel_days = sorted(
            {
                day
                for asset_days in missing_panel_by_asset.values()
                for day in asset_days
            }
        )
        panel_start, panel_end = _build_day_span(all_missing_panel_days)
        panel_backfill_cfg = dict(panel_cfg)
        panel_backfill_cfg["start"] = panel_start
        panel_backfill_cfg["end"] = panel_end
        panel_backfill_cfg["refresh"] = False
        panel_backfill_cfg["overwrite"] = False
        print(
            "live backfill panel:",
            f"start={panel_start}",
            f"end={panel_end}",
            f"assets={','.join(sorted(missing_panel_by_asset.keys()))}",
            f"missing_days={len(all_missing_panel_days)}",
        )
        panel_result = run_panel_build(
            start=panel_start,
            end=panel_end,
            refresh=False,
            overwrite=False,
            cfg=panel_backfill_cfg,
        )
        print("live backfill panel done:", panel_result)
        for asset, asset_days in missing_panel_by_asset.items():
            for day in asset_days:
                _require_existing(
                    _panel_day_path(paths_cfg, day, asset=asset, panel_name=panel_name),
                    name=f"{asset} panel(backfill)",
                )

    missing_label_days = _missing_days(
        history_days,
        path_builder=lambda d: _label_day_path(paths_cfg, d),
    )
    if missing_label_days:
        label_start, _ = _build_day_span(missing_label_days)
        label_end = score_day
        label_backfill_cfg = dict(label_cfg)
        label_backfill_cfg["start"] = label_start
        label_backfill_cfg["end"] = label_end
        label_backfill_cfg["refresh"] = False
        label_backfill_cfg["overwrite"] = False
        print(
            "live backfill labels:",
            f"start={label_start}",
            f"end={label_end}",
            f"missing_days={len(missing_label_days)}",
        )
        label_result = run_label_build(
            start=label_start,
            end=label_end,
            refresh=False,
            overwrite=False,
            cfg=label_backfill_cfg,
            panel_cfg=panel_cfg,
        )
        print("live backfill labels done:", label_result)
        for day in missing_label_days:
            _require_existing(_label_day_path(paths_cfg, day), name="label(backfill)")

    missing_factor_days = _missing_days(
        history_days,
        path_builder=lambda d: _factor_day_path(paths_cfg, d, panel_name=panel_name),
    )
    if missing_factor_days:
        factor_start, factor_end = _build_day_span(missing_factor_days)
        factor_backfill_cfg = dict(factor_cfg)
        factor_backfill_cfg["start"] = factor_start
        factor_backfill_cfg["end"] = factor_end
        factor_backfill_cfg["refresh"] = False
        factor_backfill_cfg["overwrite"] = False
        factor_backfill_cfg["panel_name"] = panel_name
        print(
            "live backfill factors:",
            f"start={factor_start}",
            f"end={factor_end}",
            f"missing_days={len(missing_factor_days)}",
        )
        factor_result = run_factor_build(
            start=factor_start,
            end=factor_end,
            refresh=False,
            overwrite=False,
            cfg=factor_backfill_cfg,
        )
        print("live backfill factors done:", factor_result)
        for day in missing_factor_days:
            _require_existing(_factor_day_path(paths_cfg, day, panel_name=panel_name), name="factor(backfill)")


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


def _assert_live_data_boundary(live_cfg: dict) -> None:
    """Live consumes DataHub clean data, then builds CBOND_ON local derived artifacts."""

    forbidden_top_level = {"source", "redis"}
    present_top_level = sorted(k for k in forbidden_top_level if k in live_cfg)
    if present_top_level:
        raise ValueError(
            "live_config must not own raw/redis processing; remove sections: "
            + ", ".join(present_top_level)
        )

    data_cfg = dict(live_cfg.get("data", {}))
    forbidden_data_keys = {
        "refresh",
        "overwrite",
        "lookback_days",
        "kline_enabled",
        "snapshot_source",
        "raw_sync_mode_when_redis",
        "redis_sync_day",
        "redis_source",
        "redis_stage",
        "redis_asset_type",
        "redis_incremental",
        "redis_full_day",
        "min_price",
        "max_price",
        "min_amount",
        "min_volume",
    }
    present_data_keys = sorted(k for k in forbidden_data_keys if k in data_cfg)
    if present_data_keys:
        raise ValueError(
            "live_config.data must not own raw/redis processing; remove keys: "
            + ", ".join(present_data_keys)
        )

    if "security_banlist" in live_cfg:
        raise ValueError("live universe filter must only use o_0005 allowlist; remove security_banlist")

    allowlist_raw = live_cfg.get("allowlist")
    if not isinstance(allowlist_raw, dict):
        raise ValueError("live allowlist must be configured as an o_0005 allowlist block")
    if not bool(allowlist_raw.get("enabled", True)):
        raise ValueError("live allowlist must be enabled; o_0005 is the only allowed universe filter")
    table = str(allowlist_raw.get("table", allowlist_raw.get("pool_table", ""))).strip()
    if table != O005_ALLOWLIST_TABLE:
        raise ValueError(
            "live allowlist table must be "
            f"{O005_ALLOWLIST_TABLE}; got {table or '<missing>'}"
        )


def _prev_trading_day(raw_root: str, day: date) -> date:
    prev_days = prev_trading_days_from_raw(
        raw_root,
        day,
        1,
        kind="snapshot",
        asset="cbond",
    )
    if not prev_days:
        raise RuntimeError(f"cannot resolve previous trading day before {day}")
    return prev_days[-1]


def run_once(
    *,
    start: str | date | None = None,
    target: str | date | None = None,
    mode: str = "default",
) -> Path:
    _ = mode
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    _assert_live_data_boundary(live_cfg)

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    allowlist_raw = live_cfg.get("allowlist", {})
    allowlist_cfg = allowlist_raw if isinstance(allowlist_raw, dict) else {}
    allowlist_enabled = bool(allowlist_cfg.get("enabled", True)) if isinstance(allowlist_raw, dict) else bool(allowlist_raw)
    factor_cfg_key, live_factor_cfg = load_live_factor_runtime(live_cfg)
    model_cfg_key, live_model_score_cfg, model_id = load_live_model_runtime(live_cfg)

    assert_no_date_fields_in_live_config(schedule_cfg, model_cfg)

    today = today_shanghai()
    target_day = parse_date(target) if target is not None else today
    score_day = parse_date(start) if start is not None else (
        today if target_day >= today else target_day
    )

    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"))
    data_hub = data_hub_runtime_from_live(
        live_cfg,
        raw_root=raw_root,
        clean_root=clean_root,
    )

    prev_trade_day = _prev_trading_day(raw_root, score_day)

    print(
        "live run window:",
        f"score_day={score_day}",
        f"target_day={target_day}",
        f"prev_trading_day={prev_trade_day}",
        "mode=clean_consumer_build_local",
    )

    if not bool(data_hub.get("ready_gate_enabled", True)):
        raise ValueError("live_config.data_hub.ready_gate_enabled must be true in consumer-only mode")
    ensure_publish_ready(
        runtime=data_hub,
        trade_day=score_day,
    )

    panel_cfg = dict(load_config_file("panel"))
    panel_cfg["start"] = score_day
    panel_cfg["end"] = score_day
    panel_cfg["refresh"] = True
    panel_cfg["overwrite"] = True
    panel_name = str(panel_cfg.get("panel_name") or live_factor_cfg.get("panel_name") or "T1430")
    panel_cfg["panel_name"] = panel_name

    label_cfg = dict(load_config_file("label"))
    label_cfg["start"] = prev_trade_day
    label_cfg["end"] = score_day
    label_cfg["refresh"] = True
    label_cfg["overwrite"] = True

    factor_runtime_cfg = dict(live_factor_cfg)
    factor_runtime_cfg["start"] = score_day
    factor_runtime_cfg["end"] = score_day
    factor_runtime_cfg["refresh"] = True
    factor_runtime_cfg["overwrite"] = True
    factor_runtime_cfg["panel_name"] = panel_name
    panel_source_mode = _factor_panel_source_mode(factor_runtime_cfg)
    use_clean_direct_panel_source = panel_source_mode == "clean_direct"

    window_days = _parse_live_model_window_days(live_model_score_cfg, model_id)
    if window_days > 0:
        print(
            "live backfill check:",
            f"window_days={window_days}",
            f"score_day={score_day}",
        )
        _backfill_live_history(
            paths_cfg=paths_cfg,
            raw_root=raw_root,
            score_day=score_day,
            prev_trade_day=prev_trade_day,
            panel_cfg=panel_cfg,
            label_cfg=label_cfg,
            factor_cfg=factor_runtime_cfg,
            panel_name=panel_name,
            window_days=window_days,
            skip_panel_build=use_clean_direct_panel_source,
        )

    if use_clean_direct_panel_source:
        print(
            "live build panel skipped:",
            f"day={score_day}",
            f"panel={panel_name}",
            f"panel_source={panel_source_mode}",
        )
    else:
        print("live build panel:", f"day={score_day}", f"panel={panel_name}")
        panel_result = run_panel_build(
            start=score_day,
            end=score_day,
            refresh=True,
            overwrite=True,
            cfg=panel_cfg,
        )
        _require_existing(_panel_day_path(paths_cfg, score_day, asset="cbond", panel_name=panel_name), name="cbond panel")
        if "stock" in _normalize_assets(panel_cfg.get("assets", [])):
            _require_existing(_panel_day_path(paths_cfg, score_day, asset="stock", panel_name=panel_name), name="stock panel")
        print("live build panel done:", panel_result)

    print("live build labels:", f"day={prev_trade_day}", f"next_day={score_day}")
    label_result = run_label_build(
        start=prev_trade_day,
        end=score_day,
        refresh=True,
        overwrite=True,
        cfg=label_cfg,
        panel_cfg=panel_cfg,
    )
    _require_existing(_label_day_path(paths_cfg, prev_trade_day), name="label")
    print("live build labels done:", label_result)

    print("live build factors:", f"day={score_day}", f"panel={panel_name}")
    factor_result = run_factor_build(
        start=score_day,
        end=score_day,
        refresh=True,
        overwrite=True,
        cfg=factor_runtime_cfg,
    )
    _require_existing(_factor_day_path(paths_cfg, score_day, panel_name=panel_name), name="factor")
    print("live build factors done:", factor_result)

    model_start = score_day
    model_end = score_day
    model_label_cutoff = model_cfg.get("label_cutoff")
    if model_label_cutoff is None:
        model_label_cutoff = prev_trade_day
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
        raise RuntimeError(
            f"clean daily data missing for {score_day}; live will not fall back to score-only universe"
        )
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
        raise RuntimeError(
            "[allowlist] required pool is unavailable; live does not allow no-filter fallback: "
            f"trade_day={score_day:%Y-%m-%d} "
            f"expected_pool_day={pool_info.get('pool_day_expected')} "
            f"reason={pool_info.get('fallback_reason')} "
            f"nearest_pool_day={pool_info.get('nearest_pool_day')}"
        )
    pre_allowlist_count = int(len(universe))
    universe = apply_allowlist_filter_to_universe(universe, allowlist_codes=pool_codes)
    allowlist_diag = _allowlist_diagnostics(pool_info, pre_allowlist_count, int(len(universe)))
    if universe.empty:
        raise ValueError("live universe is empty after allowlist filter")

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
    picks["signal_day"] = score_day
    picks["buy_day"] = score_day
    picks["sell_day"] = target_day
    picks["target_day"] = target_day
    picks["score_day"] = score_day
    picks["trade_date"] = target_day
    picks.to_csv(out_dir / "trade_list.csv", index=False)
    allowlist_summary = {
        **allowlist_diag,
        "universe_filter": "o_0005_only",
        "target_day": target_day,
        "score_day": score_day,
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

