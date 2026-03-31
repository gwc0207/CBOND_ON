from __future__ import annotations

from bisect import bisect_left
import json
from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_clean_daily
from cbond_on.models.score_io import load_scores_by_date
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.model.model_score_service import run as run_model_score
from cbond_on.strategies import StrategyRegistry
from cbond_on.strategies.base import StrategyContext


def _today_shanghai() -> date:
    return pd.Timestamp.now(tz="Asia/Shanghai").date()


def _assert_no_date_fields_in_live_config(schedule_cfg: dict, model_cfg: dict) -> None:
    schedule_forbidden = ["start", "target"]
    model_forbidden = ["start", "end"]

    bad_schedule = [
        key for key in schedule_forbidden
        if key in schedule_cfg and str(schedule_cfg.get(key)).strip() not in {"", "None", "none", "null"}
    ]
    bad_model = [
        key for key in model_forbidden
        if key in model_cfg and str(model_cfg.get(key)).strip() not in {"", "None", "none", "null"}
    ]

    if bad_schedule or bad_model:
        parts: list[str] = []
        if bad_schedule:
            parts.append(f"schedule.{','.join(bad_schedule)}")
        if bad_model:
            parts.append(f"model_score.{','.join(bad_model)}")
        raise ValueError(
            "live_config date fields are not allowed; "
            "live always resolves runtime day from current date. "
            f"remove: {', '.join(parts)}"
        )


def _parse_redis_symbols(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        text = str(value).replace("\n", ",").replace(";", ",")
        items = text.split(",")
    out = []
    for item in items:
        sym = str(item).strip()
        if sym:
            out.append(sym)
    return sorted(set(out))


def _redis_snapshot_enabled(data_cfg: dict, live_cfg: dict | None = None) -> bool:
    source_raw = data_cfg.get("snapshot_source")
    if source_raw in (None, "") and live_cfg:
        source_cfg = dict(live_cfg.get("source", {}))
        intraday_cfg = dict(source_cfg.get("intraday", {}))
        source_raw = intraday_cfg.get("type")
    source = str(source_raw or "raw").strip().lower()
    return source in {"redis", "redis_snapshot"}


def _resolve_redis_sync_day(data_cfg: dict, target_day: date) -> date:
    mode_raw = str(data_cfg.get("redis_sync_day", "today")).strip()
    mode = mode_raw.lower()
    if mode in ("", "today", "current"):
        return pd.Timestamp.now(tz="Asia/Shanghai").date()
    if mode == "target":
        return target_day
    return parse_date(mode_raw)


def _default_manifest_root(raw_root: str, clean_root: str) -> Path:
    raw_parent = Path(raw_root).resolve().parent
    clean_parent = Path(clean_root).resolve().parent
    if clean_parent == raw_parent:
        return raw_parent / "manifests"
    return raw_parent / "manifests"


def _data_hub_runtime(live_cfg: dict, *, raw_root: str, clean_root: str) -> dict:
    cfg = dict(live_cfg.get("data_hub", {}))
    manifest_root = str(cfg.get("manifest_root", "")).strip()
    if not manifest_root:
        manifest_root = str(_default_manifest_root(raw_root, clean_root))
    require_datasets = [str(x).strip().lower() for x in _parse_redis_symbols(cfg.get("require_datasets"))]
    if not require_datasets:
        require_datasets = ["raw", "clean"]
    return {
        "manifest_root": manifest_root,
        "require_datasets": require_datasets,
        "allow_partial_manifest": bool(cfg.get("allow_partial_manifest", False)),
        "require_done_marker": bool(cfg.get("require_done_marker", True)),
        "ready_gate_enabled": bool(cfg.get("ready_gate_enabled", True)),
    }


def _resolve_rebuild_window_local(
    *,
    raw_root: str,
    run_day: date,
    lookback_days: int,
) -> tuple[date, date]:
    days = list_available_trading_days_from_raw(
        raw_root,
        kind="snapshot",
        asset="cbond",
    )
    if not days:
        raise RuntimeError(f"no trading days found from raw root: {raw_root}")

    idx = bisect_left(days, run_day)
    prev_idx = max(0, idx - 1)
    prev_trade_day = days[prev_idx]

    lookback_n = max(0, int(lookback_days))
    if lookback_n <= 0:
        lookback_start = run_day
    else:
        start_idx = max(0, idx - lookback_n)
        lookback_start = days[start_idx] if start_idx < len(days) else run_day
    return lookback_start, prev_trade_day


def _publish_ready(status: dict, *, require_done_marker: bool) -> bool:
    manifests_ready = bool(status.get("manifests_ready", False))
    if not manifests_ready:
        return False
    if require_done_marker:
        return bool(status.get("done_exists", False))
    return True


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return dict(obj) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _run_publish_status_local(
    *,
    runtime: dict,
    trade_day: date,
) -> dict:
    manifest_root = Path(str(runtime["manifest_root"]))
    datasets = [
        str(x).strip().lower()
        for x in runtime.get("require_datasets", [])
        if str(x).strip()
    ]
    allow_partial = bool(runtime.get("allow_partial_manifest", False))
    require_done = bool(runtime.get("require_done_marker", True))

    manifests: dict[str, dict] = {}
    ready_flags: dict[str, bool] = {}
    run_ids: list[str] = []
    missing: list[str] = []
    failed: list[str] = []

    for ds in datasets:
        path = manifest_root / ds / f"{trade_day:%Y-%m-%d}.json"
        payload = _read_json_file(path)
        status = str(payload.get("status", "")).strip().lower()
        ok = bool(payload) and (status in {"", "success"})
        if not path.exists():
            ok = False
            missing.append(ds)
        elif not ok:
            failed.append(ds)
        run_id = str(payload.get("run_id", "")).strip()
        if run_id:
            run_ids.append(run_id)
        manifests[ds] = {
            "path": str(path),
            "status": status or ("success" if ok else "missing"),
            "run_id": run_id,
            "produced_at": str(payload.get("produced_at", "")).strip(),
        }
        ready_flags[ds] = ok

    if not datasets:
        manifests_ready = True
    elif allow_partial:
        manifests_ready = any(ready_flags.values())
    else:
        manifests_ready = all(ready_flags.values())

    done_path = manifest_root / "publish" / f"{trade_day:%Y-%m-%d}.done"
    done_exists = done_path.exists()
    done_payload = _read_json_file(done_path)
    done_run_id = str(done_payload.get("run_id", "")).strip()
    active_run_id = done_run_id or (run_ids[-1] if run_ids else "")
    run_id_complete = bool(active_run_id) and all((not rid) or rid == active_run_id for rid in run_ids)
    run_id_consistent = len(set([rid for rid in run_ids if rid])) <= 1

    reason = ""
    if missing:
        reason = f"missing manifests={','.join(missing)}"
    elif failed:
        reason = f"failed manifests={','.join(failed)}"

    ready = bool(manifests_ready) and (bool(done_exists) or not require_done)
    return {
        "trade_day": str(trade_day),
        "manifest_root": str(manifest_root),
        "require_datasets": datasets,
        "allow_partial_manifest": allow_partial,
        "manifests_ready": bool(manifests_ready),
        "done_exists": bool(done_exists),
        "done_exists_raw": bool(done_exists),
        "ready": bool(ready),
        "reason": reason,
        "done_path": str(done_path),
        "done_run_id": done_run_id,
        "active_run_id": active_run_id,
        "manifest_run_id_complete": bool(run_id_complete),
        "manifest_run_id_consistent": bool(run_id_consistent),
        "manifests": manifests,
    }


def _ensure_publish_ready_local(
    *,
    runtime: dict,
    trade_day: date,
) -> dict:
    status = _run_publish_status_local(
        runtime=runtime,
        trade_day=trade_day,
    )
    ready = _publish_ready(status, require_done_marker=bool(runtime["require_done_marker"]))
    print(
        "data hub publish status:",
        f"trade_day={trade_day}",
        f"ready={ready}",
        f"reason={status.get('reason', '')}",
    )
    if ready:
        return status

    raise RuntimeError(
        f"data hub publish not ready for {trade_day}: {status.get('reason', 'unknown')}"
    )


def _load_strategy_config(path_text: str | None) -> dict:
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)


def _load_live_factor_runtime(live_cfg: dict) -> tuple[str, dict]:
    factor_group = dict(live_cfg.get("factor", {}))
    factor_cfg_key = str(factor_group.get("config", "live/live_factors")).strip()
    if not factor_cfg_key:
        raise ValueError("live_config.factor.config must not be empty")
    factor_cfg = dict(load_config_file(factor_cfg_key))

    inline_factors = factor_cfg.get("factors")
    factor_files = factor_cfg.get("factor_files")
    has_inline = isinstance(inline_factors, list) and len(inline_factors) > 0
    has_files = isinstance(factor_files, list) and len(factor_files) > 0
    if has_files and len(factor_files) != 1:
        raise ValueError("live_factors must contain exactly one factor_files entry")
    if not has_inline and not has_files:
        raise ValueError("live_factors must define non-empty factors or one factor_files entry")
    return factor_cfg_key, factor_cfg


def _load_live_model_runtime(live_cfg: dict) -> tuple[str, dict, str]:
    model_group = dict(live_cfg.get("model_score", {}))
    model_cfg_key = str(model_group.get("config", "live/live_models")).strip()
    if not model_cfg_key:
        raise ValueError("live_config.model_score.config must not be empty")
    model_score_cfg = dict(load_config_file(model_cfg_key))

    models_raw = model_score_cfg.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ValueError("live_models.models must be a non-empty object")
    models = {str(k).strip(): v for k, v in models_raw.items() if str(k).strip()}
    if len(models) != 1:
        raise ValueError("live_models must contain exactly one model entry")
    only_model_id = next(iter(models.keys()))

    requested_model_id = str(
        model_group.get("model_id")
        or model_score_cfg.get("model_id")
        or model_score_cfg.get("default_model_id")
        or ""
    ).strip()
    if requested_model_id and requested_model_id != only_model_id:
        raise ValueError(
            "live model mismatch: "
            f"live_config.model_score.model_id={requested_model_id}, "
            f"live_models only model={only_model_id}"
        )

    model_score_cfg["model_id"] = only_model_id
    model_score_cfg["default_model_id"] = only_model_id
    return model_cfg_key, model_score_cfg, only_model_id


def _prev_holdings(results_root: Path, day: date) -> pd.DataFrame:
    if not results_root.exists():
        return pd.DataFrame(columns=["code", "weight"])
    dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name < f"{day:%Y-%m-%d}"])
    if not dirs:
        return pd.DataFrame(columns=["code", "weight"])
    latest = dirs[-1] / "trade_list.csv"
    if not latest.exists():
        return pd.DataFrame(columns=["code", "weight"])
    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(columns=["code", "weight"])
    if "code" not in df.columns:
        return pd.DataFrame(columns=["code", "weight"])
    if "weight" not in df.columns:
        df["weight"] = 0.0
    return df[["code", "weight"]].copy()


def _write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
    backend: str | None = None,
) -> None:
    if trades is None or trades.empty:
        return
    from cbond_on.data.extract import (
        connect_backend,
        get_db_backend,
        normalize_table_name_for_backend,
        resolve_table_target_for_backend,
    )

    backend_name = str(backend or get_db_backend())
    db_override, resolved_table = resolve_table_target_for_backend(table, backend_name)
    table_name = normalize_table_name_for_backend(
        resolved_table,
        backend_name,
        database=db_override,
    )
    marker = "%s" if backend_name == "postgres" else "?"

    work = trades.copy()
    if "trade_date" not in work.columns:
        work["trade_date"] = trade_day
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date.fillna(trade_day)
    parts = work["code"].astype(str).str.split(".", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("code must use instrument.exchange format")
    work["instrument_code"] = parts[0]
    work["exchange_code"] = parts[1]
    if "score" not in work.columns:
        work["score"] = pd.NA
    if "weight" not in work.columns:
        work["weight"] = pd.NA
    if "rank" not in work.columns:
        work["rank"] = pd.NA
    payload = work[
        ["instrument_code", "exchange_code", "trade_date", "score", "weight", "rank"]
    ].values.tolist()

    insert_sql = (
        f"INSERT INTO {table_name} "
        "(instrument_code, exchange_code, trade_date, factor_value, weight, rank) "
        f"VALUES ({marker}, {marker}, {marker}, {marker}, {marker}, {marker})"
    )
    with connect_backend(backend_name, database=db_override) as conn:
        cursor = conn.cursor()
        if mode == "replace_date":
            cursor.execute(
                f"DELETE FROM {table_name} WHERE trade_date = {marker}",
                (trade_day,),
            )
        if backend_name != "postgres":
            try:
                cursor.fast_executemany = True
            except Exception:
                pass
        cursor.executemany(insert_sql, payload)
        conn.commit()


def _resolve_score_df_for_target(
    score_cache: dict[date, pd.DataFrame],
    score_day: date,
    score_path: Path,
) -> pd.DataFrame:
    score_df = score_cache.get(score_day, pd.DataFrame())
    if score_df is None or score_df.empty:
        raise ValueError(f"no scores for {score_day} in {score_path}")
    return score_df


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
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))
    factor_cfg_key, live_factor_cfg = _load_live_factor_runtime(live_cfg)
    model_cfg_key, live_model_score_cfg, model_id = _load_live_model_runtime(live_cfg)

    _assert_no_date_fields_in_live_config(schedule_cfg, model_cfg)

    today = _today_shanghai()
    target_day = parse_date(target) if target is not None else today
    start_day = parse_date(start) if start is not None else target_day

    refresh_data = bool(data_cfg.get("refresh", False))
    overwrite_data = bool(data_cfg.get("overwrite", False))
    lookback_days = max(0, int(data_cfg.get("lookback_days", 0)))
    use_redis_snapshot = _redis_snapshot_enabled(data_cfg, live_cfg)

    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"))
    data_hub = _data_hub_runtime(
        live_cfg,
        raw_root=raw_root,
        clean_root=clean_root,
    )

    run_day = _resolve_redis_sync_day(data_cfg, target_day) if use_redis_snapshot else target_day
    rebuild_start_day, prev_trade_day = _resolve_rebuild_window_local(
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
    _ensure_publish_ready_local(
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
    score_path = Path(model_result.get("score_output") or (Path(paths_cfg["results_root"]) / "scores" / model_id))
    score_cache = load_scores_by_date(score_path)
    score_df = _resolve_score_df_for_target(score_cache, score_day, score_path)

    clean_daily = read_clean_daily(clean_root, score_day)
    if clean_daily.empty:
        universe = score_df[["code", "score"]].copy()
    else:
        universe = clean_daily.merge(score_df[["code", "score"]], on="code", how="inner")
        if universe.empty:
            raise ValueError("no score matched to clean data")
        buy_col = str(output_cfg.get("buy_twap_col", data_cfg.get("buy_twap_col", "twap_1442_1457")))
        sell_col = str(output_cfg.get("sell_twap_col", data_cfg.get("sell_twap_col", "twap_0930_1000")))
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
    strategy = StrategyRegistry.get(strategy_id)
    strategy_config = _load_strategy_config(strategy_cfg.get("strategy_config_path"))
    strategy_config = strategy_config or {k: v for k, v in strategy_cfg.items() if k != "strategy_id"}
    prev_positions = _prev_holdings(Path(paths_cfg["results_root"]) / "live", target_day)
    picks = strategy.select(
        universe[["code", "score"]],
        ctx=StrategyContext(trade_date=target_day, prev_positions=prev_positions, config=strategy_config),
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
            _write_trades_to_db(
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
