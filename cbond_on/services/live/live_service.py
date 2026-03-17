from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date
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


def _data_hub_runtime(live_cfg: dict) -> dict:
    cfg = dict(live_cfg.get("data_hub", {}))
    return {
        "python_exe": str(cfg.get("python_exe") or sys.executable),
        "module": str(cfg.get("module") or "cbond_data_hub"),
        "project_root": Path(str(cfg.get("project_root") or "C:/Users/BaiYang/CBOND_DATA_HUB")),
        "pg_config_path": str(cfg.get("pg_config_path") or "").strip(),
    }


def _append_bool_flag(args: list[str], name: str, enabled: bool) -> None:
    flag = f"--{name}" if bool(enabled) else f"--no-{name}"
    args.append(flag)


def _parse_json_output(text: str) -> dict:
    payload = str(text or "").strip()
    if not payload:
        return {}
    try:
        obj = json.loads(payload)
        return dict(obj) if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass

    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(payload[start : end + 1])
            return dict(obj) if isinstance(obj, dict) else {"value": obj}
        except Exception:
            pass
    raise ValueError(f"failed to parse Data Hub JSON output: {payload[:400]}")


def _run_data_hub(runtime: dict, args: list[str], *, expect_json: bool = True) -> dict:
    if not Path(runtime["project_root"]).exists():
        raise FileNotFoundError(
            f"data hub project_root not found: {runtime['project_root']}"
        )
    cmd = [runtime["python_exe"], "-m", runtime["module"], *args]
    proc = subprocess.run(
        cmd,
        cwd=str(runtime["project_root"]),
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    if proc.returncode != 0:
        raise RuntimeError(
            f"data hub command failed rc={proc.returncode}: {' '.join(cmd)}"
        )
    if not expect_json:
        return {}
    return _parse_json_output(stdout)


def _resolve_rebuild_window_via_hub(
    *,
    runtime: dict,
    raw_root: str,
    run_day: date,
    lookback_days: int,
) -> tuple[date, date]:
    payload = _run_data_hub(
        runtime,
        [
            "context",
            "resolve",
            "--raw-root",
            str(raw_root),
            "--today",
            str(run_day),
            "--now",
            f"{run_day}T12:00:00",
            "--lookback-days",
            str(max(0, int(lookback_days))),
            "--cutoff-time",
            "00:00",
            "--target-policy",
            "today",
        ],
        expect_json=True,
    )
    lookback_start = parse_date(str(payload["lookback_start_day"]))
    prev_trade_day = parse_date(str(payload["prev_trade_day"]))
    return lookback_start, prev_trade_day


def _run_history_sync_via_hub(
    *,
    runtime: dict,
    raw_root: str,
    start_day: date,
    end_day: date,
    mode: str,
    refresh: bool,
    overwrite: bool,
    raw_cfg: dict,
) -> dict:
    args = [
        "raw",
        "sync-history",
        "--raw-root",
        str(raw_root),
        "--start",
        str(start_day),
        "--end",
        str(end_day),
        "--mode",
        str(mode).strip().lower(),
    ]
    if refresh:
        args.append("--refresh")
    if overwrite:
        args.append("--overwrite")

    db_tables = list(dict(raw_cfg.get("db", {})).get("sync_tables", []) or [])
    if db_tables:
        args.extend(["--sync-tables", ",".join(str(x) for x in db_tables)])

    nfs_cfg = dict(raw_cfg.get("nfs", {}))
    nfs_root = str(nfs_cfg.get("nfs_root", "")).strip()
    nfs_base_dir = str(nfs_cfg.get("base_dir", "")).strip()
    if nfs_root:
        args.extend(["--nfs-root", nfs_root])
    if nfs_base_dir:
        args.extend(["--nfs-base-dir", nfs_base_dir])

    if runtime["pg_config_path"]:
        args.extend(["--pg-config-path", runtime["pg_config_path"]])

    return _run_data_hub(runtime, args, expect_json=True)


def _run_intraday_sync_via_hub(
    *,
    runtime: dict,
    raw_root: str,
    target_day: date,
    live_cfg: dict,
    data_cfg: dict,
) -> dict:
    source_cfg = dict(live_cfg.get("source", {}))
    intraday_cfg = dict(source_cfg.get("intraday", {}))
    redis_cfg = dict(live_cfg.get("redis", {}))

    host = str(redis_cfg.get("host", live_cfg.get("redis_host", ""))).strip()
    if not host:
        raise ValueError("live_config.redis.host is required when data.snapshot_source=redis")
    port = int(redis_cfg.get("port", live_cfg.get("redis_port", 6379)))
    db = int(redis_cfg.get("db", live_cfg.get("redis_db", 0)))
    password = redis_cfg.get("password", live_cfg.get("redis_password"))
    socket_timeout = redis_cfg.get("socket_timeout", live_cfg.get("redis_socket_timeout", 5))

    source = str(
        data_cfg.get("redis_source")
        or intraday_cfg.get("source")
        or live_cfg.get("source")
        or "combiner"
    ).strip() or "combiner"
    stage = str(
        data_cfg.get("redis_stage")
        or intraday_cfg.get("stage")
        or live_cfg.get("stage")
        or "raw"
    ).strip() or "raw"
    asset = str(
        data_cfg.get("redis_asset_type")
        or intraday_cfg.get("asset")
        or live_cfg.get("asset_type")
        or "cbond"
    ).strip() or "cbond"

    incremental = bool(
        data_cfg.get("redis_incremental", intraday_cfg.get("incremental", live_cfg.get("redis_incremental", True)))
    )
    full_day = bool(
        data_cfg.get("redis_full_day", intraday_cfg.get("full_day", live_cfg.get("redis_full_today", False)))
    )
    symbols = _parse_redis_symbols(data_cfg.get("redis_symbols"))
    watermark_path = str(
        data_cfg.get("redis_watermark_path")
        or redis_cfg.get("watermark_path")
        or intraday_cfg.get("watermark_path")
        or live_cfg.get("redis_watermark_path")
        or ""
    ).strip()

    args = [
        "raw",
        "sync-intraday",
        "--raw-root",
        str(raw_root),
        "--target-day",
        str(target_day),
        "--host",
        host,
        "--port",
        str(port),
        "--db",
        str(db),
        "--socket-timeout",
        str(socket_timeout),
        "--source",
        source,
        "--stage",
        stage,
        "--asset",
        asset,
    ]
    if password not in (None, ""):
        args.extend(["--password", str(password)])
    if symbols:
        args.extend(["--symbols", ",".join(symbols)])
    _append_bool_flag(args, "incremental", incremental)
    if full_day:
        args.append("--full-day")
    if watermark_path:
        args.extend(["--watermark-path", watermark_path])
    return _run_data_hub(runtime, args, expect_json=True)


def _run_clean_build_via_hub(
    *,
    runtime: dict,
    raw_root: str,
    clean_root: str,
    start_day: date,
    end_day: date,
    refresh: bool,
    overwrite: bool,
    data_cfg: dict,
    cleaned_cfg: dict,
) -> dict:
    snapshot_cfg = dict(cleaned_cfg.get("snapshot", {}))
    kline_enabled = bool(data_cfg.get("kline_enabled", cleaned_cfg.get("kline_enabled", True)))

    args = [
        "clean",
        "build",
        "--raw-root",
        str(raw_root),
        "--clean-root",
        str(clean_root),
        "--start",
        str(start_day),
        "--end",
        str(end_day),
        "--allowed-phases",
        ",".join(str(x) for x in snapshot_cfg.get("allowed_phases", ["T", "T0"])),
        "--price-field",
        str(snapshot_cfg.get("price_field", "last")),
    ]
    if refresh:
        args.append("--refresh")
    if overwrite:
        args.append("--overwrite")

    _append_bool_flag(args, "kline-enabled", kline_enabled)
    _append_bool_flag(args, "filter-trading-phase", bool(snapshot_cfg.get("filter_trading_phase", True)))
    _append_bool_flag(args, "drop-no-trade", bool(snapshot_cfg.get("drop_no_trade", True)))
    _append_bool_flag(args, "use-prev-snapshot", bool(snapshot_cfg.get("use_prev_snapshot", True)))
    return _run_data_hub(runtime, args, expect_json=True)


def _load_strategy_config(path_text: str | None) -> dict:
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)


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
    raw_cfg = load_config_file("raw_data")
    cleaned_cfg = load_config_file("cleaned_data")

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    data_cfg = dict(live_cfg.get("data", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))

    target_day = parse_date(target or schedule_cfg.get("target"))
    start_day = parse_date(start or schedule_cfg.get("start") or target_day)

    refresh_data = bool(data_cfg.get("refresh", False))
    overwrite_data = bool(data_cfg.get("overwrite", False))
    lookback_days = max(0, int(data_cfg.get("lookback_days", 0)))
    use_redis_snapshot = _redis_snapshot_enabled(data_cfg, live_cfg)

    raw_root = str(paths_cfg["raw_data_root"])
    clean_root = str(paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"))
    data_hub = _data_hub_runtime(live_cfg)

    run_day = _resolve_redis_sync_day(data_cfg, target_day) if use_redis_snapshot else start_day
    rebuild_start_day, prev_trade_day = _resolve_rebuild_window_via_hub(
        runtime=data_hub,
        raw_root=raw_root,
        run_day=run_day,
        lookback_days=lookback_days,
    )

    if use_redis_snapshot:
        raw_start_day = rebuild_start_day
        raw_end_day = prev_trade_day
        if raw_end_day < raw_start_day:
            raw_end_day = raw_start_day
        raw_mode = str(data_cfg.get("raw_sync_mode_when_redis", raw_cfg.get("mode", "both")))
        print(
            "live run window:",
            f"run_day={run_day}",
            f"target_day={target_day}",
            f"lookback_start={rebuild_start_day}",
            f"prev_trading_day={prev_trade_day}",
            f"raw_nfs_end={raw_end_day}",
        )
    else:
        raw_start_day = start_day
        raw_end_day = target_day
        raw_mode = str(raw_cfg.get("mode", "both"))

    _run_history_sync_via_hub(
        runtime=data_hub,
        raw_root=raw_root,
        start_day=raw_start_day,
        end_day=raw_end_day,
        mode=raw_mode,
        refresh=refresh_data,
        overwrite=overwrite_data,
        raw_cfg=raw_cfg,
    )

    if use_redis_snapshot:
        sync_result = _run_intraday_sync_via_hub(
            runtime=data_hub,
            raw_root=raw_root,
            target_day=run_day,
            live_cfg=live_cfg,
            data_cfg=data_cfg,
        )
        print(
            "redis snapshot sync:",
            f"day={run_day}",
            f"symbols={sync_result.get('symbols', 0)}",
            f"pulled_rows={sync_result.get('pulled_rows', 0)}",
            f"written_rows={sync_result.get('written_rows', 0)}",
        )

        _run_clean_build_via_hub(
            runtime=data_hub,
            raw_root=raw_root,
            clean_root=clean_root,
            start_day=rebuild_start_day,
            end_day=run_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
            data_cfg=data_cfg,
            cleaned_cfg=cleaned_cfg,
        )
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
        )
    else:
        _run_clean_build_via_hub(
            runtime=data_hub,
            raw_root=raw_root,
            clean_root=clean_root,
            start_day=start_day,
            end_day=target_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
            data_cfg=data_cfg,
            cleaned_cfg=cleaned_cfg,
        )
        label_end = target_day - timedelta(days=1)
        run_panel(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
        run_label(start=start_day, end=label_end, refresh=refresh_data, overwrite=overwrite_data)
        run_factor_build(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)

    model_id = str(model_cfg.get("model_id", "")).strip()
    if not model_id:
        raise ValueError("live_config missing model_score.model_id")

    model_start = model_cfg.get("start", start_day)
    model_end = model_cfg.get("end", target_day)
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

    model_result = run_model_score(
        model_id=model_id,
        start=model_start,
        end=model_end,
        label_cutoff=model_label_cutoff,
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
        try:
            _write_trades_to_db(
                trades=picks,
                trade_day=target_day,
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
