from __future__ import annotations

from bisect import bisect_left, bisect_right
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.universe import filter_tradable
from cbond_on.data.io import read_clean_daily, read_trading_calendar
from cbond_on.live.redis_snapshot import RedisConfig, SnapshotRedisClient
from cbond_on.models.score_io import load_scores_by_date
from cbond_on.services.data.clean_service import run as run_clean
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.data.raw_service import run as run_raw
from cbond_on.services.model.model_score_service import run as run_model_score
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.strategies import StrategyRegistry
from cbond_on.strategies.base import StrategyContext


def _raw_snapshot_path(raw_root: Path, day: date) -> Path:
    month = f"{day:%Y-%m}"
    filename = f"{day:%Y%m%d}.parquet"
    return raw_root / "snapshot" / "cbond" / "raw_data" / month / filename


def _load_watermark(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    for k, v in dict(data).items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _save_watermark(path: Path, data: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k): float(v) for k, v in data.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_trade_time(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    valid_num = num.dropna()
    if not valid_num.empty:
        sample = float(valid_num.abs().median())
        if sample >= 1e12:
            out = pd.to_datetime(num, unit="ms", utc=True, errors="coerce")
            return out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
        if sample >= 1e9:
            out = pd.to_datetime(num, unit="s", utc=True, errors="coerce")
            return out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

    out = pd.to_datetime(series, errors="coerce")
    tz = getattr(out.dt, "tz", None)
    if tz is not None:
        return out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return out


def _max_trade_time_ms(series: pd.Series) -> float | None:
    num = pd.to_numeric(series, errors="coerce")
    valid_num = num.dropna()
    if not valid_num.empty:
        sample = float(valid_num.abs().median())
        scale = 1.0 if sample >= 1e12 else 1000.0
        return float(valid_num.max() * scale)

    dt = _normalize_trade_time(series).dropna()
    if dt.empty:
        return None
    ts = pd.Timestamp(dt.max())
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    else:
        ts = ts.tz_convert("Asia/Shanghai")
    return float(ts.tz_convert("UTC").timestamp() * 1000.0)


def _append_raw_snapshot(raw_root: Path, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    work = df.copy()
    if "trade_time" not in work.columns:
        if "timestamp" in work.columns:
            work["trade_time"] = work["timestamp"]
        elif "ts" in work.columns:
            work["trade_time"] = work["ts"]
        else:
            raise KeyError("redis snapshot missing trade_time/timestamp/ts")
    work["trade_time"] = _normalize_trade_time(work["trade_time"])
    work = work[work["trade_time"].notna()].copy()
    if work.empty:
        return 0

    if "code" not in work.columns and "symbol" in work.columns:
        work["code"] = work["symbol"]
    if "symbol" not in work.columns and "code" in work.columns:
        work["symbol"] = work["code"]
    if "code" not in work.columns:
        raise KeyError("redis snapshot missing code/symbol")

    work["trade_date"] = work["trade_time"].dt.date
    written = 0
    for day, group in work.groupby("trade_date"):
        path = _raw_snapshot_path(raw_root, day)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = group.copy()
        if path.exists():
            existing = pd.read_parquet(path)
            if existing is not None and not existing.empty:
                merged = pd.concat([existing, group], ignore_index=True)
        dedup_keys = [c for c in ("code", "trade_time") if c in merged.columns]
        if len(dedup_keys) == 2:
            merged = merged.sort_values("trade_time")
            merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
        merged.to_parquet(path, index=False)
        written += int(len(group))
    return written


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


def _redis_snapshot_enabled(data_cfg: dict) -> bool:
    source = str(data_cfg.get("snapshot_source", "raw")).strip().lower()
    return source == "redis"


def _resolve_redis_sync_day(data_cfg: dict, target_day: date) -> date:
    mode_raw = str(data_cfg.get("redis_sync_day", "today")).strip()
    mode = mode_raw.lower()
    if mode in ("", "today", "current"):
        return pd.Timestamp.now(tz="Asia/Shanghai").date()
    if mode == "target":
        return target_day
    return parse_date(mode_raw)


def _load_open_trading_days(raw_root: Path) -> list[date]:
    cal = read_trading_calendar(raw_root)
    if cal is None or cal.empty or "calendar_date" not in cal.columns:
        return []
    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    work = work[work["calendar_date"].notna()]
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = sorted(set(work["calendar_date"].tolist()))
    return days


def _previous_trading_day(open_days: list[date], ref_day: date) -> date:
    if not open_days:
        return ref_day - timedelta(days=1)
    idx = bisect_left(open_days, ref_day) - 1
    if idx >= 0:
        return open_days[idx]
    return ref_day - timedelta(days=1)


def _lookback_start_day(open_days: list[date], end_day: date, lookback_days: int) -> date:
    lb = max(0, int(lookback_days))
    if lb <= 0:
        return end_day
    if not open_days:
        return end_day - timedelta(days=lb)
    end_idx = bisect_right(open_days, end_day) - 1
    if end_idx < 0:
        return end_day - timedelta(days=lb)
    start_idx = max(0, end_idx - lb)
    return open_days[start_idx]


def _sync_snapshot_from_redis(
    *,
    raw_root: Path,
    target_day: date,
    live_cfg: dict,
    data_cfg: dict,
    run_day_root: Path,
) -> dict[str, int]:
    redis_cfg = dict(live_cfg.get("redis", {}))
    host = str(redis_cfg.get("host", live_cfg.get("redis_host", ""))).strip()
    if not host:
        raise ValueError("live_config.redis.host is required when data.snapshot_source=redis")
    port = int(redis_cfg.get("port", live_cfg.get("redis_port", 6379)))
    db = int(redis_cfg.get("db", live_cfg.get("redis_db", 0)))
    password = redis_cfg.get("password", live_cfg.get("redis_password"))
    socket_timeout = redis_cfg.get("socket_timeout", live_cfg.get("redis_socket_timeout"))
    timeout_val = None if socket_timeout in (None, "", 0, "0") else float(socket_timeout)

    source = str(data_cfg.get("redis_source", live_cfg.get("source", "combiner"))).strip() or "combiner"
    stage = str(data_cfg.get("redis_stage", live_cfg.get("stage", "raw"))).strip() or "raw"
    asset_type = str(data_cfg.get("redis_asset_type", live_cfg.get("asset_type", "cbond"))).strip() or "cbond"

    client = SnapshotRedisClient(
        RedisConfig(
            host=host,
            port=port,
            db=db,
            password=None if password in ("", None) else str(password),
            socket_timeout=timeout_val,
        )
    )
    symbols = _parse_redis_symbols(data_cfg.get("redis_symbols"))
    if not symbols:
        symbols = client.list_symbols(source=source, stage=stage, asset_type=asset_type)
    if not symbols:
        return {"symbols": 0, "pulled_rows": 0, "written_rows": 0, "watermark_updated": 0}

    day_start = pd.Timestamp(f"{target_day:%Y-%m-%d} 00:00:00", tz="Asia/Shanghai")
    day_end = day_start + pd.Timedelta(days=1)
    day_start_ms = int(day_start.tz_convert("UTC").timestamp() * 1000)
    day_end_ms = int(day_end.tz_convert("UTC").timestamp() * 1000) - 1
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    query_end_ms = min(day_end_ms, now_ms)

    incremental = bool(data_cfg.get("redis_incremental", live_cfg.get("redis_incremental", True)))
    full_day = bool(data_cfg.get("redis_full_day", live_cfg.get("redis_full_today", False)))

    watermark_path_raw = (
        data_cfg.get("redis_watermark_path")
        or redis_cfg.get("watermark_path")
        or live_cfg.get("redis_watermark_path")
        or ""
    )
    watermark_path = Path(str(watermark_path_raw)) if str(watermark_path_raw).strip() else (run_day_root / "redis_watermark.json")
    watermark = _load_watermark(watermark_path) if incremental else {}
    if full_day:
        watermark = {}

    frames: list[pd.DataFrame] = []
    pulled_rows = 0
    watermark_updated = False
    for sym in symbols:
        start_ms = day_start_ms
        if incremental:
            start_ms = int(max(day_start_ms, watermark.get(sym, day_start_ms))) + 1
        records = client.read_range(
            sym,
            source=source,
            stage=stage,
            asset_type=asset_type,
            start_time=start_ms,
            end_time=query_end_ms,
        )
        if not records:
            continue
        df_sym = pd.DataFrame(records)
        if df_sym.empty:
            continue
        frames.append(df_sym)
        pulled_rows += int(len(df_sym))
        if incremental and "trade_time" in df_sym.columns:
            max_ms = _max_trade_time_ms(df_sym["trade_time"])
            if max_ms is not None:
                prev_ms = float(watermark.get(sym, day_start_ms))
                if max_ms > prev_ms:
                    watermark[sym] = max_ms
                    watermark_updated = True

    if not frames:
        return {
            "symbols": int(len(symbols)),
            "pulled_rows": 0,
            "written_rows": 0,
            "watermark_updated": 0,
        }

    merged = pd.concat(frames, ignore_index=True)
    written_rows = _append_raw_snapshot(raw_root, merged)
    if incremental and watermark_updated:
        _save_watermark(watermark_path, watermark)

    return {
        "symbols": int(len(symbols)),
        "pulled_rows": int(pulled_rows),
        "written_rows": int(written_rows),
        "watermark_updated": int(watermark_updated),
    }


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

    schedule_cfg = dict(live_cfg.get("schedule", {}))
    data_cfg = dict(live_cfg.get("data", {}))
    model_cfg = dict(live_cfg.get("model_score", {}))
    strategy_cfg = dict(live_cfg.get("strategy", {}))
    output_cfg = dict(live_cfg.get("output", {}))

    target_day = parse_date(target or schedule_cfg.get("target"))
    start_day = parse_date(start or schedule_cfg.get("start") or target_day)

    refresh_data = bool(data_cfg.get("refresh", False))
    overwrite_data = bool(data_cfg.get("overwrite", False))
    use_redis_snapshot = _redis_snapshot_enabled(data_cfg)
    lookback_days = max(0, int(data_cfg.get("lookback_days", 0)))
    raw_root = Path(paths_cfg["raw_data_root"])
    run_day = _resolve_redis_sync_day(data_cfg, target_day) if use_redis_snapshot else start_day
    open_days = _load_open_trading_days(raw_root)
    prev_trade_day = _previous_trading_day(open_days, run_day)
    rebuild_start_day = _lookback_start_day(open_days, run_day, lookback_days)
    rebuild_end_day = run_day

    raw_start_day = start_day
    raw_end_day = target_day
    raw_cfg_override = None
    if use_redis_snapshot:
        # NFS keeps historical lookback; today's increment comes from Redis snapshot.
        raw_start_day = rebuild_start_day
        raw_end_day = prev_trade_day
        if raw_end_day < raw_start_day:
            raw_end_day = raw_start_day
        raw_cfg_override = dict(load_config_file("raw_data"))
        raw_cfg_override["mode"] = (
            str(data_cfg.get("raw_sync_mode_when_redis", raw_cfg_override.get("mode", "both")))
            .strip()
            .lower()
            or "both"
        )
        print(
            "live run window:",
            f"run_day={run_day}",
            f"target_day={target_day}",
            f"lookback_start={rebuild_start_day}",
            f"prev_trading_day={prev_trade_day}",
            f"raw_nfs_end={raw_end_day}",
        )
    run_raw(
        start=raw_start_day,
        end=raw_end_day,
        refresh=refresh_data,
        overwrite=overwrite_data,
        cfg=raw_cfg_override,
    )
    if use_redis_snapshot:
        day_root = Path(paths_cfg["results_root"]) / "live" / f"{run_day:%Y-%m-%d}"
        day_root.mkdir(parents=True, exist_ok=True)
        redis_sync_day = run_day
        sync_result = _sync_snapshot_from_redis(
            raw_root=raw_root,
            target_day=redis_sync_day,
            live_cfg=live_cfg,
            data_cfg=data_cfg,
            run_day_root=day_root,
        )
        print(
            "redis snapshot sync:",
            f"day={redis_sync_day}",
            f"symbols={sync_result['symbols']}",
            f"pulled_rows={sync_result['pulled_rows']}",
            f"written_rows={sync_result['written_rows']}",
        )
    if use_redis_snapshot:
        run_clean(
            start=rebuild_start_day,
            end=rebuild_end_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
        )
        run_panel(
            start=rebuild_start_day,
            end=rebuild_end_day,
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
            end=rebuild_end_day,
            refresh=refresh_data,
            overwrite=overwrite_data,
        )
    else:
        label_end = target_day - timedelta(days=1)
        run_clean(start=start_day, end=target_day, refresh=refresh_data, overwrite=overwrite_data)
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
        # Score today's cross section only; labels are available up to previous trading day.
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

    clean_daily = read_clean_daily(paths_cfg["clean_data_root"], score_day)
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

