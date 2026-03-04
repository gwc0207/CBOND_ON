from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, parse_time
from cbond_on.core.universe import filter_tradable
from cbond_on.core.trading_days import list_trading_days_from_raw, _raw_path
from cbond_on.data.io import read_clean_daily, read_table_range
from cbond_on.data.clean import build_cleaned_snapshot, build_cleaned_kline
from cbond_on.models.score_io import load_scores_by_date
from cbond_on.run import model_score
from cbond_on.run import sync_data, build_cleaned_data, build_panels, factor_batch
from cbond_on.factor_batch.runner import run_factor_batch, build_signal_specs
from cbond_on.config import SnapshotConfig, ScheduleConfig
from cbond_on.data.panel import build_panels_with_labels, write_panel_data, _build_day_snapshot_sequence
from cbond_on.data.snapshot import read_snapshot_day
from cbond_on.live.redis_snapshot import RedisConfig, SnapshotRedisClient
from cbond_on.backtest.execution import apply_twap_bps


def _parse_target(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return parse_date(value)


def _last_trading_day(raw_root: str, target: date) -> date:
    days = list_trading_days_from_raw(raw_root, target - timedelta(days=10), target, kind="snapshot")
    if not days:
        return target
    days = sorted(days)
    return days[-1]


def _prev_trading_day(raw_root: str, target: date, lag_days: int = 1) -> date:
    lag_days = max(1, int(lag_days))
    # Scan enough calendar span to safely collect lag trading days.
    scan_days = max(20, lag_days * 8)
    days = list_trading_days_from_raw(
        raw_root, target - timedelta(days=scan_days), target - timedelta(days=1), kind="snapshot"
    )
    if not days:
        return target - timedelta(days=lag_days)
    days = sorted(days)
    if len(days) >= lag_days:
        return days[-lag_days]
    return days[0]


def _recent_trading_days(raw_root: str, target: date, lookback_days: int) -> list[date]:
    lookback_days = max(1, int(lookback_days))
    scan_days = max(lookback_days * 8, 60)
    scan_start = target - timedelta(days=scan_days)
    days = list_trading_days_from_raw(raw_root, scan_start, target, kind="snapshot")
    if not days:
        return []
    days.sort()
    return days[-lookback_days:]


def _read_twap_daily(raw_data_root: str, day: date) -> pd.DataFrame:
    df = read_table_range(raw_data_root, "market_cbond.daily_twap", day, day)
    if df.empty:
        return df
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
    return df


def _build_live_bin_history(
    *,
    raw_data_root: str,
    scores_by_date: dict[date, pd.DataFrame],
    target: date,
    bin_count: int,
    lookback_days: int,
    buy_twap_col: str,
    sell_twap_col: str,
    cost_bps: float,
) -> list[dict[int, float]]:
    lookback_days = max(1, int(lookback_days))
    if bin_count <= 1:
        return []
    all_days = sorted(d for d in scores_by_date.keys() if d < target)
    if not all_days:
        return []
    trading_days = list_trading_days_from_raw(
        raw_data_root, all_days[0], target, kind="snapshot"
    )
    if len(trading_days) < 2:
        return []
    day_to_next = {trading_days[i]: trading_days[i + 1] for i in range(len(trading_days) - 1)}
    history: list[dict[int, float]] = []
    for day in all_days:
        next_day = day_to_next.get(day)
        if next_day is None:
            continue
        score_df = scores_by_date.get(day, pd.DataFrame())
        if score_df.empty or "code" not in score_df.columns or "score" not in score_df.columns:
            continue
        buy_df = _read_twap_daily(raw_data_root, day)
        sell_df = _read_twap_daily(raw_data_root, next_day)
        if buy_df.empty or sell_df.empty:
            continue
        merged = buy_df.merge(
            sell_df[["code", sell_twap_col]], on="code", how="left", suffixes=("", "_next")
        )
        merged = merged.merge(score_df[["code", "score"]], on="code", how="left")
        required = [buy_twap_col, sell_twap_col]
        if any(c not in merged.columns for c in required):
            continue
        merged = merged[
            merged["score"].notna()
            & merged[buy_twap_col].notna()
            & merged[sell_twap_col].notna()
            & (merged[buy_twap_col] > 0)
            & (merged[sell_twap_col] > 0)
        ]
        if merged.empty:
            continue
        buy_all = apply_twap_bps(merged[buy_twap_col], cost_bps, side="buy")
        sell_all = apply_twap_bps(merged[sell_twap_col], cost_bps, side="sell")
        returns_all = (sell_all - buy_all) / buy_all
        try:
            bins_cat = pd.qcut(merged["score"], bin_count, labels=False, duplicates="drop")
        except Exception:
            bins_cat = None
        if bins_cat is None or bins_cat.dropna().empty:
            continue
        bin_df = pd.DataFrame(
            {"bin": bins_cat.values, "ret": returns_all.values}
        ).dropna()
        if bin_df.empty:
            continue
        means = bin_df.groupby("bin")["ret"].mean().to_dict()
        history.append({int(k): float(v) for k, v in means.items()})
    if len(history) > lookback_days:
        history = history[-lookback_days:]
    return history


def _normalize_trade_time(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_time" in df.columns:
        col = "trade_time"
    elif "timestamp" in df.columns:
        col = "timestamp"
    elif "ts" in df.columns:
        col = "ts"
    else:
        raise KeyError("redis snapshot missing trade_time/timestamp/ts column")
    out = df.copy()
    if pd.api.types.is_numeric_dtype(out[col]):
        # guess ms vs s
        sample = float(out[col].dropna().iloc[0]) if out[col].dropna().any() else 0.0
        unit = "ms" if sample > 1e12 else "s"
        out["trade_time"] = pd.to_datetime(out[col], unit=unit, errors="coerce")
    else:
        out["trade_time"] = pd.to_datetime(out[col], errors="coerce")
    return out


def _build_panel_from_redis(
    *,
    target: date,
    panel_data_root: str,
    raw_root: str,
    panel_cfg: dict,
    cleaned_cfg: dict,
    live_cfg: dict,
    write_panel: bool = True,
) -> bool:
    if not bool(live_cfg.get("redis_enabled", True)):
        return False
    cfg = RedisConfig(
        host=str(live_cfg.get("redis_host", "")),
        port=int(live_cfg.get("redis_port", 6379)),
        db=int(live_cfg.get("redis_db", 0)),
        password=live_cfg.get("redis_password"),
        socket_timeout=live_cfg.get("redis_socket_timeout", 5),
    )
    client = SnapshotRedisClient(cfg)
    asset_type = str(live_cfg.get("redis_asset_type", "cbond"))
    source = str(live_cfg.get("redis_source", "combiner"))
    stage = str(live_cfg.get("redis_stage", "raw"))
    limit = int(live_cfg.get("redis_limit", panel_cfg.get("count_points", 3000)))
    symbols = client.list_symbols(source=source, stage=stage, asset_type=asset_type)
    if not symbols:
        print("[live] redis symbols empty")
        return False
    df = client.read_latest_df(symbols, source, stage, asset_type=asset_type, limit=limit)
    if df.empty:
        print("[live] redis snapshot empty")
        return False
    if "code" not in df.columns:
        raise KeyError("redis snapshot missing code column")
    df = _normalize_trade_time(df)
    df = df.dropna(subset=["trade_time"])
    cutoff = parse_time(
        live_cfg.get("cutoff_time", live_cfg.get("data_cutoff", live_cfg.get("run_after", "14:30")))
    )
    cutoff_dt = pd.Timestamp.combine(target, cutoff)
    df = df[df["trade_time"] <= cutoff_dt]
    if df.empty:
        print("[live] redis snapshot empty after cutoff")
        return False
    if bool(live_cfg.get("redis_write_raw", True)):
        raw_path = _raw_path(Path(raw_root), target, kind="snapshot")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_path, index=False)
        print(f"[live] redis snapshot saved: {raw_path}")

    snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
    schedule = ScheduleConfig.from_dict(panel_cfg["schedule"]).to_schedule()
    count_points = int(panel_cfg.get("count_points", 3000))
    live_count_points = int(live_cfg.get("panel_count_points", 0))
    if live_count_points > 0:
        count_points = live_count_points
    if limit <= 0:
        limit = count_points
    count_points = min(count_points, limit)
    snapshot_columns = panel_cfg.get("snapshot_columns")
    lead_minutes = int(panel_cfg.get("lead_minutes", 0))

    if not write_panel:
        return True

    panel_df = _build_day_snapshot_sequence(
        df,
        target,
        schedule,
        snapshot_cfg,
        count_points=count_points,
        snapshot_columns=snapshot_columns,
        lead_minutes=lead_minutes,
    )
    if panel_df is None or panel_df.empty:
        lookback = int(live_cfg.get("lookback_days", 5))
        start_day = target - pd.Timedelta(days=lookback * 8)
        raw_days = list_trading_days_from_raw(raw_root, start_day, target - pd.Timedelta(days=1))
        raw_days = [d for d in raw_days if d < target][-lookback:]
        extra_frames = []
        for day in raw_days:
            raw_path = _raw_path(Path(raw_root), day, kind="snapshot")
            if not raw_path.exists():
                continue
            extra_frames.append(read_snapshot_day(raw_path, snapshot_cfg))
        if extra_frames:
            raw_df = pd.concat(extra_frames + [df], ignore_index=True)
            if "trade_time" in raw_df.columns:
                raw_df = raw_df.dropna(subset=["trade_time"])
            panel_df = _build_day_snapshot_sequence(
                raw_df,
                target,
                schedule,
                snapshot_cfg,
                count_points=count_points,
                snapshot_columns=snapshot_columns,
                lead_minutes=lead_minutes,
            )
        if panel_df is None or panel_df.empty:
            raise RuntimeError(
                f"[live] redis panel empty with count_points={count_points}; "
                f"raw history insufficient for {target}"
            )
    write_panel_data(
        panel_data_root,
        target,
        window_minutes=int(panel_cfg.get("window_minutes", [15])[0]),
        panel_name=panel_cfg.get("panel_name"),
        df=panel_df,
    )
    print(f"[live] redis panel written for {target}")
    return True


def _write_trades(out_dir: Path, trades: pd.DataFrame, trade_day: date) -> None:
    if trades is None or trades.empty:
        return
    work = trades.copy()
    if "trade_date" not in work.columns:
        work["trade_date"] = trade_day
    cols = [c for c in ["trade_date", "code", "weight", "score", "rank"] if c in work.columns]
    work = work[cols]
    work.to_csv(out_dir / "trade_list.csv", index=False)


def _write_trades_to_db(
    *,
    trades: pd.DataFrame,
    trade_day: date,
    table: str,
    mode: str = "replace_date",
) -> None:
    if trades is None or trades.empty:
        return
    if "code" not in trades.columns:
        raise ValueError("trade_list missing code column")

    from cbond_on.data.extract import connect

    work = trades.copy()
    # Be robust when trade_date comes from index instead of a normal column.
    if "trade_date" not in work.columns and work.index.name == "trade_date":
        work = work.reset_index()
    if "trade_date" not in work.columns:
        work["trade_date"] = trade_day
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date
    work["trade_date"] = work["trade_date"].fillna(trade_day)

    parts = work["code"].astype(str).str.split(".", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("code must be in instrument.exchange format, e.g. 110084.SH")
    work["instrument_code"] = parts[0]
    work["exchange_code"] = parts[1]

    if "weight" not in work.columns:
        work["weight"] = None
    if "score" not in work.columns:
        work["score"] = None
    if "rank" not in work.columns:
        work["rank"] = None

    cols = [
        "instrument_code",
        "exchange_code",
        "trade_date",
        "score",
        "weight",
        "rank",
    ]
    for col in cols:
        if col not in work.columns:
            work[col] = None
    payload = work[cols]

    insert_sql = (
        f"INSERT INTO {table} "
        "(instrument_code, exchange_code, trade_date, factor_value, weight, rank) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )

    with connect() as conn:
        cursor = conn.cursor()
        if mode == "replace_date":
            cursor.execute(f"DELETE FROM {table} WHERE trade_date = ?", trade_day)
        cursor.fast_executemany = True
        cursor.executemany(insert_sql, payload.values.tolist())
        conn.commit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CBOND_ON live pipeline")
    parser.add_argument(
        "--start",
        help="score/model start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--target",
        help="target date (YYYY-MM-DD) or 'today' to use current date",
    )
    args = parser.parse_args(argv)
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    panel_cfg = load_config_file("panel")
    cleaned_cfg = load_config_file("cleaned_data")
    model_type = str(live_cfg.get("model_type", "linear"))
    model_config = str(live_cfg.get("model_config", "models/linear/model"))
    model_cfg = load_config_file(model_config)

    raw_root = paths_cfg["raw_data_root"]
    clean_root = paths_cfg["clean_data_root"]
    results_root = Path(paths_cfg["results_root"])

    target_override = args.target if args.target else live_cfg.get("target")
    if not target_override:
        raise ValueError("live config missing target date")
    target = _parse_target(target_override)
    start_override = args.start if args.start else live_cfg.get("start", target)
    live_start = parse_date(start_override)
    trade_date_lag_days = int(live_cfg.get("trade_date_lag_days", 2))
    trade_date = _prev_trading_day(raw_root, target, lag_days=trade_date_lag_days)
    # Label causality cutoff is independent from DB trade_date lag:
    # for target T, training labels can use at most T-1 trading day.
    label_cutoff = _prev_trading_day(raw_root, target, lag_days=1)
    run_after = parse_time(
        live_cfg.get("cutoff_time", live_cfg.get("run_after", live_cfg.get("data_cutoff", "14:30")))
    )
    if pd.Timestamp.now().time() < run_after:
        print("warning: running before scheduled time")

    def _sync_segment(seg_start: date, seg_end: date, *, force_refresh: bool, tag: str) -> None:
        if seg_start > seg_end:
            return
        print(f"[live] sync raw [{tag}]: {seg_start} -> {seg_end} ...")
        sync_cfg = load_config_file("raw_data")
        sync_cfg = dict(sync_cfg)
        mode = str(sync_cfg.get("mode", "both")).lower()
        if "db" in sync_cfg:
            sync_cfg["db"] = dict(sync_cfg["db"])
            sync_cfg["db"]["start"] = str(seg_start)
            sync_cfg["db"]["end"] = str(seg_end)
            sync_cfg["db"]["refresh"] = bool(force_refresh)
            sync_cfg["db"]["overwrite"] = bool(force_refresh)
        if "nfs" in sync_cfg:
            sync_cfg["nfs"] = dict(sync_cfg["nfs"])
            sync_cfg["nfs"]["start"] = str(seg_start)
            sync_cfg["nfs"]["end"] = str(seg_end)
            sync_cfg["nfs"]["refresh"] = bool(force_refresh)
            sync_cfg["nfs"]["overwrite"] = bool(force_refresh)
        elif "ftp" in sync_cfg:
            # Backward compatibility
            sync_cfg["ftp"] = dict(sync_cfg["ftp"])
            sync_cfg["ftp"]["start"] = str(seg_start)
            sync_cfg["ftp"]["end"] = str(seg_end)
            sync_cfg["ftp"]["refresh"] = bool(force_refresh)
            sync_cfg["ftp"]["overwrite"] = bool(force_refresh)
        if mode in ("db", "both"):
            sync_data._sync_db(raw_root, sync_cfg.get("db", {}))
        if mode in ("nfs", "both"):
            sync_data._sync_nfs(raw_root, sync_cfg.get("nfs", {}))
        elif mode == "ftp":
            sync_data._sync_ftp(raw_root, sync_cfg.get("ftp", {}))

    def _clean_segment(seg_start: date, seg_end: date, *, overwrite: bool, tag: str) -> None:
        if seg_start > seg_end:
            return
        print(f"[live] build cleaned data [{tag}]: {seg_start} -> {seg_end} ...")
        snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
        build_cleaned_snapshot(
            raw_root,
            clean_root,
            seg_start,
            seg_end,
            snapshot_cfg,
            overwrite=overwrite,
        )
        build_cleaned_kline(
            raw_root,
            clean_root,
            seg_start,
            seg_end,
            overwrite=overwrite,
        )

    def _panel_segment(seg_start: date, seg_end: date, *, overwrite: bool, tag: str) -> None:
        if seg_start > seg_end:
            return
        print(f"[live] build panels [{tag}]: {seg_start} -> {seg_end} ...")
        schedule = ScheduleConfig.from_dict(panel_cfg["schedule"]).to_schedule()
        panel_name = panel_cfg.get("panel_name")
        windows = panel_cfg.get("window_minutes", [15])
        for w in windows:
            build_panels_with_labels(
                clean_root,
                paths_cfg["panel_data_root"],
                paths_cfg["label_data_root"],
                raw_root,
                seg_start,
                seg_end,
                schedule,
                SnapshotConfig.from_dict(cleaned_cfg["snapshot"]),
                panel_cfg.get("label", {}),
                window_minutes=int(w),
                panel_name=panel_name,
                overwrite=overwrite,
                panel_mode=str(panel_cfg.get("panel_mode", "snapshot_sequence")),
                count_points=int(panel_cfg.get("count_points", 3000)),
                max_lookback_days=int(panel_cfg.get("max_lookback_days", 3)),
                snapshot_columns=panel_cfg.get("snapshot_columns"),
                lead_minutes=int(panel_cfg.get("lead_minutes", 0)),
                label_end=label_cutoff,
            )

    def _factor_segment(seg_start: date, seg_end: date, *, overwrite: bool, tag: str) -> None:
        if seg_start > seg_end:
            return
        print(f"[live] build factors [{tag}]: {seg_start} -> {seg_end} ...")
        fb_cfg = load_config_file("factor_batch")
        fb_cfg = dict(fb_cfg)
        fb_cfg["backtest_enabled"] = False
        specs = build_signal_specs(fb_cfg)
        run_factor_batch(
            fb_cfg,
            panel_data_root=paths_cfg["panel_data_root"],
            factor_data_root=paths_cfg["factor_data_root"],
            label_data_root=paths_cfg["label_data_root"],
            raw_data_root=raw_root,
            results_root=paths_cfg["results_root"],
            start=seg_start,
            end=seg_end,
            window_minutes=int(fb_cfg.get("window_minutes", 15)),
            panel_name=fb_cfg.get("panel_name"),
            overwrite=overwrite,
            specs=specs,
        )

    lookback_days = int(live_cfg.get("lookback_days", 5))
    recent_days = _recent_trading_days(raw_root, target, lookback_days)
    history_start = recent_days[0] if recent_days else parse_date(live_cfg.get("start", target))
    history_end = recent_days[-1] if recent_days else target

    if bool(live_cfg.get("redis_enabled", True)) and bool(live_cfg.get("redis_write_raw", True)):
        raw_path = _raw_path(Path(paths_cfg["raw_data_root"]), target, kind="snapshot")
        redis_force = bool(live_cfg.get("redis_force", False))
        if raw_path.exists() and not redis_force:
            print(f"[live] redis -> raw skip (exists): {raw_path}")
        else:
            print(f"[live] redis -> raw: {target} ...")
            _build_panel_from_redis(
                target=target,
                panel_data_root=paths_cfg["panel_data_root"],
                raw_root=paths_cfg["raw_data_root"],
                panel_cfg=panel_cfg,
                cleaned_cfg=cleaned_cfg,
                live_cfg=live_cfg,
                write_panel=False,
            )
        if raw_path.exists():
            history_end = target

    hybrid_refresh = bool(live_cfg.get("hybrid_refresh", False))
    inner_start = max(live_start, history_start)
    outer_start = history_start
    outer_end = min(history_end, inner_start - timedelta(days=1))

    if hybrid_refresh:
        _sync_segment(outer_start, outer_end, force_refresh=False, tag="outer-incremental")
        _sync_segment(inner_start, target, force_refresh=True, tag="inner-refresh")
    else:
        _sync_segment(history_start, history_end, force_refresh=False, tag="single")

    if hybrid_refresh:
        _clean_segment(outer_start, outer_end, overwrite=False, tag="outer-incremental")
        _clean_segment(inner_start, target, overwrite=True, tag="inner-refresh")
    else:
        overwrite = bool(cleaned_cfg.get("overwrite", False))
        _clean_segment(history_start, history_end, overwrite=overwrite, tag="single")

    if hybrid_refresh:
        _panel_segment(outer_start, outer_end, overwrite=False, tag="outer-incremental")
        _panel_segment(inner_start, target, overwrite=True, tag="inner-refresh")
    else:
        overwrite = bool(panel_cfg.get("overwrite", False))
        _panel_segment(history_start, history_end, overwrite=overwrite, tag="single")
    _build_panel_from_redis(
        target=target,
        panel_data_root=paths_cfg["panel_data_root"],
        raw_root=paths_cfg["raw_data_root"],
        panel_cfg=panel_cfg,
        cleaned_cfg=cleaned_cfg,
        live_cfg=live_cfg,
    )

    if hybrid_refresh:
        _factor_segment(outer_start, outer_end, overwrite=False, tag="outer-incremental")
        _factor_segment(inner_start, target, overwrite=True, tag="inner-refresh")
    else:
        fb_cfg = load_config_file("factor_batch")
        overwrite = bool(fb_cfg.get("overwrite", False))
        _factor_segment(history_start, target, overwrite=overwrite, tag="single")

    print("[live] build model scores ...")
    model_score.main(
        model_type=model_type,
        model_config=model_config,
        start=str(live_start),
        end=str(target),
        label_cutoff=str(label_cutoff),
    )

    score_path = Path(model_cfg["score_output"])
    refresh_scores = bool(live_cfg.get("refresh_scores", False))
    if refresh_scores or not score_path.exists():
        model_score.main(
            model_type=model_type,
            model_config=model_config,
            start=str(live_start),
            end=str(target),
            label_cutoff=str(label_cutoff),
        )
    scores_by_date = load_scores_by_date(score_path)
    score_df = scores_by_date.get(target)
    if score_df is None or score_df.empty:
        raise ValueError(f"no scores for target date {target} in {score_path}")

    daily = read_clean_daily(clean_root, target)
    if daily.empty:
        print("[live] clean daily data empty for target; skip tradable filter")
        daily = None

    if daily is not None:
        merged = daily.merge(score_df[["code", "score"]], on="code", how="left")
        merged = merged[merged["score"].notna()]
        if merged.empty:
            raise ValueError("no scores matched to daily data")
    else:
        merged = score_df.copy()

    if daily is not None:
        buy_col = str(live_cfg.get("buy_twap_col", "twap_1442_1457"))
        sell_col = str(live_cfg.get("sell_twap_col", "twap_0930_1000"))
        if buy_col in merged.columns:
            merged = filter_tradable(
                merged,
                buy_twap_col=buy_col,
                sell_twap_col=sell_col,
                min_amount=float(live_cfg.get("min_amount", 0)),
                min_volume=float(live_cfg.get("min_volume", 0)),
            )
        else:
            print(f"[live] skip tradable filter: missing {buy_col}")
    if merged.empty:
        raise ValueError("no tradable symbols after filtering")

    bin_source = str(live_cfg.get("bin_source", "auto")).lower()
    bin_top_k = max(1, int(live_cfg.get("bin_top_k", 1)))
    bin_count = int(live_cfg.get("ic_bins", 20))
    bin_lookback_days = max(1, int(live_cfg.get("bin_lookback_days", 60)))
    buy_col = str(live_cfg.get("buy_twap_col", "twap_1442_1457"))
    sell_col = str(live_cfg.get("sell_twap_col", "twap_0930_1000"))
    cost_bps = float(live_cfg.get("twap_bps", 1.5)) + float(live_cfg.get("fee_bps", 0.7))
    if bin_count <= 1:
        bin_count = 2
    try:
        bins_cat = pd.qcut(merged["score"], bin_count, labels=False, duplicates="drop")
    except Exception:
        bins_cat = None
    if bins_cat is None or bins_cat.dropna().empty:
        raise ValueError("binning failed for live picks")
    merged = merged.copy()
    merged["bin"] = bins_cat.values
    available_bins = sorted(merged["bin"].dropna().unique().tolist())
    if not available_bins:
        raise ValueError("no bins available for live picks")
    if bin_source == "auto":
        bin_history = _build_live_bin_history(
            raw_data_root=raw_root,
            scores_by_date=scores_by_date,
            target=target,
            bin_count=bin_count,
            lookback_days=bin_lookback_days,
            buy_twap_col=buy_col,
            sell_twap_col=sell_col,
            cost_bps=cost_bps,
        )
        if bin_history:
            agg: dict[int, list[float]] = {}
            for rec in bin_history:
                for b, v in rec.items():
                    agg.setdefault(int(b), []).append(float(v))
            ranked = sorted(
                [(b, float(pd.Series(v).mean())) for b, v in agg.items()],
                key=lambda x: x[1],
                reverse=True,
            )
            chosen_bins = [b for b, _ in ranked][:bin_top_k] if ranked else []
            chosen_bins = [b for b in chosen_bins if b in set(available_bins)]
            if not chosen_bins:
                chosen_bins = sorted(available_bins, reverse=True)[:bin_top_k]
        else:
            chosen_bins = sorted(available_bins, reverse=True)[:bin_top_k]
    else:
        manual_bins = live_cfg.get("bin_select")
        if isinstance(manual_bins, list) and manual_bins:
            chosen_bins = [int(x) for x in manual_bins if int(x) in set(available_bins)]
            if not chosen_bins:
                chosen_bins = sorted(available_bins, reverse=True)[:bin_top_k]
        else:
            chosen_bins = sorted(available_bins, reverse=True)[:bin_top_k]
    print(
        f"[live] bin_select mode={bin_source} bins_actual={len(available_bins)} "
        f"chosen={chosen_bins} lookback_days={bin_lookback_days}"
    )
    picks = merged[merged["bin"].isin(chosen_bins)].sort_values("score", ascending=False).copy()
    if picks.empty:
        raise ValueError("no picks after bin selection")
    picks["rank"] = range(1, len(picks) + 1)
    weight = min(1.0 / len(picks), float(live_cfg.get("max_weight", 0.05)))
    picks["weight"] = weight

    out_dir = results_root / "live" / f"{target:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trades(out_dir, picks, trade_date)

    if bool(live_cfg.get("db_write", False)):
        _write_trades_to_db(
            trades=picks,
            trade_day=trade_date,
            table=live_cfg["db_table"],
            mode=live_cfg.get("db_mode", "replace_date"),
        )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
