from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, parse_time
from cbond_on.data.io import read_table_all


def _snapshot_path(snapshot_root: str, day: pd.Timestamp) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return Path(snapshot_root) / month / filename


def _load_snapshot_day(snapshot_root: str, day: pd.Timestamp) -> pd.DataFrame:
    path = _snapshot_path(snapshot_root, day)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["trade_time"]
    ):
        df = df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df


def _load_twap_day(raw_root: str, day: pd.Timestamp) -> pd.DataFrame:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = Path(raw_root) / "market_cbond__daily_twap" / month / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "instrument_code" in df.columns and "exchange_code" in df.columns:
        df = df.copy()
        df["code"] = (
            df["instrument_code"].astype(str) + "." + df["exchange_code"].astype(str)
        )
    return df


def _lob_columns(depth_levels: int) -> tuple[list[str], list[str]]:
    if depth_levels % 2 != 0:
        raise ValueError("depth_levels must be even (ask+bid)")
    half = depth_levels // 2
    ask_price = [f"ask_price{i}" for i in range(1, half + 1)]
    bid_price = [f"bid_price{i}" for i in range(1, half + 1)]
    ask_volume = [f"ask_volume{i}" for i in range(1, half + 1)]
    bid_volume = [f"bid_volume{i}" for i in range(1, half + 1)]
    price_cols = ask_price[::-1] + bid_price
    volume_cols = ask_volume[::-1] + bid_volume
    return price_cols, volume_cols


def _filter_trade_window(
    df: pd.DataFrame, day: pd.Timestamp, start_t: time, end_t: time
) -> pd.DataFrame:
    start_dt = datetime.combine(day.date(), start_t)
    end_dt = datetime.combine(day.date(), end_t)
    return df[(df["trade_time"] >= start_dt) & (df["trade_time"] <= end_dt)]


def _build_one_day(
    *,
    prev_days: list[pd.Timestamp],
    day: pd.Timestamp,
    next_day: pd.Timestamp,
    snapshot_root: str,
    raw_root: str,
    day_start_t: time,
    cutoff_t: time,
    prev_day_cutoff_t: time,
    sample_count: int,
    trade_phase_prefix: str | None,
    depth_levels: int,
    buy_twap_col: str,
    sell_twap_col: str,
    output_dir: Path,
    overwrite: bool,
) -> dict:
    df_day = _load_snapshot_day(snapshot_root, day)
    if df_day.empty:
        return {"day": day.date(), "status": "skip", "reason": "missing_snapshot"}

    df_day = df_day.copy()
    df_day["trade_time"] = pd.to_datetime(df_day["trade_time"])
    df_day = _filter_trade_window(df_day, day, day_start_t, cutoff_t)
    if df_day.empty:
        return {"day": day.date(), "status": "skip", "reason": "empty_window"}

    if "trading_phase_code" in df_day.columns and trade_phase_prefix:
        df_day = df_day[
            df_day["trading_phase_code"]
            .astype(str)
            .str.startswith(trade_phase_prefix)
        ]
        if df_day.empty:
            return {"day": day.date(), "status": "skip", "reason": "no_trading_phase"}

    price_cols, volume_cols = _lob_columns(depth_levels)
    required = ["code", "trade_time"] + price_cols + volume_cols
    missing_cols = [c for c in required if c not in df_day.columns]
    if missing_cols:
        return {"day": day.date(), "status": "skip", "reason": "missing_cols"}

    df_day = df_day[required].dropna()
    if df_day.empty:
        return {"day": day.date(), "status": "skip", "reason": "all_nan"}

    df_day = df_day.sort_values(["code", "trade_time"])

    twap_day = _load_twap_day(raw_root, day)
    twap_next = _load_twap_day(raw_root, next_day)
    if twap_day.empty or twap_next.empty:
        return {"day": day.date(), "status": "skip", "reason": "missing_twap"}

    merged_twap = twap_day[["code", buy_twap_col]].merge(
        twap_next[["code", sell_twap_col]], on="code", how="inner"
    )
    merged_twap = merged_twap[
        merged_twap[buy_twap_col].notna()
        & merged_twap[sell_twap_col].notna()
        & (merged_twap[buy_twap_col] > 0)
        & (merged_twap[sell_twap_col] > 0)
    ]
    if merged_twap.empty:
        return {"day": day.date(), "status": "skip", "reason": "no_label"}

    y_map = (
        merged_twap.set_index("code")[sell_twap_col]
        / merged_twap.set_index("code")[buy_twap_col]
        - 1.0
    )
    tradable_codes = set(y_map.index)
    df_day = df_day[df_day["code"].isin(tradable_codes)]
    if df_day.empty:
        return {"day": day.date(), "status": "skip", "reason": "no_tradable_snapshot"}

    prev_frames = []
    for prev_day in prev_days:
        df_prev = _load_snapshot_day(snapshot_root, prev_day)
        if df_prev.empty:
            continue
        df_prev = df_prev.copy()
        df_prev["trade_time"] = pd.to_datetime(df_prev["trade_time"])
        if "trading_phase_code" in df_prev.columns and trade_phase_prefix:
            df_prev = df_prev[
                df_prev["trading_phase_code"]
                .astype(str)
                .str.startswith(trade_phase_prefix)
            ]
        if df_prev.empty:
            continue
        missing_prev = [c for c in required if c not in df_prev.columns]
        if missing_prev:
            continue
        df_prev = _filter_trade_window(
            df_prev, prev_day, day_start_t, prev_day_cutoff_t
        )
        if df_prev.empty:
            continue
        df_prev = df_prev[required].dropna()
        df_prev = df_prev[df_prev["code"].isin(tradable_codes)]
        if df_prev.empty:
            continue
        df_prev = df_prev.sort_values(["code", "trade_time"])
        prev_frames.append(df_prev)

    samples = []
    labels = []
    codes = []
    used_counts = []
    used_day_counts = []
    used_prev_counts = []

    day_groups = {code: g for code, g in df_day.groupby("code", sort=False)}
    if prev_frames:
        df_prev_all = pd.concat(prev_frames, axis=0).sort_values(["code", "trade_time"])
    else:
        df_prev_all = pd.DataFrame(columns=required)
    prev_groups = {code: g for code, g in df_prev_all.groupby("code", sort=False)}

    for code in sorted(tradable_codes):
        day_group = day_groups.get(code)
        prev_group = prev_groups.get(code)
        if day_group is None and prev_group is None:
            continue
        parts = []
        if prev_group is not None and not prev_group.empty:
            parts.append(prev_group)
        if day_group is not None and not day_group.empty:
            parts.append(day_group)
        if not parts:
            continue

        combined = pd.concat(parts, axis=0).sort_values("trade_time")
        if len(combined) < sample_count:
            continue

        tail = combined.tail(sample_count)
        price = tail[price_cols].to_numpy(dtype=np.float32)
        volume = tail[volume_cols].to_numpy(dtype=np.float32)
        x = np.stack([price, volume], axis=0)  # (2, T, L)
        samples.append(x)
        labels.append(float(y_map.loc[code]))
        codes.append(code)
        used_counts.append(int(len(tail)))
        day_mask = tail["trade_time"].dt.date == day.date()
        used_day_counts.append(int(day_mask.sum()))
        used_prev_counts.append(int((~day_mask).sum()))

    if not samples:
        return {"day": day.date(), "status": "skip", "reason": "no_samples"}

    X = np.stack(samples, axis=0)
    y = np.asarray(labels, dtype=np.float32)
    meta = pd.DataFrame(
        {
            "trade_date": day.date(),
            "code": codes,
            "used_count": used_counts,
            "used_day_count": used_day_counts,
            "used_prev_count": used_prev_counts,
            "label": labels,
        }
    )

    out_day_dir = output_dir / f"{day:%Y%m%d}"
    if out_day_dir.exists() and not overwrite:
        return {"day": day.date(), "status": "skip", "reason": "exists"}
    out_day_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_day_dir / "X.npy", X)
    np.save(out_day_dir / "y.npy", y)
    meta.to_parquet(out_day_dir / "meta.parquet", index=False)

    return {"day": day.date(), "status": "ok", "samples": len(samples)}


def main() -> None:
    paths_cfg = load_config_file("paths")
    sync_cfg = load_config_file("sync_data")
    data_cfg = sync_cfg.get("data", {})
    ds_cfg = load_config_file("dataset")
    bt_cfg = load_config_file("backtest")

    raw_root = paths_cfg["raw_data_root"]
    clean_root = Path(paths_cfg["clean_data_root"])
    snapshot_root = data_cfg.get("snapshot_root")
    if not snapshot_root:
        raise ValueError("sync_data_config.json5 missing data.snapshot_root")

    start = parse_date(ds_cfg["start"])
    end = parse_date(ds_cfg["end"])
    day_start_t = parse_time(ds_cfg.get("day_start_time", "09:35"))
    cutoff_t = parse_time(ds_cfg.get("cutoff_time", "14:30"))
    prev_day_cutoff_t = parse_time(ds_cfg.get("prev_day_start_time", "15:00"))
    sample_count = int(ds_cfg.get("sample_count", 10000))
    max_lookback_days = int(ds_cfg.get("max_lookback_days", 5))
    trade_phase_prefix = ds_cfg.get("trade_phase_prefix", "T")
    depth_levels = int(ds_cfg.get("depth_levels", 10))
    output_dir = clean_root / str(ds_cfg.get("output_dir", "LOBDS"))
    refresh = bool(ds_cfg.get("refresh", False))
    overwrite = bool(ds_cfg.get("overwrite", False)) or refresh

    buy_twap_col = bt_cfg["buy_twap_col"]
    sell_twap_col = bt_cfg["sell_twap_col"]

    cal = read_table_all(raw_root, "metadata.trading_calendar")
    if cal.empty or "calendar_date" not in cal.columns:
        raise ValueError("trading_calendar not found in raw_data_root")
    cal = cal.copy()
    cal["calendar_date"] = pd.to_datetime(cal["calendar_date"]).dt.date
    if "prev_trade_date" in cal.columns:
        cal["prev_trade_date"] = pd.to_datetime(cal["prev_trade_date"]).dt.date
    if "next_trade_date" in cal.columns:
        cal["next_trade_date"] = pd.to_datetime(cal["next_trade_date"]).dt.date
    if "is_open" in cal.columns:
        cal_open = cal[cal["is_open"].astype(bool)]
    else:
        cal_open = cal

    days = cal_open["calendar_date"].dropna().unique().tolist()
    days.sort()

    prev_map = (
        cal.set_index("calendar_date")["prev_trade_date"].dropna().to_dict()
        if "prev_trade_date" in cal.columns
        else {}
    )
    next_map = (
        cal.set_index("calendar_date")["next_trade_date"].dropna().to_dict()
        if "next_trade_date" in cal.columns
        else {}
    )

    day_list = [pd.Timestamp(d) for d in days if start <= d <= end]
    if len(day_list) < 2:
        raise ValueError("need at least two trading days for labels")
    print(f"dataset range: {day_list[0].date()} -> {day_list[-1].date()}")
    print(f"trading days: {len(day_list)}")

    results = []
    for day in day_list:
        day_date = day.date()
        prev_day_date = prev_map.get(day_date)
        next_day_date = next_map.get(day_date)
        if not prev_day_date or not next_day_date:
            results.append(
                {"day": day_date, "status": "skip", "reason": "missing_prev_next_day"}
            )
            print(results[-1])
            continue
        prev_days = []
        cursor = prev_day_date
        while cursor and len(prev_days) < max_lookback_days:
            prev_days.append(pd.Timestamp(cursor))
            cursor = prev_map.get(cursor)

        next_day = pd.Timestamp(next_day_date)
        res = _build_one_day(
            prev_days=prev_days,
            day=day,
            next_day=next_day,
            snapshot_root=snapshot_root,
            raw_root=raw_root,
            day_start_t=day_start_t,
            cutoff_t=cutoff_t,
            prev_day_cutoff_t=prev_day_cutoff_t,
            sample_count=sample_count,
            trade_phase_prefix=trade_phase_prefix,
            depth_levels=depth_levels,
            buy_twap_col=buy_twap_col,
            sell_twap_col=sell_twap_col,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        results.append(res)
        print(res)

    summary = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    skipped = summary[summary["status"] != "ok"]
    print(f"ok days: {int((summary['status'] == 'ok').sum())}")
    print(f"skipped days: {len(skipped)}")


if __name__ == "__main__":
    main()
