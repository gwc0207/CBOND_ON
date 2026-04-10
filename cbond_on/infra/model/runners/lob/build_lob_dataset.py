from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, parse_time
from cbond_on.core.universe import filter_tradable
from cbond_on.infra.data.io import read_table_all


def _snapshot_path(snapshot_root: str, day: pd.Timestamp) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return Path(snapshot_root) / month / filename


def _load_snapshot_day(
    snapshot_root: str,
    day: pd.Timestamp,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    path = _snapshot_path(snapshot_root, day)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, columns=columns)
    if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["trade_time"]
    ):
        df = df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])
    return df


def _load_snapshot_codes(
    snapshot_root: str,
    day: pd.Timestamp,
    columns: list[str],
    codes: list[str],
) -> pd.DataFrame:
    path = _snapshot_path(snapshot_root, day)
    if not path.exists() or not codes:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(
            path,
            columns=columns,
            filters=[("code", "in", codes)],
        )
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        if "code" in df.columns:
            df = df[df["code"].isin(codes)]
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


def _normalize_x(x: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore_sample":
        mean = x.mean(axis=(1, 2), keepdims=True)
        std = x.std(axis=(1, 2), keepdims=True)
        std = np.where(std > 0, std, 1.0)
        return (x - mean) / std
    if method == "minmax_sample":
        x_min = x.min(axis=(1, 2), keepdims=True)
        x_max = x.max(axis=(1, 2), keepdims=True)
        denom = np.where((x_max - x_min) > 0, (x_max - x_min), 1.0)
        return (x - x_min) / denom
    return x


def _normalize_y(y: np.ndarray, method: str) -> tuple[np.ndarray, float, float]:
    if y.size == 0:
        return y, float("nan"), float("nan")
    if method == "zscore_day":
        mean = float(y.mean())
        std = float(y.std())
        if std > 0:
            return (y - mean) / std, mean, std
        return y - mean, mean, std
    return y, float("nan"), float("nan")


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
    code_batch_size: int,
    depth_levels: int,
    normalize_x: bool,
    normalize_y: bool,
    x_norm_method: str,
    y_norm_method: str,
    buy_twap_col: str,
    sell_twap_col: str,
    min_amount: float,
    min_volume: float,
    output_dir: Path,
    overwrite: bool,
) -> dict:
    price_cols, volume_cols = _lob_columns(depth_levels)
    required = ["code", "trade_time"] + price_cols + volume_cols
    if trade_phase_prefix:
        required.append("trading_phase_code")

    twap_day = _load_twap_day(raw_root, day)
    twap_next = _load_twap_day(raw_root, next_day)
    if twap_day.empty or twap_next.empty:
        return {"day": day.date(), "status": "skip", "reason": "missing_twap"}

    left_cols = ["code", buy_twap_col]
    if "amount" in twap_day.columns:
        left_cols.append("amount")
    if "volume" in twap_day.columns:
        left_cols.append("volume")
    merged_twap = twap_day[left_cols].merge(
        twap_next[["code", sell_twap_col]], on="code", how="inner"
    )
    merged_twap = filter_tradable(
        merged_twap,
        buy_twap_col=buy_twap_col,
        sell_twap_col=sell_twap_col,
        min_amount=float(min_amount),
        min_volume=float(min_volume),
    )
    if merged_twap.empty:
        return {"day": day.date(), "status": "skip", "reason": "no_label"}

    y_map = (
        merged_twap.set_index("code")[sell_twap_col]
        / merged_twap.set_index("code")[buy_twap_col]
        - 1.0
    )
    samples = []
    labels = []
    codes = []
    used_counts = []
    used_day_counts = []
    used_prev_counts = []
    tradable_list = sorted(y_map.index)
    if not tradable_list:
        return {"day": day.date(), "status": "skip", "reason": "no_tradable_snapshot"}

    batch_size = max(1, int(code_batch_size))
    for start in range(0, len(tradable_list), batch_size):
        batch_codes = tradable_list[start : start + batch_size]
        df_day = _load_snapshot_codes(snapshot_root, day, required, batch_codes)
        if df_day.empty:
            continue

        df_day = _filter_trade_window(df_day, day, day_start_t, cutoff_t)
        if df_day.empty:
            continue

        if "trading_phase_code" in df_day.columns and trade_phase_prefix:
            df_day = df_day[
                df_day["trading_phase_code"]
                .astype(str)
                .str.startswith(trade_phase_prefix)
            ]
            if df_day.empty:
                continue

        missing_cols = [c for c in required if c not in df_day.columns]
        if missing_cols:
            return {"day": day.date(), "status": "skip", "reason": "missing_cols"}

        df_day = df_day[required].dropna()
        if df_day.empty:
            continue
        df_day = df_day.sort_values(["code", "trade_time"])

        prev_frames = []
        for prev_day in prev_days:
            df_prev = _load_snapshot_codes(snapshot_root, prev_day, required, batch_codes)
            if df_prev.empty:
                continue
            if "trading_phase_code" in df_prev.columns and trade_phase_prefix:
                df_prev = df_prev[
                    df_prev["trading_phase_code"]
                    .astype(str)
                    .str.startswith(trade_phase_prefix)
                ]
            if df_prev.empty:
                continue
            df_prev = _filter_trade_window(
                df_prev, prev_day, day_start_t, prev_day_cutoff_t
            )
            if df_prev.empty:
                continue
            df_prev = df_prev[required].dropna()
            if df_prev.empty:
                continue
            df_prev = df_prev.sort_values(["code", "trade_time"])
            prev_frames.append(df_prev)

        day_groups = {code: g for code, g in df_day.groupby("code", sort=False)}
        if prev_frames:
            df_prev_all = pd.concat(prev_frames, axis=0).sort_values(["code", "trade_time"])
        else:
            df_prev_all = pd.DataFrame(columns=required)
        prev_groups = {code: g for code, g in df_prev_all.groupby("code", sort=False)}

        for code in batch_codes:
            if code not in y_map.index:
                continue
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
            if normalize_x:
                x = _normalize_x(x, x_norm_method)
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
    y_mean = float("nan")
    y_std = float("nan")
    if normalize_y:
        y, y_mean, y_std = _normalize_y(y, y_norm_method)
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
    if normalize_y:
        meta["label_norm"] = y
        meta["label_mean"] = y_mean
        meta["label_std"] = y_std

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
    sync_cfg = load_config_file("raw_data")
    data_cfg = sync_cfg.get("data", {})
    ds_cfg = load_config_file("models/lob/dataset")
    bt_cfg = load_config_file("backtest")

    raw_root = paths_cfg["raw_data_root"]
    clean_root = Path(paths_cfg["clean_data_root"])
    snapshot_root = data_cfg.get("snapshot_root")
    if not snapshot_root:
        raise ValueError("data/raw_data_config.json5 missing data.snapshot_root")

    start = parse_date(ds_cfg["start"])
    end = parse_date(ds_cfg["end"])
    day_start_t = parse_time(ds_cfg.get("day_start_time", "09:35"))
    cutoff_t = parse_time(ds_cfg.get("cutoff_time", "14:30"))
    prev_day_cutoff_t = parse_time(ds_cfg.get("prev_day_start_time", "15:00"))
    sample_count = int(ds_cfg.get("sample_count", 10000))
    max_lookback_days = int(ds_cfg.get("max_lookback_days", 5))
    trade_phase_prefix = ds_cfg.get("trade_phase_prefix", "T")
    code_batch_size = int(ds_cfg.get("code_batch_size", 200))
    depth_levels = int(ds_cfg.get("depth_levels", 10))
    normalize_x = bool(ds_cfg.get("normalize_x", False))
    normalize_y = bool(ds_cfg.get("normalize_y", False))
    x_norm_method = str(ds_cfg.get("x_norm_method", "zscore_sample"))
    y_norm_method = str(ds_cfg.get("y_norm_method", "zscore_day"))
    output_dir = clean_root / str(ds_cfg.get("output_dir", "LOBDS"))
    refresh = bool(ds_cfg.get("refresh", False))
    overwrite = bool(ds_cfg.get("overwrite", False)) or refresh

    buy_twap_col = bt_cfg["buy_twap_col"]
    sell_twap_col = bt_cfg["sell_twap_col"]
    min_amount = float(bt_cfg.get("min_amount", 0.0))
    min_volume = float(bt_cfg.get("min_volume", 0.0))

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
            code_batch_size=code_batch_size,
            depth_levels=depth_levels,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            x_norm_method=x_norm_method,
            y_norm_method=y_norm_method,
            buy_twap_col=buy_twap_col,
            sell_twap_col=sell_twap_col,
            min_amount=min_amount,
            min_volume=min_volume,
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


