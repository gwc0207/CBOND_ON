from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from cbond_on.config import SnapshotConfig
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.utils import progress


@dataclass
class CleanedBuildResult:
    snapshot_written: int = 0
    snapshot_skipped: int = 0
    kline_written: int = 0
    kline_skipped: int = 0


def build_cleaned_snapshot(
    raw_data_root: str | Path,
    cleaned_data_root: str | Path,
    start: date,
    end: date,
    snapshot_config: SnapshotConfig,
    *,
    overwrite: bool = False,
) -> CleanedBuildResult:
    result = CleanedBuildResult()
    raw_data_root = Path(raw_data_root)
    cleaned_data_root = Path(cleaned_data_root)
    # Keep tail lookahead so end-day can still use next trading files in lagged NFS.
    lookahead_end = end + pd.Timedelta(days=10)
    all_days = list_trading_days_from_raw(raw_data_root, start, lookahead_end, kind="snapshot")
    process_days = [d for d in all_days if start <= d <= end]

    for day in progress(
        process_days,
        desc="clean_snapshot",
        unit="day",
    ):
        try:
            idx = all_days.index(day)
        except ValueError:
            idx = -1
        dst = _cleaned_snapshot_path(cleaned_data_root, day)
        if dst.exists() and not overwrite:
            result.snapshot_skipped += 1
            continue

        # NFS source can contain mixed-day rows in a single file and may lag by
        # one to two trading files around holidays. Read day + next 2 trading days.
        if idx < 0:
            src_days = [day]
        else:
            src_days = all_days[idx : min(len(all_days), idx + 3)]
        frames: list[pd.DataFrame] = []
        for src_day in src_days:
            src = _raw_snapshot_path(raw_data_root, src_day)
            if not src.exists():
                continue
            raw_df = pd.read_parquet(src)
            if raw_df is not None and not raw_df.empty:
                frames.append(raw_df)
        if not frames:
            if overwrite and dst.exists():
                dst.unlink()
            continue

        df = pd.concat(frames, ignore_index=True)
        df = _clean_snapshot(df, snapshot_config, day=day)
        if df.empty:
            if overwrite and dst.exists():
                dst.unlink()
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        result.snapshot_written += 1

    return result


def build_cleaned_kline(
    raw_data_root: str | Path,
    cleaned_data_root: str | Path,
    start: date,
    end: date,
    *,
    overwrite: bool = False,
) -> CleanedBuildResult:
    result = CleanedBuildResult()
    raw_data_root = Path(raw_data_root)
    cleaned_data_root = Path(cleaned_data_root)

    for day in progress(
        list_trading_days_from_raw(raw_data_root, start, end, kind="kline"),
        desc="clean_kline",
        unit="day",
    ):
        src = _raw_kline_path(raw_data_root, day)
        if not src.exists():
            continue

        dst = _cleaned_kline_path(cleaned_data_root, day)
        if dst.exists() and not overwrite:
            result.kline_skipped += 1
            continue

        df = pd.read_parquet(src)
        df = _clean_kline(df)
        if df.empty:
            if overwrite and dst.exists():
                dst.unlink()
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        result.kline_written += 1

    return result


def _clean_snapshot(
    df: pd.DataFrame,
    cfg: SnapshotConfig,
    *,
    day: date | None = None,
) -> pd.DataFrame:
    work = df.copy()

    if "trade_time" in work.columns:
        work["trade_time"] = _normalize_trade_time_local(work["trade_time"])
    elif "timestamp" in work.columns:
        work["trade_time"] = _normalize_timestamp_col(work["timestamp"])
    elif "ts" in work.columns:
        work["trade_time"] = _normalize_timestamp_col(work["ts"])

    if "trade_time" in work.columns:
        work = work[work["trade_time"].notna()]
        if day is not None:
            work = work[work["trade_time"].dt.date == day]
    if work.empty:
        return work

    if cfg.filter_trading_phase and "trading_phase_code" in work.columns:
        allowed = set(cfg.allowed_phases or [])
        if allowed:
            work = work[work["trading_phase_code"].isin(allowed)]
    if work.empty:
        return work

    work = _fix_sh_volume_unit(work)

    if "code" in work.columns and "trade_time" in work.columns:
        work = work.sort_values(["code", "trade_time"])
        work = work.drop_duplicates(subset=["code", "trade_time"], keep="last")
    return work


def _fix_sh_volume_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize SH volume unit using amount/last/volume ratio."""
    required = {"code", "volume", "last", "amount"}
    if not required.issubset(df.columns):
        return df
    work = df
    tmp = work.loc[:, ["code", "volume", "last", "amount"]].copy()
    tmp = tmp[(tmp["volume"] > 0) & (tmp["last"] > 0) & (tmp["amount"] > 0)]
    if tmp.empty:
        return df
    tmp["ratio"] = tmp["amount"] / (tmp["last"] * tmp["volume"])
    med = tmp.groupby("code")["ratio"].median()
    need_fix = med[(med >= 8.0) & (med <= 12.0)].index
    if len(need_fix) == 0:
        return df

    volume_cols = ["volume"]
    for i in range(1, 6):
        volume_cols.append(f"ask_volume{i}")
        volume_cols.append(f"bid_volume{i}")
    cols = [c for c in volume_cols if c in work.columns]
    if not cols:
        return df

    mask = work["code"].isin(need_fix)
    work.loc[mask, cols] = work.loc[mask, cols] * 10
    return work


def _clean_kline(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "code" not in work.columns:
        if "instrument_code" in work.columns and "exchange_code" in work.columns:
            work["code"] = work["instrument_code"] + "." + work["exchange_code"]

    if "trade_time" in work.columns:
        work["trade_time"] = _normalize_trade_time_local(work["trade_time"])
        if (
            work["trade_time"].isna().all()
            and "trade_date" in work.columns
            and "trade_time" in work.columns
        ):
            work["trade_time"] = pd.to_datetime(
                work["trade_date"].astype(str) + " " + work["trade_time"].astype(str),
                errors="coerce",
            )
    elif "timestamp" in work.columns:
        work["trade_time"] = _normalize_timestamp_col(work["timestamp"])

    if "trade_time" in work.columns:
        work = work[work["trade_time"].notna()]

    if "code" in work.columns and "trade_time" in work.columns:
        work = work.sort_values(["code", "trade_time"])
        work = work.drop_duplicates(subset=["code", "trade_time"], keep="last")
    return work


def _normalize_timestamp_col(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    valid = num.dropna()
    if valid.empty:
        return pd.to_datetime(series, errors="coerce")

    sample = float(valid.abs().median())
    if sample >= 1e12:
        out = pd.to_datetime(num, unit="ms", errors="coerce", utc=True)
    else:
        out = pd.to_datetime(num, unit="s", errors="coerce", utc=True)
    return out.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def _normalize_trade_time_local(series: pd.Series) -> pd.Series:
    raw = pd.to_datetime(series, errors="coerce")
    if raw.empty:
        return raw

    tz = getattr(raw.dt, "tz", None)
    if tz is not None:
        return raw.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)

    out = raw.copy()
    valid = out.dropna()
    if valid.empty:
        return out

    # NFS snapshots may carry UTC-like naive times (01:xx~07:xx).
    # In mixed files, shift only those early-hour rows.
    early_mask = out.dt.hour.between(0, 7)
    has_early = bool(early_mask.fillna(False).any())
    has_daytime = bool(out.dt.hour.between(8, 16).fillna(False).any())
    if has_early and (has_daytime or float(valid.dt.hour.between(0, 7).mean()) >= 0.8):
        out.loc[early_mask] = out.loc[early_mask] + pd.Timedelta(hours=8)
    return out


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + pd.Timedelta(days=1)


def _raw_snapshot_path(raw_data_root: Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return raw_data_root / "snapshot" / "cbond" / "raw_data" / month / filename


def _raw_kline_path(raw_data_root: Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y-%m-%d')}.parquet"
    return raw_data_root / "kline" / "cbond" / "from_snapshot" / month / filename


def _cleaned_snapshot_path(cleaned_data_root: Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return cleaned_data_root / "snapshot" / month / filename


def _cleaned_kline_path(cleaned_data_root: Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y-%m-%d')}.parquet"
    return cleaned_data_root / "kline" / month / filename


def read_cleaned_snapshot(cleaned_data_root: str | Path, day: date) -> pd.DataFrame:
    path = _cleaned_snapshot_path(Path(cleaned_data_root), day)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_cleaned_kline(cleaned_data_root: str | Path, day: date) -> pd.DataFrame:
    path = _cleaned_kline_path(Path(cleaned_data_root), day)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
