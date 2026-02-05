from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

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

    for day in progress(
        list_trading_days_from_raw(raw_data_root, start, end, kind="snapshot"),
        desc="clean_snapshot",
        unit="day",
    ):
        src = _raw_snapshot_path(raw_data_root, day)
        if not src.exists():
            continue
        dst = _cleaned_snapshot_path(cleaned_data_root, day)
        if dst.exists() and not overwrite:
            result.snapshot_skipped += 1
            continue

        df = pd.read_parquet(src)
        df = _clean_snapshot(df, snapshot_config)
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
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        result.kline_written += 1

    return result


def _clean_snapshot(df: pd.DataFrame, cfg: SnapshotConfig) -> pd.DataFrame:
    if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["trade_time"]
    ):
        df = df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])

    if cfg.filter_trading_phase and "trading_phase_code" in df.columns:
        allowed = set(cfg.allowed_phases or [])
        if allowed:
            df = df[df["trading_phase_code"].isin(allowed)]

    df = _fix_sh_volume_unit(df)

    if "code" in df.columns and "trade_time" in df.columns:
        df = df.sort_values(["code", "trade_time"])
        df = df.drop_duplicates(subset=["code", "trade_time"], keep="last")
    return df


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
        if not pd.api.types.is_datetime64_any_dtype(work["trade_time"]):
            if "trade_date" in work.columns:
                work["trade_time"] = pd.to_datetime(
                    work["trade_date"].astype(str) + " " + work["trade_time"].astype(str)
                )
    elif "timestamp" in work.columns:
        work["trade_time"] = pd.to_datetime(work["timestamp"], unit="s")

    if "code" in work.columns and "trade_time" in work.columns:
        work = work.sort_values(["code", "trade_time"])
        work = work.drop_duplicates(subset=["code", "trade_time"], keep="last")
    return work


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
