from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class SnapshotConfig:
    price_field: str = "last"
    drop_no_trade: bool = True
    use_prev_snapshot: bool = True


def iter_snapshot_files(
    root: str | Path,
    start: date,
    end: date,
) -> Iterable[Path]:
    base = Path(root)
    for path in sorted(base.rglob("*.parquet")):
        day = _date_from_path(path)
        if day is None:
            continue
        if start <= day <= end:
            yield path


def read_snapshot_day(
    path: str | Path,
    cfg: SnapshotConfig,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "trade_time" in df.columns and not pd.api.types.is_datetime64_any_dtype(
        df["trade_time"]
    ):
        df = df.copy()
        df["trade_time"] = pd.to_datetime(df["trade_time"])

    if "code" in df.columns and "trade_time" in df.columns:
        df = df.sort_values(["code", "trade_time"])
        df = df.drop_duplicates(subset=["code", "trade_time"], keep="last")
    return df


def load_snapshot_up_to(
    root: str | Path,
    day: date,
    cutoff: time,
    cfg: SnapshotConfig,
) -> pd.DataFrame:
    path = _snapshot_path(root, day)
    if not path.exists():
        return pd.DataFrame()
    df = read_snapshot_day(path, cfg)
    if df.empty:
        return df
    if "trade_time" not in df.columns:
        raise KeyError("missing trade_time in snapshot")
    cutoff_dt = datetime.combine(day, cutoff)
    return df[df["trade_time"] <= cutoff_dt].copy()


def _snapshot_path(root: str | Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return Path(root) / month / filename


def _date_from_path(path: Path) -> Optional[date]:
    name = path.stem
    if len(name) != 8 or not name.isdigit():
        return None
    return datetime.strptime(name, "%Y%m%d").date()
