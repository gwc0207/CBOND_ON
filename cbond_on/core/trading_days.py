from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def list_trading_days_from_raw(
    raw_data_root: str | Path,
    start: date,
    end: date,
    *,
    kind: str = "snapshot",
) -> List[date]:
    raw_data_root = Path(raw_data_root)
    days: list[date] = []
    for day in _iter_dates(start, end):
        if _raw_path(raw_data_root, day, kind=kind).exists():
            days.append(day)
    return days


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + pd.Timedelta(days=1)


def _raw_path(raw_data_root: Path, day: date, *, kind: str) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    if kind == "snapshot":
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        return raw_data_root / "snapshot" / "cbond" / "raw_data" / month / filename
    if kind == "kline":
        filename = f"{day.strftime('%Y-%m-%d')}.parquet"
        return raw_data_root / "kline" / "cbond" / "from_snapshot" / month / filename
    raise ValueError(f"unknown raw kind: {kind}")
