from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from cbond_on.data.io import read_trading_calendar


def list_trading_days_from_raw(
    raw_data_root: str | Path,
    start: date,
    end: date,
    *,
    kind: str = "snapshot",
) -> List[date]:
    _ = kind  # keep signature stable for callers
    cal = read_trading_calendar(raw_data_root)
    if cal.empty:
        raise ValueError("trading_calendar is empty; sync raw metadata first")
    if "calendar_date" not in cal.columns:
        raise KeyError("trading_calendar missing calendar_date column")

    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    work = work[work["calendar_date"].notna()]
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]

    days = sorted(d for d in work["calendar_date"].tolist() if start <= d <= end)
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
