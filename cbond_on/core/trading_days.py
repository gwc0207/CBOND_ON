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
    asset: str = "cbond",
) -> List[date]:
    root = Path(raw_data_root)
    cal_days = _list_trading_days_from_calendar(root, start, end)
    if cal_days is not None:
        return cal_days

    # Fallback to file presence when calendar is missing/incomplete.
    days: list[date] = []
    for day in _iter_dates(start, end):
        if _raw_path(root, day, kind=kind, asset=asset).exists():
            days.append(day)
    return days


def _list_trading_days_from_calendar(
    raw_data_root: Path,
    start: date,
    end: date,
) -> list[date] | None:
    cal = read_trading_calendar(raw_data_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return None

    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    work = work[work["calendar_date"].notna()]
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]

    days = sorted(set(work["calendar_date"].tolist()))
    return [d for d in days if start <= d <= end]


def _iter_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current = current + pd.Timedelta(days=1)


def _raw_path(raw_data_root: Path, day: date, *, kind: str, asset: str = "cbond") -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    asset_name = str(asset or "cbond").strip().lower()
    if not asset_name:
        asset_name = "cbond"
    if kind == "snapshot":
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        return raw_data_root / "snapshot" / asset_name / "raw_data" / month / filename
    if kind == "kline":
        filename = f"{day.strftime('%Y-%m-%d')}.parquet"
        return raw_data_root / "kline" / asset_name / "from_snapshot" / month / filename
    raise ValueError(f"unknown raw kind: {kind}")
