from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import List

import pandas as pd
from cbond_on.infra.data.io import read_trading_calendar


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

    # Fallback to file-date enumeration when calendar is missing/incomplete.
    return _list_trading_days_from_files(
        root,
        start=start,
        end=end,
        kind=kind,
        asset=asset,
    )


def list_available_trading_days_from_raw(
    raw_data_root: str | Path,
    *,
    kind: str = "snapshot",
    asset: str = "cbond",
) -> List[date]:
    root = Path(raw_data_root)
    cal = _calendar_open_days(root)
    if cal is not None:
        return cal
    return _list_trading_days_from_files(root, start=None, end=None, kind=kind, asset=asset)


def next_trading_days_from_raw(
    raw_data_root: str | Path,
    day: date,
    count: int,
    *,
    kind: str = "snapshot",
    asset: str = "cbond",
) -> list[date]:
    n = max(0, int(count))
    if n == 0:
        return []
    days = list_available_trading_days_from_raw(raw_data_root, kind=kind, asset=asset)
    if not days:
        return []
    idx = bisect_right(days, day)
    return days[idx : idx + n]


def prev_trading_days_from_raw(
    raw_data_root: str | Path,
    day: date,
    count: int,
    *,
    kind: str = "snapshot",
    asset: str = "cbond",
) -> list[date]:
    n = max(0, int(count))
    if n == 0:
        return []
    days = list_available_trading_days_from_raw(raw_data_root, kind=kind, asset=asset)
    if not days:
        return []
    idx = bisect_left(days, day)
    start_idx = max(0, idx - n)
    return days[start_idx:idx]


def _list_trading_days_from_calendar(
    raw_data_root: Path,
    start: date,
    end: date,
) -> list[date] | None:
    days = _calendar_open_days(raw_data_root)
    if days is None:
        return None
    return [d for d in days if start <= d <= end]


def _calendar_open_days(raw_data_root: Path) -> list[date] | None:
    cal = read_trading_calendar(raw_data_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return None

    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    work = work[work["calendar_date"].notna()]
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    return sorted(set(work["calendar_date"].tolist()))


def _list_trading_days_from_files(
    raw_data_root: Path,
    *,
    start: date | None,
    end: date | None,
    kind: str,
    asset: str,
) -> list[date]:
    base = _raw_base(raw_data_root, kind=kind, asset=asset)
    if not base.exists():
        return []
    days: set[date] = set()
    for path in base.rglob("*.parquet"):
        parsed = _parse_day_from_stem(path.stem, kind=kind)
        if parsed is None:
            continue
        if start is not None and parsed < start:
            continue
        if end is not None and parsed > end:
            continue
        days.add(parsed)
    return sorted(days)


def _raw_base(raw_data_root: Path, *, kind: str, asset: str = "cbond") -> Path:
    asset_name = str(asset or "cbond").strip().lower() or "cbond"
    if kind == "snapshot":
        return raw_data_root / "snapshot" / asset_name / "raw_data"
    if kind == "kline":
        return raw_data_root / "kline" / asset_name / "from_snapshot"
    raise ValueError(f"unknown raw kind: {kind}")


def _parse_day_from_stem(stem: str, *, kind: str) -> date | None:
    text = str(stem).strip()
    try:
        if kind == "snapshot":
            if len(text) == 8 and text.isdigit():
                return datetime.strptime(text, "%Y%m%d").date()
            return None
        if kind == "kline":
            return datetime.strptime(text, "%Y-%m-%d").date()
    except Exception:
        return None
    raise ValueError(f"unknown raw kind: {kind}")


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

