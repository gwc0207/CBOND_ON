from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from .ftp_source import FtpParquetSource, load_ftp_config


@dataclass
class FtpSyncResult:
    snapshot_downloaded: int = 0
    snapshot_skipped: int = 0


def sync_snapshot_from_ftp(
    ftp: FtpParquetSource,
    raw_root: str | Path,
    start: date,
    end: date,
    *,
    base_dir: str = "snapshot/cbond/raw_data",
    overwrite: bool = False,
) -> FtpSyncResult:
    result = FtpSyncResult()
    raw_root = Path(raw_root)

    local_base = raw_root / "snapshot" / "cbond" / "raw_data"
    for path, day in _iter_ftp_files(ftp, base_dir, start, end, _parse_snapshot_date):
        target = _local_path_for_day(local_base, day, path)
        if _write_if_needed(ftp, path, target, overwrite=overwrite):
            result.snapshot_downloaded += 1
        else:
            result.snapshot_skipped += 1
    return result


def sync_snapshot_for_backtest(
    start: date,
    end: date,
    raw_root: str | Path,
    ftp_config_path: str | Path | None = None,
    *,
    base_dir: str = "snapshot/cbond/raw_data",
    overwrite: bool = False,
) -> FtpSyncResult:
    ftp_cfg = load_ftp_config(ftp_config_path)
    ftp = FtpParquetSource(ftp_cfg)
    return sync_snapshot_from_ftp(
        ftp,
        raw_root,
        start,
        end,
        base_dir=base_dir,
        overwrite=overwrite,
    )


def _iter_ftp_files(
    ftp: FtpParquetSource,
    base_dir: str,
    start: date,
    end: date,
    parse_date,
):
    for month in _month_range(start, end):
        dir_path = f"{base_dir}/{month}"
        try:
            files = ftp.list_dir(dir_path)
        except Exception:
            continue
        for name in files:
            day = parse_date(name)
            if day is None:
                continue
            if start <= day <= end:
                yield f"{dir_path}/{name}", day


def _month_range(start: date, end: date) -> Iterable[str]:
    year, month = start.year, start.month
    end_year, end_month = end.year, end.month
    while (year, month) <= (end_year, end_month):
        yield f"{year:04d}-{month:02d}"
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def _parse_snapshot_date(name: str) -> Optional[date]:
    stem = name.rsplit(".", 1)[0]
    if len(stem) != 8 or not stem.isdigit():
        return None
    return datetime.strptime(stem, "%Y%m%d").date()


def _local_path_for_day(local_base: Path, day: date, ftp_path: str) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = ftp_path.rsplit("/", 1)[-1]
    return local_base / month / filename


def _write_if_needed(
    ftp: FtpParquetSource,
    ftp_path: str,
    target: Path,
    *,
    overwrite: bool,
) -> bool:
    if target.exists() and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    data = ftp.read_bytes(ftp_path)
    with open(target, "wb") as f:
        f.write(data)
    return True
