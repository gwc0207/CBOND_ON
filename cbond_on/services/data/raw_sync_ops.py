from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from cbond_on.core.config import parse_date
from cbond_on.data.extract import DATE_COLUMNS, fetch_table
from cbond_on.data.ftp_source import FtpParquetSource, load_ftp_config
from cbond_on.data.ftp_sync import sync_snapshot_from_ftp
from cbond_on.data.io import get_latest_table_date, table_has_data, write_table_by_date


def sync_db(raw_root: str, cfg: dict) -> None:
    start = parse_date(cfg.get("start", "2000-01-01"))
    end = parse_date(cfg.get("end"))
    refresh = bool(cfg.get("refresh", False))
    overwrite = bool(cfg.get("overwrite", False))
    tables = cfg.get("sync_tables", [])
    if refresh:
        overwrite = True

    for table in tables:
        last_date = None if refresh else get_latest_table_date(raw_root, table)
        date_based = table in DATE_COLUMNS
        if date_based:
            fetch_start = max(start, last_date + pd.Timedelta(days=1)) if last_date else start
            if fetch_start > end:
                continue
            df = fetch_table(table, start=str(fetch_start), end=str(end))
        else:
            if not refresh and table_has_data(raw_root, table):
                continue
            df = fetch_table(table)
        if df.empty:
            continue
        write_table_by_date(
            df,
            raw_root,
            table,
            date_col=DATE_COLUMNS.get(table, "trade_date"),
            overwrite=overwrite,
        )
        print(f"synced db {table}: {len(df)}")


def sync_ftp(raw_root: str, cfg: dict) -> None:
    start = parse_date(cfg.get("start"))
    end = parse_date(cfg.get("end"))
    refresh = bool(cfg.get("refresh", False))
    overwrite = bool(cfg.get("overwrite", False)) or refresh
    base_dir = str(cfg.get("base_dir", "snapshot/cbond/raw_data"))
    cfg_path = cfg.get("ftp_config_path")

    ftp = FtpParquetSource(load_ftp_config(cfg_path))
    result = sync_snapshot_from_ftp(
        ftp,
        raw_root,
        start,
        end,
        base_dir=base_dir,
        overwrite=overwrite,
    )
    print(result)


def sync_nfs(raw_root: str, cfg: dict) -> None:
    start = parse_date(cfg.get("start"))
    end = parse_date(cfg.get("end"))
    refresh = bool(cfg.get("refresh", False))
    overwrite = bool(cfg.get("overwrite", False)) or refresh
    nfs_root = str(cfg.get("nfs_root", "")).strip()
    if not nfs_root:
        raise ValueError("nfs.nfs_root is required")
    base_dir = str(cfg.get("base_dir", "snapshot/cbond/raw_data")).strip("/\\")

    src_base = Path(nfs_root) / Path(base_dir)
    dst_base = Path(raw_root) / "snapshot" / "cbond" / "raw_data"
    downloaded = 0
    skipped = 0

    cursor = date(start.year, start.month, 1)
    end_month = date(end.year, end.month, 1)
    while cursor <= end_month:
        month = f"{cursor.year:04d}-{cursor.month:02d}"
        month_src = src_base / month
        if month_src.exists():
            for src in sorted(month_src.glob("*.parquet")):
                stem = src.stem
                if len(stem) != 8 or not stem.isdigit():
                    continue
                day = datetime.strptime(stem, "%Y%m%d").date()
                if day < start or day > end:
                    continue
                dst = dst_base / month / src.name
                if dst.exists() and not overwrite:
                    skipped += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                downloaded += 1
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    print(f"NfsSyncResult(snapshot_downloaded={downloaded}, snapshot_skipped={skipped})")

