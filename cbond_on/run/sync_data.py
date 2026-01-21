from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
import pandas as pd
from cbond_on.data.extract import DATE_COLUMNS, fetch_table
from cbond_on.data.ftp_source import FtpParquetSource, load_ftp_config
from cbond_on.data.ftp_sync import sync_snapshot_from_ftp
from cbond_on.data.io import get_latest_table_date, table_has_data, write_table_by_date


def _sync_db(raw_root: str, cfg: dict) -> None:
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


def _sync_ftp(raw_root: str, cfg: dict) -> None:
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


def main() -> None:
    paths_cfg = load_config_file("paths")
    sync_cfg = load_config_file("sync_data")

    raw_root = paths_cfg["raw_data_root"]
    mode = str(sync_cfg.get("mode", "both")).lower()

    if mode in ("db", "both"):
        _sync_db(raw_root, sync_cfg.get("db", {}))
    if mode in ("ftp", "both"):
        _sync_ftp(raw_root, sync_cfg.get("ftp", {}))


if __name__ == "__main__":
    main()
