from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.config import SnapshotConfig
from cbond_on.data.clean import build_cleaned_snapshot, build_cleaned_kline


def main() -> None:  
    paths_cfg = load_config_file("paths")
    cleaned_cfg = load_config_file("cleaned_data")
    raw_data_root = paths_cfg["raw_data_root"]
    cleaned_data_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")
    start = parse_date(cleaned_cfg.get("start"))
    end = parse_date(cleaned_cfg.get("end"))
    full_refresh = bool(cleaned_cfg.get("full_refresh", False))
    overwrite = bool(cleaned_cfg.get("overwrite", False)) or full_refresh

    print(f"[clean] start={start} end={end} overwrite={overwrite}")
    snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
    print("[clean] building snapshot ...")
    res1 = build_cleaned_snapshot(
        raw_data_root,
        cleaned_data_root,
        start,
        end,
        snapshot_cfg,
        overwrite=overwrite,
    )
    print("[clean] building kline ...")
    res2 = build_cleaned_kline(
        raw_data_root,
        cleaned_data_root,
        start,
        end,
        overwrite=overwrite,
    )
    print(f"[clean] done snapshot={res1.snapshot_written} skipped={res1.snapshot_skipped}")
    print(f"[clean] done kline={res2.kline_written} skipped={res2.kline_skipped}")


if __name__ == "__main__":
    main()
