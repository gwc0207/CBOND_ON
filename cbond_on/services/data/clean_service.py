from __future__ import annotations

from datetime import date

from cbond_on.config import SnapshotConfig
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.data.clean import build_cleaned_kline, build_cleaned_snapshot


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    cleaned_cfg = dict(cfg or load_config_file("cleaned_data"))

    start_day = parse_date(start or cleaned_cfg.get("start"))
    end_day = parse_date(end or cleaned_cfg.get("end"))
    refresh_val = bool(cleaned_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(cleaned_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if refresh_val:
        overwrite_val = True

    snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
    clean_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")
    kline_enabled = bool(cleaned_cfg.get("kline_enabled", True))

    snapshot_result = build_cleaned_snapshot(
        paths_cfg["raw_data_root"],
        clean_root,
        start_day,
        end_day,
        snapshot_cfg,
        overwrite=overwrite_val,
    )
    if kline_enabled:
        kline_result = build_cleaned_kline(
            paths_cfg["raw_data_root"],
            clean_root,
            start_day,
            end_day,
            overwrite=overwrite_val,
        )
    else:
        kline_result = snapshot_result.__class__()
    return {
        "start": start_day,
        "end": end_day,
        "snapshot_written": snapshot_result.snapshot_written,
        "snapshot_skipped": snapshot_result.snapshot_skipped,
        "kline_written": kline_result.kline_written,
        "kline_skipped": kline_result.kline_skipped,
    }

