from __future__ import annotations

from datetime import date

from cbond_on.config import ScheduleConfig, SnapshotConfig
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.data.panel import build_labels_for_day


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    label_cfg = dict(cfg or load_config_file("label"))
    panel_cfg = load_config_file("panel")
    cleaned_cfg = load_config_file("cleaned_data")

    start_day = parse_date(start or label_cfg.get("start"))
    end_day = parse_date(end or label_cfg.get("end"))
    refresh_val = bool(label_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(label_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if refresh_val:
        overwrite_val = True
    mode = "overwrite" if overwrite_val else str(label_cfg.get("mode", "upsert"))

    schedule = ScheduleConfig.from_dict(panel_cfg["schedule"]).to_schedule()
    snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
    trading_days = list_trading_days_from_raw(
        paths_cfg["raw_data_root"],
        start_day,
        end_day,
        kind="snapshot",
    )

    written = 0
    skipped = 0
    clean_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")
    for day in trading_days:
        ok = build_labels_for_day(
            clean_root,
            paths_cfg["label_data_root"],
            day,
            schedule,
            snapshot_cfg,
            label_cfg,
            mode=mode,
            max_lookahead_days=int(label_cfg.get("max_lookahead_days", 7)),
        )
        if ok:
            written += 1
        else:
            skipped += 1
    return {
        "start": start_day,
        "end": end_day,
        "written": written,
        "skipped": skipped,
    }

