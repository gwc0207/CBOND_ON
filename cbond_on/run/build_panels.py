from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.config import SnapshotConfig, ScheduleConfig
from cbond_on.data.panel import build_panel_data, build_panels_with_labels


def main() -> None:
    paths_cfg = load_config_file("paths")
    panel_cfg = load_config_file("panel")
    label_cfg = panel_cfg.get("label", {})
    cleaned_cfg = load_config_file("cleaned_data")
    raw_data_root = paths_cfg["raw_data_root"]
    cleaned_data_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")
    panel_data_root = paths_cfg["panel_data_root"]
    label_data_root = paths_cfg["label_data_root"]
    start = parse_date(panel_cfg.get("start"))
    end = parse_date(panel_cfg.get("end"))

    snapshot_cfg = SnapshotConfig.from_dict(cleaned_cfg["snapshot"])
    schedule = ScheduleConfig.from_dict(panel_cfg["schedule"]).to_schedule()

    full_refresh = bool(panel_cfg.get("full_refresh", False))
    overwrite = bool(panel_cfg.get("overwrite", False)) or full_refresh
    windows = panel_cfg.get("window_minutes", [15])
    panel_name = panel_cfg.get("panel_name")
    panel_mode = panel_cfg.get("panel_mode", "snapshot_sequence")
    count_points = int(panel_cfg.get("count_points", 3000))
    max_lookback_days = int(panel_cfg.get("max_lookback_days", 3))
    snapshot_columns = panel_cfg.get("snapshot_columns")
    lead_minutes = int(panel_cfg.get("lead_minutes", 0))
    if str(panel_mode).lower() != "snapshot_sequence":
        raise ValueError("panel_mode must be 'snapshot_sequence'")
    print(
        f"[panel] start={start} end={end} windows={windows} "
        f"panel_name={panel_name} overwrite={overwrite} full_refresh={full_refresh}"
    )
    for w in windows:
        print(f"[panel] building panels for window={w} ...")
        res = build_panels_with_labels(
            cleaned_data_root,
            panel_data_root,
            label_data_root,
            raw_data_root,
            start,
            end,
            schedule,
            snapshot_cfg,
            label_cfg,
            window_minutes=int(w),
            panel_name=panel_name,
            overwrite=overwrite,
            panel_mode=panel_mode,
            count_points=count_points,
            max_lookback_days=max_lookback_days,
            snapshot_columns=snapshot_columns,
            lead_minutes=lead_minutes,
        )
        print(f"[panel] done window={w} -> {res}")


if __name__ == "__main__":
    main()
