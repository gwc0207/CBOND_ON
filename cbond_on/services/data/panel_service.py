from __future__ import annotations

from datetime import date

from cbond_on.config import ScheduleConfig, SnapshotConfig
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.data.panel import build_panel_data


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    panel_cfg = dict(cfg or load_config_file("panel"))

    start_day = parse_date(start or panel_cfg.get("start"))
    end_day = parse_date(end or panel_cfg.get("end"))
    refresh_val = bool(panel_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(panel_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if refresh_val:
        overwrite_val = True

    schedule = ScheduleConfig.from_dict(panel_cfg["schedule"]).to_schedule()
    snapshot_cfg = SnapshotConfig.from_dict(dict(panel_cfg.get("snapshot", {})))
    windows = panel_cfg.get("window_minutes", [15])
    panel_name = panel_cfg.get("panel_name")
    assets_raw = panel_cfg.get("assets", ["cbond"])
    if isinstance(assets_raw, str):
        assets = [a.strip().lower() for a in assets_raw.replace(";", ",").split(",") if a.strip()]
    elif isinstance(assets_raw, (list, tuple)):
        assets = [str(a).strip().lower() for a in assets_raw if str(a).strip()]
    else:
        assets = ["cbond"]
    if not assets:
        assets = ["cbond"]
    panel_mode = str(panel_cfg.get("panel_mode", "snapshot_sequence"))
    count_points = int(panel_cfg.get("count_points", 3000))
    max_lookback_days = int(panel_cfg.get("max_lookback_days", 3))
    workers = int(panel_cfg.get("workers", 1))
    snapshot_columns = panel_cfg.get("snapshot_columns")
    lead_minutes = int(panel_cfg.get("lead_minutes", 0))
    clean_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")

    wrote = 0
    skipped = 0
    diagnostics_rows = 0
    missing_snapshot_days = 0
    by_asset: dict[str, dict[str, int]] = {}
    for asset in assets:
        asset_written = 0
        asset_skipped = 0
        asset_diag_rows = 0
        asset_missing_days = 0
        for window in windows:
            result = build_panel_data(
                clean_root,
                paths_cfg["panel_data_root"],
                paths_cfg["raw_data_root"],
                start_day,
                end_day,
                schedule,
                snapshot_cfg,
                window_minutes=int(window),
                panel_name=panel_name,
                asset=asset,
                overwrite=overwrite_val,
                panel_mode=panel_mode,
                count_points=count_points,
                max_lookback_days=max_lookback_days,
                workers=workers,
                snapshot_columns=snapshot_columns,
                lead_minutes=lead_minutes,
                compute_cfg=panel_cfg.get("compute"),
            )
            wrote += int(result.written)
            skipped += int(result.skipped)
            diagnostics_rows += int(result.diagnostics_rows)
            missing_snapshot_days += int(result.missing_snapshot_days)
            asset_written += int(result.written)
            asset_skipped += int(result.skipped)
            asset_diag_rows += int(result.diagnostics_rows)
            asset_missing_days += int(result.missing_snapshot_days)
        by_asset[asset] = {
            "written": asset_written,
            "skipped": asset_skipped,
            "diagnostics_rows": asset_diag_rows,
            "missing_snapshot_days": asset_missing_days,
        }

    return {
        "start": start_day,
        "end": end_day,
        "written": wrote,
        "skipped": skipped,
        "diagnostics_rows": diagnostics_rows,
        "missing_snapshot_days": missing_snapshot_days,
        "assets": assets,
        "workers": workers,
        "by_asset": by_asset,
    }

