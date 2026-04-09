from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from cbond_on.config import ScheduleConfig, SnapshotConfig
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.core.utils import progress
from cbond_on.infra.data.panel import build_labels_for_day


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
    panel_cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    label_cfg = dict(cfg or load_config_file("label"))
    panel_runtime_cfg: dict | None = None
    if isinstance(panel_cfg, dict):
        panel_runtime_cfg = dict(panel_cfg)
    else:
        panel_inline = label_cfg.get("panel")
        if isinstance(panel_inline, dict):
            panel_runtime_cfg = dict(panel_inline)
    if panel_runtime_cfg is None:
        panel_runtime_cfg = load_config_file("panel")

    start_day = parse_date(start or label_cfg.get("start"))
    end_day = parse_date(end or label_cfg.get("end"))
    refresh_val = bool(label_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(label_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if refresh_val:
        overwrite_val = True
    skip_existing = bool(label_cfg.get("skip_existing_when_no_overwrite", True))
    mode = "overwrite" if overwrite_val else str(label_cfg.get("mode", "upsert"))

    schedule_raw = panel_runtime_cfg.get("schedule")
    if not isinstance(schedule_raw, dict):
        raise KeyError("label runtime requires panel.schedule config")
    schedule = ScheduleConfig.from_dict(schedule_raw).to_schedule()
    snapshot_cfg = SnapshotConfig.from_dict(dict(panel_runtime_cfg.get("snapshot", {})))
    trading_days = list_trading_days_from_raw(
        paths_cfg["raw_data_root"],
        start_day,
        end_day,
        kind="snapshot",
    )
    workers = int(label_cfg.get("workers", panel_runtime_cfg.get("workers", 1)))
    workers = max(1, workers)

    written = 0
    skipped = 0
    clean_root = paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root")
    tasks = [
        (day, trading_days[idx + 1] if idx + 1 < len(trading_days) else None)
        for idx, day in enumerate(trading_days)
    ]

    def _run_one(day: date, next_day: date | None) -> str:
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        out_path = Path(paths_cfg["label_data_root"]) / month / filename
        if (not overwrite_val) and skip_existing and out_path.exists():
            return "skipped"
        ok = build_labels_for_day(
            clean_root,
            paths_cfg["label_data_root"],
            day,
            schedule,
            snapshot_cfg,
            label_cfg,
            mode=mode,
            next_day=next_day,
        )
        return "written" if ok else "skipped"

    if workers == 1:
        for day, next_day in progress(
            tasks,
            desc="build_labels",
            unit="day",
            total=len(tasks),
        ):
            result = _run_one(day, next_day)
            if result == "written":
                written += 1
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_one, day, next_day) for day, next_day in tasks]
            for future in progress(
                as_completed(futures),
                desc="build_labels",
                unit="day",
                total=len(futures),
            ):
                result = future.result()
                if result == "written":
                    written += 1
                else:
                    skipped += 1
    return {
        "start": start_day,
        "end": end_day,
        "written": written,
        "skipped": skipped,
        "workers": workers,
    }



