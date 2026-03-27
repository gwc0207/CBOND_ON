from __future__ import annotations

from datetime import date

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.factor_batch.runner import build_signal_specs
from cbond_on.factors import defs  # noqa: F401
from cbond_on.factors.pipeline import run_factor_pipeline


def run(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    paths_cfg = load_config_file("paths")
    factor_cfg = dict(cfg or load_config_file("factor"))

    start_day = parse_date(start or factor_cfg.get("start"))
    end_day = parse_date(end or factor_cfg.get("end"))
    refresh_val = bool(factor_cfg.get("refresh", False) if refresh is None else refresh)
    overwrite_val = bool(factor_cfg.get("overwrite", False) if overwrite is None else overwrite)
    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    workers = int(factor_cfg.get("workers", 1))
    factor_workers = int(factor_cfg.get("factor_workers", 1))

    result = run_factor_pipeline(
        paths_cfg["panel_data_root"],
        paths_cfg["factor_data_root"],
        start_day,
        end_day,
        panel_name=panel_name,
        refresh=refresh_val,
        overwrite=overwrite_val,
        workers=workers,
        factor_workers=factor_workers,
        raw_data_root=paths_cfg.get("raw_data_root"),
        context_cfg=factor_cfg.get("context"),
        specs=build_signal_specs(factor_cfg),
    )
    return {
        "start": start_day,
        "end": end_day,
        "workers": workers,
        "factor_workers": factor_workers,
        "written": int(result.written),
        "skipped": int(result.skipped),
    }
