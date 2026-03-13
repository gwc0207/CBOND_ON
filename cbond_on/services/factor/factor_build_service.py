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
    if refresh_val:
        overwrite_val = True

    result = run_factor_pipeline(
        paths_cfg["panel_data_root"],
        paths_cfg["factor_data_root"],
        start_day,
        end_day,
        window_minutes=int(factor_cfg.get("window_minutes", 15)),
        panel_name=factor_cfg.get("panel_name"),
        overwrite=overwrite_val,
        specs=build_signal_specs(factor_cfg),
    )
    return {
        "start": start_day,
        "end": end_day,
        "written": int(result.written),
        "skipped": int(result.skipped),
    }
