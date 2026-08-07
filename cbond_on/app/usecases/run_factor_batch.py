from __future__ import annotations

from datetime import date
from pathlib import Path

from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs, run_factor_batch
from cbond_on.common.factor_execution_policy import validate_factor_execution_policy


def execute(
    *,
    cfg: dict,
    paths_cfg: dict,
    start: date,
    end: date,
    refresh: bool,
    overwrite: bool,
) -> Path:
    # This is the application-level entry point as well as the workflow target.
    # Keep the research Rust-first gate here so a direct programmatic caller
    # cannot bypass the workflow and begin factor I/O on a Python/hybrid path.
    validate_factor_execution_policy(cfg, scope="factor_batch")
    panel_name = str(cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    return run_factor_batch(
        cfg,
        panel_data_root=paths_cfg["panel_data_root"],
        factor_data_root=paths_cfg["factor_data_root"],
        label_data_root=paths_cfg["label_data_root"],
        raw_data_root=paths_cfg["raw_data_root"],
        results_root=paths_cfg["results_root"],
        start=start,
        end=end,
        window_minutes=15,
        panel_name=panel_name,
        refresh=refresh,
        overwrite=overwrite,
        specs=build_signal_specs(cfg),
    )


