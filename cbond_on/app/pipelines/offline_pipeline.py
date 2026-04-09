from __future__ import annotations

from typing import Any

from cbond_on.app.pipelines.panel_pipeline import execute as run_panel_pipeline
from cbond_on.app.pipelines.label_pipeline import execute as run_label_pipeline
from cbond_on.app.pipelines.factor_pipeline import execute as run_factor_pipeline


def execute(panel_cfg: dict[str, Any], label_cfg: dict[str, Any], factor_cfg: dict[str, Any]) -> dict[str, Any]:
    panel_result = run_panel_pipeline(panel_cfg)
    label_result = run_label_pipeline(label_cfg, panel_cfg=panel_cfg)
    factor_result = run_factor_pipeline(factor_cfg)
    return {
        "panel": panel_result,
        "label": label_result,
        "factor": factor_result,
    }

