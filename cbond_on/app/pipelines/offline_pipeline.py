from __future__ import annotations

from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.build_panel import execute as build_panel
from cbond_on.app.usecases.build_labels import execute as build_labels
from cbond_on.app.usecases.build_factors import execute as build_factors


def execute(panel_cfg: dict[str, Any], label_cfg: dict[str, Any], factor_cfg: dict[str, Any]) -> dict[str, Any]:
    panel_result = build_panel(
        start=parse_date(panel_cfg.get("start")),
        end=parse_date(panel_cfg.get("end")),
        refresh=bool(panel_cfg.get("refresh", False)),
        overwrite=bool(panel_cfg.get("overwrite", False)),
        cfg=panel_cfg,
    )
    label_result = build_labels(
        start=parse_date(label_cfg.get("start")),
        end=parse_date(label_cfg.get("end")),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
        panel_cfg=panel_cfg,
    )
    factor_result = build_factors(
        start=parse_date(factor_cfg.get("start")),
        end=parse_date(factor_cfg.get("end")),
        refresh=bool(factor_cfg.get("refresh", False)),
        overwrite=bool(factor_cfg.get("overwrite", False)),
        cfg=factor_cfg,
    )
    return {
        "panel": panel_result,
        "label": label_result,
        "factor": factor_result,
    }

