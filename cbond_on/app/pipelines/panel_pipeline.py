from __future__ import annotations

from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.build_panel import execute as build_panel


def execute(panel_cfg: dict[str, Any]) -> dict[str, Any]:
    return build_panel(
        start=parse_date(panel_cfg.get("start")),
        end=parse_date(panel_cfg.get("end")),
        refresh=bool(panel_cfg.get("refresh", False)),
        overwrite=bool(panel_cfg.get("overwrite", False)),
        cfg=panel_cfg,
    )
