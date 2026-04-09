from __future__ import annotations

from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.build_labels import execute as build_labels


def execute(
    label_cfg: dict[str, Any],
    *,
    panel_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_labels(
        start=parse_date(label_cfg.get("start")),
        end=parse_date(label_cfg.get("end")),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
        panel_cfg=panel_cfg,
    )
