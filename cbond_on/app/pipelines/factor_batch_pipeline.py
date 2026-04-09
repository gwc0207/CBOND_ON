from __future__ import annotations

from pathlib import Path
from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.run_factor_batch import execute as run_factor_batch


def execute(
    factor_cfg: dict[str, Any],
    *,
    paths_cfg: dict[str, Any],
) -> Path:
    return run_factor_batch(
        cfg=factor_cfg,
        paths_cfg=paths_cfg,
        start=parse_date(factor_cfg.get("start")),
        end=parse_date(factor_cfg.get("end")),
        refresh=bool(factor_cfg.get("refresh", False)),
        overwrite=bool(factor_cfg.get("overwrite", False)),
    )
