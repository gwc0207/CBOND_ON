from __future__ import annotations

from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.build_factors import execute as build_factors


def execute(factor_cfg: dict[str, Any]) -> dict[str, Any]:
    return build_factors(
        start=parse_date(factor_cfg.get("start")),
        end=parse_date(factor_cfg.get("end")),
        refresh=bool(factor_cfg.get("refresh", False)),
        overwrite=bool(factor_cfg.get("overwrite", False)),
        cfg=factor_cfg,
    )
