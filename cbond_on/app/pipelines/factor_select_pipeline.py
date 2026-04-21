from __future__ import annotations

from typing import Any

from cbond_on.app.usecases.select_factors import execute as select_factors


def execute(
    cfg: dict[str, Any] | None = None,
    *,
    config_name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    return select_factors(
        cfg=cfg,
        config_name=config_name,
        start=start,
        end=end,
    )

