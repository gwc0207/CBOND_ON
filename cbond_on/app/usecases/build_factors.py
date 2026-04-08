from __future__ import annotations

from datetime import date

from cbond_on.services.factor.factor_build_service import run as run_factor_build


def execute(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    return run_factor_build(
        start=start,
        end=end,
        refresh=refresh,
        overwrite=overwrite,
        cfg=cfg,
    )

