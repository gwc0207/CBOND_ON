from __future__ import annotations

from datetime import date

from cbond_on.app.usecases.panel_runtime import run as run_panel


def execute(
    *,
    start: date | None = None,
    end: date | None = None,
    refresh: bool | None = None,
    overwrite: bool | None = None,
    cfg: dict | None = None,
) -> dict:
    return run_panel(
        start=start,
        end=end,
        refresh=refresh,
        overwrite=overwrite,
        cfg=cfg,
    )

