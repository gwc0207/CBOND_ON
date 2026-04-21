from __future__ import annotations

from cbond_on.app.usecases.factor_select_runtime import run as run_factor_select


def execute(
    *,
    cfg: dict | None = None,
    config_name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    return run_factor_select(
        cfg=cfg,
        config_name=config_name,
        start=start,
        end=end,
    )

