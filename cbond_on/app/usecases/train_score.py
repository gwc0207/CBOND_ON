from __future__ import annotations

from datetime import date

from cbond_on.app.usecases.model_score_runtime import run as run_model_score


def execute(
    *,
    model_id: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    label_cutoff: str | date | None = None,
    cfg: dict | None = None,
) -> dict:
    return run_model_score(
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
        cfg=cfg,
    )

