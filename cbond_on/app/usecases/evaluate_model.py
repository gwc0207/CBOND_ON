from __future__ import annotations

from cbond_on.app.usecases.model_eval_runtime import run as run_model_eval


def execute(
    *,
    cfg: dict | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict:
    return run_model_eval(
        cfg=cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )

