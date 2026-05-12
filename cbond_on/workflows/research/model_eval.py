from __future__ import annotations

from typing import Any

from cbond_on.app.pipelines.model_eval_pipeline import execute as run_model_eval_pipeline


def run(
    eval_cfg: dict[str, Any],
    *,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict[str, Any]:
    return run_model_eval_pipeline(
        eval_cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )

