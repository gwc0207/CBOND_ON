from __future__ import annotations

from typing import Any

from cbond_on.app.usecases.evaluate_model import execute as evaluate_model


def execute(
    eval_cfg: dict[str, Any],
    *,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict[str, Any]:
    return evaluate_model(
        cfg=eval_cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )
