from __future__ import annotations

from typing import Any

from cbond_on.app.usecases.train_score import execute as train_score


def execute(
    model_cfg: dict[str, Any],
    *,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict:
    return train_score(
        model_id=model_id or model_cfg.get("model_id") or model_cfg.get("default_model_id"),
        start=start or model_cfg.get("start"),
        end=end or model_cfg.get("end"),
        label_cutoff=label_cutoff or model_cfg.get("label_cutoff"),
        cfg=model_cfg,
    )

