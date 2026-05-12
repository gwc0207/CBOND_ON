from __future__ import annotations

from typing import Any

from cbond_on.schemas.config._base import require_mapping


def validate_model_eval_config(cfg: Any) -> dict[str, Any]:
    return require_mapping(cfg, name="model_eval_config")

