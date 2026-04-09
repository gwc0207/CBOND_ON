from __future__ import annotations

from pathlib import Path

from cbond_on.infra.model.adapters import build_adapter


def build_model_runner(*, model_type: str, model_config_path: Path):
    return build_adapter(model_type, model_config_path=model_config_path)

