from __future__ import annotations

from pathlib import Path
from typing import Any

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file


def load_eval_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is not None:
        return dict(cfg)
    return dict(load_config_file("score/model_eval"))


def load_score_config(path_key: str) -> dict[str, Any]:
    path = resolve_config_path(path_key)
    return dict(load_json_like(path))


def resolve_single_model(score_cfg: dict[str, Any], model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"model_id not found in score config: {model_id}")
    model_entry = dict(models[model_id])
    model_cfg_key = str(model_entry.get("model_config", "")).strip()
    if not model_cfg_key:
        raise ValueError(f"model entry missing model_config: {model_id}")
    model_cfg = dict(load_json_like(resolve_config_path(model_cfg_key)))
    return model_entry, model_cfg


def build_output_dir(paths_cfg: dict[str, Any], *, experiment_name: str, timestamp: str) -> Path:
    return Path(paths_cfg["results_root"]) / "model_eval" / experiment_name / timestamp
