from __future__ import annotations

from typing import Any

from cbond_on.config.loader import load_config_file
from cbond_on.schemas.config.factor_batch import validate_factor_batch_config
from cbond_on.schemas.config.model_eval import validate_model_eval_config
from cbond_on.schemas.config.model_score import validate_model_score_config
from cbond_on.schemas.config.shared import validate_paths_config


def load_factor_batch_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = validate_factor_batch_config(load_config_file("factor"))
    paths_cfg = validate_paths_config(load_config_file("paths"))
    return cfg, paths_cfg


def load_model_score_config() -> dict[str, Any]:
    return validate_model_score_config(load_config_file("model_score"))


def load_model_eval_config(config_name: str = "score/evaluation/model_eval") -> dict[str, Any]:
    return validate_model_eval_config(load_config_file(config_name))

