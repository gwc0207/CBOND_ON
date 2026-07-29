from __future__ import annotations

from typing import Any

from cbond_on.config.loader import load_config_file
from cbond_on.schemas.config.risk import validate_risk_config
from cbond_on.schemas.config.shared import validate_paths_config


def load_risk_inputs(
    config_name: str = "risk/cb_risk_v1",
    paths_config_name: str = "paths",
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        validate_risk_config(load_config_file(config_name)),
        validate_paths_config(load_config_file(paths_config_name)),
    )
