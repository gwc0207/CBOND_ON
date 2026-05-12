from __future__ import annotations

from pathlib import Path
from typing import Any

from cbond_on.app.pipelines.factor_batch_pipeline import execute as run_factor_batch_pipeline


def run(factor_cfg: dict[str, Any], *, paths_cfg: dict[str, Any]) -> Path:
    return run_factor_batch_pipeline(factor_cfg, paths_cfg=paths_cfg)

