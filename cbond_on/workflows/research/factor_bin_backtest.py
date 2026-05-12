from __future__ import annotations

from pathlib import Path
from typing import Any

from cbond_on.workflows.research.factor_batch import run as run_factor_batch


def run(factor_cfg: dict[str, Any], *, paths_cfg: dict[str, Any]) -> Path:
    """Run the single-factor bin backtest/report path embedded in factor_batch."""
    return run_factor_batch(factor_cfg, paths_cfg=paths_cfg)

