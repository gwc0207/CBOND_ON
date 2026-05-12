from __future__ import annotations

from typing import Any

from cbond_on.app.pipelines.backtest_pipeline import execute as run_backtest_pipeline


def run(backtest_cfg: dict[str, Any]):
    return run_backtest_pipeline(backtest_cfg)

