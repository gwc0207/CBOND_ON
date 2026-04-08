from __future__ import annotations

from typing import Any

from cbond_on.core.config import parse_date
from cbond_on.app.usecases.run_backtest import execute as run_backtest


def execute(backtest_cfg: dict[str, Any]):
    return run_backtest(
        start=parse_date(backtest_cfg.get("start")),
        end=parse_date(backtest_cfg.get("end")),
        cfg=backtest_cfg,
    )

