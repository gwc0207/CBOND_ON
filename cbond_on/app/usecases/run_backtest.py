from __future__ import annotations

from datetime import date

from cbond_on.app.usecases.backtest_runtime import BacktestRunResult
from cbond_on.app.usecases.backtest_runtime import run as run_backtest


def execute(
    *,
    start: date | None = None,
    end: date | None = None,
    cfg: dict | None = None,
) -> BacktestRunResult:
    return run_backtest(
        start=start,
        end=end,
        cfg=cfg,
    )
