from __future__ import annotations

from datetime import date

from cbond_on.services.backtest.backtest_service import run as run_backtest
from cbond_on.services.backtest.backtest_service import BacktestRunResult


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

