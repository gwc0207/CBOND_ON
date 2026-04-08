from __future__ import annotations

# Backward-compatible facade:
# primary implementation moved to app.usecases.backtest_runtime

from cbond_on.app.usecases.backtest_runtime import BacktestRunResult, run

__all__ = ["BacktestRunResult", "run"]

