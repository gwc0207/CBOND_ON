from __future__ import annotations

from pathlib import Path

from cbond_on.factor_batch.runner import FactorBacktestResult
from cbond_on.report.factor_report import save_single_factor_report


def save_report(
    result: FactorBacktestResult,
    out_dir: Path,
    *,
    factor_name: str,
    factor_col: str,
    trading_days: set,
) -> None:
    save_single_factor_report(
        result,
        out_dir,
        factor_name=factor_name,
        factor_col=factor_col,
        trading_days=trading_days,
    )

