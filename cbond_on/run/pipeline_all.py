from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.run.sync_data import main as sync_raw_data
from cbond_on.run.build_cleaned_data import main as build_cleaned_data
from cbond_on.run.build_panels import main as build_panels
from cbond_on.run.factor_batch import main as run_factor_batch
from cbond_on.run.model_score import main as run_model_score
from cbond_on.run.backtest import main as run_backtest


def main() -> None:
    sync_raw_data()
    build_cleaned_data()
    build_panels()
    run_factor_batch()
    run_model_score()
    run_backtest()


if __name__ == "__main__":
    main()
