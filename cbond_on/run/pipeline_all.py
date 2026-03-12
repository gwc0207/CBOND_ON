from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file
from cbond_on.run.sync_data import main as sync_raw_data
from cbond_on.run.build_cleaned_data import main as build_cleaned_data
from cbond_on.run.build_panels import main as build_panels
from cbond_on.run.factor_batch import main as run_factor_batch
from cbond_on.run.model_score import main as run_model_score
from cbond_on.run.backtest import main as run_backtest


def main() -> None:
    try:
        pipeline_cfg = load_config_file("pipeline_all")
    except FileNotFoundError:
        pipeline_cfg = {}
    model_cfg = pipeline_cfg.get("model_score", {})

    sync_raw_data()
    build_cleaned_data()
    build_panels()
    run_factor_batch()
    run_model_score(
        model_type=model_cfg.get("model_type"),
        model_config=model_cfg.get("model_config"),
        start=model_cfg.get("start"),
        end=model_cfg.get("end"),
        label_cutoff=model_cfg.get("label_cutoff"),
    )
    run_backtest()


if __name__ == "__main__":
    main()
