from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.app.usecases.backtest_runtime import run
from cbond_on.config.loader import load_config_file


if __name__ == "__main__":
    cfg = load_config_file("backtest_pipeline/backtest_clean82_top30")
    result = run(cfg=cfg)
    print(f"saved: {result.out_dir}")
    print(f"days: {result.days}")
