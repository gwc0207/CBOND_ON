from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.backtest.backtest_service import run as run_backtest


def main() -> None:
    cfg = load_config_file("backtest")
    result = run_backtest(
        start=parse_date(cfg.get("start")),
        end=parse_date(cfg.get("end")),
        cfg=cfg,
    )
    print(f"saved: {result.out_dir}")


if __name__ == "__main__":
    main()

