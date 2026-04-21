from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.interfaces.cli.factor_select import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run factor selection by model uplift")
    parser.add_argument("--config", default="score/factor_select")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    main(
        config_name=args.config,
        start=args.start,
        end=args.end,
    )

