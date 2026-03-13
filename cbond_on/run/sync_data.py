from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.data.raw_service import run as run_raw


def main() -> None:
    cfg = load_config_file("raw_data")
    result = run_raw(
        start=parse_date(cfg.get("start")),
        end=parse_date(cfg.get("end")),
        refresh=bool(cfg.get("refresh", False)),
        overwrite=bool(cfg.get("overwrite", False)),
        cfg=cfg,
    )
    print(result)


if __name__ == "__main__":
    main()
