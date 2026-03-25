from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.data.label_service import run as run_label


def main() -> None:
    label_cfg = load_config_file("label")
    result = run_label(
        start=parse_date(label_cfg.get("start")),
        end=parse_date(label_cfg.get("end")),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
    )
    print(result)


if __name__ == "__main__":
    main()
