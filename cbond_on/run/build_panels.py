from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel


def main() -> None:
    panel_cfg = load_config_file("panel")
    label_cfg = load_config_file("label")
    start = parse_date(panel_cfg.get("start"))
    end = parse_date(panel_cfg.get("end"))
    run_panel(
        start=start,
        end=end,
        refresh=bool(panel_cfg.get("refresh", False)),
        overwrite=bool(panel_cfg.get("overwrite", False)),
        cfg=panel_cfg,
    )
    run_label(
        start=parse_date(label_cfg.get("start", start)),
        end=parse_date(label_cfg.get("end", end)),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
    )


if __name__ == "__main__":
    main()

