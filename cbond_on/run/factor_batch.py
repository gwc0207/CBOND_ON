from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.factor_batch.runner import build_signal_specs, run_factor_batch
from cbond_on.factors import defs  # noqa: F401


def main() -> None:
    cfg = load_config_file("factor")
    paths_cfg = load_config_file("paths")
    start_day = parse_date(cfg.get("start"))
    end_day = parse_date(cfg.get("end"))
    panel_name = str(cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("factor_config.panel_name is required; window_minutes fallback is disabled")
    refresh = bool(cfg.get("refresh", False))
    overwrite = bool(cfg.get("overwrite", False))

    out_root = run_factor_batch(
        cfg,
        panel_data_root=paths_cfg["panel_data_root"],
        factor_data_root=paths_cfg["factor_data_root"],
        label_data_root=paths_cfg["label_data_root"],
        raw_data_root=paths_cfg["raw_data_root"],
        results_root=paths_cfg["results_root"],
        start=start_day,
        end=end_day,
        window_minutes=15,
        panel_name=panel_name,
        refresh=refresh,
        overwrite=overwrite,
        specs=build_signal_specs(cfg),
    )
    print({"out_root": str(out_root)})


if __name__ == "__main__":
    main()
