from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.factor_batch.runner import build_signal_specs, run_factor_batch
from cbond_on.factors.defs import aacb, volen  # noqa: F401


def main() -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("factor_batch")

    start = parse_date(cfg["start"])
    end = parse_date(cfg["end"])
    overwrite = bool(cfg.get("overwrite", False))
    full_refresh = bool(cfg.get("full_refresh", False))
    window_minutes = int(cfg.get("window_minutes", 15))
    panel_name = cfg.get("panel_name")

    panel_data_root = paths_cfg.get("panel_data_root")
    factor_data_root = paths_cfg["factor_data_root"]
    label_data_root = paths_cfg.get("label_data_root")
    results_root = paths_cfg.get("results_root") or paths_cfg.get("results_root")
    raw_data_root = paths_cfg["raw_data_root"]

    specs = build_signal_specs(cfg)
    print(
        f"[factor_batch] start={start} end={end} panel_name={panel_name} "
        f"window_minutes={window_minutes} overwrite={overwrite} full_refresh={full_refresh} "
        f"factor_time={cfg.get('factor_time')} label_time={cfg.get('label_time')} specs={len(specs)}"
    )
    out_dir = run_factor_batch(
        cfg,
        panel_data_root=panel_data_root,
        factor_data_root=factor_data_root,
        label_data_root=label_data_root,
        raw_data_root=raw_data_root,
        results_root=results_root,
        start=start,
        end=end,
        window_minutes=window_minutes,
        panel_name=panel_name,
        overwrite=overwrite,
        full_refresh=full_refresh,
        specs=specs,
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
