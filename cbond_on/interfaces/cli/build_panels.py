from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.pipelines.panel_pipeline import execute as run_panel_pipeline
from cbond_on.app.pipelines.label_pipeline import execute as run_label_pipeline


def main() -> None:
    panel_cfg = load_config_file("panel")
    label_cfg = load_config_file("label")
    start = parse_date(panel_cfg.get("start"))
    end = parse_date(panel_cfg.get("end"))
    panel_cfg["start"] = start
    panel_cfg["end"] = end
    label_cfg["start"] = parse_date(label_cfg.get("start", start))
    label_cfg["end"] = parse_date(label_cfg.get("end", end))
    run_panel_pipeline(panel_cfg)
    run_label_pipeline(label_cfg, panel_cfg=panel_cfg)


if __name__ == "__main__":
    main()

