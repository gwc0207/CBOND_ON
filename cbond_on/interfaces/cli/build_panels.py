from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.build_panel import execute as build_panel
from cbond_on.app.usecases.build_labels import execute as build_labels


def main() -> None:
    panel_cfg = load_config_file("panel")
    label_cfg = load_config_file("label")
    start = parse_date(panel_cfg.get("start"))
    end = parse_date(panel_cfg.get("end"))
    build_panel(
        start=start,
        end=end,
        refresh=bool(panel_cfg.get("refresh", False)),
        overwrite=bool(panel_cfg.get("overwrite", False)),
        cfg=panel_cfg,
    )
    build_labels(
        start=parse_date(label_cfg.get("start", start)),
        end=parse_date(label_cfg.get("end", end)),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
    )


if __name__ == "__main__":
    main()

