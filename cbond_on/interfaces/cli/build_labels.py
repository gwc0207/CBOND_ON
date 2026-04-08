from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.build_labels import execute as build_labels


def main() -> None:
    label_cfg = load_config_file("label")
    result = build_labels(
        start=parse_date(label_cfg.get("start")),
        end=parse_date(label_cfg.get("end")),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
    )
    print(result)


if __name__ == "__main__":
    main()

