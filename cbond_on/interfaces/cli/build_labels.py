from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.pipelines.label_pipeline import execute as run_label_pipeline


def main() -> None:
    label_cfg = load_config_file("label")
    label_cfg["start"] = parse_date(label_cfg.get("start"))
    label_cfg["end"] = parse_date(label_cfg.get("end"))
    result = run_label_pipeline(label_cfg)
    print(result)


if __name__ == "__main__":
    main()

