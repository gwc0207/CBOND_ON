from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.run_factor_batch import execute as run_factor_batch
from cbond_on.factors import defs  # noqa: F401


def main() -> None:
    cfg = load_config_file("factor")
    paths_cfg = load_config_file("paths")
    start_day = parse_date(cfg.get("start"))
    end_day = parse_date(cfg.get("end"))
    refresh = bool(cfg.get("refresh", False))
    overwrite = bool(cfg.get("overwrite", False))

    out_root = run_factor_batch(
        cfg=cfg,
        paths_cfg=paths_cfg,
        start=start_day,
        end=end_day,
        refresh=refresh,
        overwrite=overwrite,
    )
    print({"out_root": str(out_root)})


if __name__ == "__main__":
    main()

