from __future__ import annotations

from cbond_on.config.loader import load_config_file
from cbond_on.app.pipelines.factor_batch_pipeline import execute as run_factor_batch_pipeline
from cbond_on.domain.factors import defs  # noqa: F401


def main() -> None:
    cfg = load_config_file("factor")
    paths_cfg = load_config_file("paths")
    out_root = run_factor_batch_pipeline(cfg, paths_cfg=paths_cfg)
    print({"out_root": str(out_root)})


if __name__ == "__main__":
    main()


