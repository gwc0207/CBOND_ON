from __future__ import annotations

import argparse

from cbond_on.config.loader import load_config_file
from cbond_on.app.pipelines.factor_select_pipeline import execute as run_factor_select_pipeline


def main(
    *,
    config_name: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> None:
    cfg_name = config_name or "score/factor_select"
    cfg = load_config_file(cfg_name)
    result = run_factor_select_pipeline(
        cfg,
        config_name=cfg_name,
        start=start,
        end=end,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run factor-to-model uplift selection")
    parser.add_argument("--config", default="score/factor_select")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    main(
        config_name=args.config,
        start=args.start,
        end=args.end,
    )

