from __future__ import annotations

import argparse

from cbond_on.app.pipelines.pipeline_all import execute as run_pipeline_all


def main(*, config_name: str = "pipeline_all") -> None:
    run_pipeline_all(config_name=config_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline using one pipeline_all config.")
    parser.add_argument(
        "--config",
        default="pipeline_all",
        help="config key or config path for pipeline_all (default: pipeline_all)",
    )
    args = parser.parse_args()
    main(config_name=args.config)

