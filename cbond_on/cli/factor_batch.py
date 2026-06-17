from __future__ import annotations

import argparse

from cbond_on.bootstrap.research import load_factor_batch_inputs
from cbond_on.domain.factors import defs  # noqa: F401
from cbond_on.workflows.research.factor_batch import run


def main(
    *,
    config_name: str | None = None,
    paths_config_name: str | None = None,
    factor_files: list[str] | None = None,
    results_root: str | None = None,
) -> None:
    cfg, paths_cfg = load_factor_batch_inputs(
        config_name or "factor",
        paths_config_name or "paths",
    )
    if factor_files:
        cfg["factor_files"] = factor_files
        cfg["factors"] = []
    if results_root:
        paths_cfg["results_root"] = results_root
    out_root = run(cfg, paths_cfg=paths_cfg)
    print({"out_root": str(out_root)})


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run factor batch")
    parser.add_argument("--config", default="factor")
    parser.add_argument("--paths-config", default="paths")
    parser.add_argument("--factor-file", action="append", dest="factor_files")
    parser.add_argument("--results-root")
    args = parser.parse_args(argv)
    main(
        config_name=args.config,
        paths_config_name=args.paths_config,
        factor_files=args.factor_files,
        results_root=args.results_root,
    )


if __name__ == "__main__":
    cli_main()
