from __future__ import annotations

import argparse
from pathlib import Path

from cbond_on.bootstrap.risk import load_risk_inputs
from cbond_on.workflows.research.cb_risk import run


def main(
    *,
    config_name: str = "risk/cb_risk_v1",
    paths_config_name: str = "paths",
    start: str | None = None,
    end: str | None = None,
    strategy_positions_path: str | Path | None = None,
    benchmark_positions_path: str | Path | None = None,
    output_root: str | Path | None = None,
    write_outputs: bool = True,
) -> dict:
    cfg, paths_cfg = load_risk_inputs(config_name, paths_config_name)
    result = run(
        cfg,
        paths_cfg=paths_cfg,
        start=start,
        end=end,
        strategy_positions_path=strategy_positions_path,
        benchmark_positions_path=benchmark_positions_path,
        output_root=output_root,
        write_outputs=write_outputs,
    )
    print(result)
    return result


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run isolated CB-Risk offline replay/shadow analysis")
    parser.add_argument("--config", default="risk/cb_risk_v1")
    parser.add_argument("--paths-config", default="paths")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--positions", dest="strategy_positions_path")
    parser.add_argument("--benchmark-positions")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true", help="calculate only; do not write output artifacts")
    args = parser.parse_args(argv)
    main(
        config_name=args.config,
        paths_config_name=args.paths_config,
        start=args.start,
        end=args.end,
        strategy_positions_path=args.strategy_positions_path,
        benchmark_positions_path=args.benchmark_positions,
        output_root=args.output_root,
        write_outputs=not bool(args.dry_run),
    )


if __name__ == "__main__":
    cli_main()
