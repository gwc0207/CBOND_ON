from __future__ import annotations

import argparse

from cbond_on.bootstrap.backtest import load_strategy_backtest_config
from cbond_on.workflows.backtest.strategy_backtest import run


def main(*, config_name: str | None = None) -> None:
    cfg = load_strategy_backtest_config(config_name or "backtest")
    result = run(cfg)
    print(f"saved: {result.out_dir}")


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--config", default="backtest")
    args = parser.parse_args(argv)
    main(config_name=args.config)


if __name__ == "__main__":
    cli_main()
