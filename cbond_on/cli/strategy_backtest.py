from __future__ import annotations

from cbond_on.bootstrap.backtest import load_strategy_backtest_config
from cbond_on.workflows.backtest.strategy_backtest import run


def main() -> None:
    cfg = load_strategy_backtest_config()
    result = run(cfg)
    print(f"saved: {result.out_dir}")


if __name__ == "__main__":
    main()

