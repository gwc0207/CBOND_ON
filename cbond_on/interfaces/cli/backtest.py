from __future__ import annotations

from cbond_on.config.loader import load_config_file
from cbond_on.app.pipelines.backtest_pipeline import execute as run_backtest_pipeline


def main() -> None:
    cfg = load_config_file("backtest")
    result = run_backtest_pipeline(cfg)
    print(f"saved: {result.out_dir}")


if __name__ == "__main__":
    main()

