from __future__ import annotations

from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.run_backtest import execute as run_backtest


def main() -> None:
    cfg = load_config_file("backtest")
    result = run_backtest(
        start=parse_date(cfg.get("start")),
        end=parse_date(cfg.get("end")),
        cfg=cfg,
    )
    print(f"saved: {result.out_dir}")


if __name__ == "__main__":
    main()

