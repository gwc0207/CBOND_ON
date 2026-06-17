from __future__ import annotations

from typing import Any

from cbond_on.config.loader import load_config_file
from cbond_on.schemas.config.strategy_backtest import validate_strategy_backtest_config


def load_strategy_backtest_config(config_name: str = "backtest") -> dict[str, Any]:
    return validate_strategy_backtest_config(load_config_file(config_name))
