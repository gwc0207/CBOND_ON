from __future__ import annotations

from typing import Any

from cbond_on.schemas.config._base import require_keys, require_mapping


def validate_strategy_backtest_config(cfg: Any) -> dict[str, Any]:
    out = require_mapping(cfg, name="strategy_backtest_config")
    require_keys(out, name="strategy_backtest_config", keys=("start", "end"))
    return out

