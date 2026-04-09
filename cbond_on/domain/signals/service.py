from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from cbond_on.domain.strategies import StrategyRegistry
from cbond_on.domain.strategies.base import StrategyContext


@dataclass
class SignalSelectionRequest:
    universe: pd.DataFrame
    trade_date: date
    strategy_id: str
    strategy_config: dict
    prev_positions: pd.DataFrame | None = None


def select_signals(req: SignalSelectionRequest) -> pd.DataFrame:
    strategy = StrategyRegistry.get(req.strategy_id)
    picks = strategy.select(
        req.universe,
        ctx=StrategyContext(
            trade_date=req.trade_date,
            prev_positions=req.prev_positions,
            config=dict(req.strategy_config or {}),
        ),
    )
    if picks is None:
        return pd.DataFrame(columns=["code", "score", "weight", "rank"])
    if picks.empty:
        return pd.DataFrame(columns=["code", "score", "weight", "rank"])
    if "code" not in picks.columns:
        raise KeyError("strategy output missing required column: code")
    return picks


