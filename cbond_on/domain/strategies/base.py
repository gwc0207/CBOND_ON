from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class StrategyContext:
    trade_date: date
    prev_positions: pd.DataFrame | None
    config: dict


class Strategy(ABC):
    strategy_id: str = ""

    @abstractmethod
    def select(self, universe: pd.DataFrame, ctx: StrategyContext) -> pd.DataFrame:
        """
        Return columns: code, score, weight, rank
        """
        raise NotImplementedError

