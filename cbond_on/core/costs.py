from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """
    交易成本模型：
    - 买入：TWAP + bps（更优）
    - 卖出：TWAP - bps（更优）
    - 佣金万 0.7（单边）
    """

    buy_bps: float = 1.0
    sell_bps: float = 1.0
    commission_rate: float = 0.00007

    def execution_price(self, twap: float, side: str) -> float:
        if side not in ("buy", "sell"):
            raise ValueError("side 只能是 'buy' 或 'sell'")
        if side == "buy":
            adj = self.buy_bps / 10000.0
            return twap * (1.0 + adj)
        adj = self.sell_bps / 10000.0
        return twap * (1.0 - adj)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_rate
