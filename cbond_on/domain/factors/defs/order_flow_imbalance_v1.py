from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("order_flow_imbalance_v1")
class OrderFlowImbalanceV1Factor(Factor):
    name = "order_flow_imbalance_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["bid_volume1", "ask_volume1"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"order_flow_imbalance_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            bid = float(row["bid_volume1"])
            ask = float(row["ask_volume1"])
            denom = bid + ask + 1e-8
            return float((bid - ask) / denom)

        out = _group_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out



