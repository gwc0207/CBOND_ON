from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("bid_ask_spread_v1")
class BidAskSpreadV1Factor(Factor):
    name = "bid_ask_spread_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["bid_price1", "ask_price1"]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"bid_ask_spread_v1 missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            row = df.sort_values("trade_time").iloc[-1]
            bid = float(row["bid_price1"])
            ask = float(row["ask_price1"])
            mid = (ask + bid) / 2.0
            return float((ask - bid) / (mid + 1e-8))

        out = _group_scalar(panel, _calc).fillna(0.0)
        out.name = self.output_name(self.name)
        return out



