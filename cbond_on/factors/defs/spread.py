from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("spread")
class SpreadFactor(Factor):
    name = "spread"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        bid_col = str(ctx.params.get("bid_col", "bid_price1"))
        ask_col = str(ctx.params.get("ask_col", "ask_price1"))
        if bid_col not in panel.columns or ask_col not in panel.columns:
            missing = [c for c in [bid_col, ask_col] if c not in panel.columns]
            raise KeyError(f"spread missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            bid = float(df[bid_col].iloc[-1])
            ask = float(df[ask_col].iloc[-1])
            mid = (bid + ask) / 2.0
            if mid == 0:
                return 0.0
            return float((ask - bid) / mid)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
