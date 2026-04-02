from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("microprice_bias")
class MicropriceBiasFactor(Factor):
    name = "microprice_bias"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        bid_col = str(ctx.params.get("bid_col", "bid_price1"))
        ask_col = str(ctx.params.get("ask_col", "ask_price1"))
        bid_vol_col = str(ctx.params.get("bid_vol_col", "bid_volume1"))
        ask_vol_col = str(ctx.params.get("ask_vol_col", "ask_volume1"))
        required = [bid_col, ask_col, bid_vol_col, ask_vol_col]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"microprice_bias missing columns: {missing}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            row = df.iloc[-1]
            bid = float(row[bid_col])
            ask = float(row[ask_col])
            bid_vol = float(row[bid_vol_col])
            ask_vol = float(row[ask_vol_col])
            denom = bid_vol + ask_vol
            if bid <= 0 or ask <= 0 or denom <= 0:
                return 0.0
            micro = (ask * bid_vol + bid * ask_vol) / denom
            mid = (ask + bid) / 2.0
            if mid <= 0:
                return 0.0
            return float((micro - mid) / mid)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

