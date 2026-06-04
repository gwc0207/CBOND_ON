import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_microprice_mid_spread_v1")
class T1430MicropriceMidSpreadV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")

        def _calc(window):
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            
            # Guard against zero volumes
            total_v = bid_v + ask_v
            total_v = total_v.replace(0, np.nan)
            
            # Microprice: weighted average of bid/ask by opposite volume
            microprice = (bid_p * ask_v + ask_p * bid_v) / total_v
            
            # Midprice
            midprice = (bid_p + ask_p) / 2.0
            midprice = midprice.replace(0, np.nan)
            
            # Deviation
            diff = microprice - midprice
            
            # Normalize by midprice to make it scale-invariant
            result = diff / midprice
            return result.iloc[-1] if not result.empty else np.nan

        return _group_scalar(panel, _calc)