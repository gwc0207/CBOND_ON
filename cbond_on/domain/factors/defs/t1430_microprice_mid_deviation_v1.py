import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_microprice_mid_deviation_v1")
class T1430MicropriceMidDeviationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            last = window["last"].iloc[-1]
            bid_p1 = window["bid_price1"].iloc[-1]
            ask_p1 = window["ask_price1"].iloc[-1]
            bid_v1 = window["bid_volume1"].iloc[-1]
            ask_v1 = window["ask_volume1"].iloc[-1]
            
            total_vol = bid_v1 + ask_v1
            if total_vol <= 0:
                return np.nan
                
            # Microprice formula: Bid + (Ask-Bid) * (BidVol / TotalVol)
            microprice = bid_p1 + (ask_p1 - bid_p1) * (bid_v1 / total_vol)
            
            if microprice <= 0:
                return np.nan
                
            return (last - microprice) / microprice
            
        result = _group_scalar(panel, _calc)
        return result