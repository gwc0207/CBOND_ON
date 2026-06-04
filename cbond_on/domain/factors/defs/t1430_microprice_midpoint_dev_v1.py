import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_microprice_midpoint_dev_v1")
class T1430MicropriceMidpointDevV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Required field {f} missing")
                
        def _calc(window):
            last = window["last"].iloc[-1]
            bid_p = window["bid_price1"].iloc[-1]
            ask_p = window["ask_price1"].iloc[-1]
            bid_v = window["bid_volume1"].iloc[-1]
            ask_v = window["ask_volume1"].iloc[-1]
            
            # Calculate microprice
            total_v = bid_v + ask_v
            if total_v <= 0 or pd.isna(total_v):
                return np.nan
                
            microprice = (bid_p * ask_v + ask_p * bid_v) / total_v
            
            if microprice <= 0 or pd.isna(microprice):
                return np.nan
                
            dev = (last - microprice) / microprice
            return dev
            
        result = _group_scalar(panel, _calc)
        return result