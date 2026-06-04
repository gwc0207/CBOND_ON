import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_micro_price_deviation_v1")
class T1430MicroPriceDeviationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            last = window["last"].astype(float)
            bid_p = window["bid_price1"].astype(float)
            ask_p = window["ask_price1"].astype(float)
            bid_v = window["bid_volume1"].astype(float)
            ask_v = window["ask_volume1"].astype(float)
            
            denom = bid_v + ask_v
            denom = denom.replace(0, np.nan)
            
            micro_price = (bid_p * ask_v + ask_p * bid_v) / denom
            micro_price = micro_price.replace(0, np.nan)
            
            dev = (last - micro_price) / micro_price
            return dev.iloc[-1] if not dev.empty else np.nan
            
        result = _group_scalar(panel, _calc)
        return result