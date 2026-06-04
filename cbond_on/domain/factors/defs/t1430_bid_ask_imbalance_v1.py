import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_bid_ask_imbalance_v1")
class T1430BidAskImbalanceV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if "bid_volume1" not in panel.columns or "ask_volume1" not in panel.columns:
            raise KeyError("Required fields bid_volume1 or ask_volume1 missing")
            
        def _calc(window):
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            
            # Guard against division by zero
            denom = bid_v + ask_v
            denom = denom.replace(0, np.nan)
            
            imb = (bid_v - ask_v) / denom
            return imb.iloc[-1] if not imb.empty else np.nan
            
        result = _group_scalar(panel, _calc)
        return result