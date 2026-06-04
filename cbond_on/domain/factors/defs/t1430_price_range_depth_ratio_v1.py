import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_price_range_depth_ratio_v1")
class T1430PriceRangeDepthRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["high", "low", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            last_row = window.iloc[-1]
            
            high = last_row["high"]
            low = last_row["low"]
            bid_vol = last_row["bid_volume1"]
            ask_vol = last_row["ask_volume1"]
            
            price_range = high - low
            if price_range < 0:
                return pd.NA
                
            total_depth = bid_vol + ask_vol
            if total_depth <= 0:
                return pd.NA
                
            ratio = price_range / total_depth
            return ratio
            
        result = _group_scalar(panel, _calc)
        return result