import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_depth_pressure_return_v1")
class T1430DepthPressureReturnV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["pre_close", "last", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            # Use the last available snapshot in the window for T1430 factor
            last_row = window.iloc[-1]
            
            pre_close = last_row["pre_close"]
            last_price = last_row["last"]
            bid_vol = last_row["bid_volume1"]
            ask_vol = last_row["ask_volume1"]
            
            # Calculate return
            if pre_close <= 0:
                return pd.NA
            ret = (last_price - pre_close) / pre_close
            
            # Calculate total depth
            total_depth = bid_vol + ask_vol
            if total_depth <= 0:
                return pd.NA
                
            # Depth pressure: Return per unit of depth
            # Higher value means more price change for less depth (stronger pressure)
            score = ret / total_depth
            return score
            
        result = _group_scalar(panel, _calc)
        return result