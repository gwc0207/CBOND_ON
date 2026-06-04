import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("spread_depth_pressure_v1")
class SpreadDepthPressureV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["ask_price1", "bid_price1", "last", "ask_volume1", "bid_volume1", "volume"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(group):
            last = group["last"].iloc[-1]
            ask_p1 = group["ask_price1"].iloc[-1]
            bid_p1 = group["bid_price1"].iloc[-1]
            ask_v1 = group["ask_volume1"].iloc[-1]
            bid_v1 = group["bid_volume1"].iloc[-1]
            vol = group["volume"].iloc[-1]
            
            if last <= 0 or vol <= 0:
                return pd.NA
                
            spread = ask_p1 - bid_p1
            depth = ask_v1 + bid_v1
            
            # Normalized pressure: spread cost weighted by depth/turnover
            val = -(spread / last) * (depth / vol)
            return val
            
        return _group_scalar(panel, _calc)