import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_pv_ratio_v1")
class T1430PvRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "high", "low", "volume", "amount"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(group):
            last = group["last"].iloc[-1]
            high = group["high"].max()
            low = group["low"].min()
            volume = group["volume"].sum()
            amount = group["amount"].sum()
            
            # Guard denominators
            range_val = high - low
            if range_val <= 0:
                range_val = 1e-8
                
            avg_price = amount / volume if volume > 0 else last
            if avg_price <= 0:
                avg_price = 1e-8
                
            pos_in_range = (last - low) / range_val
            vol_intensity = volume / avg_price
            
            return pos_in_range * vol_intensity
            
        return _group_scalar(panel, _calc)