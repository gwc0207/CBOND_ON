import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("microprice_deviation_liq_v1")
class MicropriceDeviationLiqV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "ask_price1", "bid_price1", "ask_volume1", "bid_volume1", "volume"]
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
                
            total_depth = ask_v1 + bid_v1
            if total_depth <= 0:
                return pd.NA
                
            # Microprice calculation
            microprice = (ask_p1 * bid_v1 + bid_p1 * ask_v1) / total_depth
            
            # Deviation normalized by price and scaled by liquidity/turnover ratio
            deviation = (last - microprice) / last
            liq_scale = total_depth / vol
            
            return deviation * liq_scale
            
        return _group_scalar(panel, _calc)