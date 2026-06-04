import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("microprice_deviation_liq_cond_v1")
class MicropriceDeviationLiqCondV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["last", "ask_price1", "ask_volume1", "bid_price1", "bid_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
        
        def _calc(group):
            last = group["last"].iloc[-1]
            ask_p1 = group["ask_price1"].iloc[-1]
            ask_v1 = group["ask_volume1"].iloc[-1]
            bid_p1 = group["bid_price1"].iloc[-1]
            bid_v1 = group["bid_volume1"].iloc[-1]
            
            total_depth = ask_v1 + bid_v1
            if total_depth <= 0:
                return 0.0
            
            microprice = (ask_p1 * bid_v1 + bid_p1 * ask_v1) / total_depth
            deviation = last - microprice
            
            # Scale by inverse depth to highlight low-liquidity deviations
            return deviation / (total_depth + 1e-8)
        
        return _group_scalar(panel, _calc)