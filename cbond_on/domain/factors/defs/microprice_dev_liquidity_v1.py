import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("microprice_dev_liquidity_v1")
class MicropriceDevLiquidityV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
        
        def _calc(group):
            last = group["last"].values[-1]
            bp1 = group["bid_price1"].values[-1]
            ap1 = group["ask_price1"].values[-1]
            bv1 = group["bid_volume1"].values[-1]
            av1 = group["ask_volume1"].values[-1]
            
            if bp1 <= 0 or ap1 <= 0:
                return np.nan
                
            total_vol = bv1 + av1
            if total_vol <= 0:
                return 0.0
                
            micro_price = (bp1 * av1 + ap1 * bv1) / total_vol
            
            return last - micro_price

        return _group_scalar(panel, _calc)