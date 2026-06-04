import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("ob_pressure_spread_recovery_v1")
class ObPressureSpreadRecoveryV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = ["last", "ask_price1", "bid_price1", "ask_volume1", "bid_volume1"]
        for f in required:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(group):
            last = group["last"].iloc[-1]
            ap1 = group["ask_price1"].iloc[-1]
            bp1 = group["bid_price1"].iloc[-1]
            av1 = group["ask_volume1"].iloc[-1]
            bv1 = group["bid_volume1"].iloc[-1]
            
            if last <= 0 or not np.isfinite(last):
                return np.nan
            if ap1 <= 0 or bp1 <= 0 or not np.isfinite(ap1) or not np.isfinite(bp1):
                return np.nan
            if av1 < 0 or bv1 < 0 or not np.isfinite(av1) or not np.isfinite(bv1):
                return np.nan
                
            spread = ap1 - bp1
            norm_spread = spread / last
            
            denom_vol = bv1 + av1 + 1e-8
            imbalance = (bv1 - av1) / denom_vol
            
            return norm_spread * imbalance
            
        return _group_scalar(panel, _calc)