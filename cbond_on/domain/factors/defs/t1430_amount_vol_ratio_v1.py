import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_amount_vol_ratio_v1")
class T1430AmountVolRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["volume", "amount"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(group):
            volume = group["volume"].sum()
            amount = group["amount"].sum()
            
            eps = 1e-8
            denom = volume + eps
            
            if denom <= 0:
                return pd.NA
                
            return amount / denom
            
        return _group_scalar(panel, _calc)