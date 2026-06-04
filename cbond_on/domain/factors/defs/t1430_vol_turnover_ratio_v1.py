import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_vol_turnover_ratio_v1")
class T1430VolTurnoverRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "volume", "amount"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(group):
            last = group["last"].iloc[-1]
            volume = group["volume"].sum()
            amount = group["amount"].sum()
            
            if last <= 0:
                return pd.NA
                
            est_vol = amount / last
            if est_vol <= 1e-8:
                return pd.NA
                
            return volume / est_vol
            
        result = _group_scalar(panel, _calc)
        return result