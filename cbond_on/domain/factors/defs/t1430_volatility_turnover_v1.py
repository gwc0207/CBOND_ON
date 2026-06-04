import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volatility_turnover_v1")
class T1430VolatilityTurnoverV1(Factor):
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
            
            if last <= 0:
                return pd.NA
                
            eps = 1e-8
            vol = (high - low) / last
            turnover = amount / (volume * last + eps)
            
            return vol * turnover
            
        return _group_scalar(panel, _calc)