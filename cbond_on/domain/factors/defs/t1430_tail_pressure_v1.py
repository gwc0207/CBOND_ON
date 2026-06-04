import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_tail_pressure_v1")
class T1430TailPressureV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            last = group["last"]
            high = group["high"]
            low = group["low"]
            volume = group["volume"]
            amount = group["amount"]
            
            range_denom = (high - low).replace(0, 1e-8)
            # Amount / Last approximates 'effective volume' if price was constant at Last
            eff_vol_denom = (amount / last).replace(0, 1e-8)
            
            pos = (last - low) / range_denom
            vol_ratio = volume / eff_vol_denom
            
            return (pos * vol_ratio).iloc[-1]
            
        result = _group_scalar(panel, _calc)
        return result