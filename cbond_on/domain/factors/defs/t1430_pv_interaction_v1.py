import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_pv_interaction_v1")
class T1430PvInteractionV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            last = group["last"]
            high = group["high"]
            low = group["low"]
            amount = group["amount"]
            volume = group["volume"]
            
            # Guard denominators
            range_denom = (high - low).replace(0, 1e-8)
            vwap_denom = (volume * last).replace(0, 1e-8)
            
            # Price position in range
            pos = (last - low) / range_denom
            
            # Turnover intensity (approx VWA P / Last ~ Avg Price / Last, but here just Amount/Vol/Last = 1 if constant price)
            # Actually Amount / (Volume * Last) is close to 1.0 usually. 
            # Let's use Amount / Volume as proxy for average trade price, normalized by last.
            intensity = amount / vwap_denom
            
            return (pos * intensity).iloc[-1]
            
        result = _group_scalar(panel, _calc)
        return result