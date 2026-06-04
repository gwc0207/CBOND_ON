import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_amount_intensity_v1")
class T1430AmountIntensityV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            amount = group["amount"]
            last = group["last"]
            volume = group["volume"]
            
            # Guard denominator
            denom = last * volume
            valid = denom > 0
            
            res = pd.Series(index=group.index, dtype=float)
            res[:] = float('nan')
            
            if valid.any():
                res[valid] = amount[valid] / denom[valid]
                
            return res.iloc[-1] if not res.empty else float('nan')

        return _group_scalar(panel, _calc)