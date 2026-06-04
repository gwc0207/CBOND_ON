import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_close_position_v1")
class T1430ClosePositionV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            last = group["last"]
            high = group["high"]
            low = group["low"]
            
            denom = high - low
            valid = denom > 0
            
            res = pd.Series(index=group.index, dtype=float)
            res[:] = float('nan')
            
            if valid.any():
                res[valid] = (last[valid] - low[valid]) / denom[valid]
                
            return res.iloc[-1] if not res.empty else float('nan')

        return _group_scalar(panel, _calc)