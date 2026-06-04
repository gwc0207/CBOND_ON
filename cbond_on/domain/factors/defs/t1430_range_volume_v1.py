import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_range_volume_v1")
class T1430RangeVolumeV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            high = group["high"]
            low = group["low"]
            last = group["last"]
            volume = group["volume"]
            
            valid_last = last > 0
            valid_hl = high >= low
            
            mask = valid_last & valid_hl
            
            res = pd.Series(index=group.index, dtype=float)
            res[:] = float('nan')
            
            if mask.any():
                range_norm = (high[mask] - low[mask]) / last[mask]
                res[mask] = range_norm * volume[mask]
                
            return res.iloc[-1] if not res.empty else float('nan')

        return _group_scalar(panel, _calc)