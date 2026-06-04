import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_range_vol_norm_v1")
class T1430RangeVolNormV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            high = group["high"]
            low = group["low"]
            last = group["last"]
            volume = group["volume"]
            
            # Guard denominator
            last_safe = last.replace(0, 1e-8)
            
            # Normalized range
            norm_range = (high - low) / last_safe
            
            # Log volume
            log_vol = np.log1p(volume)
            
            return (norm_range * log_vol).iloc[-1]
            
        result = _group_scalar(panel, _calc)
        return result