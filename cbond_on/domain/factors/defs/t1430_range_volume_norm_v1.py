import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_range_volume_norm_v1")
class T1430RangeVolumeNormV1(Factor):
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
            
            result = pd.Series(index=group.index, dtype=float)
            result[:] = float('nan')
            
            if mask.any():
                rng = (high[mask] - low[mask]) / last[mask]
                log_vol = np.log1p(volume[mask])
                result[mask] = rng * log_vol
                
            return result.iloc[-1] if not result.empty else float('nan')

        return _group_scalar(panel, _calc)