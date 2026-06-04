import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar

@FactorRegistry.register("tail_volume_absorption_v1")
class TailVolumeAbsorptionV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            vol = group['volume'].values
            last = group['last'].values
            pre = group['pre_close'].values
            
            valid_mask = (vol >= 0) & (pre > 0) & (last > 0)
            
            ret = np.abs(last - pre) / (pre + 1e-8)
            
            # Absorption ratio: Volume per unit of absolute return
            absorption = vol / (ret + 1e-8)
            
            result = absorption
            result[~valid_mask] = np.nan
            
            return pd.Series(result, index=group.index).mean()
            
        return _group_scalar(panel, _calc)