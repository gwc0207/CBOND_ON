import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_concentration_gini_v1")
class T1430VolumeConcentrationGiniV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")
            
        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            vols = window_df["volume"].values.astype(float)
            vols = vols[vols > 0]
            
            if len(vols) < 2:
                return np.nan
                
            # Gini coefficient calculation
            sorted_vols = np.sort(vols)
            n = len(sorted_vols)
            index = np.arange(1, n + 1)
            
            sum_num = np.sum((2 * index - n - 1) * sorted_vols)
            sum_den = n * np.sum(sorted_vols)
            
            if sum_den == 0:
                return np.nan
                
            return sum_num / sum_den
            
        return _group_scalar(panel, _calc)