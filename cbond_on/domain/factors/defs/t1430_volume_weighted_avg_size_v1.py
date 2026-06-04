import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_weighted_avg_size_v1")
class T1430VolumeWeightedAvgSizeV1(Factor):
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
            
            if len(vols) == 0:
                return np.nan
                
            sum_vol_sq = np.sum(vols * vols)
            sum_vol = np.sum(vols)
            
            if sum_vol <= 0:
                return np.nan
                
            return sum_vol_sq / sum_vol
            
        return _group_scalar(panel, _calc)