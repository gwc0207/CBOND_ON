import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_cumulative_volume_ratio_v1")
class T1430CumulativeVolumeRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ctx.panel
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")
            
        panel = ensure_trade_time(panel)
        
        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            tail_minutes = int(ctx.params.get("tail_minutes", 5))
            
            sorted_group = group.sort_values("trade_time")
            window_df = slice_window(sorted_group, window_minutes)
            
            if len(window_df) == 0:
                return np.nan
                
            total_vol = window_df["volume"].sum()
            
            if total_vol <= 0:
                return np.nan
                
            # Get tail window
            tail_df = slice_window(sorted_group, tail_minutes)
            tail_vol = tail_df["volume"].sum()
            
            ratio = tail_vol / total_vol
            return float(ratio)
            
        return _group_scalar(panel, _calc)