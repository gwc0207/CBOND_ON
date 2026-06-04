import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_weighted_time_center_v1")
class T1430VolumeWeightedTimeCenterV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["trade_time", "volume"]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required field: {field}")

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            if window_df.empty:
                return np.nan
                
            volumes = window_df["volume"].values
            times = window_df["trade_time"].values
            
            if not np.all(np.isfinite(volumes)):
                return np.nan
                
            total_vol = np.sum(volumes)
            if total_vol <= 0 or not np.isfinite(total_vol):
                return np.nan
                
            min_time = np.min(times)
            max_time = np.max(times)
            duration = max_time - min_time
            
            if duration <= 0:
                return 0.5
                
            weighted_time_sum = np.sum(volumes * (times - min_time))
            center = weighted_time_sum / (total_vol * duration)
            
            if not np.isfinite(center):
                return np.nan
                
            return center

        return _group_scalar(panel, _calc)