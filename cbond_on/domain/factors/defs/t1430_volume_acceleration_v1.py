import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_acceleration_v1")
class T1430VolumeAccelerationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            if len(window_df) < 2:
                return np.nan
            
            volumes = window_df["volume"].values
            # Check for invalid volumes
            if np.any(volumes <= 0):
                return np.nan
            
            # Cumulative volume
            cum_vol = np.cumsum(volumes)
            n = len(cum_vol)
            x = np.arange(n, dtype=float)
            
            # Linear regression slope
            x_mean = np.mean(x)
            y_mean = np.mean(cum_vol)
            denom = np.sum((x - x_mean) ** 2)
            
            if denom == 0:
                return np.nan
                
            slope = np.sum((x - x_mean) * (cum_vol - y_mean)) / denom
            
            # Normalize by total volume to make it scale-invariant
            total_vol = np.sum(volumes)
            if total_vol <= 0:
                return np.nan
                
            return slope / total_vol

        return _group_scalar(panel, _calc)