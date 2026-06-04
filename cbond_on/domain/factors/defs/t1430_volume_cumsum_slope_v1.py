import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_cumsum_slope_v1")
class T1430VolumeCumsumSlopeV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["trade_time", "volume"]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required field: {field}")

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            n = len(window_df)
            if n < 3:
                return np.nan
                
            volumes = window_df["volume"].values
            if not np.all(np.isfinite(volumes)):
                return np.nan
                
            cum_vol = np.cumsum(volumes)
            x = np.arange(n, dtype=float)
            
            mean_x = np.mean(x)
            mean_y = np.mean(cum_vol)
            
            denom = np.sum((x - mean_x) ** 2)
            if denom <= 0:
                return np.nan
                
            slope = np.sum((x - mean_x) * (cum_vol - mean_y)) / denom
            
            if not np.isfinite(slope):
                return np.nan
                
            return slope

        return _group_scalar(panel, _calc)