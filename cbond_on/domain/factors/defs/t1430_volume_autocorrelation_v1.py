import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_autocorrelation_v1")
class T1430VolumeAutocorrelationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ctx.panel
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")
            
        panel = ensure_trade_time(panel)
        
        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            if len(window_df) < 3:
                return np.nan
                
            volumes = window_df["volume"].values.astype(float)
            
            if np.any(volumes < 0):
                return np.nan
                
            # Calculate first-order autocorrelation
            # r = cov(x_t, x_{t-1}) / var(x)
            x_t = volumes[1:]
            x_tm1 = volumes[:-1]
            
            mean_x = np.mean(volumes)
            
            cov_tt1 = np.mean((x_t - mean_x) * (x_tm1 - mean_x))
            var_x = np.var(volumes)
            
            if var_x == 0:
                return np.nan
                
            autocorr = cov_tt1 / var_x
            return float(autocorr)
            
        return _group_scalar(panel, _calc)