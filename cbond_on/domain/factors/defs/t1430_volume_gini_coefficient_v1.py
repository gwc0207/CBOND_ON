import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_gini_coefficient_v1")
class T1430VolumeGiniCoefficientV1(Factor):
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
            
            if len(window_df) < 2:
                return np.nan
                
            volumes = window_df["volume"].values.astype(float)
            
            if np.any(volumes < 0):
                return np.nan
                
            # Remove zero volumes for Gini calculation? Standard Gini handles zeros.
            # But if all are zero, denom is 0.
            total_vol = np.sum(volumes)
            if total_vol <= 0:
                return np.nan
                
            n = len(volumes)
            # Sort volumes
            sorted_vols = np.sort(volumes)
            
            # Gini formula: (2 * sum(i * y_i) - (n + 1) * sum(y_i)) / (n * sum(y_i))
            indices = np.arange(1, n + 1)
            numerator = 2.0 * np.sum(indices * sorted_vols) - (n + 1) * total_vol
            denominator = n * total_vol
            
            if denominator == 0:
                return np.nan
                
            gini = numerator / denominator
            return float(gini)
            
        return _group_scalar(panel, _calc)