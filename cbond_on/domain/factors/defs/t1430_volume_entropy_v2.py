import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_entropy_v2")
class T1430VolumeEntropyV2(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        if "trade_time" not in ctx.panel.columns or "volume" not in ctx.panel.columns:
            raise KeyError("Missing required fields: trade_time, volume")
        
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            if window_df.empty:
                return np.nan
            
            vols = window_df["volume"].values
            vols = vols[vols > 0]
            
            if len(vols) == 0:
                return np.nan
            
            total_vol = np.sum(vols)
            if total_vol <= 0 or not np.isfinite(total_vol):
                return np.nan
            
            probs = vols / total_vol
            # Filter out zero probabilities to avoid log(0)
            probs = probs[probs > 0]
            
            if len(probs) == 0:
                return np.nan
            
            entropy = -np.sum(probs * np.log(probs))
            return float(entropy)
        
        return _group_scalar(panel, _calc)