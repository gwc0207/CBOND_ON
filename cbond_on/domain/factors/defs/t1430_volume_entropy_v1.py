import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_entropy_v1")
class T1430VolumeEntropyV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            if len(window_df) < 1:
                return np.nan
            
            volumes = window_df["volume"].values
            valid_vols = volumes[volumes > 0]
            
            if len(valid_vols) == 0:
                return np.nan
                
            total_vol = np.sum(valid_vols)
            if total_vol <= 0:
                return np.nan
                
            probs = valid_vols / total_vol
            # Shannon entropy: -sum(p * log(p))
            # Filter out zero probabilities to avoid log(0)
            nonzero_probs = probs[probs > 0]
            entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
            return entropy

        return _group_scalar(panel, _calc)