import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_gini_simple_v1")
class T1430VolumeGiniSimpleV1(Factor):
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
            valid_vols = volumes[volumes > 0]
            
            if len(valid_vols) < 2:
                return np.nan
                
            n = len(valid_vols)
            mean_vol = np.mean(valid_vols)
            if mean_vol <= 0:
                return np.nan
                
            # Gini = Mean Absolute Difference / (2 * Mean)
            mad = np.mean(np.abs(valid_vols[:, None] - valid_vols[None, :]))
            gini = mad / (2 * mean_vol)
            return gini

        return _group_scalar(panel, _calc)