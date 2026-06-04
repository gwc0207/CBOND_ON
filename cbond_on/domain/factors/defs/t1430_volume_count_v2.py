import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_volume_count_v2")
class T1430VolumeCountV2(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")

        def _calc(group: pd.DataFrame) -> float:
            window_minutes = int(ctx.params.get("window_minutes", 30))
            window_df = slice_window(group.sort_values("trade_time"), window_minutes)
            
            if window_df.empty:
                return np.nan
            
            vol_series = window_df["volume"]
            
            # Count snapshots with strictly positive volume
            valid_vol = vol_series[vol_series > 0]
            count = len(valid_vol)
            
            if count == 0:
                return np.nan
                
            return float(count)

        return _group_scalar(panel, _calc)