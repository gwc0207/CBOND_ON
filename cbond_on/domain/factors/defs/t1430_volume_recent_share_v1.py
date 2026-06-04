import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_recent_share_v1")
class T1430VolumeRecentShareV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["trade_time", "volume"]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required field: {field}")

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            tail_minutes = int(ctx.params.get("tail_minutes", 1))
            
            sorted_group = group.sort_values("trade_time")
            window_df = slice_window(sorted_group, window_minutes)
            
            if window_df.empty:
                return np.nan
                
            total_vol = window_df["volume"].sum()
            if not np.isfinite(total_vol) or total_vol <= 0:
                return np.nan
                
            max_time = window_df["trade_time"].max()
            cutoff_time = max_time - pd.Timedelta(minutes=tail_minutes)
            
            tail_vol = window_df.loc[window_df["trade_time"] > cutoff_time, "volume"].sum()
            
            if not np.isfinite(tail_vol):
                return np.nan
                
            return tail_vol / total_vol

        return _group_scalar(panel, _calc)