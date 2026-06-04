import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_volume_max_v1")
class T1430VolumeMaxV1(Factor):
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
            
            # Guard against invalid volume values
            if (vol_series < 0).any() or vol_series.isna().all():
                return np.nan
                
            max_vol = vol_series.max()
            
            if np.isnan(max_vol) or max_vol <= 0:
                return np.nan
                
            return float(max_vol)

        return _group_scalar(panel, _calc)