import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_count_v3")
class T1430VolumeCountV3(Factor):
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
            if window_df.empty:
                return np.nan
            
            # Count snapshots where volume is valid and non-zero
            vol = window_df["volume"]
            # Check for invalid values (NaN or <= 0)
            valid_vol = vol[vol.notna() & (vol > 0)]
            count = len(valid_vol)
            
            if count == 0:
                return np.nan
            return float(count)

        return _group_scalar(panel, _calc)