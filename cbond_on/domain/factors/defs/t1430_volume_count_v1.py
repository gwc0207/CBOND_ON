import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_count_v1")
class T1430VolumeCountV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ctx.panel
        if "trade_time" not in panel.columns:
            raise KeyError("Missing required field: trade_time")
        if "volume" not in panel.columns:
            raise KeyError("Missing required field: volume")

        panel = ensure_trade_time(panel)

        def _calc(group):
            window_minutes = int(ctx.params.get("window_minutes", 30))
            sliced = slice_window(group.sort_values("trade_time"), window_minutes)
            if sliced.empty:
                return np.nan
            # Count non-zero volume entries
            valid_vol = sliced[sliced["volume"] > 0]
            count = len(valid_vol)
            if count == 0:
                return np.nan
            return float(count)

        return _group_scalar(panel, _calc)