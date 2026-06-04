import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_relative_strength_v1")
class T1430RelativeStrengthV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "pre_close"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            last = window["last"].astype(float)
            pre_close = window["pre_close"].astype(float)
            
            pre_close = pre_close.replace(0, np.nan)
            ret = (last - pre_close) / pre_close
            return ret.iloc[-1] if not ret.empty else np.nan
            
        result = _group_scalar(panel, _calc)
        return result