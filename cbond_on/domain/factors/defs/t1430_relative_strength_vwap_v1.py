import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_relative_strength_vwap_v1")
class T1430RelativeStrengthVwapV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required = ["last", "amount", "volume"]
        for f in required:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(window):
            last = window["last"]
            amt = window["amount"]
            vol = window["volume"]
            
            vol_safe = vol.replace(0, np.nan)
            
            # Cumulative VWAP
            cum_amt = amt.cumsum()
            cum_vol = vol_safe.cumsum()
            
            vwap = cum_amt / cum_vol
            
            # Current deviation
            curr_last = last.iloc[-1]
            curr_vwap = vwap.iloc[-1]
            
            if pd.isna(curr_vwap) or curr_vwap == 0:
                return np.nan
                
            factor_val = (curr_last - curr_vwap) / curr_vwap
            
            return factor_val
            
        return _group_scalar(panel, _calc)