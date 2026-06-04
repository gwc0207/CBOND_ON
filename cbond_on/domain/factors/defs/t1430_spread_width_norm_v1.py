import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

EPS = 1e-8

@FactorRegistry.register("t1430_spread_width_norm_v1")
class T1430SpreadWidthNormV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if "bid_price1" not in panel.columns or "ask_price1" not in panel.columns:
            raise KeyError("Required fields missing")
            
        def _calc(window):
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            
            mid_price = (bid_p + ask_p) / 2.0
            spread = ask_p - bid_p
            
            # Guard against zero mid_price
            valid_mid = mid_price.replace(0, np.nan)
            if valid_mid.isna().all():
                return np.nan
                
            norm_spread = spread / valid_mid
            return norm_spread.mean()
            
        result = _group_scalar(panel, _calc)
        return result