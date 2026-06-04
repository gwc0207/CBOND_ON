import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

EPS = 1e-8

@FactorRegistry.register("t1430_mid_price_return_v1")
class T1430MidPriceReturnV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if "bid_price1" not in panel.columns or "ask_price1" not in panel.columns or "pre_close" not in panel.columns:
            raise KeyError("Required fields missing")
            
        def _calc(window):
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            pre_close = window["pre_close"]
            
            mid_price = (bid_p + ask_p) / 2.0
            
            # Guard against zero pre_close
            valid_pre_close = pre_close.replace(0, np.nan)
            if valid_pre_close.isna().all():
                return np.nan
                
            ret = (mid_price - pre_close) / valid_pre_close
            return ret.mean()
            
        result = _group_scalar(panel, _calc)
        return result