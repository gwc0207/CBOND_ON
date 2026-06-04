import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_bid_ask_spread_v1")
class T1430BidAskSpreadV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(window: pd.DataFrame) -> float:
            if "ask_price1" not in window.columns or "bid_price1" not in window.columns:
                return np.nan
            ask_p1 = window["ask_price1"].iloc[-1]
            bid_p1 = window["bid_price1"].iloc[-1]
            
            if pd.isna(ask_p1) or pd.isna(bid_p1):
                return np.nan
                
            mid_price = (ask_p1 + bid_p1) / 2.0
            if mid_price <= 0:
                return np.nan
                
            spread = ask_p1 - bid_p1
            return spread / mid_price

        return _group_scalar(panel, _calc)