import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_order_imbalance_v1")
class T1430OrderImbalanceV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(window: pd.DataFrame) -> float:
            if "bid_volume1" not in window.columns or "ask_volume1" not in window.columns:
                return np.nan
                
            bid_v1 = window["bid_volume1"].iloc[-1]
            ask_v1 = window["ask_volume1"].iloc[-1]
            
            if pd.isna(bid_v1) or pd.isna(ask_v1):
                return np.nan
                
            total_vol = bid_v1 + ask_v1
            if total_vol <= 0:
                return np.nan
                
            return (bid_v1 - ask_v1) / total_vol

        return _group_scalar(panel, _calc)