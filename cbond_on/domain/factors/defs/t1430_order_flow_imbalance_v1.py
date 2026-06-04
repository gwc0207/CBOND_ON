import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_order_flow_imbalance_v1")
class T1430OrderFlowImbalanceV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")

        def _calc(window):
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            
            total_v = bid_v + ask_v
            total_v = total_v.replace(0, np.nan)
            
            # Imbalance: (Ask - Bid) / Total
            # Positive means more Ask volume (Selling pressure)
            imbalance = (ask_v - bid_v) / total_v
            
            return imbalance.iloc[-1] if not imbalance.empty else np.nan

        return _group_scalar(panel, _calc)