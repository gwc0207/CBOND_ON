import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_ob_imbalance_flow_v1")
class T1430ObImbalanceFlowV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1", "volume", "amount"]
        for f in required:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(window):
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            vol = window["volume"]
            amt = window["amount"]
            
            # Guard denominators
            total_ob_vol = bid_v + ask_v
            total_ob_vol = total_ob_vol.replace(0, np.nan)
            vol = vol.replace(0, np.nan)
            
            # OB Imbalance: (Bid - Ask) / (Bid + Ask)
            ob_imb = (bid_v - ask_v) / total_ob_vol
            
            # Trade Flow Intensity: Amount / Volume (Avg Price) normalized by current price?
            # Or just Volume relative to OB depth?
            # Let's use Volume / Total OB Volume as a measure of "throughput" vs "stock"
            throughput = vol / total_ob_vol
            
            # Interaction
            factor_val = ob_imb * throughput
            
            return factor_val.iloc[-1] if not factor_val.empty else np.nan
            
        return _group_scalar(panel, _calc)