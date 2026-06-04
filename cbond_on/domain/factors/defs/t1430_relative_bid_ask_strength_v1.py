import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_relative_bid_ask_strength_v1")
class T1430RelativeBidAskStrengthV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["bid_price1", "ask_price1", "bid_volume1", "ask_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")

        def _calc(window):
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            
            # Calculate value on books
            bid_val = bid_p * bid_v
            ask_val = ask_p * ask_v
            
            # Guard zeros/negatives for log
            bid_val = bid_val.replace(0, np.nan)
            ask_val = ask_val.replace(0, np.nan)
            
            log_bid = np.log(bid_val)
            log_ask = np.log(ask_val)
            
            strength = log_bid - log_ask
            
            return strength.iloc[-1] if not strength.empty else np.nan

        return _group_scalar(panel, _calc)