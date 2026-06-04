import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_spread_volatility_ratio_v1")
class T1430SpreadVolatilityRatioV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required = ["high", "low", "bid_price1", "ask_price1", "pre_close"]
        for f in required:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(window):
            high = window["high"]
            low = window["low"]
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            pre_close = window["pre_close"]
            
            # Spread
            spread = ask_p - bid_p
            
            # Range
            range_val = high - low
            range_val = range_val.replace(0, np.nan)
            
            # Normalize spread by pre_close to make it comparable across bonds? 
            # Or just spread/range. Let's do spread / range.
            # If range is 0 (flat day), this is NaN/Inf. Guarded by replace(0, nan).
            
            factor_val = spread / range_val
            
            return factor_val.iloc[-1] if not factor_val.empty else np.nan
            
        return _group_scalar(panel, _calc)