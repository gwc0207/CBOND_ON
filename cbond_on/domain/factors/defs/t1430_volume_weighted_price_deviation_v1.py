import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_volume_weighted_price_deviation_v1")
class T1430VolumeWeightedPriceDeviationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ["last", "amount", "volume"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required field: {f}")
                
        def _calc(window):
            last = window["last"].astype(float)
            amount = window["amount"].astype(float)
            volume = window["volume"].astype(float)
            
            # Cumulative VWAP approximation
            cum_amount = amount.cumsum()
            cum_volume = volume.cumsum()
            
            cum_volume = cum_volume.replace(0, np.nan)
            vwap = cum_amount / cum_volume
            
            vwap = vwap.replace(0, np.nan)
            dev = (last - vwap) / vwap
            
            return dev.iloc[-1] if not dev.empty else np.nan
            
        result = _group_scalar(panel, _calc)
        return result