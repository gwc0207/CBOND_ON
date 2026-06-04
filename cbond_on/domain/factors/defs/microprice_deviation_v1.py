import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("microprice_deviation_v1")
class MicropriceDeviationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        required_fields = ['last', 'ask_price1', 'ask_volume1', 'bid_price1', 'bid_volume1']
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(group):
            # Use the last snapshot in the group (which is T1430 slice)
            row = group.iloc[-1]
            
            ask_p = row['ask_price1']
            ask_v = row['ask_volume1']
            bid_p = row['bid_price1']
            bid_v = row['bid_volume1']
            last_p = row['last']
            
            if ask_p <= 0 or bid_p <= 0 or (ask_v + bid_v) == 0:
                return np.nan
                
            # Microprice formula: weighted mid
            microprice = (ask_p * bid_v + bid_p * ask_v) / (ask_v + bid_v)
            
            if microprice <= 0:
                return np.nan
                
            return (last_p - microprice) / microprice
            
        return _group_scalar(panel, _calc)