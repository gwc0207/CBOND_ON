import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar

@FactorRegistry.register("ob_pressure_spread_v1")
class ObPressureSpreadV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            if 'ask_price1' not in group.columns or 'bid_price1' not in group.columns:
                raise KeyError("Missing OB fields")
            
            ap1 = group['ask_price1'].iloc[-1]
            av1 = group['ask_volume1'].iloc[-1]
            bp1 = group['bid_price1'].iloc[-1]
            bv1 = group['bid_volume1'].iloc[-1]
            
            # Guard invalid prices/volumes
            if ap1 <= 0 or bp1 <= 0 or av1 < 0 or bv1 < 0:
                return np.nan
            
            spread = ap1 - bp1
            if spread < 0:
                return np.nan # Invalid OB
            
            # Imbalance
            imb_num = bv1 - av1
            imb_denom = bv1 + av1 + 1e-8
            imbalance = imb_num / imb_denom
            
            # Spread inverse (liquidity cost)
            # Normalize spread by price level to make it comparable
            mid_price = (ap1 + bp1) / 2.0
            if mid_price <= 0:
                return np.nan
                
            rel_spread = spread / mid_price
            
            # Signal: Imbalance weighted by inverse relative spread
            # Tighter spread (lower rel_spread) -> higher weight
            signal = imbalance / (rel_spread + 1e-8)
            
            return signal

        return _group_scalar(panel, _calc)