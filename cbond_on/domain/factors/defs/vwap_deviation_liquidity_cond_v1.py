import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar

@FactorRegistry.register("vwap_deviation_liquidity_cond_v1")
class VwapDeviationLiquidityCondV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        def _calc(group):
            # Calculate intraday VWAP up to current point
            # Assuming panel is sorted by time within group
            valid = group.dropna(subset=['last', 'volume', 'amount'])
            if len(valid) == 0:
                return pd.NA
            
            total_amount = valid['amount'].sum()
            total_volume = valid['volume'].sum()
            
            if total_volume <= 0:
                return pd.NA
                
            vwap = total_amount / total_volume
            if vwap <= 0:
                return pd.NA
                
            last = valid['last'].iloc[-1]
            amount_total = valid['amount'].iloc[-1] # Cumulative amount if provided, else sum
            
            dev = (last - vwap) / vwap
            liq_scale = np.log(amount_total + 1)
            
            return dev * liq_scale

        return _group_scalar(panel, _calc)