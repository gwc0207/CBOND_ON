import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar

@FactorRegistry.register("ob_pressure_spread_persistence_v1")
class ObPressureSpreadPersistenceV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required_fields = ["ask_price1", "bid_price1", "last", "ask_volume1", "bid_volume1"]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing panel field: {f}")

        def _calc(group):
            ap1 = group["ask_price1"].iloc[-1]
            bp1 = group["bid_price1"].iloc[-1]
            last = group["last"].iloc[-1]
            av1 = group["ask_volume1"].iloc[-1]
            bv1 = group["bid_volume1"].iloc[-1]
            
            if last <= 0:
                return np.nan
            
            spread = ap1 - bp1
            if spread < 0:
                return np.nan
                
            spread_rel = spread / last
            
            denom_vol = av1 + bv1
            if denom_vol <= 0:
                return np.nan
                
            imbalance = (av1 - bv1) / denom_vol
            
            # Interaction: Spread * Imbalance
            # If Ask Side is heavy (positive imbalance) and spread is wide, price might be suppressed?
            # Or if Bid Side is heavy (negative imbalance), price supported?
            # Let's assume positive factor -> positive return.
            # If Bid Heavy (imbalance < 0), we want positive return? Then factor should be negative? 
            # Usually Bid Heavy -> Price Up. So Imbalance < 0 -> Return > 0. 
            # So we need -Imbalance.
            
            return spread_rel * (-imbalance)

        return _group_scalar(panel, _calc)