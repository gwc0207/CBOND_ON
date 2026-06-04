import numpy as np
import pandas as pd
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window

@FactorRegistry.register("t1430_microprice_liquidity_v1")
class T1430MicropriceLiquidityV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        # Check required fields
        required = ["last", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1", "volume"]
        for f in required:
            if f not in panel.columns:
                raise KeyError(f"Missing field: {f}")
                
        def _calc(window):
            last = window["last"]
            bid_p = window["bid_price1"]
            ask_p = window["ask_price1"]
            bid_v = window["bid_volume1"]
            ask_v = window["ask_volume1"]
            vol = window["volume"]
            
            # Guard denominators
            bid_v = bid_v.replace(0, np.nan)
            ask_v = ask_v.replace(0, np.nan)
            vol = vol.replace(0, np.nan)
            
            # Microprice calculation
            microprice = (bid_p * ask_v + ask_p * bid_v) / (bid_v + ask_v)
            
            # Mid price
            mid = (bid_p + ask_p) / 2.0
            
            # Spread
            spread = ask_p - bid_p
            spread = spread.replace(0, np.nan)
            
            # Deviation normalized by spread
            dev_norm = (microprice - mid) / spread
            
            # Liquidity proxy: spread * volume (higher value = lower liquidity impact per unit volume? No, usually spread*vol is cost)
            # We want to scale by inverse liquidity. Let's use 1 / (spread * volume) as a weight for "fragility"
            liq_proxy = spread * vol
            liq_proxy = liq_proxy.replace(0, np.nan)
            
            # Factor: Deviation * Inverse Liquidity Proxy
            # High value means price is pushed away from mid in a illiquid state
            factor_val = dev_norm / liq_proxy
            
            return factor_val.iloc[-1] if not factor_val.empty else np.nan
            
        return _group_scalar(panel, _calc)