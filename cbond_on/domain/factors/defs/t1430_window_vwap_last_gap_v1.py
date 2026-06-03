import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


def _calc(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    
    # Sort by trade_time to ensure correct slicing
    df_sorted = df.sort_values('trade_time')
    
    # Slice the last 30 minutes
    df_window = slice_window(df_sorted, 30)
    
    if df_window.empty:
        return np.nan
    
    # Calculate sums for VWAP
    amount_sum = df_window['amount'].sum()
    volume_sum = df_window['volume'].sum()
    
    # Get the last price from the latest snapshot in the window (which is the end of the 30 min window)
    # Since df_window is sorted, the last row has the latest trade_time within the window
    last_price = df_window['last'].iloc[-1]
    
    # Guard against division by zero or invalid prices
    if volume_sum <= 0 or last_price <= 0:
        return np.nan
    
    window_vwap = amount_sum / volume_sum
    
    return window_vwap / last_price - 1.0


@FactorRegistry.register("t1430_window_vwap_last_gap_v1")
class T1430WindowVwapLastGapV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        # Ensure trade_time is available and processed
        panel = ensure_trade_time(ctx.panel)
        
        # Check required fields exist
        required_fields = ['trade_time', 'last', 'amount', 'volume']
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Required field {field} not found in panel")
                
        # Apply calculation per group (dt, code)
        result = _group_scalar(panel, _calc)
        return result