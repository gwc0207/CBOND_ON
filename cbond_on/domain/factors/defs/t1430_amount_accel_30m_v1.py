import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_amount_accel_30m_v1")
class T1430AmountAccel30mV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        if 'trade_time' not in panel.columns or 'amount' not in panel.columns:
            raise KeyError("Required columns 'trade_time' and 'amount' are missing from panel.")

        def _calc(df: pd.DataFrame) -> float:
            # Sort by trade_time to ensure chronological order
            df_sorted = df.sort_values('trade_time')
            
            # Slice the last window_minutes
            window_minutes = int(ctx.params.get('window_minutes', 30))
            df_window = slice_window(df_sorted, window_minutes)
            
            if len(df_window) < 2:
                return np.nan
            
            # Split into first half and second half by row order
            mid_idx = len(df_window) // 2
            if mid_idx == 0:
                return np.nan
                
            first_half = df_window.iloc[:mid_idx]
            second_half = df_window.iloc[mid_idx:]
            
            sum_first = first_half['amount'].sum()
            sum_second = second_half['amount'].sum()
            
            denominator = sum_second + sum_first
            
            # Guard against division by zero or invalid denominators
            if denominator <= 0:
                return np.nan
                
            return (sum_second - sum_first) / denominator

        return _group_scalar(panel, _calc)