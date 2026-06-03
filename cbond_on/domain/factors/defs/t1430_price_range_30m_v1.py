import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_price_range_30m_v1")
class T1430PriceRange30mV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        
        # Check required fields exist
        if 'trade_time' not in panel.columns or 'last' not in panel.columns:
            raise KeyError("Required columns 'trade_time' and 'last' not found in panel")

        def _calc(df: pd.DataFrame) -> float:
            # Sort by trade time to ensure correct window slicing
            df = df.sort_values('trade_time')
            
            # Slice the last 30 minutes of data
            window_minutes = int(ctx.params.get('window_minutes', 30))
            df_window = slice_window(df, window_minutes)
            
            if df_window.empty:
                return np.nan
                
            last_values = df_window['last'].values
            
            # Guard against invalid last values (NaN or <= 0)
            valid_last = last_values[~np.isnan(last_values) & (last_values > 0)]
            if len(valid_last) == 0:
                return np.nan
                
            max_last = np.max(valid_last)
            min_last = np.min(valid_last)
            last_end = valid_last[-1] # Last available price in the window
            
            # Guard against division by zero or invalid denominator
            if last_end <= 0:
                return np.nan
                
            return (max_last - min_last) / last_end

        return _group_scalar(panel, _calc)