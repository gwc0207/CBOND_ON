import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_mid_return_30m_v1")
class T1430MidReturn30mV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        required_fields = ['trade_time', 'ask_price1', 'bid_price1']
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required panel field: {field}")

        window_minutes = int(ctx.params.get('window_minutes', 30))

        def _calc(df: pd.DataFrame) -> float:
            if df.empty:
                return np.nan
            
            df_sorted = df.sort_values('trade_time')
            df_window = slice_window(df_sorted, window_minutes)
            
            if df_window.empty:
                return np.nan
            
            ask_start = df_window.iloc[0]['ask_price1']
            bid_start = df_window.iloc[0]['bid_price1']
            ask_end = df_window.iloc[-1]['ask_price1']
            bid_end = df_window.iloc[-1]['bid_price1']
            
            if ask_start <= 0 or bid_start <= 0 or ask_end <= 0 or bid_end <= 0:
                return np.nan
            
            mid_start = (ask_start + bid_start) / 2.0
            mid_end = (ask_end + bid_end) / 2.0
            
            if mid_start <= 0 or mid_end <= 0:
                return np.nan
            
            return mid_end / mid_start - 1.0

        return _group_scalar(panel, _calc)