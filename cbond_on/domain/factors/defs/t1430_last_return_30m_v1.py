import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_last_return_30m_v1")
class T1430LastReturn30mV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        if "trade_time" not in panel.columns:
            raise KeyError("Missing required panel field: trade_time")
        if "last" not in panel.columns:
            raise KeyError("Missing required panel field: last")

        window_minutes = int(ctx.params.get("window_minutes", 30))

        def _calc(df: pd.DataFrame) -> float:
            if df.empty:
                return np.nan
            
            df_sorted = df.sort_values("trade_time")
            df_window = slice_window(df_sorted, window_minutes)
            
            if df_window.empty:
                return np.nan
                
            start_price = df_window["last"].iloc[0]
            end_price = df_window["last"].iloc[-1]
            
            if start_price <= 0 or end_price <= 0:
                return np.nan
                
            return end_price / start_price - 1.0

        return _group_scalar(panel, _calc)