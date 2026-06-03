import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_depth_concentration_v1")
class T1430DepthConcentrationV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        required_fields = [
            "trade_time",
            "bid_volume1", "bid_volume2", "bid_volume3", "bid_volume4", "bid_volume5",
            "ask_volume1", "ask_volume2", "ask_volume3", "ask_volume4", "ask_volume5"
        ]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required panel field: {field}")

        def _calc(df: pd.DataFrame) -> float:
            if df.empty:
                return np.nan
            
            # Sort by trade_time and take the last snapshot within the window
            df_sorted = df.sort_values("trade_time")
            window_minutes = int(ctx.params.get("window_minutes", 30))
            df_window = slice_window(df_sorted, window_minutes)
            
            if df_window.empty:
                return np.nan
            
            # Get the latest snapshot
            last_snapshot = df_window.iloc[-1]
            
            bid_v1 = last_snapshot["bid_volume1"]
            bid_v2 = last_snapshot["bid_volume2"]
            bid_v3 = last_snapshot["bid_volume3"]
            bid_v4 = last_snapshot["bid_volume4"]
            bid_v5 = last_snapshot["bid_volume5"]
            
            ask_v1 = last_snapshot["ask_volume1"]
            ask_v2 = last_snapshot["ask_volume2"]
            ask_v3 = last_snapshot["ask_volume3"]
            ask_v4 = last_snapshot["ask_volume4"]
            ask_v5 = last_snapshot["ask_volume5"]
            
            top1_depth = bid_v1 + ask_v1
            total5_depth = (
                bid_v1 + bid_v2 + bid_v3 + bid_v4 + bid_v5 +
                ask_v1 + ask_v2 + ask_v3 + ask_v4 + ask_v5
            )
            
            if total5_depth <= 0:
                return np.nan
                
            return top1_depth / total5_depth

        return _group_scalar(panel, _calc)