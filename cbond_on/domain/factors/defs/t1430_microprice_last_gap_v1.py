import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_microprice_last_gap_v1")
class T1430MicropriceLastGapV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        required_fields = [
            "trade_time",
            "ask_price1",
            "bid_price1",
            "ask_volume1",
            "bid_volume1",
            "last",
        ]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required panel field: {field}")

        def _calc(df: pd.DataFrame) -> float:
            if df.empty:
                return np.nan

            df_sorted = df.sort_values("trade_time")
            window_minutes = int(ctx.params.get("window_minutes", 30))
            df_window = slice_window(df_sorted, window_minutes)

            if df_window.empty:
                return np.nan

            last_row = df_window.iloc[-1]

            ask_price1 = last_row["ask_price1"]
            bid_price1 = last_row["bid_price1"]
            ask_volume1 = last_row["ask_volume1"]
            bid_volume1 = last_row["bid_volume1"]
            last_price = last_row["last"]

            if (
                pd.isna(ask_price1)
                or pd.isna(bid_price1)
                or pd.isna(ask_volume1)
                or pd.isna(bid_volume1)
                or pd.isna(last_price)
            ):
                return np.nan

            if ask_volume1 <= 0 or bid_volume1 <= 0:
                return np.nan

            if last_price <= 0:
                return np.nan

            denominator = bid_volume1 + ask_volume1
            if denominator <= 0:
                return np.nan

            microprice = (
                ask_price1 * bid_volume1 + bid_price1 * ask_volume1
            ) / denominator

            return microprice / last_price - 1.0

        return _group_scalar(panel, _calc)