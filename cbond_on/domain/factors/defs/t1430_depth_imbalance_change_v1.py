import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_depth_imbalance_change_v1")
class T1430DepthImbalanceChangeV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        required_fields = [
            "trade_time",
            "bid_volume1",
            "bid_volume2",
            "bid_volume3",
            "ask_volume1",
            "ask_volume2",
            "ask_volume3",
        ]
        for f in required_fields:
            if f not in panel.columns:
                raise KeyError(f"Missing required panel field: {f}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            window_minutes = int(ctx.params.get("window_minutes", 30))
            df = slice_window(df, window_minutes)

            if len(df) < 2:
                return np.nan

            bid_sum_start = (
                df.iloc[0]["bid_volume1"]
                + df.iloc[0]["bid_volume2"]
                + df.iloc[0]["bid_volume3"]
            )
            ask_sum_start = (
                df.iloc[0]["ask_volume1"]
                + df.iloc[0]["ask_volume2"]
                + df.iloc[0]["ask_volume3"]
            )

            bid_sum_end = (
                df.iloc[-1]["bid_volume1"]
                + df.iloc[-1]["bid_volume2"]
                + df.iloc[-1]["bid_volume3"]
            )
            ask_sum_end = (
                df.iloc[-1]["ask_volume1"]
                + df.iloc[-1]["ask_volume2"]
                + df.iloc[-1]["ask_volume3"]
            )

            denom_start = bid_sum_start + ask_sum_start
            denom_end = bid_sum_end + ask_sum_end

            if denom_start <= 0 or denom_end <= 0:
                return np.nan

            imb_start = (bid_sum_start - ask_sum_start) / denom_start
            imb_end = (bid_sum_end - ask_sum_end) / denom_end

            return imb_end - imb_start

        return _group_scalar(panel, _calc)