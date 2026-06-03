import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_spread_mean_guarded_30m_v1")
class T1430SpreadMeanGuarded30mV1(Factor):
    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)

        required_fields = ["trade_time", "ask_price1", "bid_price1", "last"]
        for field in required_fields:
            if field not in panel.columns:
                raise KeyError(f"Missing required panel field: {field}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            window_minutes = int(ctx.params.get("window_minutes", 30))
            df = slice_window(df, window_minutes)

            if df.empty:
                return np.nan

            ask = df["ask_price1"].values
            bid = df["bid_price1"].values
            last = df["last"].values

            valid_mask = (
                (ask > 0)
                & (bid > 0)
                & (last > 0)
                & (ask >= bid)
            )

            valid_ask = ask[valid_mask]
            valid_bid = bid[valid_mask]
            valid_last = last[valid_mask]

            if len(valid_last) == 0:
                return np.nan

            spreads = (valid_ask - valid_bid) / valid_last
            return float(np.mean(spreads))

        return _group_scalar(panel, _calc)