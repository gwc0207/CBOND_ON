import pandas as pd
import numpy as np
from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, slice_window


@FactorRegistry.register("t1430_spread_change_30m_v1")
class T1430SpreadChange30mV1(Factor):
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

            if len(df) < 2:
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

            valid_df = df.loc[valid_mask].copy()
            if len(valid_df) < 2:
                return np.nan

            valid_ask = valid_df["ask_price1"].values
            valid_bid = valid_df["bid_price1"].values
            valid_last = valid_df["last"].values

            spreads = (valid_ask - valid_bid) / valid_last

            spread_start = spreads[0]
            spread_end = spreads[-1]

            return spread_end - spread_start

        return _group_scalar(panel, _calc)