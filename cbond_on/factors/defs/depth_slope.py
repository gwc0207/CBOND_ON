from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time, _group_scalar


@FactorRegistry.register("depth_slope")
class DepthSlopeFactor(Factor):
    name = "depth_slope"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        levels = int(ctx.params.get("levels", 5))
        bid_price_cols = [f"bid_price{i}" for i in range(1, levels + 1)]
        ask_price_cols = [f"ask_price{i}" for i in range(1, levels + 1)]
        bid_vol_cols = [f"bid_volume{i}" for i in range(1, levels + 1)]
        ask_vol_cols = [f"ask_volume{i}" for i in range(1, levels + 1)]
        required = bid_price_cols + ask_price_cols + bid_vol_cols + ask_vol_cols
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise KeyError(f"depth_slope missing columns: {missing}")

        def _book_slope(prices: np.ndarray, vols: np.ndarray) -> float:
            if len(prices) < 2:
                return 0.0
            dp = np.abs(np.diff(prices))
            dv = (vols[:-1] + vols[1:]) / 2.0
            weight = float(np.sum(dv))
            if weight <= 0:
                return 0.0
            return float(np.sum(dp * dv) / weight)

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            row = df.iloc[-1]
            bid_prices = row[bid_price_cols].astype("float64").to_numpy()
            ask_prices = row[ask_price_cols].astype("float64").to_numpy()
            bid_vols = row[bid_vol_cols].astype("float64").to_numpy()
            ask_vols = row[ask_vol_cols].astype("float64").to_numpy()
            bid_slope = _book_slope(bid_prices, bid_vols)
            ask_slope = _book_slope(ask_prices, ask_vols)
            mid = (float(row["ask_price1"]) + float(row["bid_price1"])) / 2.0
            if mid <= 0:
                return 0.0
            return float((ask_slope - bid_slope) / mid)

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

