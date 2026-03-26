from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha035_volume_price_momentum_v1")
class Alpha035VolumePriceMomentumV1Factor(_AlphaBase):
    name = "alpha035_volume_price_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window_long = int(ctx.params.get("ts_rank_window_long", 20))
        ts_rank_window_short = int(ctx.params.get("ts_rank_window_short", 16))
        frame = _prepare_panel(ctx, ["volume", "last", "high", "low", "pre_close"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            ts_rank_vol = _ts_rank_last(volume, ts_rank_window_long)
            price_range = (last_px + high) - low
            ts_rank_price = 1.0 - _ts_rank_last(price_range, ts_rank_window_short)
            returns = (last_px - pre_close) / (pre_close + EPS)
            ts_rank_ret = 1.0 - _ts_rank_last(returns, ts_rank_window_long)
            return float(ts_rank_vol * ts_rank_price * ts_rank_ret)

        return _group_scalar(frame, _calc)

