from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha043_volume_delay_momentum_v1")
class Alpha043VolumeDelayMomentumV1Factor(_AlphaBase):
    name = "alpha043_volume_delay_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 10))
        ts_rank_window_1 = int(ctx.params.get("ts_rank_window_1", 10))
        delta_window = int(ctx.params.get("delta_window", 5))
        ts_rank_window_2 = int(ctx.params.get("ts_rank_window_2", 5))
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            vol_ratio = volume / (adv + EPS)
            ts_rank_vol = _ts_rank_last(vol_ratio, ts_rank_window_1)
            delta_close = last_px.diff(max(1, delta_window))
            ts_rank_delta = _ts_rank_last(-delta_close, ts_rank_window_2)
            return float(ts_rank_vol * ts_rank_delta)

        return _group_scalar(frame, _calc)

