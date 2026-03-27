from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _delta_last, _group_scalar, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha039_volume_decay_momentum_v1")
class Alpha039VolumeDecayMomentumV1Factor(_AlphaBase):
    name = "alpha039_volume_decay_momentum_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 10))
        decay_window = int(ctx.params.get("decay_window", 9))
        delta_window = int(ctx.params.get("delta_window", 7))
        sum_window = int(ctx.params.get("sum_window", 60))
        frame = _prepare_panel(ctx, ["amount", "volume", "last", "pre_close"])

        def _term(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            vol_ratio = volume / (adv + EPS)
            vol_decay_rank = _ts_rank_last(vol_ratio, decay_window)
            delta_close = _delta_last(last_px, delta_window)
            return float(delta_close * (1.0 - vol_decay_rank))

        def _sum_ret(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            return float(returns.rolling(max(1, sum_window), min_periods=1).sum().iloc[-1])

        term = _group_scalar(frame, _term)
        sum_ret = _group_scalar(frame, _sum_ret)
        return (-_cs_rank(term)) * (1.0 + _cs_rank(sum_ret + 1.0))


