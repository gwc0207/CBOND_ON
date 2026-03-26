from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha101_utils import (
    EPS,
    _AlphaBase,
    _corr_last,
    _cov_last,
    _cs_rank,
    _delay_last,
    _delta_last,
    _group_scalar,
    _open_like,
    _prepare_panel,
    _ts_rank_last,
)

@FactorRegistry.register("alpha017_close_rank_volume_v1")
class Alpha017CloseRankVolumeV1Factor(_AlphaBase):
    name = "alpha017_close_rank_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        ts_rank_close_window = int(ctx.params.get("ts_rank_close_window", 10))
        ts_rank_vol_window = int(ctx.params.get("ts_rank_vol_window", 5))
        frame = _prepare_panel(ctx, ["last", "amount", "volume"])

        def _term1(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return _ts_rank_last(last_px, ts_rank_close_window)

        def _term2(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            delta2 = last_px.diff(1).diff(1)
            return float(delta2.iloc[-1]) if pd.notna(delta2.iloc[-1]) else 0.0

        def _term3(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            ratio = volume / (adv + EPS)
            return _ts_rank_last(ratio, ts_rank_vol_window)

        t1 = _group_scalar(frame, _term1)
        t2 = _group_scalar(frame, _term2)
        t3 = _group_scalar(frame, _term3)
        return (-_cs_rank(t1)) * _cs_rank(t2) * _cs_rank(t3)
