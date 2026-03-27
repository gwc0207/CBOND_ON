from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import (
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

@FactorRegistry.register("alpha007_volume_breakout_v1")
class Alpha007VolumeBreakoutV1Factor(_AlphaBase):
    name = "alpha007_volume_breakout_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        delta_window = int(ctx.params.get("delta_window", 7))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 60))
        frame = _prepare_panel(ctx, ["amount", "volume", "last"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            delta_last = last_px.diff(max(1, delta_window))
            if float(adv.iloc[-1]) >= float(volume.iloc[-1]):
                return -1.0
            sign_delta = float(np.sign(delta_last.iloc[-1])) if pd.notna(delta_last.iloc[-1]) else 0.0
            ts_rank_abs = _ts_rank_last(delta_last.abs(), ts_rank_window)
            return float((-1.0 * ts_rank_abs) * sign_delta)

        return _group_scalar(frame, _calc)

