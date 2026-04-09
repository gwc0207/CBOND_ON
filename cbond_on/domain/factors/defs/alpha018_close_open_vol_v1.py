from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import (
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

@FactorRegistry.register("alpha018_close_open_vol_v1")
class Alpha018CloseOpenVolV1Factor(_AlphaBase):
    name = "alpha018_close_open_vol_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        stddev_window = int(ctx.params.get("stddev_window", 5))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            diff = last_px - open_
            std_diff = diff.abs().rolling(max(2, stddev_window), min_periods=2).std().fillna(0.0)
            corr_co = _corr_last(last_px, open_, corr_window)
            return float(std_diff.iloc[-1] + diff.iloc[-1] + corr_co)

        raw = _group_scalar(frame, _calc)
        return -_cs_rank(raw)


