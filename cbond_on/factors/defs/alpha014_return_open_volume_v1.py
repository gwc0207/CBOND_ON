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

@FactorRegistry.register("alpha014_return_open_volume_v1")
class Alpha014ReturnOpenVolumeV1Factor(_AlphaBase):
    name = "alpha014_return_open_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window = int(ctx.params.get("delta_window", 3))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(
            ctx,
            ["last", "pre_close", "open", "ask_price1", "bid_price1", "volume"],
        )

        def _delta_ret(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            return _delta_last(returns, delta_window)

        def _corr_ov(g: pd.DataFrame) -> float:
            open_ = _open_like(g)
            volume = g["volume"].astype("float64")
            return _corr_last(open_, volume, corr_window)

        delta_ret = _group_scalar(frame, _delta_ret)
        corr_ov = _group_scalar(frame, _corr_ov)
        return (-_cs_rank(delta_ret)) * corr_ov

