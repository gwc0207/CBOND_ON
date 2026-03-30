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

@FactorRegistry.register("alpha002_corr_volume_return_v1")
class Alpha002CorrVolumeReturnV1Factor(_AlphaBase):
    name = "alpha002_corr_volume_return_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 6))
        frame = _prepare_panel(ctx, ["volume", "last", "open", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64").clip(lower=0.0)
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            log_volume = np.log(volume + EPS)
            delta_log_vol = log_volume.diff(2)
            ret = (last_px - open_) / (open_ + EPS)
            x = delta_log_vol.rank(pct=True, method="average")
            y = ret.rank(pct=True, method="average")
            return -_corr_last(x, y, corr_window)

        return _group_scalar(
            frame,
            _calc,
            kernel_name="alpha002_corr_volume_return_v1",
            kernel_params={"corr_window": corr_window},
        )

