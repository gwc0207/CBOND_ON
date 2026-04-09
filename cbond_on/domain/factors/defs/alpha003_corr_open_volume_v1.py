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

@FactorRegistry.register("alpha003_corr_open_volume_v1")
class Alpha003CorrOpenVolumeV1Factor(_AlphaBase):
    name = "alpha003_corr_open_volume_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["open", "ask_price1", "bid_price1", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            open_rank = _open_like(g).rank(pct=True, method="average")
            vol_rank = g["volume"].astype("float64").rank(pct=True, method="average")
            return -_corr_last(open_rank, vol_rank, corr_window)

        return _group_scalar(
            frame,
            _calc,
            kernel_name="alpha003_corr_open_volume_v1",
            kernel_params={"corr_window": corr_window},
        )


