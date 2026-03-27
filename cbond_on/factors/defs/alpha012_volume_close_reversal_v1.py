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

@FactorRegistry.register("alpha012_volume_close_reversal_v1")
class Alpha012VolumeCloseReversalV1Factor(_AlphaBase):
    name = "alpha012_volume_close_reversal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        frame = _prepare_panel(ctx, ["volume", "last"])

        def _calc(g: pd.DataFrame) -> float:
            volume = g["volume"].astype("float64")
            last_px = g["last"].astype("float64")
            d_vol = _delta_last(volume, 1)
            d_last = _delta_last(last_px, 1)
            return float(np.sign(d_vol) * (-d_last))

        return _group_scalar(frame, _calc)

