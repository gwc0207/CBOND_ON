from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import _AlphaBase, _corr_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha044_high_volume_rank_corr_v1")
class Alpha044HighVolumeRankCorrV1Factor(_AlphaBase):
    name = "alpha044_high_volume_rank_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 5))
        frame = _prepare_panel(ctx, ["high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            high = g["high"].astype("float64")
            volume = g["volume"].astype("float64")
            rank_vol = volume.rank(pct=True, method="average")
            return float(-_corr_last(high, rank_vol, corr_window))

        return _group_scalar(frame, _calc)

