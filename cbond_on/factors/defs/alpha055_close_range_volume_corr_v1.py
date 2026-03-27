from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _corr_last, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha055_close_range_volume_corr_v1")
class Alpha055CloseRangeVolumeCorrV1Factor(_AlphaBase):
    name = "alpha055_close_range_volume_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_window = int(ctx.params.get("ts_window", 12))
        corr_window = int(ctx.params.get("corr_window", 6))
        frame = _prepare_panel(ctx, ["last", "low", "high", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            low = g["low"].astype("float64")
            high = g["high"].astype("float64")
            volume = g["volume"].astype("float64")
            ts_min_low = low.rolling(max(1, ts_window), min_periods=1).min()
            ts_max_high = high.rolling(max(1, ts_window), min_periods=1).max()
            range_pos = (last_px - ts_min_low) / (ts_max_high - ts_min_low + EPS)
            rank_range = range_pos.rank(pct=True, method="average")
            rank_volume = volume.rank(pct=True, method="average")
            return float(-_corr_last(rank_range, rank_volume, corr_window))

        return _group_scalar(frame, _calc)

