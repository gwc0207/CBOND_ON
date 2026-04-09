from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha050_volume_vwap_corr_max_v1")
class Alpha050VolumeVwapCorrMaxV1Factor(_AlphaBase):
    name = "alpha050_volume_vwap_corr_max_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window = int(ctx.params.get("corr_window", 5))
        ts_max_window = int(ctx.params.get("ts_max_window", 5))
        frame = _prepare_panel(ctx, ["amount", "volume"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            vwap = amount / (volume + EPS)
            rank_vol = volume.rank(pct=True, method="average")
            rank_vwap = vwap.rank(pct=True, method="average")
            corr = rank_vol.rolling(max(2, corr_window), min_periods=2).corr(rank_vwap)
            rank_corr = corr.rank(pct=True, method="average")
            ts_max = rank_corr.rolling(max(1, ts_max_window), min_periods=1).max().iloc[-1]
            if pd.isna(ts_max):
                return 0.0
            return float(-ts_max)

        return _group_scalar(frame, _calc)


