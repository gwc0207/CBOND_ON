from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha075_vwap_volume_low_adv_corr_v1")
class Alpha075VwapVolumeLowAdvCorrV1Factor(_AlphaBase):
    name = "alpha075_vwap_volume_low_adv_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window_1 = int(ctx.params.get("corr_window_1", 4))
        adv_window = int(ctx.params.get("adv_window", 30))
        corr_window_2 = int(ctx.params.get("corr_window_2", 12))
        frame = _prepare_panel(ctx, ["amount", "volume", "low"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            low = g["low"].astype("float64")

            vwap = amount / (volume + EPS)
            corr1 = vwap.rolling(max(2, corr_window_1), min_periods=2).corr(volume)
            rank_corr1 = corr1.rank(pct=True, method="average")

            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            rank_low = low.rank(pct=True, method="average")
            rank_adv = adv.rank(pct=True, method="average")
            corr2 = rank_low.rolling(max(2, corr_window_2), min_periods=2).corr(rank_adv)
            rank_corr2 = corr2.rank(pct=True, method="average")

            r1 = rank_corr1.iloc[-1] if len(rank_corr1) else float("nan")
            r2 = rank_corr2.iloc[-1] if len(rank_corr2) else float("nan")
            if pd.isna(r1) or pd.isna(r2):
                return 0.0
            return 1.0 if r1 < r2 else 0.0

        return _group_scalar(frame, _calc)

