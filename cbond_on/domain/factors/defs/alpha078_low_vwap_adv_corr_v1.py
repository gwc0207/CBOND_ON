from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha078_low_vwap_adv_corr_v1")
class Alpha078LowVwapAdvCorrV1Factor(_AlphaBase):
    name = "alpha078_low_vwap_adv_corr_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        weight = float(ctx.params.get("weight", 0.352))
        sum_window_1 = int(ctx.params.get("sum_window_1", 20))
        adv_window = int(ctx.params.get("adv_window", 20))
        sum_window_2 = int(ctx.params.get("sum_window_2", 20))
        corr_window_1 = int(ctx.params.get("corr_window_1", 7))
        corr_window_2 = int(ctx.params.get("corr_window_2", 6))
        frame = _prepare_panel(ctx, ["amount", "volume", "low"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            low = g["low"].astype("float64")

            vwap = amount / (volume + EPS)
            weighted = low * weight + vwap * (1.0 - weight)
            sum_weighted = weighted.rolling(max(1, sum_window_1), min_periods=1).sum()

            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            sum_adv = adv.rolling(max(1, sum_window_2), min_periods=1).sum()
            corr1 = sum_weighted.rolling(max(2, corr_window_1), min_periods=2).corr(sum_adv)
            rank_corr1 = corr1.rank(pct=True, method="average")

            rank_vwap = vwap.rank(pct=True, method="average")
            rank_vol = volume.rank(pct=True, method="average")
            corr2 = rank_vwap.rolling(max(2, corr_window_2), min_periods=2).corr(rank_vol)
            rank_corr2 = corr2.rank(pct=True, method="average")

            r1 = rank_corr1.iloc[-1] if len(rank_corr1) else float("nan")
            r2 = rank_corr2.iloc[-1] if len(rank_corr2) else float("nan")
            if pd.isna(r1):
                r1 = 0.0
            if pd.isna(r2):
                r2 = 0.0
            return float(r1 ** r2)

        return _group_scalar(frame, _calc)

