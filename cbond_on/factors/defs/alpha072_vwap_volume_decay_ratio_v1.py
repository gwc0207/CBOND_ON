from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha_group5_utils import rolling_last_rank_pct, rolling_linear_decay
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha072_vwap_volume_decay_ratio_v1")
class Alpha072VwapVolumeDecayRatioV1Factor(_AlphaBase):
    name = "alpha072_vwap_volume_decay_ratio_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        corr_window_1 = int(ctx.params.get("corr_window_1", 9))
        decay_window_1 = int(ctx.params.get("decay_window_1", 10))
        ts_rank_window_1 = int(ctx.params.get("ts_rank_window_1", 4))
        ts_rank_window_2 = int(ctx.params.get("ts_rank_window_2", 19))
        corr_window_2 = int(ctx.params.get("corr_window_2", 7))
        decay_window_2 = int(ctx.params.get("decay_window_2", 3))
        frame = _prepare_panel(ctx, ["amount", "volume", "high", "low"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")

            vwap = amount / (volume + EPS)
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            mid_price = (high + low) / 2.0

            corr1 = mid_price.rolling(max(2, corr_window_1), min_periods=2).corr(adv)
            decay1 = rolling_linear_decay(corr1, decay_window_1)
            rank_decay1 = decay1.rank(pct=True, method="average")

            ts_rank_vwap = rolling_last_rank_pct(vwap, ts_rank_window_1)
            ts_rank_vol = rolling_last_rank_pct(volume, ts_rank_window_2)
            corr2 = ts_rank_vwap.rolling(max(2, corr_window_2), min_periods=2).corr(ts_rank_vol)
            decay2 = rolling_linear_decay(corr2, decay_window_2)
            rank_decay2 = decay2.rank(pct=True, method="average")

            r1 = rank_decay1.iloc[-1] if len(rank_decay1) else float("nan")
            r2 = rank_decay2.iloc[-1] if len(rank_decay2) else float("nan")
            if pd.isna(r1):
                r1 = 0.0
            if pd.isna(r2):
                r2 = 0.0
            return float(r1 / (r2 + EPS))

        return _group_scalar(frame, _calc)

