from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._alpha_group5_utils import rolling_linear_decay
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha077_mid_price_adv_decay_min_v1")
class Alpha077MidPriceAdvDecayMinV1Factor(_AlphaBase):
    name = "alpha077_mid_price_adv_decay_min_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        decay_window_1 = int(ctx.params.get("decay_window_1", 10))
        adv_window = int(ctx.params.get("adv_window", 20))
        corr_window = int(ctx.params.get("corr_window", 3))
        decay_window_2 = int(ctx.params.get("decay_window_2", 6))
        frame = _prepare_panel(ctx, ["amount", "volume", "high", "low"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")

            vwap = amount / (volume + EPS)
            mid_price = (high + low) / 2.0
            diff1 = mid_price - vwap
            decay1 = rolling_linear_decay(diff1, decay_window_1)
            rank_decay1 = decay1.rank(pct=True, method="average")

            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            corr = mid_price.rolling(max(2, corr_window), min_periods=2).corr(adv)
            decay2 = rolling_linear_decay(corr, decay_window_2)
            rank_decay2 = decay2.rank(pct=True, method="average")

            r1 = rank_decay1.iloc[-1] if len(rank_decay1) else float("nan")
            r2 = rank_decay2.iloc[-1] if len(rank_decay2) else float("nan")
            if pd.isna(r1):
                r1 = 0.0
            if pd.isna(r2):
                r2 = 0.0
            return float(min(r1, r2))

        return _group_scalar(frame, _calc)

