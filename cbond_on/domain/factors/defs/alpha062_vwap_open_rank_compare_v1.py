from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._alpha_group5_utils import mid_price1_series
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha062_vwap_open_rank_compare_v1")
class Alpha062VwapOpenRankCompareV1Factor(_AlphaBase):
    name = "alpha062_vwap_open_rank_compare_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 20))
        sum_window = int(ctx.params.get("sum_window", 22))
        corr_window = int(ctx.params.get("corr_window", 10))
        frame = _prepare_panel(ctx, ["amount", "volume", "high", "low", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            high = g["high"].astype("float64")
            low = g["low"].astype("float64")
            mid_price1 = mid_price1_series(g)

            vwap = amount / (volume + EPS)
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            sum_adv = adv.rolling(max(1, sum_window), min_periods=1).sum()
            corr = vwap.rolling(max(2, corr_window), min_periods=2).corr(sum_adv)
            rank_corr = corr.rank(pct=True, method="average")

            rank_open_sum = mid_price1.rank(pct=True, method="average") * 2.0
            mid_price = (high + low) / 2.0
            rank_mid = mid_price.rank(pct=True, method="average") + high.rank(pct=True, method="average")
            compare = (rank_open_sum < rank_mid).astype("float64")
            rank_compare = compare.rank(pct=True, method="average")

            rc = rank_corr.iloc[-1] if len(rank_corr) else float("nan")
            rp = rank_compare.iloc[-1] if len(rank_compare) else float("nan")
            if pd.isna(rc) or pd.isna(rp):
                return 0.0
            return -1.0 if rc < rp else 0.0

        return _group_scalar(frame, _calc)


