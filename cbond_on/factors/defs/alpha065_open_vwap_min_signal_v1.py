from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha_group5_utils import mid_price1_series
from cbond_on.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha065_open_vwap_min_signal_v1")
class Alpha065OpenVwapMinSignalV1Factor(_AlphaBase):
    name = "alpha065_open_vwap_min_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        weight = float(ctx.params.get("weight", 0.008))
        adv_window = int(ctx.params.get("adv_window", 30))
        sum_window = int(ctx.params.get("sum_window", 9))
        corr_window = int(ctx.params.get("corr_window", 6))
        ts_min_window = int(ctx.params.get("ts_min_window", 14))
        frame = _prepare_panel(ctx, ["amount", "volume", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            mid_price1 = mid_price1_series(g)
            vwap = amount / (volume + EPS)

            weighted = mid_price1 * weight + vwap * (1.0 - weight)
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            sum_adv = adv.rolling(max(1, sum_window), min_periods=1).sum()
            corr = weighted.rolling(max(2, corr_window), min_periods=2).corr(sum_adv)
            rank_corr = corr.rank(pct=True, method="average")

            ts_min_open = mid_price1.rolling(max(1, ts_min_window), min_periods=1).min()
            rank_diff = (mid_price1 - ts_min_open).rank(pct=True, method="average")

            rc = rank_corr.iloc[-1] if len(rank_corr) else float("nan")
            rd = rank_diff.iloc[-1] if len(rank_diff) else float("nan")
            if pd.isna(rc) or pd.isna(rd):
                return 0.0
            return -1.0 if rc < rd else 0.0

        return _group_scalar(frame, _calc)

