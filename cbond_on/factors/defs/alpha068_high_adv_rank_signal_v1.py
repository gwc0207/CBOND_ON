from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._alpha_group5_utils import rolling_last_rank_pct
from cbond_on.factors.defs._intraday_utils import _AlphaBase, _group_scalar, _prepare_panel


@FactorRegistry.register("alpha068_high_adv_rank_signal_v1")
class Alpha068HighAdvRankSignalV1Factor(_AlphaBase):
    name = "alpha068_high_adv_rank_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        adv_window = int(ctx.params.get("adv_window", 15))
        corr_window = int(ctx.params.get("corr_window", 9))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 14))
        weight = float(ctx.params.get("weight", 0.518))
        delta_window = int(ctx.params.get("delta_window", 1))
        frame = _prepare_panel(ctx, ["amount", "high", "last", "low"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            high = g["high"].astype("float64")
            last_px = g["last"].astype("float64")
            low = g["low"].astype("float64")

            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            corr = high.rank(pct=True, method="average").rolling(max(2, corr_window), min_periods=2).corr(
                adv.rank(pct=True, method="average")
            )
            ts_rank_corr = rolling_last_rank_pct(corr, ts_rank_window)
            rank_ts = ts_rank_corr.rank(pct=True, method="average")

            weighted = last_px * weight + low * (1.0 - weight)
            delta_weighted = weighted.diff(max(1, delta_window))
            rank_delta = delta_weighted.rank(pct=True, method="average")

            rts = rank_ts.iloc[-1] if len(rank_ts) else float("nan")
            rd = rank_delta.iloc[-1] if len(rank_delta) else float("nan")
            if pd.isna(rts) or pd.isna(rd):
                return 0.0
            return -1.0 if rts < rd else 0.0

        return _group_scalar(frame, _calc)
