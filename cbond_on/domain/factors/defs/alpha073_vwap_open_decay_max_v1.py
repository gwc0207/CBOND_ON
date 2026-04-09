from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._alpha_group5_utils import mid_price1_series, rolling_linear_decay
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _group_scalar, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha073_vwap_open_decay_max_v1")
class Alpha073VwapOpenDecayMaxV1Factor(_AlphaBase):
    name = "alpha073_vwap_open_decay_max_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        delta_window_1 = int(ctx.params.get("delta_window_1", 5))
        decay_window_1 = int(ctx.params.get("decay_window_1", 3))
        weight = float(ctx.params.get("weight", 0.147))
        delta_window_2 = int(ctx.params.get("delta_window_2", 2))
        decay_window_2 = int(ctx.params.get("decay_window_2", 3))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 17))
        frame = _prepare_panel(ctx, ["amount", "volume", "low", "ask_price1", "bid_price1"])

        def _calc(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            low = g["low"].astype("float64")
            mid_price1 = mid_price1_series(g)

            vwap = amount / (volume + EPS)
            delta_vwap = vwap.diff(max(1, delta_window_1))
            decay1 = rolling_linear_decay(delta_vwap, decay_window_1)
            rank_decay1 = decay1.rank(pct=True, method="average")
            rank_decay1_last = rank_decay1.iloc[-1] if len(rank_decay1) else float("nan")

            weighted = mid_price1 * weight + low * (1.0 - weight)
            delta_weighted = weighted.diff(max(1, delta_window_2))
            diff = delta_weighted / (weighted + EPS)
            decay2 = rolling_linear_decay(-diff, decay_window_2)
            ts_rank = _ts_rank_last(decay2, ts_rank_window)

            if pd.isna(rank_decay1_last):
                rank_decay1_last = 0.0
            if pd.isna(ts_rank):
                ts_rank = 0.0
            return float(-max(rank_decay1_last, ts_rank))

        return _group_scalar(frame, _calc)

