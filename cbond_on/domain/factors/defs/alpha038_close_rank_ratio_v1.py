from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import EPS, _AlphaBase, _cs_rank, _group_scalar, _open_like, _prepare_panel, _ts_rank_last


@FactorRegistry.register("alpha038_close_rank_ratio_v1")
class Alpha038CloseRankRatioV1Factor(_AlphaBase):
    name = "alpha038_close_rank_ratio_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        ts_rank_window = int(ctx.params.get("ts_rank_window", 10))
        frame = _prepare_panel(ctx, ["last", "open", "ask_price1", "bid_price1"])

        def _rank_close(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            return float(_ts_rank_last(last_px, ts_rank_window))

        def _ratio(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            return float(last_px.iloc[-1] / (open_.iloc[-1] + EPS))

        rank_close = _group_scalar(frame, _rank_close)
        ratio = _group_scalar(frame, _ratio)
        return (-_cs_rank(rank_close)) * _cs_rank(ratio)



