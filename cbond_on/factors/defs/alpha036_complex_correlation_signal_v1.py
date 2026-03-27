from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.factors.base import FactorComputeContext
from cbond_on.factors.defs._intraday_utils import (
    EPS,
    _AlphaBase,
    _corr_last,
    _cs_rank,
    _group_scalar,
    _open_like,
    _prepare_panel,
    _ts_rank_last,
)


@FactorRegistry.register("alpha036_complex_correlation_signal_v1")
class Alpha036ComplexCorrelationSignalV1Factor(_AlphaBase):
    name = "alpha036_complex_correlation_signal_v1"

    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        corr_window_1 = int(ctx.params.get("corr_window_1", 15))
        corr_window_2 = int(ctx.params.get("corr_window_2", 6))
        sum_window = int(ctx.params.get("sum_window", 60))
        ts_rank_window = int(ctx.params.get("ts_rank_window", 5))
        delay_window = int(ctx.params.get("delay_window", 6))
        adv_window = int(ctx.params.get("adv_window", 10))
        frame = _prepare_panel(
            ctx,
            ["last", "open", "ask_price1", "bid_price1", "volume", "pre_close", "amount"],
        )

        def _term1(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            volume = g["volume"].astype("float64")
            diff1 = last_px - open_
            delay_vol = volume.shift(1)
            return float(_corr_last(diff1, delay_vol, corr_window_1))

        def _term2(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            return float((open_ - last_px).iloc[-1])

        def _term3(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            pre_close = g["pre_close"].astype("float64")
            returns = (last_px - pre_close) / (pre_close + EPS)
            delay_ret = (-returns).shift(max(1, delay_window))
            return float(_ts_rank_last(delay_ret, ts_rank_window))

        def _term4(g: pd.DataFrame) -> float:
            amount = g["amount"].astype("float64")
            volume = g["volume"].astype("float64")
            vwap = amount / (volume + EPS)
            adv = amount.rolling(max(1, adv_window), min_periods=1).mean()
            return float(abs(_corr_last(vwap, adv, corr_window_2)))

        def _term5(g: pd.DataFrame) -> float:
            last_px = g["last"].astype("float64")
            open_ = _open_like(g)
            avg_close = float(last_px.rolling(max(1, sum_window), min_periods=1).mean().iloc[-1])
            return float((avg_close - open_.iloc[-1]) * (last_px.iloc[-1] - open_.iloc[-1]))

        t1 = _group_scalar(frame, _term1)
        t2 = _group_scalar(frame, _term2)
        t3 = _group_scalar(frame, _term3)
        t4 = _group_scalar(frame, _term4)
        t5 = _group_scalar(frame, _term5)
        return 2.21 * _cs_rank(t1) + 0.70 * _cs_rank(t2) + 0.73 * _cs_rank(t3) + _cs_rank(t4) + 0.60 * _cs_rank(t5)


