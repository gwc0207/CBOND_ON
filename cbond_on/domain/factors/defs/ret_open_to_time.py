from __future__ import annotations

import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _group_scalar, parse_hhmm, first_last_price


@FactorRegistry.register("ret_open_to_time")
class ReturnOpenToTimeFactor(Factor):
    name = "ret_open_to_time"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        price_col = str(ctx.params.get("price_col", "last"))
        start_time = parse_hhmm(str(ctx.params.get("start_time", "09:30")))
        end_time = parse_hhmm(str(ctx.params.get("end_time", "14:30")))
        if price_col not in panel.columns:
            raise KeyError(f"ret_open_to_time missing column: {price_col}")

        def _calc(df: pd.DataFrame) -> float:
            df = df.sort_values("trade_time")
            dt_value = df.index.get_level_values("dt")[0]
            base_date = pd.Timestamp(dt_value).normalize()
            start_dt = base_date + pd.Timedelta(hours=start_time.hour, minutes=start_time.minute)
            end_dt = base_date + pd.Timedelta(hours=end_time.hour, minutes=end_time.minute)
            window = df[(df["trade_time"] >= start_dt) & (df["trade_time"] <= end_dt)]
            if window.empty:
                return 0.0
            first, last = first_last_price(window, price_col)
            if first is None or first == 0:
                return 0.0
            return (last - first) / first

        out = _group_scalar(panel, _calc)
        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out

