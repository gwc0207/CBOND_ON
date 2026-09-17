from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("daily_overnight_return_mean_v1")
class DailyOvernightReturnMeanV1Factor(Factor):
    name = "daily_overnight_return_mean_v1"

    @staticmethod
    def _to_instrument_code(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip().str.upper()
        return s.str.split(".", n=1).str[0]

    def _empty_result(self, panel: pd.DataFrame) -> pd.Series:
        keys = panel.index.droplevel("seq").unique()
        out = pd.Series(index=keys, dtype="float64")
        out.index = pd.MultiIndex.from_tuples(out.index.tolist(), names=["dt", "code"])
        out.name = self.output_name(self.name)
        return out

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        params = dict(params or {})

        source = str(params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        buy_col = str(params.get("buy_col", "twap_1442_1457")).strip() or "twap_1442_1457"

        time_tag = str(params.get("time_tag", "0930_1000")).strip() or "0930_1000"
        sell_col = str(params.get("sell_col", f"twap_{time_tag}")).strip() or f"twap_{time_tag}"

        window_raw = params.get("window", params.get("lookback_days", 20))
        window = max(1, int(window_raw or 20))

        context_lookback = int(params.get("context_lookback_days", window + 2) or (window + 2))
        context_lookback = max(window + 2, context_lookback)

        return [
            DailyFactorRequirement(
                source=source,
                columns=(buy_col, sell_col),
                lookback_days=context_lookback,
            )
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)

        source = str(ctx.params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        buy_col = str(ctx.params.get("buy_col", "twap_1442_1457")).strip() or "twap_1442_1457"

        time_tag = str(ctx.params.get("time_tag", "0930_1000")).strip() or "0930_1000"
        sell_col = str(ctx.params.get("sell_col", f"twap_{time_tag}")).strip() or f"twap_{time_tag}"

        window_raw = ctx.params.get("window", ctx.params.get("lookback_days", 20))
        window = max(1, int(window_raw or 20))
        min_periods = max(1, int(ctx.params.get("min_periods", window) or window))
        min_periods = min(min_periods, window)

        daily = ctx.daily_data.get(source)
        if daily is None or daily.empty:
            return self._empty_result(panel)

        required_cols = {"trade_date", "code", buy_col, sell_col}
        missing = required_cols.difference(daily.columns)
        if missing:
            raise KeyError(
                f"daily_overnight_return_mean_v1 source={source} must include columns: {sorted(missing)}"
            )

        work = daily[["trade_date", "code", buy_col, sell_col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
        work["instrument_code"] = self._to_instrument_code(work["code"])

        work[buy_col] = pd.to_numeric(work[buy_col], errors="coerce")
        work[sell_col] = pd.to_numeric(work[sell_col], errors="coerce")

        work = work.dropna(subset=["trade_date", "instrument_code", buy_col, sell_col])
        if work.empty:
            return self._empty_result(panel)

        work = work.sort_values(["instrument_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        work["prev_buy"] = work.groupby("instrument_code", sort=False)[buy_col].shift(1)

        valid_px = (
            work["prev_buy"].notna()
            & work[sell_col].notna()
            & (work["prev_buy"] > 0)
            & (work[sell_col] > 0)
        )
        work["overnight_ret"] = np.where(
            valid_px,
            work[sell_col] / work["prev_buy"] - 1.0,
            np.nan,
        )

        rolling_mean = (
            work.groupby("instrument_code", sort=False)["overnight_ret"]
            .rolling(window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        work["overnight_return_mean"] = rolling_mean
        work["overnight_return_mean"] = work.groupby("instrument_code", sort=False)[
            "overnight_return_mean"
        ].shift(1)
        work["overnight_return_mean"] = work["overnight_return_mean"].replace([np.inf, -np.inf], np.nan)

        daily_factor = work[["trade_date", "instrument_code", "overnight_return_mean"]].dropna(
            subset=["overnight_return_mean"]
        )

        out = self._empty_result(panel)
        if daily_factor.empty:
            return out

        key_df = out.index.to_frame(index=False)
        key_df["trade_date"] = pd.to_datetime(key_df["dt"], errors="coerce").dt.normalize()
        key_df["instrument_code"] = self._to_instrument_code(key_df["code"])

        merged = key_df.merge(
            daily_factor,
            on=["trade_date", "instrument_code"],
            how="left",
            sort=False,
        )

        values = pd.to_numeric(merged["overnight_return_mean"], errors="coerce").to_numpy(dtype="float64")
        out.iloc[:] = values
        return out
