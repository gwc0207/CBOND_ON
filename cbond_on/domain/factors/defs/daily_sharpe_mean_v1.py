from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("daily_sharpe_mean_v1")
class DailySharpeMeanV1Factor(Factor):
    name = "daily_sharpe_mean_v1"

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
        price_col = str(params.get("price_col", "twap_1442_1457")).strip() or "twap_1442_1457"
        lookback_days = int(params.get("lookback_days", 20) or 20)
        smooth_days = int(params.get("smooth_days", 5) or 5)
        context_lookback = int(params.get("context_lookback_days", lookback_days + smooth_days + 2))
        context_lookback = max(1, context_lookback)
        return [
            DailyFactorRequirement(
                source=source,
                columns=(price_col,),
                lookback_days=context_lookback,
            )
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)
        source = str(ctx.params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        price_col = str(ctx.params.get("price_col", "twap_1442_1457")).strip() or "twap_1442_1457"
        lookback_days = max(2, int(ctx.params.get("lookback_days", 20) or 20))
        smooth_days = max(1, int(ctx.params.get("smooth_days", 5) or 5))
        min_periods = max(2, int(ctx.params.get("min_periods", lookback_days) or lookback_days))
        annualize = bool(ctx.params.get("annualize", False))

        daily = ctx.daily_data.get(source)
        if daily is None or daily.empty:
            return self._empty_result(panel)
        if "trade_date" not in daily.columns or "code" not in daily.columns or price_col not in daily.columns:
            raise KeyError(
                f"daily_sharpe_mean_v1 source={source} must include trade_date/code/{price_col}"
            )

        work = daily[["trade_date", "code", price_col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
        work["instrument_code"] = self._to_instrument_code(work["code"])
        work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
        work = work.dropna(subset=["trade_date", "instrument_code", price_col])
        if work.empty:
            return self._empty_result(panel)

        work = work.sort_values(["instrument_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        work["ret"] = work.groupby("instrument_code", sort=False)[price_col].pct_change()
        mean_s = (
            work.groupby("instrument_code", sort=False)["ret"]
            .rolling(lookback_days, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        std_s = (
            work.groupby("instrument_code", sort=False)["ret"]
            .rolling(lookback_days, min_periods=min_periods)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
        )
        sharpe = mean_s / std_s
        if annualize:
            sharpe = sharpe * math.sqrt(252.0)
        if smooth_days > 1:
            sharpe = (
                sharpe.groupby(work["instrument_code"], sort=False)
                .rolling(smooth_days, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        work["sharpe"] = sharpe.replace([np.inf, -np.inf], np.nan)
        daily_sharpe = work[["trade_date", "instrument_code", "sharpe"]].dropna(subset=["sharpe"])

        out = self._empty_result(panel)

        if daily_sharpe.empty:
            return out

        key_df = out.index.to_frame(index=False)
        key_df["trade_date"] = pd.to_datetime(key_df["dt"], errors="coerce").dt.normalize()
        key_df["instrument_code"] = self._to_instrument_code(key_df["code"])

        for instrument_code, code_keys in key_df.groupby("instrument_code", sort=False):
            src = daily_sharpe[
                daily_sharpe["instrument_code"] == instrument_code
            ][["trade_date", "sharpe"]].sort_values(
                "trade_date", kind="mergesort"
            )
            if src.empty:
                continue
            tgt = code_keys[["dt", "trade_date", "code"]].sort_values("trade_date", kind="mergesort")
            joined = pd.merge_asof(
                tgt,
                src,
                on="trade_date",
                direction="backward",
            )
            for dt_value, code_value, val in zip(joined["dt"], joined["code"], joined["sharpe"]):
                out.loc[(dt_value, code_value)] = float(val) if pd.notna(val) else np.nan

        return out
