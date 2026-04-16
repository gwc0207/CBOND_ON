from __future__ import annotations

import math

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("daily_sharpe_mean_v1")
class DailySharpeMeanV1Factor(Factor):
    name = "daily_sharpe_mean_v1"

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
            raise RuntimeError(f"daily_sharpe_mean_v1 missing daily context source: {source}")
        if "trade_date" not in daily.columns or "code" not in daily.columns or price_col not in daily.columns:
            raise KeyError(
                f"daily_sharpe_mean_v1 source={source} must include trade_date/code/{price_col}"
            )

        work = daily[["trade_date", "code", price_col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
        work["code"] = work["code"].astype(str).str.strip().str.upper()
        work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
        work = work.dropna(subset=["trade_date", "code", price_col])
        if work.empty:
            raise RuntimeError(f"daily_sharpe_mean_v1 source={source} has no valid rows")

        work = work.sort_values(["code", "trade_date"], kind="mergesort").reset_index(drop=True)
        work["ret"] = work.groupby("code", sort=False)[price_col].pct_change()
        mean_s = (
            work.groupby("code", sort=False)["ret"]
            .rolling(lookback_days, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
        std_s = (
            work.groupby("code", sort=False)["ret"]
            .rolling(lookback_days, min_periods=min_periods)
            .std(ddof=1)
            .reset_index(level=0, drop=True)
        )
        sharpe = mean_s / std_s
        if annualize:
            sharpe = sharpe * math.sqrt(252.0)
        if smooth_days > 1:
            sharpe = (
                sharpe.groupby(work["code"], sort=False)
                .rolling(smooth_days, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
        work["sharpe"] = sharpe.replace([np.inf, -np.inf], np.nan)
        daily_sharpe = work[["trade_date", "code", "sharpe"]].dropna(subset=["sharpe"])

        keys = panel.index.droplevel("seq").unique()
        out = pd.Series(index=keys, dtype="float64")
        out.index = pd.MultiIndex.from_tuples(out.index.tolist(), names=["dt", "code"])

        if daily_sharpe.empty:
            out = out.fillna(0.0)
            out.name = self.output_name(self.name)
            return out

        key_df = out.index.to_frame(index=False)
        key_df["trade_date"] = pd.to_datetime(key_df["dt"], errors="coerce").dt.normalize()
        key_df["code"] = key_df["code"].astype(str).str.strip().str.upper()

        for code, code_keys in key_df.groupby("code", sort=False):
            src = daily_sharpe[daily_sharpe["code"] == code][["trade_date", "sharpe"]].sort_values(
                "trade_date", kind="mergesort"
            )
            if src.empty:
                continue
            tgt = code_keys[["dt", "trade_date"]].sort_values("trade_date", kind="mergesort")
            joined = pd.merge_asof(
                tgt,
                src,
                on="trade_date",
                direction="backward",
            )
            for dt_value, val in zip(joined["dt"], joined["sharpe"]):
                out.loc[(dt_value, code)] = float(val) if pd.notna(val) else np.nan

        out = out.fillna(0.0)
        out.name = self.output_name(self.name)
        return out
