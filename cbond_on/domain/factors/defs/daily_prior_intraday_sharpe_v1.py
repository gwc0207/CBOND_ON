from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("daily_prior_intraday_sharpe_v1")
class DailyPriorIntradaySharpeV1Factor(Factor):
    """Strictly prior-session intraday TWAP Sharpe for research-only use."""

    name = "daily_prior_intraday_sharpe_v1"

    @staticmethod
    def _to_instrument_code(series: pd.Series) -> pd.Series:
        value = series.astype(str).str.strip().str.upper()
        return value.str.split(".", n=1).str[0]

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
        open_col = str(params.get("open_col", "twap_0930_0935")).strip() or "twap_0930_0935"
        close_col = str(params.get("close_col", "twap_1442_1457")).strip() or "twap_1442_1457"
        window = int(params.get("window", 120) or 120)
        if window < 2:
            raise ValueError("daily_prior_intraday_sharpe_v1 requires window >= 2")
        # The daily loader includes the score-date file in an on-or-before
        # slice.  Request that file plus exactly ``window`` earlier sessions;
        # compute() then enforces the causal ``trade_date < signal_date`` cut.
        lookback = int(params.get("context_lookback_days", window + 1) or (window + 1))
        lookback = max(window + 1, lookback)
        return [
            DailyFactorRequirement(
                source=source,
                columns=(open_col, close_col),
                lookback_days=lookback,
            )
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)
        out = self._empty_result(panel)

        source = str(ctx.params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        open_col = str(ctx.params.get("open_col", "twap_0930_0935")).strip() or "twap_0930_0935"
        close_col = str(ctx.params.get("close_col", "twap_1442_1457")).strip() or "twap_1442_1457"
        window = int(ctx.params.get("window", 120) or 120)
        if window < 2:
            raise ValueError("daily_prior_intraday_sharpe_v1 requires window >= 2")

        daily = ctx.daily_data.get(source)
        if daily is None or daily.empty:
            return out
        required = {"trade_date", "code", open_col, close_col}
        missing = sorted(required.difference(daily.columns))
        if missing:
            raise KeyError(
                "daily_prior_intraday_sharpe_v1 "
                f"source={source} must include columns: {missing}"
            )

        work = daily[["trade_date", "code", open_col, close_col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
        work["instrument_code"] = self._to_instrument_code(work["code"])
        work[open_col] = pd.to_numeric(work[open_col], errors="coerce")
        work[close_col] = pd.to_numeric(work[close_col], errors="coerce")
        work = work.dropna(subset=["trade_date", "instrument_code"])
        work = work.loc[work["instrument_code"] != ""].copy()
        if work.empty:
            return out
        duplicate = work.duplicated(["trade_date", "instrument_code"], keep=False)
        if duplicate.any():
            raise ValueError("daily_prior_intraday_sharpe_v1 received duplicate daily date/code rows")

        valid_price = (
            np.isfinite(work[open_col])
            & np.isfinite(work[close_col])
            & (work[open_col] > 0.0)
            & (work[close_col] > 0.0)
        )
        work["intraday_return"] = np.where(
            valid_price,
            work[close_col] / work[open_col] - 1.0,
            np.nan,
        )
        work = work.sort_values(["trade_date", "instrument_code"], kind="mergesort").reset_index(drop=True)

        key = out.index.to_frame(index=False)
        key["signal_date"] = pd.to_datetime(key["dt"], errors="coerce").dt.normalize()
        if key["signal_date"].isna().any():
            raise ValueError("daily_prior_intraday_sharpe_v1 received an invalid panel dt")

        for signal_date in sorted(key["signal_date"].unique()):
            # This is the causal boundary.  The daily context intentionally
            # supplies the score-date file too, so filtering it here is
            # mandatory rather than an assumption about loader behavior.
            prior_dates = pd.Index(work.loc[work["trade_date"] < signal_date, "trade_date"].unique()).sort_values()
            if len(prior_dates) < window:
                continue
            session_dates = prior_dates[-window:]
            history = work.loc[work["trade_date"].isin(session_dates)].copy()
            if history.empty:
                continue

            grouped = history.groupby("instrument_code", sort=False)["intraday_return"]
            count = grouped.count()
            mean = grouped.mean()
            std = grouped.std(ddof=1)
            sharpe = (mean / std).replace([np.inf, -np.inf], np.nan)
            sharpe = sharpe.loc[(count == window) & std.notna() & (std > 0.0)]
            if sharpe.empty:
                continue

            target_positions = key.index[key["signal_date"] == signal_date]
            target_codes = self._to_instrument_code(key.loc[target_positions, "code"])
            values = target_codes.map(sharpe)
            out.iloc[target_positions] = values.to_numpy(dtype="float64")

        return out
