from __future__ import annotations

from datetime import time as dt_time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import DailyFactorRequirement, Factor, FactorComputeContext, ensure_panel_index


@FactorRegistry.register("daily_prior_intraday_return_surprise_v1")
class DailyPriorIntradayReturnSurpriseV1Factor(Factor):
    """Current T1430 intraday return relative to strictly prior daily sessions.

    This research-only factor makes the current-day numerator explicit while
    keeping all daily-TWAP inputs strictly before the signal date.  It does not
    use labels, pools, masks, scores, or result data.
    """

    name = "daily_prior_intraday_return_surprise_v1"

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

    @staticmethod
    def _cutoff_time(value: object) -> dt_time:
        raw = str(value or "14:30").strip()
        parts = raw.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("daily_prior_intraday_return_surprise_v1 cutoff_time must be HH:MM")
        hour, minute = (int(part) for part in parts)
        return dt_time(hour, minute)

    @classmethod
    def daily_requirements(cls, params: dict | None = None) -> list[DailyFactorRequirement]:
        params = dict(params or {})
        source = str(params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        open_col = str(params.get("open_col", "twap_0930_0935")).strip() or "twap_0930_0935"
        late_col = str(params.get("late_col", "twap_1430_1442")).strip() or "twap_1430_1442"
        window = int(params.get("window", 120) or 120)
        if window < 2:
            raise ValueError("daily_prior_intraday_return_surprise_v1 requires window >= 2")
        lookback = int(params.get("context_lookback_days", window + 1) or (window + 1))
        return [
            DailyFactorRequirement(
                source=source,
                columns=(open_col, late_col),
                lookback_days=max(window + 1, lookback),
            )
        ]

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_panel_index(ctx.panel)
        out = self._empty_result(panel)

        source = str(ctx.params.get("source", "market_cbond.daily_twap")).strip() or "market_cbond.daily_twap"
        open_col = str(ctx.params.get("open_col", "twap_0930_0935")).strip() or "twap_0930_0935"
        late_col = str(ctx.params.get("late_col", "twap_1430_1442")).strip() or "twap_1430_1442"
        window = int(ctx.params.get("window", 120) or 120)
        if window < 2:
            raise ValueError("daily_prior_intraday_return_surprise_v1 requires window >= 2")
        cutoff = self._cutoff_time(ctx.params.get("cutoff_time", "14:30"))

        panel_frame = panel.reset_index()
        panel_required = {"dt", "code", "trade_time", "open", "last"}
        missing_panel = sorted(panel_required.difference(panel_frame.columns))
        if missing_panel:
            raise KeyError(
                "daily_prior_intraday_return_surprise_v1 panel must include columns: "
                f"{missing_panel}"
            )
        panel_frame["trade_time"] = pd.to_datetime(panel_frame["trade_time"], errors="coerce")
        panel_frame["signal_date"] = pd.to_datetime(panel_frame["dt"], errors="coerce").dt.normalize()
        panel_frame["open"] = pd.to_numeric(panel_frame["open"], errors="coerce")
        panel_frame["last"] = pd.to_numeric(panel_frame["last"], errors="coerce")
        panel_frame = panel_frame.loc[
            panel_frame["trade_time"].notna()
            & panel_frame["signal_date"].notna()
            & (panel_frame["trade_time"].dt.time < cutoff)
        ].copy()

        daily = ctx.daily_data.get(source)
        if daily is None or daily.empty:
            return out
        daily_required = {"trade_date", "code", open_col, late_col}
        missing_daily = sorted(daily_required.difference(daily.columns))
        if missing_daily:
            raise KeyError(
                "daily_prior_intraday_return_surprise_v1 "
                f"source={source} must include columns: {missing_daily}"
            )

        work = daily[["trade_date", "code", open_col, late_col]].copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
        work["instrument_code"] = self._to_instrument_code(work["code"])
        work[open_col] = pd.to_numeric(work[open_col], errors="coerce")
        work[late_col] = pd.to_numeric(work[late_col], errors="coerce")
        work = work.dropna(subset=["trade_date", "instrument_code"])
        work = work.loc[work["instrument_code"] != ""].copy()
        if work.empty:
            return out
        if work.duplicated(["trade_date", "instrument_code"], keep=False).any():
            raise ValueError("daily_prior_intraday_return_surprise_v1 received duplicate daily date/code rows")
        valid_daily_price = (
            np.isfinite(work[open_col])
            & np.isfinite(work[late_col])
            & (work[open_col] > 0.0)
            & (work[late_col] > 0.0)
        )
        work["historical_return"] = np.where(
            valid_daily_price,
            work[late_col] / work[open_col] - 1.0,
            np.nan,
        )
        work = work.sort_values(["trade_date", "instrument_code"], kind="mergesort").reset_index(drop=True)

        key = out.index.to_frame(index=False)
        key["signal_date"] = pd.to_datetime(key["dt"], errors="coerce").dt.normalize()
        if key["signal_date"].isna().any():
            raise ValueError("daily_prior_intraday_return_surprise_v1 received an invalid panel dt")

        for signal_date in sorted(key["signal_date"].unique()):
            prior_dates = pd.Index(
                work.loc[work["trade_date"] < signal_date, "trade_date"].unique()
            ).sort_values()
            if len(prior_dates) < window:
                continue
            session_dates = prior_dates[-window:]
            history = work.loc[work["trade_date"].isin(session_dates)].copy()
            if history.empty:
                continue
            grouped = history.groupby("instrument_code", sort=False)["historical_return"]
            count = grouped.count()
            mean = grouped.mean()
            std = grouped.std(ddof=1)
            eligible_stats = (count == window) & std.notna() & (std > 0.0)
            mean = mean.loc[eligible_stats]
            std = std.loc[eligible_stats]
            if mean.empty:
                continue

            current = panel_frame.loc[panel_frame["signal_date"] == signal_date].copy()
            current = current.loc[
                np.isfinite(current["open"])
                & np.isfinite(current["last"])
                & (current["open"] > 0.0)
                & (current["last"] > 0.0)
            ]
            if current.empty:
                continue
            current = current.sort_values(["dt", "code", "trade_time"], kind="mergesort")
            current = current.groupby(["dt", "code"], sort=False, as_index=False).tail(1).copy()
            current["instrument_code"] = self._to_instrument_code(current["code"])
            current["current_return"] = current["last"] / current["open"] - 1.0
            current = current.drop_duplicates("instrument_code", keep=False).set_index("instrument_code")

            positions = key.index[key["signal_date"] == signal_date]
            target_codes = self._to_instrument_code(key.loc[positions, "code"])
            current_return = target_codes.map(current["current_return"])
            values = (current_return - target_codes.map(mean)) / target_codes.map(std)
            values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            out.iloc[positions] = values.to_numpy(dtype="float64")

        return out
