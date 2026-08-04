"""Research-only causal tail-path efficiency from valid L1 midpoint quotes.

The factor uses only the six complete clock-minute buckets from 14:24 through
14:29 on the factor day.  It deliberately excludes 14:30 and later snapshots,
labels, masks, daily data, and fallback prices.  A missing/invalid bucket is a
missing signal, never a zero-filled path.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.base import Factor, FactorComputeContext
from cbond_on.domain.factors.defs._intraday_utils import ensure_trade_time, _iter_dt_code_groups


_START_TIME = time(14, 24)
_END_TIME = time(14, 30)  # Exclusive: 14:30 itself is never visible.
_MINUTE_COUNT = 6
_EPS = 1e-12


def _tail_path_efficiency(dt: object, group: pd.DataFrame) -> float:
    signal_dt = pd.Timestamp(dt)
    if pd.isna(signal_dt):
        raise ValueError("tail_path_efficiency_5m_v1 received an invalid panel dt")
    signal_day = signal_dt.date()
    start = pd.Timestamp.combine(signal_day, _START_TIME)
    end = pd.Timestamp.combine(signal_day, _END_TIME)

    frame = group.reset_index().copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    if frame["trade_time"].isna().any():
        raise ValueError("tail_path_efficiency_5m_v1 received an invalid trade_time")
    frame = frame.loc[
        (frame["trade_time"] >= start) & (frame["trade_time"] < end)
    ].copy()
    if frame.empty:
        return float("nan")

    bid = pd.to_numeric(frame["bid_price1"], errors="coerce")
    ask = pd.to_numeric(frame["ask_price1"], errors="coerce")
    valid = (
        np.isfinite(bid)
        & np.isfinite(ask)
        & (bid > 0.0)
        & (ask > 0.0)
        & (bid <= ask)
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        return float("nan")
    frame["_mid"] = (bid.loc[valid] + ask.loc[valid]) / 2.0
    frame["_minute"] = frame["trade_time"].dt.floor("min")
    sort_columns = ["trade_time"] + (["seq"] if "seq" in frame.columns else [])
    frame = frame.sort_values(sort_columns, kind="mergesort")
    # Updates arrive at uneven per-bond frequencies.  One last valid midpoint
    # per clock minute gives every bond the same six-step path definition.
    buckets = frame.groupby("_minute", sort=True, group_keys=False).tail(1)
    buckets = buckets.sort_values(["_minute"] + (["seq"] if "seq" in buckets.columns else []), kind="mergesort")
    expected = pd.date_range(start, periods=_MINUTE_COUNT, freq="min")
    if len(buckets) != _MINUTE_COUNT or not buckets["_minute"].reset_index(drop=True).equals(
        pd.Series(expected)
    ):
        return float("nan")

    log_mid = np.log(buckets["_mid"].to_numpy(dtype=float, copy=False))
    if not np.isfinite(log_mid).all():
        return float("nan")
    path_length = float(np.abs(np.diff(log_mid)).sum())
    if not np.isfinite(path_length) or path_length <= _EPS:
        return float("nan")
    result = float((log_mid[-1] - log_mid[0]) / path_length)
    return result if np.isfinite(result) else float("nan")


@FactorRegistry.register("tail_path_efficiency_5m_v1")
class TailPathEfficiency5mV1Factor(Factor):
    """Signed straightness of the fixed 14:24--14:29 L1-midpoint path."""

    name = "tail_path_efficiency_5m_v1"

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        panel = ensure_trade_time(ctx.panel)
        required = {"trade_time", "bid_price1", "ask_price1"}
        missing = sorted(required.difference(panel.columns))
        if missing:
            raise KeyError(f"tail_path_efficiency_5m_v1 missing panel columns: {missing}")
        rows = [
            (dt, str(code), _tail_path_efficiency(dt, group))
            for dt, code, group in _iter_dt_code_groups(panel)
        ]
        if not rows:
            raise ValueError("tail_path_efficiency_5m_v1 received an empty panel")
        index = pd.MultiIndex.from_tuples(
            [(dt, code) for dt, code, _ in rows], names=["dt", "code"]
        )
        output = pd.Series([value for _, _, value in rows], index=index, dtype="float64")
        output = output.replace([np.inf, -np.inf], np.nan)
        output.name = self.output_name(self.name)
        return output
