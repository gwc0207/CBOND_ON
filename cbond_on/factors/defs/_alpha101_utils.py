from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.factors.base import Factor, FactorComputeContext
from cbond_on.factors.defs._intraday_utils import ensure_trade_time

EPS = 1e-8


def _prepare_panel(ctx: FactorComputeContext, required: list[str]) -> pd.DataFrame:
    panel = ensure_trade_time(ctx.panel)
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"alpha101 missing columns: {missing}")
    frame = panel.reset_index()[["dt", "code", "seq", *required]].copy()
    frame = frame.sort_values(["dt", "code", "seq"], kind="mergesort")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _group_scalar(frame: pd.DataFrame, func) -> pd.Series:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for (dt, code), g in frame.groupby(["dt", "code"], sort=False):
        g = g.sort_values("seq", kind="mergesort")
        try:
            val = float(func(g))
        except Exception:
            val = np.nan
        rows.append((dt, str(code), val))
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.Series([v for _, _, v in rows], index=idx, dtype="float64")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _cs_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.groupby(level="dt").rank(pct=True, method="average")


def _delta_last(series: pd.Series, periods: int) -> float:
    periods = max(1, int(periods))
    if len(series) <= periods:
        return 0.0
    return float(series.iloc[-1] - series.iloc[-1 - periods])


def _delay_last(series: pd.Series, periods: int) -> float:
    periods = max(1, int(periods))
    if series.empty:
        return 0.0
    if len(series) <= periods:
        return float(series.iloc[0])
    return float(series.iloc[-1 - periods])


def _ts_rank_last(series: pd.Series, window: int) -> float:
    window = max(1, int(window))
    tail = series.tail(window).dropna()
    if tail.empty:
        return 0.0
    ranked = tail.rank(pct=True, method="average")
    return float(ranked.iloc[-1])


def _corr_last(x: pd.Series, y: pd.Series, window: int) -> float:
    window = max(2, int(window))
    tail = pd.concat([x, y], axis=1).dropna().tail(window)
    if len(tail) < 2:
        return 0.0
    a = tail.iloc[:, 0].astype("float64")
    b = tail.iloc[:, 1].astype("float64")
    if float(a.std(ddof=0)) <= EPS or float(b.std(ddof=0)) <= EPS:
        return 0.0
    corr = a.corr(b)
    if pd.isna(corr):
        return 0.0
    return float(corr)


def _cov_last(x: pd.Series, y: pd.Series, window: int) -> float:
    window = max(2, int(window))
    tail = pd.concat([x, y], axis=1).dropna().tail(window)
    if len(tail) < 2:
        return 0.0
    cov = tail.iloc[:, 0].astype("float64").cov(tail.iloc[:, 1].astype("float64"))
    if pd.isna(cov):
        return 0.0
    return float(cov)


def _open_like(g: pd.DataFrame) -> pd.Series:
    open_px: pd.Series | None = None
    if "open" in g.columns:
        open_px = g["open"].astype("float64")
    mid: pd.Series | None = None
    if "ask_price1" in g.columns and "bid_price1" in g.columns:
        ask = g["ask_price1"].astype("float64")
        bid = g["bid_price1"].astype("float64")
        mid = (ask + bid) / 2.0
    if mid is not None and open_px is not None:
        return mid.where(mid.notna(), open_px)
    if mid is not None:
        return mid
    if open_px is not None:
        return open_px
    raise KeyError("open-like requires open or [ask_price1, bid_price1]")


class _AlphaBase(Factor):
    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        raise NotImplementedError

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        out = self._compute_series(ctx)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out.name = self.output_name(self.name)
        return out
