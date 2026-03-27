from __future__ import annotations

from datetime import time as dt_time
from typing import Callable

import numpy as np
import pandas as pd

from cbond_on.factors.base import Factor, FactorComputeContext, ensure_panel_index

EPS = 1e-8
OPEN_LIKE_CACHE_COL = "__open_like__"
OHLC_COLS = ("open", "high", "low", "close")


def _resolve_ohlc_rebuild_params(params: dict, required: list[str]) -> tuple[bool, int]:
    need_ohlc = any(col in required for col in OHLC_COLS)
    if not need_ohlc:
        return False, 0

    # Single-param mode: pass `windowsize` (or alias `window_size`) to enable rolling OHLC rebuild.
    raw_size = params.get("windowsize", params.get("window_size"))
    if raw_size is None:
        return False, 0
    try:
        window_points = max(1, int(raw_size))
    except Exception:
        return False, 0
    return True, window_points


def _build_rolling_ohlc(
    base_frame: pd.DataFrame,
    price: pd.Series,
    *,
    window_points: int,
) -> dict[str, pd.Series]:
    w = max(1, int(window_points))
    group_keys = [base_frame["dt"], base_frame["code"]]
    grouped = price.groupby(group_keys, sort=False)

    close_series = price
    high_series = grouped.rolling(w, min_periods=1).max().reset_index(level=[0, 1], drop=True)
    low_series = grouped.rolling(w, min_periods=1).min().reset_index(level=[0, 1], drop=True)
    if w <= 1:
        open_series = price
    else:
        shifted = grouped.shift(w - 1)
        first = grouped.transform("first")
        open_series = shifted.where(shifted.notna(), first)

    return {
        "open": pd.to_numeric(open_series, errors="coerce"),
        "high": pd.to_numeric(high_series, errors="coerce"),
        "low": pd.to_numeric(low_series, errors="coerce"),
        "close": pd.to_numeric(close_series, errors="coerce"),
    }


def ensure_trade_time(panel: pd.DataFrame) -> pd.DataFrame:
    if "trade_time" not in panel.columns:
        raise KeyError("panel missing column: trade_time")
    if not pd.api.types.is_datetime64_any_dtype(panel["trade_time"]):
        panel = panel.copy()
        panel["trade_time"] = pd.to_datetime(panel["trade_time"], errors="coerce")
    return ensure_panel_index(panel)


def group_apply_scalar(panel: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> pd.Series:
    grouped = panel.groupby(level=["dt", "code"], sort=False)
    out = grouped.apply(func)
    if isinstance(out.index, pd.MultiIndex) and out.index.nlevels > 2:
        out = out.droplevel(-1)
    return out


def slice_window(df: pd.DataFrame, window_minutes: int | None) -> pd.DataFrame:
    if window_minutes is None or int(window_minutes) <= 0:
        return df
    end_time = df["trade_time"].iloc[-1]
    start_time = end_time - pd.Timedelta(minutes=int(window_minutes))
    return df[df["trade_time"] >= start_time]


def parse_hhmm(value: str) -> dt_time:
    parts = str(value).split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time value: {value}")
    return dt_time(int(parts[0]), int(parts[1]))


def first_last_price(df: pd.DataFrame, price_col: str) -> tuple[float | None, float | None]:
    if df.empty or price_col not in df.columns:
        return None, None
    series = df[price_col].astype("float64")
    if series.empty:
        return None, None
    return float(series.iloc[0]), float(series.iloc[-1])


def open_like_series(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    ask_col: str = "ask_price1",
    bid_col: str = "bid_price1",
) -> pd.Series:
    """Return open-like price series: prefer mid_price1, fallback to open."""
    mid: pd.Series | None = None
    if ask_col in df.columns and bid_col in df.columns:
        ask = pd.to_numeric(df[ask_col], errors="coerce")
        bid = pd.to_numeric(df[bid_col], errors="coerce")
        mid = (ask + bid) / 2.0

    open_px: pd.Series | None = None
    if open_col in df.columns:
        open_px = pd.to_numeric(df[open_col], errors="coerce")

    if mid is not None and open_px is not None:
        return mid.where(mid.notna(), open_px)
    if mid is not None:
        return mid
    if open_px is not None:
        return open_px
    raise KeyError(
        f"open-like requires [{open_col}] or [{ask_col}, {bid_col}] in panel columns"
    )


def _prepare_panel(ctx: FactorComputeContext, required: list[str]) -> pd.DataFrame:
    cache_key = "_alpha101_prepare_cache"
    with ctx.cache_lock:
        cache = ctx.cache.get(cache_key)
        if cache is None:
            cache = {"base_frame": None, "numeric_cols": {}, "derived_cols": {}}
            ctx.cache[cache_key] = cache

    base_frame = cache.get("base_frame")
    if base_frame is None:
        panel = ensure_trade_time(ctx.panel)
        built = panel.reset_index().sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
        with ctx.cache_lock:
            if cache.get("base_frame") is None:
                cache["base_frame"] = built
            base_frame = cache["base_frame"]

    ohlc_rebuild, ohlc_window_points = _resolve_ohlc_rebuild_params(ctx.params, required)
    derivable_ohlc_cols = set(OHLC_COLS) if ohlc_rebuild else set()
    required_source_cols = [c for c in required if c not in derivable_ohlc_cols]

    missing = [c for c in required_source_cols if c not in base_frame.columns]
    if missing:
        raise KeyError(f"alpha101 missing columns: {missing}")
    ohlc_base_price_col = "last"
    if ohlc_rebuild and ohlc_base_price_col not in base_frame.columns:
        raise KeyError("alpha101 ohlc rebuild requires column: last")

    numeric_cols: dict[str, pd.Series] = cache["numeric_cols"]
    derived_cols: dict[str, pd.Series] = cache["derived_cols"]

    def _ensure_numeric(col: str) -> pd.Series:
        with ctx.cache_lock:
            cached_col = numeric_cols.get(col)
        if cached_col is not None:
            return cached_col
        converted = pd.to_numeric(base_frame[col], errors="coerce")
        with ctx.cache_lock:
            numeric_cols.setdefault(col, converted)
            return numeric_cols[col]

    for col in required_source_cols:
        _ensure_numeric(col)

    rebuilt_ohlc: dict[str, pd.Series] = {}
    if ohlc_rebuild and any(c in required for c in OHLC_COLS):
        key_base = f"__ohlc_rebuild__{ohlc_base_price_col}__w{ohlc_window_points}"
        key_map = {
            "open": f"{key_base}__open",
            "high": f"{key_base}__high",
            "low": f"{key_base}__low",
            "close": f"{key_base}__close",
        }
        with ctx.cache_lock:
            cached = {
                k: derived_cols.get(v)
                for k, v in key_map.items()
            }
        if not all(isinstance(cached.get(k), pd.Series) for k in key_map):
            base_price = _ensure_numeric(ohlc_base_price_col)
            built_ohlc = _build_rolling_ohlc(
                base_frame,
                base_price,
                window_points=ohlc_window_points,
            )
            with ctx.cache_lock:
                for col_name, cache_key_name in key_map.items():
                    derived_cols.setdefault(cache_key_name, built_ohlc[col_name])
        with ctx.cache_lock:
            for col_name, cache_key_name in key_map.items():
                series = derived_cols.get(cache_key_name)
                if isinstance(series, pd.Series):
                    rebuilt_ohlc[col_name] = series

    need_open_like = any(c in required for c in ("open", "ask_price1", "bid_price1"))
    if need_open_like:
        if ohlc_rebuild and "open" in rebuilt_ohlc:
            with ctx.cache_lock:
                derived_cols[OPEN_LIKE_CACHE_COL] = rebuilt_ohlc["open"]
        else:
            with ctx.cache_lock:
                open_like_cached = derived_cols.get(OPEN_LIKE_CACHE_COL)
            if open_like_cached is None:
                open_px = _ensure_numeric("open") if "open" in base_frame.columns else None
                ask_px = _ensure_numeric("ask_price1") if "ask_price1" in base_frame.columns else None
                bid_px = _ensure_numeric("bid_price1") if "bid_price1" in base_frame.columns else None
                mid_px = ((ask_px + bid_px) / 2.0) if (ask_px is not None and bid_px is not None) else None
                if mid_px is not None and open_px is not None:
                    open_like = mid_px.where(mid_px.notna(), open_px)
                elif mid_px is not None:
                    open_like = mid_px
                elif open_px is not None:
                    open_like = open_px
                else:
                    open_like = pd.Series(np.nan, index=base_frame.index, dtype="float64")
                with ctx.cache_lock:
                    derived_cols.setdefault(OPEN_LIKE_CACHE_COL, open_like)

    frame = base_frame.loc[:, ["dt", "code", "seq"]].copy(deep=False)
    for col in required:
        if ohlc_rebuild and col in rebuilt_ohlc:
            frame[col] = rebuilt_ohlc[col]
        else:
            frame[col] = numeric_cols[col]
    if need_open_like and OPEN_LIKE_CACHE_COL in derived_cols:
        frame[OPEN_LIKE_CACHE_COL] = derived_cols[OPEN_LIKE_CACHE_COL]
    return frame


def _group_scalar(frame: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> pd.Series:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for (dt, code), g in frame.groupby(["dt", "code"], sort=False):
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
    if OPEN_LIKE_CACHE_COL in g.columns:
        return pd.to_numeric(g[OPEN_LIKE_CACHE_COL], errors="coerce")
    return open_like_series(g)


class _AlphaBase(Factor):
    def _compute_series(self, ctx: FactorComputeContext) -> pd.Series:
        raise NotImplementedError

    def compute(self, ctx: FactorComputeContext) -> pd.Series:
        out = self._compute_series(ctx)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out.name = self.output_name(self.name)
        return out
