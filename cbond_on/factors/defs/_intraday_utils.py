from __future__ import annotations

from datetime import time as dt_time
from typing import Callable

import numpy as np
import pandas as pd

from cbond_on.factors.base import Factor, FactorComputeContext, ensure_panel_index

EPS = 1e-8
OPEN_LIKE_CACHE_COL = "__open_like__"
OHLC_COLS = ("open", "high", "low", "close")
REBUILD_PRICE_COLS = ("open", "high", "low", "close", "last")
REBUILD_VALUE_COLS = ("volume", "amount", "num_trades")
REBUILD_PREV_COLS = ("pre_close", "prev_bar_close")
REBUILD_TRIGGER_COLS = REBUILD_PRICE_COLS + REBUILD_VALUE_COLS + REBUILD_PREV_COLS + (
    "ask_price1",
    "bid_price1",
)


def _resolve_ohlc_rebuild_params(params: dict, required: list[str]) -> tuple[bool, int]:
    need_rebuild = any(col in required for col in REBUILD_TRIGGER_COLS)
    if not need_rebuild:
        return False, 0

    # Single-param mode: pass `windowsize` (or alias `window_size`) to enable
    # intraday bar rebuild (non-overlap buckets by seq order).
    raw_size = params.get("windowsize", params.get("window_size"))
    if raw_size is None:
        return False, 0
    try:
        window_points = max(1, int(raw_size))
    except Exception:
        return False, 0
    return True, window_points


def _normalize_ohlc_windows_plan(params: dict) -> list[int]:
    raw = params.get("__ohlc_windows_plan__")
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        try:
            w = int(item)
        except Exception:
            continue
        if w <= 0 or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return sorted(out)


def _build_rebuilt_bar_frame(
    base_frame: pd.DataFrame,
    *,
    window_points: int,
) -> pd.DataFrame:
    w = max(1, int(window_points))
    if base_frame.empty:
        return base_frame.loc[:, ["dt", "code", "seq"]].copy(deep=False)

    row_no = base_frame.groupby(["dt", "code"], sort=False).cumcount()
    bar_seq = (row_no // w).astype("int64")
    bar_seq.name = "bar_seq"
    group_keys = [base_frame["dt"], base_frame["code"], bar_seq]

    last_px = pd.to_numeric(base_frame["last"], errors="coerce")
    grouped_last = last_px.groupby(group_keys, sort=False)
    bars = pd.DataFrame(
        {
            "open": grouped_last.first(),
            "high": grouped_last.max(),
            "low": grouped_last.min(),
            "close": grouped_last.last(),
        }
    )
    if isinstance(bars.index, pd.MultiIndex) and bars.index.nlevels == 3:
        bars.index = bars.index.set_names(["dt", "code", "bar_seq"])
    bars["last"] = bars["close"]

    if "trade_time" in base_frame.columns:
        bars["trade_time"] = pd.to_datetime(base_frame["trade_time"], errors="coerce").groupby(
            group_keys, sort=False
        ).last()

    for col in REBUILD_VALUE_COLS:
        if col not in base_frame.columns:
            continue
        cum = pd.to_numeric(base_frame[col], errors="coerce")
        delta = cum.groupby([base_frame["dt"], base_frame["code"]], sort=False).diff()
        delta = delta.where(delta.notna(), cum)
        delta = delta.where(delta >= 0.0, cum)
        delta = delta.fillna(0.0)
        bars[col] = delta.groupby(group_keys, sort=False).sum(min_count=1).fillna(0.0)

    carry_cols = [
        c
        for c in base_frame.columns
        if c not in {"dt", "code", "seq"}
        and c not in bars.columns
        and c not in REBUILD_VALUE_COLS
    ]
    for col in carry_cols:
        src = base_frame[col]
        if pd.api.types.is_numeric_dtype(src):
            bars[col] = pd.to_numeric(src, errors="coerce").groupby(group_keys, sort=False).last()
        else:
            bars[col] = src.groupby(group_keys, sort=False).last()

    prev_bar_close = bars.groupby(level=["dt", "code"], sort=False)["close"].shift(1)
    if "pre_close" in base_frame.columns:
        pre_close_day = pd.to_numeric(base_frame["pre_close"], errors="coerce").groupby(
            [base_frame["dt"], base_frame["code"]],
            sort=False,
        ).first()
        lookup_idx = pd.MultiIndex.from_arrays(
            [bars.index.get_level_values(0), bars.index.get_level_values(1)],
            names=["dt", "code"],
        )
        fallback = pre_close_day.reindex(lookup_idx).to_numpy()
        prev_bar_close = prev_bar_close.fillna(
            pd.Series(fallback, index=bars.index, dtype="float64")
        )
    bars["prev_bar_close"] = pd.to_numeric(prev_bar_close, errors="coerce")
    bars["pre_close"] = bars["prev_bar_close"]

    out = bars.reset_index()
    if "bar_seq" in out.columns:
        out = out.rename(columns={"bar_seq": "seq"})
    elif "level_2" in out.columns:
        out = out.rename(columns={"level_2": "seq"})
    out["seq"] = pd.to_numeric(out["seq"], errors="coerce").fillna(0).astype("int64")
    out = out.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
    return out


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
            cache = {
                "base_frame": None,
                "numeric_cols": {},
                "derived_cols": {},
                "rebuild_frames": {},
            }
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
    if ohlc_rebuild and "last" not in base_frame.columns:
        raise KeyError("alpha101 ohlc rebuild requires column: last")

    rebuild_windows = _normalize_ohlc_windows_plan(ctx.params)
    if ohlc_rebuild and ohlc_window_points not in rebuild_windows:
        rebuild_windows.append(ohlc_window_points)
        rebuild_windows = sorted(set(rebuild_windows))

    rebuild_frames: dict[int, pd.DataFrame] = cache["rebuild_frames"]
    if ohlc_rebuild and rebuild_windows:
        to_build: list[int] = []
        with ctx.cache_lock:
            for w in rebuild_windows:
                if w not in rebuild_frames:
                    to_build.append(w)
        for w in to_build:
            built = _build_rebuilt_bar_frame(base_frame, window_points=w)
            with ctx.cache_lock:
                rebuild_frames.setdefault(w, built)

    frame_source = base_frame
    mode_key = "base"
    if ohlc_rebuild:
        with ctx.cache_lock:
            frame_source = rebuild_frames[ohlc_window_points]
        mode_key = f"w{ohlc_window_points}"

    missing = [c for c in required if c not in frame_source.columns]
    if missing:
        raise KeyError(f"alpha101 missing columns: {missing}")

    numeric_cols: dict[str, pd.Series] = cache["numeric_cols"]
    derived_cols: dict[str, pd.Series] = cache["derived_cols"]

    def _ensure_numeric(col: str) -> pd.Series:
        cache_name = f"{mode_key}:{col}"
        with ctx.cache_lock:
            cached_col = numeric_cols.get(cache_name)
        if cached_col is not None:
            return cached_col
        converted = pd.to_numeric(frame_source[col], errors="coerce")
        with ctx.cache_lock:
            numeric_cols.setdefault(cache_name, converted)
            return numeric_cols[cache_name]

    for col in required:
        _ensure_numeric(col)

    need_open_like = any(c in required for c in ("open", "ask_price1", "bid_price1"))
    open_like_cache_key = f"{mode_key}:{OPEN_LIKE_CACHE_COL}"
    if need_open_like:
        with ctx.cache_lock:
            open_like_cached = derived_cols.get(open_like_cache_key)
        if open_like_cached is None:
            if ohlc_rebuild and "open" in frame_source.columns:
                open_like = _ensure_numeric("open")
            else:
                open_px = _ensure_numeric("open") if "open" in frame_source.columns else None
                ask_px = _ensure_numeric("ask_price1") if "ask_price1" in frame_source.columns else None
                bid_px = _ensure_numeric("bid_price1") if "bid_price1" in frame_source.columns else None
                mid_px = ((ask_px + bid_px) / 2.0) if (ask_px is not None and bid_px is not None) else None
                if mid_px is not None and open_px is not None:
                    open_like = mid_px.where(mid_px.notna(), open_px)
                elif mid_px is not None:
                    open_like = mid_px
                elif open_px is not None:
                    open_like = open_px
                else:
                    open_like = pd.Series(np.nan, index=frame_source.index, dtype="float64")
            with ctx.cache_lock:
                derived_cols.setdefault(open_like_cache_key, open_like)

    frame = frame_source.loc[:, ["dt", "code", "seq"]].copy(deep=False)
    for col in required:
        frame[col] = _ensure_numeric(col)
    if need_open_like:
        frame[OPEN_LIKE_CACHE_COL] = derived_cols[open_like_cache_key]
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
