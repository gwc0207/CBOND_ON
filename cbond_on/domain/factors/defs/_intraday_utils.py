from __future__ import annotations

from datetime import time as dt_time
import threading
from typing import Any, Callable

import numpy as np
import pandas as pd

from cbond_on.domain.factors.base import Factor, FactorComputeContext, ensure_panel_index

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

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
_RECORD_LOG_LOCK = threading.Lock()


def _compute_backend_runtime(params: dict) -> tuple[str, str]:
    raw = params.get("__compute_backend__", {})
    if not isinstance(raw, dict):
        return "cpu", "cuda"
    active = str(raw.get("active", "cpu")).strip().lower() or "cpu"
    device = str(raw.get("torch_device", "cuda")).strip() or "cuda"
    return active, device


def _panel_compute_backend_runtime(panel: pd.DataFrame) -> tuple[str, str]:
    backend = "cpu"
    device = "cuda"
    raw_attr = panel.attrs.get("__compute_backend__")
    if isinstance(raw_attr, dict):
        backend = str(raw_attr.get("active", backend)).strip().lower() or backend
        device = str(raw_attr.get("torch_device", device)).strip() or device
    if "__compute_backend__" in panel.columns and not panel.empty:
        raw_col = panel["__compute_backend__"].iloc[0]
        if isinstance(raw_col, dict):
            backend = str(raw_col.get("active", backend)).strip().lower() or backend
            device = str(raw_col.get("torch_device", device)).strip() or device
    return backend, device


def _panel_debug_log_each_record(panel: pd.DataFrame) -> bool:
    raw_attr = panel.attrs.get("__compute_backend__")
    if isinstance(raw_attr, dict):
        if bool(raw_attr.get("debug_log_each_record", False)):
            return True
    if "__compute_backend__" in panel.columns and not panel.empty:
        raw_col = panel["__compute_backend__"].iloc[0]
        if isinstance(raw_col, dict):
            return bool(raw_col.get("debug_log_each_record", False))
    return False


def _panel_factor_spec_name(panel: pd.DataFrame) -> str:
    raw_attr = panel.attrs.get("__factor_spec_name__")
    if isinstance(raw_attr, str) and raw_attr.strip():
        return raw_attr.strip()
    if "__factor_spec_name__" in panel.columns and not panel.empty:
        raw_col = panel["__factor_spec_name__"].iloc[0]
        if isinstance(raw_col, str) and raw_col.strip():
            return raw_col.strip()
    raw_kernel = panel.attrs.get("__factor_kernel_name__")
    if isinstance(raw_kernel, str) and raw_kernel.strip():
        return raw_kernel.strip()
    if "__factor_kernel_name__" in panel.columns and not panel.empty:
        raw_col_kernel = panel["__factor_kernel_name__"].iloc[0]
        if isinstance(raw_col_kernel, str) and raw_col_kernel.strip():
            return raw_col_kernel.strip()
    return "unknown_factor"


def _record_log_context(panel: pd.DataFrame) -> tuple[bool, str]:
    return _panel_debug_log_each_record(panel), _panel_factor_spec_name(panel)


def _emit_record_log(
    enabled: bool,
    factor_name: str,
    *,
    stage: str,
    dt: Any,
    code: Any,
    value: Any,
    rows: int | None = None,
) -> None:
    if not enabled:
        return
    try:
        if value is None:
            value_text = "None"
        else:
            value_text = f"{float(value):.10g}"
    except Exception:
        value_text = str(value)
    msg = (
        f"[factor-record] factor={factor_name} stage={stage} dt={dt} code={code} value={value_text}"
    )
    if rows is not None:
        msg += f" rows={int(rows)}"
    with _RECORD_LOG_LOCK:
        print(msg, flush=True)


def _groupby_dt_code(frame: pd.DataFrame):
    if "dt" in frame.columns and "code" in frame.columns:
        return frame.groupby(["dt", "code"], sort=False)
    if isinstance(frame.index, pd.MultiIndex) and frame.index.nlevels >= 2:
        names = list(frame.index.names[:2])
        if all(isinstance(n, str) and n for n in names):
            return frame.groupby(level=names, sort=False)
        return frame.groupby(level=[0, 1], sort=False)
    raise ValueError("frame must contain dt/code columns or MultiIndex(dt, code, ...)")


def _iter_dt_code_groups(frame: pd.DataFrame):
    grouped = _groupby_dt_code(frame)
    for key, g in grouped:
        if isinstance(key, tuple):
            dt = key[0]
            code = key[1] if len(key) > 1 else ""
        else:
            dt = key
            code = ""
        yield dt, code, g


def _dataframe_backend_runtime(params: dict) -> str:
    raw = params.get("__compute_backend__", {})
    if not isinstance(raw, dict):
        return "pandas"
    active = str(raw.get("dataframe_active", "pandas")).strip().lower() or "pandas"
    return active


def _panel_dataframe_backend_runtime(panel: pd.DataFrame) -> str:
    active = "pandas"
    raw_attr = panel.attrs.get("__compute_backend__")
    if isinstance(raw_attr, dict):
        active = str(raw_attr.get("dataframe_active", active)).strip().lower() or active
    if "__compute_backend__" in panel.columns and not panel.empty:
        raw_col = panel["__compute_backend__"].iloc[0]
        if isinstance(raw_col, dict):
            active = str(raw_col.get("dataframe_active", active)).strip().lower() or active
    return active


def _try_sort_frame_with_cudf(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        import cudf  # type: ignore
    except Exception as exc:
        raise RuntimeError("cudf is not available") from exc
    try:
        gdf = cudf.from_pandas(frame)
        gdf = gdf.sort_values(["dt", "code", "seq"])
        out = gdf.to_pandas()
        out = out.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
        return out
    except Exception as exc:
        raise RuntimeError("cudf frame sort failed") from exc


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


def _build_rebuilt_bar_frame_pandas(
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


def _build_rebuilt_bar_frame_torch(
    base_frame: pd.DataFrame,
    *,
    window_points: int,
    torch_device: str,
) -> pd.DataFrame:
    if torch is None:
        raise RuntimeError("torch is not available")
    if not torch.cuda.is_available():
        raise RuntimeError("torch cuda is not available")

    w = max(1, int(window_points))
    if base_frame.empty:
        return base_frame.loc[:, ["dt", "code", "seq"]].copy(deep=False)

    frame = base_frame.copy()
    row_no = frame.groupby(["dt", "code"], sort=False).cumcount()
    frame["bar_seq"] = (row_no // w).astype("int64")

    ohlc_rows: list[tuple[pd.Timestamp, str, int, float, float, float, float]] = []
    for (dt, code), g in frame.groupby(["dt", "code"], sort=False):
        arr = pd.to_numeric(g["last"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        n = int(arr.size)
        if n <= 0:
            continue
        px = torch.as_tensor(arr, dtype=torch.float64, device=torch_device)
        n_bars = (n + w - 1) // w
        pad = n_bars * w - n
        if pad > 0:
            px_padded = torch.cat(
                [px, torch.full((pad,), float("nan"), dtype=torch.float64, device=torch_device)]
            )
        else:
            px_padded = px
        px_mat = px_padded.view(n_bars, w)
        open_v = px_mat[:, 0]
        high_v = torch.nanmax(px_mat, dim=1).values
        low_v = torch.nanmin(px_mat, dim=1).values

        start_idx = torch.arange(0, n_bars * w, w, device=torch_device, dtype=torch.long)
        end_idx = torch.clamp(start_idx + (w - 1), max=n - 1)
        close_v = px[end_idx]

        open_np = open_v.detach().cpu().numpy()
        high_np = high_v.detach().cpu().numpy()
        low_np = low_v.detach().cpu().numpy()
        close_np = close_v.detach().cpu().numpy()
        for i in range(n_bars):
            ohlc_rows.append(
                (
                    pd.Timestamp(dt),
                    str(code),
                    int(i),
                    float(open_np[i]),
                    float(high_np[i]),
                    float(low_np[i]),
                    float(close_np[i]),
                )
            )

    if not ohlc_rows:
        return frame.loc[:, ["dt", "code", "seq"]].copy(deep=False)

    bars = pd.DataFrame(
        ohlc_rows,
        columns=["dt", "code", "bar_seq", "open", "high", "low", "close"],
    ).set_index(["dt", "code", "bar_seq"])
    bars["last"] = bars["close"]

    if "trade_time" in frame.columns:
        bars["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce").groupby(
            [frame["dt"], frame["code"], frame["bar_seq"]],
            sort=False,
        ).last().reindex(bars.index)

    for col in REBUILD_VALUE_COLS:
        if col not in frame.columns:
            continue
        cum = pd.to_numeric(frame[col], errors="coerce")
        delta = cum.groupby([frame["dt"], frame["code"]], sort=False).diff()
        delta = delta.where(delta.notna(), cum)
        delta = delta.where(delta >= 0.0, cum)
        delta = delta.fillna(0.0)
        bars[col] = (
            delta.groupby([frame["dt"], frame["code"], frame["bar_seq"]], sort=False)
            .sum(min_count=1)
            .fillna(0.0)
            .reindex(bars.index)
        )

    carry_cols = [
        c
        for c in frame.columns
        if c not in {"dt", "code", "seq", "bar_seq"}
        and c not in bars.columns
        and c not in REBUILD_VALUE_COLS
    ]
    for col in carry_cols:
        src = frame[col]
        if pd.api.types.is_numeric_dtype(src):
            bars[col] = pd.to_numeric(src, errors="coerce").groupby(
                [frame["dt"], frame["code"], frame["bar_seq"]],
                sort=False,
            ).last().reindex(bars.index)
        else:
            bars[col] = src.groupby(
                [frame["dt"], frame["code"], frame["bar_seq"]],
                sort=False,
            ).last().reindex(bars.index)

    prev_bar_close = bars.groupby(level=["dt", "code"], sort=False)["close"].shift(1)
    if "pre_close" in frame.columns:
        pre_close_day = pd.to_numeric(frame["pre_close"], errors="coerce").groupby(
            [frame["dt"], frame["code"]],
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


def _build_rebuilt_bar_frame_cudf(
    base_frame: pd.DataFrame,
    *,
    window_points: int,
) -> pd.DataFrame:
    sorted_frame = _try_sort_frame_with_cudf(base_frame)
    return _build_rebuilt_bar_frame_pandas(
        sorted_frame,
        window_points=window_points,
    )


def _build_rebuilt_bar_frame(
    base_frame: pd.DataFrame,
    *,
    window_points: int,
    backend_mode: str = "cpu",
    torch_device: str = "cuda",
    dataframe_backend: str = "pandas",
) -> pd.DataFrame:
    if backend_mode == "torch_cuda":
        try:
            return _build_rebuilt_bar_frame_torch(
                base_frame,
                window_points=window_points,
                torch_device=torch_device,
            )
        except Exception:
            pass
    if str(dataframe_backend).strip().lower() == "cudf":
        try:
            return _build_rebuilt_bar_frame_cudf(
                base_frame,
                window_points=window_points,
            )
        except Exception:
            pass
    return _build_rebuilt_bar_frame_pandas(
        base_frame,
        window_points=window_points,
    )


def ensure_trade_time(panel: pd.DataFrame) -> pd.DataFrame:
    if "trade_time" not in panel.columns:
        raise KeyError("panel missing column: trade_time")
    if not pd.api.types.is_datetime64_any_dtype(panel["trade_time"]):
        panel = panel.copy()
        panel["trade_time"] = pd.to_datetime(panel["trade_time"], errors="coerce")
    return ensure_panel_index(panel)


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
        dataframe_backend = _dataframe_backend_runtime(ctx.params)
        built = panel.reset_index()
        if dataframe_backend == "cudf":
            try:
                built = _try_sort_frame_with_cudf(built)
            except Exception:
                built = built.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
        else:
            built = built.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
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
        backend_mode, torch_device = _compute_backend_runtime(ctx.params)
        dataframe_backend = _dataframe_backend_runtime(ctx.params)
        to_build: list[int] = []
        with ctx.cache_lock:
            for w in rebuild_windows:
                if w not in rebuild_frames:
                    to_build.append(w)
        for w in to_build:
            built = _build_rebuilt_bar_frame(
                base_frame,
                window_points=w,
                backend_mode=backend_mode,
                torch_device=torch_device,
                dataframe_backend=dataframe_backend,
            )
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
    backend_mode, torch_device = _compute_backend_runtime(ctx.params)
    frame["__compute_backend__"] = {
        "active": backend_mode,
        "torch_device": torch_device,
        "debug_log_each_record": bool(
            dict(ctx.params.get("__compute_backend__", {})).get("debug_log_each_record", False)
        ),
    }
    frame["__factor_spec_name__"] = str(ctx.params.get("__factor_spec_name__", "")).strip()
    frame["__factor_kernel_name__"] = str(ctx.params.get("__factor_kernel_name__", "")).strip()
    frame["__factor_kernel_params__"] = (
        dict(ctx.params.get("__factor_kernel_params__", {}))
        if isinstance(ctx.params.get("__factor_kernel_params__", {}), dict)
        else {}
    )
    frame.attrs["__compute_backend__"] = frame["__compute_backend__"].iloc[0]
    frame.attrs["__factor_spec_name__"] = frame["__factor_spec_name__"].iloc[0]
    frame.attrs["__factor_kernel_name__"] = frame["__factor_kernel_name__"].iloc[0]
    return frame


TorchKernelFn = Callable[[pd.DataFrame, dict[str, Any], str], float]
_TORCH_SCALAR_KERNELS: dict[str, TorchKernelFn] = {}


def _register_torch_scalar_kernel(name: str) -> Callable[[TorchKernelFn], TorchKernelFn]:
    def _decorator(fn: TorchKernelFn) -> TorchKernelFn:
        _TORCH_SCALAR_KERNELS[str(name).strip()] = fn
        return fn

    return _decorator


def _torch_tensor_from_series(series: pd.Series, *, device: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is not available")
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def _torch_device_available(device: str) -> bool:
    dev = str(device or "").strip().lower()
    if dev.startswith("cpu"):
        return torch is not None
    return torch is not None and torch.cuda.is_available()


def _torch_scalar_to_float(x: Any) -> float:
    if torch is None:
        return float(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        if x.ndim == 0:
            return float(x.item())
        return float(x.reshape(-1)[-1].item())
    return float(x)


def _torch_open_like_from_group(g: pd.DataFrame, *, device: str) -> Any:
    if OPEN_LIKE_CACHE_COL in g.columns:
        return _torch_tensor_from_series(g[OPEN_LIKE_CACHE_COL], device=device)
    if "open" in g.columns:
        open_t = _torch_tensor_from_series(g["open"], device=device)
    else:
        open_t = None
    ask_t = _torch_tensor_from_series(g["ask_price1"], device=device) if "ask_price1" in g.columns else None
    bid_t = _torch_tensor_from_series(g["bid_price1"], device=device) if "bid_price1" in g.columns else None
    if ask_t is not None and bid_t is not None:
        mid = (ask_t + bid_t) / 2.0
        if open_t is not None:
            return torch.where(torch.isnan(mid), open_t, mid)
        return mid
    if open_t is not None:
        return open_t
    raise KeyError("open-like requires open or ask/bid columns")


def _torch_rank_pct(x: Any) -> Any:
    if torch is None:
        raise RuntimeError("torch is not available")
    out = torch.full_like(x, float("nan"))
    valid = ~torch.isnan(x)
    if int(valid.sum().item()) <= 0:
        return out
    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(1)
    xv = x[idx_valid]
    sorted_vals, order = torch.sort(xv)
    m = int(sorted_vals.numel())
    ranks_sorted = torch.empty_like(sorted_vals)
    i = 0
    while i < m:
        j = i + 1
        while j < m and bool(sorted_vals[j] == sorted_vals[i]):
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        ranks_sorted[i:j] = avg_rank
        i = j
    ranks = torch.empty_like(xv)
    ranks[order] = ranks_sorted / float(max(1, m))
    out[idx_valid] = ranks
    return out


def _torch_corr_last(x: Any, y: Any, window: int) -> float:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(2, int(window))
    pair = torch.stack([x, y], dim=1)
    valid = ~torch.isnan(pair).any(dim=1)
    pair = pair[valid]
    if int(pair.shape[0]) < 2:
        return 0.0
    pair = pair[-w:]
    a = pair[:, 0]
    b = pair[:, 1]
    a_std = torch.std(a, unbiased=False)
    b_std = torch.std(b, unbiased=False)
    if float(a_std.item()) <= EPS or float(b_std.item()) <= EPS:
        return 0.0
    am = torch.mean(a)
    bm = torch.mean(b)
    cov = torch.mean((a - am) * (b - bm))
    corr = cov / (a_std * b_std + EPS)
    if torch.isnan(corr):
        return 0.0
    return float(corr.item())


def _torch_ts_rank_last(x: Any, window: int) -> float:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(1, int(window))
    tail = x[-w:]
    tail = tail[~torch.isnan(tail)]
    if int(tail.numel()) <= 0:
        return 0.0
    ranks = _torch_rank_pct(tail)
    return float(ranks[-1].item())


def _torch_rolling_mean_last(x: Any, window: int) -> float:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(1, int(window))
    tail = x[-w:]
    if int(tail.numel()) <= 0:
        return float("nan")
    valid = tail[~torch.isnan(tail)]
    if int(valid.numel()) <= 0:
        return float("nan")
    return float(torch.mean(valid).item())


def _torch_rolling_min_last(x: Any, window: int) -> float:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(1, int(window))
    tail = x[-w:]
    valid = tail[~torch.isnan(tail)]
    if int(valid.numel()) <= 0:
        return float("nan")
    return float(torch.min(valid).item())


def _torch_rolling_max_last(x: Any, window: int) -> float:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(1, int(window))
    tail = x[-w:]
    valid = tail[~torch.isnan(tail)]
    if int(valid.numel()) <= 0:
        return float("nan")
    return float(torch.max(valid).item())


def _torch_rolling_std_series(x: Any, window: int, *, min_periods: int = 2) -> Any:
    if torch is None:
        raise RuntimeError("torch is not available")
    w = max(1, int(window))
    out = torch.full_like(x, float("nan"))
    n = int(x.numel())
    for i in range(n):
        left = max(0, i - w + 1)
        seg = x[left : i + 1]
        valid = seg[~torch.isnan(seg)]
        if int(valid.numel()) < int(min_periods):
            continue
        out[i] = torch.std(valid, unbiased=True)
    return out


def _to_torch_tensor(value: Any, *, device: str, dtype: Any | None = None, like: Any | None = None) -> Any:
    if torch is None:
        raise RuntimeError("torch is not available")
    if isinstance(value, _TorchSeries):
        t = value.tensor
    elif isinstance(value, torch.Tensor):
        t = value
    elif np.isscalar(value):
        if like is not None:
            base = like.tensor if isinstance(like, _TorchSeries) else like
            t = torch.full_like(base, float(value), dtype=dtype or base.dtype, device=device)
        else:
            t = torch.tensor(float(value), dtype=dtype or torch.float64, device=device)
    else:
        arr = np.asarray(value)
        t = torch.as_tensor(arr, dtype=dtype or torch.float64, device=device)
    if dtype is not None and t.dtype != dtype:
        t = t.to(dtype=dtype)
    if t.device.type != torch.device(device).type:
        t = t.to(device=device)
    return t


class _TorchILoc:
    def __init__(self, series: "_TorchSeries") -> None:
        self._s = series

    def __getitem__(self, key: int) -> float:
        t = self._s.tensor
        if t.numel() == 0:
            return float("nan")
        idx = int(key)
        if idx < 0:
            idx = int(t.numel()) + idx
        idx = min(max(0, idx), int(t.numel()) - 1)
        return float(t[idx].item())


class _TorchRolling:
    def __init__(self, series: "_TorchSeries", window: int, min_periods: int = 1) -> None:
        self._s = series
        self._w = max(1, int(window))
        self._minp = max(1, int(min_periods))

    def _apply(self, op: str) -> "_TorchSeries":
        x = self._s.tensor
        n = int(x.numel())
        out = torch.full_like(x, float("nan"))
        for i in range(n):
            left = max(0, i - self._w + 1)
            seg = x[left : i + 1]
            valid = seg[~torch.isnan(seg)]
            if int(valid.numel()) < self._minp:
                continue
            if op == "mean":
                out[i] = torch.mean(valid)
            elif op == "sum":
                out[i] = torch.sum(valid)
            elif op == "min":
                out[i] = torch.min(valid)
            elif op == "max":
                out[i] = torch.max(valid)
            elif op == "std":
                if int(valid.numel()) < 2:
                    continue
                out[i] = torch.std(valid, unbiased=True)
            else:
                raise ValueError(f"unknown rolling op: {op}")
        return _TorchSeries(out, self._s.device)

    def mean(self) -> "_TorchSeries":
        return self._apply("mean")

    def sum(self) -> "_TorchSeries":
        return self._apply("sum")

    def min(self) -> "_TorchSeries":
        return self._apply("min")

    def max(self) -> "_TorchSeries":
        return self._apply("max")

    def std(self) -> "_TorchSeries":
        return self._apply("std")


class _TorchSeries:
    __array_priority__ = 1000

    def __init__(self, tensor: Any, device: str) -> None:
        if torch is None:
            raise RuntimeError("torch is not available")
        if not isinstance(tensor, torch.Tensor):
            tensor = _to_torch_tensor(tensor, device=device)
        self.tensor = tensor
        self.device = str(device)

    @property
    def iloc(self) -> _TorchILoc:
        return _TorchILoc(self)

    def __len__(self) -> int:
        return int(self.tensor.numel())

    def __iter__(self):
        arr = self.tensor.detach().cpu().numpy().reshape(-1)
        for v in arr:
            yield float(v)

    def _binary(self, other: Any, op: str) -> "_TorchSeries":
        t = self.tensor
        o = _to_torch_tensor(other, device=self.device, like=self)
        if op == "add":
            out = t + o
        elif op == "sub":
            out = t - o
        elif op == "rsub":
            out = o - t
        elif op == "mul":
            out = t * o
        elif op == "div":
            out = t / o
        elif op == "rdiv":
            out = o / t
        elif op == "pow":
            out = torch.pow(t, o)
        elif op == "rpow":
            out = torch.pow(o, t)
        elif op == "lt":
            out = t < o
        elif op == "le":
            out = t <= o
        elif op == "gt":
            out = t > o
        elif op == "ge":
            out = t >= o
        elif op == "eq":
            out = t == o
        elif op == "ne":
            out = t != o
        else:
            raise ValueError(f"unknown op: {op}")
        return _TorchSeries(out, self.device)

    def __add__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "add")

    def __radd__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "add")

    def __sub__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "sub")

    def __rsub__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "rsub")

    def __mul__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "mul")

    def __rmul__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "mul")

    def __truediv__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "div")

    def __rtruediv__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "rdiv")

    def __pow__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "pow")

    def __rpow__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "rpow")

    def __lt__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "lt")

    def __le__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "le")

    def __gt__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "gt")

    def __ge__(self, other: Any) -> "_TorchSeries":
        return self._binary(other, "ge")

    def __eq__(self, other: Any) -> "_TorchSeries":  # type: ignore[override]
        return self._binary(other, "eq")

    def __ne__(self, other: Any) -> "_TorchSeries":  # type: ignore[override]
        return self._binary(other, "ne")

    def __neg__(self) -> "_TorchSeries":
        return _TorchSeries(-self.tensor, self.device)

    def __abs__(self) -> "_TorchSeries":
        return self.abs()

    def abs(self) -> "_TorchSeries":
        return _TorchSeries(torch.abs(self.tensor), self.device)

    def astype(self, _dtype: str) -> "_TorchSeries":
        return _TorchSeries(self.tensor.to(dtype=torch.float64), self.device)

    def clip(self, lower: float | None = None, upper: float | None = None) -> "_TorchSeries":
        t = self.tensor
        if lower is not None:
            t = torch.clamp(t, min=float(lower))
        if upper is not None:
            t = torch.clamp(t, max=float(upper))
        return _TorchSeries(t, self.device)

    def diff(self, periods: int = 1) -> "_TorchSeries":
        p = max(1, int(periods))
        t = self.tensor
        out = torch.full_like(t, float("nan"))
        if int(t.numel()) > p:
            out[p:] = t[p:] - t[:-p]
        return _TorchSeries(out, self.device)

    def rank(self, pct: bool = False, method: str = "average") -> "_TorchSeries":
        if method != "average":
            raise ValueError("torch proxy rank only supports method='average'")
        ranks = _torch_rank_pct(self.tensor)
        if not pct:
            valid = ~torch.isnan(ranks)
            n = int(valid.sum().item())
            if n > 0:
                ranks = ranks * float(n)
        return _TorchSeries(ranks, self.device)

    def rolling(self, window: int, min_periods: int = 1) -> _TorchRolling:
        return _TorchRolling(self, window=window, min_periods=min_periods)

    def fillna(self, value: float) -> "_TorchSeries":
        fill = _to_torch_tensor(value, device=self.device, like=self)
        out = torch.where(torch.isnan(self.tensor), fill, self.tensor)
        return _TorchSeries(out, self.device)

    def shift(self, periods: int = 1) -> "_TorchSeries":
        p = int(periods)
        t = self.tensor
        out = torch.full_like(t, float("nan"))
        n = int(t.numel())
        if p == 0:
            out = t.clone()
        elif p > 0 and n > p:
            out[p:] = t[:-p]
        elif p < 0 and n > (-p):
            out[:p] = t[-p:]
        return _TorchSeries(out, self.device)

    def tail(self, n: int) -> "_TorchSeries":
        n = max(0, int(n))
        if n == 0:
            return _TorchSeries(self.tensor[:0], self.device)
        return _TorchSeries(self.tensor[-n:], self.device)

    def dropna(self) -> "_TorchSeries":
        valid = ~torch.isnan(self.tensor)
        return _TorchSeries(self.tensor[valid], self.device)

    def where(self, cond: Any, other: Any) -> "_TorchSeries":
        c = _to_torch_tensor(cond, device=self.device, dtype=torch.bool, like=self)
        o = _to_torch_tensor(other, device=self.device, like=self)
        out = torch.where(c, self.tensor, o)
        return _TorchSeries(out, self.device)

    def sum(self) -> float:
        valid = self.tensor[~torch.isnan(self.tensor)]
        if int(valid.numel()) <= 0:
            return float("nan")
        return float(torch.sum(valid).item())

    def mean(self) -> float:
        valid = self.tensor[~torch.isnan(self.tensor)]
        if int(valid.numel()) <= 0:
            return float("nan")
        return float(torch.mean(valid).item())

    def std(self, ddof: int = 0) -> float:
        valid = self.tensor[~torch.isnan(self.tensor)]
        if int(valid.numel()) <= int(ddof):
            return float("nan")
        return float(torch.std(valid, unbiased=bool(ddof)).item())

    def min(self) -> float:
        valid = self.tensor[~torch.isnan(self.tensor)]
        if int(valid.numel()) <= 0:
            return float("nan")
        return float(torch.min(valid).item())

    def max(self) -> float:
        valid = self.tensor[~torch.isnan(self.tensor)]
        if int(valid.numel()) <= 0:
            return float("nan")
        return float(torch.max(valid).item())

    def corr(self, other: Any, method: str = "pearson") -> float:
        y = _to_torch_tensor(other, device=self.device, like=self)
        x = self.tensor
        pair = torch.stack([x, y], dim=1)
        valid = ~torch.isnan(pair).any(dim=1)
        pair = pair[valid]
        if int(pair.shape[0]) < 2:
            return float("nan")
        a = pair[:, 0]
        b = pair[:, 1]
        if method == "spearman":
            a = _torch_rank_pct(a)
            b = _torch_rank_pct(b)
            pair2 = torch.stack([a, b], dim=1)
            valid2 = ~torch.isnan(pair2).any(dim=1)
            pair2 = pair2[valid2]
            if int(pair2.shape[0]) < 2:
                return float("nan")
            a = pair2[:, 0]
            b = pair2[:, 1]
        a_std = torch.std(a, unbiased=False)
        b_std = torch.std(b, unbiased=False)
        if float(a_std.item()) <= EPS or float(b_std.item()) <= EPS:
            return float("nan")
        am = torch.mean(a)
        bm = torch.mean(b)
        cov = torch.mean((a - am) * (b - bm))
        r = cov / (a_std * b_std + EPS)
        if torch.isnan(r):
            return float("nan")
        return float(r.item())

    def cov(self, other: Any) -> float:
        y = _to_torch_tensor(other, device=self.device, like=self)
        x = self.tensor
        pair = torch.stack([x, y], dim=1)
        valid = ~torch.isnan(pair).any(dim=1)
        pair = pair[valid]
        if int(pair.shape[0]) < 2:
            return float("nan")
        a = pair[:, 0]
        b = pair[:, 1]
        am = torch.mean(a)
        bm = torch.mean(b)
        denom = max(1, int(a.numel()) - 1)
        cov = torch.sum((a - am) * (b - bm)) / float(denom)
        return float(cov.item())

    def to_numpy(self) -> np.ndarray:
        return self.tensor.detach().cpu().numpy()

    @property
    def values(self) -> np.ndarray:
        return self.to_numpy()

    def __array_function__(self, func, types, args, kwargs):
        if func is np.where:
            cond, x, y = args
            if isinstance(x, _TorchSeries):
                dev = x.device
                like = x
            elif isinstance(y, _TorchSeries):
                dev = y.device
                like = y
            else:
                dev = self.device
                like = self
            c = _to_torch_tensor(cond, device=dev, dtype=torch.bool, like=like)
            tx = _to_torch_tensor(x, device=dev, like=like)
            ty = _to_torch_tensor(y, device=dev, like=like)
            return _TorchSeries(torch.where(c, tx, ty), dev)
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        tensors = []
        device = self.device
        like: _TorchSeries | None = None
        for inp in inputs:
            if isinstance(inp, _TorchSeries):
                like = inp
                device = inp.device
                break
        for inp in inputs:
            tensors.append(_to_torch_tensor(inp, device=device, like=like or self))
        out = getattr(torch, ufunc.__name__)(*tensors, **kwargs)
        return _TorchSeries(out, device)


class _TorchGroupProxy:
    def __init__(self, group_df: pd.DataFrame, *, device: str) -> None:
        self._g = group_df
        self._device = device

    @property
    def empty(self) -> bool:
        return bool(self._g.empty)

    @property
    def columns(self):
        return self._g.columns

    def __len__(self) -> int:
        return int(len(self._g))

    def __getitem__(self, key: str) -> _TorchSeries:
        if key not in self._g.columns:
            raise KeyError(key)
        return _TorchSeries(_torch_tensor_from_series(self._g[key], device=self._device), self._device)

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._g.columns:
            return self[key]
        return default


@_register_torch_scalar_kernel("alpha001_signed_power_v1")
def _kernel_alpha001(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    stddev_window = int(params.get("stddev_window", 20))
    ts_max_window = int(params.get("ts_max_window", 5))
    last_px = _torch_tensor_from_series(g["last"], device=device)
    pre_close = _torch_tensor_from_series(g["prev_bar_close"], device=device)
    ret = (last_px - pre_close) / (pre_close + EPS)
    std_ret = _torch_rolling_std_series(ret, max(2, stddev_window), min_periods=2)
    std_ret = torch.nan_to_num(std_ret, nan=0.0)
    base = torch.where(ret < 0.0, std_ret, last_px)
    sp = torch.sign(base) * torch.pow(torch.abs(base), 2.0)
    n = int(sp.numel())
    if n <= 0:
        return 0.0
    out = torch.full_like(sp, float("nan"))
    w = max(1, int(ts_max_window))
    for i in range(n):
        left = max(0, i - w + 1)
        seg = sp[left : i + 1]
        valid = seg[~torch.isnan(seg)]
        if int(valid.numel()) <= 0:
            continue
        out[i] = torch.max(valid)
    val = out[-1]
    if torch.isnan(val):
        return 0.0
    return float(val.item())


@_register_torch_scalar_kernel("alpha002_corr_volume_return_v1")
def _kernel_alpha002(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    corr_window = int(params.get("corr_window", 6))
    volume = torch.clamp(_torch_tensor_from_series(g["volume"], device=device), min=0.0)
    last_px = _torch_tensor_from_series(g["last"], device=device)
    open_like = _torch_open_like_from_group(g, device=device)
    log_volume = torch.log(volume + EPS)
    delta = torch.full_like(log_volume, float("nan"))
    if int(log_volume.numel()) > 2:
        delta[2:] = log_volume[2:] - log_volume[:-2]
    ret = (last_px - open_like) / (open_like + EPS)
    x = _torch_rank_pct(delta)
    y = _torch_rank_pct(ret)
    return -_torch_corr_last(x, y, corr_window)


@_register_torch_scalar_kernel("alpha003_corr_open_volume_v1")
def _kernel_alpha003(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    corr_window = int(params.get("corr_window", 10))
    open_like = _torch_open_like_from_group(g, device=device)
    volume = _torch_tensor_from_series(g["volume"], device=device)
    open_rank = _torch_rank_pct(open_like)
    vol_rank = _torch_rank_pct(volume)
    return -_torch_corr_last(open_rank, vol_rank, corr_window)


@_register_torch_scalar_kernel("alpha004_ts_rank_low_v1")
def _kernel_alpha004(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    ts_rank_window = int(params.get("ts_rank_window", 9))
    low = _torch_tensor_from_series(g["low"], device=device)
    low_rank = _torch_rank_pct(low)
    return -_torch_ts_rank_last(low_rank, ts_rank_window)


@_register_torch_scalar_kernel("alpha005_vwap_gap_v1_gap1")
def _kernel_alpha005_gap1(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    vwap_window = int(params.get("vwap_window", 10))
    amount = _torch_tensor_from_series(g["amount"], device=device)
    volume = _torch_tensor_from_series(g["volume"], device=device)
    open_like = _torch_open_like_from_group(g, device=device)
    vwap = amount / (volume + EPS)
    avg_vwap_last = _torch_rolling_mean_last(vwap, vwap_window)
    open_last = float(open_like[-1].item()) if int(open_like.numel()) else float("nan")
    return float(open_last - avg_vwap_last)


@_register_torch_scalar_kernel("alpha005_vwap_gap_v1_gap2")
def _kernel_alpha005_gap2(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    amount = _torch_tensor_from_series(g["amount"], device=device)
    volume = _torch_tensor_from_series(g["volume"], device=device)
    last_px = _torch_tensor_from_series(g["last"], device=device)
    if int(last_px.numel()) <= 0:
        return 0.0
    vwap = amount / (volume + EPS)
    return float((last_px[-1] - vwap[-1]).item())


@_register_torch_scalar_kernel("alpha006_corr_open_volume_neg_v1")
def _kernel_alpha006(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    corr_window = int(params.get("corr_window", 10))
    open_like = _torch_open_like_from_group(g, device=device)
    volume = _torch_tensor_from_series(g["volume"], device=device)
    return -_torch_corr_last(open_like, volume, corr_window)


@_register_torch_scalar_kernel("alpha008_open_return_momentum_v1")
def _kernel_alpha008(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    sum_window = int(params.get("sum_window", 5))
    delay_window = int(params.get("delay_window", 10))
    open_like = _torch_open_like_from_group(g, device=device)
    last_px = _torch_tensor_from_series(g["last"], device=device)
    ret = (last_px - open_like) / (open_like + EPS)
    n = int(last_px.numel())
    if n <= 0:
        return 0.0
    sum_open = torch.full_like(open_like, float("nan"))
    sum_ret = torch.full_like(ret, float("nan"))
    w = max(1, int(sum_window))
    for i in range(n):
        left = max(0, i - w + 1)
        seg_o = open_like[left : i + 1]
        seg_r = ret[left : i + 1]
        valid_o = seg_o[~torch.isnan(seg_o)]
        valid_r = seg_r[~torch.isnan(seg_r)]
        if int(valid_o.numel()) > 0:
            sum_open[i] = torch.sum(valid_o)
        if int(valid_r.numel()) > 0:
            sum_ret[i] = torch.sum(valid_r)
    prod = sum_open * sum_ret
    d = max(1, int(delay_window))
    delayed = torch.full_like(prod, float("nan"))
    if n > d:
        delayed[d:] = prod[:-d]
    delay_val = delayed[-1]
    if torch.isnan(delay_val):
        delay_val = prod[0]
    out = prod[-1] - delay_val
    if torch.isnan(out):
        return 0.0
    return float(out.item())


@_register_torch_scalar_kernel("alpha009_close_change_filter_v1")
def _kernel_alpha009(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    ts_window = int(params.get("ts_window", 5))
    last_px = _torch_tensor_from_series(g["last"], device=device)
    n = int(last_px.numel())
    if n <= 0:
        return 0.0
    delta = torch.full_like(last_px, float("nan"))
    if n > 1:
        delta[1:] = last_px[1:] - last_px[:-1]
    d = delta[-1]
    if torch.isnan(d):
        d = torch.tensor(0.0, dtype=last_px.dtype, device=last_px.device)
    ts_min = _torch_rolling_min_last(delta, ts_window)
    ts_max = _torch_rolling_max_last(delta, ts_window)
    if np.isfinite(ts_min) and ts_min > 0.0:
        return float(d.item())
    if np.isfinite(ts_max) and ts_max < 0.0:
        return float(d.item())
    return float((-d).item())


@_register_torch_scalar_kernel("alpha010_close_change_rank_v1")
def _kernel_alpha010(g: pd.DataFrame, params: dict[str, Any], device: str) -> float:
    return _kernel_alpha009(g, params, device)


def _group_scalar_cpu(frame: pd.DataFrame, func: Callable[[pd.DataFrame], float]) -> pd.Series:
    log_enabled, factor_name = _record_log_context(frame)
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for dt, code, g in _iter_dt_code_groups(frame):
        try:
            val = float(func(g))
        except Exception:
            val = np.nan
        rows.append((dt, str(code), val))
        _emit_record_log(
            log_enabled,
            factor_name,
            stage="group_scalar_cpu",
            dt=dt,
            code=code,
            value=val,
            rows=len(g),
        )
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.Series([v for _, _, v in rows], index=idx, dtype="float64")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _group_scalar_torch_kernel(
    frame: pd.DataFrame,
    *,
    kernel_name: str,
    kernel_params: dict[str, Any],
    torch_device: str,
) -> pd.Series:
    log_enabled, factor_name = _record_log_context(frame)
    kernel = _TORCH_SCALAR_KERNELS.get(str(kernel_name).strip())
    if kernel is None:
        raise KeyError(f"unknown torch kernel: {kernel_name}")
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for dt, code, g in _iter_dt_code_groups(frame):
        try:
            val = float(kernel(g, kernel_params, torch_device))
        except Exception:
            val = np.nan
        rows.append((dt, str(code), val))
        _emit_record_log(
            log_enabled,
            factor_name,
            stage="group_scalar_torch_kernel",
            dt=dt,
            code=code,
            value=val,
            rows=len(g),
        )
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.Series([v for _, _, v in rows], index=idx, dtype="float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _group_scalar_torch_proxy(
    frame: pd.DataFrame,
    func: Callable[[pd.DataFrame], float],
    *,
    torch_device: str,
) -> pd.Series:
    log_enabled, factor_name = _record_log_context(frame)
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for dt, code, g in _iter_dt_code_groups(frame):
        proxy = _TorchGroupProxy(g, device=torch_device)
        raw = func(proxy)
        if isinstance(raw, _TorchSeries):
            val = _torch_scalar_to_float(raw.tensor)
        elif torch is not None and isinstance(raw, torch.Tensor):
            val = _torch_scalar_to_float(raw)
        else:
            val = float(raw)
        rows.append((dt, str(code), val))
        _emit_record_log(
            log_enabled,
            factor_name,
            stage="group_scalar_torch_proxy",
            dt=dt,
            code=code,
            value=val,
            rows=len(g),
        )
    if not rows:
        return pd.Series(dtype="float64")
    idx = pd.MultiIndex.from_tuples([(dt, code) for dt, code, _ in rows], names=["dt", "code"])
    out = pd.Series([v for _, _, v in rows], index=idx, dtype="float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _group_scalar(
    frame: pd.DataFrame,
    func: Callable[[pd.DataFrame], float],
    *,
    kernel_name: str | None = None,
    kernel_params: dict[str, Any] | None = None,
) -> pd.Series:
    backend, device = _panel_compute_backend_runtime(frame)
    if backend == "torch_cuda" and kernel_name and _torch_device_available(device):
        try:
            return _group_scalar_torch_kernel(
                frame,
                kernel_name=kernel_name,
                kernel_params=dict(kernel_params or {}),
                torch_device=device,
            )
        except Exception:
            pass
    if backend == "torch_cuda" and _torch_device_available(device):
        auto_name = ""
        auto_params: dict[str, Any] = {}
        if "__factor_kernel_name__" in frame.columns:
            auto_name = str(frame["__factor_kernel_name__"].iloc[0] or "").strip()
        if "__factor_kernel_params__" in frame.columns:
            raw = frame["__factor_kernel_params__"].iloc[0]
            if isinstance(raw, dict):
                auto_params = dict(raw)
        if auto_name:
            try:
                return _group_scalar_torch_kernel(
                    frame,
                    kernel_name=auto_name,
                    kernel_params=auto_params,
                    torch_device=device,
                )
            except Exception:
                pass
    if backend == "torch_cuda" and _torch_device_available(device):
        try:
            return _group_scalar_torch_proxy(
                frame,
                func,
                torch_device=device,
            )
        except Exception:
            pass
    return _group_scalar_cpu(frame, func)


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

