from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context, shared_memory
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from cbond_on.domain.factors.base import ensure_panel_index
from cbond_on.infra.factors.rust_backend import build_factor_frame_rust
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col


@dataclass(frozen=True)
class _ShmArrayMeta:
    name: str
    shape: tuple[int, ...]
    dtype: str


# Linux fork zero-copy path:
# parent prepares one pandas frame, children inherit it via fork (COW),
# avoiding explicit shared_memory publish copy.
_FORK_PANEL_DF: pd.DataFrame | None = None


def _import_rust_module():
    from importlib import import_module

    try:
        return import_module("cbond_on_rust")
    except Exception as exc:
        raise RuntimeError(
            "factor engine=rust_shm_exp but module 'cbond_on_rust' is not available. "
            "Build/install the Rust extension first (see rust/factor_engine/README.md)."
        ) from exc


def _specs_payload(specs: Sequence[FactorSpec]) -> list[dict]:
    payload: list[dict] = []
    for spec in specs:
        payload.append(
            {
                "name": str(spec.name),
                "factor": str(spec.factor),
                "params": dict(spec.params or {}),
                "output_col": spec.output_col,
            }
        )
    return payload


def _call_compute_factor_frame(
    module: object,
    *,
    panel_df: pd.DataFrame,
    specs_payload: list[dict],
    daily_payload: dict[str, pd.DataFrame] | None,
    compute_backend_params: dict,
):
    fn = getattr(module, "compute_factor_frame")
    try:
        return fn(
            panel_df,
            specs_payload,
            None,
            None,
            daily_payload or None,
            compute_backend_params,
        )
    except TypeError as exc:
        if daily_payload:
            raise RuntimeError(
                "rust extension is outdated for daily factor context; rebuild cbond_on_rust first"
            ) from exc
        return fn(
            panel_df,
            specs_payload,
            None,
            None,
            compute_backend_params,
        )


def _chunk_specs(specs: Sequence[FactorSpec], parts: int) -> list[list[FactorSpec]]:
    slots = max(1, min(int(parts), len(specs)))
    buckets: list[list[FactorSpec]] = [[] for _ in range(slots)]
    for idx, spec in enumerate(specs):
        buckets[idx % slots].append(spec)
    return [bucket for bucket in buckets if bucket]


def _resolve_mp_start_method(requested: str) -> str:
    raw = str(requested or "auto").strip().lower()
    methods = set(get_all_start_methods())
    if raw in {"", "auto"}:
        if "fork" in methods:
            return "fork"
        if "forkserver" in methods:
            return "forkserver"
        return "spawn"
    if raw in methods:
        return raw
    if "spawn" in methods:
        return "spawn"
    if methods:
        return sorted(methods)[0]
    return "spawn"


def _fixed_unicode(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype("<U1")
    str_values = values.astype(str, copy=False)
    max_len = max(1, max(len(v) for v in str_values.tolist()))
    return str_values.astype(f"<U{max_len}")


def _share_array(arr: np.ndarray) -> tuple[_ShmArrayMeta, shared_memory.SharedMemory]:
    data = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    view = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    view[:] = data
    meta = _ShmArrayMeta(
        name=shm.name,
        shape=tuple(int(v) for v in data.shape),
        dtype=str(data.dtype),
    )
    return meta, shm


def _attach_array(meta: _ShmArrayMeta) -> tuple[np.ndarray, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=meta.name)
    arr = np.ndarray(meta.shape, dtype=np.dtype(meta.dtype), buffer=shm.buf)
    return arr, shm


def _normalize_panel_for_rust(panel: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_panel_index(panel).reset_index().copy()
    frame = frame.sort_values(["dt", "code", "seq"], kind="mergesort").reset_index(drop=True)
    frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce")
    frame["code"] = frame["code"].astype(str)
    if "trade_time" in frame.columns:
        frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
        frame["__trade_time_ns__"] = frame["trade_time"].astype("int64")
    elif "__trade_time_ns__" not in frame.columns:
        raise RuntimeError("panel missing trade_time/__trade_time_ns__ column for rust_shm_exp")
    frame["seq"] = pd.to_numeric(frame["seq"], errors="coerce").fillna(0).astype("int64")
    return frame


def _build_shm_package(panel_df: pd.DataFrame) -> tuple[dict[str, Any], list[shared_memory.SharedMemory]]:
    owned: list[shared_memory.SharedMemory] = []
    arrays: dict[str, _ShmArrayMeta] = {}

    dt_arr = _fixed_unicode(
        panel_df["dt"].dt.strftime("%Y-%m-%d").fillna("").to_numpy(dtype=str, copy=False)
    )
    code_arr = _fixed_unicode(panel_df["code"].fillna("").to_numpy(dtype=str, copy=False))
    seq_arr = panel_df["seq"].to_numpy(dtype=np.int64, copy=False)
    time_ns_arr = pd.to_numeric(panel_df["__trade_time_ns__"], errors="coerce").fillna(0).to_numpy(
        dtype=np.int64, copy=False
    )

    for name, arr in (
        ("dt", dt_arr),
        ("code", code_arr),
        ("seq", seq_arr),
        ("__trade_time_ns__", time_ns_arr),
    ):
        meta, shm = _share_array(arr)
        arrays[name] = meta
        owned.append(shm)

    reserved = {"dt", "code", "seq", "trade_time", "__trade_time_ns__"}
    numeric_cols: list[str] = []
    for col in panel_df.columns:
        if col in reserved:
            continue
        series = panel_df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        numeric_cols.append(col)
        meta, shm = _share_array(series.to_numpy(copy=False))
        arrays[col] = meta
        owned.append(shm)

    return (
        {
            "row_count": int(len(panel_df)),
            "arrays": arrays,
            "numeric_cols": numeric_cols,
        },
        owned,
    )


def _close_shm_refs(refs: list[shared_memory.SharedMemory]) -> None:
    for shm in refs:
        try:
            shm.close()
        except Exception:
            pass


def _cleanup_owned_shm(owned: list[shared_memory.SharedMemory]) -> None:
    for shm in owned:
        try:
            shm.close()
        except Exception:
            pass
        try:
            shm.unlink()
        except Exception:
            pass


def _rust_shm_worker(task: dict[str, Any]) -> pd.DataFrame:
    module = _import_rust_module()
    package = dict(task.get("package") or {})
    arrays_meta = package.get("arrays") or {}
    refs: list[shared_memory.SharedMemory] = []
    arrays: dict[str, np.ndarray] = {}
    for col, meta_dict in arrays_meta.items():
        meta = _ShmArrayMeta(
            name=str(meta_dict["name"]),
            shape=tuple(meta_dict["shape"]),
            dtype=str(meta_dict["dtype"]),
        )
        arr, shm = _attach_array(meta)
        arrays[col] = arr
        refs.append(shm)
    try:
        data: dict[str, Any] = {
            "dt": arrays["dt"],
            "code": arrays["code"],
            "seq": arrays["seq"],
            "__trade_time_ns__": arrays["__trade_time_ns__"],
        }
        for col in package.get("numeric_cols") or []:
            data[col] = arrays[col]
        panel_df = pd.DataFrame(data, copy=False)
        out = _call_compute_factor_frame(
            module,
            panel_df=panel_df,
            specs_payload=list(task.get("spec_payload") or []),
            daily_payload=task.get("daily_payload"),
            compute_backend_params=dict(task.get("compute_backend_params") or {}),
        )
        if not isinstance(out, pd.DataFrame):
            raise RuntimeError("cbond_on_rust.compute_factor_frame must return pandas.DataFrame")
        return out
    finally:
        _close_shm_refs(refs)


def _rust_fork_worker(task: dict[str, Any]) -> pd.DataFrame:
    module = _import_rust_module()
    panel_df = _FORK_PANEL_DF
    if panel_df is None:
        raise RuntimeError("fork_zero_copy panel is not initialized in worker")
    out = _call_compute_factor_frame(
        module,
        panel_df=panel_df,
        specs_payload=list(task.get("spec_payload") or []),
        daily_payload=task.get("daily_payload"),
        compute_backend_params=dict(task.get("compute_backend_params") or {}),
    )
    if not isinstance(out, pd.DataFrame):
        raise RuntimeError("cbond_on_rust.compute_factor_frame must return pandas.DataFrame")
    return out


def build_factor_frame_rust_shm(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
    compute_backend_params: dict | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    if not specs:
        return pd.DataFrame()
    if workers <= 1:
        return build_factor_frame_rust(
            panel,
            specs,
            stock_panel=stock_panel,
            bond_stock_map=bond_stock_map,
            daily_data=daily_data,
            compute_backend_params=compute_backend_params,
        )
    if (stock_panel is not None and not stock_panel.empty) or (
        bond_stock_map is not None and not bond_stock_map.empty
    ):
        # Keep experiment scope bounded to cbond-only path to reduce debug surface.
        return build_factor_frame_rust(
            panel,
            specs,
            stock_panel=stock_panel,
            bond_stock_map=bond_stock_map,
            daily_data=daily_data,
            compute_backend_params=compute_backend_params,
        )

    backend_cfg = {}
    if isinstance(compute_backend_params, dict):
        raw = compute_backend_params.get("__compute_backend__")
        if isinstance(raw, dict):
            backend_cfg = raw
    mp_start_method = _resolve_mp_start_method(
        str(backend_cfg.get("shm_mp_start_method", "auto") or "auto")
    )
    fallback_to_rust = bool(backend_cfg.get("shm_fallback_to_rust", True))
    fork_zero_copy = bool(backend_cfg.get("shm_fork_zero_copy", True))

    panel_df = _normalize_panel_for_rust(panel)

    daily_payload: dict[str, pd.DataFrame] = {}
    for source, raw_df in dict(daily_data or {}).items():
        if raw_df is None or raw_df.empty:
            continue
        df = raw_df.copy()
        if "trade_date" not in df.columns or "code" not in df.columns:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["code"] = df["code"].astype(str)
        numeric_cols: list[str] = []
        for col in df.columns:
            if col in {"trade_date", "code"}:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                df[col] = series.astype("float64")
                numeric_cols.append(col)
        if not numeric_cols:
            continue
        keep_cols = ["trade_date", "code", *numeric_cols]
        daily_payload[str(source)] = df[keep_cols].reset_index(drop=True)

    chunks = _chunk_specs(specs, workers)
    use_fork_zero_copy = mp_start_method == "fork" and fork_zero_copy

    t_share = perf_counter()
    tasks: list[dict[str, Any]]
    owned: list[shared_memory.SharedMemory] = []
    worker_fn = _rust_shm_worker
    if use_fork_zero_copy:
        tasks = [
            {
                "spec_payload": _specs_payload(chunk),
                "daily_payload": daily_payload or None,
                "compute_backend_params": dict(compute_backend_params or {}),
            }
            for chunk in chunks
        ]
        t_share = perf_counter() - t_share
        print(
            "rust_shm_exp:",
            f"mp_start_method={mp_start_method}",
            "fork_zero_copy=True",
            "shared_arrays=0",
            f"rows={len(panel_df)}",
            f"t_shm_publish={t_share:.2f}s",
            flush=True,
        )
        worker_fn = _rust_fork_worker
    else:
        package, owned = _build_shm_package(panel_df)
        tasks = [
            {
                "package": {
                    "row_count": package["row_count"],
                    "numeric_cols": list(package["numeric_cols"]),
                    "arrays": {
                        col: {
                            "name": meta.name,
                            "shape": list(meta.shape),
                            "dtype": meta.dtype,
                        }
                        for col, meta in package["arrays"].items()
                    },
                },
                "spec_payload": _specs_payload(chunk),
                "daily_payload": daily_payload or None,
                "compute_backend_params": dict(compute_backend_params or {}),
            }
            for chunk in chunks
        ]
        t_share = perf_counter() - t_share
        print(
            "rust_shm_exp:",
            f"mp_start_method={mp_start_method}",
            "fork_zero_copy=False",
            f"shared_arrays={len(package['arrays'])}",
            f"rows={package['row_count']}",
            f"t_shm_publish={t_share:.2f}s",
            flush=True,
        )

    ctx = get_context(mp_start_method)
    t_compute = perf_counter()
    out_frames: list[pd.DataFrame] = []
    global _FORK_PANEL_DF
    if use_fork_zero_copy:
        _FORK_PANEL_DF = panel_df
    try:
        try:
            with ProcessPoolExecutor(max_workers=len(tasks), mp_context=ctx) as executor:
                future_map = {
                    executor.submit(worker_fn, task): idx for idx, task in enumerate(tasks)
                }
                ordered: dict[int, pd.DataFrame] = {}
                for future in as_completed(future_map):
                    idx = future_map[future]
                    ordered[idx] = future.result()
                for idx in range(len(tasks)):
                    out_frames.append(ordered[idx])
        except Exception as exc:
            if not fallback_to_rust:
                raise
            print(
                "rust_shm_exp:",
                f"fallback_to_rust reason={type(exc).__name__}:{exc}",
                flush=True,
            )
            return build_factor_frame_rust(
                panel,
                specs,
                stock_panel=stock_panel,
                bond_stock_map=bond_stock_map,
                daily_data=daily_data,
                compute_backend_params=compute_backend_params,
            )
    finally:
        _FORK_PANEL_DF = None
        if owned:
            _cleanup_owned_shm(owned)
    t_compute = perf_counter() - t_compute

    t_merge = perf_counter()
    merged: pd.DataFrame | None = None
    for frame in out_frames:
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["dt", "code"], how="outer", sort=False)
    if merged is None:
        return pd.DataFrame()
    merged = merged.copy()
    merged["dt"] = pd.to_datetime(merged["dt"], errors="coerce")
    merged["code"] = merged["code"].astype(str)
    merged = merged.set_index(["dt", "code"]).sort_index()
    t_merge = perf_counter() - t_merge
    print(
        "rust_shm_exp:",
        f"worker_groups={len(tasks)}",
        f"t_mp_compute={t_compute:.2f}s",
        f"t_merge={t_merge:.2f}s",
        flush=True,
    )

    cols = [build_factor_col(spec) for spec in specs]
    missing = [c for c in cols if c not in merged.columns]
    if missing:
        raise RuntimeError(f"cbond_on_rust shm output missing factor columns: {missing}")
    return merged[cols]


