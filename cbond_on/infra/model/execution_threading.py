from __future__ import annotations

import os


def _parse_positive_int(value) -> int | None:
    if value is None:
        return None
    try:
        n = int(value)
    except Exception:
        return None
    if n <= 0:
        return None
    return n


def apply_execution_threading(execution_cfg: dict) -> None:
    threading_cfg = dict(execution_cfg.get("threading", {}) or {})
    if not threading_cfg:
        return

    env_map = {
        "omp_num_threads": "OMP_NUM_THREADS",
        "mkl_num_threads": "MKL_NUM_THREADS",
        "openblas_num_threads": "OPENBLAS_NUM_THREADS",
        "numexpr_num_threads": "NUMEXPR_NUM_THREADS",
        "pyarrow_num_threads": "PYARROW_NUM_THREADS",
    }
    applied: dict[str, int] = {}
    for key, env_name in env_map.items():
        n = _parse_positive_int(threading_cfg.get(key))
        if n is None:
            continue
        os.environ[env_name] = str(n)
        applied[key] = n

    pyarrow_cpu_n = _parse_positive_int(threading_cfg.get("pyarrow_num_threads"))
    pyarrow_io_n = _parse_positive_int(threading_cfg.get("pyarrow_io_threads"))
    if pyarrow_cpu_n is not None or pyarrow_io_n is not None:
        try:
            import pyarrow as pa

            if pyarrow_cpu_n is not None and hasattr(pa, "set_cpu_count"):
                pa.set_cpu_count(int(pyarrow_cpu_n))
            if pyarrow_io_n is not None and hasattr(pa, "set_io_thread_count"):
                pa.set_io_thread_count(int(pyarrow_io_n))
            if pyarrow_cpu_n is not None:
                applied["pyarrow_num_threads"] = int(pyarrow_cpu_n)
            if pyarrow_io_n is not None:
                applied["pyarrow_io_threads"] = int(pyarrow_io_n)
        except Exception:
            pass

    numexpr_n = _parse_positive_int(threading_cfg.get("numexpr_num_threads"))
    if numexpr_n is not None:
        try:
            import numexpr as ne

            ne.set_num_threads(int(numexpr_n))
        except Exception:
            pass

    torch_n = _parse_positive_int(threading_cfg.get("torch_num_threads"))
    if torch_n is not None:
        try:
            import torch

            torch.set_num_threads(int(torch_n))
            if hasattr(torch, "set_num_interop_threads"):
                interop_n = _parse_positive_int(threading_cfg.get("torch_num_interop_threads"))
                if interop_n is not None:
                    torch.set_num_interop_threads(int(interop_n))
                    applied["torch_num_interop_threads"] = int(interop_n)
            applied["torch_num_threads"] = int(torch_n)
        except Exception:
            pass

    if applied:
        info = " ".join([f"{k}={v}" for k, v in sorted(applied.items())])
        print(f"[model_score] execution threading applied: {info}")
