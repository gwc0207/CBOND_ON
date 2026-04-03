from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.services.model.adapters import build_adapter


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


def _apply_execution_threading(execution_cfg: dict) -> None:
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


def run(
    *,
    model_id: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    label_cutoff: str | date | None = None,
    cfg: dict | None = None,
) -> dict:
    score_cfg = dict(cfg or load_config_file("model_score"))
    model_id = str(model_id or score_cfg.get("model_id") or score_cfg.get("default_model_id", "")).strip()
    if not model_id:
        raise ValueError("model_score config missing model_id/default_model_id")

    models = score_cfg.get("models", {})
    if model_id not in models:
        raise KeyError(f"model_id not configured: {model_id}")
    model_entry = dict(models[model_id])
    model_type = str(model_entry["model_type"])
    model_config_key = str(model_entry.get("model_config", "")).strip()
    if not model_config_key:
        raise ValueError(f"model entry missing model_config: {model_id}")

    model_config_path = resolve_config_path(model_config_key)
    model_cfg = load_json_like(model_config_path)
    paths_cfg = load_config_file("paths")

    start_day = parse_date(start or score_cfg.get("start") or model_cfg.get("start"))
    end_day = parse_date(end or score_cfg.get("end") or model_cfg.get("end"))
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    execution_cfg = score_cfg.get("execution")
    if execution_cfg is None:
        execution_cfg = {}
    if not isinstance(execution_cfg, dict):
        raise ValueError("model_score.execution must be an object")
    execution_cfg = dict(execution_cfg)
    _apply_execution_threading(execution_cfg)

    adapter = build_adapter(model_type, model_config_path=model_config_path)
    artifact = adapter.fit(
        start=str(start_day),
        end=str(end_day),
        label_cutoff=str(cutoff_day) if cutoff_day else None,
        execution=execution_cfg,
    )
    adapter.predict(
        start=str(start_day),
        end=str(end_day),
        artifact=artifact,
        label_cutoff=str(cutoff_day) if cutoff_day else None,
        execution=execution_cfg,
    )
    score_output_resolved = None
    score_output_raw = model_cfg.get("score_output")
    if score_output_raw:
        score_output_resolved = str(
            resolve_output_path(
                score_output_raw,
                default_path=Path(paths_cfg["results_root"]) / "scores" / str(model_cfg.get("model_name", model_id)),
                results_root=paths_cfg["results_root"],
            )
        )

    return {
        "model_id": model_id,
        "model_type": model_type,
        "model_config_path": str(model_config_path),
        "score_output": score_output_resolved,
        "start": start_day,
        "end": end_day,
    }

