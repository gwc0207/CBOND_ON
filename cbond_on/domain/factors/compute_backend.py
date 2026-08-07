from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComputeBackendState:
    requested: str
    active: str
    torch_device: str
    fallback_cpu: bool
    reason: str

    def to_params(self) -> dict[str, Any]:
        return {
            "__compute_backend__": {
                "requested": self.requested,
                "active": self.active,
                "torch_device": self.torch_device,
                "fallback_cpu": self.fallback_cpu,
                "reason": self.reason,
            }
        }


@dataclass(frozen=True)
class DataFrameBackendState:
    requested: str
    active: str
    reason: str

    def to_params(self) -> dict[str, Any]:
        return {
            "__compute_backend__": {
                "dataframe_requested": self.requested,
                "dataframe_active": self.active,
                "dataframe_reason": self.reason,
            }
        }


@dataclass(frozen=True)
class FactorEngineState:
    requested: str
    active: str
    reason: str

    def to_params(self) -> dict[str, Any]:
        return {
            "__compute_backend__": {
                "engine_requested": self.requested,
                "engine_active": self.active,
                "engine_reason": self.reason,
            }
        }


def _parse_requested_backend(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("backend", "cpu")).strip().lower()
    if requested in {"", "none"}:
        return "cpu"
    if requested in {"gpu"}:
        return "auto"
    return requested


def _parse_requested_engine(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("engine", cfg.get("factor_engine", "rust"))).strip().lower()
    if requested in {"", "none"}:
        return "rust"
    if requested == "auto":
        return "rust"
    return requested


def _parse_requested_dataframe_backend(cfg: dict[str, Any]) -> str:
    requested = str(
        cfg.get(
            "dataframe_backend",
            cfg.get("frame_backend", cfg.get("table_backend", "pandas")),
        )
    ).strip().lower()
    if requested in {"", "none", "gpu", "auto", "cudf"}:
        return "pandas"
    return requested


def resolve_compute_backend(cfg: dict[str, Any] | None = None) -> ComputeBackendState:
    runtime = dict(cfg or {})
    requested = _parse_requested_backend(runtime)
    fallback_cpu = bool(runtime.get("fallback_cpu", True))
    torch_device = str(runtime.get("torch_device", "cuda")).strip() or "cuda"

    if requested == "cpu":
        return ComputeBackendState(
            requested=requested,
            active="cpu",
            torch_device=torch_device,
            fallback_cpu=fallback_cpu,
            reason="forced_cpu",
        )

    if requested not in {"auto", "torch"}:
        if not fallback_cpu:
            raise ValueError(f"unsupported compute backend: {requested}")
        return ComputeBackendState(
            requested=requested,
            active="cpu",
            torch_device=torch_device,
            fallback_cpu=fallback_cpu,
            reason=f"unsupported_backend:{requested}",
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        if not fallback_cpu:
            raise RuntimeError(f"torch import failed: {exc}") from exc
        return ComputeBackendState(
            requested=requested,
            active="cpu",
            torch_device=torch_device,
            fallback_cpu=fallback_cpu,
            reason=f"torch_import_failed:{type(exc).__name__}",
        )

    if not torch.cuda.is_available():
        if not fallback_cpu:
            raise RuntimeError("torch cuda is not available and fallback_cpu=false")
        return ComputeBackendState(
            requested=requested,
            active="cpu",
            torch_device=torch_device,
            fallback_cpu=fallback_cpu,
            reason="cuda_unavailable",
        )

    try:
        _ = torch.zeros(1, device=torch_device)
    except Exception as exc:  # pragma: no cover
        if not fallback_cpu:
            raise RuntimeError(f"torch device probe failed: {exc}") from exc
        return ComputeBackendState(
            requested=requested,
            active="cpu",
            torch_device=torch_device,
            fallback_cpu=fallback_cpu,
            reason=f"cuda_probe_failed:{type(exc).__name__}",
        )

    return ComputeBackendState(
        requested=requested,
        active="torch_cuda",
        torch_device=torch_device,
        fallback_cpu=fallback_cpu,
        reason="cuda_ready",
    )


def resolve_factor_engine(cfg: dict[str, Any] | None = None) -> FactorEngineState:
    runtime = dict(cfg or {})
    requested = _parse_requested_engine(runtime)
    if requested == "rust":
        return FactorEngineState(
            requested=requested,
            active=requested,
            reason=f"forced_{requested}",
        )
    if requested in {"rust_shm_exp", "rust_python_hybrid", "python"}:
        raise ValueError(
            f"retired factor engine: {requested}; every executable factor path uses "
            "the single public Rust compute_factor_frame API"
        )
    raise ValueError(
        f"unsupported factor engine: {requested}; expected the public rust engine"
    )


def resolve_dataframe_backend(cfg: dict[str, Any] | None = None) -> DataFrameBackendState:
    runtime = dict(cfg or {})
    requested = _parse_requested_dataframe_backend(runtime)
    return DataFrameBackendState(
        requested=requested,
        active="pandas",
        reason="forced_pandas_only",
    )
