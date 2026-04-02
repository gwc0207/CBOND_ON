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
    fallback_pandas: bool
    reason: str

    def to_params(self) -> dict[str, Any]:
        return {
            "__compute_backend__": {
                "dataframe_requested": self.requested,
                "dataframe_active": self.active,
                "fallback_pandas": self.fallback_pandas,
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
    requested = str(cfg.get("backend", "auto")).strip().lower()
    if requested in {"", "none"}:
        return "cpu"
    if requested in {"gpu"}:
        return "auto"
    return requested


def _parse_requested_engine(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("engine", cfg.get("factor_engine", "python"))).strip().lower()
    if requested in {"", "none"}:
        return "python"
    if requested == "auto":
        return "python"
    return requested


def _parse_requested_dataframe_backend(cfg: dict[str, Any]) -> str:
    requested = str(
        cfg.get(
            "dataframe_backend",
            cfg.get("frame_backend", cfg.get("table_backend", "pandas")),
        )
    ).strip().lower()
    if requested in {"", "none"}:
        return "pandas"
    if requested in {"gpu"}:
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
    if requested in {"python", "rust", "rust_shm_exp"}:
        return FactorEngineState(
            requested=requested,
            active=requested,
            reason=f"forced_{requested}",
        )
    raise ValueError(
        f"unsupported factor engine: {requested}; expected python|rust|rust_shm_exp|auto"
    )


def resolve_dataframe_backend(cfg: dict[str, Any] | None = None) -> DataFrameBackendState:
    runtime = dict(cfg or {})
    requested = _parse_requested_dataframe_backend(runtime)
    fallback_pandas = bool(runtime.get("fallback_pandas", True))

    if requested == "pandas":
        return DataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason="forced_pandas",
        )

    if requested not in {"auto", "cudf"}:
        if not fallback_pandas:
            raise ValueError(f"unsupported dataframe backend: {requested}")
        return DataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"unsupported_backend:{requested}",
        )

    try:
        import cudf  # type: ignore
    except Exception as exc:  # pragma: no cover
        if not fallback_pandas:
            raise RuntimeError(f"cudf import failed: {exc}") from exc
        return DataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"cudf_import_failed:{type(exc).__name__}",
        )

    try:
        probe = cudf.DataFrame({"k": [0, 0], "v": [1.0, 2.0]})
        _ = probe.groupby("k")["v"].sum()
    except Exception as exc:  # pragma: no cover
        if not fallback_pandas:
            raise RuntimeError(f"cudf probe failed: {exc}") from exc
        return DataFrameBackendState(
            requested=requested,
            active="pandas",
            fallback_pandas=fallback_pandas,
            reason=f"cudf_probe_failed:{type(exc).__name__}",
        )

    return DataFrameBackendState(
        requested=requested,
        active="cudf",
        fallback_pandas=fallback_pandas,
        reason="cudf_ready",
    )
