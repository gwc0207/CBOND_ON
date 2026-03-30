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


def _parse_requested_backend(cfg: dict[str, Any]) -> str:
    requested = str(cfg.get("backend", "auto")).strip().lower()
    if requested in {"", "none"}:
        return "cpu"
    if requested in {"gpu"}:
        return "auto"
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

