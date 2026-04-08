from __future__ import annotations


class ArchitectureError(RuntimeError):
    """Raised when layer boundaries are violated or required runtime wiring is missing."""

