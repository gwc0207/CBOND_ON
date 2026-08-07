"""Execution-policy guard shared by research workflows and app entrypoints.

This module deliberately depends on neither ``workflows`` nor factor-runtime
implementations.  The application usecase can therefore enforce the same
Rust-first boundary even when it is called programmatically instead of through
the research workflow or CLI.
"""

from __future__ import annotations

from typing import Any, Mapping


RUST_FIRST_POLICY = "rust_first"
_RETIRED_TOP_LEVEL_REFERENCE_FIELDS = (
    "legacy_reference_only",
    "legacy_reference_reason",
)


def _require_rust_first_compute(compute: Mapping[str, Any], *, scope: str) -> None:
    """Validate the only executable factor-compute policy.

    Python implementations remain useful as unit-test/parity references, but
    they are deliberately not a batch, research, FactorStore, model, or live
    execution mode.  Keeping this rule in a small shared helper makes the
    application/workflow boundary reject a stale config before it can start
    loading panels or writing a store.
    """

    policy = str(compute.get("execution_policy", "")).strip().lower()
    engine = str(compute.get("engine", compute.get("factor_engine", ""))).strip().lower()
    if policy != RUST_FIRST_POLICY:
        raise ValueError(
            f"{scope} factor batches must use compute.execution_policy='rust_first'; "
            "Python is available only to isolated parity/reference tests"
        )
    if engine != "rust":
        raise ValueError(
            f"{scope} factor batches require compute.engine='rust' when "
            "execution_policy='rust_first'"
        )
    if "allow_python_engine" in compute:
        raise ValueError(
            f"{scope} factor batches must not declare compute.allow_python_engine; "
            "normal factor execution has no Python route"
        )


def validate_factor_execution_policy(
    factor_cfg: Mapping[str, Any],
    *,
    scope: str = "factor",
) -> None:
    """Reject every non-Rust executable factor path before runtime I/O.

    This applies uniformly to batch, research, and direct application entry
    points.  Historical Python code remains a test/parity reference only; it
    has no executable configuration exception and cannot reopen a
    FactorStore-producing pipeline.
    """

    raw_compute = factor_cfg.get("compute", {})
    if not isinstance(raw_compute, Mapping):
        raise TypeError(f"{scope} factor_config.compute must be an object")
    compute = dict(raw_compute)
    _require_rust_first_compute(compute, scope=scope)
    retired = [field for field in _RETIRED_TOP_LEVEL_REFERENCE_FIELDS if field in factor_cfg]
    if retired:
        raise ValueError(
            f"{scope} factor configs must not declare retired Python-reference fields: "
            + ", ".join(retired)
        )


def validate_research_execution_policy(factor_cfg: Mapping[str, Any]) -> None:
    """Research-name compatibility wrapper for the uniform Rust-only policy."""

    if factor_cfg.get("research_only") is not True:
        return
    validate_factor_execution_policy(factor_cfg, scope="research")
