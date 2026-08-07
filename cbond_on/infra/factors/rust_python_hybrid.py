"""Strict Rust/Python factor-frame split for the explicit live-50 path.

This backend never guesses which implementation should compute a factor.  The
configuration must name every output column in exactly one of ``rust_columns``
or ``python_columns``; the pipeline then preserves the factor-pack column
order when it combines the two results.
"""

"""Historical Rust/Python parity reference only.

This module is deliberately outside every normal factor execution path.  It
cannot be selected by config, FactorPipeline, batch, research, model, or live
runtime, and its compute function requires an explicit isolated-parity marker.
It must never be used to populate a FactorStore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from cbond_on.domain.factors.builder import build_factor_frame
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col
from cbond_on.infra.factors.rust_backend import build_factor_frame_rust


@dataclass(frozen=True)
class RustPythonHybridPlan:
    """Fully validated column routing for one factor pipeline invocation."""

    rust_columns: tuple[str, ...]
    python_columns: tuple[str, ...]
    output_columns: tuple[str, ...]

    def rust_specs(self, specs: Sequence[FactorSpec]) -> list[FactorSpec]:
        rust_set = set(self.rust_columns)
        return [spec for spec in specs if build_factor_col(spec) in rust_set]

    def python_specs(self, specs: Sequence[FactorSpec]) -> list[FactorSpec]:
        python_set = set(self.python_columns)
        return [spec for spec in specs if build_factor_col(spec) in python_set]

    def for_pending_specs(self, specs: Sequence[FactorSpec]) -> "RustPythonHybridPlan":
        """Restrict a validated full plan to missing columns without rerouting them.

        This preserves the normal non-refresh behavior of the factor store: if
        the original Rust columns are already present, an incremental run may
        compute only the pending Python columns.  It never moves an existing
        Rust column to Python (or the reverse).
        """

        pending_columns = tuple(build_factor_col(spec) for spec in specs)
        if len(set(pending_columns)) != len(pending_columns):
            raise ValueError("rust_python_hybrid pending factor specs have duplicate output columns")
        unknown = sorted(set(pending_columns).difference(self.output_columns))
        if unknown:
            raise ValueError(
                "rust_python_hybrid pending factor specs are outside the validated plan: "
                + ", ".join(unknown)
            )
        pending_set = set(pending_columns)
        return RustPythonHybridPlan(
            rust_columns=tuple(column for column in self.rust_columns if column in pending_set),
            python_columns=tuple(column for column in self.python_columns if column in pending_set),
            output_columns=pending_columns,
        )


def _require_column_list(compute_cfg: dict[str, Any], *, field: str) -> tuple[str, ...]:
    raw = compute_cfg.get(field)
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"compute.{field} must be a list for engine=rust_python_hybrid")
    columns = tuple(str(item).strip() for item in raw)
    if not columns:
        raise ValueError(f"compute.{field} must be non-empty for engine=rust_python_hybrid")
    if any(not column for column in columns):
        raise ValueError(f"compute.{field} must not contain an empty column")
    if len(set(columns)) != len(columns):
        raise ValueError(f"compute.{field} must not contain duplicate columns")
    return columns


def resolve_rust_python_hybrid_plan(
    specs: Sequence[FactorSpec],
    *,
    compute_cfg: dict[str, Any] | None,
) -> RustPythonHybridPlan:
    """Validate an explicit split and freeze the requested final column order."""

    cfg = dict(compute_cfg or {})
    rust_columns = _require_column_list(cfg, field="rust_columns")
    python_columns = _require_column_list(cfg, field="python_columns")
    overlap = sorted(set(rust_columns).intersection(python_columns))
    if overlap:
        raise ValueError(
            "compute.rust_columns and compute.python_columns overlap: " + ", ".join(overlap)
        )

    output_columns = tuple(build_factor_col(spec) for spec in specs)
    if not output_columns:
        raise ValueError("engine=rust_python_hybrid requires at least one factor spec")
    duplicate_specs = sorted(
        column for column in set(output_columns) if output_columns.count(column) > 1
    )
    if duplicate_specs:
        raise ValueError(
            "factor specs have duplicate output columns for engine=rust_python_hybrid: "
            + ", ".join(duplicate_specs)
        )

    configured = set(rust_columns).union(python_columns)
    spec_columns = set(output_columns)
    missing = sorted(spec_columns.difference(configured))
    unknown = sorted(configured.difference(spec_columns))
    if missing or unknown:
        raise ValueError(
            "rust_python_hybrid column routing must cover factor specs exactly: "
            f"missing={missing}, unknown={unknown}"
        )
    return RustPythonHybridPlan(
        rust_columns=rust_columns,
        python_columns=python_columns,
        output_columns=output_columns,
    )


def _normalize_and_validate_frame(
    frame: pd.DataFrame,
    *,
    expected_columns: Sequence[str],
    source: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"rust_python_hybrid {source} result must be pandas.DataFrame, got {type(frame).__name__}"
        )
    out = frame.copy()
    if {"dt", "code"}.issubset(out.columns):
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels >= 2:
            raise RuntimeError(
                f"rust_python_hybrid {source} result exposes dt/code in both index and columns"
            )
        out = out.set_index(["dt", "code"])
    if not isinstance(out.index, pd.MultiIndex) or out.index.nlevels != 2:
        raise RuntimeError(
            f"rust_python_hybrid {source} result must use a two-level (dt, code) index"
        )

    if out.columns.duplicated().any():
        duplicates = sorted(set(out.columns[out.columns.duplicated()].astype(str)))
        raise RuntimeError(
            f"rust_python_hybrid {source} result has duplicate factor columns: {duplicates}"
        )
    expected = tuple(str(column) for column in expected_columns)
    missing_columns = [column for column in expected if column not in out.columns]
    unexpected_columns = [str(column) for column in out.columns if str(column) not in set(expected)]
    if missing_columns or unexpected_columns:
        raise RuntimeError(
            f"rust_python_hybrid {source} columns do not match its configured route: "
            f"missing={missing_columns}, unexpected={unexpected_columns}"
        )

    keys = out.index.to_frame(index=False)
    keys.columns = ["dt", "code"]
    if keys.duplicated(["dt", "code"]).any():
        duplicate_rows = keys.loc[keys.duplicated(["dt", "code"], keep=False)].head(5)
        raise RuntimeError(
            f"rust_python_hybrid {source} result has duplicate raw (dt, code) keys: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )
    if keys["dt"].isna().any() or keys["code"].isna().any():
        raise RuntimeError(f"rust_python_hybrid {source} result has null (dt, code) keys")
    keys["dt"] = pd.to_datetime(keys["dt"], errors="coerce")
    keys["code"] = keys["code"].astype(str).str.strip()
    if keys["dt"].isna().any() or (keys["code"] == "").any():
        raise RuntimeError(f"rust_python_hybrid {source} result has invalid (dt, code) keys")

    normalized = out.copy()
    normalized.index = pd.MultiIndex.from_frame(keys, names=["dt", "code"])
    if normalized.index.has_duplicates:
        duplicate_rows = normalized.index.to_frame(index=False)
        duplicate_rows = duplicate_rows.loc[duplicate_rows.duplicated(["dt", "code"], keep=False)].head(5)
        raise RuntimeError(
            f"rust_python_hybrid {source} result has duplicate normalized (dt, code) keys: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )
    return normalized.loc[:, list(expected)].sort_index()


def _key_mismatch_message(rust_frame: pd.DataFrame, python_frame: pd.DataFrame) -> str:
    rust_only = rust_frame.index.difference(python_frame.index)
    python_only = python_frame.index.difference(rust_frame.index)
    return (
        "rust_python_hybrid Rust/Python (dt, code) key mismatch: "
        f"rust_only={rust_only.to_list()[:5]}, python_only={python_only.to_list()[:5]}"
    )


def build_factor_frame_rust_python_hybrid(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    plan: RustPythonHybridPlan,
    stock_panel: pd.DataFrame | None = None,
    bond_stock_map: pd.DataFrame | None = None,
    daily_data: dict[str, pd.DataFrame] | None = None,
    workers: int = 1,
    compute_backend_params: dict | None = None,
    reference_mode: str | None = None,
) -> pd.DataFrame:
    """Compute a one-off historical parity reference, never a normal factor frame."""

    if reference_mode != "isolated_parity_reference":
        raise RuntimeError(
            "rust_python_hybrid is an isolated parity reference only; normal factor execution "
            "must use Rust compute_factor_frame through FactorPipeline"
        )

    spec_list = list(specs)
    current_columns = tuple(build_factor_col(spec) for spec in spec_list)
    if current_columns != plan.output_columns:
        raise ValueError(
            "rust_python_hybrid plan/spec mismatch; resolve a new plan for the current factor specs"
        )
    rust_specs = plan.rust_specs(spec_list)
    python_specs = plan.python_specs(spec_list)
    if tuple(build_factor_col(spec) for spec in rust_specs) != tuple(
        column for column in plan.output_columns if column in set(plan.rust_columns)
    ):
        raise RuntimeError("rust_python_hybrid Rust spec routing drift")
    if tuple(build_factor_col(spec) for spec in python_specs) != tuple(
        column for column in plan.output_columns if column in set(plan.python_columns)
    ):
        raise RuntimeError("rust_python_hybrid Python spec routing drift")

    rust_frame: pd.DataFrame | None = None
    if rust_specs:
        rust_frame = _normalize_and_validate_frame(
            build_factor_frame_rust(
                panel,
                rust_specs,
                stock_panel=stock_panel,
                bond_stock_map=bond_stock_map,
                daily_data=daily_data,
                compute_backend_params=compute_backend_params,
            ),
            expected_columns=plan.rust_columns,
            source="Rust",
        )
    python_frame: pd.DataFrame | None = None
    if python_specs:
        python_frame = _normalize_and_validate_frame(
            build_factor_frame(
                panel,
                python_specs,
                stock_panel=stock_panel,
                bond_stock_map=bond_stock_map,
                daily_data=daily_data,
                workers=workers,
                compute_backend_params=compute_backend_params,
            ),
            expected_columns=plan.python_columns,
            source="Python",
        )
    if rust_frame is None and python_frame is None:
        raise RuntimeError("rust_python_hybrid has no routed factor specs to compute")
    if rust_frame is None:
        return python_frame.loc[:, list(plan.output_columns)]  # type: ignore[union-attr]
    if python_frame is None:
        return rust_frame.loc[:, list(plan.output_columns)]
    if not rust_frame.index.equals(python_frame.index):
        raise RuntimeError(_key_mismatch_message(rust_frame, python_frame))

    merged = rust_frame.join(python_frame, how="inner")
    if len(merged) != len(rust_frame) or len(merged) != len(python_frame):
        raise RuntimeError("rust_python_hybrid merge unexpectedly changed the (dt, code) universe")
    if merged.columns.duplicated().any():
        duplicates = sorted(set(merged.columns[merged.columns.duplicated()].astype(str)))
        raise RuntimeError(
            f"rust_python_hybrid merge has duplicate factor columns: {duplicates}"
        )
    return merged.loc[:, list(plan.output_columns)]


__all__ = [
    "RustPythonHybridPlan",
    "build_factor_frame_rust_python_hybrid",
    "resolve_rust_python_hybrid_plan",
]
