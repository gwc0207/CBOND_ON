"""Research-only daily factor supplement.

At 23:59 an independent task reconciles the full registered catalog against
the immutable live release and, in explicit ``execute`` mode, computes only
the non-live factor instances in ephemeral scratch staging before publishing
the complete family set into the canonical factor-library table.  It never
invokes the live runtime or scheduler and never writes model scores, trade
lists, or the production database.

Normal factor paths remain Rust-first.  This module is the sole research-only
exception: it can use ``research_python`` only after an opaque
``ResearchCatalogExecutionPermit`` has pinned every selected contract,
operator source hash, catalog snapshot and scratch output root.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import mkdtemp
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    FactorColumnContract,
    FactorTableContract,
    FactorTableKind,
)
from cbond_on.infra.factors.research_catalog_permit import (
    RESEARCH_CATALOG_PYTHON_ENGINE,
    RESEARCH_CATALOG_PYTHON_POLICY,
    ResearchCatalogExecutionPermit,
    ResearchCatalogPermitError,
    build_permitted_factor_specs,
    issue_research_catalog_execution_permit,
)
from cbond_on.infra.factors.pipeline import run_factor_pipeline
from cbond_on.infra.live.publish_gate import run_publish_status


SUPPLEMENT_SCHEMA = "cbond_on_factor_supplement/v1"
LEDGER_SCHEMA = "cbond_on_factor_supplement_ledger/v1"
DEFAULT_CONFIG_REF = "factor/research/factor_supplement_v1"
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
CANONICAL_FACTOR_STORE_ROOT = Path(r"D:/cbond_on/factor_store")
_SUPPORTED_MODES = frozenset({"plan", "dry-run", "execute"})
_EXECUTION_MANIFEST_SCHEMA = "cbond_on_factor_supplement_execution/v2"


class FactorSupplementError(RuntimeError):
    """Base error for the isolated research supplement boundary."""


class FactorSupplementLockError(FactorSupplementError):
    """Raised when a same-day supplement attempt already owns the lock."""


@dataclass(frozen=True)
class CatalogFactor:
    """One registered factor identity from the full research catalog."""

    factor_id: str
    primary_family: str
    factor_version: str | None = None
    contract_hash: str | None = None
    output_column: str | None = None
    rust_contract_id: str | None = None


@dataclass(frozen=True)
class _LockHandle:
    path: Path
    token: str


def _as_mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return dict(value)


def _platform_value(value: object) -> object:
    """Choose a platform-specific scalar without importing a paths profile."""

    if not isinstance(value, Mapping):
        return value
    normalized = {str(key).strip().lower(): item for key, item in value.items()}
    primary = ("windows", "win") if sys.platform.startswith("win") else ("linux", "unix", "posix", "server")
    for key in (*primary, "default", "common", "all", "value"):
        picked = normalized.get(key)
        if picked not in (None, ""):
            return picked
    return next((item for item in normalized.values() if item not in (None, "")), None)


def _path_from(value: object, *, field: str) -> Path:
    picked = _platform_value(value)
    text = str(picked or "").strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return Path(text).expanduser().resolve(strict=False)


def _is_strict_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_safely(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def load_supplement_config(config_ref: str | Path = DEFAULT_CONFIG_REF) -> tuple[Path, dict[str, Any]]:
    """Load the dedicated research-only configuration and validate its boundary."""

    path = resolve_config_path(str(config_ref)).resolve()
    payload = load_json_like(path)
    config = _as_mapping(payload, field="factor supplement config")
    if str(config.get("schema", "")).strip() != SUPPLEMENT_SCHEMA:
        raise ValueError(f"factor supplement config.schema must equal {SUPPLEMENT_SCHEMA!r}")
    if config.get("research_only") is not True:
        raise ValueError("factor supplement config requires research_only=true")
    compute = _as_mapping(config.get("compute"), field="factor supplement config.compute")
    if str(compute.get("engine", "")).strip().lower() != RESEARCH_CATALOG_PYTHON_ENGINE:
        raise ValueError(
            f"factor supplement V1 requires compute.engine={RESEARCH_CATALOG_PYTHON_ENGINE!r}"
        )
    if str(compute.get("execution_policy", "")).strip().lower() != RESEARCH_CATALOG_PYTHON_POLICY:
        raise ValueError(
            "factor supplement V1 requires "
            f"compute.execution_policy={RESEARCH_CATALOG_PYTHON_POLICY!r}"
        )
    output = _as_mapping(config.get("output"), field="factor supplement config.output")
    if output.get("scratch_root") in (None, "") or output.get("canonical_factor_store_root") in (None, ""):
        raise ValueError(
            "factor supplement output requires scratch_root for runtime audit staging and "
            "canonical_factor_store_root for persistent factor publication"
        )
    inputs = _as_mapping(config.get("inputs"), field="factor supplement config.inputs")
    if inputs.get("live_factor_table") in (None, ""):
        raise ValueError(
            "factor supplement inputs requires manifest-bound live_factor_table for the complete library day"
        )
    return path, config


def configured_schedule_time(config: Mapping[str, Any]) -> str:
    schedule = _as_mapping(config.get("schedule"), field="factor supplement config.schedule")
    value = str(schedule.get("time", "")).strip()
    if value != "23:59":
        raise ValueError("factor supplement V1 schedule.time must be exactly '23:59'")
    return value


def configured_scratch_root(config: Mapping[str, Any]) -> Path:
    output = _as_mapping(config.get("output"), field="factor supplement config.output")
    return _validate_scratch_root(_path_from(output.get("scratch_root"), field="output.scratch_root"))


def configured_canonical_factor_store_root(config: Mapping[str, Any]) -> Path:
    """Return the one persistent factor-result destination for this task."""

    output = _as_mapping(config.get("output"), field="factor supplement config.output")
    root = _path_from(
        output.get("canonical_factor_store_root"),
        field="output.canonical_factor_store_root",
    )
    # The complement job is a designated writer to the existing three-table
    # store, never a creator of a parallel local data layer.
    expected = CANONICAL_FACTOR_STORE_ROOT.expanduser().resolve(strict=False)
    if root != expected:
        raise ValueError(
            "factor supplement canonical_factor_store_root must be the established "
            f"canonical root {expected.as_posix()}, got {root.as_posix()}"
        )
    return root


def _validate_scratch_root(path: Path) -> Path:
    parent = RESEARCH_SCRATCH_PARENT.expanduser().resolve(strict=False)
    root = path.expanduser().resolve(strict=False)
    if not _is_strict_child(root, parent):
        raise ValueError(
            "factor supplement scratch root must resolve strictly below "
            f"{parent.as_posix()}, got {root.as_posix()}"
        )
    return root


def _configured_live_factor_table(config: Mapping[str, Any]) -> CanonicalFactorStore:
    """Resolve the published live table as a strictly manifest-bound input."""

    inputs = _as_mapping(config.get("inputs"), field="factor supplement config.inputs")
    table = _as_mapping(inputs.get("live_factor_table"), field="inputs.live_factor_table")
    if str(table.get("table_id", "")).strip() != FactorTableKind.LIVE.value:
        raise ValueError("supplement live_factor_table.table_id must be exactly 'live'")
    root = _path_from(table.get("root"), field="inputs.live_factor_table.root")
    expected = CANONICAL_FACTOR_STORE_ROOT.expanduser().resolve(strict=False)
    if root != expected:
        raise ValueError(
            "supplement live_factor_table.root must be the established canonical root "
            f"{expected.as_posix()}, got {root.as_posix()}"
        )
    store = CanonicalFactorStore(root)
    store.require_consumer_ready(FactorTableKind.LIVE)
    return store


def _canonical_factor_library_store(config: Mapping[str, Any]) -> CanonicalFactorStore:
    store = CanonicalFactorStore(configured_canonical_factor_store_root(config))
    # A daily writer must not bootstrap or write into an unverified migration
    # target.  The full-table migration owns initialization and readiness.
    store.require_consumer_ready(FactorTableKind.FACTOR_LIBRARY)
    return store


def _load_csv_catalog(path: Path, cfg: Mapping[str, Any]) -> list[CatalogFactor]:
    factor_id_column = str(cfg.get("factor_id_column", "factor")).strip()
    family_column = str(cfg.get("family_column", "family")).strip()
    if not factor_id_column:
        raise ValueError("catalog.factor_id_column must not be empty")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise FileNotFoundError(f"cannot read factor catalog: {path}") from exc
    if not rows:
        raise ValueError(f"factor catalog has no rows: {path}")
    factors: list[CatalogFactor] = []
    for position, row in enumerate(rows, start=2):
        raw_factor_id = row.get(factor_id_column)
        factor_id = str(raw_factor_id or "").strip()
        if not factor_id:
            raise ValueError(f"factor catalog row {position} has empty {factor_id_column!r}")
        family = str(row.get(family_column) or "").strip()
        factors.append(CatalogFactor(factor_id=factor_id, primary_family=family))
    return factors


def _load_json_catalog(path: Path) -> list[CatalogFactor]:
    payload = _as_mapping(load_json_like(path), field="factor catalog")
    if str(payload.get("schema_version", payload.get("schema", ""))).strip() != "factor_catalog/v1":
        raise ValueError("JSON factor catalog schema must equal 'factor_catalog/v1'")
    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, list):
        raise TypeError("JSON factor catalog.factors must be a list")
    factors: list[CatalogFactor] = []
    for position, raw in enumerate(raw_factors):
        item = _as_mapping(raw, field=f"factor catalog.factors[{position}]")
        factor_id = str(item.get("factor_id", "")).strip()
        if not factor_id:
            raise ValueError(f"factor catalog.factors[{position}].factor_id must not be empty")
        rust_contract_id = str(item.get("rust_contract_id") or "").strip() or None
        factors.append(
            CatalogFactor(
                factor_id=factor_id,
                primary_family=str(item.get("primary_family") or "").strip(),
                factor_version=str(item.get("factor_version") or "").strip() or None,
                contract_hash=str(item.get("contract_hash") or "").strip().lower() or None,
                output_column=str(item.get("output_column") or factor_id).strip() or None,
                rust_contract_id=rust_contract_id,
            )
        )
    return factors


def _verify_canonical_catalog_api(path: Path, *, factor_ids: Sequence[str]) -> None:
    """Ensure a plan uses the catalog agent's public read-only resolver."""

    try:
        from cbond_on.domain.factor_catalog import catalog_path, load_factor_catalog
    except ImportError as exc:  # pragma: no cover - checked-in package invariant.
        raise RuntimeError("canonical factor catalog API is unavailable") from exc
    repo_root = path.parents[2]
    expected = catalog_path(repo_root).resolve(strict=False)
    if path != expected:
        raise ValueError(
            "factor_catalog_json path must be the canonical catalog API path: "
            f"expected={expected.as_posix()}, actual={path.as_posix()}"
        )
    resolved = load_factor_catalog(repo_root)
    if set(resolved) != set(factor_ids):
        raise ValueError("canonical factor catalog API keys differ from catalog JSON keys")


def _load_catalog(config: Mapping[str, Any]) -> tuple[Path, list[CatalogFactor], dict[str, Any]]:
    cfg = _as_mapping(config.get("catalog"), field="factor supplement config.catalog")
    path = _path_from(cfg.get("path"), field="catalog.path")
    format_name = str(cfg.get("format", "csv_factor_family_catalog")).strip().lower()
    if format_name == "csv_factor_family_catalog":
        factors = _load_csv_catalog(path, cfg)
    elif format_name == "factor_catalog_json":
        factors = _load_json_catalog(path)
    else:
        raise ValueError(
            "catalog.format must be 'csv_factor_family_catalog' or 'factor_catalog_json'"
        )
    ids = [factor.factor_id for factor in factors]
    duplicate_ids = sorted({factor_id for factor_id in ids if ids.count(factor_id) > 1})
    if duplicate_ids:
        raise ValueError("factor catalog has duplicate factor_id values: " + ", ".join(duplicate_ids[:20]))
    expected_count = int(cfg.get("expected_factor_count", 0) or 0)
    if expected_count and len(factors) != expected_count:
        raise ValueError(
            "factor catalog count does not match expected_factor_count: "
            f"expected={expected_count}, actual={len(factors)}"
        )
    if format_name == "factor_catalog_json":
        _verify_canonical_catalog_api(path, factor_ids=ids)
    return path, factors, {
        "path": str(path),
        "sha256": _sha256(path),
        "format": format_name,
        "factor_count": len(factors),
        "expected_factor_count": expected_count,
    }


def _load_release_exclude(config: Mapping[str, Any]) -> tuple[set[str], dict[str, Any]]:
    """Read only the release factor names, which are used solely as exclusions."""

    cfg = _as_mapping(config.get("release_exclude"), field="factor supplement config.release_exclude")
    path = _path_from(cfg.get("manifest"), field="release_exclude.manifest")
    payload = _as_mapping(load_json_like(path), field="release exclusion manifest")
    format_name = str(cfg.get("format", "live_release_instances")).strip().lower()
    if format_name == "live_release_instances":
        if str(payload.get("schema_version", "")).strip() != "factor_catalog_live_release/v1":
            raise ValueError("release exclusion manifest must be factor_catalog_live_release/v1")
        raw_instances = payload.get("instances")
        if not isinstance(raw_instances, list):
            raise TypeError("release exclusion manifest.instances must be a list")
        names = tuple(
            str(item.get("factor_id", "")).strip() if isinstance(item, Mapping) else ""
            for item in raw_instances
        )
        declared_count = int(payload.get("factor_count", -1))
        if declared_count != len(names):
            raise ValueError("release exclusion manifest factor_count does not match instances")
        for position, item in enumerate(raw_instances):
            if not isinstance(item, Mapping):
                raise TypeError(f"release exclusion manifest.instances[{position}] must be an object")
            for field in ("factor_id", "factor_version", "contract_hash", "operator_id"):
                if not str(item.get(field, "")).strip():
                    raise ValueError(f"release exclusion instance missing {field}: position={position}")
        # The release is validated against canonical identity/version/hash
        # metadata.  Its values are still used only as an exclusion set below.
        try:
            from cbond_on.domain.factor_catalog import load_live_release
        except ImportError as exc:  # pragma: no cover - checked-in package invariant.
            raise RuntimeError("canonical live release resolver is unavailable") from exc
        repo_root = path.parents[3]
        resolved = load_live_release(str(payload.get("release_id", "")).strip(), repo_root=repo_root)
        if list(resolved.get("instances", [])) != raw_instances:
            raise ValueError("release exclusion manifest differs from catalog API release metadata")
    elif format_name == "legacy_factors_list_for_test_only":
        raw_factors = payload.get("factors")
        if not isinstance(raw_factors, list):
            raise TypeError("legacy release exclusion manifest.factors must be a list")
        names = tuple(str(item).strip() for item in raw_factors)
    else:
        raise ValueError(
            "release_exclude.format must be 'live_release_instances' or "
            "'legacy_factors_list_for_test_only'"
        )
    if any(not name for name in names):
        raise ValueError("release exclusion manifest must not contain empty factor IDs")
    if len(set(names)) != len(names):
        raise ValueError("release exclusion manifest must not contain duplicate factor IDs")
    return set(names), {
        "path": str(path),
        "sha256": _sha256(path),
        "format": format_name,
        "factor_count": len(names),
        "factor_ids": list(names),
        "use": "exclude_only",
    }


def _datahub_gate(config: Mapping[str, Any], *, score_day: date) -> dict[str, Any]:
    cfg = _as_mapping(config.get("data_hub"), field="factor supplement config.data_hub")
    required_datasets = [str(item).strip().lower() for item in cfg.get("require_datasets", ["clean"])]
    if not required_datasets or any(not item for item in required_datasets):
        raise ValueError("data_hub.require_datasets must be a non-empty list of names")
    runtime = {
        "manifest_root": str(_path_from(cfg.get("manifest_root"), field="data_hub.manifest_root")),
        "require_datasets": required_datasets,
        "allow_partial_manifest": False,
        "require_done_marker": True,
        "ready_gate_enabled": True,
    }
    try:
        status = dict(run_publish_status(runtime=runtime, trade_day=score_day))
    except Exception as exc:
        return {
            "passed": False,
            "runtime": runtime,
            "status": {},
            "reasons": [f"publish_status_error:{type(exc).__name__}"],
            "error": str(exc),
        }
    reasons: list[str] = []
    if not bool(status.get("ready", False)):
        reasons.append("publish_not_ready")
    if not bool(status.get("done_exists", False)):
        reasons.append("done_marker_missing")
    if not bool(status.get("manifest_run_id_consistent", False)):
        reasons.append("manifest_run_id_inconsistent")
    if not bool(status.get("manifest_run_id_complete", False)):
        reasons.append("manifest_run_id_incomplete")
    expected_profile = str(cfg.get("required_profile", "")).strip()
    if expected_profile:
        clean_manifest = dict(status.get("manifests", {}).get("clean", {}))
        clean_payload = _read_json_safely(Path(str(clean_manifest.get("path", ""))))
        if str(clean_payload.get("required_profile", "")).strip() != expected_profile:
            reasons.append("clean_manifest_profile_mismatch")
        validation = clean_payload.get("validation")
        if not isinstance(validation, Mapping) or validation.get("passed") is not True:
            reasons.append("clean_manifest_validation_not_passed")
    return {
        "passed": not reasons,
        "runtime": runtime,
        "status": status,
        "reasons": reasons,
    }


def _canonical_factor_library_plan_evidence(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Inspect, but never initialize or mutate, the canonical writer target."""

    try:
        root = configured_canonical_factor_store_root(config)
        store = _canonical_factor_library_store(config)
        manifest = store.read_table_manifest(FactorTableKind.FACTOR_LIBRARY)
        migration = _as_mapping(manifest.get("migration"), field="canonical factor-library migration")
        return {
            "root": str(root),
            "table_id": FactorTableKind.FACTOR_LIBRARY.value,
            "migration": dict(migration),
            "registered_contract_count": len(manifest.get("registered_contracts", [])),
            "writer": "CanonicalFactorStore.publish_factor_library_day",
        }, []
    except Exception as exc:
        return {
            "root": str(
                _path_from(
                    _as_mapping(config.get("output"), field="factor supplement config.output").get(
                        "canonical_factor_store_root"
                    ),
                    field="output.canonical_factor_store_root",
                )
            )
            if isinstance(config.get("output"), Mapping)
            and _as_mapping(config.get("output"), field="factor supplement config.output").get(
                "canonical_factor_store_root"
            )
            not in (None, "")
            else "",
            "table_id": FactorTableKind.FACTOR_LIBRARY.value,
            "writer": "CanonicalFactorStore.publish_factor_library_day",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, [f"canonical_factor_library_unavailable:{type(exc).__name__}"]


def _live_factor_input_plan_evidence(config: Mapping[str, Any], *, score_day: date) -> tuple[dict[str, Any], list[str]]:
    """Verify the manifest-bound live companion without touching a legacy root."""

    try:
        store = _configured_live_factor_table(config)
        path = store.partition_paths(FactorTableKind.LIVE, score_day).parquet_path
        frame = store.read_day(FactorTableKind.LIVE, score_day)
        evidence = {
            "root": str(store.root),
            "table_id": FactorTableKind.LIVE.value,
            "path": str(path),
            "exists": True,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
            "row_count": int(len(frame)),
            "read_only": True,
        }
        return evidence, []
    except Exception as exc:
        return {
            "read_only": True,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, [f"live_factor_input_unavailable:{type(exc).__name__}"]


def build_plan(config: Mapping[str, Any], *, score_day: date) -> dict[str, Any]:
    """Create a no-write daily plan and block before any factor computation."""

    if config.get("research_only") is not True:
        raise ValueError("factor supplement requires research_only=true")
    configured_schedule_time(config)
    catalog_path, catalog, catalog_evidence = _load_catalog(config)
    _ = catalog_path  # Keep the checked path in evidence; no catalog data is written.
    catalog_ids = {factor.factor_id for factor in catalog}
    release_exclude, release_evidence = _load_release_exclude(config)
    excluded_ids = sorted(catalog_ids.intersection(release_exclude))
    release_only_ids = sorted(release_exclude.difference(catalog_ids))
    supplement_ids = sorted(catalog_ids.difference(release_exclude))
    datahub = _datahub_gate(config, score_day=score_day)
    canonical_execution = str(catalog_evidence.get("format", "")).strip().lower() == "factor_catalog_json"
    if canonical_execution:
        canonical_target, canonical_reasons = _canonical_factor_library_plan_evidence(config)
        live_input, live_input_reasons = _live_factor_input_plan_evidence(config, score_day=score_day)
    else:
        # CSV fixtures are retained only for plan-contract tests.  They have no
        # version/hash metadata and are never eligible to publish a canonical
        # factor-library partition.
        canonical_target, canonical_reasons = (
            {"writer": "not_applicable_legacy_plan_only", "read_only": True},
            [],
        )
        live_input, live_input_reasons = ({"read_only": True, "not_applicable": True}, [])
    family_members: dict[str, list[str]] = {}
    for factor in catalog:
        if factor.factor_id not in supplement_ids:
            continue
        family = str(factor.primary_family).strip()
        if not family:
            raise ValueError(f"canonical catalog factor has no primary_family: {factor.factor_id}")
        family_members.setdefault(family, []).append(factor.factor_id)

    blocking_reasons: list[str] = []
    if not datahub["passed"]:
        blocking_reasons.append("datahub_publish_gate_not_ready")
    blocking_reasons.extend(canonical_reasons)
    blocking_reasons.extend(live_input_reasons)
    if not blocking_reasons:
        disposition = "PLAN_READY_NO_EXECUTION"
    elif not datahub["passed"] and len(blocking_reasons) == 1:
        disposition = "BLOCKED_DATAHUB"
    else:
        disposition = "BLOCKED_PRECHECK"
    return {
        "schema_version": SUPPLEMENT_SCHEMA,
        "research_only": True,
        "score_day": score_day.isoformat(),
        "disposition": disposition,
        "blocking_reasons": blocking_reasons,
        "datahub": datahub,
        "canonical_factor_library": canonical_target,
        "live_factor_input": live_input,
        "catalog": catalog_evidence,
        "release_exclude": {
            **release_evidence,
            "excluded_catalog_factor_ids": excluded_ids,
            "release_factor_ids_not_in_catalog": release_only_ids,
        },
        "supplement_scope": {
            "catalog_factor_count": len(catalog),
            "live_excluded_factor_count": len(excluded_ids),
            "non_live_catalog_factor_count": len(supplement_ids),
            "non_live_catalog_factor_ids": supplement_ids,
            "non_live_family_factor_ids": {
                family: list(factor_ids)
                for family, factor_ids in sorted(family_members.items())
            },
        },
        "research_python_admission": {
            "engine": RESEARCH_CATALOG_PYTHON_ENGINE,
            "execution_policy": RESEARCH_CATALOG_PYTHON_POLICY,
            "permit_required": True,
            "canonical_catalog_path": str(catalog_path),
            "catalog_snapshot_sha256": catalog_evidence["sha256"],
            "permit_issued": False,
            "reason": (
                "Execute mode constructs an opaque ResearchCatalogExecutionPermit from the exact "
                "canonical catalog, every selected factor contract, and ephemeral scratch staging "
                "before Python operator modules can import."
            ),
        },
        "execution_boundary": {
            "allowed_mode": "execute_one_score_day_research_only",
            "factor_compute": "not_started",
            "factor_store_write": "canonical_factor_library_only_when_execute",
            "legacy_scratch_factor_store": "forbidden_writer",
            "model_score": "forbidden",
            "trade_list": "forbidden",
            "database_write": "forbidden",
            "live_scheduler": "forbidden",
        },
    }


def issue_catalog_execution_permit(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    scratch_root: Path | None = None,
    staging_factor_root: Path | None = None,
) -> ResearchCatalogExecutionPermit:
    """Construct the only research Python permit for a previously planned scope.

    This helper deliberately does not execute a factor.  It simply converts the
    planned non-live catalog set into a contract- and hash-bound opaque permit.
    The canonical catalog itself is re-read and re-validated by the permit
    factory, so a stale plan cannot authorize changed definitions.
    """

    if str(plan.get("disposition", "")) != "PLAN_READY_NO_EXECUTION":
        raise ResearchCatalogPermitError(
            "cannot issue research catalog execution permit for a blocked supplement plan"
        )
    catalog = _as_mapping(plan.get("catalog"), field="supplement plan.catalog")
    if str(catalog.get("format", "")).strip().lower() != "factor_catalog_json":
        raise ResearchCatalogPermitError(
            "research Python execution requires the canonical factor_catalog_json; "
            "legacy CSV inventory cannot authorize execution"
        )
    scope = _as_mapping(plan.get("supplement_scope"), field="supplement plan.supplement_scope")
    factor_ids = scope.get("non_live_catalog_factor_ids")
    if not isinstance(factor_ids, list) or not factor_ids:
        raise ResearchCatalogPermitError("supplement plan has no non-live catalog factor IDs")
    resolved_scratch_root = _validate_scratch_root(scratch_root) if scratch_root is not None else configured_scratch_root(config)
    factor_data_root = (
        staging_factor_root.expanduser().resolve(strict=False)
        if staging_factor_root is not None
        else resolved_scratch_root / "staging" / "unbound" / "factor_data"
    )
    if not _is_strict_child(factor_data_root, resolved_scratch_root):
        raise ResearchCatalogPermitError(
            "research supplement permit staging FactorStore must be strictly below its scratch root"
        )
    if factor_data_root == (resolved_scratch_root / "factor_data"):
        raise ResearchCatalogPermitError(
            "legacy scratch_root/factor_data is retired and must not be used as a supplement writer"
        )
    permit_cfg = {
        "research_only": config.get("research_only"),
        "compute": dict(_as_mapping(config.get("compute"), field="factor supplement config.compute")),
        "output": {},
    }
    return issue_research_catalog_execution_permit(
        research_cfg=permit_cfg,
        catalog_path=str(catalog["path"]),
        factor_data_root=factor_data_root,
        factor_ids=[str(item) for item in factor_ids],
    )


def _execution_input_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Resolve explicit read-only DataHub/local inputs for the research stage."""

    cfg = _as_mapping(config.get("inputs"), field="factor supplement config.inputs")
    roots = {
        "raw_data_root": _path_from(cfg.get("raw_data_root"), field="inputs.raw_data_root"),
        "cleaned_data_root": _path_from(cfg.get("cleaned_data_root"), field="inputs.cleaned_data_root"),
    }
    # The supplement is fixed to clean-direct panels.  ``run_factor_pipeline``
    # still requires a positional panel root for API compatibility, but this
    # value is never read in clean-direct mode; bind it to the already audited
    # clean root rather than inventing a cache or a write target.
    roots["panel_data_root"] = roots["cleaned_data_root"]
    scratch_parent = RESEARCH_SCRATCH_PARENT.expanduser().resolve(strict=False)
    bad = [field for field, path in roots.items() if _is_strict_child(path, scratch_parent) or path == scratch_parent]
    if bad:
        raise ValueError(
            "factor supplement inputs must be read-only external inputs, not research scratch roots: "
            + ", ".join(bad)
        )
    missing = [field for field, path in roots.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "factor supplement input root(s) do not exist: " + ", ".join(missing)
        )
    return roots


def _assert_plan_artifacts_current(plan: Mapping[str, Any]) -> None:
    """Reject execution if the catalog or release changed after planning."""

    catalog = _as_mapping(plan.get("catalog"), field="supplement plan.catalog")
    release = _as_mapping(plan.get("release_exclude"), field="supplement plan.release_exclude")
    for label, evidence in (("catalog", catalog), ("release exclusion", release)):
        path = _path_from(evidence.get("path"), field=f"supplement plan {label}.path")
        expected_hash = str(evidence.get("sha256", "")).strip().lower()
        actual_hash = _sha256(path)
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"supplement {label} changed after plan; refusing execution: "
                f"expected={expected_hash}, actual={actual_hash}"
            )


def _staging_factor_store(staging_factor_root: Path) -> FactorStore:
    """Return the ephemeral permit-bound staging store, never a durable root."""

    return FactorStore(staging_factor_root, panel_name="T1430")


def _new_staging_factor_root(scratch_root: Path, *, attempt_id: str) -> Path:
    parent = scratch_root / "staging"
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix=f"{attempt_id}_", dir=str(parent))).resolve(strict=False) / "factor_data"
    if not _is_strict_child(root, scratch_root):  # pragma: no cover - mkdtemp invariant.
        raise RuntimeError("supplement staging root escaped scratch root")
    if root == scratch_root / "factor_data":  # pragma: no cover - explicit legacy guard.
        raise RuntimeError("supplement staging selected the retired scratch FactorStore root")
    return root


def _catalog_factor_map(config: Mapping[str, Any], *, plan: Mapping[str, Any]) -> dict[str, CatalogFactor]:
    catalog_path, factors, evidence = _load_catalog(config)
    expected = _as_mapping(plan.get("catalog"), field="supplement plan.catalog")
    if str(evidence.get("sha256", "")) != str(expected.get("sha256", "")):
        raise RuntimeError("supplement catalog changed after planning")
    if str(catalog_path) != str(expected.get("path", "")):
        raise RuntimeError("supplement catalog path changed after planning")
    by_id = {factor.factor_id: factor for factor in factors}
    if len(by_id) != len(factors):  # pragma: no cover - _load_catalog already guards this.
        raise RuntimeError("supplement catalog factor IDs are not unique")
    return by_id


def _canonical_contract_for(factors: Sequence[CatalogFactor]) -> FactorTableContract:
    columns: list[FactorColumnContract] = []
    for factor in factors:
        if not factor.factor_version or not factor.contract_hash or not factor.output_column:
            raise RuntimeError(
                "canonical factor-library publication requires catalog version/hash/output metadata: "
                f"{factor.factor_id}"
            )
        columns.append(
            FactorColumnContract(
                factor_id=factor.factor_id,
                factor_version=factor.factor_version,
                contract_hash=factor.contract_hash,
                output_column=factor.output_column,
                materialization_contract_origin="current_catalog",
            )
        )
    return FactorTableContract(tuple(columns))


def _require_exact_factor_frame(
    frame: pd.DataFrame,
    *,
    factor_ids: Sequence[str],
    label: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"{label} factor frame is missing or empty")
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["dt", "code"]:
        raise RuntimeError(f"{label} factor frame must preserve a (dt, code) MultiIndex")
    actual = [str(column) for column in frame.columns]
    expected = list(factor_ids)
    if actual != expected:
        raise RuntimeError(
            f"{label} factor columns differ from the exact catalog/release contract: "
            f"expected_count={len(expected)} actual_count={len(actual)}"
        )
    return frame


def _read_live_factor_frame(
    config: Mapping[str, Any],
    *,
    score_day: date,
    live_ids: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    store = _configured_live_factor_table(config)
    path = store.partition_paths(FactorTableKind.LIVE, score_day).parquet_path
    frame = _require_exact_factor_frame(
        store.read_day(FactorTableKind.LIVE, score_day),
        factor_ids=live_ids,
        label="read-only live",
    )
    return frame, {
        "source": "canonical_live_release_read_only",
        "source_root": str(store.root),
        "source_table_id": FactorTableKind.LIVE.value,
        "source_partition": {
            "path": str(path),
            "sha256": _sha256(path),
            "trade_day": score_day.isoformat(),
        },
        "read_only": True,
    }


def _family_publication_evidence(
    *,
    plan: Mapping[str, Any],
    score_day: date,
    attempt_id: str,
    family: str,
    family_ids: Sequence[str],
    coverage: Mapping[str, Any],
    stage_evidence: Mapping[str, Any] | None,
    live_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    catalog = _as_mapping(plan.get("catalog"), field="supplement plan.catalog")
    release = _as_mapping(plan.get("release_exclude"), field="supplement plan.release_exclude")
    ids = list(family_ids)
    per_factor = dict(coverage.get("per_factor", {}))
    return {
        "schema_version": _EXECUTION_MANIFEST_SCHEMA,
        "source": "factor_supplement_2359_canonical_publish",
        "score_day": score_day.isoformat(),
        "attempt_id": attempt_id,
        "family": family,
        "factor_ids": ids,
        "catalog": {"path": str(catalog.get("path", "")), "sha256": str(catalog.get("sha256", ""))},
        "release_exclude": {
            "path": str(release.get("path", "")),
            "sha256": str(release.get("sha256", "")),
        },
        "coverage_quality": {
            "row_count": int(coverage.get("row_count", 0) or 0),
            "all_nan_factor_ids": [factor_id for factor_id in ids if factor_id in set(coverage.get("all_nan_factor_ids", []))],
            "infinite_factor_ids": [factor_id for factor_id in ids if factor_id in set(coverage.get("infinite_factor_ids", []))],
            "per_factor": {factor_id: per_factor.get(factor_id, {}) for factor_id in ids},
        },
        "staging": dict(stage_evidence) if stage_evidence is not None else None,
        "live_input": dict(live_evidence) if live_evidence is not None else None,
        "legacy_scratch_factor_store": "not_used",
    }


def _factor_frame_index_hash(frame: pd.DataFrame) -> str:
    if not isinstance(frame.index, pd.MultiIndex):
        raise RuntimeError("research supplement FactorStore must preserve a MultiIndex")
    normalized = frame.index.to_frame(index=False).astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _factor_column_hashes(frame: pd.DataFrame) -> dict[str, str]:
    """Hash each scratch output column together with its persisted index."""

    hashes: dict[str, str] = {}
    for column in frame.columns:
        series = frame[column]
        values = pd.util.hash_pandas_object(series, index=True, categorize=False).to_numpy()
        digest = hashlib.sha256()
        digest.update(str(series.dtype).encode("utf-8"))
        digest.update(values.tobytes())
        hashes[str(column)] = digest.hexdigest()
    return hashes


def _coverage_quality(frame: pd.DataFrame, *, expected_ids: Sequence[str]) -> dict[str, Any]:
    """Describe coverage without treating expected missing values as failure.

    A factor can be fully materialized yet legitimately unavailable on a sparse
    source day.  That is surfaced as a warning (`all_nan_factor_ids`) rather
    than a compute failure.  Infinite values are different: they violate the
    persisted numeric output contract and remain an integrity failure.
    """

    row_count = int(len(frame))
    per_factor: dict[str, dict[str, Any]] = {}
    all_nan: list[str] = []
    infinite: list[str] = []
    for factor_id in expected_ids:
        series = pd.to_numeric(frame[factor_id], errors="coerce")
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite_mask = np.isfinite(values)
        inf_mask = np.isinf(values)
        finite_count = int(finite_mask.sum())
        inf_count = int(inf_mask.sum())
        nan_count = int(pd.isna(series).sum())
        if bool(pd.isna(series).all()):
            all_nan.append(factor_id)
        if inf_count:
            infinite.append(factor_id)
        per_factor[factor_id] = {
            "finite_count": finite_count,
            "finite_rate": float(finite_count / row_count) if row_count else 0.0,
            "nan_count": nan_count,
            "infinite_count": inf_count,
        }
    return {
        "row_count": row_count,
        "all_nan_factor_ids": all_nan,
        "all_nan_factor_count": len(all_nan),
        "infinite_factor_ids": infinite,
        "infinite_factor_count": len(infinite),
        "per_factor": per_factor,
    }


def _execution_manifest_path(scratch_root: Path, *, score_day: date, attempt_id: str) -> Path:
    return scratch_root / "manifests" / f"{score_day:%Y-%m-%d}" / f"{attempt_id}.json"


def _build_execution_manifest(
    *,
    plan: Mapping[str, Any],
    score_day: date,
    attempt_id: str,
    scratch_root: Path,
    catalog_ids: Sequence[str],
    non_live_ids: Sequence[str],
    result_frame: pd.DataFrame | None,
    canonical_root: Path | None,
    family_write_status: Mapping[str, str],
    logical_publish_status: str | None,
    staging_evidence: Mapping[str, Any] | None,
    staging_removed: bool,
    error: Exception | None,
) -> dict[str, Any]:
    expected_ids = list(catalog_ids)
    observed_ids = list(result_frame.columns) if result_frame is not None and not result_frame.empty else []
    observed_set = set(observed_ids)
    missing_ids = [factor_id for factor_id in expected_ids if factor_id not in observed_set]
    catalog = _as_mapping(plan.get("catalog"), field="supplement plan.catalog")
    release = _as_mapping(plan.get("release_exclude"), field="supplement plan.release_exclude")
    column_hashes = _factor_column_hashes(result_frame) if result_frame is not None and not result_frame.empty else {}
    compute_complete = error is None and not missing_ids and tuple(observed_ids) == tuple(expected_ids)
    coverage = _coverage_quality(result_frame, expected_ids=expected_ids) if compute_complete and result_frame is not None else {
        "row_count": 0,
        "all_nan_factor_ids": [],
        "all_nan_factor_count": 0,
        "infinite_factor_ids": [],
        "infinite_factor_count": 0,
        "per_factor": {},
    }
    integrity_passed = bool(compute_complete) and int(coverage["infinite_factor_count"]) == 0
    complete = bool(compute_complete) and bool(integrity_passed) and logical_publish_status in {
        "published",
        "already_published",
    }
    if not compute_complete or not integrity_passed:
        status = "failed"
    elif int(coverage["all_nan_factor_count"]) > 0:
        status = "completed_with_coverage_gaps"
    else:
        status = "completed"
    return {
        "schema_version": _EXECUTION_MANIFEST_SCHEMA,
        "research_only": True,
        "attempt_id": attempt_id,
        "score_day": score_day.isoformat(),
        "status": status,
        "complete": complete,
        "compute_complete": compute_complete,
        "integrity_passed": integrity_passed,
        "catalog": {
            "path": str(catalog.get("path", "")),
            "sha256": str(catalog.get("sha256", "")),
            "factor_count": int(catalog.get("factor_count", 0) or 0),
        },
        "release_exclude": {
            "path": str(release.get("path", "")),
            "sha256": str(release.get("sha256", "")),
            "factor_count": int(release.get("factor_count", 0) or 0),
        },
        "scratch_root": str(scratch_root),
        "canonical_factor_library": {
            "root": str(canonical_root) if canonical_root is not None else "",
            "table_id": FactorTableKind.FACTOR_LIBRARY.value,
            "family_write_status": dict(family_write_status),
            "logical_publish_status": logical_publish_status or "not_started",
        },
        "ephemeral_staging": {
            **(dict(staging_evidence) if staging_evidence is not None else {}),
            "removed": bool(staging_removed),
        },
        "legacy_scratch_factor_store": {
            "path": str(scratch_root / "factor_data"),
            "writer": "forbidden",
            "accessed": False,
        },
        "factor_output": {
            "row_count": int(len(result_frame)) if result_frame is not None else 0,
            "index_sha256": _factor_frame_index_hash(result_frame) if result_frame is not None and not result_frame.empty else "",
            "columns": observed_ids,
            "column_sha256": column_hashes,
            "column_order_sha256": hashlib.sha256(
                json.dumps(observed_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        },
        "expected_factor_ids": list(expected_ids),
        "non_live_factor_ids": list(non_live_ids),
        "requested_execution_factor_ids": list(non_live_ids),
        "executed_factor_ids": list(non_live_ids) if compute_complete else [],
        "complete_factor_ids": [factor_id for factor_id in expected_ids if factor_id in observed_set],
        "missing_factor_ids": missing_ids,
        "coverage_quality": coverage,
        "error": (
            {"type": type(error).__name__, "message": str(error)} if error is not None else None
        ),
        "side_effect_boundary": {
            "live_runtime": "not_called",
            "live_scheduler": "not_called",
            "database_write": "not_called",
            "model_score": "not_called",
            "trade_list": "not_called",
            "canonical_factor_library_write": "only_when_execute",
            "legacy_scratch_factor_store_write": "forbidden",
            "runtime_audit_root": str(scratch_root),
        },
    }


def execute_one_score_day(
    config: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    score_day: date,
    attempt_id: str,
    scratch_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Compute one non-live stage then publish the complete canonical library day."""

    if str(plan.get("disposition", "")) != "PLAN_READY_NO_EXECUTION":
        raise RuntimeError("cannot execute a blocked factor supplement plan")
    scope = _as_mapping(plan.get("supplement_scope"), field="supplement plan.supplement_scope")
    expected_ids_raw = scope.get("non_live_catalog_factor_ids")
    if not isinstance(expected_ids_raw, list) or not expected_ids_raw:
        raise RuntimeError("supplement plan has no non-live factor IDs")
    non_live_ids = tuple(str(item).strip() for item in expected_ids_raw)
    if len(non_live_ids) != len(set(non_live_ids)) or any(not item for item in non_live_ids):
        raise RuntimeError("supplement plan has invalid non-live factor IDs")
    catalog_by_id = _catalog_factor_map(config, plan=plan)
    catalog_ids = tuple(catalog_by_id)
    if set(non_live_ids).difference(catalog_by_id):
        raise RuntimeError("supplement plan includes non-live factors absent from the current catalog")
    release = _as_mapping(plan.get("release_exclude"), field="supplement plan.release_exclude")
    raw_live_ids = release.get("factor_ids")
    if not isinstance(raw_live_ids, list):
        raise RuntimeError("supplement plan release exclusion has no ordered catalog factor IDs")
    live_ids = tuple(str(item).strip() for item in raw_live_ids)
    live_id_set = set(live_ids)
    if not live_ids or len(live_ids) != len(live_id_set) or any(not item for item in live_ids):
        raise RuntimeError("supplement plan live factor IDs are invalid")
    if not live_id_set.issubset(catalog_by_id) or set(catalog_ids) != set(non_live_ids).union(live_id_set):
        raise RuntimeError("supplement catalog does not partition into live and non-live factor IDs")
    if release.get("release_factor_ids_not_in_catalog"):
        raise RuntimeError("supplement cannot publish a complete catalog day when live release IDs are absent from catalog")

    error: Exception | None = None
    result_frame: pd.DataFrame | None = None
    canonical_root: Path | None = None
    family_write_status: dict[str, str] = {}
    logical_publish_status: str | None = None
    staging_root: Path | None = None
    staging_evidence: dict[str, Any] | None = None
    staging_removed = False
    try:
        _assert_plan_artifacts_current(plan)
        canonical_store = _canonical_factor_library_store(config)
        canonical_root = canonical_store.root
        staging_root = _new_staging_factor_root(scratch_root, attempt_id=attempt_id)
        permit = issue_catalog_execution_permit(
            config,
            plan=plan,
            scratch_root=scratch_root,
            staging_factor_root=staging_root,
        )
        specs = build_permitted_factor_specs(permit)
        if tuple(spec.name for spec in specs) != non_live_ids:
            raise RuntimeError("permit-derived FactorSpecs differ from the planned catalog scope")
        inputs = _execution_input_paths(config)
        panel_cfg = dict(load_config_file("panel"))
        if str(panel_cfg.get("panel_name", "")).strip() != "T1430":
            raise RuntimeError("research supplement requires panel_config.panel_name='T1430'")
        if int(panel_cfg.get("lead_minutes", -1)) != 1:
            raise RuntimeError("research supplement requires panel_config.lead_minutes=1 for strict 14:29 visibility")
        factor_cfg = _as_mapping(config.get("compute"), field="factor supplement config.compute")
        run_factor_pipeline(
            inputs["panel_data_root"],
            staging_root,
            score_day,
            score_day,
            panel_name="T1430",
            refresh=False,
            overwrite=False,
            workers=1,
            factor_workers=int(config.get("factor_workers", 1) or 1),
            raw_data_root=inputs["raw_data_root"],
            cleaned_data_root=inputs["cleaned_data_root"],
            context_cfg={"mode": "auto"},
            compute_cfg=dict(factor_cfg),
            panel_source_cfg={"mode": "clean_direct"},
            panel_build_cfg=panel_cfg,
            research_catalog_execution_permit=permit,
            specs=specs,
        )
        stage_path = _staging_factor_store(staging_root).day_path(score_day)
        stage_frame = _require_exact_factor_frame(
            _staging_factor_store(staging_root).read_day(score_day),
            factor_ids=non_live_ids,
            label="ephemeral non-live staging",
        )
        stage_evidence = {
            "path": str(stage_path),
            "sha256": _sha256(stage_path),
            "row_count": int(len(stage_frame)),
            "factor_count": len(non_live_ids),
            "ephemeral": True,
            "permit_catalog_sha256": permit.catalog_sha256,
        }
        live_frame, live_evidence = _read_live_factor_frame(
            config,
            score_day=score_day,
            live_ids=live_ids,
        )
        if not stage_frame.index.equals(live_frame.index):
            raise RuntimeError(
                "supplement staging and read-only live factor inputs have different (dt, code) indexes; "
                "refusing to fabricate a combined catalog day"
            )
        combined = pd.concat([live_frame, stage_frame], axis=1)
        if combined.columns.duplicated().any():
            raise RuntimeError("supplement live and non-live factor columns overlap")
        result_frame = _require_exact_factor_frame(
            combined.loc[:, list(catalog_ids)],
            factor_ids=catalog_ids,
            label="combined canonical factor-library",
        )
        coverage = _coverage_quality(result_frame, expected_ids=catalog_ids)
        if int(coverage["infinite_factor_count"]) > 0:
            raise RuntimeError("supplement combined factor output contains Inf values")
        families: dict[str, list[CatalogFactor]] = {}
        for factor_id in catalog_ids:
            factor = catalog_by_id[factor_id]
            family = str(factor.primary_family).strip()
            if not family:
                raise RuntimeError(f"catalog factor has no primary_family: {factor_id}")
            families.setdefault(family, []).append(factor)
        contracts: dict[str, FactorTableContract] = {}
        for family, factors in sorted(families.items()):
            factor_ids = [factor.factor_id for factor in factors]
            contract = _canonical_contract_for(factors)
            contracts[family] = contract
            source_roles = {
                "stage": any(factor_id in set(non_live_ids) for factor_id in factor_ids),
                "live": any(factor_id in set(live_ids) for factor_id in factor_ids),
            }
            evidence = _family_publication_evidence(
                plan=plan,
                score_day=score_day,
                attempt_id=attempt_id,
                family=family,
                family_ids=factor_ids,
                coverage=coverage,
                stage_evidence=stage_evidence if source_roles["stage"] else None,
                live_evidence=live_evidence if source_roles["live"] else None,
            )
            outcome = canonical_store.write_day(
                FactorTableKind.FACTOR_LIBRARY,
                score_day,
                result_frame.loc[:, list(contract.output_columns)],
                contract=contract,
                family=family,
                source_evidence=evidence,
            )
            family_write_status[family] = outcome.status
        logical_publish_status = canonical_store.publish_factor_library_day(
            score_day,
            family_contracts=contracts,
        )
    except Exception as exc:
        error = exc
    finally:
        if staging_root is not None:
            stage_container = staging_root.parent
            if _is_strict_child(stage_container, scratch_root) and stage_container.parent == scratch_root / "staging":
                try:
                    shutil.rmtree(stage_container, ignore_errors=False)
                    staging_removed = not stage_container.exists()
                except Exception as cleanup_exc:
                    staging_removed = False
                    if error is None:
                        error = RuntimeError(
                            "supplement canonical publish completed but ephemeral staging cleanup failed: "
                            f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        )
            else:  # pragma: no cover - defensive path proof.
                if error is None:
                    error = RuntimeError("supplement refused to clean an unverified staging directory")
    manifest = _build_execution_manifest(
        plan=plan,
        score_day=score_day,
        attempt_id=attempt_id,
        scratch_root=scratch_root,
        catalog_ids=catalog_ids,
        non_live_ids=non_live_ids,
        result_frame=result_frame,
        canonical_root=canonical_root,
        family_write_status=family_write_status,
        logical_publish_status=logical_publish_status,
        staging_evidence=staging_evidence,
        staging_removed=staging_removed,
        error=error,
    )
    manifest_path = _execution_manifest_path(scratch_root, score_day=score_day, attempt_id=attempt_id)
    _write_json_atomic(manifest_path, manifest)
    return manifest, manifest_path


def _acquire_lock(scratch_root: Path, *, score_day: date) -> _LockHandle:
    lock_path = scratch_root / "locks" / f"{score_day:%Y-%m-%d}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    payload = {
        "schema_version": SUPPLEMENT_SCHEMA,
        "token": token,
        "pid": os.getpid(),
        "score_day": score_day.isoformat(),
        "acquired_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        evidence = _read_json_safely(lock_path)
        raise FactorSupplementLockError(
            "factor supplement lock already exists for "
            f"{score_day}: path={lock_path} evidence={evidence}"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            lock_path.unlink()
        except OSError:
            pass
        raise
    return _LockHandle(path=lock_path, token=token)


def _release_lock(lock: _LockHandle) -> None:
    evidence = _read_json_safely(lock.path)
    if evidence.get("token") != lock.token:
        raise RuntimeError(f"refusing to release a lock no longer owned by this attempt: {lock.path}")
    lock.path.unlink(missing_ok=False)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _failure_plan(*, score_day: date, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": SUPPLEMENT_SCHEMA,
        "research_only": True,
        "score_day": score_day.isoformat(),
        "disposition": "BLOCKED_PRECHECK_ERROR",
        "blocking_reasons": [f"precheck_error:{type(error).__name__}"],
        "error": str(error),
        "execution_boundary": {
            "allowed_mode": "execute_one_score_day_research_only",
            "factor_compute": "not_started",
            "factor_store_write": "forbidden",
            "model_score": "forbidden",
            "trade_list": "forbidden",
            "database_write": "forbidden",
            "live_scheduler": "forbidden",
        },
    }


def run(
    config: Mapping[str, Any],
    *,
    score_day: date | str,
    mode: str = "plan",
    scratch_root: Path | str | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Run an isolated plan, dry-run ledger, or one-score-day research stage.

    ``plan`` is read-only. ``dry-run`` writes only a ledger. ``execute`` uses
    the catalog-bound permit with ephemeral scratch staging, then publishes
    the canonical factor-library logical day.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in _SUPPORTED_MODES:
        raise ValueError(f"factor supplement mode must be one of {sorted(_SUPPORTED_MODES)}")
    day = parse_date(score_day)
    if normalized_mode == "plan":
        return build_plan(config, score_day=day), None

    root = _validate_scratch_root(
        _path_from(scratch_root, field="scratch_root")
        if scratch_root is not None
        else configured_scratch_root(config)
    )
    lock = _acquire_lock(root, score_day=day)
    attempt_id = f"{datetime.now():%Y%m%dT%H%M%S}_{os.getpid()}_{uuid4().hex[:8]}"
    ledger_path = root / "ledger" / f"{day:%Y-%m-%d}" / f"{attempt_id}.json"
    try:
        try:
            plan = build_plan(config, score_day=day)
        except Exception as exc:
            plan = _failure_plan(score_day=day, error=exc)
        execution_manifest_path: Path | None = None
        if normalized_mode == "execute":
            if str(plan.get("disposition", "")) != "PLAN_READY_NO_EXECUTION":
                plan = {
                    **plan,
                    "execution": {
                        "requested": True,
                        "started": False,
                        "status": "blocked_by_plan",
                        "manifest_path": "",
                    },
                }
            else:
                try:
                    manifest, execution_manifest_path = execute_one_score_day(
                        config,
                        plan=plan,
                        score_day=day,
                        attempt_id=attempt_id,
                        scratch_root=root,
                    )
                except Exception as exc:
                    manifest = {
                        "status": "failed_before_manifest",
                        "complete": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                    plan = {
                        **plan,
                        "disposition": "EXECUTION_FAILED",
                        "blocking_reasons": [*list(plan.get("blocking_reasons", [])), "execution_exception"],
                        "execution": {
                            "requested": True,
                            "started": True,
                            "status": "failed_before_manifest",
                            "manifest_path": "",
                            "error": manifest["error"],
                        },
                    }
                else:
                    if bool(manifest.get("complete", False)):
                        manifest_status = str(manifest.get("status", "completed")).strip()
                        disposition = (
                            "EXECUTION_COMPLETED_WITH_COVERAGE_GAPS"
                            if manifest_status == "completed_with_coverage_gaps"
                            else "EXECUTION_COMPLETED"
                        )
                        plan = {
                            **plan,
                            "disposition": disposition,
                            "execution": {
                                "requested": True,
                                "started": True,
                                "status": manifest_status,
                                "manifest_path": str(execution_manifest_path),
                                "coverage_warning": manifest_status == "completed_with_coverage_gaps",
                            },
                        }
                    else:
                        reasons = [*list(plan.get("blocking_reasons", [])), "execution_incomplete_or_failed"]
                        plan = {
                            **plan,
                            "disposition": "EXECUTION_FAILED",
                            "blocking_reasons": reasons,
                            "execution": {
                                "requested": True,
                                "started": True,
                                "status": str(manifest.get("status", "failed")),
                                "manifest_path": str(execution_manifest_path),
                            },
                        }
        ledger = {
            "schema_version": LEDGER_SCHEMA,
            "research_only": True,
            "attempt_id": attempt_id,
            "mode": normalized_mode,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "scratch_root": str(root),
            "ledger_path": str(ledger_path),
            "execution_manifest_path": str(execution_manifest_path) if execution_manifest_path else "",
            "plan": plan,
        }
        _write_json_atomic(ledger_path, ledger)
        return plan, ledger_path
    finally:
        _release_lock(lock)


def compact_summary(plan: Mapping[str, Any], *, ledger_path: Path | None = None) -> dict[str, Any]:
    """Produce a terminal-safe summary without printing hundreds of factor IDs."""

    scope = plan.get("supplement_scope") if isinstance(plan.get("supplement_scope"), Mapping) else {}
    admission = (
        plan.get("research_python_admission")
        if isinstance(plan.get("research_python_admission"), Mapping)
        else {}
    )
    datahub = plan.get("datahub") if isinstance(plan.get("datahub"), Mapping) else {}
    execution = plan.get("execution") if isinstance(plan.get("execution"), Mapping) else {}
    disposition = str(plan.get("disposition", "")).strip()
    execution_requested = bool(execution.get("requested", False))
    execution_started = bool(execution.get("started", False))
    execution_status = str(execution.get("status", "")).strip().lower()
    if disposition in {"EXECUTION_COMPLETED", "EXECUTION_COMPLETED_WITH_COVERAGE_GAPS"} or execution_status in {
        "completed",
        "completed_with_coverage_gaps",
    }:
        permit_issued = True
        factor_compute = "completed_with_coverage_gaps" if execution_status == "completed_with_coverage_gaps" else "completed"
    elif execution_started and execution_status in {"failed", "incomplete"}:
        # These terminal states are written only after the permit-bound stage
        # returned an execution manifest, so compute had begun.
        permit_issued = True
        factor_compute = execution_status
    elif execution_requested and execution_started:
        permit_issued = bool(admission.get("permit_issued", False))
        factor_compute = "not_started" if execution_status == "failed_before_manifest" else "started"
    else:
        # Keep plan/dry-run and data-gate-blocked behavior unchanged.
        permit_issued = bool(admission.get("permit_issued", False))
        factor_compute = "not_started"
    return {
        "research_only": True,
        "score_day": plan.get("score_day"),
        "disposition": disposition,
        "blocking_reasons": list(plan.get("blocking_reasons", [])),
        "datahub_passed": datahub.get("passed"),
        "catalog_factor_count": scope.get("catalog_factor_count"),
        "live_excluded_factor_count": scope.get("live_excluded_factor_count"),
        "non_live_catalog_factor_count": scope.get("non_live_catalog_factor_count"),
        "research_python_permit_required": admission.get("permit_required"),
        "research_python_permit_issued": permit_issued,
        "execution": dict(execution),
        "ledger_path": str(ledger_path) if ledger_path is not None else "",
        "factor_compute": factor_compute,
    }


__all__ = [
    "DEFAULT_CONFIG_REF",
    "FactorSupplementError",
    "FactorSupplementLockError",
    "RESEARCH_SCRATCH_PARENT",
    "SUPPLEMENT_SCHEMA",
    "build_plan",
    "compact_summary",
    "configured_schedule_time",
    "configured_scratch_root",
    "execute_one_score_day",
    "issue_catalog_execution_permit",
    "load_supplement_config",
    "run",
]
