"""Build or verify the source-controlled full factor and operator catalogue.

P0/P1 boundary:
* reads frozen research metadata and canonical operator implementation modules;
* writes only beneath ``factor_engine/`` when ``--write`` is supplied;
* never modifies runtime operator source, ``FactorRegistry``, live configuration, Rust,
  scheduler state, FactorStore, or database assets.

The output creates one parameterized factor definition and one contract for
every factor instance, plus one governance contract per registered operator.
Runtime source lives under ``cbond_on.domain.factors.operators``; the
top-level ``factor_engine`` remains the identity/provenance layer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import pprint
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import json5


sys.dont_write_bytecode = True

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbond_on.core.registry import FactorRegistry  # noqa: E402
from cbond_on.domain.factor_catalog.validation import (  # noqa: E402
    FactorCatalogValidationError,
    json_sha256,
    validate_factor_catalog,
)


EXPECTED_RESEARCH_FACTOR_COUNT = 773
EXPECTED_LEGACY_LIVE_FACTOR_COUNT = 27
EXPECTED_FACTOR_COUNT = EXPECTED_RESEARCH_FACTOR_COUNT + EXPECTED_LEGACY_LIVE_FACTOR_COUNT
EXPECTED_OPERATOR_COUNT = 266
SNAPSHOT_ID = "canonical_factor_catalog_20260826_800"
_SAFE_COMPONENT = re.compile(r"^[a-z][a-z0-9_]*$")
_OPERATOR_MODULE_PREFIX = "cbond_on.domain.factors.operators."
_RESEARCH_MODULE_PREFIX = _OPERATOR_MODULE_PREFIX + "research_"
_LEGACY_LIVE27_PACK = _REPO_ROOT / "cbond_on" / "config" / "factor" / "packs" / "live_screened_no_winsor_27.json5"
_LEGACY_FACTOR_REGISTRY = _REPO_ROOT / "cbond_on" / "factor_contracts" / "registry.json5"
_LIVE50_PACK = _REPO_ROOT / "cbond_on" / "config" / "factor" / "packs" / "live_screened_no_winsor_50_20260805.json5"
_LIVE50_PROFILE = _REPO_ROOT / "cbond_on" / "factor_contracts" / "profiles" / "live50_rust50_20260806.json5"
LIVE50_RELEASE_ID = "live50_rust50_operator_source_20260826"
LIVE50_PROFILE_ID = "live50_rust50_20260806"
_OPERATOR_MIGRATION_MANIFEST = (
    _REPO_ROOT / "factor_engine" / "migrations" / "operator_source_migration_20260826.json"
)


class CatalogBuildError(RuntimeError):
    """Raised for any ambiguity or provenance break in catalogue generation."""


@dataclass(frozen=True)
class FrozenFactor:
    factor_id: str
    primary_family: str
    source_set: str
    source_position: int | None
    source_root: str
    fixed_params: Mapping[str, Any]
    frozen_operator_id: str | None = None
    rust_contract_id: str | None = None


@dataclass(frozen=True)
class Candidate:
    factor_id: str
    primary_family: str
    operator_id: str
    implementation_module: str
    implementation_path: str
    implementation_sha256: str
    legacy_implementation_path: str
    legacy_implementation_sha256: str
    operator_version: str
    hypothesis: str
    fixed_params: Mapping[str, Any]
    rust_contract_id: str | None = None


@dataclass(frozen=True)
class SourceMigrationRecord:
    """Auditable relationship between a retired source path and runtime operator code."""

    legacy_path: str
    legacy_sha256: str
    target_path: str
    target_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _relative_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError as exc:
        raise CatalogBuildError(f"implementation escaped repository: {path}") from exc


def _safe_component(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not _SAFE_COMPONENT.fullmatch(text):
        raise CatalogBuildError(f"unsafe {label}: {value!r}")
    return text


def _source_snapshot_path() -> Path:
    return _REPO_ROOT / "factor_engine" / "catalog" / "frozen_canonical_20260826_800.json"


def _overrides_path() -> Path:
    return _REPO_ROOT / "factor_engine" / "catalog" / "manual_overrides.json"


def _factor_engine_root() -> Path:
    return _REPO_ROOT / "factor_engine"


def _operator_source_root() -> Path:
    """Return the only runtime Python source root for reusable operators."""

    return _REPO_ROOT / "cbond_on" / "domain" / "factors" / "operators"


def _operator_contract_path(operator_id: str) -> Path:
    return Path("operators") / operator_id / "contract.json"


def _operator_definition_path(operator_id: str) -> Path:
    return Path("operators") / operator_id / "definition.py"


def _load_source_migration_records() -> dict[str, SourceMigrationRecord]:
    """Load the deterministic source relocation ledger before generating contracts.

    A physical source move changes module/path evidence even when the numerical
    formula is unchanged.  The migration record is therefore part of every
    new operator contract; it prevents path-only refactors from becoming
    unauditable silent rewrites.
    """

    try:
        raw = json.loads(_OPERATOR_MIGRATION_MANIFEST.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogBuildError(
            "operator-source migration manifest is required before catalog generation: "
            f"{_OPERATOR_MIGRATION_MANIFEST}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CatalogBuildError("operator-source migration manifest is invalid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "operator_source_migration/v1":
        raise CatalogBuildError("unexpected operator-source migration manifest schema")
    if str(raw.get("new_module_prefix", "")).strip() != _OPERATOR_MODULE_PREFIX.rstrip("."):
        raise CatalogBuildError("operator-source migration target prefix drift")
    rows = raw.get("files")
    if not isinstance(rows, list):
        raise CatalogBuildError("operator-source migration manifest files must be a list")
    result: dict[str, SourceMigrationRecord] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise CatalogBuildError("operator-source migration row must be an object")
        legacy_path = str(item.get("legacy_path", "")).strip()
        legacy_sha256 = str(item.get("legacy_sha256", "")).strip().lower()
        target_path = str(item.get("target_path", "")).strip()
        target_sha256 = str(item.get("target_sha256", "")).strip().lower()
        if (
            not legacy_path
            or not target_path
            or len(legacy_sha256) != 64
            or len(target_sha256) != 64
        ):
            raise CatalogBuildError("invalid operator-source migration row")
        if target_path in result:
            raise CatalogBuildError(f"duplicate migrated operator target path: {target_path}")
        target_file = _REPO_ROOT / target_path
        if not target_file.is_file() or _sha256_file(target_file) != target_sha256:
            raise CatalogBuildError(f"migrated operator source hash drift: {target_path}")
        result[target_path] = SourceMigrationRecord(
            legacy_path=legacy_path,
            legacy_sha256=legacy_sha256,
            target_path=target_path,
            target_sha256=target_sha256,
        )
    if len(result) != 290:
        raise CatalogBuildError(
            f"operator-source migration expected 290 Python files, found {len(result)}"
        )
    return result


def _load_csv_snapshot(
    *,
    source_catalog: Path,
    source_audit: Path,
    screen_manifest: Path | None,
    legacy_live_pack: Path,
    legacy_factor_registry: Path,
) -> dict[str, Any]:
    source_catalog = source_catalog.resolve(strict=True)
    source_audit = source_audit.resolve(strict=True)
    legacy_live_pack = legacy_live_pack.resolve(strict=True)
    legacy_factor_registry = legacy_factor_registry.resolve(strict=True)
    with source_catalog.open(encoding="utf-8-sig", newline="") as handle:
        catalog_rows = list(csv.DictReader(handle))
    with source_audit.open(encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))

    if not catalog_rows or set(catalog_rows[0]) != {"family", "factor"}:
        raise CatalogBuildError("source factor catalogue must contain exactly family,factor")
    audit_by_factor: dict[str, tuple[int, str]] = {}
    for row in audit_rows:
        factor_id = str(row.get("factor", "")).strip()
        source_root = str(row.get("source_root", "")).strip()
        try:
            position = int(row.get("source_position", ""))
        except (TypeError, ValueError) as exc:
            raise CatalogBuildError(f"invalid source_position for {factor_id!r}") from exc
        if not factor_id or not source_root or factor_id in audit_by_factor:
            raise CatalogBuildError(f"invalid or duplicate source-audit entry: {factor_id!r}")
        audit_by_factor[factor_id] = (position, source_root)

    factors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in catalog_rows:
        factor_id = _safe_component(str(row.get("factor", "")), label="factor_id")
        family = _safe_component(str(row.get("family", "")), label="family")
        if factor_id in seen:
            raise CatalogBuildError(f"duplicate factor in frozen catalogue: {factor_id}")
        seen.add(factor_id)
        audit = audit_by_factor.get(factor_id)
        if audit is None:
            raise CatalogBuildError(f"frozen factor lacks source audit: {factor_id}")
        factors.append(
            {
                "factor_id": factor_id,
                "primary_family": family,
                "source_set": "research773",
                "source_position": audit[0],
                "source_root": audit[1],
                "fixed_params": {"signal": factor_id, "family": family},
                "frozen_operator_id": None,
                "rust_contract_id": None,
            }
        )
    if len(factors) != EXPECTED_RESEARCH_FACTOR_COUNT:
        raise CatalogBuildError(
            "frozen research catalogue expected "
            f"{EXPECTED_RESEARCH_FACTOR_COUNT} factors, found {len(factors)}"
        )
    if set(audit_by_factor) != seen:
        missing = sorted(seen.difference(audit_by_factor))
        extra = sorted(set(audit_by_factor).difference(seen))
        raise CatalogBuildError(f"source-audit mismatch: missing={missing}, extra={extra}")

    try:
        with legacy_live_pack.open(encoding="utf-8") as handle:
            legacy_pack_payload = json5.load(handle)
        with legacy_factor_registry.open(encoding="utf-8") as handle:
            registry_payload = json5.load(handle)
    except Exception as exc:
        raise CatalogBuildError("cannot parse source-controlled legacy live metadata") from exc
    if not isinstance(legacy_pack_payload, dict) or not isinstance(legacy_pack_payload.get("factors"), list):
        raise CatalogBuildError("legacy live27 pack must contain factors")
    if not isinstance(registry_payload, dict) or not isinstance(registry_payload.get("factors"), list):
        raise CatalogBuildError("factor contract registry must contain factors")
    registry_by_name: dict[str, Mapping[str, Any]] = {}
    for item in registry_payload["factors"]:
        if not isinstance(item, dict):
            raise CatalogBuildError("factor contract registry row must be object")
        name = str(item.get("name", "")).strip()
        if not name or name in registry_by_name:
            raise CatalogBuildError(f"invalid factor contract registry name: {name!r}")
        registry_by_name[name] = item
    legacy_seen: set[str] = set()
    for item in legacy_pack_payload["factors"]:
        if not isinstance(item, dict):
            raise CatalogBuildError("legacy live27 factor row must be object")
        factor_id = _safe_component(str(item.get("name", "")), label="legacy factor_id")
        operator_id = _safe_component(str(item.get("factor", "")), label="legacy operator_id")
        if factor_id in seen or factor_id in legacy_seen:
            raise CatalogBuildError(f"legacy live27 factor overlaps or duplicates a factor: {factor_id}")
        registry_row = registry_by_name.get(factor_id)
        if registry_row is None:
            raise CatalogBuildError(f"legacy live27 factor missing registry family: {factor_id}")
        if str(registry_row.get("implementation", "")).strip() != operator_id:
            raise CatalogBuildError(f"legacy live27 implementation drift: {factor_id}")
        family = _safe_component(str(registry_row.get("family", "")), label="legacy family")
        fixed_params_raw = item.get("params", {})
        if not isinstance(fixed_params_raw, dict):
            raise CatalogBuildError(f"legacy live27 params must be object: {factor_id}")
        fixed_params = json.loads(json.dumps(fixed_params_raw, ensure_ascii=False))
        factors.append(
            {
                "factor_id": factor_id,
                "primary_family": family,
                "source_set": "legacy_live27",
                "source_position": None,
                "source_root": _relative_repo_path(legacy_live_pack),
                "fixed_params": fixed_params,
                "frozen_operator_id": operator_id,
                "rust_contract_id": None,
            }
        )
        legacy_seen.add(factor_id)
    if len(legacy_seen) != EXPECTED_LEGACY_LIVE_FACTOR_COUNT:
        raise CatalogBuildError(
            f"legacy live pack expected {EXPECTED_LEGACY_LIVE_FACTOR_COUNT} factors, found {len(legacy_seen)}"
        )
    if len(factors) != EXPECTED_FACTOR_COUNT:
        raise CatalogBuildError(f"canonical snapshot expected {EXPECTED_FACTOR_COUNT} factors, found {len(factors)}")

    source_artifacts: dict[str, Any] = {
        "family_catalog": {
            "path": str(source_catalog),
            "sha256": _sha256_file(source_catalog),
        },
        "factor_source_audit": {
            "path": str(source_audit),
            "sha256": _sha256_file(source_audit),
        },
        "legacy_live27_pack": {
            "path": _relative_repo_path(legacy_live_pack),
            "sha256": _sha256_file(legacy_live_pack),
        },
        "legacy_factor_registry": {
            "path": _relative_repo_path(legacy_factor_registry),
            "sha256": _sha256_file(legacy_factor_registry),
        },
    }
    if screen_manifest is not None:
        manifest_path = screen_manifest.resolve(strict=True)
        source_artifacts["screen_manifest"] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
    return {
        "schema_version": "frozen_factor_catalog/v1",
        "snapshot_id": SNAPSHOT_ID,
        "factor_count": len(factors),
        "source_artifacts": source_artifacts,
        "time_contract": {
            "panel_name": "T1430",
            "factor_time": "14:30",
            "research_label_time_reference": "14:42",
            "scope": "canonical research773 plus legacy_live27 provenance; not live admission",
        },
        "factors": factors,
    }


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogBuildError(f"frozen snapshot does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogBuildError(f"frozen snapshot is invalid JSON: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "frozen_factor_catalog/v1":
        raise CatalogBuildError(f"unexpected frozen snapshot schema: {path}")
    factors = raw.get("factors")
    if not isinstance(factors, list) or raw.get("factor_count") != len(factors):
        raise CatalogBuildError("frozen snapshot factor_count mismatch")
    if len(factors) != EXPECTED_FACTOR_COUNT:
        raise CatalogBuildError(
            f"frozen snapshot expected {EXPECTED_FACTOR_COUNT} factors, found {len(factors)}"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in factors:
        if not isinstance(row, dict):
            raise CatalogBuildError("frozen snapshot factor row must be object")
        factor_id = _safe_component(str(row.get("factor_id", "")), label="factor_id")
        family = _safe_component(str(row.get("primary_family", "")), label="family")
        if factor_id in seen:
            raise CatalogBuildError(f"duplicate factor in frozen snapshot: {factor_id}")
        seen.add(factor_id)
        source_set = str(row.get("source_set", "")).strip()
        if source_set not in {"research773", "legacy_live27"}:
            raise CatalogBuildError(f"invalid source_set for {factor_id}: {source_set!r}")
        raw_position = row.get("source_position")
        if source_set == "research773":
            try:
                position = int(raw_position)
            except (TypeError, ValueError) as exc:
                raise CatalogBuildError(f"invalid research source_position for {factor_id}") from exc
        else:
            if raw_position is not None:
                raise CatalogBuildError(f"legacy source_position must be null for {factor_id}")
            position = None
        source_root = str(row.get("source_root", "")).strip()
        if not source_root:
            raise CatalogBuildError(f"missing source_root for {factor_id}")
        fixed_params = row.get("fixed_params")
        if not isinstance(fixed_params, dict):
            raise CatalogBuildError(f"fixed_params must be object for {factor_id}")
        frozen_operator_raw = row.get("frozen_operator_id")
        frozen_operator_id = None if frozen_operator_raw is None else _safe_component(
            str(frozen_operator_raw), label="frozen_operator_id"
        )
        rust_contract_raw = row.get("rust_contract_id")
        rust_contract_id = None if rust_contract_raw is None else str(rust_contract_raw).strip()
        if source_set == "research773":
            if frozen_operator_id is not None or fixed_params != {
                "signal": factor_id,
                "family": family,
            }:
                raise CatalogBuildError(f"invalid research frozen mapping for {factor_id}")
        elif frozen_operator_id is None:
            raise CatalogBuildError(f"legacy factor missing frozen_operator_id: {factor_id}")
        normalized.append(
            {
                "factor_id": factor_id,
                "primary_family": family,
                "source_set": source_set,
                "source_position": position,
                "source_root": source_root,
                "fixed_params": fixed_params,
                "frozen_operator_id": frozen_operator_id,
                "rust_contract_id": rust_contract_id,
            }
        )
    if sum(row["source_set"] == "research773" for row in normalized) != EXPECTED_RESEARCH_FACTOR_COUNT:
        raise CatalogBuildError("frozen snapshot research773 count mismatch")
    if sum(row["source_set"] == "legacy_live27" for row in normalized) != EXPECTED_LEGACY_LIVE_FACTOR_COUNT:
        raise CatalogBuildError("frozen snapshot legacy_live27 count mismatch")
    out = dict(raw)
    out["factors"] = normalized
    return out


def _frozen_factors(snapshot: Mapping[str, Any]) -> list[FrozenFactor]:
    rows = snapshot.get("factors")
    if not isinstance(rows, list):
        raise CatalogBuildError("snapshot factors must be a list")
    return [
        FrozenFactor(
            factor_id=str(row["factor_id"]),
            primary_family=str(row["primary_family"]),
            source_set=str(row["source_set"]),
            source_position=(int(row["source_position"]) if row["source_position"] is not None else None),
            source_root=str(row["source_root"]),
            fixed_params=dict(row["fixed_params"]),
            frozen_operator_id=(str(row["frozen_operator_id"]) if row["frozen_operator_id"] is not None else None),
            rust_contract_id=(str(row["rust_contract_id"]) if row["rust_contract_id"] is not None else None),
        )
        for row in rows
    ]


def _read_overrides(path: Path) -> dict[str, dict[str, str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogBuildError(f"manual override file missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogBuildError(f"manual override file invalid: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != "factor_catalog_manual_overrides/v1":
        raise CatalogBuildError("unexpected manual override schema")
    entries = raw.get("overrides")
    if not isinstance(entries, list):
        raise CatalogBuildError("manual overrides must be a list")
    result: dict[str, dict[str, str]] = {}
    required = ("factor_id", "primary_family", "operator_id", "implementation_module", "reason")
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogBuildError("manual override entry must be an object")
        normalized = {field: str(entry.get(field, "")).strip() for field in required}
        if any(not value for value in normalized.values()):
            raise CatalogBuildError(f"manual override missing field: {entry!r}")
        factor_id = normalized["factor_id"]
        if factor_id in result:
            raise CatalogBuildError(f"duplicate manual override: {factor_id}")
        result[factor_id] = normalized
    return result


def _all_research_modules() -> Iterable[str]:
    operators_root = _operator_source_root()
    for path in sorted(operators_root.glob("research_*.py")):
        yield _RESEARCH_MODULE_PREFIX + path.stem.removeprefix("research_")


def _all_definition_modules() -> Iterable[str]:
    operators_root = _operator_source_root()
    for path in sorted(operators_root.glob("*.py")):
        if path.name == "__init__.py" or path.name.startswith("_"):
            continue
        yield _OPERATOR_MODULE_PREFIX + path.stem


def _discover_all_static_operators() -> dict[str, dict[str, Any]]:
    """Load every static operator only inside this isolated tool process."""

    migration_records = _load_source_migration_records()
    importlib.import_module(_OPERATOR_MODULE_PREFIX.rstrip("."))
    default_loaded = {str(operator_id) for operator_id in FactorRegistry.names()}
    for module_name in _all_definition_modules():
        importlib.import_module(module_name)
    operator_ids = sorted(str(operator_id) for operator_id in FactorRegistry.names())
    if len(operator_ids) != EXPECTED_OPERATOR_COUNT:
        raise CatalogBuildError(
            f"expected {EXPECTED_OPERATOR_COUNT} static operators, discovered {len(operator_ids)}"
        )
    rows: dict[str, dict[str, Any]] = {}
    for operator_id in operator_ids:
        operator_cls = FactorRegistry.get(operator_id)
        implementation_module = str(operator_cls.__module__)
        implementation_file = inspect.getsourcefile(operator_cls)
        if not implementation_file:
            raise CatalogBuildError(f"cannot resolve source file for operator {operator_id}")
        implementation_path = Path(implementation_file).resolve()
        implementation_rel = _relative_repo_path(implementation_path)
        migration = migration_records.get(implementation_rel)
        if migration is None:
            raise CatalogBuildError(
                "operator implementation lacks a migration record: "
                f"{implementation_rel}"
            )
        implementation_mod = importlib.import_module(implementation_module)
        rows[operator_id] = {
            "operator_id": operator_id,
            "operator_class": str(operator_cls.__name__),
            "operator_version": str(
                getattr(implementation_mod, "CATALOG_VERSION", "legacy-unversioned")
            ),
            "requires_stock_panel": bool(getattr(operator_cls, "requires_stock_panel", False)),
            "requires_bond_stock_map": bool(
                getattr(operator_cls, "requires_bond_stock_map", False)
            ),
            "implementation_module": implementation_module,
            "implementation_path": implementation_rel,
            "implementation_sha256": _sha256_file(implementation_path),
            "legacy_implementation_path": migration.legacy_path,
            "legacy_implementation_sha256": migration.legacy_sha256,
            "language": "python_runtime_operator",
            "availability_status": "registered_static_operator",
            "default_import_status": (
                "default_loaded" if operator_id in default_loaded else "explicit_module_only"
            ),
        }
    return rows


def _candidate_from_operator(
    *,
    factor: FrozenFactor,
    primary_family: str,
    operator_id: str,
    operator_metadata: Mapping[str, Mapping[str, Any]],
    hypothesis: str,
) -> Candidate:
    metadata = operator_metadata.get(operator_id)
    if metadata is None:
        raise CatalogBuildError(
            f"factor {factor.factor_id} references unknown static operator {operator_id}"
        )
    return Candidate(
        factor_id=factor.factor_id,
        primary_family=primary_family,
        operator_id=operator_id,
        implementation_module=str(metadata["implementation_module"]),
        implementation_path=str(metadata["implementation_path"]),
        implementation_sha256=str(metadata["implementation_sha256"]),
        legacy_implementation_path=str(metadata["legacy_implementation_path"]),
        legacy_implementation_sha256=str(metadata["legacy_implementation_sha256"]),
        operator_version=str(metadata["operator_version"]),
        hypothesis=hypothesis,
        fixed_params=dict(factor.fixed_params),
        rust_contract_id=factor.rust_contract_id,
    )


def _discover_candidates(
    frozen: Sequence[FrozenFactor],
    *,
    operator_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[Candidate]]:
    """Discover legacy mappings without registering or exporting new factors.

    Importing a legacy research module registers its existing shared operator in
    this isolated tool process only.  The generated wrappers never import it.
    """

    research_by_id = {factor.factor_id: factor for factor in frozen if factor.source_set == "research773"}
    candidates: dict[str, dict[tuple[str, str, str], Candidate]] = {
        factor.factor_id: {} for factor in frozen
    }
    for module_name in _all_research_modules():
        module = importlib.import_module(module_name)
        factory = getattr(module, "factor_mining_catalog", None)
        if not callable(factory):
            continue
        for entry in tuple(factory()):
            if not all(hasattr(entry, field) for field in ("signal", "family", "kernel")):
                continue
            factor_id = str(entry.signal).strip()
            factor = research_by_id.get(factor_id)
            if factor is None:
                continue
            primary_family = str(entry.family).strip()
            operator_id = str(entry.kernel).strip()
            candidate = _candidate_from_operator(
                factor=factor,
                primary_family=primary_family,
                operator_id=operator_id,
                operator_metadata=operator_metadata,
                hypothesis=str(getattr(entry, "hypothesis", "")).strip(),
            )
            key = (candidate.primary_family, candidate.operator_id, candidate.implementation_module)
            existing = candidates[factor_id].get(key)
            # Direct leaf catalogues are preferred for descriptive text over
            # aggregate/re-export wrappers; execution mapping is identical.
            if existing is None or module_name == candidate.implementation_module:
                candidates[factor_id][key] = candidate
    for factor in frozen:
        if factor.source_set != "legacy_live27":
            continue
        if factor.frozen_operator_id is None:
            raise CatalogBuildError(f"legacy factor lacks frozen operator: {factor.factor_id}")
        candidate = _candidate_from_operator(
            factor=factor,
            primary_family=factor.primary_family,
            operator_id=factor.frozen_operator_id,
            operator_metadata=operator_metadata,
            hypothesis="Frozen legacy live27 instance; identity is independent of live admission.",
        )
        key = (candidate.primary_family, candidate.operator_id, candidate.implementation_module)
        candidates[factor.factor_id][key] = candidate
    return {factor_id: list(rows.values()) for factor_id, rows in candidates.items()}


def _select_candidates(
    frozen: Sequence[FrozenFactor],
    *,
    discovered: Mapping[str, Sequence[Candidate]],
    overrides: Mapping[str, Mapping[str, str]],
) -> list[Candidate]:
    selected: list[Candidate] = []
    seen_overrides: set[str] = set()
    for factor in frozen:
        options = list(discovered.get(factor.factor_id, ()))
        if not options:
            raise CatalogBuildError(f"no legacy operator mapping for factor: {factor.factor_id}")
        matching_family = [option for option in options if option.primary_family == factor.primary_family]
        if not matching_family:
            listed = sorted({option.primary_family for option in options})
            raise CatalogBuildError(
                f"family mismatch for {factor.factor_id}: frozen={factor.primary_family}, candidates={listed}"
            )
        override = overrides.get(factor.factor_id)
        if len(options) > 1:
            if override is None:
                descriptions = [
                    {
                        "family": option.primary_family,
                        "operator": option.operator_id,
                        "implementation_module": option.implementation_module,
                    }
                    for option in options
                ]
                raise CatalogBuildError(
                    f"ambiguous legacy mapping requires explicit override for {factor.factor_id}: "
                    + json.dumps(descriptions, ensure_ascii=False, sort_keys=True)
                )
            seen_overrides.add(factor.factor_id)
            matching_override = [
                option
                for option in options
                if option.primary_family == override["primary_family"]
                and option.operator_id == override["operator_id"]
                and option.implementation_module == override["implementation_module"]
            ]
            if len(matching_override) != 1:
                raise CatalogBuildError(
                    f"manual override does not match an ambiguous candidate: {factor.factor_id}"
                )
            choice = matching_override[0]
        else:
            if override is not None:
                raise CatalogBuildError(
                    f"manual override is stale because mapping is no longer ambiguous: {factor.factor_id}"
                )
            choice = options[0]
        if choice.primary_family != factor.primary_family:
            raise CatalogBuildError(f"selected mapping family drift: {factor.factor_id}")
        selected.append(choice)
    extra_overrides = sorted(set(overrides).difference(seen_overrides))
    if extra_overrides:
        raise CatalogBuildError(f"manual overrides did not correspond to an ambiguity: {extra_overrides}")
    return selected


def _daily_requirements(candidate: Candidate) -> list[dict[str, Any]]:
    operator_cls = FactorRegistry.get(candidate.operator_id)
    try:
        requirements = operator_cls.daily_requirements(dict(candidate.fixed_params))
    except Exception as exc:
        raise CatalogBuildError(
            f"cannot infer daily requirements for {candidate.factor_id} from {candidate.operator_id}"
        ) from exc
    output: list[dict[str, Any]] = []
    for requirement in requirements:
        output.append(
            {
                "source": str(requirement.source),
                "columns": [str(column) for column in requirement.columns],
                "lookback_days": int(requirement.lookback_days),
            }
        )
    return output


def _definition_path(factor: FrozenFactor) -> Path:
    return Path("factors") / factor.primary_family / factor.factor_id / "definition.py"


def _contract_path(factor: FrozenFactor) -> Path:
    return Path("factors") / factor.primary_family / factor.factor_id / "contract.json"


def _factor_version(candidate: Candidate) -> str:
    params_hash = json_sha256(candidate.fixed_params)[:12]
    return f"catalog-20260826-{candidate.implementation_sha256[:12]}-{params_hash}"


def _render_definition(
    *,
    factor: FrozenFactor,
    candidate: Candidate,
    factor_version: str,
    contract_rel: Path,
) -> str:
    output_col = None
    rust_contract_id = candidate.rust_contract_id
    return "\n".join(
        [
            '"""Generated parameterized factor definition; never registers a runtime operator."""',
            "",
            "from __future__ import annotations",
            "",
            "GENERATED_FACTOR_DEFINITION = True",
            f"FACTOR_ID = {factor.factor_id!r}",
            f"FACTOR_VERSION = {factor_version!r}",
            f"PRIMARY_FAMILY = {factor.primary_family!r}",
            f"OPERATOR_ID = {candidate.operator_id!r}",
            "FIXED_PARAMS = " + pprint.pformat(dict(candidate.fixed_params), sort_dicts=True, width=100),
            f"OUTPUT_COL = {output_col!r}",
            f"RUST_CONTRACT_ID = {rust_contract_id!r}",
            f"CONTRACT_PATH = {contract_rel.as_posix()!r}",
            f"OPERATOR_CONTRACT_PATH = {_operator_contract_path(candidate.operator_id).as_posix()!r}",
            "",
            "",
            "def definition_payload() -> dict[str, object]:",
            "    \"\"\"Return this immutable instance's execution metadata as a fresh mapping.\"\"\"",
            "",
            "    return {",
            "        'factor_id': FACTOR_ID,",
            "        'factor_version': FACTOR_VERSION,",
            "        'primary_family': PRIMARY_FAMILY,",
            "        'operator_id': OPERATOR_ID,",
            "        'fixed_params': dict(FIXED_PARAMS),",
            "        'output_col': OUTPUT_COL,",
            "        'rust_contract_id': RUST_CONTRACT_ID,",
            "        'operator_contract_path': OPERATOR_CONTRACT_PATH,",
            "    }",
            "",
            "",
            "def build_factor_spec():",
            "    \"\"\"Build an exact ``FactorSpec`` without importing or registering an operator.\"\"\"",
            "",
            "    from cbond_on.domain.factors.spec import FactorSpec",
            "",
            "    return FactorSpec(",
            "        name=FACTOR_ID,",
            "        factor=OPERATOR_ID,",
            "        params=dict(FIXED_PARAMS),",
            "        output_col=OUTPUT_COL,",
            "        rust_contract_id=RUST_CONTRACT_ID,",
            "    )",
            "",
        ]
    )


def _contract_payload(
    *,
    factor: FrozenFactor,
    candidate: Candidate,
    factor_version: str,
    definition_rel: Path,
    definition_sha256: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    operator_cls = FactorRegistry.get(candidate.operator_id)
    payload: dict[str, Any] = {
        "schema_version": "factor_contract/v2",
        "identity": {
            "factor_id": factor.factor_id,
            "factor_version": factor_version,
            "primary_family": factor.primary_family,
            "tags": [],
        },
        "definition": {
            "path": definition_rel.as_posix(),
            "sha256": definition_sha256,
            "kind": "parameterized_operator_definition",
        },
        "operator_bindings": [
            {
                "operator_id": candidate.operator_id,
                "operator_version": candidate.operator_version,
                "implementation_module": candidate.implementation_module,
                "implementation_path": candidate.implementation_path,
                "implementation_sha256": candidate.implementation_sha256,
                "operator_contract_path": _operator_contract_path(candidate.operator_id).as_posix(),
                "legacy_implementation_path": candidate.legacy_implementation_path,
                "legacy_implementation_sha256": candidate.legacy_implementation_sha256,
                "language": "python_runtime_operator",
                "rust_contract_id": candidate.rust_contract_id,
            }
        ],
        "fixed_params": dict(candidate.fixed_params),
        "output": {
            "column": factor.factor_id,
            "dtype": "float64",
        },
        "context_requirements": {
            "panel_name": str(snapshot["time_contract"]["panel_name"]),
            "requires_stock_panel": bool(getattr(operator_cls, "requires_stock_panel", False)),
            "requires_bond_stock_map": bool(getattr(operator_cls, "requires_bond_stock_map", False)),
            "daily": _daily_requirements(candidate),
            "panel_fields": [],
            "panel_field_schema_status": "operator_declared_pending_factor_minimal_audit",
        },
        "time_contract": {
            "factor_time": str(snapshot["time_contract"]["factor_time"]),
            "research_label_time_reference": str(
                snapshot["time_contract"]["research_label_time_reference"]
            ),
            "pit_validation_status": "operator_migration_pending_factor_pit_audit",
        },
        "source_lineage": {
            "snapshot_id": str(snapshot["snapshot_id"]),
            "source_set": factor.source_set,
            "source_position": factor.source_position,
            "source_root": factor.source_root,
            "hypothesis": candidate.hypothesis,
        },
        "lifecycle": {
            "catalog_status": "registered",
            "research_status": (
                "legacy_mapping_discovered" if factor.source_set == "research773" else "not_evaluated"
            ),
            "live_admission_status": (
                "live_released" if candidate.rust_contract_id is not None else "not_evaluated_by_catalog_builder"
            ),
            "rust_status": (
                "live50_rust_capability_required"
                if candidate.rust_contract_id is not None
                else "not_evaluated_by_catalog_builder"
            ),
        },
        "test_reference": {
            "contract_test": "tests/test_factor_engine_governance.py::test_all_factor_definitions_and_contracts",
            "operator_test": (
                "tests/test_factor_engine_governance.py::test_every_operator_contract_matches_runtime_source"
            ),
            "behavioral_parity_status": "pending_baseline_replay",
        },
    }
    payload["contract_hash"] = json_sha256(payload)
    return payload


def _operator_contract_payload(
    *,
    operator_id: str,
    metadata: Mapping[str, Any],
    factor_ids: Sequence[str],
    definition_rel: Path,
    definition_sha256: str,
) -> dict[str, Any]:
    """Build one independently registered contract for a reusable operator."""

    payload: dict[str, Any] = {
        "schema_version": "operator_contract/v1",
        "identity": {
            "operator_id": operator_id,
            "operator_version": str(metadata["operator_version"]),
            "operator_class": str(metadata["operator_class"]),
        },
        "definition": {
            "path": definition_rel.as_posix(),
            "sha256": definition_sha256,
            "kind": "runtime_operator_reference",
        },
        "implementation": {
            "module": str(metadata["implementation_module"]),
            "path": str(metadata["implementation_path"]),
            "sha256": str(metadata["implementation_sha256"]),
            "language": "python_runtime_operator",
        },
        "migration": {
            "migration_kind": "namespace_relocation_without_formula_change",
            "legacy_path": str(metadata["legacy_implementation_path"]),
            "legacy_sha256": str(metadata["legacy_implementation_sha256"]),
            "behavior_parity_status": "pending_baseline_replay",
        },
        "runtime_contract": {
            "registration_key": operator_id,
            "requires_stock_panel": bool(metadata["requires_stock_panel"]),
            "requires_bond_stock_map": bool(metadata["requires_bond_stock_map"]),
            "input_schema_status": "operator_declared_pending_field_audit",
            "pit_validation_status": "operator_migration_pending_pit_audit",
        },
        "factor_instances": list(factor_ids),
        "test_reference": "tests/test_factor_engine_governance.py::test_every_operator_contract_matches_runtime_source",
        "lifecycle": {
            "operator_catalog_status": "registered",
            "runtime_source_status": "migrated_pending_cutover",
            "rust_status": "not_evaluated_by_catalog_builder",
        },
    }
    payload["contract_hash"] = json_sha256(payload)
    return payload


def _render_operator_definition(
    *,
    operator_id: str,
    metadata: Mapping[str, Any],
    contract_rel: Path,
) -> str:
    """Give each registered operator a local definition without copying its formula."""

    return "\n".join(
        [
            '"""Generated operator identity entry; executable source stays in domain/factors/operators."""',
            "",
            "from __future__ import annotations",
            "",
            "GENERATED_OPERATOR_DEFINITION = True",
            f"OPERATOR_ID = {operator_id!r}",
            f"OPERATOR_VERSION = {str(metadata['operator_version'])!r}",
            f"OPERATOR_CLASS = {str(metadata['operator_class'])!r}",
            f"IMPLEMENTATION_MODULE = {str(metadata['implementation_module'])!r}",
            f"IMPLEMENTATION_PATH = {str(metadata['implementation_path'])!r}",
            f"IMPLEMENTATION_SHA256 = {str(metadata['implementation_sha256'])!r}",
            f"CONTRACT_PATH = {contract_rel.as_posix()!r}",
            "",
            "",
            "def definition_payload() -> dict[str, object]:",
            "    \"\"\"Return immutable identity metadata without importing the runtime implementation.\"\"\"",
            "",
            "    return {",
            "        'operator_id': OPERATOR_ID,",
            "        'operator_version': OPERATOR_VERSION,",
            "        'operator_class': OPERATOR_CLASS,",
            "        'implementation_module': IMPLEMENTATION_MODULE,",
            "        'implementation_path': IMPLEMENTATION_PATH,",
            "        'implementation_sha256': IMPLEMENTATION_SHA256,",
            "        'contract_path': CONTRACT_PATH,",
            "    }",
            "",
        ]
    )


def _load_json5_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            raw = json5.load(handle)
    except FileNotFoundError as exc:
        raise CatalogBuildError(f"missing {label}: {path}") from exc
    except Exception as exc:
        raise CatalogBuildError(f"invalid {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise CatalogBuildError(f"{label} must be an object: {path}")
    return raw


def _load_live50_specs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and cross-check the frozen live50 metadata without changing it."""

    pack = _load_json5_object(_LIVE50_PACK, label="live50 pack")
    profile = _load_json5_object(_LIVE50_PROFILE, label="live50 profile")
    pack_specs = pack.get("factors")
    profile_ids = profile.get("factors")
    if not isinstance(pack_specs, list) or not isinstance(profile_ids, list):
        raise CatalogBuildError("live50 pack/profile must each contain factors")
    if len(pack_specs) != 50 or len(profile_ids) != 50:
        raise CatalogBuildError("live50 pack/profile must each contain exactly 50 factors")
    pack_ids: list[str] = []
    normalized_specs: list[dict[str, Any]] = []
    for spec in pack_specs:
        if not isinstance(spec, dict):
            raise CatalogBuildError("live50 pack factor spec must be object")
        factor_id = _safe_component(str(spec.get("name", "")), label="live50 factor_id")
        operator_id = _safe_component(str(spec.get("factor", "")), label="live50 operator_id")
        rust_contract_id = str(spec.get("rust_contract_id", "")).strip()
        params = spec.get("params", {})
        if not rust_contract_id or not isinstance(params, dict):
            raise CatalogBuildError(f"invalid live50 factor spec: {factor_id}")
        pack_ids.append(factor_id)
        normalized_specs.append(
            {
                "factor_id": factor_id,
                "operator_id": operator_id,
                "rust_contract_id": rust_contract_id,
                "runtime_params": json.loads(json.dumps(params, ensure_ascii=False)),
                "output_col": spec.get("output_col"),
            }
        )
    normalized_profile_ids = [_safe_component(str(item), label="live50 profile factor_id") for item in profile_ids]
    if pack_ids != normalized_profile_ids or len(set(pack_ids)) != len(pack_ids):
        raise CatalogBuildError("live50 pack/profile factor order or uniqueness drift")
    return normalized_specs, profile


def _enrich_live50_candidates(selected: Sequence[Candidate]) -> list[Candidate]:
    """Attach frozen live50 facts to the matching canonical factor identities."""

    specs, _ = _load_live50_specs()
    by_factor_id = {str(spec["factor_id"]): spec for spec in specs}
    enriched: list[Candidate] = []
    for candidate in selected:
        spec = by_factor_id.get(candidate.factor_id)
        if spec is None:
            enriched.append(candidate)
            continue
        if candidate.operator_id != spec["operator_id"]:
            raise CatalogBuildError(f"live50 operator mismatch for canonical factor: {candidate.factor_id}")
        if dict(candidate.fixed_params) != dict(spec["runtime_params"]):
            raise CatalogBuildError(
                f"live50 params mismatch for canonical factor: {candidate.factor_id}"
            )
        enriched.append(replace(candidate, rust_contract_id=str(spec["rust_contract_id"])))
    if sum(candidate.rust_contract_id is not None for candidate in enriched) != 50:
        raise CatalogBuildError("live50 enrichment expected exactly 50 canonical instances")
    return enriched


def _live50_release_payload(factor_catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the migrated runtime source binding for the existing live50 formula."""

    normalized_specs, profile = _load_live50_specs()
    catalog_rows = factor_catalog.get("factors")
    if not isinstance(catalog_rows, list):
        raise CatalogBuildError("factor catalog must contain factor rows for live release")
    catalog_by_id = {str(row.get("factor_id")): row for row in catalog_rows if isinstance(row, dict)}
    instances: list[dict[str, Any]] = []
    for position, spec in enumerate(normalized_specs, start=1):
        factor_id = str(spec["factor_id"])
        factor = catalog_by_id.get(factor_id)
        if factor is None:
            raise CatalogBuildError(f"live50 factor absent from canonical catalog: {factor_id}")
        operator_id = str(spec["operator_id"])
        if factor.get("operator_id") != operator_id:
            raise CatalogBuildError(f"live50 operator mismatch for canonical factor: {factor_id}")
        rust_contract_id = str(spec["rust_contract_id"])
        if factor.get("rust_contract_id") != rust_contract_id:
            raise CatalogBuildError(f"canonical rust contract mismatch for live50 factor: {factor_id}")
        instances.append(
            {
                "position": position,
                "factor_id": factor_id,
                "factor_version": factor["factor_version"],
                "contract_hash": factor["contract_hash"],
                "operator_id": operator_id,
                "rust_contract_id": rust_contract_id,
                "runtime_params": dict(spec["runtime_params"]),
                "output_col": spec.get("output_col"),
            }
        )
    return {
        "schema_version": "factor_catalog_live_release/v1",
        "release_id": LIVE50_RELEASE_ID,
        "release_kind": "operator_source_migration_binding",
        "migration_from": {
            "release_id": "live50_rust50_20260806",
            "formula_change": False,
            "feature_order_change": False,
            "rust_contract_change": False,
            "activation_requires_no_db_fullchain_parity": True,
        },
        "source_artifacts": {
            "live50_pack": {
                "path": _relative_repo_path(_LIVE50_PACK),
                "sha256": _sha256_file(_LIVE50_PACK),
            },
            "live50_profile": {
                "path": _relative_repo_path(_LIVE50_PROFILE),
                "sha256": _sha256_file(_LIVE50_PROFILE),
                "specs_sha256": str(profile.get("specs_sha256", "")).strip(),
            },
            "operator_source_migration": {
                "path": _relative_repo_path(_OPERATOR_MIGRATION_MANIFEST),
                "sha256": _sha256_file(_OPERATOR_MIGRATION_MANIFEST),
            },
        },
        "factor_count": len(instances),
        "instances": instances,
        "admission_status": "metadata_only_not_runtime_authorization",
    }


def _build_outputs(
    *,
    snapshot: Mapping[str, Any],
    selected: Sequence[Candidate],
    operator_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[Path, str], dict[str, Any], dict[str, Any]]:
    frozen = _frozen_factors(snapshot)
    selected_by_id = {candidate.factor_id: candidate for candidate in selected}
    if set(selected_by_id) != {factor.factor_id for factor in frozen}:
        raise CatalogBuildError("selected mappings do not cover frozen factor snapshot")
    files: dict[Path, str] = {}
    catalog_rows: list[dict[str, Any]] = []
    operator_members: dict[str, list[str]] = {}
    for factor in frozen:
        candidate = selected_by_id[factor.factor_id]
        definition_rel = _definition_path(factor)
        contract_rel = _contract_path(factor)
        factor_version = _factor_version(candidate)
        definition_text = _render_definition(
            factor=factor,
            candidate=candidate,
            factor_version=factor_version,
            contract_rel=contract_rel,
        )
        definition_sha256 = hashlib.sha256(definition_text.encode("utf-8")).hexdigest()
        contract = _contract_payload(
            factor=factor,
            candidate=candidate,
            factor_version=factor_version,
            definition_rel=definition_rel,
            definition_sha256=definition_sha256,
            snapshot=snapshot,
        )
        files[definition_rel] = definition_text
        files[contract_rel] = _canonical_text(contract)
        catalog_rows.append(
            {
                "factor_id": factor.factor_id,
                "factor_version": factor_version,
                "primary_family": factor.primary_family,
                "source_set": factor.source_set,
                "operator_id": candidate.operator_id,
                "definition_path": definition_rel.as_posix(),
                "contract_path": contract_rel.as_posix(),
                "contract_hash": contract["contract_hash"],
                "source_position": factor.source_position,
                "rust_contract_id": candidate.rust_contract_id,
                "lifecycle": {
                    "catalog_status": "registered",
                    "live_admission_status": (
                        "live_released"
                        if candidate.rust_contract_id is not None
                        else "not_evaluated_by_catalog_builder"
                    ),
                    "rust_status": (
                        "live50_rust_capability_required"
                        if candidate.rust_contract_id is not None
                        else "not_evaluated_by_catalog_builder"
                    ),
                },
            }
        )
        operator_members.setdefault(candidate.operator_id, []).append(factor.factor_id)

    operator_rows: list[dict[str, Any]] = []
    for operator_id in sorted(operator_metadata):
        metadata = dict(operator_metadata[operator_id])
        factor_ids = sorted(operator_members.get(operator_id, []))
        operator_contract_rel = _operator_contract_path(operator_id)
        operator_definition_rel = _operator_definition_path(operator_id)
        operator_definition = _render_operator_definition(
            operator_id=operator_id,
            metadata=metadata,
            contract_rel=operator_contract_rel,
        )
        operator_definition_sha256 = hashlib.sha256(
            operator_definition.encode("utf-8")
        ).hexdigest()
        operator_contract = _operator_contract_payload(
            operator_id=operator_id,
            metadata=metadata,
            factor_ids=factor_ids,
            definition_rel=operator_definition_rel,
            definition_sha256=operator_definition_sha256,
        )
        files[operator_definition_rel] = operator_definition
        files[operator_contract_rel] = _canonical_text(operator_contract)
        operator_rows.append(
            {
                "operator_id": operator_id,
                "operator_version": metadata["operator_version"],
                "operator_class": metadata["operator_class"],
                "implementation_module": metadata["implementation_module"],
                "implementation_path": metadata["implementation_path"],
                "implementation_sha256": metadata["implementation_sha256"],
                "legacy_implementation_path": metadata["legacy_implementation_path"],
                "legacy_implementation_sha256": metadata["legacy_implementation_sha256"],
                "language": metadata["language"],
                "availability_status": metadata["availability_status"],
                "default_import_status": metadata["default_import_status"],
                "operator_contract_path": operator_contract_rel.as_posix(),
                "operator_definition_path": operator_definition_rel.as_posix(),
                "operator_contract_hash": operator_contract["contract_hash"],
                "requires_stock_panel": metadata["requires_stock_panel"],
                "requires_bond_stock_map": metadata["requires_bond_stock_map"],
                "factor_count": len(factor_ids),
                "factor_ids": factor_ids,
                "lifecycle": {
                    "operator_catalog_status": "registered",
                    "rust_status": "not_evaluated_by_catalog_builder",
                },
            }
        )

    if len(operator_rows) != EXPECTED_OPERATOR_COUNT:
        raise CatalogBuildError(f"operator catalog expected {EXPECTED_OPERATOR_COUNT} rows")
    snapshot_sha256 = json_sha256(snapshot)
    source_set_counts = {
        source_set: sum(row["source_set"] == source_set for row in catalog_rows)
        for source_set in ("research773", "legacy_live27")
    }
    factor_catalog = {
        "schema_version": "factor_catalog/v1",
        "snapshot_id": str(snapshot["snapshot_id"]),
        "snapshot_sha256": snapshot_sha256,
        "factor_count": len(catalog_rows),
        "operator_count": len(operator_rows),
        "referenced_operator_count": sum(bool(row["factor_ids"]) for row in operator_rows),
        "source_set_counts": source_set_counts,
        "manual_overrides_path": "catalog/manual_overrides.json",
        "factors": catalog_rows,
    }
    operator_catalog = {
        "schema_version": "operator_catalog/v1",
        "snapshot_id": str(snapshot["snapshot_id"]),
        "factor_count": len(catalog_rows),
        "operator_count": len(operator_rows),
        "operators": operator_rows,
    }
    files[Path("catalog") / "factor_catalog.json"] = _canonical_text(factor_catalog)
    files[Path("catalog") / "operator_catalog.json"] = _canonical_text(operator_catalog)
    files[Path("releases") / "live" / f"{LIVE50_RELEASE_ID}.json"] = _canonical_text(
        _live50_release_payload(factor_catalog)
    )
    return files, factor_catalog, operator_catalog


def _assert_output_root(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    expected = _factor_engine_root().resolve(strict=False)
    if resolved != expected:
        raise CatalogBuildError(f"output root must be the isolated factor_engine directory: {expected}")
    return resolved


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True


def _write_outputs(
    *,
    output_root: Path,
    snapshot: Mapping[str, Any],
    write_snapshot: bool,
    files: Mapping[Path, str],
) -> int:
    target = _assert_output_root(output_root)
    changed = 0
    if write_snapshot:
        changed += int(_write_if_changed(target / "catalog" / _source_snapshot_path().name, _canonical_text(snapshot)))
    for relative, content in files.items():
        if relative.is_absolute() or ".." in relative.parts:
            raise CatalogBuildError(f"unsafe generated output path: {relative}")
        changed += int(_write_if_changed(target / relative, content))
    # P0 generated ``implementation_ref.json`` files were replaced by the
    # stronger per-operator ``definition.py`` entry.  Prune only this exact,
    # generator-owned retired filename; do not touch factor contracts, release
    # history, user research assets, or any runtime source tree.
    for stale in target.glob("operators/*/implementation_ref.json"):
        relative = stale.relative_to(target)
        if relative not in files:
            stale.unlink()
            changed += 1
    return changed


def _check_rendered_outputs(
    *,
    output_root: Path,
    files: Mapping[Path, str],
) -> None:
    root = _assert_output_root(output_root)
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
        elif path.read_text(encoding="utf-8") != expected:
            mismatched.append(relative.as_posix())
    if missing or mismatched:
        raise CatalogBuildError(
            "generated factor-engine output drift: "
            f"missing={missing[:10]}, mismatched={mismatched[:10]}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-catalog",
        type=Path,
        help="Frozen external factor_family_catalog.csv used only to bootstrap the committed snapshot.",
    )
    parser.add_argument(
        "--source-audit",
        type=Path,
        help="Frozen external factor_source_audit.csv paired with --source-catalog.",
    )
    parser.add_argument(
        "--screen-manifest",
        type=Path,
        help="Optional frozen external screen_manifest.json for provenance hashing.",
    )
    parser.add_argument(
        "--legacy-live-pack",
        type=Path,
        default=_LEGACY_LIVE27_PACK,
        help="Source-controlled legacy live27 pack incorporated as its own source_set.",
    )
    parser.add_argument(
        "--legacy-factor-registry",
        type=Path,
        default=_LEGACY_FACTOR_REGISTRY,
        help="Source-controlled registry used only to obtain legacy live27 families.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=_source_snapshot_path(),
        help="Committed frozen snapshot used for ordinary generation/checks.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=_overrides_path(),
        help="Committed explicit ambiguity overrides.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_factor_engine_root(),
        help="Must remain the repository's factor_engine directory.",
    )
    parser.add_argument("--write", action="store_true", help="Write deterministic source-controlled assets.")
    parser.add_argument("--check", action="store_true", help="Fail if committed generated assets drift.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source_catalog is None and args.source_audit is not None:
        raise CatalogBuildError("--source-audit requires --source-catalog")
    if args.source_catalog is not None and args.source_audit is None:
        raise CatalogBuildError("--source-catalog requires --source-audit")
    if args.check and args.write:
        raise CatalogBuildError("--check and --write are mutually exclusive")

    write_snapshot = args.source_catalog is not None
    snapshot = (
        _load_csv_snapshot(
            source_catalog=args.source_catalog,
            source_audit=args.source_audit,
            screen_manifest=args.screen_manifest,
            legacy_live_pack=args.legacy_live_pack,
            legacy_factor_registry=args.legacy_factor_registry,
        )
        if write_snapshot
        else _load_snapshot(args.snapshot)
    )
    frozen = _frozen_factors(snapshot)
    overrides = _read_overrides(args.overrides)
    operator_metadata = _discover_all_static_operators()
    discovered = _discover_candidates(frozen, operator_metadata=operator_metadata)
    selected = _enrich_live50_candidates(
        _select_candidates(frozen, discovered=discovered, overrides=overrides)
    )
    files, factor_catalog, operator_catalog = _build_outputs(
        snapshot=snapshot,
        selected=selected,
        operator_metadata=operator_metadata,
    )

    if args.check:
        _check_rendered_outputs(output_root=args.output_root, files=files)
        summary = validate_factor_catalog(_REPO_ROOT)
        print(json.dumps({"mode": "check", **summary}, ensure_ascii=False, sort_keys=True))
        return 0

    if args.write:
        changed = _write_outputs(
            output_root=args.output_root,
            snapshot=snapshot,
            write_snapshot=write_snapshot,
            files=files,
        )
        print(
            json.dumps(
                {
                    "mode": "write",
                    "changed_files": changed,
                    "factor_count": factor_catalog["factor_count"],
                    "operator_count": operator_catalog["operator_count"],
                    "output_root": str(_assert_output_root(args.output_root)),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "mode": "plan",
                "factor_count": factor_catalog["factor_count"],
                "operator_count": operator_catalog["operator_count"],
                "generated_file_count": len(files) + int(write_snapshot),
                "output_root": str(_assert_output_root(args.output_root)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CatalogBuildError, FactorCatalogValidationError) as exc:
        print(f"factor catalog build failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
