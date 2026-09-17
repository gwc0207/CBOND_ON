"""Narrow permit for the research-only catalog Python execution path.

This module does not alter the normal Rust-first policy.  It admits Python
execution only when a research process presents an opaque permit bound to an
immutable full-catalog snapshot, exact factor contracts, and a scratch-only
FactorStore root.  It is intentionally unusable by live configuration and has
no dependency on live admission code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from cbond_on.core.registry import FactorRegistry, RegistryError
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col


RESEARCH_CATALOG_PYTHON_ENGINE = "research_python"
RESEARCH_CATALOG_PYTHON_POLICY = "research_catalog_python"
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_PERMIT_SEAL = object()
_REQUIRED_FACTOR_FIELDS = (
    "factor_id",
    "factor_version",
    "source_set",
    "operator_id",
    "definition_path",
    "contract_path",
    "contract_hash",
)


class ResearchCatalogPermitError(PermissionError):
    """Raised when a request does not meet the isolated research boundary."""


@dataclass(frozen=True, init=False)
class ResearchCatalogExecutionPermit:
    """Opaque proof of one catalog-pinned research Python execution request.

    ``init=False`` and the private identity seal deliberately prevent normal
    callers from constructing a usable permit.  The sole factory below derives
    all fields from a checked-in catalog plus its individual contracts.
    """

    catalog_path: str
    catalog_sha256: str
    output_root: str
    factor_ids: tuple[str, ...]
    _bindings: Mapping[str, Mapping[str, Any]]
    _operator_modules: Mapping[str, Mapping[str, str]]
    _seal: object


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_strict_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchCatalogPermitError(f"missing {field}: {path}") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchCatalogPermitError(f"invalid {field}: {path}") from exc
    if not isinstance(raw, Mapping):
        raise ResearchCatalogPermitError(f"{field} must be an object: {path}")
    return dict(raw)


def _require_sha256(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ResearchCatalogPermitError(f"{field} must be a SHA-256 hex string")
    return text


def _safe_relative_path(value: object, *, field: str) -> Path:
    text = str(value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ResearchCatalogPermitError(f"unsafe {field}: {value!r}")
    return path


def _validate_research_cfg(research_cfg: Mapping[str, Any]) -> None:
    if research_cfg.get("research_only") is not True:
        raise ResearchCatalogPermitError("research catalog Python execution requires research_only=true")
    forbidden_top_level = {
        "live_factor_admission",
        "live",
        "live_config",
        "model_score",
        "strategy",
        "database",
        "db",
    }
    present = sorted(key for key in forbidden_top_level if key in research_cfg)
    if present:
        raise ResearchCatalogPermitError(
            "research catalog execution config must not declare live/model/DB fields: " + ", ".join(present)
        )
    output = research_cfg.get("output", {})
    if output is not None and not isinstance(output, Mapping):
        raise ResearchCatalogPermitError("research catalog execution output must be an object")
    output_map = dict(output or {})
    forbidden_output = {"db_write", "db_table", "db_backend", "score_output", "trade_list", "live_root"}
    present_output = sorted(key for key in forbidden_output if key in output_map)
    if present_output:
        raise ResearchCatalogPermitError(
            "research catalog execution output must not declare live/DB outputs: " + ", ".join(present_output)
        )
    compute = research_cfg.get("compute")
    if not isinstance(compute, Mapping):
        raise ResearchCatalogPermitError("research catalog execution requires compute object")
    if str(compute.get("engine", "")).strip().lower() != RESEARCH_CATALOG_PYTHON_ENGINE:
        raise ResearchCatalogPermitError(
            f"research catalog execution requires compute.engine={RESEARCH_CATALOG_PYTHON_ENGINE!r}"
        )
    if str(compute.get("execution_policy", "")).strip().lower() != RESEARCH_CATALOG_PYTHON_POLICY:
        raise ResearchCatalogPermitError(
            "research catalog execution requires "
            f"compute.execution_policy={RESEARCH_CATALOG_PYTHON_POLICY!r}"
        )


def _catalog_entries(catalog_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if str(catalog_payload.get("schema_version", "")).strip() != "factor_catalog/v1":
        raise ResearchCatalogPermitError("research Python execution requires canonical factor_catalog/v1")
    raw_factors = catalog_payload.get("factors")
    if not isinstance(raw_factors, list) or not raw_factors:
        raise ResearchCatalogPermitError("canonical factor catalog must contain non-empty factors list")
    entries: dict[str, dict[str, Any]] = {}
    for raw in raw_factors:
        if not isinstance(raw, Mapping):
            raise ResearchCatalogPermitError("canonical factor catalog has invalid factor row")
        row = dict(raw)
        factor_id = str(row.get("factor_id", "")).strip()
        if not factor_id or factor_id in entries:
            raise ResearchCatalogPermitError(f"canonical factor catalog has duplicate/empty factor_id: {factor_id!r}")
        missing = [field for field in _REQUIRED_FACTOR_FIELDS if not str(row.get(field, "")).strip()]
        if missing:
            raise ResearchCatalogPermitError(
                f"canonical factor catalog row {factor_id} missing required field(s): {', '.join(missing)}"
            )
        entries[factor_id] = row
    declared = int(catalog_payload.get("factor_count", -1))
    if declared != len(entries):
        raise ResearchCatalogPermitError("canonical factor catalog factor_count does not match factor rows")
    return entries


def _contract_binding(*, engine_root: Path, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    factor_id = str(row["factor_id"])
    contract_rel = _safe_relative_path(row["contract_path"], field=f"contract_path for {factor_id}")
    definition_rel = _safe_relative_path(row["definition_path"], field=f"definition_path for {factor_id}")
    contract_path = (engine_root / contract_rel).resolve(strict=False)
    definition_path = (engine_root / definition_rel).resolve(strict=False)
    if not _is_strict_child(contract_path, engine_root) or not _is_strict_child(definition_path, engine_root):
        raise ResearchCatalogPermitError(f"catalog paths escape factor_engine for {factor_id}")
    if not definition_path.is_file():
        raise ResearchCatalogPermitError(f"missing factor definition for {factor_id}: {definition_path}")
    contract = _read_json(contract_path, field=f"factor contract for {factor_id}")
    semantic_contract = dict(contract)
    declared_contract_hash = _require_sha256(contract.get("contract_hash"), field=f"contract hash for {factor_id}")
    semantic_contract.pop("contract_hash", None)
    if _canonical_sha256(semantic_contract) != declared_contract_hash:
        raise ResearchCatalogPermitError(f"factor contract semantic hash mismatch: {factor_id}")
    if declared_contract_hash != _require_sha256(row.get("contract_hash"), field=f"catalog contract hash for {factor_id}"):
        raise ResearchCatalogPermitError(f"catalog/contract hash mismatch: {factor_id}")
    identity = contract.get("identity")
    if not isinstance(identity, Mapping):
        raise ResearchCatalogPermitError(f"factor contract identity missing: {factor_id}")
    if str(identity.get("factor_id", "")).strip() != factor_id:
        raise ResearchCatalogPermitError(f"factor contract identity mismatch: {factor_id}")
    if str(identity.get("factor_version", "")).strip() != str(row["factor_version"]).strip():
        raise ResearchCatalogPermitError(f"factor contract version mismatch: {factor_id}")
    output = contract.get("output")
    if not isinstance(output, Mapping) or str(output.get("column", "")).strip() != factor_id:
        raise ResearchCatalogPermitError(f"factor contract output mismatch: {factor_id}")
    bindings = contract.get("operator_bindings")
    if not isinstance(bindings, list) or len(bindings) != 1 or not isinstance(bindings[0], Mapping):
        raise ResearchCatalogPermitError(f"factor contract must bind exactly one operator: {factor_id}")
    operator_binding = dict(bindings[0])
    operator_id = str(operator_binding.get("operator_id", "")).strip()
    if operator_id != str(row["operator_id"]).strip():
        raise ResearchCatalogPermitError(f"factor catalog/operator contract mismatch: {factor_id}")
    implementation_module = str(operator_binding.get("implementation_module", "")).strip()
    implementation_path = str(operator_binding.get("implementation_path", "")).strip()
    implementation_sha256 = _require_sha256(
        operator_binding.get("implementation_sha256"),
        field=f"operator implementation hash for {factor_id}",
    )
    if not implementation_module or not implementation_path:
        raise ResearchCatalogPermitError(f"factor contract operator implementation missing: {factor_id}")
    fixed_params = contract.get("fixed_params", {})
    if not isinstance(fixed_params, Mapping):
        raise ResearchCatalogPermitError(f"factor contract fixed_params must be an object: {factor_id}")
    binding = {
        "factor_id": factor_id,
        "factor_version": str(row["factor_version"]).strip(),
        "operator_id": operator_id,
        "contract_hash": declared_contract_hash,
        "params_sha256": _canonical_sha256(dict(fixed_params)),
        "fixed_params": MappingProxyType(dict(fixed_params)),
    }
    operator = {
        "operator_id": operator_id,
        "implementation_module": implementation_module,
        "implementation_path": implementation_path,
        "implementation_sha256": implementation_sha256,
    }
    return binding, operator


def _resolve_through_catalog_api(
    *,
    catalog_file: Path,
    factor_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    """Resolve canonical metadata through the public catalog API only.

    The raw JSON checks below defend the contract payload itself.  This helper
    ensures a permit cannot point at a look-alike JSON file outside the
    canonical ``factor_engine/catalog`` topology or bypass the catalog agent's
    read-only resolver.
    """

    try:
        from cbond_on.domain.factor_catalog import (
            catalog_path as public_catalog_path,
            load_factor_catalog,
            resolve_factor_instance,
            resolve_operator_modules,
        )
    except ImportError as exc:  # pragma: no cover - source checkout invariant.
        raise ResearchCatalogPermitError("canonical factor catalog API is unavailable") from exc
    repo_root = catalog_file.parents[2]
    expected_path = public_catalog_path(repo_root).resolve(strict=False)
    if catalog_file != expected_path:
        raise ResearchCatalogPermitError(
            "research Python permit must use the canonical catalog API path: "
            f"expected={expected_path.as_posix()}, actual={catalog_file.as_posix()}"
        )
    catalog = load_factor_catalog(repo_root)
    rows: dict[str, Mapping[str, Any]] = {}
    for factor_id in factor_ids:
        row = catalog.get(factor_id)
        if row is None:
            raise ResearchCatalogPermitError(f"requested factor is absent from catalog API: {factor_id}")
        resolved = resolve_factor_instance(
            factor_id,
            factor_version=str(row.get("factor_version", "")),
            contract_hash=str(row.get("contract_hash", "")),
            repo_root=repo_root,
        )
        rows[factor_id] = resolved
    operators = {
        str(item.get("operator_id", "")).strip(): item
        for item in resolve_operator_modules(factor_ids, repo_root=repo_root)
    }
    if not operators or any(not key for key in operators):
        raise ResearchCatalogPermitError("catalog API returned invalid operator metadata")
    return rows, operators


def issue_research_catalog_execution_permit(
    *,
    research_cfg: Mapping[str, Any],
    catalog_path: str | Path,
    factor_data_root: str | Path,
    factor_ids: Sequence[str],
) -> ResearchCatalogExecutionPermit:
    """Issue the sole permit for the research-only Python catalog path.

    The factory validates catalog bytes, each requested factor contract, exact
    fixed parameters, operator implementation provenance, and scratch-only
    output before returning an opaque permit.  It does not import an operator
    or execute a factor.
    """

    _validate_research_cfg(research_cfg)
    requested = tuple(str(item).strip() for item in factor_ids)
    if not requested or any(not item for item in requested) or len(set(requested)) != len(requested):
        raise ResearchCatalogPermitError("factor_ids must be a non-empty unique sequence")
    catalog_file = _resolved(catalog_path)
    payload = _read_json(catalog_file, field="canonical factor catalog")
    entries = _catalog_entries(payload)
    unknown = sorted(set(requested).difference(entries))
    if unknown:
        raise ResearchCatalogPermitError(
            "requested factor_ids are absent from canonical catalog: " + ", ".join(unknown[:20])
        )
    api_rows, api_operators = _resolve_through_catalog_api(
        catalog_file=catalog_file,
        factor_ids=requested,
    )
    for factor_id in requested:
        raw_row = entries[factor_id]
        api_row = api_rows[factor_id]
        for field in _REQUIRED_FACTOR_FIELDS:
            if str(raw_row.get(field, "")) != str(api_row.get(field, "")):
                raise ResearchCatalogPermitError(
                    f"canonical catalog/API drift for {factor_id}.{field}"
                )
    root = _resolved(factor_data_root)
    scratch_parent = _resolved(RESEARCH_SCRATCH_PARENT)
    if not _is_strict_child(root, scratch_parent):
        raise ResearchCatalogPermitError(
            "research Python FactorStore must resolve strictly below "
            f"{scratch_parent.as_posix()}, got {root.as_posix()}"
        )
    engine_root = catalog_file.parent.parent.resolve(strict=False)
    bindings: dict[str, Mapping[str, str]] = {}
    operators: dict[str, Mapping[str, str]] = {}
    for factor_id in requested:
        binding, operator = _contract_binding(engine_root=engine_root, row=entries[factor_id])
        api_operator = api_operators.get(operator["operator_id"])
        if api_operator is None:
            raise ResearchCatalogPermitError(
                f"catalog API has no operator metadata for {operator['operator_id']}"
            )
        for field in ("implementation_module", "implementation_path", "implementation_sha256"):
            if str(api_operator.get(field, "")).strip() != operator[field]:
                raise ResearchCatalogPermitError(
                    f"operator catalog/contract drift for {factor_id}.{field}"
                )
        bindings[factor_id] = MappingProxyType(binding)
        previous = operators.get(operator["operator_id"])
        if previous is not None and dict(previous) != operator:
            raise ResearchCatalogPermitError(
                f"operator provenance conflicts across requested factors: {operator['operator_id']}"
            )
        operators[operator["operator_id"]] = MappingProxyType(operator)
    permit = object.__new__(ResearchCatalogExecutionPermit)
    object.__setattr__(permit, "catalog_path", str(catalog_file))
    object.__setattr__(permit, "catalog_sha256", _sha256_file(catalog_file))
    object.__setattr__(permit, "output_root", str(root))
    object.__setattr__(permit, "factor_ids", requested)
    object.__setattr__(permit, "_bindings", MappingProxyType(bindings))
    object.__setattr__(permit, "_operator_modules", MappingProxyType(operators))
    object.__setattr__(permit, "_seal", _PERMIT_SEAL)
    return permit


def validate_research_catalog_execution_permit(
    permit: ResearchCatalogExecutionPermit | None,
    *,
    factor_data_root: str | Path,
    specs: Sequence[FactorSpec],
) -> ResearchCatalogExecutionPermit:
    """Validate a permit before any panel read, context load, or store I/O."""

    if not isinstance(permit, ResearchCatalogExecutionPermit) or permit._seal is not _PERMIT_SEAL:
        raise ResearchCatalogPermitError(
            "research_python factor execution requires a ResearchCatalogExecutionPermit issued by the catalog gate"
        )
    if _sha256_file(_resolved(permit.catalog_path)) != permit.catalog_sha256:
        raise ResearchCatalogPermitError(
            "canonical factor catalog changed after research Python permit issuance"
        )
    if _resolved(factor_data_root) != _resolved(permit.output_root):
        raise ResearchCatalogPermitError(
            "research Python FactorStore root differs from the permit-bound scratch root"
        )
    actual_ids = tuple(str(spec.name).strip() for spec in specs)
    if actual_ids != permit.factor_ids:
        raise ResearchCatalogPermitError(
            "research Python specs must exactly match the permit-bound ordered catalog factor IDs"
        )
    if len(actual_ids) != len(set(actual_ids)) or any(not item for item in actual_ids):
        raise ResearchCatalogPermitError("research Python specs must have unique non-empty names")
    for spec in specs:
        factor_id = str(spec.name).strip()
        binding = permit._bindings.get(factor_id)
        if binding is None:
            raise ResearchCatalogPermitError(f"research Python factor is not permit-bound: {factor_id}")
        if build_factor_col(spec) != factor_id:
            raise ResearchCatalogPermitError(
                f"research Python factor output must equal catalog factor_id: {factor_id}"
            )
        if str(spec.factor).strip() != binding["operator_id"]:
            raise ResearchCatalogPermitError(
                f"research Python factor/operator mismatch for {factor_id}: "
                f"expected={binding['operator_id']!r}, actual={spec.factor!r}"
            )
        if _canonical_sha256(dict(spec.params or {})) != binding["params_sha256"]:
            raise ResearchCatalogPermitError(f"research Python factor parameters differ from contract: {factor_id}")
    return permit


def build_permitted_factor_specs(permit: ResearchCatalogExecutionPermit) -> tuple[FactorSpec, ...]:
    """Materialize FactorSpecs only from permit-bound catalog contracts.

    Callers must not construct a parallel config/pack for this path.  The
    factor name, operator key, output column, and parameters all derive from
    the contract snapshot held in the opaque permit.
    """

    if not isinstance(permit, ResearchCatalogExecutionPermit) or permit._seal is not _PERMIT_SEAL:
        raise ResearchCatalogPermitError("cannot materialize factor specs without a valid research catalog permit")
    specs: list[FactorSpec] = []
    for factor_id in permit.factor_ids:
        binding = permit._bindings.get(factor_id)
        if binding is None:
            raise ResearchCatalogPermitError(f"permit binding missing factor: {factor_id}")
        fixed_params = binding.get("fixed_params")
        if not isinstance(fixed_params, Mapping):
            raise ResearchCatalogPermitError(f"permit binding fixed_params missing: {factor_id}")
        specs.append(
            FactorSpec(
                name=factor_id,
                factor=str(binding["operator_id"]),
                params=dict(fixed_params),
                output_col=None,
                rust_contract_id=None,
            )
        )
    return tuple(specs)


def load_permitted_operator_modules(permit: ResearchCatalogExecutionPermit) -> tuple[str, ...]:
    """Import and provenance-check only the modules bound into one permit.

    This is deliberately separate from permit issuance: resolving a catalog is
    metadata-only, while an actual research compute process has to make an
    explicit, audited transition to importing legacy operator code.
    """

    if not isinstance(permit, ResearchCatalogExecutionPermit) or permit._seal is not _PERMIT_SEAL:
        raise ResearchCatalogPermitError("cannot import operators without a valid research catalog permit")
    loaded: list[str] = []
    repo_root = _resolved(permit.catalog_path).parents[2]
    for operator_id in sorted(permit._operator_modules):
        metadata = permit._operator_modules[operator_id]
        module_name = metadata["implementation_module"]
        module = importlib.import_module(module_name)
        source_path = inspect.getsourcefile(module)
        if not source_path:
            raise ResearchCatalogPermitError(
                f"cannot resolve imported research operator source: {operator_id}"
            )
        actual_path = _resolved(source_path)
        expected_rel = _safe_relative_path(metadata["implementation_path"], field=f"operator path {operator_id}")
        expected_path = (repo_root / expected_rel).resolve(strict=False)
        if actual_path != expected_path:
            raise ResearchCatalogPermitError(
                f"research operator source path drift for {operator_id}: "
                f"expected={expected_path.as_posix()}, actual={actual_path.as_posix()}"
            )
        if _sha256_file(actual_path) != metadata["implementation_sha256"]:
            raise ResearchCatalogPermitError(f"research operator source hash drift for {operator_id}")
        try:
            registered = FactorRegistry.get(operator_id)
        except RegistryError as exc:
            raise ResearchCatalogPermitError(
                f"permitted research operator module did not register {operator_id}"
            ) from exc
        if registered.__module__ != module_name:
            raise ResearchCatalogPermitError(
                f"permitted research operator registry provenance drift for {operator_id}: "
                f"expected={module_name}, actual={registered.__module__}"
            )
        loaded.append(module_name)
    return tuple(loaded)


__all__ = [
    "RESEARCH_CATALOG_PYTHON_ENGINE",
    "RESEARCH_CATALOG_PYTHON_POLICY",
    "RESEARCH_SCRATCH_PARENT",
    "ResearchCatalogExecutionPermit",
    "ResearchCatalogPermitError",
    "build_permitted_factor_specs",
    "issue_research_catalog_execution_permit",
    "load_permitted_operator_modules",
    "validate_research_catalog_execution_permit",
]
