"""Validation helpers for the source-controlled factor-engine catalogue.

The catalogue is deliberately data-first.  Importing this module never
imports legacy factor definitions, changes ``FactorRegistry``, or resolves a
live release.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class FactorCatalogValidationError(ValueError):
    """Raised when a generated catalogue loses identity or provenance integrity."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the deterministic representation used by generated contracts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FactorCatalogValidationError(f"missing catalogue artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FactorCatalogValidationError(f"invalid JSON catalogue artifact: {path}") from exc


def _repository_root(candidate: Path | None = None) -> Path:
    start = (candidate or Path(__file__)).resolve()
    for parent in (start, *start.parents):
        if (parent / "cbond_on").is_dir() and (parent / "harness").is_dir():
            return parent
    raise FactorCatalogValidationError("cannot resolve CBOND_ON repository root")


def _relative_path(value: object, *, field: str) -> Path:
    text = str(value or "").strip().replace("\\", "/")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise FactorCatalogValidationError(f"unsafe {field}: {value!r}")
    return path


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FactorCatalogValidationError(f"{label} must be an object")
    return value


def _require_list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FactorCatalogValidationError(f"{label} must be a list")
    return value


def _expected_contract_hash(contract: Mapping[str, Any]) -> str:
    semantic = dict(contract)
    semantic.pop("contract_hash", None)
    return json_sha256(semantic)


def _release_path(engine: Path, release_id: str) -> Path:
    text = str(release_id).strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        raise FactorCatalogValidationError(f"unsafe release_id: {release_id!r}")
    return engine / "releases" / "live" / f"{text}.json"


def validate_factor_catalog(repo_root: Path | None = None) -> dict[str, int]:
    """Validate the P0/P1 identity catalogue without touching runtime assets.

    Returns a compact count summary.  The function validates all definition
    wrappers and contracts, so a successful result proves the catalogue is a
    complete, internally consistent source-controlled asset graph.
    """

    root = _repository_root(repo_root)
    engine = root / "factor_engine"
    catalog_dir = engine / "catalog"
    factor_catalog = _require_mapping(
        load_json(catalog_dir / "factor_catalog.json"),
        label="factor_catalog",
    )
    operator_catalog = _require_mapping(
        load_json(catalog_dir / "operator_catalog.json"),
        label="operator_catalog",
    )
    snapshot = _require_mapping(
        load_json(catalog_dir / "frozen_canonical_20260826_800.json"),
        label="frozen snapshot",
    )
    overrides = _require_mapping(
        load_json(catalog_dir / "manual_overrides.json"),
        label="manual overrides",
    )

    if factor_catalog.get("schema_version") != "factor_catalog/v1":
        raise FactorCatalogValidationError("unexpected factor catalogue schema")
    if operator_catalog.get("schema_version") != "operator_catalog/v1":
        raise FactorCatalogValidationError("unexpected operator catalogue schema")
    if snapshot.get("schema_version") != "frozen_factor_catalog/v1":
        raise FactorCatalogValidationError("unexpected frozen snapshot schema")

    factors = _require_list(factor_catalog.get("factors"), label="factor_catalog.factors")
    operators = _require_list(operator_catalog.get("operators"), label="operator_catalog.operators")
    snapshot_factors = _require_list(snapshot.get("factors"), label="snapshot.factors")
    override_rows = _require_list(overrides.get("overrides"), label="manual_overrides.overrides")

    declared_factor_count = int(factor_catalog.get("factor_count", -1))
    declared_operator_count = int(operator_catalog.get("operator_count", -1))
    if declared_factor_count != len(factors):
        raise FactorCatalogValidationError("factor_count does not match factor rows")
    if declared_operator_count != len(operators):
        raise FactorCatalogValidationError("operator_count does not match operator rows")
    if int(snapshot.get("factor_count", -1)) != len(snapshot_factors):
        raise FactorCatalogValidationError("snapshot factor_count does not match rows")

    operator_by_id: dict[str, Mapping[str, Any]] = {}
    for row_raw in operators:
        row = _require_mapping(row_raw, label="operator row")
        operator_id = str(row.get("operator_id", "")).strip()
        if not operator_id or operator_id in operator_by_id:
            raise FactorCatalogValidationError(f"duplicate or empty operator_id: {operator_id!r}")
        implementation_rel = _relative_path(
            row.get("implementation_path"), field=f"operator implementation_path {operator_id}"
        )
        implementation_path = root / implementation_rel
        if not implementation_path.is_file():
            raise FactorCatalogValidationError(
                f"missing runtime operator implementation for {operator_id}: {implementation_rel}"
            )
        if not str(row.get("implementation_module", "")).startswith(
            "cbond_on.domain.factors.operators."
        ):
            raise FactorCatalogValidationError(f"operator implementation module is outside operators: {operator_id}")
        if str(row.get("implementation_sha256", "")).strip().lower() != hashlib.sha256(
            implementation_path.read_bytes()
        ).hexdigest():
            raise FactorCatalogValidationError(f"operator implementation hash mismatch: {operator_id}")
        contract_rel = _relative_path(
            row.get("operator_contract_path"), field=f"operator contract path {operator_id}"
        )
        contract_path = engine / contract_rel
        operator_contract = _require_mapping(
            load_json(contract_path), label=f"operator contract {operator_id}"
        )
        if operator_contract.get("schema_version") != "operator_contract/v1":
            raise FactorCatalogValidationError(f"unexpected operator contract schema: {operator_id}")
        expected_operator_hash = _expected_contract_hash(operator_contract)
        if (
            operator_contract.get("contract_hash") != expected_operator_hash
            or row.get("operator_contract_hash") != expected_operator_hash
        ):
            raise FactorCatalogValidationError(f"operator contract hash mismatch: {operator_id}")
        operator_identity = _require_mapping(
            operator_contract.get("identity"), label=f"operator identity {operator_id}"
        )
        if operator_identity.get("operator_id") != operator_id:
            raise FactorCatalogValidationError(f"operator contract identity mismatch: {operator_id}")
        operator_definition_rel = _relative_path(
            row.get("operator_definition_path"), field=f"operator definition path {operator_id}"
        )
        operator_definition_path = engine / operator_definition_rel
        if not operator_definition_path.is_file():
            raise FactorCatalogValidationError(f"missing operator definition: {operator_id}")
        operator_definition_text = operator_definition_path.read_text(encoding="utf-8")
        if "GENERATED_OPERATOR_DEFINITION = True" not in operator_definition_text:
            raise FactorCatalogValidationError(f"operator definition marker missing: {operator_id}")
        definition = _require_mapping(
            operator_contract.get("definition"), label=f"operator definition {operator_id}"
        )
        if definition.get("path") != operator_definition_rel.as_posix():
            raise FactorCatalogValidationError(f"operator definition path mismatch: {operator_id}")
        if definition.get("sha256") != hashlib.sha256(
            operator_definition_text.encode("utf-8")
        ).hexdigest():
            raise FactorCatalogValidationError(f"operator definition hash mismatch: {operator_id}")
        operator_impl = _require_mapping(
            operator_contract.get("implementation"), label=f"operator implementation {operator_id}"
        )
        for field, actual in {
            "module": row.get("implementation_module"),
            "path": row.get("implementation_path"),
            "sha256": row.get("implementation_sha256"),
        }.items():
            if operator_impl.get(field) != actual:
                raise FactorCatalogValidationError(
                    f"operator contract implementation {field} mismatch: {operator_id}"
                )
        operator_by_id[operator_id] = row

    snapshot_by_id: dict[str, Mapping[str, Any]] = {}
    for row_raw in snapshot_factors:
        row = _require_mapping(row_raw, label="snapshot factor row")
        factor_id = str(row.get("factor_id", "")).strip()
        if not factor_id or factor_id in snapshot_by_id:
            raise FactorCatalogValidationError(f"duplicate or empty snapshot factor_id: {factor_id!r}")
        snapshot_by_id[factor_id] = row

    overrides_by_id: dict[str, Mapping[str, Any]] = {}
    for row_raw in override_rows:
        row = _require_mapping(row_raw, label="manual override row")
        factor_id = str(row.get("factor_id", "")).strip()
        if not factor_id or factor_id in overrides_by_id:
            raise FactorCatalogValidationError(f"duplicate or empty override factor_id: {factor_id!r}")
        if not str(row.get("reason", "")).strip():
            raise FactorCatalogValidationError(f"override missing reason: {factor_id}")
        overrides_by_id[factor_id] = row

    factor_ids: set[str] = set()
    live_released_factor_ids: set[str] = set()
    factors_by_operator: dict[str, set[str]] = {operator_id: set() for operator_id in operator_by_id}
    for row_raw in factors:
        row = _require_mapping(row_raw, label="factor row")
        factor_id = str(row.get("factor_id", "")).strip()
        if not factor_id or factor_id in factor_ids:
            raise FactorCatalogValidationError(f"duplicate or empty factor_id: {factor_id!r}")
        factor_ids.add(factor_id)

        snapshot_row = snapshot_by_id.get(factor_id)
        if snapshot_row is None:
            raise FactorCatalogValidationError(f"factor absent from frozen snapshot: {factor_id}")
        if row.get("primary_family") != snapshot_row.get("primary_family"):
            raise FactorCatalogValidationError(f"family drift for factor: {factor_id}")
        if row.get("source_set") != snapshot_row.get("source_set"):
            raise FactorCatalogValidationError(f"source-set drift for factor: {factor_id}")
        if row.get("source_position") != snapshot_row.get("source_position"):
            raise FactorCatalogValidationError(f"source-position drift for factor: {factor_id}")

        definition_rel = _relative_path(row.get("definition_path"), field="definition_path")
        contract_rel = _relative_path(row.get("contract_path"), field="contract_path")
        definition_path = engine / definition_rel
        contract_path = engine / contract_rel
        if not definition_path.is_file() or not contract_path.is_file():
            raise FactorCatalogValidationError(f"missing definition or contract for {factor_id}")
        definition_text = definition_path.read_text(encoding="utf-8")
        if "GENERATED_FACTOR_DEFINITION = True" not in definition_text:
            raise FactorCatalogValidationError(f"definition wrapper marker missing: {factor_id}")
        if "def definition_payload()" not in definition_text:
            raise FactorCatalogValidationError(f"definition payload entry missing: {factor_id}")
        if "def build_factor_spec():" not in definition_text:
            raise FactorCatalogValidationError(f"FactorSpec build entry missing: {factor_id}")
        if "from cbond_on.domain.factors.spec import FactorSpec" not in definition_text:
            raise FactorCatalogValidationError(f"FactorSpec lazy import missing: {factor_id}")
        if "cbond_on.domain.factors.operators" in definition_text:
            raise FactorCatalogValidationError(f"definition imports retired defs: {factor_id}")
        if "FactorRegistry" in definition_text or "@" in definition_text:
            raise FactorCatalogValidationError(f"definition wrapper is not runtime-isolated: {factor_id}")

        contract = _require_mapping(load_json(contract_path), label=f"contract {factor_id}")
        if contract.get("schema_version") != "factor_contract/v2":
            raise FactorCatalogValidationError(f"unexpected factor contract schema: {factor_id}")
        identity = _require_mapping(contract.get("identity"), label=f"contract identity {factor_id}")
        if identity.get("factor_id") != factor_id:
            raise FactorCatalogValidationError(f"contract identity mismatch: {factor_id}")
        if identity.get("primary_family") != row.get("primary_family"):
            raise FactorCatalogValidationError(f"contract family mismatch: {factor_id}")
        if identity.get("factor_version") != row.get("factor_version"):
            raise FactorCatalogValidationError(f"contract version mismatch: {factor_id}")
        lineage = _require_mapping(contract.get("source_lineage"), label=f"source lineage {factor_id}")
        if lineage.get("source_set") != snapshot_row.get("source_set"):
            raise FactorCatalogValidationError(f"contract source-set mismatch: {factor_id}")
        if lineage.get("source_position") != snapshot_row.get("source_position"):
            raise FactorCatalogValidationError(f"contract source-position mismatch: {factor_id}")
        if lineage.get("source_root") != snapshot_row.get("source_root"):
            raise FactorCatalogValidationError(f"contract source-root mismatch: {factor_id}")
        if contract.get("fixed_params") != snapshot_row.get("fixed_params"):
            raise FactorCatalogValidationError(f"contract fixed-params mismatch: {factor_id}")
        expected_hash = _expected_contract_hash(contract)
        if contract.get("contract_hash") != expected_hash or row.get("contract_hash") != expected_hash:
            raise FactorCatalogValidationError(f"contract hash mismatch: {factor_id}")
        output = _require_mapping(contract.get("output"), label=f"contract output {factor_id}")
        if output.get("column") != factor_id:
            raise FactorCatalogValidationError(f"output column mismatch: {factor_id}")
        bindings = _require_list(contract.get("operator_bindings"), label=f"bindings {factor_id}")
        if len(bindings) != 1:
            raise FactorCatalogValidationError(f"factor must bind exactly one operator: {factor_id}")
        binding = _require_mapping(bindings[0], label=f"binding {factor_id}")
        operator_id = str(binding.get("operator_id", "")).strip()
        if operator_id not in operator_by_id:
            raise FactorCatalogValidationError(f"unknown operator for factor {factor_id}: {operator_id}")
        if row.get("operator_id") != operator_id:
            raise FactorCatalogValidationError(f"catalog/operator mismatch: {factor_id}")
        expected_operator_contract_path = operator_by_id[operator_id].get("operator_contract_path")
        if binding.get("operator_contract_path") != expected_operator_contract_path:
            raise FactorCatalogValidationError(f"factor/operator contract path mismatch: {factor_id}")
        if binding.get("implementation_module") != operator_by_id[operator_id].get(
            "implementation_module"
        ):
            raise FactorCatalogValidationError(f"factor/operator implementation mismatch: {factor_id}")
        rust_contract_id = row.get("rust_contract_id")
        lifecycle = _require_mapping(row.get("lifecycle"), label=f"catalog lifecycle {factor_id}")
        contract_lifecycle = _require_mapping(
            contract.get("lifecycle"), label=f"contract lifecycle {factor_id}"
        )
        if binding.get("rust_contract_id") != rust_contract_id:
            raise FactorCatalogValidationError(f"rust contract mismatch: {factor_id}")
        if rust_contract_id is not None:
            if lifecycle.get("live_admission_status") != "live_released":
                raise FactorCatalogValidationError(f"live release lifecycle mismatch: {factor_id}")
            if lifecycle.get("rust_status") != "live50_rust_capability_required":
                raise FactorCatalogValidationError(f"rust lifecycle mismatch: {factor_id}")
            if contract_lifecycle.get("live_admission_status") != "live_released":
                raise FactorCatalogValidationError(f"contract live lifecycle mismatch: {factor_id}")
            if contract_lifecycle.get("rust_status") != "live50_rust_capability_required":
                raise FactorCatalogValidationError(f"contract rust lifecycle mismatch: {factor_id}")
            live_released_factor_ids.add(factor_id)
        factors_by_operator[operator_id].add(factor_id)

    if factor_ids != set(snapshot_by_id):
        missing = sorted(set(snapshot_by_id).difference(factor_ids))
        extra = sorted(factor_ids.difference(snapshot_by_id))
        raise FactorCatalogValidationError(f"factor snapshot mismatch: missing={missing}, extra={extra}")
    if len(live_released_factor_ids) != 50:
        raise FactorCatalogValidationError(
            f"expected exactly 50 live-released factors, found {len(live_released_factor_ids)}"
        )

    for operator_id, row in operator_by_id.items():
        declared = _require_list(row.get("factor_ids"), label=f"operator factors {operator_id}")
        declared_set = {str(item).strip() for item in declared}
        if len(declared_set) != len(declared) or any(not item for item in declared_set):
            raise FactorCatalogValidationError(f"invalid factor_ids for operator: {operator_id}")
        if declared_set != factors_by_operator[operator_id]:
            raise FactorCatalogValidationError(f"operator membership mismatch: {operator_id}")

    release_dir = engine / "releases" / "live"
    release_count = 0
    if release_dir.is_dir():
        for release_path in sorted(release_dir.glob("*.json")):
            release = _require_mapping(load_json(release_path), label=f"live release {release_path.name}")
            if release.get("schema_version") != "factor_catalog_live_release/v1":
                raise FactorCatalogValidationError(f"unexpected live release schema: {release_path}")
            release_id = str(release.get("release_id", "")).strip()
            if _release_path(engine, release_id) != release_path:
                raise FactorCatalogValidationError(f"live release path/id mismatch: {release_path}")
            instances = _require_list(release.get("instances"), label=f"live release instances {release_id}")
            if int(release.get("factor_count", -1)) != len(instances):
                raise FactorCatalogValidationError(f"live release factor_count mismatch: {release_id}")
            seen_positions: set[int] = set()
            seen_factor_ids: set[str] = set()
            for raw in instances:
                instance = _require_mapping(raw, label=f"live release instance {release_id}")
                factor_id = str(instance.get("factor_id", "")).strip()
                try:
                    position = int(instance.get("position", 0))
                except (TypeError, ValueError) as exc:
                    raise FactorCatalogValidationError(f"invalid release position: {release_id}") from exc
                if not factor_id or factor_id in seen_factor_ids or position in seen_positions:
                    raise FactorCatalogValidationError(f"duplicate live release instance: {release_id}")
                seen_factor_ids.add(factor_id)
                seen_positions.add(position)
                factor = next((row for row in factors if row.get("factor_id") == factor_id), None)
                if factor is None:
                    raise FactorCatalogValidationError(f"live release unknown factor: {factor_id}")
                for field in ("factor_version", "contract_hash", "operator_id"):
                    if instance.get(field) != factor.get(field):
                        raise FactorCatalogValidationError(
                            f"live release {field} mismatch for {factor_id}"
                        )
                if not str(instance.get("rust_contract_id", "")).strip():
                    raise FactorCatalogValidationError(f"live release missing rust contract: {factor_id}")
            if seen_positions != set(range(1, len(instances) + 1)):
                raise FactorCatalogValidationError(f"live release positions are not 1..N: {release_id}")
            release_count += 1

    return {
        "factor_count": len(factors),
        "operator_count": len(operators),
        "family_count": len({str(row["primary_family"]) for row in factors}),
        "manual_override_count": len(overrides_by_id),
        "live_release_count": release_count,
        "live_released_factor_count": len(live_released_factor_ids),
    }
