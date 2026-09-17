"""Read-only public resolver for the generated factor identity catalogue.

These functions intentionally inspect only source-controlled JSON metadata.
They do not import ``cbond_on.domain.factors.operators`` and therefore cannot make a
research or live factor executable merely by resolving its identity.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .validation import FactorCatalogValidationError, _repository_root, load_json


def catalog_path(repo_root: Path | None = None) -> Path:
    """Return the canonical full Factor Catalog JSON path."""

    return _repository_root(repo_root) / "factor_engine" / "catalog" / "factor_catalog.json"


def operator_catalog_path(repo_root: Path | None = None) -> Path:
    """Return the canonical full Operator Catalog JSON path."""

    return _repository_root(repo_root) / "factor_engine" / "catalog" / "operator_catalog.json"


def live_release_path(release_id: str, repo_root: Path | None = None) -> Path:
    """Return one immutable live-release metadata path without resolving it."""

    text = str(release_id).strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        raise FactorCatalogValidationError(f"unsafe release_id: {release_id!r}")
    return _repository_root(repo_root) / "factor_engine" / "releases" / "live" / f"{text}.json"


def _mapping_rows(payload: object, *, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("factors"), list):
        raise FactorCatalogValidationError(f"invalid {label}")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in payload["factors"]:
        if not isinstance(raw, dict):
            raise FactorCatalogValidationError(f"invalid {label} row")
        factor_id = str(raw.get("factor_id", "")).strip()
        if not factor_id or factor_id in result:
            raise FactorCatalogValidationError(f"duplicate factor_id in {label}: {factor_id!r}")
        result[factor_id] = MappingProxyType(dict(raw))
    return result


def load_factor_catalog(repo_root: Path | None = None) -> Mapping[str, Mapping[str, Any]]:
    """Load every registered factor instance keyed by immutable ``factor_id``."""

    return MappingProxyType(_mapping_rows(load_json(catalog_path(repo_root)), label="factor catalog"))


def resolve_factor_instance(
    factor_id: str,
    *,
    factor_version: str | None = None,
    contract_hash: str | None = None,
    repo_root: Path | None = None,
) -> Mapping[str, Any]:
    """Resolve one factor and optionally pin its version and contract hash.

    A future live admission resolver should pass all three identity values.  A
    mismatch is fail-closed rather than silently accepting a newer definition.
    """

    key = str(factor_id).strip()
    row = load_factor_catalog(repo_root).get(key)
    if row is None:
        raise FactorCatalogValidationError(f"unknown factor_id: {key!r}")
    if factor_version is not None and row.get("factor_version") != str(factor_version).strip():
        raise FactorCatalogValidationError(f"factor_version mismatch: {key}")
    if contract_hash is not None and row.get("contract_hash") != str(contract_hash).strip():
        raise FactorCatalogValidationError(f"contract_hash mismatch: {key}")
    return row


def resolve_operator_modules(
    factor_ids: Iterable[str],
    *,
    repo_root: Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return exact operator metadata needed by a resolved factor-id set.

    Results are metadata only: caller-owned admission code decides whether a
    module is allowed to be imported in a particular environment.
    """

    requested = [str(factor_id).strip() for factor_id in factor_ids]
    if not requested or any(not factor_id for factor_id in requested):
        raise FactorCatalogValidationError("factor_ids must contain at least one non-empty factor_id")
    operator_ids = {str(resolve_factor_instance(factor_id, repo_root=repo_root)["operator_id"]) for factor_id in requested}
    payload = load_json(operator_catalog_path(repo_root))
    if not isinstance(payload, dict) or not isinstance(payload.get("operators"), list):
        raise FactorCatalogValidationError("invalid operator catalog")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in payload["operators"]:
        if not isinstance(raw, dict):
            raise FactorCatalogValidationError("invalid operator catalog row")
        operator_id = str(raw.get("operator_id", "")).strip()
        if not operator_id or operator_id in by_id:
            raise FactorCatalogValidationError(f"duplicate operator_id: {operator_id!r}")
        by_id[operator_id] = MappingProxyType(dict(raw))
    missing = sorted(operator_ids.difference(by_id))
    if missing:
        raise FactorCatalogValidationError(f"factor catalog references unregistered operator(s): {missing}")
    return tuple(by_id[operator_id] for operator_id in sorted(operator_ids))


def load_live_release(
    release_id: str,
    *,
    repo_root: Path | None = None,
) -> Mapping[str, Any]:
    """Load an immutable release and fail if it disagrees with the full catalog.

    This is metadata resolution only.  It does not authorize the release or
    import the referenced operator modules.
    """

    payload = load_json(live_release_path(release_id, repo_root))
    if not isinstance(payload, dict) or payload.get("release_id") != str(release_id).strip():
        raise FactorCatalogValidationError(f"invalid live release: {release_id!r}")
    instances = payload.get("instances")
    if not isinstance(instances, list) or int(payload.get("factor_count", -1)) != len(instances):
        raise FactorCatalogValidationError(f"invalid live release instances: {release_id!r}")
    for item in instances:
        if not isinstance(item, dict):
            raise FactorCatalogValidationError(f"invalid live release instance: {release_id!r}")
        resolved = resolve_factor_instance(
            str(item.get("factor_id", "")),
            factor_version=str(item.get("factor_version", "")),
            contract_hash=str(item.get("contract_hash", "")),
            repo_root=repo_root,
        )
        if item.get("operator_id") != resolved.get("operator_id"):
            raise FactorCatalogValidationError(
                f"live release operator mismatch: {item.get('factor_id')!r}"
            )
        if not str(item.get("rust_contract_id", "")).strip():
            raise FactorCatalogValidationError(
                f"live release missing rust contract: {item.get('factor_id')!r}"
            )
        if item.get("rust_contract_id") != resolved.get("rust_contract_id"):
            raise FactorCatalogValidationError(
                f"live release rust contract mismatch: {item.get('factor_id')!r}"
            )
    return MappingProxyType(dict(payload))
