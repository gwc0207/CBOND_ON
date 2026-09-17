"""Canonical, local storage for the three CBOND_ON factor tables.

This module owns *only* the storage contract.  It does not compute factors,
resolve a live release, read a panel, or migrate an existing FactorStore.

The root has exactly three data-table children:

``factor_library``
    ``因子库内因子表``.  Its contents are partitioned by registered factor
    family, while still belonging to one canonical table.
``experiment``
    ``实验用因子表``.  One wide table for the currently declared experiment
    factor set.
``live``
    ``实盘所需要的因子表``.  One wide table for the currently admitted live
    factor set.

Every published day is a small commit bundle: parquet first, then an atomic
day manifest, then an atomic ``.done`` marker.  Consumers must treat the done
marker as the visibility boundary.  A missing member of a bundle is an
integrity failure, never a signal to overwrite or recompute it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


TABLE_MANIFEST_SCHEMA = "cbond_on_canonical_factor_table/v1"
CONTRACT_SCHEMA = "cbond_on_canonical_factor_table_contract/v1"
DAY_MANIFEST_SCHEMA = "cbond_on_canonical_factor_day_manifest/v1"
DAY_DONE_SCHEMA = "cbond_on_canonical_factor_day_done/v1"
FRAME_HASH_SEMANTICS = "cbond_on_factor_frame_semantic_datetime_ns/v1"
LIBRARY_DAY_MANIFEST_SCHEMA = "cbond_on_canonical_factor_library_day_manifest/v1"
LIBRARY_DAY_DONE_SCHEMA = "cbond_on_canonical_factor_library_day_done/v1"
EXPERIMENT_GENERATION_POINTER_SCHEMA = "cbond_on_canonical_factor_experiment_generation_pointer/v1"
EXPERIMENT_GENERATION_SCHEMA = "cbond_on_canonical_factor_experiment_generation/v1"
EXPERIMENT_GENERATION_DONE_SCHEMA = "cbond_on_canonical_factor_experiment_generation_done/v1"
LEGACY_EXPERIMENT_GENERATION_ID = "legacy_flat"

PANEL_NAME = "T1430"
LOGICAL_PANEL_TIME = "14:30"


class CanonicalFactorStoreError(RuntimeError):
    """Base error for the canonical factor-table contract."""


class CanonicalFactorStoreValidationError(CanonicalFactorStoreError, ValueError):
    """Caller-provided frame, identity, or path data violates the contract."""


class CanonicalFactorStoreIntegrityError(CanonicalFactorStoreError):
    """A previously persisted canonical artifact is incomplete or corrupt."""


class CanonicalFactorStoreConflictError(CanonicalFactorStoreError):
    """A complete day exists with different immutable values or contracts."""


class CanonicalFactorStoreLockError(CanonicalFactorStoreError):
    """A different writer already owns the requested canonical partition."""


class FactorTableKind(str, Enum):
    """The only three legal canonical factor-table identities."""

    FACTOR_LIBRARY = "factor_library"
    EXPERIMENT = "experiment"
    LIVE = "live"

    @property
    def display_name(self) -> str:
        return _DISPLAY_NAMES[self]

    @property
    def family_aware(self) -> bool:
        return self is FactorTableKind.FACTOR_LIBRARY


_DISPLAY_NAMES: dict[FactorTableKind, str] = {
    FactorTableKind.FACTOR_LIBRARY: "因子库内因子表",
    FactorTableKind.EXPERIMENT: "实验用因子表",
    FactorTableKind.LIVE: "实盘所需要的因子表",
}

_KIND_ALIASES: dict[str, FactorTableKind] = {
    **{kind.value: kind for kind in FactorTableKind},
    **{kind.display_name: kind for kind in FactorTableKind},
}
_TABLE_IDS = tuple(kind.value for kind in FactorTableKind)
_FAMILY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GENERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FACTOR_LIBRARY_TABLE_METADATA_DIRS = frozenset({"locks", "day_manifests", "day_done"})
_WIDE_TABLE_METADATA_DIRS = frozenset({"factors", "manifests", "done", "contracts", "locks"})
_EXPERIMENT_TABLE_METADATA_DIRS = frozenset({*_WIDE_TABLE_METADATA_DIRS, "generations"})


def coerce_factor_table_kind(value: FactorTableKind | str) -> FactorTableKind:
    """Resolve only a canonical table ID or its exact owner-facing label."""

    if isinstance(value, FactorTableKind):
        return value
    text = str(value).strip()
    kind = _KIND_ALIASES.get(text)
    if kind is None:
        raise CanonicalFactorStoreValidationError(
            "factor table kind must be one of "
            f"{list(_TABLE_IDS)!r} or {[kind.display_name for kind in FactorTableKind]!r}; got {value!r}"
        )
    return kind


@dataclass(frozen=True)
class FactorColumnContract:
    """One registered factor identity as materialised in a wide factor frame."""

    factor_id: str
    factor_version: str
    contract_hash: str
    output_column: str | None = None
    # ``contract_hash`` above is the Catalog identity contract.  These optional
    # fields state whether the persisted values were actually materialised by
    # that current contract, or are an auditable historical source copy.
    materialization_contract_origin: str = "current_catalog"
    materialization_contract_hash: str | None = None
    materialization_source_attestation_sha256: str | None = None

    _MATERIALIZATION_ORIGINS = frozenset(
        {
            "current_catalog",
            "source_attested",
            "source_manifest_unversioned",
        }
    )

    def __post_init__(self) -> None:
        factor_id = str(self.factor_id).strip()
        version = str(self.factor_version).strip()
        contract_hash = str(self.contract_hash).strip().lower()
        output_column = str(self.output_column or factor_id).strip()
        materialization_origin = str(self.materialization_contract_origin).strip().lower()
        materialization_contract_hash = (
            str(self.materialization_contract_hash).strip().lower()
            if self.materialization_contract_hash is not None
            else None
        )
        materialization_attestation = (
            str(self.materialization_source_attestation_sha256).strip().lower()
            if self.materialization_source_attestation_sha256 is not None
            else None
        )
        if not factor_id:
            raise CanonicalFactorStoreValidationError("factor contract factor_id must be non-empty")
        if not version:
            raise CanonicalFactorStoreValidationError(f"factor {factor_id!r} has an empty factor_version")
        if not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
            raise CanonicalFactorStoreValidationError(
                f"factor {factor_id!r} contract_hash must be a 64-character lowercase SHA-256"
            )
        if not output_column or output_column in {"dt", "code"}:
            raise CanonicalFactorStoreValidationError(
                f"factor {factor_id!r} has an invalid output column {output_column!r}"
            )
        if materialization_origin not in self._MATERIALIZATION_ORIGINS:
            raise CanonicalFactorStoreValidationError(
                f"factor {factor_id!r} has an invalid materialization_contract_origin "
                f"{materialization_origin!r}"
            )
        if materialization_contract_hash is not None and not re.fullmatch(
            r"[0-9a-f]{64}", materialization_contract_hash
        ):
            raise CanonicalFactorStoreValidationError(
                f"factor {factor_id!r} materialization_contract_hash must be a 64-character SHA-256"
            )
        if materialization_attestation is not None and not re.fullmatch(r"[0-9a-f]{64}", materialization_attestation):
            raise CanonicalFactorStoreValidationError(
                f"factor {factor_id!r} materialization_source_attestation_sha256 must be a 64-character SHA-256"
            )
        if materialization_origin == "current_catalog":
            if materialization_contract_hash not in {None, contract_hash}:
                raise CanonicalFactorStoreValidationError(
                    f"factor {factor_id!r} current_catalog materialization hash differs from Catalog identity"
                )
            if materialization_attestation is not None:
                raise CanonicalFactorStoreValidationError(
                    f"factor {factor_id!r} current_catalog materialization must not carry a source attestation hash"
                )
            materialization_contract_hash = contract_hash
        elif materialization_origin == "source_attested":
            if materialization_contract_hash is None or materialization_attestation is None:
                raise CanonicalFactorStoreValidationError(
                    f"factor {factor_id!r} source_attested materialization requires source contract and attestation hashes"
                )
        elif materialization_origin == "source_manifest_unversioned":
            if materialization_contract_hash is not None or materialization_attestation is None:
                raise CanonicalFactorStoreValidationError(
                    f"factor {factor_id!r} source_manifest_unversioned materialization requires only an attestation hash"
                )
        object.__setattr__(self, "factor_id", factor_id)
        object.__setattr__(self, "factor_version", version)
        object.__setattr__(self, "contract_hash", contract_hash)
        object.__setattr__(self, "output_column", output_column)
        object.__setattr__(self, "materialization_contract_origin", materialization_origin)
        object.__setattr__(self, "materialization_contract_hash", materialization_contract_hash)
        object.__setattr__(self, "materialization_source_attestation_sha256", materialization_attestation)

    def to_dict(self) -> dict[str, str]:
        return {
            "factor_id": self.factor_id,
            "factor_version": self.factor_version,
            "contract_hash": self.contract_hash,
            "output_column": str(self.output_column),
            "materialization_contract_origin": self.materialization_contract_origin,
            "materialization_contract_hash": self.materialization_contract_hash,
            "materialization_source_attestation_sha256": self.materialization_source_attestation_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorColumnContract":
        return cls(
            factor_id=str(payload.get("factor_id", "")),
            factor_version=str(payload.get("factor_version", "")),
            contract_hash=str(payload.get("contract_hash", "")),
            output_column=str(payload.get("output_column", "") or ""),
            materialization_contract_origin=str(payload.get("materialization_contract_origin", "current_catalog")),
            materialization_contract_hash=(
                str(payload.get("materialization_contract_hash"))
                if payload.get("materialization_contract_hash") is not None
                else None
            ),
            materialization_source_attestation_sha256=(
                str(payload.get("materialization_source_attestation_sha256"))
                if payload.get("materialization_source_attestation_sha256") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class FactorTableContract:
    """Ordered factor identity/contract list for one canonical table day.

    A contract list is deliberately immutable.  Adding or removing a factor is
    represented by a new list hash and a new contract document; it never
    overwrites a completed date partition.
    """

    columns: tuple[FactorColumnContract, ...]

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        if not columns:
            raise CanonicalFactorStoreValidationError("factor table contract must contain at least one factor")
        if not all(isinstance(column, FactorColumnContract) for column in columns):
            raise CanonicalFactorStoreValidationError("factor table contract columns must be FactorColumnContract values")
        factor_ids = [column.factor_id for column in columns]
        output_columns = [str(column.output_column) for column in columns]
        if len(factor_ids) != len(set(factor_ids)):
            raise CanonicalFactorStoreValidationError("factor table contract has duplicate factor_id values")
        if len(output_columns) != len(set(output_columns)):
            raise CanonicalFactorStoreValidationError("factor table contract has duplicate output_column values")
        object.__setattr__(self, "columns", columns)

    @property
    def factor_ids(self) -> tuple[str, ...]:
        return tuple(column.factor_id for column in self.columns)

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(str(column.output_column) for column in self.columns)

    def to_dict(self) -> dict[str, Any]:
        origins = {column.materialization_contract_origin for column in self.columns}
        materialization_contract_status = next(iter(origins)) if len(origins) == 1 else "mixed"
        return {
            "schema_version": CONTRACT_SCHEMA,
            "factor_count": len(self.columns),
            "factor_ids": list(self.factor_ids),
            "output_columns": list(self.output_columns),
            "materialization_contract_status": materialization_contract_status,
            "columns": [column.to_dict() for column in self.columns],
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))

    @classmethod
    def from_catalog_entries(cls, entries: Sequence[Mapping[str, Any]]) -> "FactorTableContract":
        """Build an ordered storage contract from canonical Catalog rows.

        Catalog rows use ``factor_id``, ``factor_version`` and
        ``contract_hash``.  The materialised output normally has the same name
        as ``factor_id``; ``output_column`` remains available for a registered
        factor whose runtime output name is deliberately different.
        """

        return cls(
            tuple(
                FactorColumnContract(
                    factor_id=str(entry.get("factor_id", "")),
                    factor_version=str(entry.get("factor_version", "")),
                    contract_hash=str(entry.get("contract_hash", "")),
                    output_column=(
                        str(entry.get("output_column", "")).strip()
                        or str(entry.get("factor_id", ""))
                    ),
                )
                for entry in entries
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FactorTableContract":
        if str(payload.get("schema_version", "")) != CONTRACT_SCHEMA:
            raise CanonicalFactorStoreIntegrityError("factor contract document has an unexpected schema_version")
        columns = payload.get("columns")
        if not isinstance(columns, list):
            raise CanonicalFactorStoreIntegrityError("factor contract document has no columns list")
        contract = cls(tuple(FactorColumnContract.from_dict(item) for item in columns if isinstance(item, Mapping)))
        if len(contract.columns) != len(columns):
            raise CanonicalFactorStoreIntegrityError("factor contract document has a non-object column entry")
        if list(payload.get("factor_ids", [])) != list(contract.factor_ids):
            raise CanonicalFactorStoreIntegrityError("factor contract document factor_ids differ from columns")
        if list(payload.get("output_columns", [])) != list(contract.output_columns):
            raise CanonicalFactorStoreIntegrityError("factor contract document output_columns differ from columns")
        if int(payload.get("factor_count", -1)) != len(contract.columns):
            raise CanonicalFactorStoreIntegrityError("factor contract document factor_count differs from columns")
        origins = {column.materialization_contract_origin for column in contract.columns}
        expected_status = next(iter(origins)) if len(origins) == 1 else "mixed"
        observed_status = payload.get("materialization_contract_status", "current_catalog")
        if observed_status != expected_status:
            raise CanonicalFactorStoreIntegrityError(
                "factor contract document materialization_contract_status differs from columns"
            )
        return contract


@dataclass(frozen=True)
class FactorDayPaths:
    """The three published artifacts for one canonical table day."""

    table_root: Path
    scope_root: Path
    parquet_path: Path
    manifest_path: Path
    done_path: Path
    lock_path: Path


@dataclass(frozen=True)
class FactorLibraryLogicalDayPaths:
    """The commit bundle that makes all family partitions visible as one day."""

    table_root: Path
    manifest_path: Path
    done_path: Path
    lock_path: Path


@dataclass(frozen=True)
class ExperimentGenerationPaths:
    """One immutable internal generation of the experiment factor table.

    ``legacy_flat`` deliberately points at the historical flat experiment
    layout.  It exists so the first generation migration never has to move or
    rewrite previously published parquet bundles.  New generations are always
    isolated below ``experiment/generations/<generation_id>`` until their
    own full-verification ``generation.done`` marker is committed and the
    table pointer is atomically activated.
    """

    table_root: Path
    generation_id: str
    generation_root: Path
    manifest_path: Path
    done_path: Path | None
    rebuild_lock_path: Path

    @property
    def is_legacy_flat(self) -> bool:
        return self.generation_id == LEGACY_EXPERIMENT_GENERATION_ID


@dataclass(frozen=True)
class FactorTableWriteResult:
    status: str
    table_kind: FactorTableKind
    day: date
    family: str | None
    paths: FactorDayPaths
    contract_sha256: str
    row_count: int


@dataclass(frozen=True)
class CanonicalFactorTableReport:
    table_id: str
    display_name: str
    root: str
    exists: bool
    manifest_valid: bool
    family_aware: bool
    families: tuple[str, ...]
    registered_contract_count: int
    completed_day_count: int
    incomplete_day_count: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "display_name": self.display_name,
            "root": self.root,
            "exists": self.exists,
            "manifest_valid": self.manifest_valid,
            "family_aware": self.family_aware,
            "families": list(self.families),
            "registered_contract_count": self.registered_contract_count,
            "completed_day_count": self.completed_day_count,
            "incomplete_day_count": self.incomplete_day_count,
            "error": self.error,
        }


@dataclass(frozen=True)
class CanonicalFactorStoreReport:
    root: str
    allowed_table_ids: tuple[str, ...]
    unexpected_root_entries: tuple[str, ...]
    tables: tuple[CanonicalFactorTableReport, ...]

    @property
    def is_canonical_layout(self) -> bool:
        return not self.unexpected_root_entries

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "cbond_on_canonical_factor_store_report/v1",
            "root": self.root,
            "allowed_table_ids": list(self.allowed_table_ids),
            "unexpected_root_entries": list(self.unexpected_root_entries),
            "is_canonical_layout": self.is_canonical_layout,
            "tables": [table.to_dict() for table in self.tables],
        }


@dataclass(frozen=True)
class _LockHandle:
    path: Path
    token: str


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _canonical_json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CanonicalFactorStoreIntegrityError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CanonicalFactorStoreIntegrityError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CanonicalFactorStoreIntegrityError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact in its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_json_bytes(dict(payload)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:  # pragma: no cover - internal invariant guard.
        raise CanonicalFactorStoreIntegrityError(f"artifact is outside its table root: {path}") from exc


def _coerce_day(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise CanonicalFactorStoreValidationError(f"day must be datetime.date, got {type(value).__name__}")


def _normalise_family(kind: FactorTableKind, family: str | None) -> str | None:
    if kind.family_aware:
        text = str(family or "").strip()
        if not _FAMILY_PATTERN.fullmatch(text):
            raise CanonicalFactorStoreValidationError(
                "因子库内因子表 requires a non-empty family containing only letters, digits, '.', '_' or '-'"
            )
        return text
    if family is not None:
        raise CanonicalFactorStoreValidationError(
            f"{kind.display_name} is one wide table and does not accept a family scope"
        )
    return None


def _hash_pandas_object(value: pd.DataFrame | pd.Series) -> bytes:
    hashes = pd.util.hash_pandas_object(value, index=True, categorize=True).to_numpy(dtype=np.uint64, copy=False)
    return hashes.tobytes()


def _legacy_frame_hashes(frame: pd.DataFrame) -> dict[str, Any]:
    """The original unit-sensitive hashes retained for legacy bundle reads."""

    index_frame = frame.index.to_frame(index=False)
    index_frame.columns = ["dt", "code"]
    index_hash = _sha256_bytes(_hash_pandas_object(index_frame))
    column_hashes: dict[str, str] = {}
    for column in frame.columns:
        column_hashes[str(column)] = _sha256_bytes(_hash_pandas_object(frame[[column]]))
    schema = {
        "index_names": [str(name) for name in frame.index.names],
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json_bytes(schema))
    digest.update(_hash_pandas_object(index_frame))
    for column in frame.columns:
        digest.update(str(column).encode("utf-8"))
        digest.update(_hash_pandas_object(frame[[column]]))
    return {
        "frame_sha256": digest.hexdigest(),
        "index_sha256": index_hash,
        "column_sha256": column_hashes,
    }


def _frame_hashes(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash factor values with a unit-independent datetime index view.

    A logical T1430 timestamp can arrive as ``datetime64[s]`` and round-trip
    through Parquet as ``datetime64[ms]`` or ``datetime64[us]``.  Those are
    the same data, so newly written canonical manifests hash a temporary
    ``datetime64[ns]`` view rather than backend-specific physical units.  The
    caller's frame itself is never recast; existing readers keep their prior
    dtype behavior and legacy manifests remain verifiable separately.
    """

    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["dt", "code"]:
        raise CanonicalFactorStoreValidationError("canonical frame hash requires a (dt, code) MultiIndex")
    timestamp_values = pd.DatetimeIndex(frame.index.get_level_values("dt")).to_numpy(dtype="datetime64[ns]")
    code_values = frame.index.get_level_values("code").to_numpy()
    semantic_index = pd.MultiIndex.from_arrays([timestamp_values, code_values], names=["dt", "code"])
    semantic_frame = frame.copy(deep=False)
    semantic_frame.index = semantic_index
    return _legacy_frame_hashes(semantic_frame)


def _normalise_factor_frame(
    frame: pd.DataFrame,
    *,
    day: date,
    contract: FactorTableContract,
) -> pd.DataFrame:
    """Return a deterministic ``(dt, code)`` frame or fail before persistence."""

    if not isinstance(frame, pd.DataFrame):
        raise CanonicalFactorStoreValidationError("factor day output must be a pandas DataFrame")
    if frame.empty:
        raise CanonicalFactorStoreValidationError("canonical factor day output may not be empty")

    working = frame.copy()
    if {"dt", "code"}.issubset(working.columns):
        if isinstance(working.index, pd.MultiIndex) and set(working.index.names).intersection({"dt", "code"}):
            raise CanonicalFactorStoreValidationError("factor frame has ambiguous dt/code in both index and columns")
        dt_values = working.pop("dt")
        code_values = working.pop("code")
    elif isinstance(working.index, pd.MultiIndex) and list(working.index.names) == ["dt", "code"]:
        dt_values = pd.Series(working.index.get_level_values("dt"), index=working.index)
        code_values = pd.Series(working.index.get_level_values("code"), index=working.index)
    else:
        raise CanonicalFactorStoreValidationError(
            "factor frame must have a two-level MultiIndex named ('dt', 'code') or explicit dt/code columns"
        )

    actual_columns = [str(column) for column in working.columns]
    if actual_columns != list(contract.output_columns):
        raise CanonicalFactorStoreValidationError(
            "factor frame columns differ from its ordered factor contract: "
            f"expected={list(contract.output_columns)!r}, actual={actual_columns!r}"
        )
    if working.columns.duplicated().any():
        raise CanonicalFactorStoreValidationError("factor frame has duplicate output columns")

    parsed_dt = pd.to_datetime(dt_values, errors="coerce")
    if bool(pd.isna(parsed_dt).any()):
        raise CanonicalFactorStoreValidationError("factor frame dt contains an unparseable value")
    parsed_index = pd.DatetimeIndex(parsed_dt)
    if parsed_index.tz is not None:
        raise CanonicalFactorStoreValidationError("factor frame dt must be timezone-naive")
    if not bool((parsed_index.normalize() == pd.Timestamp(day)).all()):
        observed = sorted({str(value) for value in parsed_index.normalize().unique()})[:5]
        raise CanonicalFactorStoreValidationError(
            f"T1430 partition {day.isoformat()} contains a different dt day: {observed!r}"
        )
    expected_t1430 = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    if not bool((parsed_index == expected_t1430).all()):
        observed_times = sorted({str(value) for value in parsed_index.unique()})[:5]
        raise CanonicalFactorStoreValidationError(
            f"T1430 partition {day.isoformat()} must use exact logical timestamp "
            f"{expected_t1430.isoformat()}, observed={observed_times!r}"
        )
    raw_codes = pd.Series(code_values, copy=False)
    if bool(raw_codes.isna().any()):
        raise CanonicalFactorStoreValidationError("factor frame code contains a missing value")
    codes = raw_codes.astype(str).str.strip().str.upper()
    if bool((codes == "").any()):
        raise CanonicalFactorStoreValidationError("factor frame code contains an empty value")

    numeric = pd.DataFrame(index=working.index)
    for column in contract.output_columns:
        series = working[column]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise CanonicalFactorStoreValidationError(
                f"factor column {column!r} must be numeric (NaN is allowed), got {series.dtype}"
            )
        numeric[column] = series.astype("float64")
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    if bool(np.isinf(values).any()):
        raise CanonicalFactorStoreValidationError("factor frame contains Inf or -Inf values")

    canonical_index = pd.MultiIndex.from_arrays(
        [parsed_index.to_numpy(), codes.to_numpy()], names=["dt", "code"]
    )
    numeric.index = canonical_index
    if bool(numeric.index.duplicated().any()):
        duplicates = numeric.index[numeric.index.duplicated()].tolist()[:5]
        raise CanonicalFactorStoreValidationError(f"factor frame has duplicate (dt, code) keys: {duplicates!r}")
    return numeric.sort_index(kind="mergesort")


class CanonicalFactorStore:
    """Service for the only three allowed local CBOND_ON factor tables.

    ``root`` is always supplied by the caller/configuration.  Constructing or
    reporting on the service is read-only.  ``initialize`` and ``write_day``
    are the only methods that create local artifacts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = _resolved(root)

    def table_root(self, kind: FactorTableKind | str) -> Path:
        return self.root / coerce_factor_table_kind(kind).value

    def resolve_table_root(self, kind: FactorTableKind | str, *, require_manifest: bool = True) -> Path:
        """Return a table root only after validating the canonical table identity."""

        resolved_kind = coerce_factor_table_kind(kind)
        self._assert_root_layout(require_all_tables=require_manifest)
        path = self.table_root(resolved_kind)
        if require_manifest:
            self._load_table_manifest(resolved_kind)
        if resolved_kind is FactorTableKind.EXPERIMENT:
            paths = self.experiment_generation_paths()
            if not paths.is_legacy_flat:
                self._load_experiment_generation_manifest(paths.generation_id)
                path = paths.generation_root
        return path

    def active_experiment_generation_id(self) -> str:
        """Return the currently published experiment generation ID.

        Old flat experiment tables predate generations.  They remain valid
        read-only inputs through the explicit ``legacy_flat`` identity instead
        of being moved or rewritten as part of the generation migration.
        """

        manifest = self.read_table_manifest(FactorTableKind.EXPERIMENT)
        return self._active_experiment_generation_id_from_manifest(manifest)

    def experiment_generation_paths(
        self,
        generation_id: str | None = None,
    ) -> ExperimentGenerationPaths:
        """Resolve the active or explicitly named experiment generation.

        This performs no writes.  A non-legacy ID is restricted to the one
        approved internal namespace beneath the canonical experiment table.
        """

        self._assert_root_layout(require_all_tables=True)
        table_root = self.table_root(FactorTableKind.EXPERIMENT)
        table_manifest = self._load_table_manifest(FactorTableKind.EXPERIMENT)
        resolved_id = (
            self._active_experiment_generation_id_from_manifest(table_manifest)
            if generation_id is None
            else self._normalise_experiment_generation_id(generation_id, allow_legacy=True)
        )
        rebuild_lock = table_root / "locks" / "experiment_rebuild.lock"
        if resolved_id == LEGACY_EXPERIMENT_GENERATION_ID:
            return ExperimentGenerationPaths(
                table_root=table_root,
                generation_id=resolved_id,
                generation_root=table_root,
                manifest_path=table_root / "table_manifest.json",
                done_path=None,
                rebuild_lock_path=rebuild_lock,
            )
        generation_root = table_root / "generations" / resolved_id
        return ExperimentGenerationPaths(
            table_root=table_root,
            generation_id=resolved_id,
            generation_root=generation_root,
            manifest_path=generation_root / "generation_manifest.json",
            done_path=generation_root / "generation.done",
            rebuild_lock_path=rebuild_lock,
        )

    def read_experiment_generation_manifest(self, generation_id: str | None = None) -> dict[str, Any]:
        """Read the active or named experiment generation manifest.

        For ``legacy_flat`` this intentionally returns the historical top
        table manifest so old data stays readable without a migration write.
        """

        paths = self.experiment_generation_paths(generation_id)
        if paths.is_legacy_flat:
            return self._load_table_manifest(FactorTableKind.EXPERIMENT)
        return self._load_experiment_generation_manifest(paths.generation_id)

    def require_experiment_generation_ready(self, generation_id: str | None = None) -> dict[str, Any]:
        """Return a consumer-ready experiment generation or fail closed."""

        paths = self.experiment_generation_paths(generation_id)
        if paths.is_legacy_flat:
            return self.require_consumer_ready(FactorTableKind.EXPERIMENT)
        manifest = self._load_experiment_generation_manifest(paths.generation_id)
        self._validate_experiment_generation_done(paths, manifest=manifest)
        return manifest

    def create_stage_generation(
        self,
        generation_id: str,
        *,
        source_evidence: Mapping[str, Any],
    ) -> ExperimentGenerationPaths:
        """Create one inactive experiment generation under the rebuild lock.

        A stage is not visible to normal readers.  Only a complete day set,
        full migration verification, and ``finalize_stage_generation`` can
        make it eligible for a later atomic pointer switch.
        """

        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        evidence = self._normalise_generation_source_evidence(source_evidence)
        self.initialize()
        paths = self.experiment_generation_paths(resolved_id)
        with self._day_lock(paths.rebuild_lock_path):
            if paths.generation_root.exists():
                raise CanonicalFactorStoreConflictError(
                    f"experiment generation already exists and is immutable: {resolved_id!r}"
                )
            paths.generation_root.mkdir(parents=True, exist_ok=False)
            manifest = self._new_experiment_generation_manifest(resolved_id, source_evidence=evidence)
            _write_json_atomic(paths.manifest_path, manifest)
            self._assert_experiment_generation_layout(paths.generation_root)
        return paths

    def record_experiment_generation_attestation(
        self,
        generation_id: str,
        attestation: Mapping[str, Any],
    ) -> str:
        """Record immutable full/partial coverage evidence for one stage."""

        payload = self._normalise_migration_attestation(attestation)
        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        paths = self.experiment_generation_paths_for_id(resolved_id)
        with self._day_lock(paths.rebuild_lock_path):
            paths = self._require_staging_experiment_generation(resolved_id)
            with self._day_lock(paths.generation_root / "locks" / "generation_manifest.lock"):
                manifest = self._load_experiment_generation_manifest(paths.generation_id)
                history_raw = manifest.get("migration_attestations", [])
                if not isinstance(history_raw, list):
                    raise CanonicalFactorStoreIntegrityError("experiment generation migration_attestations must be a list")
                history = [dict(item) for item in history_raw if isinstance(item, Mapping)]
                if len(history) != len(history_raw):
                    raise CanonicalFactorStoreIntegrityError("experiment generation has a non-object migration attestation")
                matches = [item for item in history if str(item.get("migration_id", "")) == payload["migration_id"]]
                if matches:
                    if len(matches) != 1 or matches[0] != payload:
                        raise CanonicalFactorStoreConflictError(
                            "experiment generation migration attestation ID already exists with different evidence: "
                            f"{payload['migration_id']}"
                        )
                    return "already_attested"
                history.append(payload)
                manifest["migration_attestations"] = history
                _write_json_atomic(paths.manifest_path, manifest)
                return "attested"

    def mark_experiment_generation_verified(
        self,
        generation_id: str,
        *,
        migration_id: str,
        coverage: Mapping[str, Any],
    ) -> str:
        """Mark a staged generation full-verified while keeping it inactive."""

        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        paths = self.experiment_generation_paths_for_id(resolved_id)
        migration_id = str(migration_id).strip()
        if not migration_id:
            raise CanonicalFactorStoreValidationError("migration_id must be non-empty")
        if not isinstance(coverage, Mapping) or not coverage:
            raise CanonicalFactorStoreValidationError("generation verification coverage must be a non-empty mapping")
        try:
            _canonical_json_bytes(dict(coverage))
        except Exception as exc:
            raise CanonicalFactorStoreValidationError("generation verification coverage must be JSON-serialisable") from exc
        with self._day_lock(paths.rebuild_lock_path):
            paths = self._require_staging_experiment_generation(resolved_id)
            with self._day_lock(paths.generation_root / "locks" / "generation_manifest.lock"):
                manifest = self._load_experiment_generation_manifest(paths.generation_id)
                attestations = manifest.get("migration_attestations", [])
                matched = [item for item in attestations if isinstance(item, Mapping) and item.get("migration_id") == migration_id]
                if len(matched) != 1:
                    raise CanonicalFactorStoreIntegrityError(
                        "cannot mark experiment generation verified without its immutable migration attestation"
                    )
                attestation = matched[0]
                if attestation.get("scope") != "full" or attestation.get("expected_coverage") != dict(coverage):
                    raise CanonicalFactorStoreIntegrityError(
                        "experiment generation full verification must match one full immutable migration attestation"
                    )
                next_status = {
                    "status": "full_verified",
                    "ready_for_consumer": True,
                    "migration_id": migration_id,
                    "coverage": dict(coverage),
                }
                current = manifest.get("migration")
                if current == next_status:
                    return "already_verified"
                if isinstance(current, Mapping) and current.get("status") == "full_verified":
                    raise CanonicalFactorStoreConflictError(
                        "experiment generation is already verified by a different migration scope"
                    )
                manifest["migration"] = next_status
                _write_json_atomic(paths.manifest_path, manifest)
                return "verified"

    def finalize_stage_generation(
        self,
        generation_id: str,
        *,
        expected_days: Sequence[date | datetime],
        verification: Mapping[str, Any],
    ) -> str:
        """Commit ``generation.done`` after full verification of a stage.

        This does *not* alter the active pointer.  It is intentionally a
        separate operation so a complete candidate can be inspected before a
        normal reader can ever select it.
        """

        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        if not isinstance(expected_days, Sequence) or isinstance(expected_days, (str, bytes)):
            raise CanonicalFactorStoreValidationError("generation expected_days must be a non-empty date sequence")
        normalized_days = tuple(_coerce_day(day) for day in expected_days)
        if not normalized_days or tuple(sorted(normalized_days)) != normalized_days or len(set(normalized_days)) != len(normalized_days):
            raise CanonicalFactorStoreValidationError(
                "generation expected_days must be non-empty, sorted ascending, and unique"
            )
        if not isinstance(verification, Mapping) or not verification:
            raise CanonicalFactorStoreValidationError("generation verification must be a non-empty mapping")
        try:
            verification_payload = dict(verification)
            _canonical_json_bytes(verification_payload)
        except Exception as exc:
            raise CanonicalFactorStoreValidationError("generation verification must be JSON-serialisable") from exc
        self.initialize()
        paths = self.experiment_generation_paths(resolved_id)
        with self._day_lock(paths.rebuild_lock_path):
            manifest = self._load_experiment_generation_manifest(resolved_id)
            if manifest.get("lifecycle_status") != "staging":
                raise CanonicalFactorStoreConflictError(
                    f"experiment generation is no longer staging: {resolved_id!r}"
                )
            self._validate_generation_migration_ready(manifest)
            complete, incomplete = self._partition_counts_for_scope(paths.generation_root)
            actual_days = self._completed_days_for_scope(paths.generation_root)
            if complete <= 0 or incomplete or tuple(actual_days) != normalized_days:
                raise CanonicalFactorStoreIntegrityError(
                    "experiment generation cannot finalize without the exact full expected day set: "
                    f"expected={len(normalized_days)} complete={complete} actual={len(actual_days)} incomplete={incomplete}"
                )
            contract_hashes: set[str] = set()
            for day in actual_days:
                _frame, day_manifest = self._read_complete_day(
                    FactorTableKind.EXPERIMENT,
                    day,
                    family=None,
                    expected_contract=None,
                    generation_id=resolved_id,
                )
                contract = day_manifest.get("contract")
                if not isinstance(contract, Mapping):
                    raise CanonicalFactorStoreIntegrityError("experiment generation day lacks contract evidence")
                contract_hashes.add(str(contract.get("contract_list_sha256", "")).lower())
            if len(contract_hashes) != 1:
                raise CanonicalFactorStoreIntegrityError(
                    "experiment generation must have one uniform factor contract across its full expected day set"
                )
            contract_hash = next(iter(contract_hashes))
            done = {
                "schema_version": EXPERIMENT_GENERATION_DONE_SCHEMA,
                "ready": True,
                "table_id": FactorTableKind.EXPERIMENT.value,
                "display_name": FactorTableKind.EXPERIMENT.display_name,
                "generation_id": resolved_id,
                "generation_manifest_path": _relative_to(paths.manifest_path, paths.table_root),
                "generation_manifest_sha256": _sha256_file(paths.manifest_path),
                "migration": dict(manifest["migration"]),
                "completed_day_count": complete,
                "score_days_sha256": _sha256_bytes(
                    _canonical_json_bytes([day.isoformat() for day in normalized_days])
                ),
                "contract_list_sha256": contract_hash,
                "verification": verification_payload,
            }
            if paths.done_path is None:  # defensive: non-legacy is required above.
                raise AssertionError("non-legacy experiment generation has no done path")
            if paths.done_path.exists():
                observed = _read_json(paths.done_path, label="experiment generation done marker")
                if observed != done:
                    raise CanonicalFactorStoreConflictError(
                        "experiment generation is already finalized with different verification evidence"
                    )
                self._validate_experiment_generation_done(paths, manifest=manifest, full_integrity=True)
                return "already_finalized"
            _write_json_atomic(paths.done_path, done)
            self._validate_experiment_generation_done(paths, manifest=manifest)
            return "finalized"

    def activate_generation(self, generation_id: str) -> str:
        """Atomically flip the experiment active-generation pointer.

        Only a separately finalized, full-verified generation may become
        active.  The old generation is not modified, so readers already
        pinned to it can detect the pointer change and fail closed.
        """

        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        self.initialize()
        paths = self.experiment_generation_paths(resolved_id)
        with self._day_lock(paths.rebuild_lock_path):
            generation_manifest = self._load_experiment_generation_manifest(resolved_id)
            self._validate_experiment_generation_done(
                paths,
                manifest=generation_manifest,
                full_integrity=True,
            )
            table_root = self.table_root(FactorTableKind.EXPERIMENT)
            with self._day_lock(table_root / "locks" / "table_manifest.lock"):
                table_manifest = self._load_table_manifest(FactorTableKind.EXPERIMENT)
                current = self._active_experiment_generation_id_from_manifest(table_manifest)
                if current == resolved_id:
                    return "already_active"
                table_manifest["active_generation"] = {
                    "schema_version": EXPERIMENT_GENERATION_POINTER_SCHEMA,
                    "generation_id": resolved_id,
                }
                _write_json_atomic(table_root / "table_manifest.json", table_manifest)
            return "activated"

    def read_table_manifest(self, kind: FactorTableKind | str) -> dict[str, Any]:
        """Read and validate a table-level manifest without opening data days."""

        resolved_kind = coerce_factor_table_kind(kind)
        self._assert_root_layout(require_all_tables=True)
        return self._load_table_manifest(resolved_kind)

    def require_consumer_ready(self, kind: FactorTableKind | str) -> dict[str, Any]:
        """Return a table manifest only when its full migration gate is open.

        A table created by a smoke or interrupted migration is intentionally
        not consumable by configs/models.  This is a separate control from a
        syntactically valid table manifest, because a valid partial table must
        remain auditable without becoming an accidental input source.
        """

        resolved_kind = coerce_factor_table_kind(kind)
        if resolved_kind is FactorTableKind.EXPERIMENT:
            paths = self.experiment_generation_paths()
            manifest = (
                self._load_table_manifest(resolved_kind)
                if paths.is_legacy_flat
                else self._load_experiment_generation_manifest(paths.generation_id)
            )
            if not paths.is_legacy_flat:
                self._validate_experiment_generation_done(paths, manifest=manifest)
        else:
            manifest = self.read_table_manifest(resolved_kind)
        migration = manifest.get("migration")
        if not isinstance(migration, Mapping):
            raise CanonicalFactorStoreIntegrityError(
                f"{resolved_kind.display_name} has no migration readiness object"
            )
        if migration.get("ready_for_consumer") is not True or migration.get("status") != "full_verified":
            raise CanonicalFactorStoreIntegrityError(
                f"{resolved_kind.display_name} is not consumer-ready; "
                f"status={migration.get('status')!r}"
            )
        return manifest

    def initialize(self) -> None:
        """Create the three empty canonical table roots and static manifests.

        This never creates a fourth root and never changes a pre-existing table
        identity.  Contract-list registrations are appended later by
        ``write_day`` under an atomic table manifest update.
        """

        self._assert_root_layout(require_all_tables=False)
        self.root.mkdir(parents=True, exist_ok=True)
        for kind in FactorTableKind:
            table_root = self.table_root(kind)
            table_root.mkdir(parents=True, exist_ok=True)
            path = table_root / "table_manifest.json"
            if path.exists():
                self._load_table_manifest(kind)
                continue
            _write_json_atomic(path, self._new_table_manifest(kind))
        self._assert_root_layout(require_all_tables=True)

    @contextmanager
    def migration_lock(self) -> Iterator[None]:
        """Serialise one bulk migration within approved factor-library metadata.

        The lock is intentionally below an existing table root, so the
        canonical root never gains a fourth child or a parallel migration
        directory.  Per-day locks still guard each artifact bundle.
        """

        self.initialize()
        lock_path = self.table_root(FactorTableKind.FACTOR_LIBRARY) / "locks" / "migration.lock"
        with self._day_lock(lock_path):
            yield

    @contextmanager
    def experiment_rebuild_lock(self) -> Iterator[None]:
        """Serialise lifecycle changes to internal experiment generations.

        It does not lock ordinary legacy/live/library day writes.  The scope
        is exactly stage creation, finalization, and active-pointer movement.
        """

        self.initialize()
        lock_path = self.table_root(FactorTableKind.EXPERIMENT) / "locks" / "experiment_rebuild.lock"
        with self._day_lock(lock_path):
            yield

    def partition_paths(
        self,
        kind: FactorTableKind | str,
        day: date | datetime,
        *,
        family: str | None = None,
        generation_id: str | None = None,
    ) -> FactorDayPaths:
        """Resolve deterministic canonical paths without reading or writing them."""

        resolved_kind = coerce_factor_table_kind(kind)
        resolved_day = _coerce_day(day)
        resolved_family = _normalise_family(resolved_kind, family)
        if generation_id is not None and resolved_kind is not FactorTableKind.EXPERIMENT:
            raise CanonicalFactorStoreValidationError("generation_id is only valid for the experiment factor table")
        if resolved_kind is FactorTableKind.EXPERIMENT:
            generation = (
                self.experiment_generation_paths_for_id(generation_id)
                if generation_id is not None
                else self.experiment_generation_paths()
            )
            table_root = generation.generation_root
        else:
            table_root = self.table_root(resolved_kind)
        scope_root = table_root / resolved_family if resolved_family is not None else table_root
        month = f"{resolved_day:%Y-%m}"
        stem = f"{resolved_day:%Y%m%d}"
        return FactorDayPaths(
            table_root=table_root,
            scope_root=scope_root,
            parquet_path=scope_root / "factors" / PANEL_NAME / month / f"{stem}.parquet",
            manifest_path=scope_root / "manifests" / PANEL_NAME / month / f"{stem}.json",
            done_path=scope_root / "done" / PANEL_NAME / month / f"{stem}.done",
            lock_path=scope_root / "locks" / PANEL_NAME / month / f"{stem}.lock",
        )

    def library_day_paths(self, day: date | datetime) -> FactorLibraryLogicalDayPaths:
        """Resolve the logical all-family publication boundary for one day."""

        resolved_day = _coerce_day(day)
        table_root = self.table_root(FactorTableKind.FACTOR_LIBRARY)
        month = f"{resolved_day:%Y-%m}"
        stem = f"{resolved_day:%Y%m%d}"
        return FactorLibraryLogicalDayPaths(
            table_root=table_root,
            manifest_path=table_root / "day_manifests" / PANEL_NAME / month / f"{stem}.json",
            done_path=table_root / "day_done" / PANEL_NAME / month / f"{stem}.done",
            lock_path=table_root / "locks" / "logical_days" / PANEL_NAME / month / f"{stem}.lock",
        )

    def require_factor_library_day_published(
        self,
        day: date | datetime,
        *,
        families: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validate the logical library-day commit and optional exact families.

        Candidate readers can call this before iterating family partitions.  It
        never returns a half-published day or assumes a family has completed
        merely because a parquet happens to exist.
        """

        resolved_day = _coerce_day(day)
        self._assert_root_layout(require_all_tables=True)
        self._load_table_manifest(FactorTableKind.FACTOR_LIBRARY)
        manifest = self._validate_logical_library_day(resolved_day)
        if families is not None:
            expected = {
                _normalise_family(FactorTableKind.FACTOR_LIBRARY, str(family))
                for family in families
            }
            raw_families = manifest.get("families", [])
            actual = {
                _normalise_family(FactorTableKind.FACTOR_LIBRARY, str(item.get("family", "")))
                for item in raw_families
                if isinstance(item, Mapping)
            }
            if actual != expected:
                raise CanonicalFactorStoreIntegrityError(
                    "factor library logical day family set differs from caller expectation: "
                    f"expected={sorted(expected)!r} actual={sorted(actual)!r}"
                )
        return manifest

    def write_day(
        self,
        kind: FactorTableKind | str,
        day: date | datetime,
        frame: pd.DataFrame,
        *,
        contract: FactorTableContract,
        family: str | None = None,
        source_evidence: Mapping[str, Any] | None = None,
        generation_id: str | None = None,
    ) -> FactorTableWriteResult:
        """Publish a new day or verify an exactly identical idempotent retry.

        Completed partitions are immutable.  A retry with the same values and
        exact contract returns ``status='already_present'``.  Any different
        values, contract, or partial existing bundle fails closed.
        """

        if not isinstance(contract, FactorTableContract):
            raise CanonicalFactorStoreValidationError("contract must be a FactorTableContract")
        resolved_kind = coerce_factor_table_kind(kind)
        resolved_day = _coerce_day(day)
        resolved_family = _normalise_family(resolved_kind, family)
        normalized = _normalise_factor_frame(frame, day=resolved_day, contract=contract)
        if source_evidence is not None:
            if not isinstance(source_evidence, Mapping):
                raise CanonicalFactorStoreValidationError("source_evidence must be a mapping when provided")
            try:
                _canonical_json_bytes(dict(source_evidence))
            except Exception as exc:
                raise CanonicalFactorStoreValidationError("source_evidence must be JSON-serialisable") from exc

        self.initialize()
        if generation_id is not None and resolved_kind is not FactorTableKind.EXPERIMENT:
            raise CanonicalFactorStoreValidationError("generation_id is only valid for the experiment factor table")
        # Existing callers without a generation continue to address the
        # historical flat table.  New rebuild writers must explicitly name a
        # staging generation; they are never implicitly routed to active.
        resolved_generation_id = (
            self._normalise_experiment_generation_id(generation_id, allow_legacy=True)
            if resolved_kind is FactorTableKind.EXPERIMENT and generation_id is not None
            else LEGACY_EXPERIMENT_GENERATION_ID
            if resolved_kind is FactorTableKind.EXPERIMENT
            else None
        )
        if resolved_kind is FactorTableKind.EXPERIMENT and resolved_generation_id != LEGACY_EXPERIMENT_GENERATION_ID:
            self._require_staging_experiment_generation(resolved_generation_id)
        paths = self.partition_paths(
            resolved_kind,
            resolved_day,
            family=resolved_family,
            generation_id=resolved_generation_id,
        )
        with self._day_lock(paths.lock_path):
            logical_library_state = "absent"
            if resolved_kind is FactorTableKind.FACTOR_LIBRARY:
                logical_library_state = self._logical_library_day_state(self.library_day_paths(resolved_day))
                if logical_library_state == "partial":
                    raise CanonicalFactorStoreIntegrityError(
                        "factor library logical day is partial; refusing family write until an explicitly audited repair: "
                        f"{resolved_day.isoformat()}"
                    )
            state = self._partition_state(paths)
            if state == "partial":
                raise CanonicalFactorStoreIntegrityError(
                    "canonical factor day has a partial artifact bundle; refusing overwrite: "
                    f"{paths.parquet_path.parent}"
                )
            if state == "complete":
                existing, manifest = self._read_complete_day(
                    resolved_kind,
                    resolved_day,
                    family=resolved_family,
                    expected_contract=contract,
                    generation_id=resolved_generation_id,
                )
                expected_hash = _frame_hashes(normalized)["frame_sha256"]
                actual_hash = _frame_hashes(existing)["frame_sha256"]
                if expected_hash != actual_hash:
                    raise CanonicalFactorStoreConflictError(
                        "canonical factor day already exists with different values; refusing overwrite: "
                        f"{paths.parquet_path}"
                    )
                if str(manifest.get("contract", {}).get("contract_list_sha256", "")) != contract.sha256:
                    raise CanonicalFactorStoreConflictError(
                        "canonical factor day already exists with a different factor contract list"
                    )
                if source_evidence is not None and manifest.get("source_evidence") != dict(source_evidence):
                    raise CanonicalFactorStoreConflictError(
                        "canonical factor day already exists with different immutable source evidence"
                    )
                if logical_library_state == "complete":
                    self._validate_logical_library_day(
                        resolved_day,
                        expected_family={str(resolved_family): contract},
                    )
                return FactorTableWriteResult(
                    status="already_present",
                    table_kind=resolved_kind,
                    day=resolved_day,
                    family=resolved_family,
                    paths=paths,
                    contract_sha256=contract.sha256,
                    row_count=len(existing),
                )

            if logical_library_state == "complete":
                raise CanonicalFactorStoreConflictError(
                    "factor library logical day is already published; adding/replacing a family is forbidden: "
                    f"{resolved_day.isoformat()}"
                )

            contract_path = self._register_contract(
                resolved_kind,
                contract,
                family=resolved_family,
                generation_id=resolved_generation_id,
            )
            self._write_parquet_atomic(
                paths.parquet_path,
                normalized,
                day=resolved_day,
                contract=contract,
            )
            artifact_hashes = _frame_hashes(normalized)
            manifest = self._build_day_manifest(
                kind=resolved_kind,
                day=resolved_day,
                family=resolved_family,
                paths=paths,
                contract=contract,
                contract_path=contract_path,
                artifact_hashes=artifact_hashes,
                row_count=len(normalized),
                source_evidence=dict(source_evidence) if source_evidence is not None else None,
            )
            _write_json_atomic(paths.manifest_path, manifest)
            done = {
                "schema_version": DAY_DONE_SCHEMA,
                "ready": True,
                "table_id": resolved_kind.value,
                "display_name": resolved_kind.display_name,
                "family": resolved_family,
                "score_day": resolved_day.isoformat(),
                "panel_name": PANEL_NAME,
                "manifest_path": _relative_to(paths.manifest_path, paths.table_root),
                "manifest_sha256": _sha256_file(paths.manifest_path),
                "parquet_path": _relative_to(paths.parquet_path, paths.table_root),
                "parquet_sha256": _sha256_file(paths.parquet_path),
                "contract_list_sha256": contract.sha256,
            }
            _write_json_atomic(paths.done_path, done)
            return FactorTableWriteResult(
                status="written",
                table_kind=resolved_kind,
                day=resolved_day,
                family=resolved_family,
                paths=paths,
                contract_sha256=contract.sha256,
                row_count=len(normalized),
            )

    def publish_factor_library_day(
        self,
        day: date | datetime,
        *,
        family_contracts: Mapping[str, FactorTableContract],
    ) -> str:
        """Publish all completed library family bundles as one logical day.

        Family parquet/manifest/done bundles are deliberately not visible to a
        normal library consumer until this final logical-day ``.done`` marker
        exists.  A failed/interrupted family loop therefore leaves auditable
        partial material without creating a usable all-factor day.
        """

        resolved_day = _coerce_day(day)
        if not isinstance(family_contracts, Mapping) or not family_contracts:
            raise CanonicalFactorStoreValidationError("factor library day requires a non-empty family_contracts mapping")
        normalized_contracts: dict[str, FactorTableContract] = {}
        for raw_family, contract in family_contracts.items():
            family = _normalise_family(FactorTableKind.FACTOR_LIBRARY, str(raw_family))
            if not isinstance(contract, FactorTableContract):
                raise CanonicalFactorStoreValidationError(
                    f"factor library family {family!r} contract must be a FactorTableContract"
                )
            if family in normalized_contracts:
                raise CanonicalFactorStoreValidationError(f"duplicate factor library family contract: {family!r}")
            normalized_contracts[family] = contract

        self.initialize()
        logical_paths = self.library_day_paths(resolved_day)
        with self._day_lock(logical_paths.lock_path):
            state = self._logical_library_day_state(logical_paths)
            if state == "partial":
                raise CanonicalFactorStoreIntegrityError(
                    "factor library logical day has a partial manifest/done bundle; refusing overwrite: "
                    f"{logical_paths.manifest_path.parent}"
                )
            bundles: list[dict[str, Any]] = []
            actual_families = set(self._library_family_partitions_for_day(resolved_day))
            expected_families = set(normalized_contracts)
            if actual_families != expected_families:
                raise CanonicalFactorStoreIntegrityError(
                    "factor library logical-day publish requires exact complete family set: "
                    f"expected={sorted(expected_families)!r} actual={sorted(actual_families)!r}"
                )
            for family, contract in sorted(normalized_contracts.items()):
                paths = self.partition_paths(FactorTableKind.FACTOR_LIBRARY, resolved_day, family=family)
                if self._partition_state(paths) != "complete":
                    raise CanonicalFactorStoreIntegrityError(
                        "factor library family bundle is incomplete before logical-day publish: "
                        f"day={resolved_day.isoformat()} family={family!r}"
                    )
                frame, manifest = self._read_complete_day(
                    FactorTableKind.FACTOR_LIBRARY,
                    resolved_day,
                    family=family,
                    expected_contract=contract,
                )
                bundles.append(
                    {
                        "family": family,
                        "parquet_path": _relative_to(paths.parquet_path, logical_paths.table_root),
                        "parquet_sha256": _sha256_file(paths.parquet_path),
                        "manifest_path": _relative_to(paths.manifest_path, logical_paths.table_root),
                        "manifest_sha256": _sha256_file(paths.manifest_path),
                        "done_path": _relative_to(paths.done_path, logical_paths.table_root),
                        "done_sha256": _sha256_file(paths.done_path),
                        "contract_list_sha256": contract.sha256,
                        "factor_count": len(contract.columns),
                        "row_count": len(frame),
                        "frame_sha256": str(manifest.get("artifact", {}).get("frame_sha256", "")),
                    }
                )
            logical_manifest = {
                "schema_version": LIBRARY_DAY_MANIFEST_SCHEMA,
                "table_id": FactorTableKind.FACTOR_LIBRARY.value,
                "display_name": FactorTableKind.FACTOR_LIBRARY.display_name,
                "score_day": resolved_day.isoformat(),
                "panel": {"name": PANEL_NAME, "logical_time": LOGICAL_PANEL_TIME},
                "family_bundle_count": len(bundles),
                "total_factor_count": sum(int(bundle["factor_count"]) for bundle in bundles),
                "families": bundles,
            }
            if state == "complete":
                observed = _read_json(logical_paths.manifest_path, label="factor library logical day manifest")
                if observed != logical_manifest:
                    raise CanonicalFactorStoreConflictError(
                        "factor library logical day already exists with different family bundle evidence: "
                        f"{logical_paths.manifest_path}"
                    )
                self._validate_logical_library_day(resolved_day, expected_family=normalized_contracts)
                return "already_published"
            _write_json_atomic(logical_paths.manifest_path, logical_manifest)
            done = {
                "schema_version": LIBRARY_DAY_DONE_SCHEMA,
                "ready": True,
                "table_id": FactorTableKind.FACTOR_LIBRARY.value,
                "display_name": FactorTableKind.FACTOR_LIBRARY.display_name,
                "score_day": resolved_day.isoformat(),
                "panel_name": PANEL_NAME,
                "manifest_path": _relative_to(logical_paths.manifest_path, logical_paths.table_root),
                "manifest_sha256": _sha256_file(logical_paths.manifest_path),
                "family_bundle_count": len(bundles),
                "total_factor_count": sum(int(bundle["factor_count"]) for bundle in bundles),
            }
            _write_json_atomic(logical_paths.done_path, done)
            self._validate_logical_library_day(resolved_day, expected_family=normalized_contracts)
            return "published"

    def mark_migration_verified(
        self,
        kind: FactorTableKind | str,
        *,
        migration_id: str,
        coverage: Mapping[str, Any],
        ready_for_consumer: bool = True,
    ) -> str:
        """Open the consumer gate only after a full verified migration scope."""

        resolved_kind = coerce_factor_table_kind(kind)
        migration_id = str(migration_id).strip()
        if not migration_id:
            raise CanonicalFactorStoreValidationError("migration_id must be non-empty")
        if not isinstance(coverage, Mapping) or not coverage:
            raise CanonicalFactorStoreValidationError("migration verification coverage must be a non-empty mapping")
        try:
            _canonical_json_bytes(dict(coverage))
        except Exception as exc:
            raise CanonicalFactorStoreValidationError("migration coverage must be JSON-serialisable") from exc
        self.initialize()
        table_root = self.table_root(resolved_kind)
        with self._day_lock(table_root / "locks" / "table_manifest.lock"):
            manifest = self._load_table_manifest(resolved_kind)
            attestations = manifest.get("migration_attestations", [])
            matched = [item for item in attestations if isinstance(item, Mapping) and item.get("migration_id") == migration_id]
            if len(matched) != 1:
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} verified without its immutable migration attestation"
                )
            attestation = matched[0]
            if attestation.get("scope") != "full":
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} full_verified from a non-full migration attestation"
                )
            if attestation.get("expected_coverage") != dict(coverage):
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} full_verified with coverage different from its attestation"
                )
            if ready_for_consumer is not True:
                raise CanonicalFactorStoreValidationError("full_verified migration must set ready_for_consumer=true")
            next_status = {
                "status": "full_verified",
                "ready_for_consumer": True,
                "migration_id": migration_id,
                "coverage": dict(coverage),
            }
            current = manifest.get("migration")
            if current == next_status:
                return "already_verified"
            if isinstance(current, Mapping) and current.get("status") == "full_verified":
                raise CanonicalFactorStoreConflictError(
                    f"{resolved_kind.display_name} is already verified by a different migration scope"
                )
            manifest["migration"] = next_status
            _write_json_atomic(table_root / "table_manifest.json", manifest)
            return "verified"

    def mark_migration_partial(
        self,
        kind: FactorTableKind | str,
        *,
        migration_id: str,
        coverage: Mapping[str, Any],
    ) -> str:
        """Record a verified partial/smoke scope while retaining a closed consumer gate."""

        resolved_kind = coerce_factor_table_kind(kind)
        migration_id = str(migration_id).strip()
        if not migration_id:
            raise CanonicalFactorStoreValidationError("migration_id must be non-empty")
        if not isinstance(coverage, Mapping) or not coverage:
            raise CanonicalFactorStoreValidationError("migration partial coverage must be a non-empty mapping")
        try:
            _canonical_json_bytes(dict(coverage))
        except Exception as exc:
            raise CanonicalFactorStoreValidationError("migration partial coverage must be JSON-serialisable") from exc
        self.initialize()
        table_root = self.table_root(resolved_kind)
        with self._day_lock(table_root / "locks" / "table_manifest.lock"):
            manifest = self._load_table_manifest(resolved_kind)
            attestations = manifest.get("migration_attestations", [])
            matched = [item for item in attestations if isinstance(item, Mapping) and item.get("migration_id") == migration_id]
            if len(matched) != 1:
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} partial without its immutable migration attestation"
                )
            attestation = matched[0]
            if attestation.get("scope") != "partial":
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} partial from a non-partial migration attestation"
                )
            if attestation.get("expected_coverage") != dict(coverage):
                raise CanonicalFactorStoreIntegrityError(
                    f"cannot mark {resolved_kind.display_name} partial with coverage different from its attestation"
                )
            current = manifest.get("migration")
            if isinstance(current, Mapping) and current.get("status") == "full_verified":
                raise CanonicalFactorStoreConflictError(
                    f"{resolved_kind.display_name} is already full_verified and may not be downgraded"
                )
            next_status = {
                "status": "partial_verified",
                "ready_for_consumer": False,
                "migration_id": migration_id,
                "coverage": dict(coverage),
            }
            if current == next_status:
                return "already_marked_partial"
            manifest["migration"] = next_status
            _write_json_atomic(table_root / "table_manifest.json", manifest)
            return "marked_partial"

    def record_migration_attestation(
        self,
        kind: FactorTableKind | str,
        attestation: Mapping[str, Any],
    ) -> str:
        """Atomically record one immutable source-migration attestation.

        The evidence lives in the target table's existing metadata manifest,
        rather than creating a fourth root or a parallel migration-report
        directory.  Retries with byte-identical evidence are idempotent;
        reusing an ID with different evidence fails closed.
        """

        if not isinstance(attestation, Mapping):
            raise CanonicalFactorStoreValidationError("migration attestation must be a mapping")
        payload = dict(attestation)
        migration_id = str(payload.get("migration_id", "")).strip()
        if not migration_id:
            raise CanonicalFactorStoreValidationError("migration attestation requires a non-empty migration_id")
        scope = str(payload.get("scope", "")).strip().lower()
        if scope not in {"full", "partial"}:
            raise CanonicalFactorStoreValidationError("migration attestation scope must be exactly 'full' or 'partial'")
        expected_coverage = payload.get("expected_coverage")
        if not isinstance(expected_coverage, Mapping) or not expected_coverage:
            raise CanonicalFactorStoreValidationError(
                "migration attestation requires a non-empty expected_coverage mapping"
            )
        payload["migration_id"] = migration_id
        payload["scope"] = scope
        payload["expected_coverage"] = dict(expected_coverage)
        payload.pop("attestation_sha256", None)
        try:
            digest = _sha256_bytes(_canonical_json_bytes(payload))
        except Exception as exc:
            raise CanonicalFactorStoreValidationError(
                "migration attestation must be JSON-serialisable and must not contain NaN"
            ) from exc
        payload["attestation_sha256"] = digest

        resolved_kind = coerce_factor_table_kind(kind)
        self.initialize()
        table_root = self.table_root(resolved_kind)
        with self._day_lock(table_root / "locks" / "table_manifest.lock"):
            manifest = self._load_table_manifest(resolved_kind)
            history_raw = manifest.get("migration_attestations", [])
            if not isinstance(history_raw, list):
                raise CanonicalFactorStoreIntegrityError(
                    f"{resolved_kind.display_name} migration_attestations must be a list"
                )
            history = [dict(item) for item in history_raw if isinstance(item, Mapping)]
            if len(history) != len(history_raw):
                raise CanonicalFactorStoreIntegrityError(
                    f"{resolved_kind.display_name} has a non-object migration attestation"
                )
            matches = [item for item in history if str(item.get("migration_id", "")) == migration_id]
            if matches:
                if len(matches) != 1 or matches[0] != payload:
                    raise CanonicalFactorStoreConflictError(
                        "migration attestation ID already exists with different evidence: " f"{migration_id}"
                    )
                return "already_attested"
            history.append(payload)
            manifest["migration_attestations"] = history
            _write_json_atomic(table_root / "table_manifest.json", manifest)
            return "attested"

    def read_day(
        self,
        kind: FactorTableKind | str,
        day: date | datetime,
        *,
        family: str | None = None,
        expected_contract: FactorTableContract | None = None,
        generation_id: str | None = None,
    ) -> pd.DataFrame:
        """Read a complete, integrity-checked canonical day.

        An absent, partial, or corrupt day raises.  This is intentionally
        stricter than the legacy ``FactorStore`` reader: canonical consumers
        must not treat missing production/research factor input as an empty
        valid cross-section.
        """

        resolved_kind = coerce_factor_table_kind(kind)
        resolved_day = _coerce_day(day)
        resolved_family = _normalise_family(resolved_kind, family)
        self._assert_root_layout(require_all_tables=True)
        if generation_id is not None and resolved_kind is not FactorTableKind.EXPERIMENT:
            raise CanonicalFactorStoreValidationError("generation_id is only valid for the experiment factor table")
        resolved_generation_id = (
            self.experiment_generation_paths(generation_id).generation_id
            if resolved_kind is FactorTableKind.EXPERIMENT
            else None
        )
        if resolved_kind is FactorTableKind.EXPERIMENT and resolved_generation_id != LEGACY_EXPERIMENT_GENERATION_ID:
            paths_for_generation = self.experiment_generation_paths(resolved_generation_id)
            self._load_experiment_generation_manifest(resolved_generation_id)
            self._validate_experiment_generation_done(paths_for_generation)
        else:
            self._load_table_manifest(resolved_kind)
        if resolved_kind is FactorTableKind.FACTOR_LIBRARY:
            self._validate_logical_library_day(resolved_day, required_family=resolved_family)
        paths = self.partition_paths(
            resolved_kind,
            resolved_day,
            family=resolved_family,
            generation_id=resolved_generation_id,
        )
        state = self._partition_state(paths)
        if state == "absent":
            raise FileNotFoundError(f"canonical factor day is absent: {paths.parquet_path}")
        if state != "complete":
            raise CanonicalFactorStoreIntegrityError(
                "canonical factor day has a partial artifact bundle: " f"{paths.parquet_path.parent}"
            )
        output, _manifest = self._read_complete_day(
            resolved_kind,
            resolved_day,
            family=resolved_family,
            expected_contract=expected_contract,
            generation_id=resolved_generation_id,
        )
        return output

    def read_staging_experiment_day(
        self,
        generation_id: str,
        day: date | datetime,
        *,
        expected_contract: FactorTableContract | None = None,
    ) -> pd.DataFrame:
        """Read a complete staging day for the designated generation writer.

        This narrow API deliberately does not make a staging generation a
        normal consumer input.  It exists only so a writer can perform the
        legacy pipeline's idempotent ``read-before-write`` check while the
        generation remains inactive and lacks ``generation.done``.
        """

        paths = self._require_staging_experiment_generation(generation_id)
        resolved_day = _coerce_day(day)
        day_paths = self.partition_paths(
            FactorTableKind.EXPERIMENT,
            resolved_day,
            generation_id=paths.generation_id,
        )
        if self._partition_state(day_paths) == "absent":
            raise FileNotFoundError(f"canonical factor day is absent: {day_paths.parquet_path}")
        if self._partition_state(day_paths) != "complete":
            raise CanonicalFactorStoreIntegrityError(
                "canonical factor day has a partial artifact bundle: " f"{day_paths.parquet_path.parent}"
            )
        output, _manifest = self._read_complete_day(
            FactorTableKind.EXPERIMENT,
            resolved_day,
            family=None,
            expected_contract=expected_contract,
            generation_id=paths.generation_id,
        )
        return output

    def report(self) -> CanonicalFactorStoreReport:
        """Return a read-only inventory of canonical roots and committed days."""

        unexpected = tuple(self._unexpected_root_entries())
        table_reports: list[CanonicalFactorTableReport] = []
        for kind in FactorTableKind:
            table_root = self.table_root(kind)
            if not table_root.is_dir():
                table_reports.append(
                    CanonicalFactorTableReport(
                        table_id=kind.value,
                        display_name=kind.display_name,
                        root=str(table_root),
                        exists=False,
                        manifest_valid=False,
                        family_aware=kind.family_aware,
                        families=(),
                        registered_contract_count=0,
                        completed_day_count=0,
                        incomplete_day_count=0,
                    )
                )
                continue
            try:
                manifest = self._load_table_manifest(kind)
                active_paths: ExperimentGenerationPaths | None = None
                if kind is FactorTableKind.EXPERIMENT:
                    active_paths = self.experiment_generation_paths()
                    if not active_paths.is_legacy_flat:
                        manifest = self._load_experiment_generation_manifest(active_paths.generation_id)
                        self._validate_experiment_generation_done(active_paths, manifest=manifest)
                contracts = manifest.get("registered_contracts", [])
                if not isinstance(contracts, list):
                    raise CanonicalFactorStoreIntegrityError("registered_contracts is not a list")
                families = self._family_names(table_root) if kind.family_aware else ()
                complete, incomplete = (
                    self._partition_counts_for_scope(active_paths.generation_root)
                    if active_paths is not None and not active_paths.is_legacy_flat
                    else self._partition_counts(kind, table_root, families)
                )
                table_reports.append(
                    CanonicalFactorTableReport(
                        table_id=kind.value,
                        display_name=kind.display_name,
                        root=str(table_root),
                        exists=True,
                        manifest_valid=True,
                        family_aware=kind.family_aware,
                        families=tuple(families),
                        registered_contract_count=len(contracts),
                        completed_day_count=complete,
                        incomplete_day_count=incomplete,
                    )
                )
            except CanonicalFactorStoreError as exc:
                table_reports.append(
                    CanonicalFactorTableReport(
                        table_id=kind.value,
                        display_name=kind.display_name,
                        root=str(table_root),
                        exists=True,
                        manifest_valid=False,
                        family_aware=kind.family_aware,
                        families=(),
                        registered_contract_count=0,
                        completed_day_count=0,
                        incomplete_day_count=0,
                        error=str(exc),
                    )
                )
        return CanonicalFactorStoreReport(
            root=str(self.root),
            allowed_table_ids=_TABLE_IDS,
            unexpected_root_entries=unexpected,
            tables=tuple(table_reports),
        )

    def _unexpected_root_entries(self) -> list[str]:
        if not self.root.exists():
            return []
        if not self.root.is_dir():
            return [self.root.name or str(self.root)]
        return sorted(path.name for path in self.root.iterdir() if path.name not in _TABLE_IDS)

    def _assert_root_layout(self, *, require_all_tables: bool) -> None:
        if self.root.exists() and not self.root.is_dir():
            raise CanonicalFactorStoreIntegrityError(f"canonical factor-store root is not a directory: {self.root}")
        unexpected = self._unexpected_root_entries()
        if unexpected:
            raise CanonicalFactorStoreIntegrityError(
                "canonical factor-store root may contain only three table IDs "
                f"{list(_TABLE_IDS)!r}; unexpected entries={unexpected!r}"
            )
        non_directories = [
            kind.value
            for kind in FactorTableKind
            if self.table_root(kind).exists() and not self.table_root(kind).is_dir()
        ]
        if non_directories:
            raise CanonicalFactorStoreIntegrityError(
                "canonical factor-store table roots must be directories: " + ", ".join(non_directories)
            )
        if require_all_tables:
            missing = [kind.value for kind in FactorTableKind if not self.table_root(kind).is_dir()]
            if missing:
                raise CanonicalFactorStoreIntegrityError(
                    "canonical factor-store root is missing required table roots: " + ", ".join(missing)
                )

    def _new_table_manifest(self, kind: FactorTableKind) -> dict[str, Any]:
        manifest = {
            "schema_version": TABLE_MANIFEST_SCHEMA,
            "table_id": kind.value,
            "display_name": kind.display_name,
            "panel": {"name": PANEL_NAME, "logical_time": LOGICAL_PANEL_TIME},
            "family_aware": kind.family_aware,
            "layout": {
                "factor_library_scope": "<family>" if kind.family_aware else None,
                "parquet": "[<family>/]factors/T1430/YYYY-MM/YYYYMMDD.parquet",
                "manifest": "[<family>/]manifests/T1430/YYYY-MM/YYYYMMDD.json",
                "done": "[<family>/]done/T1430/YYYY-MM/YYYYMMDD.done",
                "contract": "[<family>/]contracts/<contract_list_sha256>.json",
            },
            "registered_contracts": [],
            "migration_attestations": [],
            "migration": {
                "status": "not_started",
                "ready_for_consumer": False,
                "migration_id": None,
                "coverage": {},
            },
        }
        if kind is FactorTableKind.EXPERIMENT:
            # Existing flat tables without this field are interpreted as the
            # immutable legacy generation.  New tables make that pointer
            # explicit from their first initialization onward.
            manifest["active_generation"] = {
                "schema_version": EXPERIMENT_GENERATION_POINTER_SCHEMA,
                "generation_id": LEGACY_EXPERIMENT_GENERATION_ID,
            }
        return manifest

    @staticmethod
    def _normalise_experiment_generation_id(value: str | object, *, allow_legacy: bool) -> str:
        generation_id = str(value).strip()
        if not _GENERATION_ID_PATTERN.fullmatch(generation_id):
            raise CanonicalFactorStoreValidationError(
                "experiment generation_id must match "
                "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
            )
        if generation_id == LEGACY_EXPERIMENT_GENERATION_ID and not allow_legacy:
            raise CanonicalFactorStoreValidationError(
                f"{LEGACY_EXPERIMENT_GENERATION_ID!r} is reserved for the pre-generation flat experiment table"
            )
        return generation_id

    @staticmethod
    def _normalise_generation_source_evidence(source_evidence: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(source_evidence, Mapping) or not source_evidence:
            raise CanonicalFactorStoreValidationError(
                "experiment generation source_evidence must be a non-empty mapping"
            )
        payload = dict(source_evidence)
        try:
            _canonical_json_bytes(payload)
        except Exception as exc:
            raise CanonicalFactorStoreValidationError(
                "experiment generation source_evidence must be JSON-serialisable"
            ) from exc
        return payload

    def _active_experiment_generation_id_from_manifest(self, manifest: Mapping[str, Any]) -> str:
        """Validate and return the one atomic experiment generation pointer."""

        raw = manifest.get("active_generation")
        # ``active_generation`` is intentionally optional for old table
        # manifests.  Treating its absence as legacy makes old parquet
        # readable without a metadata rewrite before the first rebuild.
        if raw is None:
            return LEGACY_EXPERIMENT_GENERATION_ID
        if isinstance(raw, str):
            # Accept the first generation implementation's scalar pointer as
            # an on-disk compatibility form; all new writes use the schema-
            # tagged object below.
            generation_id = self._normalise_experiment_generation_id(raw, allow_legacy=True)
        elif isinstance(raw, Mapping):
            if raw.get("schema_version") != EXPERIMENT_GENERATION_POINTER_SCHEMA:
                raise CanonicalFactorStoreIntegrityError("experiment active_generation pointer has an unexpected schema")
            if set(raw) != {"schema_version", "generation_id"}:
                raise CanonicalFactorStoreIntegrityError("experiment active_generation pointer has unsupported fields")
            generation_id = self._normalise_experiment_generation_id(raw.get("generation_id", ""), allow_legacy=True)
        else:
            raise CanonicalFactorStoreIntegrityError("experiment active_generation pointer must be an object")
        if generation_id == LEGACY_EXPERIMENT_GENERATION_ID:
            return generation_id
        paths = self.experiment_generation_paths_for_id(generation_id)
        self._load_experiment_generation_manifest(generation_id)
        self._validate_experiment_generation_done(paths)
        return generation_id

    def experiment_generation_paths_for_id(self, generation_id: str) -> ExperimentGenerationPaths:
        """Build bounded generation paths without recursively resolving active."""

        resolved_id = self._normalise_experiment_generation_id(generation_id, allow_legacy=True)
        table_root = self.table_root(FactorTableKind.EXPERIMENT)
        rebuild_lock = table_root / "locks" / "experiment_rebuild.lock"
        if resolved_id == LEGACY_EXPERIMENT_GENERATION_ID:
            return ExperimentGenerationPaths(
                table_root=table_root,
                generation_id=resolved_id,
                generation_root=table_root,
                manifest_path=table_root / "table_manifest.json",
                done_path=None,
                rebuild_lock_path=rebuild_lock,
            )
        generation_root = table_root / "generations" / resolved_id
        return ExperimentGenerationPaths(
            table_root=table_root,
            generation_id=resolved_id,
            generation_root=generation_root,
            manifest_path=generation_root / "generation_manifest.json",
            done_path=generation_root / "generation.done",
            rebuild_lock_path=rebuild_lock,
        )

    def _new_experiment_generation_manifest(
        self,
        generation_id: str,
        *,
        source_evidence: Mapping[str, Any],
    ) -> dict[str, Any]:
        static = self._new_table_manifest(FactorTableKind.EXPERIMENT)
        return {
            "schema_version": EXPERIMENT_GENERATION_SCHEMA,
            "table_id": FactorTableKind.EXPERIMENT.value,
            "display_name": FactorTableKind.EXPERIMENT.display_name,
            "generation_id": generation_id,
            "lifecycle_status": "staging",
            "panel": static["panel"],
            "family_aware": False,
            "layout": static["layout"],
            "registered_contracts": [],
            "migration_attestations": [],
            "migration": {
                "status": "not_started",
                "ready_for_consumer": False,
                "migration_id": None,
                "coverage": {},
            },
            "source_evidence": dict(source_evidence),
        }

    def _load_experiment_generation_manifest(self, generation_id: str) -> dict[str, Any]:
        paths = self.experiment_generation_paths_for_id(generation_id)
        if paths.is_legacy_flat:
            return self._load_table_manifest(FactorTableKind.EXPERIMENT)
        if not paths.generation_root.is_dir():
            raise FileNotFoundError(f"experiment generation is absent: {paths.generation_root}")
        self._assert_experiment_generation_layout(paths.generation_root)
        manifest = _read_json(paths.manifest_path, label="experiment generation manifest")
        expected = self._new_experiment_generation_manifest(
            paths.generation_id,
            source_evidence=self._normalise_generation_source_evidence(manifest.get("source_evidence", {})),
        )
        for key in (
            "schema_version",
            "table_id",
            "display_name",
            "generation_id",
            "panel",
            "family_aware",
            "layout",
        ):
            if manifest.get(key) != expected[key]:
                raise CanonicalFactorStoreIntegrityError(
                    f"experiment generation manifest has an incompatible {key!r}: {paths.manifest_path}"
                )
        if manifest.get("lifecycle_status") not in {"staging"}:
            raise CanonicalFactorStoreIntegrityError(
                f"experiment generation has an invalid lifecycle_status: {manifest.get('lifecycle_status')!r}"
            )
        contracts = manifest.get("registered_contracts")
        if not isinstance(contracts, list):
            raise CanonicalFactorStoreIntegrityError("experiment generation registered_contracts must be a list")
        for entry in contracts:
            self._validate_registered_contract_entry(
                FactorTableKind.EXPERIMENT,
                entry,
                table_root=paths.generation_root,
                generation_id=paths.generation_id,
            )
        attestations = manifest.get("migration_attestations", [])
        if not isinstance(attestations, list):
            raise CanonicalFactorStoreIntegrityError("experiment generation migration_attestations must be a list")
        attestation_ids: list[str] = []
        for entry in attestations:
            self._validate_migration_attestation_entry(FactorTableKind.EXPERIMENT, entry)
            assert isinstance(entry, Mapping)
            attestation_ids.append(str(entry.get("migration_id", "")))
        if len(attestation_ids) != len(set(attestation_ids)):
            raise CanonicalFactorStoreIntegrityError("experiment generation has duplicate migration attestation IDs")
        self._validate_migration_status(FactorTableKind.EXPERIMENT, manifest.get("migration"))
        return manifest

    def _require_staging_experiment_generation(self, generation_id: str) -> ExperimentGenerationPaths:
        paths = self.experiment_generation_paths_for_id(
            self._normalise_experiment_generation_id(generation_id, allow_legacy=False)
        )
        manifest = self._load_experiment_generation_manifest(paths.generation_id)
        if manifest.get("lifecycle_status") != "staging":  # defensive; validator enforces the same state.
            raise CanonicalFactorStoreConflictError(
                f"experiment generation is not writable staging: {paths.generation_id!r}"
            )
        if paths.done_path is not None and paths.done_path.exists():
            raise CanonicalFactorStoreConflictError(
                f"experiment generation is finalized and immutable: {paths.generation_id!r}"
            )
        return paths

    def _validate_generation_migration_ready(self, manifest: Mapping[str, Any]) -> None:
        migration = manifest.get("migration")
        if not isinstance(migration, Mapping):
            raise CanonicalFactorStoreIntegrityError("experiment generation has no migration status")
        if migration.get("status") != "full_verified" or migration.get("ready_for_consumer") is not True:
            raise CanonicalFactorStoreIntegrityError(
                "experiment generation must be full_verified before a generation.done marker can be committed"
            )
        migration_id = migration.get("migration_id")
        coverage = migration.get("coverage")
        if not isinstance(migration_id, str) or not migration_id.strip() or not isinstance(coverage, Mapping) or not coverage:
            raise CanonicalFactorStoreIntegrityError("experiment generation full verification lacks migration_id/coverage")

    def _validate_experiment_generation_done(
        self,
        paths: ExperimentGenerationPaths,
        *,
        manifest: Mapping[str, Any] | None = None,
        full_integrity: bool = False,
    ) -> dict[str, Any]:
        if paths.is_legacy_flat or paths.done_path is None:
            raise CanonicalFactorStoreValidationError("legacy_flat has no experiment generation.done marker")
        generation_manifest = dict(manifest) if manifest is not None else self._load_experiment_generation_manifest(paths.generation_id)
        self._validate_generation_migration_ready(generation_manifest)
        if not paths.done_path.is_file():
            raise CanonicalFactorStoreIntegrityError(
                f"experiment generation is not finalized: {paths.done_path}"
            )
        done = _read_json(paths.done_path, label="experiment generation done marker")
        expected_identity = {
            "table_id": FactorTableKind.EXPERIMENT.value,
            "display_name": FactorTableKind.EXPERIMENT.display_name,
            "generation_id": paths.generation_id,
        }
        if done.get("schema_version") != EXPERIMENT_GENERATION_DONE_SCHEMA or done.get("ready") is not True:
            raise CanonicalFactorStoreIntegrityError("experiment generation done marker is not ready")
        for key, value in expected_identity.items():
            if done.get(key) != value:
                raise CanonicalFactorStoreIntegrityError(
                    f"experiment generation done marker {key!r} mismatch"
                )
        if done.get("generation_manifest_path") != _relative_to(paths.manifest_path, paths.table_root):
            raise CanonicalFactorStoreIntegrityError("experiment generation done marker manifest path mismatch")
        if done.get("generation_manifest_sha256") != _sha256_file(paths.manifest_path):
            raise CanonicalFactorStoreIntegrityError("experiment generation manifest hash differs from done marker")
        if done.get("migration") != dict(generation_manifest["migration"]):
            raise CanonicalFactorStoreIntegrityError("experiment generation done marker migration differs from manifest")
        if not isinstance(done.get("verification"), Mapping) or not dict(done["verification"]):
            raise CanonicalFactorStoreIntegrityError("experiment generation done marker lacks verification evidence")
        if int(done.get("completed_day_count", 0)) <= 0:
            raise CanonicalFactorStoreIntegrityError("experiment generation done marker has no completed factor days")
        if full_integrity:
            complete, incomplete = self._partition_counts_for_scope(paths.generation_root)
            actual_days = self._completed_days_for_scope(paths.generation_root)
            expected_days_sha256 = _sha256_bytes(_canonical_json_bytes([day.isoformat() for day in actual_days]))
            if incomplete or complete != int(done["completed_day_count"]):
                raise CanonicalFactorStoreIntegrityError(
                    "experiment generation day coverage differs from its finalized marker"
                )
            if done.get("score_days_sha256") != expected_days_sha256:
                raise CanonicalFactorStoreIntegrityError(
                    "experiment generation score-day sequence differs from its finalized marker"
                )
            contract_hashes: set[str] = set()
            for day in actual_days:
                _frame, day_manifest = self._read_complete_day(
                    FactorTableKind.EXPERIMENT,
                    day,
                    family=None,
                    expected_contract=None,
                    generation_id=paths.generation_id,
                )
                contract = day_manifest.get("contract")
                if not isinstance(contract, Mapping):
                    raise CanonicalFactorStoreIntegrityError("experiment generation day lacks contract evidence")
                contract_hashes.add(str(contract.get("contract_list_sha256", "")).lower())
            if len(contract_hashes) != 1 or done.get("contract_list_sha256") not in contract_hashes:
                raise CanonicalFactorStoreIntegrityError(
                    "experiment generation factor contract differs from its finalized marker"
                )
        return done

    def _normalise_migration_attestation(self, attestation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(attestation, Mapping):
            raise CanonicalFactorStoreValidationError("migration attestation must be a mapping")
        payload = dict(attestation)
        migration_id = str(payload.get("migration_id", "")).strip()
        if not migration_id:
            raise CanonicalFactorStoreValidationError("migration attestation requires a non-empty migration_id")
        scope = str(payload.get("scope", "")).strip().lower()
        if scope not in {"full", "partial"}:
            raise CanonicalFactorStoreValidationError("migration attestation scope must be exactly 'full' or 'partial'")
        expected_coverage = payload.get("expected_coverage")
        if not isinstance(expected_coverage, Mapping) or not expected_coverage:
            raise CanonicalFactorStoreValidationError(
                "migration attestation requires a non-empty expected_coverage mapping"
            )
        payload["migration_id"] = migration_id
        payload["scope"] = scope
        payload["expected_coverage"] = dict(expected_coverage)
        payload.pop("attestation_sha256", None)
        try:
            digest = _sha256_bytes(_canonical_json_bytes(payload))
        except Exception as exc:
            raise CanonicalFactorStoreValidationError(
                "migration attestation must be JSON-serialisable and must not contain NaN"
            ) from exc
        payload["attestation_sha256"] = digest
        return payload

    def _load_table_manifest(self, kind: FactorTableKind) -> dict[str, Any]:
        self._assert_table_layout(kind)
        path = self.table_root(kind) / "table_manifest.json"
        manifest = _read_json(path, label=f"{kind.display_name} table manifest")
        static = self._new_table_manifest(kind)
        for key in ("schema_version", "table_id", "display_name", "panel", "family_aware", "layout"):
            if manifest.get(key) != static[key]:
                raise CanonicalFactorStoreIntegrityError(
                    f"{kind.display_name} table manifest has an incompatible {key!r}: {path}"
                )
        if kind is FactorTableKind.EXPERIMENT:
            self._active_experiment_generation_id_from_manifest(manifest)
        elif "active_generation" in manifest:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} table manifest may not declare an experiment active_generation pointer"
            )
        contracts = manifest.get("registered_contracts")
        if not isinstance(contracts, list):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} table manifest registered_contracts must be a list"
            )
        for entry in contracts:
            self._validate_registered_contract_entry(kind, entry)
        attestations = manifest.get("migration_attestations", [])
        if not isinstance(attestations, list):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} table manifest migration_attestations must be a list"
            )
        attestation_ids: list[str] = []
        for entry in attestations:
            self._validate_migration_attestation_entry(kind, entry)
            assert isinstance(entry, Mapping)  # narrowed by the validator above.
            attestation_ids.append(str(entry.get("migration_id", "")))
        if len(attestation_ids) != len(set(attestation_ids)):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} table manifest has duplicate migration attestation IDs"
            )
        self._validate_migration_status(kind, manifest.get("migration"))
        return manifest

    def _validate_migration_status(self, kind: FactorTableKind, value: object) -> None:
        if not isinstance(value, Mapping):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} table manifest lacks migration status")
        status = value.get("status")
        ready = value.get("ready_for_consumer")
        migration_id = value.get("migration_id")
        coverage = value.get("coverage")
        if status not in {"not_started", "partial_verified", "full_verified"}:
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} has an invalid migration status")
        if not isinstance(ready, bool) or not isinstance(coverage, Mapping):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} migration status has invalid readiness/coverage")
        if status == "not_started":
            if ready or migration_id is not None or dict(coverage):
                raise CanonicalFactorStoreIntegrityError(
                    f"{kind.display_name} not_started migration status must have a closed empty gate"
                )
            return
        if not isinstance(migration_id, str) or not migration_id.strip() or not dict(coverage):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} verified migration status requires migration_id and coverage"
            )
        if status == "partial_verified" and ready:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} partial migration must not be consumer-ready"
            )
        if status == "full_verified" and not ready:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} full_verified migration must be consumer-ready"
            )

    def _validate_migration_attestation_entry(self, kind: FactorTableKind, entry: object) -> None:
        if not isinstance(entry, Mapping):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} has a non-object migration attestation")
        migration_id = str(entry.get("migration_id", "")).strip()
        if not migration_id:
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} migration attestation lacks migration_id")
        if entry.get("scope") not in {"full", "partial"}:
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} migration attestation has invalid scope")
        if not isinstance(entry.get("expected_coverage"), Mapping) or not dict(entry["expected_coverage"]):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} migration attestation lacks expected_coverage"
            )
        observed = str(entry.get("attestation_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", observed):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} migration attestation has an invalid attestation_sha256"
            )
        payload = dict(entry)
        payload.pop("attestation_sha256", None)
        if _sha256_bytes(_canonical_json_bytes(payload)) != observed:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} migration attestation hash differs from its content"
            )

    def _assert_table_layout(self, kind: FactorTableKind) -> None:
        """Reject parallel data roots inside a canonical table before use.

        The top-level root is limited to the three table IDs.  This second
        check keeps an old/temporary factor store from hiding below a valid
        table directory where a normal consumer might accidentally discover
        it later.
        """

        table_root = self.table_root(kind)
        if not table_root.is_dir():
            raise CanonicalFactorStoreIntegrityError(f"canonical table root is not a directory: {table_root}")
        allowed_files = {"table_manifest.json"}
        for path in table_root.iterdir():
            if path.is_file():
                if path.name not in allowed_files:
                    raise CanonicalFactorStoreIntegrityError(
                        f"{kind.display_name} has an unapproved top-level file: {path.name!r}"
                    )
                continue
            if not path.is_dir():
                raise CanonicalFactorStoreIntegrityError(
                    f"{kind.display_name} has an unsupported top-level entry: {path.name!r}"
                )
            if kind.family_aware:
                if path.name in _FACTOR_LIBRARY_TABLE_METADATA_DIRS:
                    continue
                if not _FAMILY_PATTERN.fullmatch(path.name):
                    raise CanonicalFactorStoreIntegrityError(
                        f"因子库内因子表 has an invalid family directory: {path.name!r}"
                    )
                self._assert_scope_layout(path, label=f"因子库内因子表 family {path.name!r}")
            elif path.name == "generations" and kind is FactorTableKind.EXPERIMENT:
                self._assert_experiment_generations_layout(path)
            elif path.name not in _WIDE_TABLE_METADATA_DIRS:
                raise CanonicalFactorStoreIntegrityError(
                    f"{kind.display_name} has an unapproved top-level directory: {path.name!r}"
                )

    @staticmethod
    def _assert_scope_layout(scope_root: Path, *, label: str) -> None:
        for path in scope_root.iterdir():
            if path.is_dir() and path.name in _WIDE_TABLE_METADATA_DIRS:
                continue
            raise CanonicalFactorStoreIntegrityError(f"{label} has an unapproved entry: {path.name!r}")

    def _assert_experiment_generations_layout(self, generations_root: Path) -> None:
        """Validate the one approved internal namespace for experiment rebuilds."""

        if not generations_root.is_dir():
            raise CanonicalFactorStoreIntegrityError("experiment generations path must be a directory")
        for child in generations_root.iterdir():
            if not child.is_dir() or not _GENERATION_ID_PATTERN.fullmatch(child.name):
                raise CanonicalFactorStoreIntegrityError(
                    f"experiment generations has an invalid entry: {child.name!r}"
                )
            self._assert_experiment_generation_layout(child)

    @staticmethod
    def _assert_experiment_generation_layout(generation_root: Path) -> None:
        allowed_files = {"generation_manifest.json", "generation.done"}
        for path in generation_root.iterdir():
            if path.is_file():
                if path.name not in allowed_files:
                    raise CanonicalFactorStoreIntegrityError(
                        f"experiment generation has an unapproved top-level file: {path.name!r}"
                    )
                continue
            if not path.is_dir() or path.name not in _WIDE_TABLE_METADATA_DIRS:
                raise CanonicalFactorStoreIntegrityError(
                    f"experiment generation has an unapproved top-level entry: {path.name!r}"
                )

    def _validate_registered_contract_entry(
        self,
        kind: FactorTableKind,
        entry: object,
        *,
        table_root: Path | None = None,
        generation_id: str | None = None,
    ) -> None:
        if not isinstance(entry, Mapping):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} has a non-object registered contract")
        digest = str(entry.get("contract_list_sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} has an invalid registered contract hash")
        family = entry.get("family")
        if kind.family_aware:
            _normalise_family(kind, str(family or ""))
        elif family is not None:
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} registered contract unexpectedly has family")
        if not isinstance(entry.get("factor_ids"), list) or not isinstance(entry.get("output_columns"), list):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} registered contract lacks factor column lists")
        if int(entry.get("factor_count", -1)) != len(entry["factor_ids"]):
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} registered contract factor_count mismatch")
        contract_path_raw = str(entry.get("contract_path", "")).strip()
        if not contract_path_raw:
            raise CanonicalFactorStoreIntegrityError(f"{kind.display_name} registered contract lacks contract_path")
        resolved_table_root = table_root or self.table_root(kind)
        contract_path = _resolved(resolved_table_root / contract_path_raw)
        try:
            contract_path.relative_to(resolved_table_root)
        except ValueError as exc:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} registered contract path escapes its table root"
            ) from exc
        document = _read_json(contract_path, label="registered factor table contract")
        contract = FactorTableContract.from_dict(document)
        expected_document = self._contract_document(
            kind,
            contract,
            family=str(family) if family is not None else None,
        )
        expected_path = _resolved(
            self._contract_path(
                kind,
                contract,
                family=str(family) if family is not None else None,
                generation_id=generation_id,
            )
        )
        if contract_path != expected_path or document != expected_document:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} registered contract document/path mismatch"
            )
        if contract.sha256 != digest:
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} registered contract hash differs from document"
            )
        if list(entry["factor_ids"]) != list(contract.factor_ids) or list(entry["output_columns"]) != list(contract.output_columns):
            raise CanonicalFactorStoreIntegrityError(
                f"{kind.display_name} registered contract columns differ from document"
            )

    def _family_names(self, table_root: Path) -> list[str]:
        if not table_root.exists():
            return []
        names: list[str] = []
        for path in table_root.iterdir():
            if path.name in _FACTOR_LIBRARY_TABLE_METADATA_DIRS:
                if not path.is_dir():
                    raise CanonicalFactorStoreIntegrityError(
                        f"因子库内因子表 metadata path is not a directory: {path.name!r}"
                    )
                continue
            if not path.is_dir():
                continue
            if _FAMILY_PATTERN.fullmatch(path.name):
                names.append(path.name)
            else:
                raise CanonicalFactorStoreIntegrityError(
                    f"因子库内因子表 has an invalid family directory: {path.name!r}"
                )
        return sorted(names)

    def _partition_counts(
        self,
        kind: FactorTableKind,
        table_root: Path,
        families: Sequence[str],
    ) -> tuple[int, int]:
        scope_roots = [table_root / family for family in families] if kind.family_aware else [table_root]
        complete = 0
        incomplete = 0
        for scope in scope_roots:
            all_stems: set[tuple[str, str]] = set()
            for artifact_dir, suffix in (("factors", ".parquet"), ("manifests", ".json"), ("done", ".done")):
                base = scope / artifact_dir / PANEL_NAME
                if not base.is_dir():
                    continue
                for path in base.glob("*/*" + suffix):
                    if len(path.stem) == 8 and path.stem.isdigit():
                        all_stems.add((path.parent.name, path.stem))
            for month, stem in all_stems:
                paths = FactorDayPaths(
                    table_root=table_root,
                    scope_root=scope,
                    parquet_path=scope / "factors" / PANEL_NAME / month / f"{stem}.parquet",
                    manifest_path=scope / "manifests" / PANEL_NAME / month / f"{stem}.json",
                    done_path=scope / "done" / PANEL_NAME / month / f"{stem}.done",
                    lock_path=scope / "locks" / PANEL_NAME / month / f"{stem}.lock",
                )
                if self._partition_state(paths) == "complete":
                    complete += 1
                else:
                    incomplete += 1
        return complete, incomplete

    def _partition_counts_for_scope(self, scope_root: Path) -> tuple[int, int]:
        """Count complete and partial wide-table bundles in one generation."""

        all_stems: set[tuple[str, str]] = set()
        for artifact_dir, suffix in (("factors", ".parquet"), ("manifests", ".json"), ("done", ".done")):
            base = scope_root / artifact_dir / PANEL_NAME
            if not base.is_dir():
                continue
            for path in base.glob("*/*" + suffix):
                if len(path.stem) == 8 and path.stem.isdigit():
                    all_stems.add((path.parent.name, path.stem))
        complete = 0
        incomplete = 0
        for month, stem in all_stems:
            paths = FactorDayPaths(
                table_root=scope_root,
                scope_root=scope_root,
                parquet_path=scope_root / "factors" / PANEL_NAME / month / f"{stem}.parquet",
                manifest_path=scope_root / "manifests" / PANEL_NAME / month / f"{stem}.json",
                done_path=scope_root / "done" / PANEL_NAME / month / f"{stem}.done",
                lock_path=scope_root / "locks" / PANEL_NAME / month / f"{stem}.lock",
            )
            if self._partition_state(paths) == "complete":
                complete += 1
            else:
                incomplete += 1
        return complete, incomplete

    def _completed_days_for_scope(self, scope_root: Path) -> tuple[date, ...]:
        """List only complete day bundles from one fixed experiment scope."""

        done_root = scope_root / "done" / PANEL_NAME
        if not done_root.is_dir():
            return ()
        days: list[date] = []
        for month_root in sorted(done_root.iterdir(), key=lambda path: path.name):
            if not month_root.is_dir() or not re.fullmatch(r"\d{4}-\d{2}", month_root.name):
                raise CanonicalFactorStoreIntegrityError(
                    f"experiment generation has an invalid done month directory: {month_root}"
                )
            for done_path in sorted(month_root.iterdir(), key=lambda path: path.name):
                if not done_path.is_file() or not re.fullmatch(r"\d{8}\.done", done_path.name):
                    raise CanonicalFactorStoreIntegrityError(
                        f"experiment generation has an invalid done marker path: {done_path}"
                    )
                try:
                    day = date.fromisoformat(
                        f"{done_path.stem[:4]}-{done_path.stem[4:6]}-{done_path.stem[6:]}"
                    )
                except ValueError as exc:  # pragma: no cover - regex permits impossible calendar dates.
                    raise CanonicalFactorStoreIntegrityError(
                        f"experiment generation has an invalid done day: {done_path}"
                    ) from exc
                if month_root.name != f"{day:%Y-%m}":
                    raise CanonicalFactorStoreIntegrityError(
                        f"experiment generation done marker month differs from score day: {done_path}"
                    )
                day_paths = FactorDayPaths(
                    table_root=scope_root,
                    scope_root=scope_root,
                    parquet_path=scope_root / "factors" / PANEL_NAME / month_root.name / f"{done_path.stem}.parquet",
                    manifest_path=scope_root / "manifests" / PANEL_NAME / month_root.name / f"{done_path.stem}.json",
                    done_path=done_path,
                    lock_path=scope_root / "locks" / PANEL_NAME / month_root.name / f"{done_path.stem}.lock",
                )
                if self._partition_state(day_paths) != "complete":
                    raise CanonicalFactorStoreIntegrityError(
                        f"experiment generation done marker has an incomplete day bundle: {done_path}"
                    )
                days.append(day)
        if days != sorted(days) or len(days) != len(set(days)):
            raise CanonicalFactorStoreIntegrityError("experiment generation completed days must be sorted and unique")
        return tuple(days)

    def _partition_state(self, paths: FactorDayPaths) -> str:
        exists = (paths.parquet_path.is_file(), paths.manifest_path.is_file(), paths.done_path.is_file())
        if not any(exists):
            return "absent"
        if all(exists):
            return "complete"
        return "partial"

    @staticmethod
    def _logical_library_day_state(paths: FactorLibraryLogicalDayPaths) -> str:
        exists = (paths.manifest_path.is_file(), paths.done_path.is_file())
        if not any(exists):
            return "absent"
        if all(exists):
            return "complete"
        return "partial"

    def _library_family_partitions_for_day(self, day: date) -> dict[str, FactorDayPaths]:
        table_root = self.table_root(FactorTableKind.FACTOR_LIBRARY)
        paths_by_family: dict[str, FactorDayPaths] = {}
        for family in self._family_names(table_root):
            paths = self.partition_paths(FactorTableKind.FACTOR_LIBRARY, day, family=family)
            state = self._partition_state(paths)
            if state == "absent":
                continue
            if state != "complete":
                raise CanonicalFactorStoreIntegrityError(
                    "factor library has a partial family bundle for logical day: "
                    f"day={day.isoformat()} family={family!r}"
                )
            paths_by_family[family] = paths
        return paths_by_family

    def _validate_logical_library_day(
        self,
        day: date,
        *,
        required_family: str | None = None,
        expected_family: Mapping[str, FactorTableContract] | None = None,
    ) -> dict[str, Any]:
        """Validate the all-family manifest/done commit before any read."""

        paths = self.library_day_paths(day)
        state = self._logical_library_day_state(paths)
        if state == "absent":
            raise FileNotFoundError(
                f"factor library logical day is not published: {paths.manifest_path}"
            )
        if state != "complete":
            raise CanonicalFactorStoreIntegrityError(
                f"factor library logical day has a partial manifest/done bundle: {paths.manifest_path.parent}"
            )
        manifest = _read_json(paths.manifest_path, label="factor library logical day manifest")
        done = _read_json(paths.done_path, label="factor library logical day done marker")
        expected_identity = {
            "table_id": FactorTableKind.FACTOR_LIBRARY.value,
            "display_name": FactorTableKind.FACTOR_LIBRARY.display_name,
            "score_day": day.isoformat(),
        }
        if manifest.get("schema_version") != LIBRARY_DAY_MANIFEST_SCHEMA:
            raise CanonicalFactorStoreIntegrityError("factor library logical day manifest has unexpected schema_version")
        for key, value in expected_identity.items():
            if manifest.get(key) != value:
                raise CanonicalFactorStoreIntegrityError(
                    f"factor library logical day manifest {key!r} mismatch"
                )
        if manifest.get("panel") != {"name": PANEL_NAME, "logical_time": LOGICAL_PANEL_TIME}:
            raise CanonicalFactorStoreIntegrityError("factor library logical day manifest does not declare T1430")
        if done.get("schema_version") != LIBRARY_DAY_DONE_SCHEMA or done.get("ready") is not True:
            raise CanonicalFactorStoreIntegrityError("factor library logical day done marker is not ready")
        for key, value in expected_identity.items():
            if done.get(key) != value:
                raise CanonicalFactorStoreIntegrityError(
                    f"factor library logical day done marker {key!r} mismatch"
                )
        if done.get("panel_name") != PANEL_NAME:
            raise CanonicalFactorStoreIntegrityError("factor library logical day done marker does not declare T1430")
        if done.get("manifest_path") != _relative_to(paths.manifest_path, paths.table_root):
            raise CanonicalFactorStoreIntegrityError("factor library logical day done marker manifest path mismatch")
        if done.get("manifest_sha256") != _sha256_file(paths.manifest_path):
            raise CanonicalFactorStoreIntegrityError("factor library logical day manifest hash differs from done marker")

        raw_families = manifest.get("families")
        if not isinstance(raw_families, list) or not raw_families:
            raise CanonicalFactorStoreIntegrityError("factor library logical day manifest has no family bundle list")
        family_entries: dict[str, Mapping[str, Any]] = {}
        total_factor_count = 0
        for item in raw_families:
            if not isinstance(item, Mapping):
                raise CanonicalFactorStoreIntegrityError("factor library logical day has a non-object family entry")
            family = _normalise_family(FactorTableKind.FACTOR_LIBRARY, str(item.get("family", "")))
            if family in family_entries:
                raise CanonicalFactorStoreIntegrityError("factor library logical day has duplicate family entries")
            family_entries[family] = item
            expected_paths = self.partition_paths(FactorTableKind.FACTOR_LIBRARY, day, family=family)
            expected_values = {
                "parquet_path": _relative_to(expected_paths.parquet_path, paths.table_root),
                "parquet_sha256": _sha256_file(expected_paths.parquet_path),
                "manifest_path": _relative_to(expected_paths.manifest_path, paths.table_root),
                "manifest_sha256": _sha256_file(expected_paths.manifest_path),
                "done_path": _relative_to(expected_paths.done_path, paths.table_root),
                "done_sha256": _sha256_file(expected_paths.done_path),
            }
            for key, value in expected_values.items():
                if item.get(key) != value:
                    raise CanonicalFactorStoreIntegrityError(
                        f"factor library logical day family bundle {family!r} {key!r} mismatch"
                    )
            day_manifest = _read_json(expected_paths.manifest_path, label="factor library family day manifest")
            day_contract = day_manifest.get("contract")
            if not isinstance(day_contract, Mapping):
                raise CanonicalFactorStoreIntegrityError("factor library family day lacks contract evidence")
            if item.get("contract_list_sha256") != day_contract.get("contract_list_sha256"):
                raise CanonicalFactorStoreIntegrityError("factor library logical day family contract hash mismatch")
            if int(item.get("factor_count", -1)) != int(day_contract.get("factor_count", -1)):
                raise CanonicalFactorStoreIntegrityError("factor library logical day family factor count mismatch")
            artifact = day_manifest.get("artifact")
            if not isinstance(artifact, Mapping):
                raise CanonicalFactorStoreIntegrityError("factor library family day lacks artifact evidence")
            if int(item.get("row_count", -1)) != int(artifact.get("row_count", -1)):
                raise CanonicalFactorStoreIntegrityError("factor library logical day family row count mismatch")
            if item.get("frame_sha256") != artifact.get("frame_sha256"):
                raise CanonicalFactorStoreIntegrityError("factor library logical day family frame hash mismatch")
            total_factor_count += int(item["factor_count"])

        actual_paths = self._library_family_partitions_for_day(day)
        if set(actual_paths) != set(family_entries):
            raise CanonicalFactorStoreIntegrityError(
                "factor library logical day family set differs from committed family bundles: "
                f"committed={sorted(family_entries)!r} actual={sorted(actual_paths)!r}"
            )
        if int(manifest.get("family_bundle_count", -1)) != len(family_entries):
            raise CanonicalFactorStoreIntegrityError("factor library logical day family bundle count mismatch")
        if int(manifest.get("total_factor_count", -1)) != total_factor_count:
            raise CanonicalFactorStoreIntegrityError("factor library logical day total factor count mismatch")
        if int(done.get("family_bundle_count", -1)) != len(family_entries) or int(
            done.get("total_factor_count", -1)
        ) != total_factor_count:
            raise CanonicalFactorStoreIntegrityError("factor library logical day done count mismatch")
        if required_family is not None and required_family not in family_entries:
            raise CanonicalFactorStoreIntegrityError(
                f"factor library logical day does not commit requested family {required_family!r}"
            )
        if expected_family is not None:
            expected_normalized = {
                _normalise_family(FactorTableKind.FACTOR_LIBRARY, family): contract
                for family, contract in expected_family.items()
            }
            if not set(expected_normalized).issubset(family_entries):
                raise CanonicalFactorStoreIntegrityError(
                    "factor library logical day is missing an expected family contract"
                )
            for family, contract in expected_normalized.items():
                if family_entries[family].get("contract_list_sha256") != contract.sha256:
                    raise CanonicalFactorStoreIntegrityError(
                        f"factor library logical day contract differs for family {family!r}"
                    )
        return manifest

    def _contract_path(
        self,
        kind: FactorTableKind,
        contract: FactorTableContract,
        *,
        family: str | None,
        generation_id: str | None = None,
    ) -> Path:
        if generation_id is not None and kind is not FactorTableKind.EXPERIMENT:
            raise CanonicalFactorStoreValidationError("generation_id is only valid for the experiment factor table")
        if kind is FactorTableKind.EXPERIMENT:
            # Contract validation of the top table manifest always addresses
            # its retained flat history.  New generation callers pass their
            # ID explicitly; there is no implicit active-generation write.
            scope_root = self.experiment_generation_paths_for_id(
                generation_id or LEGACY_EXPERIMENT_GENERATION_ID
            ).generation_root
        else:
            scope_root = self.table_root(kind)
        if family is not None:
            scope_root = scope_root / family
        return scope_root / "contracts" / f"{contract.sha256}.json"

    def _contract_document(
        self,
        kind: FactorTableKind,
        contract: FactorTableContract,
        *,
        family: str | None,
    ) -> dict[str, Any]:
        payload = contract.to_dict()
        payload.update(
            {
                "table_id": kind.value,
                "display_name": kind.display_name,
                "family": family,
                "panel_name": PANEL_NAME,
                "contract_list_sha256": contract.sha256,
            }
        )
        return payload

    def _register_contract(
        self,
        kind: FactorTableKind,
        contract: FactorTableContract,
        *,
        family: str | None,
        generation_id: str | None = None,
    ) -> Path:
        """Persist/reuse a list contract and append it to the table manifest."""

        if generation_id is not None and kind is not FactorTableKind.EXPERIMENT:
            raise CanonicalFactorStoreValidationError("generation_id is only valid for the experiment factor table")
        generation_paths = (
            self.experiment_generation_paths_for_id(generation_id or LEGACY_EXPERIMENT_GENERATION_ID)
            if kind is FactorTableKind.EXPERIMENT
            else None
        )
        table_root = generation_paths.generation_root if generation_paths is not None else self.table_root(kind)
        metadata_name = "generation_manifest.lock" if generation_paths and not generation_paths.is_legacy_flat else "table_manifest.lock"
        metadata_lock = table_root / "locks" / metadata_name
        with self._day_lock(metadata_lock):
            manifest = (
                self._load_experiment_generation_manifest(generation_paths.generation_id)
                if generation_paths is not None and not generation_paths.is_legacy_flat
                else self._load_table_manifest(kind)
            )
            path = self._contract_path(kind, contract, family=family, generation_id=generation_id)
            expected_document = self._contract_document(kind, contract, family=family)
            if path.exists():
                observed = _read_json(path, label="factor table contract document")
                if observed != expected_document:
                    raise CanonicalFactorStoreConflictError(
                        "contract-list hash resolves to a different persisted document: " f"{path}"
                    )
            else:
                _write_json_atomic(path, expected_document)

            registration = {
                "contract_list_sha256": contract.sha256,
                "family": family,
                "contract_path": _relative_to(path, table_root),
                "factor_count": len(contract.columns),
                "factor_ids": list(contract.factor_ids),
                "output_columns": list(contract.output_columns),
            }
            registered = list(manifest["registered_contracts"])
            matching = [
                item
                for item in registered
                if isinstance(item, Mapping)
                and str(item.get("contract_list_sha256", "")) == contract.sha256
                and item.get("family") == family
            ]
            if matching:
                if len(matching) != 1 or dict(matching[0]) != registration:
                    raise CanonicalFactorStoreConflictError(
                        "table manifest registers the same contract hash with different metadata"
                    )
            else:
                registered.append(registration)
                manifest["registered_contracts"] = registered
                target_manifest_path = (
                    generation_paths.manifest_path
                    if generation_paths is not None and not generation_paths.is_legacy_flat
                    else table_root / "table_manifest.json"
                )
                _write_json_atomic(target_manifest_path, manifest)
            return path

    def _write_parquet_atomic(
        self,
        path: Path,
        frame: pd.DataFrame,
        *,
        day: date,
        contract: FactorTableContract,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.{uuid4().hex}.tmp.parquet")
        try:
            frame.to_parquet(temporary, index=True)
            # Read the staged artifact before making it visible.  The frame is
            # normalised again so an engine/schema surprise fails before commit.
            staged = _normalise_factor_frame(pd.read_parquet(temporary), day=day, contract=contract)
            if _frame_hashes(staged)["frame_sha256"] != _frame_hashes(frame)["frame_sha256"]:
                raise CanonicalFactorStoreIntegrityError("staged parquet does not round-trip to the canonical frame")
            # Windows rejects fsync on a read-only descriptor.
            with temporary.open("rb+") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink(missing_ok=True)

    def _build_day_manifest(
        self,
        *,
        kind: FactorTableKind,
        day: date,
        family: str | None,
        paths: FactorDayPaths,
        contract: FactorTableContract,
        contract_path: Path,
        artifact_hashes: Mapping[str, Any],
        row_count: int,
        source_evidence: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": DAY_MANIFEST_SCHEMA,
            "table_id": kind.value,
            "display_name": kind.display_name,
            "family": family,
            "score_day": day.isoformat(),
            "panel": {"name": PANEL_NAME, "logical_time": LOGICAL_PANEL_TIME},
            "contract": {
                "contract_list_sha256": contract.sha256,
                "contract_path": _relative_to(contract_path, paths.table_root),
                "document_sha256": _sha256_file(contract_path),
                "factor_count": len(contract.columns),
                "factor_ids": list(contract.factor_ids),
                "output_columns": list(contract.output_columns),
            },
            "artifact": {
                "parquet_path": _relative_to(paths.parquet_path, paths.table_root),
                "parquet_sha256": _sha256_file(paths.parquet_path),
                "row_count": int(row_count),
                "hash_semantics": FRAME_HASH_SEMANTICS,
                **dict(artifact_hashes),
            },
            "source_evidence": dict(source_evidence) if source_evidence is not None else None,
        }

    def _read_complete_day(
        self,
        kind: FactorTableKind,
        day: date,
        *,
        family: str | None,
        expected_contract: FactorTableContract | None,
        generation_id: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        paths = self.partition_paths(kind, day, family=family, generation_id=generation_id)
        if self._partition_state(paths) != "complete":
            raise CanonicalFactorStoreIntegrityError("requested factor day is not a complete artifact bundle")
        manifest = _read_json(paths.manifest_path, label="canonical factor day manifest")
        done = _read_json(paths.done_path, label="canonical factor day done marker")
        if str(manifest.get("schema_version", "")) != DAY_MANIFEST_SCHEMA:
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest has an unexpected schema_version")
        expected_identity = {
            "table_id": kind.value,
            "display_name": kind.display_name,
            "family": family,
            "score_day": day.isoformat(),
        }
        for key, expected in expected_identity.items():
            if manifest.get(key) != expected:
                raise CanonicalFactorStoreIntegrityError(
                    f"canonical factor day manifest {key!r} mismatch: {paths.manifest_path}"
                )
        panel = manifest.get("panel")
        if panel != {"name": PANEL_NAME, "logical_time": LOGICAL_PANEL_TIME}:
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest does not declare T1430")
        if str(done.get("schema_version", "")) != DAY_DONE_SCHEMA or done.get("ready") is not True:
            raise CanonicalFactorStoreIntegrityError("canonical factor day done marker is not ready")
        for key, expected in expected_identity.items():
            if done.get(key) != expected:
                raise CanonicalFactorStoreIntegrityError(f"canonical factor day done marker {key!r} mismatch")
        if done.get("panel_name") != PANEL_NAME:
            raise CanonicalFactorStoreIntegrityError("canonical factor day done marker does not declare T1430")
        if done.get("manifest_path") != _relative_to(paths.manifest_path, paths.table_root):
            raise CanonicalFactorStoreIntegrityError("canonical factor day done marker manifest path mismatch")
        if done.get("parquet_path") != _relative_to(paths.parquet_path, paths.table_root):
            raise CanonicalFactorStoreIntegrityError("canonical factor day done marker parquet path mismatch")
        if done.get("manifest_sha256") != _sha256_file(paths.manifest_path):
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest hash differs from done marker")
        if done.get("parquet_sha256") != _sha256_file(paths.parquet_path):
            raise CanonicalFactorStoreIntegrityError("canonical factor day parquet hash differs from done marker")

        raw_contract = manifest.get("contract")
        if not isinstance(raw_contract, Mapping):
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest has no contract evidence")
        contract_hash = str(raw_contract.get("contract_list_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", contract_hash):
            raise CanonicalFactorStoreIntegrityError("canonical factor day contract list hash is invalid")
        contract_path = _resolved(paths.table_root / str(raw_contract.get("contract_path", "")))
        try:
            contract_path.relative_to(paths.table_root)
        except ValueError as exc:
            raise CanonicalFactorStoreIntegrityError("canonical factor day contract path escapes table root") from exc
        contract_document = _read_json(contract_path, label="canonical factor contract document")
        contract = FactorTableContract.from_dict(contract_document)
        expected_contract_path = _resolved(
            self._contract_path(kind, contract, family=family, generation_id=generation_id)
        )
        if contract_path != expected_contract_path:
            raise CanonicalFactorStoreIntegrityError("canonical factor day contract path is not its canonical hash path")
        expected_contract_document = self._contract_document(kind, contract, family=family)
        if contract_document != expected_contract_document or contract.sha256 != contract_hash:
            raise CanonicalFactorStoreIntegrityError("canonical factor contract document differs from its day manifest")
        if raw_contract.get("document_sha256") != _sha256_file(contract_path):
            raise CanonicalFactorStoreIntegrityError("canonical factor contract document hash differs from day manifest")
        if list(raw_contract.get("factor_ids", [])) != list(contract.factor_ids):
            raise CanonicalFactorStoreIntegrityError("canonical factor day factor IDs differ from contract document")
        if list(raw_contract.get("output_columns", [])) != list(contract.output_columns):
            raise CanonicalFactorStoreIntegrityError("canonical factor day output columns differ from contract document")
        if int(raw_contract.get("factor_count", -1)) != len(contract.columns):
            raise CanonicalFactorStoreIntegrityError("canonical factor day factor count differs from contract document")
        if done.get("contract_list_sha256") != contract.sha256:
            raise CanonicalFactorStoreIntegrityError("canonical factor day done marker contract differs from manifest")
        if expected_contract is not None and expected_contract.sha256 != contract.sha256:
            raise CanonicalFactorStoreConflictError("canonical factor day has a different factor contract list")

        artifact = manifest.get("artifact")
        if not isinstance(artifact, Mapping):
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest has no artifact evidence")
        if artifact.get("parquet_path") != _relative_to(paths.parquet_path, paths.table_root):
            raise CanonicalFactorStoreIntegrityError("canonical factor day manifest parquet path mismatch")
        if artifact.get("parquet_sha256") != _sha256_file(paths.parquet_path):
            raise CanonicalFactorStoreIntegrityError("canonical factor day parquet hash differs from manifest")
        output = pd.read_parquet(paths.parquet_path)
        normalized = _normalise_factor_frame(output, day=day, contract=contract)
        hash_semantics = artifact.get("hash_semantics")
        if hash_semantics is None:
            # Existing canonical manifests were written before semantic
            # datetime-unit hashing.  Preserve their immutable evidence and
            # returned physical dtype rather than rewriting live/history.
            hashes = _legacy_frame_hashes(normalized)
        elif hash_semantics == FRAME_HASH_SEMANTICS:
            hashes = _frame_hashes(normalized)
        else:
            raise CanonicalFactorStoreIntegrityError(
                f"canonical factor day has an unsupported artifact hash semantics: {hash_semantics!r}"
            )
        if int(artifact.get("row_count", -1)) != len(normalized):
            raise CanonicalFactorStoreIntegrityError("canonical factor day row count differs from manifest")
        for key in ("frame_sha256", "index_sha256", "column_sha256"):
            if artifact.get(key) != hashes[key]:
                raise CanonicalFactorStoreIntegrityError(f"canonical factor day {key} differs from manifest")
        return normalized, manifest

    @contextmanager
    def _day_lock(self, lock_path: Path) -> Iterator[_LockHandle]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        payload = {"pid": os.getpid(), "token": token, "acquired_at": datetime.now().isoformat(timespec="seconds")}
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise CanonicalFactorStoreLockError(
                f"canonical factor-table lock is already held: {lock_path}"
            ) from exc
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical_json_bytes(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            handle_value = _LockHandle(path=lock_path, token=token)
            yield handle_value
        finally:
            try:
                current = _read_json(lock_path, label="canonical factor-table lock")
                if current.get("token") != token:
                    raise CanonicalFactorStoreLockError(
                        f"refusing to release a canonical lock no longer owned by this writer: {lock_path}"
                    )
                lock_path.unlink(missing_ok=False)
            except FileNotFoundError:
                # A missing lock after an unsuccessful acquisition is harmless;
                # after a successful acquisition it means a concurrent actor
                # tampered with the lock and must not be hidden.
                if 'handle_value' in locals():
                    raise CanonicalFactorStoreLockError(f"canonical lock disappeared before release: {lock_path}")


__all__ = [
    "CanonicalFactorStore",
    "CanonicalFactorStoreConflictError",
    "CanonicalFactorStoreError",
    "CanonicalFactorStoreIntegrityError",
    "CanonicalFactorStoreLockError",
    "CanonicalFactorStoreReport",
    "CanonicalFactorStoreValidationError",
    "CanonicalFactorTableReport",
    "CONTRACT_SCHEMA",
    "DAY_DONE_SCHEMA",
    "DAY_MANIFEST_SCHEMA",
    "EXPERIMENT_GENERATION_DONE_SCHEMA",
    "EXPERIMENT_GENERATION_POINTER_SCHEMA",
    "EXPERIMENT_GENERATION_SCHEMA",
    "ExperimentGenerationPaths",
    "FactorColumnContract",
    "FactorDayPaths",
    "FactorLibraryLogicalDayPaths",
    "FactorTableContract",
    "FactorTableKind",
    "FactorTableWriteResult",
    "FRAME_HASH_SEMANTICS",
    "LOGICAL_PANEL_TIME",
    "LIBRARY_DAY_DONE_SCHEMA",
    "LIBRARY_DAY_MANIFEST_SCHEMA",
    "LEGACY_EXPERIMENT_GENERATION_ID",
    "PANEL_NAME",
    "TABLE_MANIFEST_SCHEMA",
    "coerce_factor_table_kind",
]
