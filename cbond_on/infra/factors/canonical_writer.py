"""Designated writers for the three canonical CBOND_ON factor tables.

The legacy :class:`~cbond_on.domain.factors.storage.FactorStore` writes a
parquet file directly.  That is intentionally not valid for a canonical table:
every day must be committed through :class:`CanonicalFactorStore` so the
parquet, manifest, contract, and ``.done`` marker stay aligned.

This module supplies a deliberately small, FactorStore-shaped writer for the
two wide tables.  It is a transport adapter only: callers still own admission,
factor computation, and the authority to choose one table.  The factor-library
writer is family-aware and therefore belongs to its dedicated 23:59 workflow,
not this wide-table adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    FactorColumnContract,
    FactorTableContract,
    FactorTableKind,
    coerce_factor_table_kind,
)


_WRITER_AUTHORITY_SEAL = object()
_OFFICIAL_STORE_ROOT = Path(r"D:/cbond_on/factor_store").resolve(strict=False)


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


@dataclass(frozen=True, init=False)
class CanonicalFactorWriterAuthority:
    """Opaque authority for one designated canonical table writer."""

    table_id: str
    root: Path
    role: str
    generation_id: str | None
    _seal: object


def _issue_authority(
    *,
    root: str | Path,
    table_id: str,
    role: str,
    generation_id: str | None = None,
) -> CanonicalFactorWriterAuthority:
    authority = object.__new__(CanonicalFactorWriterAuthority)
    object.__setattr__(authority, "table_id", str(table_id))
    object.__setattr__(authority, "root", _resolved(root))
    object.__setattr__(authority, "role", str(role))
    object.__setattr__(authority, "generation_id", str(generation_id) if generation_id is not None else None)
    object.__setattr__(authority, "_seal", _WRITER_AUTHORITY_SEAL)
    return authority


def issue_admitted_live_writer_authority(*, root: str | Path) -> CanonicalFactorWriterAuthority:
    """Issue the live authority after the active-profile gate has succeeded."""

    resolved_root = _resolved(root)
    if resolved_root != _OFFICIAL_STORE_ROOT:
        raise PermissionError(
            f"admitted live writer authority requires official root {_OFFICIAL_STORE_ROOT}; got {resolved_root}"
        )
    return _issue_authority(root=resolved_root, table_id=FactorTableKind.LIVE.value, role="admitted_live")


def issue_experiment_publisher_authority(
    *,
    root: str | Path,
    research_only: bool,
    generation_id: str,
) -> CanonicalFactorWriterAuthority:
    """Issue a staging-generation authority for an explicit research publisher.

    New experiment values must never write through the flat historical table.
    The designated publisher receives authority for exactly one pre-created,
    inactive generation; activation is a later store-level operation.
    """

    resolved_root = _resolved(root)
    if resolved_root != _OFFICIAL_STORE_ROOT or not research_only:
        raise PermissionError(
            "experiment publisher authority requires research_only=true and the official canonical store root"
        )
    resolved_generation_id = CanonicalFactorStore(resolved_root)._normalise_experiment_generation_id(
        generation_id,
        allow_legacy=False,
    )
    return _issue_authority(
        root=resolved_root,
        table_id=FactorTableKind.EXPERIMENT.value,
        role="experiment_publisher",
        generation_id=resolved_generation_id,
    )


def issue_test_canonical_writer_authority(
    *,
    root: str | Path,
    table_id: FactorTableKind | str,
    generation_id: str | None = None,
) -> CanonicalFactorWriterAuthority:
    """Fixture-only authority; unavailable outside a pytest process."""

    if not os.environ.get("PYTEST_CURRENT_TEST"):
        raise PermissionError("test canonical writer authority is available only under pytest")
    kind = coerce_factor_table_kind(table_id)
    if kind.family_aware:
        raise ValueError("test wide writer authority does not support factor_library")
    if kind is FactorTableKind.EXPERIMENT:
        if generation_id is None:
            raise ValueError("test experiment writer authority requires an explicit staging generation_id")
        generation_id = CanonicalFactorStore(root)._normalise_experiment_generation_id(
            generation_id,
            allow_legacy=False,
        )
    elif generation_id is not None:
        raise ValueError("only experiment writer authority may declare generation_id")
    return _issue_authority(root=root, table_id=kind.value, role="test_fixture", generation_id=generation_id)


def current_catalog_contract(factor_ids: Sequence[str]) -> FactorTableContract:
    """Build an exact current-Catalog materialisation contract.

    This is appropriate only when the caller actually computes the values
    using the current registered factor definitions. Historical copies must
    construct their own source-attested materialisation contract instead.
    """

    from cbond_on.domain.factor_catalog import resolve_factor_instance

    factor_ids = tuple(str(value).strip() for value in factor_ids)
    if not factor_ids or len(factor_ids) != len(set(factor_ids)) or any(not value for value in factor_ids):
        raise ValueError("current catalog contract requires unique non-empty factor IDs")
    columns: list[FactorColumnContract] = []
    for factor_id in factor_ids:
        instance = resolve_factor_instance(factor_id)
        output_column = str(instance.get("output_column") or factor_id).strip()
        columns.append(
            FactorColumnContract(
                factor_id=factor_id,
                factor_version=str(instance.get("factor_version", "")).strip(),
                contract_hash=str(instance.get("contract_hash", "")).strip().lower(),
                output_column=output_column,
                materialization_contract_origin="current_catalog",
            )
        )
    return FactorTableContract(tuple(columns))


def live50_current_catalog_contract() -> FactorTableContract:
    """Return the immutable ordered current contract for the live writer."""

    from cbond_on.infra.live.factor_admission import LIVE50_COLUMNS

    return current_catalog_contract(LIVE50_COLUMNS)


@dataclass
class CanonicalFactorTableWriter:
    """A narrow read/write adapter for one canonical wide factor table.

    It intentionally implements only the three methods consumed by the factor
    pipeline: ``day_path``, ``read_day``, and ``write_day``. Direct parquet
    access is never exposed as a writer path.
    """

    root: Path
    table_id: FactorTableKind | str
    contract: FactorTableContract
    source_evidence: Mapping[str, Any]
    authority: CanonicalFactorWriterAuthority | None = None
    generation_id: str | None = None
    panel_name: str = "T1430"
    window_minutes: int = 15

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()
        self._kind = coerce_factor_table_kind(self.table_id)
        if self._kind.family_aware:
            raise ValueError(
                "CanonicalFactorTableWriter supports only wide canonical tables; "
                "factor_library must be published by its family-aware workflow"
            )
        if self.panel_name != "T1430":
            raise ValueError("canonical factor tables support only the published T1430 contract")
        if not isinstance(self.contract, FactorTableContract):
            raise TypeError("canonical factor writer requires a FactorTableContract")
        if not isinstance(self.source_evidence, Mapping):
            raise TypeError("canonical factor writer source_evidence must be a mapping")
        if (
            not isinstance(self.authority, CanonicalFactorWriterAuthority)
            or self.authority._seal is not _WRITER_AUTHORITY_SEAL
            or self.authority.table_id != self._kind.value
            or self.authority.root != self.root.resolve(strict=False)
        ):
            raise PermissionError("canonical factor writer requires a matching designated writer authority")
        allowed_roles = {
            FactorTableKind.LIVE.value: {"admitted_live", "test_fixture"},
            FactorTableKind.EXPERIMENT.value: {"experiment_publisher", "test_fixture"},
        }
        if self.authority.role not in allowed_roles[self._kind.value]:
            raise PermissionError(
                f"writer role {self.authority.role!r} is not permitted for canonical {self._kind.value}"
            )
        self._store = CanonicalFactorStore(self.root)
        if self._kind is FactorTableKind.EXPERIMENT:
            if self.generation_id is None:
                raise ValueError("canonical experiment writer requires an explicit staging generation_id")
            self.generation_id = self._store._normalise_experiment_generation_id(
                self.generation_id,
                allow_legacy=False,
            )
            if self.authority.generation_id != self.generation_id:
                raise PermissionError(
                    "canonical experiment writer authority is not bound to this staging generation"
                )
            # Prove the stage exists and has not been finalized before a
            # pipeline can obtain a writer-shaped adapter.
            self._store._require_staging_experiment_generation(self.generation_id)
        elif self.generation_id is not None or self.authority.generation_id is not None:
            raise ValueError("only canonical experiment writers may declare generation_id")

    @property
    def kind(self) -> FactorTableKind:
        return self._kind

    def day_path(self, day: date) -> Path:
        return self._store.partition_paths(self._kind, day, generation_id=self.generation_id).parquet_path

    def read_day(self, day: date) -> pd.DataFrame:
        paths = self._store.partition_paths(self._kind, day, generation_id=self.generation_id)
        if not paths.parquet_path.exists() and not self._store.root.exists():
            return pd.DataFrame()
        try:
            if self._kind is FactorTableKind.EXPERIMENT:
                assert self.generation_id is not None
                return self._store.read_staging_experiment_day(
                    self.generation_id,
                    day,
                    expected_contract=self.contract,
                )
            return self._store.read_day(self._kind, day, expected_contract=self.contract)
        except FileNotFoundError:
            # Preserve the legacy writer contract: an absent unpublished day
            # means there is nothing to merge yet, not a valid empty artifact.
            return pd.DataFrame()

    def write_day(self, day: date, frame: pd.DataFrame) -> None:
        self._store.write_day(
            self._kind,
            day,
            frame,
            contract=self.contract,
            source_evidence=dict(self.source_evidence),
            generation_id=self.generation_id,
        )


__all__ = [
    "CanonicalFactorTableWriter",
    "CanonicalFactorWriterAuthority",
    "current_catalog_contract",
    "issue_admitted_live_writer_authority",
    "issue_experiment_publisher_authority",
    "issue_test_canonical_writer_authority",
    "live50_current_catalog_contract",
]
