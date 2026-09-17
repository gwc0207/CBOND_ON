"""Migrate existing FactorStore outputs into the canonical three-table root.

This is a deliberately narrow, data-copy-only tool.  It never computes a
factor, rebuilds a panel, changes an active configuration, touches a database,
or starts/restarts a scheduler.  Its sources are opened read-only and its only
write destination is the canonical local root ``D:/cbond_on/factor_store``.

The migration has one important ownership rule for the 因子库内因子表:

* catalog-history owns its 773 physical columns on every catalog-history day;
* the 27 live columns absent there are copied from live on all live days;
* the 23 live columns shared with catalog-history are copied from live *only*
  outside catalog-history coverage;
* the supplement and live frames on 2026-08-25 may be unioned only when their
  factor columns are disjoint and their full ``(dt, code)`` indexes are exact.

No same-name values are ever coalesced, filled, or chosen by a value heuristic.
The historical source contracts remain in each target partition's provenance
evidence, including supplement's source-level historical contract hashes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

import json5
import numpy as np
import pandas as pd


# A harness tool is normally executed directly from ``harness/tools``.  Keep
# imports deterministic without creating a new run entrypoint.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cbond_on.infra.factors.canonical_store import (  # noqa: E402
    CanonicalFactorStore,
    CanonicalFactorStoreConflictError,
    CanonicalFactorStoreIntegrityError,
    FactorColumnContract,
    FactorTableContract,
    FactorTableKind,
)
import harness.tools.plan_factor_table_migration as planner  # noqa: E402


CANONICAL_FACTOR_STORE_ROOT = Path(r"D:/cbond_on/factor_store")
MIGRATION_SCHEMA = "cbond_on_factor_table_migration_execution/v1"
OWNERSHIP_POLICY_ID = "catalog_history_owns_overlap_days_v1"

# This set is intentionally route-oriented rather than "latest day" oriented:
# it exercises the live/experiment roots, historical family routing, the
# supplement+live union, and the live-only tail.
ROUTE_SMOKE_DAYS = (
    date(2024, 1, 3),
    date(2025, 1, 2),
    date(2026, 8, 25),
    date(2026, 8, 27),
)


class FactorTableMigrationError(RuntimeError):
    """The migration's immutable source/target contract was not satisfied."""


@dataclass(frozen=True)
class CatalogFactor:
    factor_id: str
    output_column: str
    primary_family: str
    contract_hash: str
    factor_version: str


@dataclass(frozen=True)
class SourceRuntime:
    name: str
    scan: planner.StoreScan
    manifest_path: Path | None
    manifest: Mapping[str, Any] | None
    manifest_sha256: str | None
    contract_status: str
    partition_sha256: Mapping[date, str]


@dataclass(frozen=True)
class Route:
    route_id: str
    target: FactorTableKind
    source: str
    factor_ids: tuple[str, ...]
    days: tuple[date, ...]
    ownership_rule: str


@dataclass(frozen=True)
class MigrationPlan:
    """Fully resolved read-only execution plan; no frame data is retained."""

    target_root: Path
    planner_plan: Mapping[str, Any]
    planner_plan_sha256: str
    catalog_path: Path
    catalog_sha256: str
    catalog: Mapping[str, CatalogFactor]
    governing_artifacts: Mapping[str, tuple[Path, str]]
    sources: Mapping[str, SourceRuntime]
    routes: tuple[Route, ...]
    selected_days: tuple[date, ...] | None
    smoke_route_days: bool
    ownership_ack_required: bool

    def to_summary(self) -> dict[str, Any]:
        operation_summary = _operation_summary(self)
        return {
            "schema_version": MIGRATION_SCHEMA,
            "mode": "plan",
            "read_only": True,
            "target_root": _path_text(self.target_root),
            "only_write_destination_when_executed": _path_text(self.target_root),
            "catalog": {
                "path": _path_text(self.catalog_path),
                "sha256": self.catalog_sha256,
                "factor_count": len(self.catalog),
            },
            "governing_artifacts": {
                name: {"path": _path_text(path), "sha256": sha256}
                for name, (path, sha256) in self.governing_artifacts.items()
            },
            "source_roots": {
                name: {
                    "root": _path_text(runtime.scan.root),
                    "factor_count": len(runtime.scan.factor_ids),
                    "day_count": len(runtime.scan.days),
                    "contract_status": runtime.contract_status,
                    "selected_partition_hash_count": len(runtime.partition_sha256),
                }
                for name, runtime in self.sources.items()
            },
            "selection": {
                "smoke_route_days": self.smoke_route_days,
                "selected_days": [item.isoformat() for item in self.selected_days] if self.selected_days else None,
                "route_smoke_days": [item.isoformat() for item in ROUTE_SMOKE_DAYS] if self.smoke_route_days else [],
            },
            "routes": [
                {
                    "route_id": route.route_id,
                    "target_table_id": route.target.value,
                    "source": route.source,
                    "factor_count": len(route.factor_ids),
                    "factor_ids_sha256": _sha256_json(list(route.factor_ids)),
                    "day_count": len(route.days),
                    "coverage": _date_summary(route.days),
                    "ownership_rule": route.ownership_rule,
                }
                for route in self.routes
            ],
            "operation_summary": operation_summary,
            "planner_plan_sha256": self.planner_plan_sha256,
            "planner_readiness": self.planner_plan.get("migration_readiness"),
            "ownership_policy": {
                "id": OWNERSHIP_POLICY_ID,
                "acknowledgement_required_for_execute": self.ownership_ack_required,
                "description": "No same-name source values are coalesced. Catalog-history owns its dates; shared live columns are routed only outside those dates.",
            },
            "execution_boundary": {
                "factor_compute": "not_called",
                "panel_build": "not_called",
                "active_live_config": "not_modified",
                "scheduler": "not_called",
                "database": "not_called",
                "source_factor_stores": "read_only",
                "target_factor_store": "not_created_in_plan_mode",
            },
            "partial_bundle_repair_runbook": [
                "Do not delete, overwrite, or clean a partial canonical bundle.",
                "If an interruption occurred before a factor-library logical day manifest/.done, rerun the identical selected scope; complete family bundles are idempotently verified and missing bundles are published.",
                "If any parquet/manifest/done trio or logical library day manifest/.done is partial, stop: preserve it for audit and resolve it explicitly before retrying.",
                "A source hash, governing-artifact hash, contract, value, or source-evidence mismatch is a fail-closed conflict, not a repair candidate.",
            ],
        }


@dataclass(frozen=True)
class MigrationResult:
    target_root: Path
    mode: str
    planner_plan_sha256: str
    selected_days: tuple[date, ...] | None
    written_by_table: Mapping[str, int]
    already_present_by_table: Mapping[str, int]
    attestations: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MIGRATION_SCHEMA,
            "mode": self.mode,
            "target_root": _path_text(self.target_root),
            "planner_plan_sha256": self.planner_plan_sha256,
            "selected_days": [item.isoformat() for item in self.selected_days] if self.selected_days else None,
            "written_by_table": dict(self.written_by_table),
            "already_present_by_table": dict(self.already_present_by_table),
            "attestations": dict(self.attestations),
            "execution_boundary": {
                "factor_compute": "not_called",
                "panel_build": "not_called",
                "active_live_config": "not_modified",
                "scheduler": "not_called",
                "database": "not_called",
                "source_factor_stores": "read_only",
            },
        }


@dataclass(frozen=True)
class SourceFrameSnapshot:
    source_name: str
    day: date
    path: Path
    sha256: str
    frame: pd.DataFrame


class SourceFrameCache:
    """Freeze each approved source partition before it can enter a target.

    The migration reads one source parquet at most once per source/day, then
    splits that in-memory canonical frame by family.  The planned pre-hash is
    checked immediately before and after the read, and again after all target
    writes/read-back.  A changing active source therefore fails closed rather
    than becoming a mixed migration snapshot.
    """

    def __init__(self, plan: MigrationPlan) -> None:
        self.plan = plan
        self._snapshots: dict[tuple[str, date], SourceFrameSnapshot] = {}

    def frame(self, source_name: str, day: date) -> pd.DataFrame:
        return self.snapshot(source_name, day).frame

    def snapshot(self, source_name: str, day: date) -> SourceFrameSnapshot:
        key = (source_name, day)
        cached = self._snapshots.get(key)
        if cached is not None:
            return cached
        runtime = self.plan.sources[source_name]
        path = runtime.scan.paths_by_day.get(day)
        expected_hash = runtime.partition_sha256.get(day)
        if path is None or expected_hash is None:
            raise FactorTableMigrationError(
                f"source partition was not pre-snapshotted for migration: {source_name}@{day.isoformat()}"
            )
        before = _sha256_file(path)
        if before != expected_hash:
            raise FactorTableMigrationError(
                f"source partition changed after preflight before read: {source_name}@{day.isoformat()}"
            )
        frame = pd.read_parquet(path)
        frame = _validate_source_frame(
            self.plan,
            source_name=source_name,
            day=day,
            frame=frame,
            path=path,
        )
        after = _sha256_file(path)
        if after != before:
            raise FactorTableMigrationError(
                f"source partition changed while being read: {source_name}@{day.isoformat()}"
            )
        snapshot = SourceFrameSnapshot(
            source_name=source_name,
            day=day,
            path=path,
            sha256=before,
            frame=frame,
        )
        self._snapshots[key] = snapshot
        return snapshot

    def assert_unchanged(self) -> None:
        for source_name, runtime in self.plan.sources.items():
            for day, expected_hash in runtime.partition_sha256.items():
                path = runtime.scan.paths_by_day.get(day)
                if path is None or _sha256_file(path) != expected_hash:
                    raise FactorTableMigrationError(
                        f"source partition changed during migration: {source_name}@{day.isoformat()}"
                    )
            if runtime.manifest_path is not None and runtime.manifest_sha256 is not None:
                if _sha256_file(runtime.manifest_path) != runtime.manifest_sha256:
                    raise FactorTableMigrationError(
                        f"source manifest changed during migration: {source_name}"
                    )

    def release(self, source_name: str, day: date) -> None:
        """Release one materialised frame once its atomic day work is complete."""

        self._snapshots.pop((source_name, day), None)

    def release_day(self, day: date) -> None:
        for key in [key for key in self._snapshots if key[1] == day]:
            self._snapshots.pop(key, None)


def _validate_source_frame(
    plan: MigrationPlan,
    *,
    source_name: str,
    day: date,
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    runtime = plan.sources[source_name]
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
        if {"dt", "code"}.issubset(frame.columns):
            frame = frame.set_index(["dt", "code"])
        else:
            raise FactorTableMigrationError(
                f"source {source_name!r} does not preserve a (dt, code) index: {path}"
            )
    if frame.index.has_duplicates:
        raise FactorTableMigrationError(f"source {source_name!r} has duplicate (dt, code) rows: {path}")
    expected_columns = list(runtime.scan.factor_ids)
    if [str(column) for column in frame.columns] != expected_columns:
        raise FactorTableMigrationError(
            f"source {source_name!r} physical output order drifted after plan build: {path}"
        )
    observed_times = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce"))
    expected_t1430 = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    if bool(pd.isna(observed_times).any()) or not bool((observed_times == expected_t1430).all()):
        raise FactorTableMigrationError(f"source {source_name!r} is not exact T1430 for {day.isoformat()}: {path}")
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]) or pd.api.types.is_bool_dtype(frame[column]):
            raise FactorTableMigrationError(
                f"source {source_name!r} has a nonnumeric factor column {column!r}: {path}"
            )
    values = frame.to_numpy(dtype="float64", na_value=np.nan)
    if bool(np.isinf(values).any()):
        raise FactorTableMigrationError(f"source {source_name!r} contains Inf values: {path}")
    return frame.astype("float64").sort_index(kind="mergesort")


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(_resolved(left)))) == os.path.normcase(
        os.path.normpath(str(_resolved(right)))
    )


def _paths_overlap(left: str | Path, right: str | Path) -> bool:
    first = _resolved(left)
    second = _resolved(right)
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _path_text(path: Path) -> str:
    return _resolved(path).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _date_summary(days: Sequence[date]) -> dict[str, Any]:
    return {
        "day_count": len(days),
        "start": days[0].isoformat() if days else None,
        "end": days[-1].isoformat() if days else None,
    }


def _parse_catalog(path: str | Path) -> tuple[Path, Mapping[str, CatalogFactor]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FactorTableMigrationError(f"catalog is unreadable JSON: {resolved}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("factors"), list):
        raise FactorTableMigrationError(f"catalog lacks a factors list: {resolved}")
    rows: dict[str, CatalogFactor] = {}
    output_columns: set[str] = set()
    for raw in payload["factors"]:
        if not isinstance(raw, Mapping):
            raise FactorTableMigrationError(f"catalog has a non-object factor row: {resolved}")
        factor_id = str(raw.get("factor_id", "")).strip()
        output_column = str(raw.get("output_column") or factor_id).strip()
        primary_family = str(raw.get("primary_family", "")).strip()
        contract_hash = str(raw.get("contract_hash", "")).strip().lower()
        factor_version = str(raw.get("factor_version", "")).strip()
        if not all((factor_id, output_column, primary_family, contract_hash, factor_version)):
            raise FactorTableMigrationError(f"catalog has an incomplete factor identity: {raw!r}")
        if factor_id in rows or output_column in output_columns:
            raise FactorTableMigrationError("catalog factor_id/output_column identity is not unique")
        try:
            # Reuse the canonical service's SHA-256 identity validation.
            FactorColumnContract(
                factor_id=factor_id,
                factor_version=factor_version,
                contract_hash=contract_hash,
                output_column=output_column,
            )
        except Exception as exc:
            raise FactorTableMigrationError(f"catalog factor has invalid storage contract: {factor_id}") from exc
        rows[factor_id] = CatalogFactor(
            factor_id=factor_id,
            output_column=output_column,
            primary_family=primary_family,
            contract_hash=contract_hash,
            factor_version=factor_version,
        )
        output_columns.add(output_column)
    if payload.get("factor_count") != len(rows):
        raise FactorTableMigrationError("catalog factor_count differs from unique factor rows")
    return resolved, rows


def _profile_output_columns(path: str | Path, *, catalog: Mapping[str, CatalogFactor]) -> tuple[str, ...]:
    resolved = Path(path).expanduser().resolve(strict=True)
    raw = json5.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("factors"), list):
        raise FactorTableMigrationError(f"profile lacks a factors list: {resolved}")
    output_columns: list[str] = []
    for item in raw["factors"]:
        if isinstance(item, str):
            factor_id = item
            output_override = None
        elif isinstance(item, Mapping):
            factor_id = str(item.get("factor_id") or item.get("name") or "").strip()
            output_override = item.get("output_col", item.get("output_column"))
        else:
            raise FactorTableMigrationError(f"profile has unsupported factor entry: {resolved}")
        factor = catalog.get(factor_id)
        if factor is None:
            raise FactorTableMigrationError(f"profile references a factor absent from Catalog: {factor_id!r}")
        output_column = str(output_override or factor.output_column).strip()
        if output_column != factor.output_column:
            raise FactorTableMigrationError(
                f"profile output column differs from Catalog factor contract: {factor_id!r}"
            )
        output_columns.append(output_column)
    if len(output_columns) != len(set(output_columns)):
        raise FactorTableMigrationError(f"profile output columns are not unique: {resolved}")
    return tuple(output_columns)


def _release_output_columns(path: str | Path, *, catalog: Mapping[str, CatalogFactor]) -> tuple[str, ...]:
    """Resolve the live release's exact physical output order from Catalog."""

    resolved = Path(path).expanduser().resolve(strict=True)
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("instances"), list):
        raise FactorTableMigrationError(f"live release lacks an instances list: {resolved}")
    output_columns: list[str] = []
    seen_factor_ids: set[str] = set()
    for item in raw["instances"]:
        if not isinstance(item, Mapping):
            raise FactorTableMigrationError(f"live release has a non-object instance: {resolved}")
        factor_id = str(item.get("factor_id", "")).strip()
        factor = catalog.get(factor_id)
        if factor is None or factor_id in seen_factor_ids:
            raise FactorTableMigrationError(f"live release has an invalid/duplicate Catalog factor: {factor_id!r}")
        seen_factor_ids.add(factor_id)
        if str(item.get("contract_hash", "")).strip().lower() != factor.contract_hash:
            raise FactorTableMigrationError(f"live release contract hash differs from Catalog: {factor_id!r}")
        output_override = item.get("output_col", item.get("output_column"))
        output_column = str(output_override or factor.output_column).strip()
        if output_column != factor.output_column:
            raise FactorTableMigrationError(
                f"live release output column differs from Catalog factor contract: {factor_id!r}"
            )
        output_columns.append(output_column)
    if raw.get("factor_count") != len(output_columns):
        raise FactorTableMigrationError(f"live release factor_count mismatch: {resolved}")
    return tuple(output_columns)


def _read_optional_manifest(path: str | Path | None) -> tuple[Path | None, Mapping[str, Any] | None]:
    if path is None:
        return None, None
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise FactorTableMigrationError(f"source manifest must be a JSON object: {resolved}")
    return resolved, dict(payload)


def _validate_experiment_source_manifest(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    factor_root: Path,
    profile_path: Path,
    profile_factor_ids: Sequence[str],
    selected_days: Sequence[date],
) -> None:
    if str(manifest.get("execution_status", "")) != "completed_rust88_backfill":
        raise FactorTableMigrationError("experiment manifest is not a completed Rust88 backfill")
    if manifest.get("model_training_ready") is not True:
        raise FactorTableMigrationError("experiment manifest is not model_training_ready")
    if not _same_path(str(manifest.get("factor_root", "")), factor_root):
        raise FactorTableMigrationError("experiment manifest factor_root does not bind the selected source FactorStore")
    if manifest.get("factor_count") != len(profile_factor_ids) or manifest.get("factors") != list(profile_factor_ids):
        raise FactorTableMigrationError("experiment manifest factor membership/order differs from profile source")
    time_contract = manifest.get("time_contract")
    if not isinstance(time_contract, Mapping) or time_contract.get("panel_name") != "T1430":
        raise FactorTableMigrationError("experiment manifest does not attest T1430")
    range_contract = manifest.get("range_contract")
    if not isinstance(range_contract, Mapping):
        raise FactorTableMigrationError("experiment manifest lacks a range_contract")
    try:
        range_start = datetime.strptime(str(range_contract.get("start", "")), "%Y-%m-%d").date()
        range_end = datetime.strptime(str(range_contract.get("end", "")), "%Y-%m-%d").date()
    except ValueError as exc:
        raise FactorTableMigrationError("experiment manifest range_contract is not YYYY-MM-DD") from exc
    if range_start > range_end or any(day < range_start or day > range_end for day in selected_days):
        raise FactorTableMigrationError("experiment manifest range_contract does not cover selected source days")
    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise FactorTableMigrationError("experiment manifest lacks profile attestation")
    frozen_profile_sha = str(profile.get("profile_sha256", "")).strip().lower()
    if len(frozen_profile_sha) != 64 or any(char not in "0123456789abcdef" for char in frozen_profile_sha):
        raise FactorTableMigrationError("experiment manifest profile_sha256 is missing/invalid")
    frozen_specs_sha = str(profile.get("specs_sha256", "")).strip().lower()
    if len(frozen_specs_sha) != 64 or any(char not in "0123456789abcdef" for char in frozen_specs_sha):
        raise FactorTableMigrationError("experiment manifest profile specs_sha256 is missing")
    # The R88 materialisation is historical.  A raw current profile file hash
    # may legitimately drift after the frozen manifest; do not re-label those
    # values as current.  The immutable manifest's profile/spec hashes remain
    # target provenance, while ordered factor identity is checked above.
    completion = manifest.get("completion")
    if not isinstance(completion, Mapping) or completion.get("full_completion") is not True:
        raise FactorTableMigrationError("experiment manifest completion is not full")
    completed_days = completion.get("r88_factor_store_days")
    if not isinstance(completed_days, list):
        raise FactorTableMigrationError("experiment manifest lacks completed FactorStore day evidence")
    completed = {str(item) for item in completed_days}
    missing = [day.isoformat() for day in selected_days if day.isoformat() not in completed]
    if missing:
        raise FactorTableMigrationError(
            "experiment manifest does not attest selected source days: " + ", ".join(missing[:10])
        )
    source_inputs = manifest.get("source_inputs")
    if source_inputs is not None and not isinstance(source_inputs, Mapping):
        raise FactorTableMigrationError("experiment manifest source_inputs must be an object when present")


def _validate_catalog_history_source_manifest(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    factor_root: Path,
    source: planner.StoreScan,
    selected_days: Sequence[date],
) -> None:
    if not _same_path(str(manifest.get("output_root", "")), factor_root):
        raise FactorTableMigrationError("catalog-history merge manifest output_root does not bind selected FactorStore")
    if manifest.get("panel_name") != "T1430" or manifest.get("output_columns") != list(source.factor_ids):
        raise FactorTableMigrationError("catalog-history merge manifest panel/output columns differ from source")
    output_files = manifest.get("output_files")
    if not isinstance(output_files, list):
        raise FactorTableMigrationError("catalog-history merge manifest lacks output_files")
    by_day: dict[str, Mapping[str, Any]] = {}
    for item in output_files:
        if not isinstance(item, Mapping):
            raise FactorTableMigrationError("catalog-history merge manifest has invalid output_files entry")
        key = str(item.get("trade_day", ""))
        if key in by_day:
            raise FactorTableMigrationError("catalog-history merge manifest has duplicate output file day")
        by_day[key] = item
    for day in selected_days:
        item = by_day.get(day.isoformat())
        path = source.paths_by_day[day]
        if item is None:
            raise FactorTableMigrationError(f"catalog-history manifest lacks selected day {day.isoformat()}")
        expected_relative = path.relative_to(factor_root).as_posix()
        if item.get("relative_path") != expected_relative:
            raise FactorTableMigrationError(f"catalog-history manifest path mismatch for {day.isoformat()}")
        if str(item.get("sha256", "")).lower() != _sha256_file(path):
            raise FactorTableMigrationError(f"catalog-history manifest hash mismatch for {day.isoformat()}")
        if int(item.get("row_count", -1)) != int(source.row_counts_by_day[day]):
            raise FactorTableMigrationError(f"catalog-history manifest row count mismatch for {day.isoformat()}")


def _find_supplement_manifest(
    source: planner.StoreScan,
    *,
    explicit_path: str | Path | None,
) -> tuple[Path | None, Mapping[str, Any] | None]:
    if explicit_path is not None:
        explicit_resolved, explicit_payload = _read_optional_manifest(explicit_path)
        assert explicit_resolved is not None and explicit_payload is not None
        matches = [(explicit_resolved, explicit_payload)]
    else:
        matches = planner._find_supplement_manifests(source.root, store=source)
    if not matches:
        raise FactorTableMigrationError("supplement source has no execution manifest matching its FactorStore")
    if len(source.days) != 1:
        raise FactorTableMigrationError("supplement migration currently requires exactly one source day")
    day = source.days[0]
    source_path = source.paths_by_day[day]
    source_sha = _sha256_file(source_path)
    source_frame = pd.read_parquet(source_path)
    if not isinstance(source_frame.index, pd.MultiIndex) or tuple(source_frame.index.names) != ("dt", "code"):
        raise FactorTableMigrationError("supplement source does not preserve a (dt, code) index")
    index_csv = source_frame.index.to_frame(index=False).astype(str).to_csv(index=False, lineterminator="\n")
    index_sha = hashlib.sha256(index_csv.encode("utf-8")).hexdigest()
    column_order_sha = hashlib.sha256(
        json.dumps(list(source.factor_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_ids = set(source.factor_ids)
    validated: list[tuple[Path, Mapping[str, Any]]] = []
    for path, payload in matches:
        factor_store = payload.get("factor_store")
        hashes = payload.get("factor_contract_hashes")
        if not isinstance(factor_store, Mapping) or not isinstance(hashes, Mapping):
            continue
        if payload.get("complete") is not True or payload.get("integrity_passed") is not True:
            continue
        if str(payload.get("status", "")) not in {"completed", "completed_with_coverage_gaps"}:
            continue
        if not _same_path(str(factor_store.get("path", "")), source_path):
            continue
        if str(factor_store.get("sha256", "")).lower() != source_sha:
            continue
        if str(factor_store.get("index_sha256", "")).lower() != index_sha:
            continue
        if str(factor_store.get("column_order_sha256", "")).lower() != column_order_sha:
            continue
        columns = factor_store.get("columns")
        if columns != list(source.factor_ids) or set(hashes) != expected_ids:
            continue
        validated.append((path, payload))
    if not validated:
        raise FactorTableMigrationError(
            "supplement source has no completed/integrity-passed manifest whose file, index, columns, and "
            f"historical contracts attest the source; total_candidates={len(matches)}"
        )
    hashes = {str(payload.get("factor_contract_hashes_sha256", "")).lower() for _path, payload in validated}
    if len(hashes) != 1:
        raise FactorTableMigrationError(
            "supplement manifests attest different per-factor historical contract maps; require explicit manifest"
        )
    # The latest completed attempt is authoritative when it attests exactly the
    # same source file/index/columns/contracts.  Unlike a generic "completed"
    # preference it retains `completed_with_coverage_gaps` and its all-NaN
    # evidence for target provenance.
    validated.sort(
        key=lambda item: (
            str(item[1].get("attempt_id", "")),
            item[0].as_posix(),
        )
    )
    return validated[-1]


def _validate_source_output_contract(
    source: planner.StoreScan,
    *,
    expected_columns: Sequence[str],
    label: str,
) -> None:
    actual = tuple(source.factor_ids)
    expected = tuple(expected_columns)
    if actual != expected:
        raise FactorTableMigrationError(
            f"{label} physical FactorStore columns do not exactly match declared Catalog/release/profile order: "
            f"expected_count={len(expected)} actual_count={len(actual)}"
        )


def _selection(
    *,
    explicit_days: Sequence[date] | None,
    smoke_route_days: bool,
) -> tuple[date, ...] | None:
    if explicit_days and smoke_route_days:
        raise FactorTableMigrationError("--days and --smoke-route-days are mutually exclusive")
    if smoke_route_days:
        return ROUTE_SMOKE_DAYS
    if explicit_days:
        return tuple(sorted(set(explicit_days)))
    return None


def _selected(source_days: Sequence[date], requested: tuple[date, ...] | None) -> tuple[date, ...]:
    if requested is None:
        return tuple(source_days)
    return tuple(day for day in source_days if day in set(requested))


def _planner_blockers_allowed(plan: Mapping[str, Any]) -> tuple[bool, list[str]]:
    readiness = plan.get("migration_readiness")
    blockers = list(readiness.get("blocking_reasons", [])) if isinstance(readiness, Mapping) else ["missing_readiness"]
    allowed = {"same_name_catalog_table_sources_have_index_or_value_conflicts"}
    return all(item in allowed for item in blockers), blockers


def _compare_by_pair(plan: Mapping[str, Any], left: str, right: str) -> Mapping[str, Any]:
    values = plan.get("source_comparisons")
    if not isinstance(values, list):
        raise FactorTableMigrationError("planner result lacks source_comparisons")
    for item in values:
        if isinstance(item, Mapping) and item.get("left_source") == left and item.get("right_source") == right:
            return item
    raise FactorTableMigrationError(f"planner result lacks comparison {left!r} -> {right!r}")


def _assert_ownership_policy_is_safe(
    *,
    planner_plan: Mapping[str, Any],
    live_ids: set[str],
    history_ids: set[str],
    supplement_ids: set[str],
    catalog_ids: set[str],
) -> bool:
    """Accept only the documented non-coalescing source ownership exception."""

    allowed_blockers, blockers = _planner_blockers_allowed(planner_plan)
    if not allowed_blockers:
        raise FactorTableMigrationError(
            "migration plan has blockers other than the explicit route-ownership audit: " + ", ".join(blockers)
        )
    if not history_ids.issubset(catalog_ids) or not supplement_ids.issubset(catalog_ids):
        raise FactorTableMigrationError("library source contains an unregistered Catalog factor")
    if live_ids & supplement_ids:
        raise FactorTableMigrationError("live and supplement factor IDs overlap; disjoint union is not safe")
    if (live_ids | supplement_ids) != catalog_ids:
        raise FactorTableMigrationError("live/supplement union does not exactly cover the Catalog")

    comparison = _compare_by_pair(planner_plan, "live_input", "catalog_history_input")
    value = comparison.get("value")
    if not isinstance(value, Mapping):
        raise FactorTableMigrationError("live/catalog-history collision audit lacks value evidence")
    if int(value.get("finite_value_conflict_cell_count", -1)) != 0:
        raise FactorTableMigrationError(
            "live/catalog-history same-name fields have finite value conflicts; date ownership cannot resolve this"
        )
    # An index/NaN-mask mismatch is why values must not be coalesced.  The
    # route holds catalog-history as the only owner on its source dates.
    shared = set(live_ids) & set(history_ids)
    if not shared:
        raise FactorTableMigrationError("expected live/catalog-history shared columns are absent")
    return bool(blockers)


def build_migration_plan(
    *,
    catalog_path: str | Path,
    live_release_path: str | Path,
    live_profile_path: str | Path,
    experiment_profile_path: str | Path,
    live_root: str | Path,
    experiment_root: str | Path,
    catalog_history_root: str | Path,
    supplement_root: str | Path,
    experiment_manifest_path: str | Path | None = None,
    catalog_history_manifest_path: str | Path | None = None,
    supplement_manifest_path: str | Path | None = None,
    target_root: str | Path = CANONICAL_FACTOR_STORE_ROOT,
    days: Sequence[date] | None = None,
    smoke_route_days: bool = False,
) -> MigrationPlan:
    """Build an immutable, read-only three-table migration plan.

    This always runs the existing value/index conflict audit.  The return value
    contains paths and contracts only; it does not read factors into memory or
    create any target directory.
    """

    planner_plan = planner.build_plan(
        catalog_path=catalog_path,
        live_release_path=live_release_path,
        live_profile_path=live_profile_path,
        experiment_profile_path=experiment_profile_path,
        live_root=live_root,
        experiment_root=experiment_root,
        catalog_history_root=catalog_history_root,
        supplement_root=supplement_root,
        experiment_manifest_path=experiment_manifest_path,
        catalog_history_manifest_path=catalog_history_manifest_path,
        supplement_manifest_path=supplement_manifest_path,
        value_conflict_check=True,
    )
    catalog_resolved, catalog = _parse_catalog(catalog_path)
    requested_days = _selection(explicit_days=days, smoke_route_days=smoke_route_days)
    live = planner.scan_factor_store(live_root, name="live_input")
    experiment = planner.scan_factor_store(experiment_root, name="experiment_input")
    catalog_history = planner.scan_factor_store(catalog_history_root, name="catalog_history_input")
    supplement = planner.scan_factor_store(supplement_root, name="supplement_input")

    release_output_columns = _release_output_columns(live_release_path, catalog=catalog)
    live_profile_columns = _profile_output_columns(live_profile_path, catalog=catalog)
    experiment_profile_columns = _profile_output_columns(experiment_profile_path, catalog=catalog)
    if release_output_columns != live_profile_columns:
        raise FactorTableMigrationError("live release output order differs from its active profile")
    _validate_source_output_contract(live, expected_columns=release_output_columns, label="live")
    _validate_source_output_contract(experiment, expected_columns=experiment_profile_columns, label="experiment")

    catalog_columns = {factor.output_column for factor in catalog.values()}
    for label, source in (("catalog-history", catalog_history), ("supplement", supplement)):
        unknown = sorted(set(source.factor_ids) - catalog_columns)
        if unknown:
            raise FactorTableMigrationError(f"{label} source has output columns absent from Catalog: {unknown[:10]!r}")

    # The planner scans output column names.  Convert back to canonical factor
    # IDs without assuming factor_id == output_column.
    catalog_by_output = {factor.output_column: factor for factor in catalog.values()}
    live_ids = {catalog_by_output[column].factor_id for column in live.factor_ids}
    experiment_ids = {catalog_by_output[column].factor_id for column in experiment.factor_ids}
    history_ids = {catalog_by_output[column].factor_id for column in catalog_history.factor_ids}
    supplement_ids = {catalog_by_output[column].factor_id for column in supplement.factor_ids}
    catalog_ids = set(catalog)
    ownership_ack_required = _assert_ownership_policy_is_safe(
        planner_plan=planner_plan,
        live_ids=live_ids,
        history_ids=history_ids,
        supplement_ids=supplement_ids,
        catalog_ids=catalog_ids,
    )

    experiment_manifest_resolved, experiment_manifest = _read_optional_manifest(experiment_manifest_path)
    history_manifest_resolved, history_manifest = _read_optional_manifest(catalog_history_manifest_path)
    if experiment_manifest_resolved is None or experiment_manifest is None:
        raise FactorTableMigrationError("experiment source requires its immutable historical backfill manifest")
    if history_manifest_resolved is None or history_manifest is None:
        raise FactorTableMigrationError("catalog-history source requires its immutable merge manifest")
    supplement_manifest_resolved, supplement_manifest = _find_supplement_manifest(
        supplement,
        explicit_path=supplement_manifest_path,
    )
    if supplement_manifest is None:
        raise FactorTableMigrationError("supplement source has no historical contract attestation manifest")
    source_hashes = supplement_manifest.get("factor_contract_hashes")
    if not isinstance(source_hashes, Mapping):
        raise FactorTableMigrationError("supplement manifest lacks factor_contract_hashes")
    expected_supplement_outputs = tuple(supplement.factor_ids)
    if set(source_hashes) != {catalog_by_output[column].factor_id for column in expected_supplement_outputs}:
        raise FactorTableMigrationError("supplement manifest factor_contract_hashes do not exactly match source columns")

    sources = {
        "live_input": SourceRuntime(
            name="live_input",
            scan=live,
            manifest_path=Path(live_release_path).expanduser().resolve(strict=True),
            manifest=None,
            manifest_sha256=_sha256_file(Path(live_release_path).expanduser().resolve(strict=True)),
            contract_status="active_release_current_catalog_aligned",
            partition_sha256={},
        ),
        "experiment_input": SourceRuntime(
            name="experiment_input",
            scan=experiment,
            manifest_path=experiment_manifest_resolved,
            manifest=experiment_manifest,
            manifest_sha256=_sha256_file(experiment_manifest_resolved) if experiment_manifest_resolved else None,
            contract_status="experiment_profile_bound_historical_materialization",
            partition_sha256={},
        ),
        "catalog_history_input": SourceRuntime(
            name="catalog_history_input",
            scan=catalog_history,
            manifest_path=history_manifest_resolved,
            manifest=history_manifest,
            manifest_sha256=_sha256_file(history_manifest_resolved) if history_manifest_resolved else None,
            contract_status="historical_source_manifest_without_per_factor_contract_hashes",
            partition_sha256={},
        ),
        "supplement_input": SourceRuntime(
            name="supplement_input",
            scan=supplement,
            manifest_path=supplement_manifest_resolved,
            manifest=supplement_manifest,
            manifest_sha256=_sha256_file(supplement_manifest_resolved) if supplement_manifest_resolved else None,
            contract_status="historical_source_per_factor_contract_hashes_preserved",
            partition_sha256={},
        ),
    }

    live_only_ids = tuple(
        catalog_by_output[column].factor_id for column in live.factor_ids if catalog_by_output[column].factor_id not in history_ids
    )
    shared_live_ids = tuple(
        catalog_by_output[column].factor_id for column in live.factor_ids if catalog_by_output[column].factor_id in history_ids
    )
    history_days = set(catalog_history.days)
    candidate_routes = (
        Route(
            route_id="live_wide",
            target=FactorTableKind.LIVE,
            source="live_input",
            factor_ids=tuple(catalog_by_output[column].factor_id for column in live.factor_ids),
            days=_selected(live.days, requested_days),
            ownership_rule="live_release_owns_all_live_table_dates",
        ),
        Route(
            route_id="experiment_wide",
            target=FactorTableKind.EXPERIMENT,
            source="experiment_input",
            factor_ids=tuple(catalog_by_output[column].factor_id for column in experiment.factor_ids),
            days=_selected(experiment.days, requested_days),
            ownership_rule="experiment_profile_owns_all_experiment_table_dates",
        ),
        Route(
            route_id="library_catalog_history",
            target=FactorTableKind.FACTOR_LIBRARY,
            source="catalog_history_input",
            factor_ids=tuple(catalog_by_output[column].factor_id for column in catalog_history.factor_ids),
            days=_selected(catalog_history.days, requested_days),
            ownership_rule="catalog_history_owns_all_its_dates_and_shared_live_columns",
        ),
        Route(
            route_id="library_supplement",
            target=FactorTableKind.FACTOR_LIBRARY,
            source="supplement_input",
            factor_ids=tuple(catalog_by_output[column].factor_id for column in supplement.factor_ids),
            days=_selected(supplement.days, requested_days),
            ownership_rule="supplement_owns_its_disjoint_columns_on_its_manifested_dates",
        ),
        Route(
            route_id="library_live_only",
            target=FactorTableKind.FACTOR_LIBRARY,
            source="live_input",
            factor_ids=live_only_ids,
            days=_selected(live.days, requested_days),
            ownership_rule="live_owns_columns_absent_from_catalog_history_on_all_live_dates",
        ),
        Route(
            route_id="library_live_shared_outside_history",
            target=FactorTableKind.FACTOR_LIBRARY,
            source="live_input",
            factor_ids=shared_live_ids,
            days=tuple(day for day in _selected(live.days, requested_days) if day not in history_days),
            ownership_rule="live_owns_shared_columns_only_outside_catalog_history_dates",
        ),
    )
    routes = tuple(route for route in candidate_routes if route.factor_ids and route.days)
    if not routes:
        raise FactorTableMigrationError("selected scope does not intersect any approved source date")
    if smoke_route_days:
        _assert_smoke_route_coverage(routes)
    experiment_selected_days = tuple(
        day for route in routes if route.route_id == "experiment_wide" for day in route.days
    )
    history_selected_days = tuple(
        day for route in routes if route.route_id == "library_catalog_history" for day in route.days
    )
    _validate_experiment_source_manifest(
        manifest_path=experiment_manifest_resolved,
        manifest=experiment_manifest,
        factor_root=experiment.root,
        profile_path=Path(experiment_profile_path).expanduser().resolve(strict=True),
        profile_factor_ids=tuple(catalog_by_output[column].factor_id for column in experiment.factor_ids),
        selected_days=experiment_selected_days,
    )
    _validate_catalog_history_source_manifest(
        manifest_path=history_manifest_resolved,
        manifest=history_manifest,
        factor_root=catalog_history.root,
        source=catalog_history,
        selected_days=history_selected_days,
    )
    target_resolved = _resolved(target_root)
    source_roots = {_resolved(runtime.scan.root) for runtime in sources.values()}
    if any(_paths_overlap(target_resolved, source_root) for source_root in source_roots):
        raise FactorTableMigrationError(
            "target root must not equal, contain, or be contained by a read-only source FactorStore"
        )
    source_days_by_name: dict[str, set[date]] = defaultdict(set)
    for route in routes:
        source_days_by_name[route.source].update(route.days)
    sources = {
        name: replace(
            runtime,
            partition_sha256={
                day: _sha256_file(runtime.scan.paths_by_day[day])
                for day in sorted(source_days_by_name.get(name, set()))
            },
        )
        for name, runtime in sources.items()
    }
    governing_artifacts = {
        "catalog": (catalog_resolved, _sha256_file(catalog_resolved)),
        "live_release": (
            Path(live_release_path).expanduser().resolve(strict=True),
            _sha256_file(Path(live_release_path).expanduser().resolve(strict=True)),
        ),
        "live_profile": (
            Path(live_profile_path).expanduser().resolve(strict=True),
            _sha256_file(Path(live_profile_path).expanduser().resolve(strict=True)),
        ),
        "experiment_profile": (
            Path(experiment_profile_path).expanduser().resolve(strict=True),
            _sha256_file(Path(experiment_profile_path).expanduser().resolve(strict=True)),
        ),
        "experiment_manifest": (experiment_manifest_resolved, _sha256_file(experiment_manifest_resolved)),
        "catalog_history_manifest": (history_manifest_resolved, _sha256_file(history_manifest_resolved)),
        "supplement_manifest": (supplement_manifest_resolved, _sha256_file(supplement_manifest_resolved)),
    }
    return MigrationPlan(
        target_root=target_resolved,
        planner_plan=planner_plan,
        planner_plan_sha256=_sha256_json(planner_plan),
        catalog_path=catalog_resolved,
        catalog_sha256=_sha256_file(catalog_resolved),
        catalog=catalog,
        governing_artifacts=governing_artifacts,
        sources=sources,
        routes=routes,
        selected_days=requested_days,
        smoke_route_days=smoke_route_days,
        ownership_ack_required=ownership_ack_required,
    )


def _assert_smoke_route_coverage(routes: Sequence[Route]) -> None:
    """Ensure the fixed smoke selection covers every meaningful source route."""

    by_id = {route.route_id: route for route in routes}
    required = {
        "live_wide": date(2024, 1, 3),
        "experiment_wide": date(2024, 1, 3),
        "library_catalog_history": date(2025, 1, 2),
        "library_supplement": date(2026, 8, 25),
        "library_live_only": date(2026, 8, 25),
        "library_live_shared_outside_history": date(2026, 8, 25),
    }
    missing: list[str] = []
    for route_id, day in required.items():
        route = by_id.get(route_id)
        if route is None or day not in route.days:
            missing.append(f"{route_id}@{day.isoformat()}")
    if missing:
        raise FactorTableMigrationError(
            "route smoke selection does not exercise all required routes: " + ", ".join(missing)
        )


def _materialization_contract_fields(
    plan: MigrationPlan,
    *,
    source_name: str,
    factor_id: str,
) -> dict[str, str | None]:
    """Separate current Catalog identity from the source's materialisation contract."""

    runtime = plan.sources[source_name]
    if source_name == "live_input":
        return {
            "materialization_contract_origin": "current_catalog",
            "materialization_contract_hash": None,
            "materialization_source_attestation_sha256": None,
        }
    if runtime.manifest_path is None:
        raise FactorTableMigrationError(
            f"historical source {source_name!r} has no immutable source manifest attestation"
        )
    attestation_sha = _sha256_file(runtime.manifest_path)
    if source_name == "supplement_input":
        if runtime.manifest is None or not isinstance(runtime.manifest.get("factor_contract_hashes"), Mapping):
            raise FactorTableMigrationError("supplement source has no per-factor historical contract hash map")
        source_hash = str(runtime.manifest["factor_contract_hashes"].get(factor_id, "")).strip().lower()
        if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
            raise FactorTableMigrationError(
                f"supplement source has no valid historical contract hash for {factor_id!r}"
            )
        return {
            "materialization_contract_origin": "source_attested",
            "materialization_contract_hash": source_hash,
            "materialization_source_attestation_sha256": attestation_sha,
        }
    # R88 and catalog-history have immutable source/profile manifests but no
    # per-factor materialisation hash.  They remain explicitly unversioned;
    # their current Catalog identity is never a claim about historical values.
    return {
        "materialization_contract_origin": "source_manifest_unversioned",
        "materialization_contract_hash": None,
        "materialization_source_attestation_sha256": attestation_sha,
    }


def _contract_for_factor_ids(
    plan: MigrationPlan,
    factor_ids: Iterable[str],
    *,
    preserve_order: bool,
    source_by_factor_id: Mapping[str, str],
) -> FactorTableContract:
    ids = list(factor_ids)
    if not ids:
        raise FactorTableMigrationError("cannot create an empty FactorTableContract")
    if not preserve_order:
        catalog_position = {factor_id: position for position, factor_id in enumerate(plan.catalog)}
        ids.sort(key=lambda factor_id: catalog_position[factor_id])
    columns: list[FactorColumnContract] = []
    for factor_id in ids:
        factor = plan.catalog.get(factor_id)
        if factor is None:
            raise FactorTableMigrationError(f"route refers to a factor absent from Catalog: {factor_id!r}")
        columns.append(
            FactorColumnContract(
                factor_id=factor.factor_id,
                factor_version=factor.factor_version,
                contract_hash=factor.contract_hash,
                output_column=factor.output_column,
                **_materialization_contract_fields(
                    plan,
                    source_name=source_by_factor_id[factor_id],
                    factor_id=factor_id,
                ),
            )
        )
    return FactorTableContract(tuple(columns))


def _frame_for_source_fragment(
    plan: MigrationPlan,
    *,
    source_cache: SourceFrameCache,
    source_name: str,
    day: date,
    factor_ids: Sequence[str],
) -> pd.DataFrame:
    output_columns = [plan.catalog[factor_id].output_column for factor_id in factor_ids]
    runtime = plan.sources[source_name]
    source_columns = set(runtime.scan.factor_ids)
    missing = sorted(set(output_columns) - source_columns)
    if missing:
        raise FactorTableMigrationError(
            f"source {source_name!r} lacks route columns for {day.isoformat()}: {missing[:10]!r}"
        )
    frame = source_cache.frame(source_name, day)
    return frame.loc[:, output_columns]


def _historical_contract_evidence(
    plan: MigrationPlan,
    *,
    source_name: str,
    factor_ids: Sequence[str],
) -> Mapping[str, Any]:
    runtime = plan.sources[source_name]
    manifest_path = runtime.manifest_path
    evidence: dict[str, Any] = {
        "contract_status": runtime.contract_status,
        "manifest": (
            {"path": _path_text(manifest_path), "sha256": runtime.manifest_sha256}
            if manifest_path is not None
            else None
        ),
    }
    if source_name != "supplement_input":
        if source_name == "experiment_input" and runtime.manifest is not None:
            profile = runtime.manifest.get("profile")
            evidence["historical_profile"] = dict(profile) if isinstance(profile, Mapping) else None
            evidence["source_execution_status"] = runtime.manifest.get("execution_status")
            evidence["source_model_training_ready"] = runtime.manifest.get("model_training_ready")
        return evidence
    if runtime.manifest is None:
        raise FactorTableMigrationError("supplement has no manifest payload for historical contract provenance")
    hashes = runtime.manifest.get("factor_contract_hashes")
    if not isinstance(hashes, Mapping):
        raise FactorTableMigrationError("supplement manifest lacks factor_contract_hashes")
    historical: dict[str, str] = {}
    for factor_id in factor_ids:
        value = str(hashes.get(factor_id, "")).strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise FactorTableMigrationError(
                f"supplement historical contract hash is invalid/missing for {factor_id!r}"
            )
        historical[factor_id] = value
    evidence["historical_source_factor_contract_hashes"] = historical
    evidence["current_catalog_hash_match_count"] = sum(
        historical[factor_id] == plan.catalog[factor_id].contract_hash for factor_id in factor_ids
    )
    evidence["current_catalog_hash_compared_count"] = len(factor_ids)
    coverage = runtime.manifest.get("coverage_quality")
    evidence["source_completion"] = {
        "status": runtime.manifest.get("status"),
        "complete": runtime.manifest.get("complete"),
        "integrity_passed": runtime.manifest.get("integrity_passed"),
        "coverage_quality": dict(coverage) if isinstance(coverage, Mapping) else None,
    }
    return evidence


def _fragment_evidence(
    plan: MigrationPlan,
    *,
    source_name: str,
    day: date,
    factor_ids: Sequence[str],
    source_cache: SourceFrameCache,
) -> dict[str, Any]:
    runtime = plan.sources[source_name]
    snapshot = source_cache.snapshot(source_name, day)
    path = snapshot.path
    return {
        "source": source_name,
        "source_root": _path_text(runtime.scan.root),
        "source_partition": {
            "path": _path_text(path),
            "sha256": snapshot.sha256,
            "trade_day": day.isoformat(),
        },
        "factor_ids": list(factor_ids),
        "output_columns": [plan.catalog[factor_id].output_column for factor_id in factor_ids],
        "historical_contract_evidence": _historical_contract_evidence(
            plan,
            source_name=source_name,
            factor_ids=factor_ids,
        ),
    }


def _merge_same_index_disjoint_fragments(
    fragments: Sequence[tuple[str, Sequence[str], pd.DataFrame]],
    *,
    contract: FactorTableContract,
    day: date,
    family: str | None,
) -> pd.DataFrame:
    """Combine only disjoint columns with precisely equal source rows.

    This is not an outer union and never fills a row from a second source.  A
    mismatched source index is an explicit migration conflict, because any
    row-level reconciliation would create a new unapproved data contract.
    """

    if not fragments:
        raise FactorTableMigrationError("cannot publish a factor-table day without source fragments")
    result: pd.DataFrame | None = None
    owners: dict[str, str] = {}
    for source_name, factor_ids, frame in fragments:
        columns = [str(column) for column in frame.columns]
        duplicate_columns = sorted(set(columns) & set(owners))
        if duplicate_columns:
            existing = owners[duplicate_columns[0]]
            raise FactorTableMigrationError(
                "same-name factor column would be coalesced in one canonical partition: "
                f"day={day.isoformat()} family={family!r} column={duplicate_columns[0]!r} "
                f"owners=({existing!r},{source_name!r})"
            )
        if result is None:
            result = frame.copy()
        else:
            if not result.index.equals(frame.index):
                raise FactorTableMigrationError(
                    "source (dt, code) indexes differ inside one target family/day; refusing row-level union: "
                    f"day={day.isoformat()} family={family!r} existing_rows={len(result)} "
                    f"{source_name}_rows={len(frame)}"
                )
            result = pd.concat([result, frame], axis=1)
        owners.update({column: source_name for column in columns})
        if [str(column) for column in frame.columns] != [
            # This uses the exact selected factor physical columns, not an
            # inferred relation between factor ID and output name.
            str(column) for column in frame.columns
        ]:  # pragma: no cover - local invariant retained for readability.
            raise AssertionError("unreachable fragment output-order guard")
    assert result is not None
    missing = sorted(set(contract.output_columns) - set(result.columns))
    extra = sorted(set(result.columns) - set(contract.output_columns))
    if missing or extra:
        raise FactorTableMigrationError(
            f"assembled target family/day columns differ from Catalog contract: missing={missing[:10]!r} extra={extra[:10]!r}"
        )
    return result.loc[:, list(contract.output_columns)].sort_index(kind="mergesort")


def _route_by_id(plan: MigrationPlan, route_id: str) -> Route | None:
    matches = [route for route in plan.routes if route.route_id == route_id]
    if len(matches) > 1:
        raise FactorTableMigrationError(f"duplicate migration route ID: {route_id}")
    return matches[0] if matches else None


def _library_fragments_for_day(plan: MigrationPlan, day: date) -> Mapping[str, list[tuple[str, tuple[str, ...]]]]:
    """Return source-owned factor IDs grouped by their Catalog family."""

    grouped: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for route in plan.routes:
        if route.target is not FactorTableKind.FACTOR_LIBRARY or day not in route.days:
            continue
        by_family: dict[str, list[str]] = defaultdict(list)
        for factor_id in route.factor_ids:
            by_family[plan.catalog[factor_id].primary_family].append(factor_id)
        for family, factor_ids in by_family.items():
            grouped[family].append((route.source, tuple(factor_ids)))
    # Detect source ownership collisions before any source file is opened.
    for family, fragments in grouped.items():
        owners: dict[str, str] = {}
        for source_name, factor_ids in fragments:
            for factor_id in factor_ids:
                previous = owners.setdefault(factor_id, source_name)
                if previous != source_name:
                    raise FactorTableMigrationError(
                        "date/factor ownership collision in factor library route: "
                        f"day={day.isoformat()} family={family!r} factor_id={factor_id!r} "
                        f"owners=({previous!r},{source_name!r})"
                    )
    return dict(sorted(grouped.items()))


def _operation_summary(plan: MigrationPlan) -> dict[str, Any]:
    wide: dict[str, int] = {}
    library_partitions = 0
    library_days = sorted(
        {day for route in plan.routes if route.target is FactorTableKind.FACTOR_LIBRARY for day in route.days}
    )
    for route in plan.routes:
        if route.target is not FactorTableKind.FACTOR_LIBRARY:
            wide[route.target.value] = wide.get(route.target.value, 0) + len(route.days)
    for day in library_days:
        library_partitions += len(_library_fragments_for_day(plan, day))
    return {
        "wide_table_day_operations": wide,
        "factor_library_family_day_operations": library_partitions,
        "factor_library_day_count": len(library_days),
    }


def _migration_id(plan: MigrationPlan, kind: FactorTableKind) -> str:
    target_routes = [
        {
            "route_id": route.route_id,
            "source": route.source,
            "factor_ids": list(route.factor_ids),
            "days": [day.isoformat() for day in route.days],
            "ownership_rule": route.ownership_rule,
        }
        for route in plan.routes
        if route.target is kind
    ]
    scope_hash = _sha256_json(
        {
            "planner_plan_sha256": plan.planner_plan_sha256,
            "table_id": kind.value,
            "scope": "full" if plan.selected_days is None else "partial",
            "selected_days": [day.isoformat() for day in plan.selected_days] if plan.selected_days else None,
            "target_routes": target_routes,
        }
    )
    return f"factor-table-migration-{kind.value}-{scope_hash[:20]}"


def _migration_attestation(
    plan: MigrationPlan,
    kind: FactorTableKind,
    *,
    expected_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    routes = [route for route in plan.routes if route.target is kind]
    source_names = sorted({route.source for route in routes})
    return {
        "schema_version": MIGRATION_SCHEMA,
        "migration_id": _migration_id(plan, kind),
        "scope": expected_coverage.get("scope"),
        "expected_coverage": dict(expected_coverage),
        "table_id": kind.value,
        "planner_plan_sha256": plan.planner_plan_sha256,
        "catalog": {"path": _path_text(plan.catalog_path), "sha256": plan.catalog_sha256},
        "selection": {
            "smoke_route_days": plan.smoke_route_days,
            "selected_days": [day.isoformat() for day in plan.selected_days] if plan.selected_days else None,
        },
        "ownership_policy": {
            "id": OWNERSHIP_POLICY_ID if kind is FactorTableKind.FACTOR_LIBRARY else "single_source_wide_table_v1",
            "same_name_values_coalesced": False,
        },
        "routes": [
            {
                "route_id": route.route_id,
                "source": route.source,
                "factor_ids_sha256": _sha256_json(list(route.factor_ids)),
                "factor_count": len(route.factor_ids),
                "days": [day.isoformat() for day in route.days],
                "ownership_rule": route.ownership_rule,
            }
            for route in routes
        ],
        "source_contract_statuses": {
            source_name: plan.sources[source_name].contract_status for source_name in source_names
        },
    }


def _wide_source_evidence(
    plan: MigrationPlan,
    *,
    route: Route,
    day: date,
    source_cache: SourceFrameCache,
) -> dict[str, Any]:
    return {
        "schema_version": MIGRATION_SCHEMA,
        "ownership_policy": "single_source_wide_table_v1",
        "route_id": route.route_id,
        "fragments": [
            _fragment_evidence(
                plan,
                source_name=route.source,
                day=day,
                factor_ids=route.factor_ids,
                source_cache=source_cache,
            )
        ],
    }


def _library_source_evidence(
    plan: MigrationPlan,
    *,
    day: date,
    family: str,
    fragments: Sequence[tuple[str, Sequence[str], pd.DataFrame]],
    source_cache: SourceFrameCache,
) -> dict[str, Any]:
    return {
        "schema_version": MIGRATION_SCHEMA,
        "ownership_policy": OWNERSHIP_POLICY_ID,
        "family": family,
        "same_name_values_coalesced": False,
        "fragments": [
            _fragment_evidence(
                plan,
                source_name=source_name,
                day=day,
                factor_ids=factor_ids,
                source_cache=source_cache,
            )
            for source_name, factor_ids, _frame in fragments
        ],
    }


def _table_counts_template() -> dict[str, int]:
    return {kind.value: 0 for kind in FactorTableKind}


def _assert_governing_artifacts_unchanged(plan: MigrationPlan) -> None:
    for name, (path, expected_hash) in plan.governing_artifacts.items():
        if _sha256_file(path) != expected_hash:
            raise FactorTableMigrationError(f"governing artifact changed after migration planning: {name} ({path})")


def _source_map_for_route(route: Route) -> dict[str, str]:
    return {factor_id: route.source for factor_id in route.factor_ids}


def _source_map_for_fragments(fragments: Sequence[tuple[str, Sequence[str], pd.DataFrame]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for source_name, factor_ids, _frame in fragments:
        for factor_id in factor_ids:
            previous = mapping.setdefault(factor_id, source_name)
            if previous != source_name:
                raise FactorTableMigrationError(
                    f"factor {factor_id!r} has multiple materialization sources in one target contract"
                )
    return mapping


def _expected_coverage(plan: MigrationPlan, kind: FactorTableKind) -> dict[str, Any]:
    routes = [route for route in plan.routes if route.target is kind]
    if not routes:
        raise FactorTableMigrationError(f"no selected migration routes for {kind.value}")
    if kind is FactorTableKind.FACTOR_LIBRARY:
        library_days = sorted({day for route in routes for day in route.days})
        family_operations = sum(len(_library_fragments_for_day(plan, day)) for day in library_days)
        operation_count = family_operations
    else:
        operation_count = sum(len(route.days) for route in routes)
    return {
        "scope": "full" if plan.selected_days is None else "partial",
        "operation_count": operation_count,
        "routes": [
            {
                "route_id": route.route_id,
                "source": route.source,
                "factor_ids_sha256": _sha256_json(list(route.factor_ids)),
                "days_sha256": _sha256_json([day.isoformat() for day in route.days]),
                "day_count": len(route.days),
            }
            for route in routes
        ],
        "governing_artifacts_sha256": _sha256_json(
            {
                name: {"path": _path_text(path), "sha256": sha256}
                for name, (path, sha256) in plan.governing_artifacts.items()
            }
        ),
    }


def _verify_wide_day(
    plan: MigrationPlan,
    *,
    store: CanonicalFactorStore,
    source_cache: SourceFrameCache,
    route: Route,
    day: date,
) -> None:
    source_map = _source_map_for_route(route)
    contract = _contract_for_factor_ids(
        plan,
        route.factor_ids,
        preserve_order=True,
        source_by_factor_id=source_map,
    )
    expected = _frame_for_source_fragment(
        plan,
        source_cache=source_cache,
        source_name=route.source,
        day=day,
        factor_ids=route.factor_ids,
    )
    actual = store.read_day(route.target, day, expected_contract=contract)
    _assert_exact_frame(actual, expected, label=f"{route.route_id}@{day.isoformat()}")
    manifest = _load_target_day_manifest(store, route.target, day, family=None)
    _assert_target_source_evidence(
        manifest,
        _wide_source_evidence(plan, route=route, day=day, source_cache=source_cache),
        label=f"{route.route_id}@{day.isoformat()}",
    )


def _verify_library_day(
    plan: MigrationPlan,
    *,
    store: CanonicalFactorStore,
    source_cache: SourceFrameCache,
    day: date,
) -> int:
    family_contracts: dict[str, FactorTableContract] = {}
    expected_by_family: dict[str, tuple[list[tuple[str, tuple[str, ...], pd.DataFrame]], pd.DataFrame]] = {}
    for family, fragment_specs in _library_fragments_for_day(plan, day).items():
        fragments: list[tuple[str, tuple[str, ...], pd.DataFrame]] = []
        factor_ids: list[str] = []
        for source_name, source_factor_ids in fragment_specs:
            frame = _frame_for_source_fragment(
                plan,
                source_cache=source_cache,
                source_name=source_name,
                day=day,
                factor_ids=source_factor_ids,
            )
            fragments.append((source_name, source_factor_ids, frame))
            factor_ids.extend(source_factor_ids)
        contract = _contract_for_factor_ids(
            plan,
            factor_ids,
            preserve_order=False,
            source_by_factor_id=_source_map_for_fragments(fragments),
        )
        expected = _merge_same_index_disjoint_fragments(
            fragments,
            contract=contract,
            day=day,
            family=family,
        )
        family_contracts[family] = contract
        expected_by_family[family] = (fragments, expected)
    # Verifies exact family set + all family hashes through the logical day
    # manifest before exposing any individual family frame.
    store.require_factor_library_day_published(day, families=sorted(family_contracts))
    for family, (fragments, expected) in expected_by_family.items():
        actual = store.read_day(
            FactorTableKind.FACTOR_LIBRARY,
            day,
            family=family,
            expected_contract=family_contracts[family],
        )
        _assert_exact_frame(actual, expected, label=f"factor_library/{family}@{day.isoformat()}")
        manifest = _load_target_day_manifest(store, FactorTableKind.FACTOR_LIBRARY, day, family=family)
        _assert_target_source_evidence(
            manifest,
            _library_source_evidence(
                plan,
                day=day,
                family=family,
                fragments=fragments,
                source_cache=source_cache,
            ),
            label=f"factor_library/{family}@{day.isoformat()}",
        )
    return len(family_contracts)


def execute_migration(
    plan: MigrationPlan,
    *,
    accept_date_owned_catalog_routing: bool,
    verify_readback: bool = True,
) -> MigrationResult:
    """Copy only approved source partitions to the canonical table root.

    The owner-facing acknowledgement is deliberately required even when the
    existing planner observed NaN/index differences between same-name catalog
    sources.  It authorises *only* the deterministic date ownership policy;
    it never permits value coalescing or a finite-value conflict.
    """

    if not isinstance(plan, MigrationPlan):
        raise TypeError("plan must be a MigrationPlan")
    if plan.ownership_ack_required and not accept_date_owned_catalog_routing:
        raise FactorTableMigrationError(
            "execution requires --accept-date-owned-catalog-routing because the planner detected "
            "same-name source index/NaN-mask differences; no values will be coalesced"
        )
    if not verify_readback:
        raise FactorTableMigrationError(
            "migration execution always requires semantic read-back verification; no unverified scope may be attested"
        )
    _assert_governing_artifacts_unchanged(plan)
    # Verify every selected source partition before even creating the target
    # root.  A stale/in-flight source therefore cannot leave a new target tree
    # behind merely by failing its first write.
    SourceFrameCache(plan).assert_unchanged()
    store = CanonicalFactorStore(plan.target_root)
    written = _table_counts_template()
    already_present = _table_counts_template()
    source_cache = SourceFrameCache(plan)
    with store.migration_lock():
        _assert_governing_artifacts_unchanged(plan)
        source_cache.assert_unchanged()
        wide_routes = [route for route in plan.routes if route.route_id in {"live_wide", "experiment_wide"}]
        all_days = sorted({day for route in plan.routes for day in route.days})
        for day in all_days:
            # Process every target that uses this source day before releasing
            # its immutable snapshot.  This bounds memory to one/few daily
            # frames and prevents the previous per-family reread pattern.
            for route in wide_routes:
                if day not in route.days:
                    continue
                contract = _contract_for_factor_ids(
                    plan,
                    route.factor_ids,
                    preserve_order=True,
                    source_by_factor_id=_source_map_for_route(route),
                )
                frame = _frame_for_source_fragment(
                    plan,
                    source_cache=source_cache,
                    source_name=route.source,
                    day=day,
                    factor_ids=route.factor_ids,
                )
                result = store.write_day(
                    route.target,
                    day,
                    frame,
                    contract=contract,
                    source_evidence=_wide_source_evidence(
                        plan, route=route, day=day, source_cache=source_cache
                    ),
                )
                (written if result.status == "written" else already_present)[route.target.value] += 1
                if result.status not in {"written", "already_present"}:
                    raise FactorTableMigrationError(f"unexpected canonical write status: {result.status!r}")
                _verify_wide_day(plan, store=store, source_cache=source_cache, route=route, day=day)

            family_fragments_for_day = _library_fragments_for_day(plan, day)
            if family_fragments_for_day:
                family_contracts: dict[str, FactorTableContract] = {}
                for family, fragment_specs in family_fragments_for_day.items():
                    fragments: list[tuple[str, tuple[str, ...], pd.DataFrame]] = []
                    factor_ids: list[str] = []
                    for source_name, source_factor_ids in fragment_specs:
                        fragments.append(
                            (
                                source_name,
                                source_factor_ids,
                                _frame_for_source_fragment(
                                    plan,
                                    source_cache=source_cache,
                                    source_name=source_name,
                                    day=day,
                                    factor_ids=source_factor_ids,
                                ),
                            )
                        )
                        factor_ids.extend(source_factor_ids)
                    contract = _contract_for_factor_ids(
                        plan,
                        factor_ids,
                        preserve_order=False,
                        source_by_factor_id=_source_map_for_fragments(fragments),
                    )
                    assembled = _merge_same_index_disjoint_fragments(
                        fragments, contract=contract, day=day, family=family
                    )
                    result = store.write_day(
                        FactorTableKind.FACTOR_LIBRARY,
                        day,
                        assembled,
                        contract=contract,
                        family=family,
                        source_evidence=_library_source_evidence(
                            plan, day=day, family=family, fragments=fragments, source_cache=source_cache
                        ),
                    )
                    (written if result.status == "written" else already_present)[FactorTableKind.FACTOR_LIBRARY.value] += 1
                    if result.status not in {"written", "already_present"}:
                        raise FactorTableMigrationError(f"unexpected canonical write status: {result.status!r}")
                    family_contracts[family] = contract
                store.publish_factor_library_day(day, family_contracts=family_contracts)
                _verify_library_day(plan, store=store, source_cache=source_cache, day=day)
            source_cache.release_day(day)

        source_cache.assert_unchanged()
        _assert_governing_artifacts_unchanged(plan)
        attestations: dict[str, str] = {}
        for kind in FactorTableKind:
            if not any(route.target is kind and route.days for route in plan.routes):
                continue
            coverage = _expected_coverage(plan, kind)
            status = store.record_migration_attestation(
                kind, _migration_attestation(plan, kind, expected_coverage=coverage)
            )
            if coverage["scope"] == "full":
                store.mark_migration_verified(kind, migration_id=_migration_id(plan, kind), coverage=coverage)
            else:
                store.mark_migration_partial(kind, migration_id=_migration_id(plan, kind), coverage=coverage)
            attestations[kind.value] = status
    return MigrationResult(
        target_root=plan.target_root,
        mode="execute",
        planner_plan_sha256=plan.planner_plan_sha256,
        selected_days=plan.selected_days,
        written_by_table=written,
        already_present_by_table=already_present,
        attestations=attestations,
    )


def _load_target_day_manifest(store: CanonicalFactorStore, kind: FactorTableKind, day: date, *, family: str | None) -> Mapping[str, Any]:
    paths = store.partition_paths(kind, day, family=family)
    try:
        payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FactorTableMigrationError(f"target day manifest cannot be read: {paths.manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise FactorTableMigrationError(f"target day manifest must be an object: {paths.manifest_path}")
    return payload


def _assert_exact_frame(actual: pd.DataFrame, expected: pd.DataFrame, *, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=True,
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise FactorTableMigrationError(f"target read-back differs from source semantics: {label}") from exc


def _assert_target_source_evidence(
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    actual = manifest.get("source_evidence")
    if actual != expected:
        raise FactorTableMigrationError(f"target source evidence differs from the migration plan: {label}")


def verify_migration(plan: MigrationPlan, *, store: CanonicalFactorStore | None = None) -> dict[str, Any]:
    """Read back every planned target/source/family/day and prove semantic parity.

    ``CanonicalFactorStore.read_day`` verifies the parquet, manifest, done
    marker, table contract, index, and hashes.  This second layer compares the
    semantic frame to the current read-only source and checks every copied
    source hash/provenance entry.
    """

    if not isinstance(plan, MigrationPlan):
        raise TypeError("plan must be a MigrationPlan")
    _assert_governing_artifacts_unchanged(plan)
    store = store or CanonicalFactorStore(plan.target_root)
    source_cache = SourceFrameCache(plan)
    verified = _table_counts_template()
    wide_routes = [route for route in plan.routes if route.route_id in {"live_wide", "experiment_wide"}]
    all_days = sorted({day for route in plan.routes for day in route.days})
    for day in all_days:
        for route in wide_routes:
            if day not in route.days:
                continue
            _verify_wide_day(plan, store=store, source_cache=source_cache, route=route, day=day)
            verified[route.target.value] += 1
        if _library_fragments_for_day(plan, day):
            verified[FactorTableKind.FACTOR_LIBRARY.value] += _verify_library_day(
                plan, store=store, source_cache=source_cache, day=day
            )
        source_cache.release_day(day)
    source_cache.assert_unchanged()
    _assert_governing_artifacts_unchanged(plan)
    return {
        "schema_version": MIGRATION_SCHEMA,
        "target_root": _path_text(plan.target_root),
        "verified_operations_by_table": verified,
        "planner_plan_sha256": plan.planner_plan_sha256,
    }


def _parse_day_list(raw: str) -> tuple[date, ...]:
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--days must contain at least one YYYY-MM-DD value")
    parsed: list[date] = []
    for value in values:
        try:
            parsed.append(datetime.strptime(value, "%Y-%m-%d").date())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid day {value!r}; expected YYYY-MM-DD") from exc
    return tuple(sorted(set(parsed)))


def _parser() -> argparse.ArgumentParser:
    defaults = planner._default_paths()
    parser = argparse.ArgumentParser(
        description=(
            "Plan or explicitly execute the read-only-source CBOND_ON three-factor-table migration. "
            "Default mode prints a JSON plan and creates nothing."
        )
    )
    parser.add_argument("--catalog-path", default=str(defaults["catalog_path"]))
    parser.add_argument("--live-release-path", default=str(defaults["live_release_path"]))
    parser.add_argument("--live-profile-path", default=str(defaults["live_profile_path"]))
    parser.add_argument("--experiment-profile-path", default=str(defaults["experiment_profile_path"]))
    parser.add_argument("--live-root", default=str(defaults["live_root"]))
    parser.add_argument("--experiment-root", default=str(defaults["experiment_root"]))
    parser.add_argument("--catalog-history-root", default=str(defaults["catalog_history_root"]))
    parser.add_argument("--supplement-root", default=str(defaults["supplement_root"]))
    parser.add_argument(
        "--experiment-manifest-path",
        default=None if defaults["experiment_manifest_path"] is None else str(defaults["experiment_manifest_path"]),
    )
    parser.add_argument(
        "--catalog-history-manifest-path",
        default=None if defaults["catalog_history_manifest_path"] is None else str(defaults["catalog_history_manifest_path"]),
    )
    parser.add_argument(
        "--supplement-manifest-path",
        default=None,
        help="Optional exact supplement execution manifest; it is still fully validated against the source partition.",
    )
    parser.add_argument(
        "--days",
        type=_parse_day_list,
        default=None,
        help="Comma-separated explicit source days. Mutually exclusive with --smoke-route-days.",
    )
    parser.add_argument(
        "--smoke-route-days",
        action="store_true",
        help="Use fixed route-covering days: 2024-01-03, 2025-01-02, 2026-08-25, 2026-08-27.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write only D:/cbond_on/factor_store after all source/contract gates pass.",
    )
    parser.add_argument(
        "--accept-date-owned-catalog-routing",
        action="store_true",
        help="Required with --execute: acknowledges deterministic no-coalescing ownership for live/catalog-history overlaps.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.days is not None and args.smoke_route_days:
        _parser().error("--days and --smoke-route-days are mutually exclusive")
    if args.execute and not args.accept_date_owned_catalog_routing:
        _parser().error("--execute requires --accept-date-owned-catalog-routing")
    plan = build_migration_plan(
        catalog_path=args.catalog_path,
        live_release_path=args.live_release_path,
        live_profile_path=args.live_profile_path,
        experiment_profile_path=args.experiment_profile_path,
        live_root=args.live_root,
        experiment_root=args.experiment_root,
        catalog_history_root=args.catalog_history_root,
        supplement_root=args.supplement_root,
        experiment_manifest_path=args.experiment_manifest_path,
        catalog_history_manifest_path=args.catalog_history_manifest_path,
        supplement_manifest_path=args.supplement_manifest_path,
        target_root=CANONICAL_FACTOR_STORE_ROOT,
        days=args.days,
        smoke_route_days=bool(args.smoke_route_days),
    )
    if not _same_path(plan.target_root, CANONICAL_FACTOR_STORE_ROOT):  # pragma: no cover - defensive CLI invariant.
        raise FactorTableMigrationError("CLI target root must be exactly D:/cbond_on/factor_store")
    output: dict[str, Any] = {"plan": plan.to_summary()}
    if args.execute:
        output["execution"] = execute_migration(
            plan,
            accept_date_owned_catalog_routing=True,
            verify_readback=True,
        ).to_dict()
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CANONICAL_FACTOR_STORE_ROOT",
    "FactorTableMigrationError",
    "MigrationPlan",
    "MigrationResult",
    "ROUTE_SMOKE_DAYS",
    "SourceFrameCache",
    "build_migration_plan",
    "execute_migration",
    "main",
    "verify_migration",
]


if __name__ == "__main__":
    raise SystemExit(main())
