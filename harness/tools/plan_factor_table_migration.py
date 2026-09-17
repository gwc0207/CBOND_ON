"""Print a read-only migration plan for the three CBOND_ON factor tables.

The tool deliberately has no ``--execute`` or output-path option.  It reads
the supplied FactorStores and governing contracts, then writes one JSON plan
to stdout.  In particular it never creates a directory, copies a parquet
file, updates configuration, or changes a consumer.

The plan distinguishes a source being *available* from it being safe to
coalesce with another source.  Sources proposed for the same target table are
always compared before a route is described as compatible; a caller must
resolve any reported collision rather than silently choosing one source.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import json5
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANEL_NAME = "T1430"

_LIVE_TABLE_LABEL = "实盘所需要的因子表"
_EXPERIMENT_TABLE_LABEL = "实验用因子表"
_CATALOG_TABLE_LABEL = "因子库内因子表"


@dataclass(frozen=True)
class CatalogFactor:
    factor_id: str
    primary_family: str
    contract_hash: str
    factor_version: str
    output_column: str


@dataclass(frozen=True)
class StoreScan:
    """Read-only, canonical-layout description of one FactorStore."""

    name: str
    root: Path
    paths_by_day: Mapping[date, Path]
    factor_ids: tuple[str, ...]
    row_counts_by_day: Mapping[date, int]
    schema_drift_days: tuple[date, ...]
    first_index_names: tuple[str, ...]

    @property
    def days(self) -> tuple[date, ...]:
        return tuple(sorted(self.paths_by_day))


def _path_text(path: Path) -> str:
    return path.resolve(strict=False).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object in {_path_text(path)}")
    return value


def _require_string(value: object, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string in {_path_text(path)}")
    return value


def _canonical_factor_dir(root: Path, *, panel_name: str) -> Path:
    factor_dir = root / "factors" / panel_name
    if not factor_dir.is_dir():
        raise FileNotFoundError(f"missing canonical FactorStore directory: {_path_text(factor_dir)}")
    return factor_dir


def _parse_day_path(path: Path, *, factor_dir: Path) -> date:
    relative = path.relative_to(factor_dir)
    if len(relative.parts) != 2 or path.suffix.lower() != ".parquet":
        raise ValueError(
            "FactorStore contains a noncanonical path; expected "
            f"YYYY-MM/YYYYMMDD.parquet: {_path_text(path)}"
        )
    month, filename = relative.parts
    try:
        day = datetime.strptime(Path(filename).stem, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"FactorStore filename is not YYYYMMDD.parquet: {_path_text(path)}") from exc
    if month != f"{day:%Y-%m}":
        raise ValueError(f"FactorStore month/day path mismatch: {_path_text(path)}")
    return day


def scan_factor_store(root: str | Path, *, name: str, panel_name: str = _PANEL_NAME) -> StoreScan:
    """Read and validate only FactorStore metadata; never mutate the source."""

    resolved_root = Path(root).expanduser().resolve(strict=True)
    factor_dir = _canonical_factor_dir(resolved_root, panel_name=panel_name)
    files = sorted((path for path in factor_dir.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    if not files:
        raise ValueError(f"FactorStore contains no files: {_path_text(factor_dir)}")

    paths_by_day: dict[date, Path] = {}
    first_factor_ids: tuple[str, ...] | None = None
    first_index_names: tuple[str, ...] | None = None
    schema_drift_days: list[date] = []
    row_counts: dict[date, int] = {}

    for path in files:
        day = _parse_day_path(path, factor_dir=factor_dir)
        if day in paths_by_day:
            raise ValueError(f"duplicate FactorStore day {day.isoformat()} in {_path_text(resolved_root)}")
        parquet = pq.ParquetFile(path)
        physical_names = tuple(parquet.schema_arrow.names)
        if "dt" not in physical_names or "code" not in physical_names:
            raise ValueError(f"FactorStore file is missing dt/code: {_path_text(path)}")
        factor_ids = tuple(name for name in physical_names if name not in {"dt", "code"})
        if not factor_ids or len(set(factor_ids)) != len(factor_ids):
            raise ValueError(f"FactorStore has invalid factor-column identity: {_path_text(path)}")
        if first_factor_ids is None:
            first_factor_ids = factor_ids
            frame = pd.read_parquet(path)
            if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
                raise ValueError(f"FactorStore requires a (dt, code) MultiIndex: {_path_text(path)}")
            if frame.index.has_duplicates:
                raise ValueError(f"FactorStore has duplicate (dt, code) rows: {_path_text(path)}")
            observed_days = set(pd.to_datetime(frame.index.get_level_values("dt")).date)
            if observed_days != {day}:
                raise ValueError(f"FactorStore filename/index day mismatch: {_path_text(path)}")
            first_index_names = tuple(str(item) for item in frame.index.names)
        elif factor_ids != first_factor_ids:
            schema_drift_days.append(day)
        paths_by_day[day] = path
        row_counts[day] = int(parquet.metadata.num_rows)

    assert first_factor_ids is not None
    assert first_index_names is not None
    return StoreScan(
        name=name,
        root=resolved_root,
        paths_by_day=dict(sorted(paths_by_day.items())),
        factor_ids=first_factor_ids,
        row_counts_by_day=dict(sorted(row_counts.items())),
        schema_drift_days=tuple(sorted(schema_drift_days)),
        first_index_names=first_index_names,
    )


def _date_summary(days: Sequence[date]) -> Mapping[str, Any]:
    return {
        "day_count": len(days),
        "start": days[0].isoformat() if days else None,
        "end": days[-1].isoformat() if days else None,
    }


def _source_summary(store: StoreScan) -> Mapping[str, Any]:
    return {
        "root": _path_text(store.root),
        "panel_name": _PANEL_NAME,
        "factor_count": len(store.factor_ids),
        "factor_ids": list(store.factor_ids),
        "coverage": _date_summary(store.days),
        "row_count_total": sum(store.row_counts_by_day.values()),
        "row_count_range": {
            "min": min(store.row_counts_by_day.values()),
            "max": max(store.row_counts_by_day.values()),
        },
        "index_contract": {
            "sampled_partition_index_names": list(store.first_index_names),
            "schema_drift_day_count": len(store.schema_drift_days),
            "schema_drift_days": [item.isoformat() for item in store.schema_drift_days],
        },
    }


def _load_catalog(path: str | Path) -> tuple[Path, Mapping[str, CatalogFactor], Mapping[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = _require_mapping(json.loads(resolved.read_text(encoding="utf-8")), field="factor catalog", path=resolved)
    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, list):
        raise ValueError(f"factor catalog factors must be a list: {_path_text(resolved)}")
    result: dict[str, CatalogFactor] = {}
    for raw in raw_factors:
        item = _require_mapping(raw, field="factor catalog factor", path=resolved)
        factor_id = _require_string(item.get("factor_id"), field="factor_id", path=resolved)
        if factor_id in result:
            raise ValueError(f"duplicate Catalog factor_id {factor_id!r}: {_path_text(resolved)}")
        output_column_raw = item.get("output_column")
        output_column = factor_id if output_column_raw is None else _require_string(
            output_column_raw,
            field="output_column",
            path=resolved,
        )
        result[factor_id] = CatalogFactor(
            factor_id=factor_id,
            primary_family=_require_string(item.get("primary_family"), field="primary_family", path=resolved),
            contract_hash=_require_string(item.get("contract_hash"), field="contract_hash", path=resolved),
            factor_version=_require_string(item.get("factor_version"), field="factor_version", path=resolved),
            output_column=output_column,
        )
    expected_count = payload.get("factor_count")
    if expected_count != len(result):
        raise ValueError(
            f"Catalog factor_count does not equal unique factor records in {_path_text(resolved)}: "
            f"declared={expected_count!r} actual={len(result)}"
        )
    return resolved, result, payload


def _load_live_release(path: str | Path, *, catalog: Mapping[str, CatalogFactor]) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = _require_mapping(json.loads(resolved.read_text(encoding="utf-8")), field="live release", path=resolved)
    raw_instances = payload.get("instances")
    if not isinstance(raw_instances, list):
        raise ValueError(f"live release instances must be a list: {_path_text(resolved)}")
    factor_ids: list[str] = []
    contract_matches = 0
    contract_mismatches: list[str] = []
    unknown_factor_ids: list[str] = []
    for raw in raw_instances:
        item = _require_mapping(raw, field="live release instance", path=resolved)
        factor_id = _require_string(item.get("factor_id"), field="factor_id", path=resolved)
        factor_ids.append(factor_id)
        catalog_factor = catalog.get(factor_id)
        if catalog_factor is None:
            unknown_factor_ids.append(factor_id)
            continue
        if item.get("contract_hash") == catalog_factor.contract_hash:
            contract_matches += 1
        else:
            contract_mismatches.append(factor_id)
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError(f"live release has duplicate factor_id entries: {_path_text(resolved)}")
    declared = payload.get("factor_count")
    if declared != len(factor_ids):
        raise ValueError(f"live release factor_count mismatch: {_path_text(resolved)}")
    return {
        "path": _path_text(resolved),
        "sha256": _sha256(resolved),
        "factor_count": len(factor_ids),
        "factor_ids": factor_ids,
        "catalog_contract_alignment": {
            "matched_factor_count": contract_matches,
            "mismatched_factor_ids": sorted(contract_mismatches),
            "unknown_factor_ids": sorted(unknown_factor_ids),
        },
    }


def _load_profile(path: str | Path, *, catalog: Mapping[str, CatalogFactor]) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = _require_mapping(json5.loads(resolved.read_text(encoding="utf-8")), field="experiment profile", path=resolved)
    raw_factors = payload.get("factors")
    if not isinstance(raw_factors, list):
        raise ValueError(f"experiment profile factors must be a list: {_path_text(resolved)}")
    factor_ids: list[str] = []
    for raw in raw_factors:
        if isinstance(raw, str):
            factor_id = raw
        elif isinstance(raw, Mapping):
            value = raw.get("name") or raw.get("factor_id") or raw.get("output_col")
            factor_id = _require_string(value, field="experiment factor identity", path=resolved)
        else:
            raise ValueError(f"experiment profile factor has unsupported shape: {_path_text(resolved)}")
        factor_ids.append(factor_id)
    if len(set(factor_ids)) != len(factor_ids):
        raise ValueError(f"experiment profile has duplicate factor IDs: {_path_text(resolved)}")
    return {
        "path": _path_text(resolved),
        "sha256": _sha256(resolved),
        "admission_profile": payload.get("admission_profile"),
        "specs_sha256": payload.get("specs_sha256"),
        "factor_count": len(factor_ids),
        "factor_ids": factor_ids,
        "unknown_catalog_factor_ids": sorted(set(factor_ids) - set(catalog)),
    }


def _load_optional_json(path: Path | None) -> tuple[Path | None, Mapping[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve(strict=True)
    return resolved, _require_mapping(json.loads(resolved.read_text(encoding="utf-8")), field="manifest", path=resolved)


def _find_supplement_manifests(root: Path, *, store: StoreScan) -> list[tuple[Path, Mapping[str, Any]]]:
    manifests_root = root.parent / "manifests"
    if not manifests_root.is_dir():
        return []
    discovered: list[tuple[Path, Mapping[str, Any]]] = []
    store_paths = {_path_text(path) for path in store.paths_by_day.values()}
    for path in sorted(manifests_root.rglob("*.json"), key=lambda item: item.as_posix()):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        factor_store = payload.get("factor_store")
        if not isinstance(factor_store, Mapping):
            continue
        stored_path = factor_store.get("path")
        if isinstance(stored_path, str) and _path_text(Path(stored_path)) in store_paths:
            discovered.append((path.resolve(strict=True), payload))
    return discovered


def _contract_snapshot_summary(
    *,
    manifests: Iterable[tuple[Path, Mapping[str, Any]]],
    catalog: Mapping[str, CatalogFactor],
) -> Mapping[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for path, payload in manifests:
        expected = payload.get("expected_factor_ids")
        hashes = payload.get("factor_contract_hashes")
        expected_ids = [item for item in expected if isinstance(item, str)] if isinstance(expected, list) else []
        hash_map = hashes if isinstance(hashes, Mapping) else {}
        compared = [factor_id for factor_id in expected_ids if factor_id in catalog and isinstance(hash_map.get(factor_id), str)]
        matches = sum(hash_map[factor_id] == catalog[factor_id].contract_hash for factor_id in compared)
        coverage_quality = payload.get("coverage_quality")
        all_nan = None
        if isinstance(coverage_quality, Mapping):
            value = coverage_quality.get("all_nan_factor_count")
            all_nan = value if isinstance(value, int) else None
        rows.append(
            {
                "path": _path_text(path),
                "sha256": _sha256(path),
                "status": payload.get("status"),
                "complete": payload.get("complete"),
                "expected_factor_count": len(expected_ids),
                "current_catalog_contract_hash_match_count": matches,
                "current_catalog_contract_hash_compared_count": len(compared),
                "all_nan_factor_count": all_nan,
            }
        )
    return {"attestation_count": len(rows), "attestations": rows}


def _frame_for_compare(path: Path, *, factor_ids: Sequence[str]) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=list(factor_ids))
    except (ValueError, KeyError):
        frame = pd.read_parquet(path)
        frame = frame.loc[:, list(factor_ids)]
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
        if {"dt", "code"}.issubset(frame.columns):
            frame = frame.set_index(["dt", "code"])
        else:
            raise ValueError(f"comparison requires a (dt, code) index: {_path_text(path)}")
    if frame.index.has_duplicates:
        raise ValueError(f"comparison found duplicate (dt, code): {_path_text(path)}")
    return frame.sort_index()


def _index_only(path: Path) -> pd.MultiIndex:
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
        if {"dt", "code"}.issubset(frame.columns):
            frame = frame.set_index(["dt", "code"])
        else:
            raise ValueError(f"comparison requires a (dt, code) index: {_path_text(path)}")
    if frame.index.has_duplicates:
        raise ValueError(f"comparison found duplicate (dt, code): {_path_text(path)}")
    return frame.index.sort_values()


def compare_source_values(left: StoreScan, right: StoreScan) -> Mapping[str, Any]:
    """Compute deterministic index/value evidence for two read-only sources."""

    common_factor_ids = tuple(sorted(set(left.factor_ids) & set(right.factor_ids)))
    common_days = tuple(sorted(set(left.paths_by_day) & set(right.paths_by_day)))
    same_index_days = 0
    index_mismatch_days: list[date] = []
    left_only_rows = 0
    right_only_rows = 0
    common_rows = 0
    equal_cells = 0
    nan_mask_conflict_cells = 0
    finite_value_conflict_cells = 0
    per_factor: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for day in common_days:
        if common_factor_ids:
            left_frame = _frame_for_compare(left.paths_by_day[day], factor_ids=common_factor_ids)
            right_frame = _frame_for_compare(right.paths_by_day[day], factor_ids=common_factor_ids)
            left_index = left_frame.index
            right_index = right_frame.index
        else:
            left_index = _index_only(left.paths_by_day[day])
            right_index = _index_only(right.paths_by_day[day])
        if left_index.equals(right_index):
            same_index_days += 1
        else:
            index_mismatch_days.append(day)
            left_only_rows += len(left_index.difference(right_index))
            right_only_rows += len(right_index.difference(left_index))
        common_index = left_index.intersection(right_index, sort=False)
        common_rows += len(common_index)
        if not common_factor_ids:
            continue
        left_values = left_frame.reindex(common_index).loc[:, list(common_factor_ids)].to_numpy(dtype=float, na_value=np.nan)
        right_values = right_frame.reindex(common_index).loc[:, list(common_factor_ids)].to_numpy(dtype=float, na_value=np.nan)
        equal = (left_values == right_values) | (np.isnan(left_values) & np.isnan(right_values))
        nan_mask_conflict = np.isnan(left_values) ^ np.isnan(right_values)
        finite_value_conflict = np.isfinite(left_values) & np.isfinite(right_values) & (left_values != right_values)
        equal_cells += int(equal.sum())
        nan_mask_conflict_cells += int(nan_mask_conflict.sum())
        finite_value_conflict_cells += int(finite_value_conflict.sum())
        for position, factor_id in enumerate(common_factor_ids):
            per_factor[factor_id]["nan_mask_conflict_cells"] += int(nan_mask_conflict[:, position].sum())
            per_factor[factor_id]["finite_value_conflict_cells"] += int(finite_value_conflict[:, position].sum())

    return {
        "left_source": left.name,
        "right_source": right.name,
        "shared_factor_count": len(common_factor_ids),
        "shared_factor_ids": list(common_factor_ids),
        "shared_day_count": len(common_days),
        "index": {
            "same_index_day_count": same_index_days,
            "mismatch_day_count": len(index_mismatch_days),
            "mismatch_day_sample": [item.isoformat() for item in index_mismatch_days[:20]],
            "left_only_row_count": left_only_rows,
            "right_only_row_count": right_only_rows,
            "common_row_count": common_rows,
        },
        "value": {
            "equal_cell_count": equal_cells,
            "nan_mask_conflict_cell_count": nan_mask_conflict_cells,
            "finite_value_conflict_cell_count": finite_value_conflict_cells,
            "finite_value_conflicting_factor_ids": sorted(
                factor_id
                for factor_id, stats in per_factor.items()
                if stats["finite_value_conflict_cells"]
            ),
            "nan_mask_conflicting_factor_ids": sorted(
                factor_id
                for factor_id, stats in per_factor.items()
                if stats["nan_mask_conflict_cells"]
            ),
        },
    }


def _schema_overlap_only(left: StoreScan, right: StoreScan, *, purpose: str) -> Mapping[str, Any]:
    """Record a separate-table overlap without reading source values."""

    return {
        "left_source": left.name,
        "right_source": right.name,
        "purpose": purpose,
        "value_check": "not_applicable_separate_target_tables",
        "shared_factor_count": len(set(left.factor_ids) & set(right.factor_ids)),
        "shared_factor_ids": sorted(set(left.factor_ids) & set(right.factor_ids)),
        "shared_day_count": len(set(left.days) & set(right.days)),
    }


def _family_routes(factor_ids: Iterable[str], *, catalog: Mapping[str, CatalogFactor]) -> list[Mapping[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for factor_id in sorted(factor_ids):
        if factor_id in catalog:
            grouped[catalog[factor_id].primary_family].append(factor_id)
    return [
        {"family": family, "factor_ids": factor_ids}
        for family, factor_ids in sorted(grouped.items())
    ]


def _membership_check(
    *,
    expected_ids: Sequence[str],
    actual_ids: Sequence[str],
    require_order: bool,
) -> Mapping[str, Any]:
    expected = tuple(expected_ids)
    actual = tuple(actual_ids)
    return {
        "expected_factor_count": len(expected),
        "actual_factor_count": len(actual),
        "missing_factor_ids": sorted(set(expected) - set(actual)),
        "unexpected_factor_ids": sorted(set(actual) - set(expected)),
        "ordered_membership_match": expected == actual if require_order else None,
    }


def _manifest_profile_alignment(
    manifest: Mapping[str, Any] | None, profile: Mapping[str, Any]) -> Mapping[str, Any]:
    if manifest is None:
        return {"manifest_found": False}
    manifest_factors = manifest.get("factors")
    listed = [item for item in manifest_factors if isinstance(item, str)] if isinstance(manifest_factors, list) else []
    manifest_profile = manifest.get("profile")
    recorded_specs_sha256 = manifest_profile.get("specs_sha256") if isinstance(manifest_profile, Mapping) else None
    recorded_profile_sha256 = manifest_profile.get("profile_sha256") if isinstance(manifest_profile, Mapping) else None
    return {
        "manifest_found": True,
        "execution_status": manifest.get("execution_status"),
        "model_training_ready": manifest.get("model_training_ready"),
        "manifest_factor_count": len(listed),
        "profile_factor_membership_match": listed == profile["factor_ids"],
        "profile_specs_sha256_match": recorded_specs_sha256 == profile.get("specs_sha256"),
        "profile_sha256_match": recorded_profile_sha256 == profile.get("sha256"),
    }


def _comparison_has_collision(comparison: Mapping[str, Any]) -> bool:
    """Return whether two same-table candidate sources cannot be coalesced."""

    index = comparison.get("index")
    value = comparison.get("value")
    if not isinstance(index, Mapping) or not isinstance(value, Mapping):
        return bool(comparison.get("shared_factor_count")) and bool(comparison.get("shared_day_count"))
    return bool(
        index.get("mismatch_day_count")
        or value.get("nan_mask_conflict_cell_count")
        or value.get("finite_value_conflict_cell_count")
    )


def _source_contract_evidence(
    *,
    source: str,
    live_release: Mapping[str, Any],
    experiment_alignment: Mapping[str, Any],
    supplement_attestations: Mapping[str, Any],
) -> Mapping[str, Any]:
    if source == "live_input":
        alignment = live_release["catalog_contract_alignment"]
        return {
            "kind": "active_release_contract",
            "current_catalog_contract_aligned": not alignment["mismatched_factor_ids"]
            and not alignment["unknown_factor_ids"],
        }
    if source == "experiment_input":
        return {
            "kind": "experiment_profile_contract",
            "profile_factor_membership_match": experiment_alignment.get("profile_factor_membership_match"),
            "profile_specs_sha256_match": experiment_alignment.get("profile_specs_sha256_match"),
        }
    if source == "catalog_history_input":
        return {
            "kind": "source_manifest_column_and_family_attestation",
            "per_factor_contract_hashes_present": False,
        }
    if source == "supplement_input":
        return {
            "kind": "source_manifest_historical_contract_attestation",
            "attestation_count": supplement_attestations["attestation_count"],
            "preserve_source_contract_hashes_in_target_manifest": True,
        }
    raise ValueError(f"unknown source role: {source}")


def build_plan(
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
    value_conflict_check: bool = True,
    cross_table_value_audit: bool = False,
) -> Mapping[str, Any]:
    """Return a complete, deterministic plan without creating any filesystem state."""

    catalog_path_resolved, catalog, catalog_payload = _load_catalog(catalog_path)
    live_release = _load_live_release(live_release_path, catalog=catalog)
    live_profile = _load_profile(live_profile_path, catalog=catalog)
    experiment_profile = _load_profile(experiment_profile_path, catalog=catalog)
    live = scan_factor_store(live_root, name="live_input")
    experiment = scan_factor_store(experiment_root, name="experiment_input")
    catalog_history = scan_factor_store(catalog_history_root, name="catalog_history_input")
    supplement = scan_factor_store(supplement_root, name="supplement_input")

    experiment_manifest_resolved, experiment_manifest = _load_optional_json(
        Path(experiment_manifest_path) if experiment_manifest_path is not None else None
    )
    catalog_history_manifest_resolved, catalog_history_manifest = _load_optional_json(
        Path(catalog_history_manifest_path) if catalog_history_manifest_path is not None else None
    )
    if supplement_manifest_path is not None:
        supplement_manifests = [_load_optional_json(Path(supplement_manifest_path))]
        resolved_supplement_manifests = [
            (path, payload) for path, payload in supplement_manifests if path is not None and payload is not None
        ]
    else:
        resolved_supplement_manifests = _find_supplement_manifests(supplement.root, store=supplement)

    catalog_ids = set(catalog)
    live_ids = set(live.factor_ids)
    experiment_ids = set(experiment.factor_ids)
    catalog_history_ids = set(catalog_history.factor_ids)
    supplement_ids = set(supplement.factor_ids)
    expected_live_ids = list(live_release["factor_ids"])
    expected_experiment_ids = list(experiment_profile["factor_ids"])

    source_summaries: dict[str, Any] = {
        "live_input": _source_summary(live),
        "experiment_input": _source_summary(experiment),
        "catalog_history_input": _source_summary(catalog_history),
        "supplement_input": _source_summary(supplement),
    }
    source_summaries["live_input"]["membership"] = _membership_check(
        expected_ids=expected_live_ids, actual_ids=live.factor_ids, require_order=True
    )
    source_summaries["experiment_input"]["membership"] = _membership_check(
        expected_ids=expected_experiment_ids, actual_ids=experiment.factor_ids, require_order=True
    )
    for source_name, source_ids in (
        ("catalog_history_input", catalog_history_ids),
        ("supplement_input", supplement_ids),
    ):
        source_summaries[source_name]["catalog_membership"] = {
            "registered_factor_count": len(source_ids & catalog_ids),
            "unregistered_factor_ids": sorted(source_ids - catalog_ids),
        }

    if catalog_history_manifest is not None:
        output_columns = catalog_history_manifest.get("output_columns")
        if isinstance(output_columns, list) and all(isinstance(item, str) for item in output_columns):
            source_summaries["catalog_history_input"]["manifest_membership"] = _membership_check(
                expected_ids=output_columns,
                actual_ids=catalog_history.factor_ids,
                require_order=True,
            )
        source_summaries["catalog_history_input"]["manifest"] = {
            "path": _path_text(catalog_history_manifest_resolved),
            "sha256": _sha256(catalog_history_manifest_resolved),
        }
    if experiment_manifest is not None:
        source_summaries["experiment_input"]["manifest"] = {
            "path": _path_text(experiment_manifest_resolved),
            "sha256": _sha256(experiment_manifest_resolved),
        }
    source_summaries["supplement_input"]["contract_attestations"] = _contract_snapshot_summary(
        manifests=resolved_supplement_manifests,
        catalog=catalog,
    )

    pair_specs = (
        (live, experiment, "cross_table_parity_audit", cross_table_value_audit),
        (live, catalog_history, "same_table_candidate_collision_check", True),
        (experiment, catalog_history, "cross_table_contract_collision_check", cross_table_value_audit),
        (live, supplement, "same_table_disjoint_column_union_check", True),
        (catalog_history, supplement, "same_table_date_overlap_check", True),
        (experiment, supplement, "cross_table_contract_collision_check", cross_table_value_audit),
    )
    comparisons: list[Mapping[str, Any]] = []
    if value_conflict_check:
        for left, right, purpose, value_required in pair_specs:
            if value_required:
                result = dict(compare_source_values(left, right))
                result["purpose"] = purpose
                comparisons.append(result)
            else:
                comparisons.append(_schema_overlap_only(left, right, purpose=purpose))
    else:
        for left, right, purpose, _value_required in pair_specs:
            comparisons.append(
                {
                    "left_source": left.name,
                    "right_source": right.name,
                    "purpose": purpose,
                    "value_check": "not_requested",
                    "shared_factor_count": len(set(left.factor_ids) & set(right.factor_ids)),
                    "shared_day_count": len(set(left.days) & set(right.days)),
                }
            )

    comparison_by_pair = {
        (item["left_source"], item["right_source"]): item
        for item in comparisons
    }
    catalog_source_collision = _comparison_has_collision(
        comparison_by_pair[("live_input", "catalog_history_input")]
    )
    catalog_source_dates_overlap = bool(set(catalog_history.days) & set(supplement.days))

    live_only_vs_catalog_history = sorted(live_ids - catalog_history_ids)
    shared_live_catalog_history = sorted(live_ids & catalog_history_ids)
    live_shared_without_catalog_history_days = tuple(
        day for day in live.days if day not in set(catalog_history.days)
    )
    supplement_live_union = supplement_ids | live_ids
    supplement_live_common_days = sorted(set(supplement.days) & set(live.days))
    blockers: list[str] = []
    live_membership = _membership_check(
        expected_ids=expected_live_ids,
        actual_ids=live.factor_ids,
        require_order=True,
    )
    experiment_membership = _membership_check(
        expected_ids=expected_experiment_ids,
        actual_ids=experiment.factor_ids,
        require_order=True,
    )
    if not live_membership["ordered_membership_match"]:
        blockers.append("live_input_does_not_match_active_release_membership_or_order")
    if not experiment_membership["ordered_membership_match"]:
        blockers.append("experiment_input_does_not_match_experiment_profile_membership_or_order")
    if catalog_source_collision:
        blockers.append("same_name_catalog_table_sources_have_index_or_value_conflicts")
    if any(store.schema_drift_days for store in (live, experiment, catalog_history, supplement)):
        blockers.append("source_factor_store_schema_drift_detected")

    return {
        "schema_version": "cbond_on_factor_table_migration_plan/v1",
        "read_only": True,
        "execution_boundary": {
            "writes_factor_tables": False,
            "writes_any_files": False,
            "stdout_only": True,
        },
        "catalog": {
            "path": _path_text(catalog_path_resolved),
            "sha256": _sha256(catalog_path_resolved),
            "factor_count": len(catalog),
            "family_count": len({item.primary_family for item in catalog.values()}),
            "declared_snapshot_id": catalog_payload.get("snapshot_id"),
        },
        "contracts": {
            _LIVE_TABLE_LABEL: {
                "release": live_release,
                "profile": live_profile,
                "release_profile_order_match": live_release["factor_ids"] == live_profile["factor_ids"],
                "source_membership": _membership_check(
                    expected_ids=expected_live_ids,
                    actual_ids=live.factor_ids,
                    require_order=True,
                ),
            },
            _EXPERIMENT_TABLE_LABEL: {
                "profile": experiment_profile,
                "source_membership": _membership_check(
                    expected_ids=expected_experiment_ids,
                    actual_ids=experiment.factor_ids,
                    require_order=True,
                ),
                "manifest_alignment": _manifest_profile_alignment(experiment_manifest, experiment_profile),
            },
            _CATALOG_TABLE_LABEL: {
                "registered_factor_count": len(catalog),
                "family_partition_count": len({item.primary_family for item in catalog.values()}),
                "storage_shape": "<primary_family>/factors/T1430/YYYY-MM/YYYYMMDD.parquet",
            },
        },
        "sources": source_summaries,
        "candidate_routes": [
            {
                "target_table": _LIVE_TABLE_LABEL,
                "source": live.name,
                "factor_ids": list(live.factor_ids),
                "coverage": _date_summary(live.days),
                "requires_conflict_resolution": False,
                "contract_evidence": _source_contract_evidence(
                    source=live.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
            {
                "target_table": _EXPERIMENT_TABLE_LABEL,
                "source": experiment.name,
                "factor_ids": list(experiment.factor_ids),
                "coverage": _date_summary(experiment.days),
                "requires_conflict_resolution": False,
                "contract_evidence": _source_contract_evidence(
                    source=experiment.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
            {
                "target_table": _CATALOG_TABLE_LABEL,
                "source": catalog_history.name,
                "factor_ids": sorted(catalog_history_ids),
                "family_routes": _family_routes(catalog_history_ids, catalog=catalog),
                "coverage": _date_summary(catalog_history.days),
                "requires_conflict_resolution": catalog_source_collision,
                "conflicting_candidate_source": live.name if catalog_source_collision else None,
                "contract_evidence": _source_contract_evidence(
                    source=catalog_history.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
            {
                "target_table": _CATALOG_TABLE_LABEL,
                "source": supplement.name,
                "factor_ids": sorted(supplement_ids),
                "family_routes": _family_routes(supplement_ids, catalog=catalog),
                "coverage": _date_summary(supplement.days),
                "requires_conflict_resolution": catalog_source_dates_overlap,
                "contract_evidence": _source_contract_evidence(
                    source=supplement.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
            {
                "target_table": _CATALOG_TABLE_LABEL,
                "source": live.name,
                "factor_ids": live_only_vs_catalog_history,
                "family_routes": _family_routes(live_only_vs_catalog_history, catalog=catalog),
                "coverage": _date_summary(live.days),
                "requires_conflict_resolution": False,
                "contract_evidence": _source_contract_evidence(
                    source=live.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
            {
                "target_table": _CATALOG_TABLE_LABEL,
                "source": live.name,
                "factor_ids": shared_live_catalog_history,
                "family_routes": _family_routes(shared_live_catalog_history, catalog=catalog),
                "coverage": _date_summary(live_shared_without_catalog_history_days),
                "eligible_source_days": [item.isoformat() for item in live_shared_without_catalog_history_days],
                "requires_conflict_resolution": False,
                "source_date_exclusion_reason": "This candidate is limited to days outside catalog_history_input coverage; any overlapping-source evidence is retained in source_comparisons.",
                "contract_evidence": _source_contract_evidence(
                    source=live.name,
                    live_release=live_release,
                    experiment_alignment=_manifest_profile_alignment(experiment_manifest, experiment_profile),
                    supplement_attestations=source_summaries["supplement_input"]["contract_attestations"],
                ),
            },
        ],
        "composition_checks": {
            "live_and_supplement_factor_ids_disjoint": not bool(live_ids & supplement_ids),
            "live_and_supplement_union_factor_count": len(supplement_live_union),
            "live_and_supplement_union_matches_catalog": supplement_live_union == catalog_ids,
            "live_and_supplement_common_days": [item.isoformat() for item in supplement_live_common_days],
            "catalog_history_registered_factor_count": len(catalog_history_ids & catalog_ids),
            "experiment_registered_factor_count": len(experiment_ids & catalog_ids),
            "same_name_live_columns_not_routed_on_catalog_history_days": shared_live_catalog_history,
            "same_name_live_columns_collision_detected": catalog_source_collision,
        },
        "source_comparisons": comparisons,
        "migration_readiness": {
            "ready_for_copy": not blockers,
            "blocking_reasons": blockers,
        },
        "migration_constraints": [
            "Copy only a route whose source membership and declared contract checks pass.",
            "Do not coalesce same-name source values when source_comparisons reports an index, NaN-mask, or finite-value conflict.",
            "The catalog table is partitioned by Catalog primary_family; its family membership comes only from the active Catalog.",
            "Historical source contract attestations remain attached to their copied partitions; they are not silently relabelled as current-contract data.",
        ],
    }


def _default_paths() -> Mapping[str, Path | None]:
    return {
        "catalog_path": _REPO_ROOT / "factor_engine" / "catalog" / "factor_catalog.json",
        "live_release_path": _REPO_ROOT / "factor_engine" / "releases" / "live" / "live50_rust50_operator_source_20260826.json",
        "live_profile_path": _REPO_ROOT / "cbond_on" / "factor_contracts" / "profiles" / "live50_rust50_20260806.json5",
        "experiment_profile_path": _REPO_ROOT / "cbond_on" / "factor_contracts" / "profiles" / "research_r88_rust88_20260825.json5",
        "live_root": Path(r"D:/cbond_on/factor_data_live50_20260805"),
        "experiment_root": Path(r"D:/cbond_on/research_scratch/r88_factor_backfill_20260825_r2/factor_data"),
        "catalog_history_root": Path(r"D:/cbond_on/research_scratch/factor_mining_20260804_unified_v7_v5_v8_merged_r2"),
        "supplement_root": Path(r"D:/cbond_on/research_scratch/factor_supplement_v1/factor_data"),
        # Historical migration manifests were retired with their noncanonical
        # factor stores.  They are audit-only optional inputs now, never a
        # default dependency of a read-only planning command.
        "experiment_manifest_path": None,
        "catalog_history_manifest_path": None,
    }


def _parser() -> argparse.ArgumentParser:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description="Print a read-only CBOND_ON three-factor-table migration plan as JSON.")
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
        "--skip-value-conflict-check",
        action="store_true",
        help="Only report schema/date overlaps. Default performs the read-only value and index comparison.",
    )
    parser.add_argument(
        "--cross-table-value-audit",
        action="store_true",
        help="Also compare same-name values across the separate live and experiment target tables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_plan(
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
        value_conflict_check=not args.skip_value_conflict_check,
        cross_table_value_audit=args.cross_table_value_audit,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
