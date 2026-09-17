"""Manifest-bound admission for the R88 experiment factor table.

R88 is a research profile, not a fourth factor-result table.  Its normal
model/backtest consumers must therefore enter through the canonical
``experiment`` table and its published commit bundles.  Historical R88 roots
remain provenance evidence for the migration/audit tools only; this module
never opens them as a factor-data input.

The narrow R88 checks live here rather than in individual model runners so
LGBM and Torch research paths cannot drift back to a direct ``factor_data``
directory independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from cbond_on.infra.factors.canonical_store import CanonicalFactorStore, FactorTableContract
from cbond_on.infra.factors.factor_table_resolution import (
    CanonicalFactorTableReader,
    FactorTableResolution,
    resolve_factor_table_reference,
)


_R88_FACTOR_COUNT = 88
_R88_PROFILE_SCHEMA = "research_r88_factor_profile/v1"
_R88_SOURCE_EXECUTION_STATUS = "completed_rust88_backfill"
_DONE_NAME = re.compile(r"^(?P<day>\d{8})\.done$")
_MONTH_NAME = re.compile(r"^\d{4}-\d{2}$")


class R88ExperimentTableError(RuntimeError):
    """The canonical experiment table cannot safely serve the R88 profile."""


@dataclass(frozen=True)
class R88ExperimentTableBinding:
    """One read-only, manifest-bound R88 experiment-table admission."""

    resolution: FactorTableResolution
    reader: CanonicalFactorTableReader
    contract: FactorTableContract
    days: tuple[date, ...]
    table_manifest_sha256: str
    source_provenance: Mapping[str, Any]

    @property
    def table_root(self) -> Path:
        return self.resolution.factor_data_root

    @property
    def table_manifest_path(self) -> Path:
        return self.resolution.manifest_path

    def to_evidence(self) -> dict[str, Any]:
        """Return serialisable immutable input evidence for a research plan."""

        return {
            "table_id": self.resolution.table_id,
            "canonical_store_root": self.resolution.store_root.as_posix(),
            "table_root": self.table_root.as_posix(),
            "table_manifest": self.table_manifest_path.as_posix(),
            "table_manifest_sha256": self.table_manifest_sha256,
            "contract_list_sha256": self.contract.sha256,
            "factor_count": len(self.contract.factor_ids),
            "day_count": len(self.days),
            "days_sha256": _sha256_json([day.isoformat() for day in self.days]),
            "source_provenance": dict(self.source_provenance),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R88ExperimentTableError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise R88ExperimentTableError(f"{label} must be a JSON object: {path}")
    return payload


def _expected_factor_ids(profile: Mapping[str, Any]) -> tuple[str, ...]:
    if str(profile.get("schema_version", "")).strip() != _R88_PROFILE_SCHEMA:
        raise R88ExperimentTableError("R88 profile has an unexpected schema_version")
    factors_raw = profile.get("factors")
    if not isinstance(factors_raw, list):
        raise R88ExperimentTableError("R88 profile factors must be a list")
    factor_ids = tuple(str(value).strip() for value in factors_raw)
    if len(factor_ids) != _R88_FACTOR_COUNT or len(set(factor_ids)) != _R88_FACTOR_COUNT or any(not value for value in factor_ids):
        raise R88ExperimentTableError("R88 profile must contain exactly 88 unique ordered factor IDs")
    return factor_ids


def _published_days(reader: CanonicalFactorTableReader) -> tuple[date, ...]:
    """List only explicitly committed canonical experiment days.

    The canonical table layout is fixed and validated by the resolver.  This
    routine intentionally reads the ``.done`` commit boundary rather than
    scanning parquet files, so a stranded parquet cannot become an input day.
    Each returned day is then opened through the canonical reader at least
    once, which validates the full bundle before admission continues.
    """

    done_root = reader.root / "done" / "T1430"
    if not done_root.is_dir():
        raise R88ExperimentTableError(f"canonical experiment table has no done root: {done_root}")
    days: list[date] = []
    for month_dir in sorted(done_root.iterdir(), key=lambda item: item.name):
        if not month_dir.is_dir() or not _MONTH_NAME.fullmatch(month_dir.name):
            raise R88ExperimentTableError(f"unexpected canonical experiment done entry: {month_dir}")
        for done_path in sorted(month_dir.iterdir(), key=lambda item: item.name):
            if not done_path.is_file():
                raise R88ExperimentTableError(f"unexpected canonical experiment done child: {done_path}")
            match = _DONE_NAME.fullmatch(done_path.name)
            if match is None:
                raise R88ExperimentTableError(f"unexpected canonical experiment done filename: {done_path}")
            try:
                day = date.fromisoformat(
                    f"{match.group('day')[:4]}-{match.group('day')[4:6]}-{match.group('day')[6:]}"
                )
            except ValueError as exc:  # pragma: no cover - regex permits impossible dates.
                raise R88ExperimentTableError(f"invalid canonical experiment score day: {done_path}") from exc
            if f"{day:%Y-%m}" != month_dir.name:
                raise R88ExperimentTableError(
                    "canonical experiment done month does not match score day: " f"{done_path}"
                )
            days.append(day)
    if not days:
        raise R88ExperimentTableError("canonical experiment table has no published days")
    if days != sorted(days) or len(days) != len(set(days)):
        raise R88ExperimentTableError("canonical experiment done days must be sorted and unique")
    # First/last validates that the table has actual complete bundles before a
    # model plan is materialised.  The caller's full coverage loop validates
    # every selected day before execution.
    reader.read_day(days[0])
    reader.read_day(days[-1])
    return tuple(days)


def _matching_contract(
    *,
    resolution: FactorTableResolution,
    expected_factor_ids: Sequence[str],
) -> FactorTableContract:
    registered = resolution.manifest.get("registered_contracts")
    if not isinstance(registered, list):
        raise R88ExperimentTableError("canonical experiment table has no registered_contracts list")
    expected = list(expected_factor_ids)
    matches = [
        entry
        for entry in registered
        if isinstance(entry, Mapping)
        and entry.get("factor_ids") == expected
        and entry.get("output_columns") == expected
        and entry.get("factor_count") == len(expected)
    ]
    if len(matches) != 1:
        raise R88ExperimentTableError(
            "canonical experiment table must expose exactly one contract matching the R88 profile "
            f"(matches={len(matches)})"
        )
    entry = matches[0]
    raw_contract_path = str(entry.get("contract_path", "")).strip()
    if not raw_contract_path:
        raise R88ExperimentTableError("matching canonical experiment contract has no contract_path")
    contract_path = (resolution.factor_data_root / raw_contract_path).resolve(strict=False)
    try:
        contract_path.relative_to(resolution.factor_data_root.resolve(strict=False))
    except ValueError as exc:
        raise R88ExperimentTableError("canonical experiment contract_path escapes its table root") from exc
    contract = FactorTableContract.from_dict(_load_json(contract_path, label="canonical experiment factor contract"))
    if contract.factor_ids != tuple(expected) or contract.output_columns != tuple(expected):
        raise R88ExperimentTableError("canonical experiment contract order differs from the R88 profile")
    if str(entry.get("contract_list_sha256", "")).lower() != contract.sha256:
        raise R88ExperimentTableError("canonical experiment contract hash differs from the table manifest")
    return contract


def _source_provenance(
    *,
    binding_reader: CanonicalFactorTableReader,
    last_day: date,
    profile_path: Path,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate migration provenance without reopening a legacy factor root."""

    day_manifest_path = binding_reader.root / "manifests" / "T1430" / f"{last_day:%Y-%m}" / f"{last_day:%Y%m%d}.json"
    payload = _load_json(day_manifest_path, label="canonical experiment day manifest")
    source_evidence = payload.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        raise R88ExperimentTableError("canonical experiment day lacks migration source evidence")
    direct_current = str(source_evidence.get("source", "")).strip() == "r88_full_current_rust88_recompute"
    if direct_current:
        historical = source_evidence
        recorded_profile = source_evidence.get("profile")
    else:
        fragments = source_evidence.get("fragments")
        if not isinstance(fragments, list):
            raise R88ExperimentTableError("canonical experiment day source evidence has no fragments list")
        matches = [
            fragment
            for fragment in fragments
            if isinstance(fragment, Mapping) and str(fragment.get("source", "")).strip() == "experiment_input"
        ]
        if len(matches) != 1:
            raise R88ExperimentTableError("canonical experiment day must have exactly one experiment source fragment")
        fragment = matches[0]
        historical = fragment.get("historical_contract_evidence")
        if not isinstance(historical, Mapping):
            raise R88ExperimentTableError("canonical experiment source fragment lacks historical contract evidence")
        recorded_profile = historical.get("historical_profile")
    if not isinstance(recorded_profile, Mapping):
        raise R88ExperimentTableError("canonical experiment source fragment lacks historical profile evidence")
    expected_relative = profile_path.as_posix()
    # The project-relative path is the durable identity; an absolute checkout
    # path would make the canonical input needlessly machine-specific.
    try:
        project_relative = profile_path.resolve(strict=False).relative_to(Path(__file__).resolve().parents[3])
        expected_relative = project_relative.as_posix()
    except ValueError:
        pass
    if str(recorded_profile.get("path", "")).replace("\\", "/") != expected_relative:
        raise R88ExperimentTableError("canonical experiment historical profile path differs from the R88 profile")
    if str(recorded_profile.get("admission_profile", "")).strip() != str(profile.get("admission_profile", "")).strip():
        raise R88ExperimentTableError("canonical experiment historical admission profile differs from R88")
    recorded_profile_sha = str(recorded_profile.get("profile_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", recorded_profile_sha):
        raise R88ExperimentTableError("canonical experiment historical profile hash is invalid")
    current_profile_sha = _sha256_file(profile_path)
    current_specs_sha = _sha256_json(profile.get("factor_specs", []))
    if str(recorded_profile.get("specs_sha256", "")).lower() != current_specs_sha:
        raise R88ExperimentTableError("canonical experiment historical factor-spec hash differs from the R88 profile")
    if not direct_current:
        if str(historical.get("source_execution_status", "")).strip() != _R88_SOURCE_EXECUTION_STATUS:
            raise R88ExperimentTableError("canonical experiment source does not attest a completed R88 backfill")
        if historical.get("source_model_training_ready") is not True:
            raise R88ExperimentTableError("canonical experiment source is not attested model-training-ready")
    return {
        "historical_profile": dict(recorded_profile),
        # The migrated values remain a historical materialisation.  The
        # profile file can legitimately receive non-formula provenance edits;
        # ordered membership and immutable factor-spec hash are the execution
        # compatibility gate.  Surface the raw-file drift explicitly instead
        # of either rejecting usable data or silently relabelling it current.
        "current_profile_sha256": current_profile_sha,
        "profile_file_hash_changed_since_materialization": recorded_profile_sha != current_profile_sha,
        "current_specs_sha256": current_specs_sha,
        "materialization": "current_catalog_direct" if direct_current else "historical_source_attested",
        "source_execution_status": "current_catalog_direct" if direct_current else _R88_SOURCE_EXECUTION_STATUS,
        "source_model_training_ready": True,
        "canonical_day": last_day.isoformat(),
        "canonical_day_manifest": day_manifest_path.as_posix(),
        "canonical_day_manifest_sha256": _sha256_file(day_manifest_path),
    }


def _assert_uniform_contract(
    *,
    reader: CanonicalFactorTableReader,
    days: Sequence[date],
    contract: FactorTableContract,
) -> None:
    """Reject a mixed experiment-table history before a rolling model sees it."""

    for day in days:
        manifest_path = reader.root / "manifests" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.json"
        manifest = _load_json(manifest_path, label="canonical experiment day manifest")
        day_contract = manifest.get("contract")
        if not isinstance(day_contract, Mapping):
            raise R88ExperimentTableError(f"canonical experiment day has no contract evidence: {day.isoformat()}")
        if str(day_contract.get("contract_list_sha256", "")).lower() != contract.sha256:
            raise R88ExperimentTableError(
                "canonical experiment table mixes factor materialisation contracts across published days; "
                f"R88 cannot form a rolling input chain (first mismatch={day.isoformat()})"
            )
        done_path = reader.root / "done" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.done"
        done = _load_json(done_path, label="canonical experiment day done marker")
        if (
            done.get("ready") is not True
            or done.get("table_id") != "experiment"
            or done.get("score_day") != day.isoformat()
            or str(done.get("contract_list_sha256", "")).lower() != contract.sha256
        ):
            raise R88ExperimentTableError(
                "canonical experiment done marker differs from the declared R88 contract "
                f"(first mismatch={day.isoformat()})"
            )


def admit_r88_experiment_table(
    paths_cfg: Mapping[str, Any],
    *,
    profile_path: str | Path,
    profile: Mapping[str, Any],
) -> R88ExperimentTableBinding:
    """Bind R88 model research to the consumer-ready canonical experiment table.

    ``paths_cfg`` must declare ``factor_table.table_id='experiment'``.  A
    direct FactorStore path is deliberately not accepted as an alternative.
    """

    raw_table = paths_cfg.get("factor_table")
    if not isinstance(raw_table, Mapping):
        raise R88ExperimentTableError("R88 research requires an explicit canonical factor_table declaration")
    resolution = resolve_factor_table_reference(raw_table)
    if resolution.table_id != "experiment":
        raise R88ExperimentTableError(
            "R88 research requires factor_table.table_id='experiment', " f"got {resolution.table_id!r}"
        )
    expected_factor_ids = _expected_factor_ids(profile)
    contract = _matching_contract(resolution=resolution, expected_factor_ids=expected_factor_ids)
    reader = CanonicalFactorTableReader(resolution, panel_name="T1430", window_minutes=15)
    days = _published_days(reader)
    _assert_uniform_contract(reader=reader, days=days, contract=contract)
    canonical = CanonicalFactorStore(resolution.store_root)
    report = canonical.report()
    table_report = next((item for item in report.tables if item.table_id == "experiment"), None)
    if table_report is None or not table_report.manifest_valid or table_report.incomplete_day_count:
        raise R88ExperimentTableError("canonical experiment table has incomplete or invalid partitions")
    resolved_profile_path = Path(profile_path).expanduser().resolve(strict=False)
    if not resolved_profile_path.is_file():
        raise FileNotFoundError(f"R88 profile is missing: {resolved_profile_path}")
    provenance = _source_provenance(
        binding_reader=reader,
        last_day=days[-1],
        profile_path=resolved_profile_path,
        profile=profile,
    )
    return R88ExperimentTableBinding(
        resolution=resolution,
        reader=reader,
        contract=contract,
        days=days,
        table_manifest_sha256=_sha256_file(resolution.manifest_path),
        source_provenance=provenance,
    )


__all__ = [
    "R88ExperimentTableBinding",
    "R88ExperimentTableError",
    "admit_r88_experiment_table",
]
