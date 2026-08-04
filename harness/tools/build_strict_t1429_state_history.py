"""Build a provenance-audited, research-only 14:29 state reconstruction.

This tool intentionally has no default input or output roots.  A caller must
provide an explicit clean-snapshot root, DataHub manifest root, expected score
calendar, and scratch root.  It never reads an existing state CSV and never
repairs, carries forward, or falls back a missing state row.

Only a row with all four evidence classes is reconstructed:

* availability: matching successful clean manifest and publish ``.done``;
* source: the canonical clean cbond snapshot exists;
* row: Parquet metadata (and any declared manifest row count) is valid;
* hash: a SHA-256 was recorded for the exact source snapshot.

Every same-day V1 publish record is a *historical reconstruction*, never a
forward-PIT certificate: the published file is a post-cutoff final snapshot,
not an immutable version demonstrably available at 14:29.  The output manifest
therefore explicitly marks reconstructed rows as
``historical_reconstruction_not_forward_certified``.  A strict candidate must
reject that manifest unless a separate forward-PIT attestation contract exists.

Every other requested date remains in the audit with ``outcome=blocked``.  The
strict cutoff is structural and fixed at 14:29; callers cannot accidentally
request the live 14:30 behaviour through a command-line option.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence
import uuid

import pandas as pd
from pyarrow.parquet import ParquetFile

from cbond_on.infra.live.model_switch import (
    T1430_DISPERSION_FEATURE_SETS,
    build_t1430_market_state_feature_row,
)


STRICT_CUTOFF_TIME = "14:29"
STRICT_CUTOFF_CLOCK = time(14, 29)
STATE_FEATURE_SET = "path_full_t1429"
STATE_FEATURE_COLUMNS = tuple(T1430_DISPERSION_FEATURE_SETS["path_full_t1430"])
STATE_HISTORY_COLUMNS = ("trade_date", "valid_count", *STATE_FEATURE_COLUMNS)
_SCORE_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATAHUB_V1_SCHEMA_VERSION = "cbond_on_t1430_v1"
DATAHUB_V1_REQUIRED_PROFILE = "cbond_on_live_t1430"
HISTORICAL_RECONSTRUCTION_STATUS = "historical_reconstruction_not_forward_certified"
BLOCKED_STATUS = "blocked"

AUDIT_COLUMNS = (
    "trade_date",
    "cutoff_time",
    "outcome",
    "reason",
    "provenance_classification",
    "forward_pit_certified",
    "availability_evidence_ok",
    "source_evidence_ok",
    "row_evidence_ok",
    "hash_evidence_ok",
    "state_row_evidence_ok",
    "clean_manifest_path",
    "publish_done_path",
    "clean_manifest_sha256",
    "publish_done_sha256",
    "manifest_run_id",
    "done_run_id",
    "manifest_schema_version",
    "manifest_required_profile",
    "publish_done_allow_partial_manifest",
    "publish_done_require_datasets",
    "clean_manifest_produced_at",
    "publish_done_produced_at",
    "source_path",
    "source_exists",
    "source_size_bytes",
    "source_row_count",
    "manifest_row_count",
    "manifest_cbond_file_path",
    "source_schema_sha256",
    "source_sha256",
    "state_row_sha256",
    "state_valid_count",
    "missing_state_columns",
)


@dataclass(frozen=True)
class StrictStateHistoryPaths:
    """All files created by a strict-state build, rooted below one scratch dir."""

    scratch_root: Path
    state_path: Path
    audit_path: Path
    calendar_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class StrictStateHistoryResult:
    """Immutable references and counts produced by one state-history build."""

    paths: StrictStateHistoryPaths
    requested_days: int
    built_days: int
    historical_reconstruction_days: int
    blocked_days: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_hash(value: object) -> str:
    return _sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _canonical_snapshot_path(clean_root: Path, score_day: date) -> Path:
    """Return the only accepted clean-snapshot location.

    ``model_switch`` supports an old alternate path for backward compatibility.
    This provenance tool deliberately does not: accepting a second location
    would make a missing canonical source look repaired.
    """

    return clean_root / "snapshot" / "cbond" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"


def _same_path(left: str | Path, right: Path) -> bool:
    try:
        return os.path.normcase(os.path.normpath(str(Path(left).resolve()))) == os.path.normcase(
            os.path.normpath(str(right.resolve()))
        )
    except OSError:
        return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(
            os.path.normpath(str(right))
        )


def _read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable:{type(exc).__name__}"
    if not isinstance(parsed, dict):
        return None, "not_object"
    return dict(parsed), None


def _record_input_hash(
    audit: dict[str, object],
    *,
    path: Path,
    audit_field: str,
    evidence_name: str,
) -> list[str]:
    """Hash a consumed input so the audit binds the parsed evidence to bytes."""

    if not path.is_file():
        return []
    try:
        audit[audit_field] = _sha256_file(path)
    except OSError as exc:
        return [f"{evidence_name}_hash_failed:{type(exc).__name__}"]
    if not str(audit[audit_field]).strip():
        return [f"{evidence_name}_hash_missing"]
    return []


def _path_is_within(path: Path, ancestor: Path) -> bool:
    try:
        path.resolve().relative_to(ancestor.resolve())
        return True
    except ValueError:
        return False


def _validate_scratch_root(
    scratch_root: str | Path,
    *,
    clean_root: Path,
    manifest_root: Path,
) -> StrictStateHistoryPaths:
    root = Path(scratch_root)
    if not root.is_absolute():
        raise ValueError("scratch_root must be an explicit absolute path")
    root = root.resolve()
    if root == root.parent:
        raise ValueError("scratch_root cannot be a filesystem root")
    # A state artifact cannot be written inside an input source, nor may a
    # broad scratch root encompass the source directories.
    for source_name, source in (("clean_root", clean_root), ("manifest_root", manifest_root)):
        if _path_is_within(root, source) or _path_is_within(source, root):
            raise ValueError(f"scratch_root must not overlap {source_name}")

    state_dir = root / "state"
    paths = StrictStateHistoryPaths(
        scratch_root=root,
        state_path=state_dir / "t1430_market_state_features_pathfull_t1429.csv",
        audit_path=state_dir / "t1430_market_state_features_pathfull_t1429_audit.csv",
        calendar_path=state_dir / "expected-score-calendar.csv",
        manifest_path=state_dir / "t1430_market_state_features_pathfull_t1429_manifest.json",
    )
    existing = [
        path
        for path in (paths.state_path, paths.audit_path, paths.calendar_path, paths.manifest_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite strict-state artifact(s): " + ", ".join(str(path) for path in existing)
        )
    return paths


def _normalise_expected_days(expected_days: Iterable[date | str]) -> list[date]:
    normalised: list[date] = []
    for raw in expected_days:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"invalid expected trade date: {raw!r}")
        normalised.append(pd.Timestamp(parsed).date())
    if not normalised:
        raise ValueError("expected_days must contain at least one score day")
    if len(set(normalised)) != len(normalised):
        raise ValueError("expected_days contains duplicates")
    return sorted(normalised)


def load_expected_days_csv(path: str | Path) -> list[date]:
    """Load the explicit score-day calendar used by the state builder."""

    calendar_path = Path(path)
    if not calendar_path.is_file():
        raise FileNotFoundError(f"expected-days CSV missing: {calendar_path}")
    frame = pd.read_csv(calendar_path)
    if "trade_date" not in frame.columns:
        raise KeyError("expected-days CSV must contain a trade_date column")
    return _normalise_expected_days(frame["trade_date"].tolist())


def discover_score_calendar(score_root: str | Path) -> list[date]:
    """Discover canonical daily score files without opening labels or scores.

    A usable score root has exactly one ``YYYY-MM/YYYY-MM-DD.csv`` file per
    score day.  The caller's selected dates are frozen into the scratch
    calendar artifact before any snapshot/state work begins.
    """

    root = Path(score_root)
    if not root.is_dir():
        raise FileNotFoundError(f"score_root missing: {root}")
    days: list[date] = []
    for path in sorted(root.glob("*/*.csv")):
        name = path.stem
        if not _SCORE_DAY_RE.fullmatch(name):
            raise ValueError(f"unexpected score filename: {path}")
        if path.parent.name != name[:7]:
            raise ValueError(f"score file must live in its YYYY-MM directory: {path}")
        parsed = pd.to_datetime(name, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"invalid score day filename: {path}")
        days.append(pd.Timestamp(parsed).date())
    return _normalise_expected_days(days)


def _base_audit_row(
    *,
    score_day: date,
    clean_root: Path,
    manifest_root: Path,
) -> dict[str, object]:
    return {
        "trade_date": score_day.isoformat(),
        "cutoff_time": STRICT_CUTOFF_TIME,
        "outcome": BLOCKED_STATUS,
        "reason": "not_checked",
        "provenance_classification": BLOCKED_STATUS,
        "forward_pit_certified": False,
        "availability_evidence_ok": False,
        "source_evidence_ok": False,
        "row_evidence_ok": False,
        "hash_evidence_ok": False,
        "state_row_evidence_ok": False,
        "clean_manifest_path": str(manifest_root / "clean" / f"{score_day:%Y-%m-%d}.json"),
        "publish_done_path": str(manifest_root / "publish" / f"{score_day:%Y-%m-%d}.done"),
        "clean_manifest_sha256": "",
        "publish_done_sha256": "",
        "manifest_run_id": "",
        "done_run_id": "",
        "manifest_schema_version": "",
        "manifest_required_profile": "",
        "publish_done_allow_partial_manifest": "",
        "publish_done_require_datasets": "",
        "clean_manifest_produced_at": "",
        "publish_done_produced_at": "",
        "source_path": str(_canonical_snapshot_path(clean_root, score_day)),
        "source_exists": False,
        "source_size_bytes": None,
        "source_row_count": None,
        "manifest_row_count": None,
        "manifest_cbond_file_path": "",
        "source_schema_sha256": "",
        "source_sha256": "",
        "state_row_sha256": "",
        "state_valid_count": None,
        "missing_state_columns": "",
    }


def _availability_evidence(
    audit: dict[str, object], *, score_day: date
) -> tuple[dict[str, Any] | None, list[str]]:
    clean_manifest_path = Path(str(audit["clean_manifest_path"]))
    publish_done_path = Path(str(audit["publish_done_path"]))
    reasons: list[str] = []
    reasons.extend(
        _record_input_hash(
            audit,
            path=clean_manifest_path,
            audit_field="clean_manifest_sha256",
            evidence_name="clean_manifest",
        )
    )
    reasons.extend(
        _record_input_hash(
            audit,
            path=publish_done_path,
            audit_field="publish_done_sha256",
            evidence_name="publish_done",
        )
    )
    manifest, manifest_error = _read_json_object(clean_manifest_path)
    done, done_error = _read_json_object(publish_done_path)

    if manifest_error is not None:
        reasons.append(f"clean_manifest_{manifest_error}")
    if done_error is not None:
        reasons.append(f"publish_done_{done_error}")
    if manifest is None or done is None:
        return None, reasons

    expected_day = score_day.isoformat()
    manifest_run_id = str(manifest.get("run_id", "")).strip()
    done_run_id = str(done.get("run_id", "")).strip()
    manifest_schema_version = str(manifest.get("schema_version", "")).strip()
    manifest_required_profile = str(manifest.get("required_profile", "")).strip()
    audit["manifest_run_id"] = manifest_run_id
    audit["done_run_id"] = done_run_id
    audit["manifest_schema_version"] = manifest_schema_version
    audit["manifest_required_profile"] = manifest_required_profile
    manifest_produced_at = str(manifest.get("produced_at", "")).strip()
    done_produced_at = str(done.get("produced_at", "")).strip()
    audit["clean_manifest_produced_at"] = manifest_produced_at
    audit["publish_done_produced_at"] = done_produced_at

    if str(manifest.get("status", "")).strip().lower() != "success":
        reasons.append("clean_manifest_not_success")
    if str(manifest.get("trade_day", "")).strip() != expected_day:
        reasons.append("clean_manifest_trade_day_mismatch")
    if not manifest_run_id:
        reasons.append("clean_manifest_run_id_missing")
    if manifest_schema_version != DATAHUB_V1_SCHEMA_VERSION:
        reasons.append("clean_manifest_schema_version_not_cbond_on_t1430_v1")
    if manifest_required_profile != DATAHUB_V1_REQUIRED_PROFILE:
        reasons.append("clean_manifest_required_profile_not_cbond_on_live_t1430")

    for name, value in (
        ("clean_manifest", manifest_produced_at),
        ("publish_done", done_produced_at),
    ):
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            reasons.append(f"{name}_produced_at_invalid")
            continue
        timestamp = pd.Timestamp(parsed)
        if timestamp.date() > score_day:
            # A clean snapshot produced on a later calendar day is an explicit
            # backfill.  It cannot be reconstructed as score-day evidence.
            reasons.append("availability_late_backfill")
        elif timestamp.date() != score_day:
            reasons.append("availability_not_same_score_day")
        elif timestamp.time() < STRICT_CUTOFF_CLOCK:
            reasons.append("availability_before_cutoff")

    asset_status = manifest.get("assets_status")
    if not isinstance(asset_status, dict) or str(asset_status.get("cbond", "")).strip().lower() != "success":
        reasons.append("clean_manifest_cbond_not_success")

    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        reasons.append("clean_manifest_validation_not_passed")

    assets = manifest.get("assets")
    cbond_asset = assets.get("cbond") if isinstance(assets, dict) else None
    if not isinstance(cbond_asset, dict):
        reasons.append("clean_manifest_cbond_asset_missing")
    else:
        declared_path = cbond_asset.get("file_path")
        if isinstance(declared_path, str):
            audit["manifest_cbond_file_path"] = declared_path.strip()
        if not isinstance(declared_path, str) or not declared_path.strip():
            reasons.append("manifest_cbond_file_path_missing")
        elif not _same_path(declared_path, Path(str(audit["source_path"]))):
            reasons.append("manifest_cbond_file_path_mismatch")

        declared_rows = cbond_asset.get("row_count")
        if declared_rows in (None, "") or isinstance(declared_rows, bool):
            reasons.append("manifest_cbond_row_count_missing")
        else:
            try:
                parsed_rows = int(declared_rows)
            except (TypeError, ValueError):
                reasons.append("manifest_cbond_row_count_invalid")
            else:
                audit["manifest_row_count"] = parsed_rows
                if parsed_rows <= 0:
                    reasons.append("manifest_cbond_row_count_nonpositive")

    if done.get("ready") is not True:
        reasons.append("publish_done_not_ready")
    allow_partial_manifest = done.get("allow_partial_manifest")
    if isinstance(allow_partial_manifest, bool):
        audit["publish_done_allow_partial_manifest"] = allow_partial_manifest
    if allow_partial_manifest is not False:
        reasons.append("publish_done_allow_partial_manifest_not_false")
    required_datasets = done.get("require_datasets")
    if isinstance(required_datasets, list):
        audit["publish_done_require_datasets"] = json.dumps(required_datasets, ensure_ascii=False)
    if not isinstance(required_datasets, list) or "clean" not in {
        str(item).strip().lower() for item in required_datasets
    }:
        reasons.append("publish_done_require_datasets_missing_clean")
    if str(done.get("trade_day", "")).strip() != expected_day:
        reasons.append("publish_done_trade_day_mismatch")
    if not done_run_id:
        reasons.append("publish_done_run_id_missing")
    elif done_run_id != manifest_run_id:
        reasons.append("manifest_done_run_id_mismatch")

    return manifest, list(dict.fromkeys(reasons))


def _source_evidence(
    audit: dict[str, object]
) -> list[str]:
    source_path = Path(str(audit["source_path"]))
    reasons: list[str] = []
    if not source_path.is_file():
        reasons.append("canonical_snapshot_missing")
        return reasons

    audit["source_exists"] = True
    try:
        audit["source_size_bytes"] = int(source_path.stat().st_size)
    except OSError as exc:
        reasons.append(f"source_stat_failed:{type(exc).__name__}")
        return reasons
    if int(audit["source_size_bytes"] or 0) <= 0:
        reasons.append("source_empty")
        return reasons

    try:
        metadata = ParquetFile(source_path).metadata
        source_rows = int(metadata.num_rows)
        audit["source_row_count"] = source_rows
        audit["source_schema_sha256"] = _sha256_text(str(metadata.schema))
    except Exception as exc:  # pyarrow surface differs for corrupt/unsupported files.
        reasons.append(f"source_metadata_failed:{type(exc).__name__}")
        return reasons

    if int(audit["source_row_count"] or 0) <= 0:
        reasons.append("source_row_count_nonpositive")
    declared_rows = audit["manifest_row_count"]
    if declared_rows is None:
        reasons.append("manifest_cbond_row_count_missing")
    elif int(declared_rows) != int(audit["source_row_count"]):
        reasons.append("manifest_cbond_row_count_mismatch")

    return reasons


def _hash_evidence(audit: dict[str, object]) -> list[str]:
    source_path = Path(str(audit["source_path"]))
    try:
        source_hash = _sha256_file(source_path)
    except OSError as exc:
        return [f"source_hash_failed:{type(exc).__name__}"]
    if not source_hash:
        return ["source_hash_missing"]
    audit["source_sha256"] = source_hash
    return []


def _normalise_state_row(row: dict[str, object], *, score_day: date) -> tuple[dict[str, object] | None, list[str]]:
    missing: list[str] = []
    normalised: dict[str, object] = {"trade_date": score_day.isoformat()}
    try:
        valid_count = int(row.get("valid_count", 0))
    except (TypeError, ValueError):
        valid_count = 0
    normalised["valid_count"] = valid_count
    if valid_count <= 0:
        missing.append("valid_count")

    for column in STATE_FEATURE_COLUMNS:
        raw = row.get(column)
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            missing.append(column)
            continue
        if not math.isfinite(value):
            missing.append(column)
            continue
        normalised[column] = value

    if missing:
        return None, missing
    return normalised, []


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_strict_t1429_state_history(
    *,
    clean_root: str | Path,
    manifest_root: str | Path,
    scratch_root: str | Path,
    expected_days: Iterable[date | str],
    calendar_evidence_path: str | Path | None = None,
    calendar_source_kind: str = "inline_expected_days",
    calendar_source_path: str | Path | None = None,
    state_builder: Callable[..., dict[str, object]] = build_t1430_market_state_feature_row,
) -> StrictStateHistoryResult:
    """Build a new historical reconstruction from immutable clean snapshots.

    The function intentionally writes a partial history if some expected days
    fail evidence checks.  Same-day post-cutoff V1 records receive only a
    clearly labelled historical-reconstruction row; no row is forward-PIT
    certified.  Failed days are represented only in the audit and never
    receive a surrogate feature row or a rolling/fallback label.
    """

    clean = Path(clean_root).resolve()
    manifests = Path(manifest_root).resolve()
    if not clean.is_dir():
        raise FileNotFoundError(f"clean_root missing: {clean}")
    if not manifests.is_dir():
        raise FileNotFoundError(f"manifest_root missing: {manifests}")
    paths = _validate_scratch_root(scratch_root, clean_root=clean, manifest_root=manifests)
    days = _normalise_expected_days(expected_days)

    calendar_path: Path | None = None
    if calendar_evidence_path is not None:
        calendar_path = Path(calendar_evidence_path)
        if not calendar_path.is_file():
            raise FileNotFoundError(f"calendar evidence file missing: {calendar_path}")
        calendar_sha256 = _sha256_file(calendar_path)
    else:
        calendar_sha256 = _json_hash([item.isoformat() for item in days])

    # Freeze the exact requested score calendar before reading state sources.
    # This gives a score-root caller an immutable artifact to use for the
    # later strict selector and makes a partial history impossible to mistake
    # for a full recent-day calendar.
    calendar_frame = pd.DataFrame({"trade_date": [item.isoformat() for item in days]})
    _atomic_write_csv(calendar_frame, paths.calendar_path)
    frozen_calendar_sha256 = _sha256_file(paths.calendar_path)

    state_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for score_day in days:
        audit = _base_audit_row(score_day=score_day, clean_root=clean, manifest_root=manifests)
        _, availability_reasons = _availability_evidence(audit, score_day=score_day)
        audit["availability_evidence_ok"] = not availability_reasons
        if availability_reasons:
            audit["reason"] = ";".join(availability_reasons)
            audit_rows.append(audit)
            continue

        source_reasons = _source_evidence(audit)
        audit["source_evidence_ok"] = not any(
            reason.startswith(("canonical_snapshot", "source_", "manifest_source_path"))
            for reason in source_reasons
        )
        audit["row_evidence_ok"] = not any(
            "row_count" in reason or reason.startswith("source_metadata") for reason in source_reasons
        )
        if source_reasons:
            audit["reason"] = ";".join(source_reasons)
            audit_rows.append(audit)
            continue

        hash_reasons = _hash_evidence(audit)
        audit["hash_evidence_ok"] = not hash_reasons
        if hash_reasons:
            audit["reason"] = ";".join(hash_reasons)
            audit_rows.append(audit)
            continue

        try:
            raw_state_row = state_builder(
                clean_root=clean,
                score_day=score_day,
                cutoff_time=STRICT_CUTOFF_TIME,
            )
        except Exception as exc:
            audit["reason"] = f"state_build_failed:{type(exc).__name__}"
            audit_rows.append(audit)
            continue
        if not isinstance(raw_state_row, dict):
            audit["reason"] = "state_build_not_object"
            audit_rows.append(audit)
            continue

        state_row, missing_columns = _normalise_state_row(raw_state_row, score_day=score_day)
        audit["state_valid_count"] = raw_state_row.get("valid_count")
        audit["missing_state_columns"] = ";".join(missing_columns)
        if state_row is None:
            audit["reason"] = "state_row_incomplete"
            audit_rows.append(audit)
            continue

        audit["state_row_evidence_ok"] = True
        audit["state_row_sha256"] = _json_hash(state_row)
        audit["outcome"] = HISTORICAL_RECONSTRUCTION_STATUS
        audit["provenance_classification"] = HISTORICAL_RECONSTRUCTION_STATUS
        audit["forward_pit_certified"] = False
        audit["reason"] = "same_day_v1_post_cutoff_reconstruction_not_forward_pit_certified"
        state_rows.append(state_row)
        audit_rows.append(audit)

    state_frame = pd.DataFrame(state_rows, columns=STATE_HISTORY_COLUMNS)
    audit_frame = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    _atomic_write_csv(state_frame, paths.state_path)
    _atomic_write_csv(audit_frame, paths.audit_path)

    historical_reconstruction_days = int(
        (audit_frame["outcome"] == HISTORICAL_RECONSTRUCTION_STATUS).sum()
    )
    blocked = audit_frame.loc[audit_frame["outcome"] == BLOCKED_STATUS, "reason"].astype(str)
    blocked_days = int((audit_frame["outcome"] == BLOCKED_STATUS).sum())
    state_sha256 = _sha256_file(paths.state_path)
    audit_sha256 = _sha256_file(paths.audit_path)
    manifest_status = (
        HISTORICAL_RECONSTRUCTION_STATUS
        if historical_reconstruction_days
        else BLOCKED_STATUS
    )
    manifest_payload: dict[str, object] = {
        "schema_version": 1,
        "status": manifest_status,
        "purpose": (
            "research-only 14:29 historical state reconstruction; no repair or fallback rows; "
            "not a forward-PIT-certified history"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strict_cutoff_time": STRICT_CUTOFF_TIME,
        "state_feature_set": STATE_FEATURE_SET,
        "state_feature_columns": list(STATE_FEATURE_COLUMNS),
        "expected_days": {
            "count": len(days),
            "source_kind": calendar_source_kind,
            "source_path": (
                str(calendar_source_path)
                if calendar_source_path is not None
                else (str(calendar_path) if calendar_path is not None else "inline_expected_days")
            ),
            "source_sha256": calendar_sha256,
            "frozen_calendar_path": str(paths.calendar_path),
            "frozen_calendar_sha256": frozen_calendar_sha256,
        },
        "inputs": {
            "clean_root": str(clean),
            "manifest_root": str(manifests),
            "historical_reconstruction_contract": (
                "same-day successful cbond_on_t1430_v1 clean manifest with required_profile "
                "cbond_on_live_t1430, validation.passed=true, cbond assets.file_path and row_count "
                "matching the canonical snapshot, plus ready publish .done with matching run_id; "
                "the .done must require clean and disallow partial manifests; both produced_at timestamps must be "
                "on score_day and at or after 14:29"
            ),
            "source_contract": "canonical clean snapshot/cbond/YYYY-MM/YYYYMMDD.parquet only",
        },
        "outputs": {
            "state_path": str(paths.state_path),
            "state_sha256": state_sha256,
            "audit_path": str(paths.audit_path),
            "audit_sha256": audit_sha256,
            "calendar_path": str(paths.calendar_path),
            "calendar_sha256": frozen_calendar_sha256,
        },
        "counts": {
            "requested_days": len(days),
            "built_days": historical_reconstruction_days,
            "historical_reconstruction_days": historical_reconstruction_days,
            "blocked_days": blocked_days,
            "forward_pit_certified_days": 0,
        },
        "blocked_reason_counts": dict(sorted(Counter(blocked).items())),
        "certification": {
            "status": manifest_status,
            "forward_pit_certified": False,
            "forward_pit_certified_days": 0,
            "reason": (
                "DataHub v1 records final post-cutoff artifacts, not an immutable version or attestation "
                "demonstrably available at the 14:29 score cutoff."
            ),
            "consumer_policy": (
                "A strict Similar60 candidate must reject this manifest. It may be used only under a separately "
                "approved and explicitly labelled non-historical-PIT reconstruction or forward-shadow study."
            ),
        },
        "invariant": (
            "No emitted state row is forward-PIT certified. Any requested day missing V1 reconstruction, source, "
            "row, hash, or complete-state evidence is audit-only with outcome=blocked; it is neither repaired nor "
            "labelled as a fallback."
        ),
    }
    _atomic_write_json(manifest_payload, paths.manifest_path)
    return StrictStateHistoryResult(
        paths=paths,
        requested_days=len(days),
        built_days=historical_reconstruction_days,
        historical_reconstruction_days=historical_reconstruction_days,
        blocked_days=blocked_days,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-root", required=True, help="read-only DataHub clean_data root")
    parser.add_argument("--manifest-root", required=True, help="read-only DataHub manifests root")
    parser.add_argument("--scratch-root", required=True, help="new explicit research scratch root")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--expected-days-csv",
        help="CSV with the exact intended score calendar in a trade_date column",
    )
    input_group.add_argument(
        "--score-root",
        help="read-only Regsim score root; discovers canonical YYYY-MM/YYYY-MM-DD.csv days only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.expected_days_csv:
        expected_days = load_expected_days_csv(args.expected_days_csv)
        calendar_evidence_path: str | Path | None = args.expected_days_csv
        calendar_source_kind = "expected_days_csv"
        calendar_source_path: str | Path = args.expected_days_csv
    else:
        expected_days = discover_score_calendar(args.score_root)
        calendar_evidence_path = None
        calendar_source_kind = "score_root"
        calendar_source_path = args.score_root
    result = build_strict_t1429_state_history(
        clean_root=args.clean_root,
        manifest_root=args.manifest_root,
        scratch_root=args.scratch_root,
        expected_days=expected_days,
        calendar_evidence_path=calendar_evidence_path,
        calendar_source_kind=calendar_source_kind,
        calendar_source_path=calendar_source_path,
    )
    print(
        json.dumps(
            {
                "state_path": str(result.paths.state_path),
                "audit_path": str(result.paths.audit_path),
                "calendar_path": str(result.paths.calendar_path),
                "manifest_path": str(result.paths.manifest_path),
                "requested_days": result.requested_days,
                "built_days": result.built_days,
                "historical_reconstruction_days": result.historical_reconstruction_days,
                "blocked_days": result.blocked_days,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
