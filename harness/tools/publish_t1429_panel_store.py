"""Publish the one shared, read-only T1430 PanelStore.

This is deliberately a *publisher*, not a factor, model, live, or scheduler
entrypoint.  It is the only new code path that may materialize
``D:/cbond_on/panel_data``.  Consumers must later use the published day
manifest and ``.done`` marker as their read-only admission boundary.

The logical panel is named T1430, but its physical market-data cutoff is
14:29:00 (``lead_minutes=1``).  Both cbond and stock panels are published as
one per-day bundle: parquet files are staged and validated first, then the
manifest and finally the ``.done`` marker are atomically written.  The done
marker is the commit point; an incomplete pair is never admitted.

No-write preflight is the default.  ``--execute --public-root`` is the only
way to write the canonical public root.  ``--execute --scratch-root`` is
available only below ``D:/cbond_on/research_scratch`` for a bounded smoke.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd


# Publishing a shared data artifact must not leave interpreter cache files in
# the repository merely because this harness script was invoked directly.
sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.config import ScheduleConfig, SnapshotConfig  # noqa: E402
from cbond_on.core.config import load_config_file, parse_date  # noqa: E402
from cbond_on.core.trading_days import list_trading_days_from_raw  # noqa: E402
from cbond_on.infra.data.panel import (  # noqa: E402
    _iter_existing_snapshot_days,
    build_panel_data,
)


PUBLIC_PANEL_ROOT = Path(r"D:/cbond_on/panel_data")
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
DATAHUB_RAW_ROOT = Path(r"D:/cbond_data_hub/raw_data")
DATAHUB_CLEAN_ROOT = Path(r"D:/cbond_data_hub/clean_data")
DATAHUB_MANIFEST_ROOT = Path(r"D:/cbond_data_hub/manifests")

PANEL_NAME = "T1430"
LOGICAL_PANEL_TIME = "14:30"
PHYSICAL_CUTOFF_TIME = "14:29:00"
LEAD_MINUTES = 1
COUNT_POINTS = 5000
MAX_LOOKBACK_DAYS = 4
ASSETS = ("cbond", "stock")
DEFAULT_START = date(2024, 1, 3)
DEFAULT_END = date(2026, 7, 30)
SCRATCH_SMOKE_MAX_DAYS = 5

CONTRACT_SCHEMA = "cbond_on_panel_store_contract/v1"
MANIFEST_SCHEMA = "cbond_on_panel_store_manifest/v1"
DONE_SCHEMA = "cbond_on_panel_store_done/v1"


class PanelStorePublisherError(RuntimeError):
    """The shared panel publication contract could not be established."""


@dataclass(frozen=True)
class PanelStorePlan:
    target_root: Path
    target_kind: str
    start: date
    end: date
    days: tuple[date, ...]
    workers: int
    panel_cfg: dict[str, Any]
    contract: dict[str, Any]
    contract_sha256: str
    calendar: dict[str, Any]
    source_by_day: dict[str, dict[str, Any]]


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.normpath(str(_resolved(left)))) == os.path.normcase(
        os.path.normpath(str(_resolved(right)))
    )


def _is_strict_child(child: str | Path, parent: str | Path) -> bool:
    try:
        _resolved(child).relative_to(_resolved(parent))
        return _resolved(child) != _resolved(parent)
    except ValueError:
        return False


def _canonical_json_bytes(payload: Mapping[str, Any] | Sequence[Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_required(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - exercised by callers' failure paths.
        raise PanelStorePublisherError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise PanelStorePublisherError(f"{label} must be a JSON object: {path}")
    return dict(raw)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one visibility marker atomically within its destination directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(payload) + b"\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalise_assets(raw: object) -> list[str]:
    if isinstance(raw, str):
        values = raw.replace(";", ",").split(",")
    elif isinstance(raw, (list, tuple)):
        values = raw
    else:
        values = []
    return [str(value).strip().lower() for value in values if str(value).strip()]


def _contract_from_panel_config(panel_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the repository's public panel contract is exact."""

    cfg = dict(panel_cfg)
    if str(cfg.get("panel_name", "")).strip() != PANEL_NAME:
        raise PanelStorePublisherError(f"panel_config.panel_name must be {PANEL_NAME!r}")
    if str(cfg.get("panel_mode", "")).strip().lower() != "snapshot_sequence":
        raise PanelStorePublisherError("panel_config.panel_mode must be 'snapshot_sequence'")
    if int(cfg.get("lead_minutes", -1)) != LEAD_MINUTES:
        raise PanelStorePublisherError(
            f"panel_config.lead_minutes must be {LEAD_MINUTES} for physical {PHYSICAL_CUTOFF_TIME}"
        )
    if int(cfg.get("count_points", -1)) != COUNT_POINTS:
        raise PanelStorePublisherError(f"panel_config.count_points must be {COUNT_POINTS}")
    if int(cfg.get("max_lookback_days", -1)) != MAX_LOOKBACK_DAYS:
        raise PanelStorePublisherError(f"panel_config.max_lookback_days must be {MAX_LOOKBACK_DAYS}")
    if _normalise_assets(cfg.get("assets")) != list(ASSETS):
        raise PanelStorePublisherError(
            f"panel_config.assets must be exactly {list(ASSETS)!r}, got {_normalise_assets(cfg.get('assets'))!r}"
        )

    schedule_raw = cfg.get("schedule")
    if not isinstance(schedule_raw, Mapping):
        raise PanelStorePublisherError("panel_config.schedule must be an object")
    windows = schedule_raw.get("windows")
    expected_windows = [{"start": LOGICAL_PANEL_TIME, "end": LOGICAL_PANEL_TIME}]
    if not isinstance(windows, list) or len(windows) != 1:
        raise PanelStorePublisherError("panel_config.schedule.windows must contain exactly one T1430 window")
    actual_windows = [
        {"start": str(dict(item).get("start", "")), "end": str(dict(item).get("end", ""))}
        for item in windows
        if isinstance(item, Mapping)
    ]
    if actual_windows != expected_windows:
        raise PanelStorePublisherError(
            f"panel_config.schedule.windows must be {expected_windows!r}, got {actual_windows!r}"
        )

    snapshot = cfg.get("snapshot")
    if snapshot is not None and not isinstance(snapshot, Mapping):
        raise PanelStorePublisherError("panel_config.snapshot must be an object when provided")
    overrides = cfg.get("asset_overrides")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise PanelStorePublisherError("panel_config.asset_overrides must be an object when provided")
    for asset, asset_cfg_raw in dict(overrides or {}).items():
        asset = str(asset).strip().lower()
        if asset not in ASSETS or not isinstance(asset_cfg_raw, Mapping):
            raise PanelStorePublisherError(f"panel_config.asset_overrides.{asset} is not a supported asset object")
        asset_cfg = dict(asset_cfg_raw)
        if int(asset_cfg.get("count_points", COUNT_POINTS)) != COUNT_POINTS:
            raise PanelStorePublisherError(
                f"panel_config.asset_overrides.{asset}.count_points must be {COUNT_POINTS}"
            )
        if int(asset_cfg.get("max_lookback_days", MAX_LOOKBACK_DAYS)) != MAX_LOOKBACK_DAYS:
            raise PanelStorePublisherError(
                f"panel_config.asset_overrides.{asset}.max_lookback_days must be {MAX_LOOKBACK_DAYS}"
            )
    return {
        "schema_version": CONTRACT_SCHEMA,
        "panel_name": PANEL_NAME,
        "logical_panel_time": LOGICAL_PANEL_TIME,
        "physical_cutoff_time": PHYSICAL_CUTOFF_TIME,
        "lead_minutes": LEAD_MINUTES,
        "panel_mode": "snapshot_sequence",
        "assets": list(ASSETS),
        "count_points": COUNT_POINTS,
        "max_lookback_days": MAX_LOOKBACK_DAYS,
        "schedule_windows": expected_windows,
        "snapshot": dict(snapshot or {}),
        "asset_overrides": dict(overrides or {}),
        "snapshot_columns": cfg.get("snapshot_columns"),
    }


def _snapshot_path(clean_root: Path, day: date, asset: str) -> Path:
    return clean_root / "snapshot" / asset / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"


def _path_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _datahub_day_evidence(day: date) -> dict[str, Any]:
    """Validate the exact DataHub day before it may enter the shared cache."""

    manifest_path = DATAHUB_MANIFEST_ROOT / "clean" / f"{day:%Y-%m-%d}.json"
    done_path = DATAHUB_MANIFEST_ROOT / "publish" / f"{day:%Y-%m-%d}.done"
    manifest = _read_json_required(manifest_path, label="DataHub clean manifest")
    done = _read_json_required(done_path, label="DataHub publish done marker")
    assets_status = manifest.get("assets_status")
    if not isinstance(assets_status, Mapping):
        raise PanelStorePublisherError(f"DataHub clean manifest has no assets_status for {day}")
    if str(manifest.get("status", "")).strip().lower() != "success":
        raise PanelStorePublisherError(f"DataHub clean manifest is not successful for {day}")
    for asset in ASSETS:
        if str(assets_status.get(asset, "")).strip().lower() != "success":
            raise PanelStorePublisherError(f"DataHub clean manifest asset={asset} is not successful for {day}")

    manifest_day = str(manifest.get("trade_day", "")).strip()
    done_day = str(done.get("trade_day", "")).strip()
    manifest_run = str(manifest.get("run_id", "")).strip()
    done_run = str(done.get("run_id", "")).strip()
    if manifest_day != day.isoformat() or done_day != day.isoformat():
        raise PanelStorePublisherError(f"DataHub manifest/done day mismatch for {day}")
    if not manifest_run or manifest_run != done_run:
        raise PanelStorePublisherError(f"DataHub manifest/done run_id mismatch for {day}")
    if done.get("ready") is not True:
        raise PanelStorePublisherError(f"DataHub publish done marker is not ready for {day}")

    snapshots: dict[str, dict[str, Any]] = {}
    for asset in ASSETS:
        path = _snapshot_path(DATAHUB_CLEAN_ROOT, day, asset)
        if not path.is_file():
            raise FileNotFoundError(f"DataHub clean {asset} snapshot is missing for {day}: {path}")
        snapshots[asset] = _path_stat(path)
    return {
        "trade_day": day.isoformat(),
        "datahub_run_id": manifest_run,
        "clean_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": _sha256_file(manifest_path),
            "status": str(manifest.get("status", "")),
            "required_profile": str(manifest.get("required_profile", "")),
        },
        "publish_done": {
            "path": done_path.as_posix(),
            "sha256": _sha256_file(done_path),
            "ready": bool(done.get("ready")),
        },
        "clean_snapshots": snapshots,
    }


def _days_sha256(days: Sequence[date]) -> str:
    return _sha256_bytes("\n".join(day.isoformat() for day in days).encode("utf-8"))


def _calendar_and_sources(start: date, end: date) -> tuple[tuple[date, ...], dict[str, Any], dict[str, dict[str, Any]]]:
    raw_days = list_trading_days_from_raw(
        DATAHUB_RAW_ROOT,
        start,
        end,
        kind="snapshot",
        asset="cbond",
    )
    raw_stock_days = list_trading_days_from_raw(
        DATAHUB_RAW_ROOT,
        start,
        end,
        kind="snapshot",
        asset="stock",
    )
    clean_cbond_days = _iter_existing_snapshot_days(DATAHUB_CLEAN_ROOT, start, end, asset="cbond")
    clean_stock_days = _iter_existing_snapshot_days(DATAHUB_CLEAN_ROOT, start, end, asset="stock")
    if not raw_days:
        raise PanelStorePublisherError("DataHub raw cbond snapshot calendar is empty")
    if raw_days != clean_cbond_days:
        raise PanelStorePublisherError(
            "raw and clean cbond snapshot calendars differ: "
            f"raw_only={[day.isoformat() for day in sorted(set(raw_days) - set(clean_cbond_days))[:5]]} "
            f"clean_only={[day.isoformat() for day in sorted(set(clean_cbond_days) - set(raw_days))[:5]]}"
        )
    if raw_days != raw_stock_days:
        raise PanelStorePublisherError(
            "raw cbond and raw stock snapshot calendars differ: "
            f"cbond_only={[day.isoformat() for day in sorted(set(raw_days) - set(raw_stock_days))[:5]]} "
            f"stock_only={[day.isoformat() for day in sorted(set(raw_stock_days) - set(raw_days))[:5]]}"
        )
    if raw_days != clean_stock_days:
        raise PanelStorePublisherError(
            "raw and clean stock snapshot calendars differ: "
            f"raw_only={[day.isoformat() for day in sorted(set(raw_days) - set(clean_stock_days))[:5]]} "
            f"clean_only={[day.isoformat() for day in sorted(set(clean_stock_days) - set(raw_days))[:5]]}"
        )
    source_by_day = {day.isoformat(): _datahub_day_evidence(day) for day in raw_days}
    calendar = {
        "source": "datahub_raw_clean_publish_cbond_stock_parity",
        "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
        "raw_cbond_snapshot_days": [day.isoformat() for day in raw_days],
        "raw_stock_snapshot_days": [day.isoformat() for day in raw_stock_days],
        "clean_cbond_snapshot_days": [day.isoformat() for day in clean_cbond_days],
        "clean_stock_snapshot_days": [day.isoformat() for day in clean_stock_days],
        "publish_ready_days": [day.isoformat() for day in raw_days],
        "days_sha256": _days_sha256(raw_days),
    }
    return tuple(raw_days), calendar, source_by_day


def _resolve_target(*, execute: bool, public_root: bool, scratch_root: str | None) -> tuple[Path, str]:
    if public_root and scratch_root:
        raise PanelStorePublisherError("--public-root and --scratch-root are mutually exclusive")
    if public_root:
        return _resolved(PUBLIC_PANEL_ROOT), "public"
    if scratch_root:
        target = _resolved(scratch_root)
        if not _is_strict_child(target, RESEARCH_SCRATCH_PARENT):
            raise PanelStorePublisherError(
                f"--scratch-root must resolve strictly below {RESEARCH_SCRATCH_PARENT.as_posix()}"
            )
        if execute and target.exists():
            raise FileExistsError(
                "scratch publication requires a previously absent root; refusing to reuse "
                f"{target.as_posix()}"
            )
        return target, "scratch"
    if execute:
        raise PanelStorePublisherError("--execute requires exactly one of --public-root or --scratch-root")
    # The default no-write plan describes the canonical destination but cannot
    # create it or write into it.
    return _resolved(PUBLIC_PANEL_ROOT), "public"


def preflight(
    *,
    start_text: str | None = None,
    end_text: str | None = None,
    workers: int = 1,
    execute: bool = False,
    public_root: bool = False,
    scratch_root: str | None = None,
) -> PanelStorePlan:
    """Return a fully validated publication plan without writing anything."""

    start = parse_date(start_text) if start_text else DEFAULT_START
    end = parse_date(end_text) if end_text else DEFAULT_END
    if start > end:
        raise ValueError("start date must be <= end date")
    if int(workers) < 1:
        raise ValueError("workers must be >= 1")
    target_root, target_kind = _resolve_target(
        execute=execute,
        public_root=public_root,
        scratch_root=scratch_root,
    )
    panel_cfg = dict(load_config_file("panel"))
    contract = _contract_from_panel_config(panel_cfg)
    contract_sha256 = _sha256_bytes(_canonical_json_bytes(contract))
    days, calendar, source_by_day = _calendar_and_sources(start, end)
    if target_kind == "scratch" and len(days) > SCRATCH_SMOKE_MAX_DAYS:
        raise PanelStorePublisherError(
            "scratch publication is smoke-only; requested "
            f"{len(days)} days exceeds SCRATCH_SMOKE_MAX_DAYS={SCRATCH_SMOKE_MAX_DAYS}"
        )
    return PanelStorePlan(
        target_root=target_root,
        target_kind=target_kind,
        start=start,
        end=end,
        days=days,
        workers=max(1, int(workers)),
        panel_cfg=panel_cfg,
        contract=contract,
        contract_sha256=contract_sha256,
        calendar=calendar,
        source_by_day=source_by_day,
    )


def _contract_path(root: Path) -> Path:
    return root / "contracts" / f"{PANEL_NAME}.json"


def _panel_path(root: Path, day: date, asset: str) -> Path:
    return root / "panels" / asset / PANEL_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"


def _day_manifest_path(root: Path, day: date) -> Path:
    return root / "manifests" / PANEL_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.json"


def _day_done_path(root: Path, day: date) -> Path:
    return root / "publish" / PANEL_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.done"


def _assert_existing_contract_compatible(plan: PanelStorePlan) -> None:
    path = _contract_path(plan.target_root)
    if not path.exists():
        # A legacy/nonempty root without this contract is deliberately not
        # adopted by the new publisher.
        visible_entries = (
            [
                entry
                for entry in plan.target_root.iterdir()
                if entry.name not in {".panel_publisher.lock", ".staging"}
            ]
            if plan.target_root.exists()
            else []
        )
        if visible_entries:
            raise PanelStorePublisherError(
                "target PanelStore is nonempty but has no publisher contract; refusing to adopt legacy files"
            )
        return
    existing = _read_json_required(path, label="PanelStore contract")
    expected = dict(plan.contract)
    existing_hash = str(existing.get("contract_sha256", "")).strip().lower()
    if existing_hash != plan.contract_sha256:
        raise PanelStorePublisherError(
            "existing PanelStore contract differs from the requested T1429/T1430 contract"
        )
    if {key: existing.get(key) for key in expected} != expected:
        raise PanelStorePublisherError("existing PanelStore contract payload differs despite its declared hash")


def _assert_days_fresh(plan: PanelStorePlan) -> None:
    conflicts: list[str] = []
    for day in plan.days:
        for asset in ASSETS:
            path = _panel_path(plan.target_root, day, asset)
            if path.exists():
                conflicts.append(path.as_posix())
        for path in (_day_manifest_path(plan.target_root, day), _day_done_path(plan.target_root, day)):
            if path.exists():
                conflicts.append(path.as_posix())
    if conflicts:
        raise FileExistsError(
            "PanelStore publisher never overwrites an existing day; conflicting paths: "
            + ", ".join(conflicts[:8])
        )


def _acquire_lock(root: Path, *, run_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".panel_publisher.lock"
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise PanelStorePublisherError(f"PanelStore publisher lock already exists: {lock_path}") from exc
    try:
        os.write(descriptor, f"run_id={run_id}\npid={os.getpid()}\n".encode("utf-8"))
    finally:
        os.close(descriptor)
    return lock_path


def _release_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock_path.unlink()


def _build_staging(plan: PanelStorePlan, *, stage_root: Path) -> dict[str, dict[str, int]]:
    schedule = ScheduleConfig.from_dict(dict(plan.panel_cfg["schedule"])).to_schedule()
    snapshot_cfg = SnapshotConfig.from_dict(dict(plan.panel_cfg.get("snapshot", {})))
    asset_overrides = plan.panel_cfg.get("asset_overrides", {})
    if not isinstance(asset_overrides, Mapping):  # guarded during preflight.
        raise PanelStorePublisherError("panel_config.asset_overrides must be an object")
    window_minutes_raw = plan.panel_cfg.get("window_minutes", [15])
    if isinstance(window_minutes_raw, (list, tuple)):
        if len(window_minutes_raw) != 1:
            raise PanelStorePublisherError("T1430 publisher requires exactly one window_minutes value")
        window_minutes = int(window_minutes_raw[0])
    else:
        window_minutes = int(window_minutes_raw)

    results: dict[str, dict[str, int]] = {}
    for asset in ASSETS:
        asset_cfg = asset_overrides.get(asset, {})
        if not isinstance(asset_cfg, Mapping):
            raise PanelStorePublisherError(f"panel_config.asset_overrides.{asset} must be an object")
        result = build_panel_data(
            DATAHUB_CLEAN_ROOT,
            stage_root,
            DATAHUB_RAW_ROOT,
            plan.start,
            plan.end,
            schedule,
            snapshot_cfg,
            window_minutes=window_minutes,
            panel_name=PANEL_NAME,
            asset=asset,
            overwrite=False,
            panel_mode="snapshot_sequence",
            count_points=int(asset_cfg.get("count_points", COUNT_POINTS)),
            max_lookback_days=int(asset_cfg.get("max_lookback_days", MAX_LOOKBACK_DAYS)),
            snapshot_columns=plan.panel_cfg.get("snapshot_columns"),
            lead_minutes=LEAD_MINUTES,
            workers=plan.workers,
            compute_cfg=plan.panel_cfg.get("compute"),
        )
        results[asset] = {
            "written": int(result.written),
            "skipped": int(result.skipped),
            "missing_snapshot_days": int(result.missing_snapshot_days),
        }
        if int(result.written) != len(plan.days) or int(result.skipped) != 0 or int(result.missing_snapshot_days) != 0:
            raise PanelStorePublisherError(
                f"staged {asset} panel build is incomplete: {results[asset]}, expected_written={len(plan.days)}"
            )
    return results


def _verify_panel_file(path: Path, *, day: date, asset: str) -> dict[str, Any]:
    """Validate index/time visibility while reading only the needed field."""

    if not path.is_file():
        raise FileNotFoundError(f"staged {asset} panel is missing: {path}")
    frame = pd.read_parquet(path, columns=["trade_time"])
    if frame.empty:
        raise PanelStorePublisherError(f"staged {asset} panel is empty for {day}")
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["dt", "code", "seq"]:
        raise PanelStorePublisherError(f"staged {asset} panel index must be (dt, code, seq) for {day}")
    if "trade_time" not in frame.columns:
        raise PanelStorePublisherError(f"staged {asset} panel lacks trade_time for {day}")

    logical = pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce")
    expected_logical = pd.Timestamp.combine(day, datetime.strptime(LOGICAL_PANEL_TIME, "%H:%M").time())
    if logical.isna().any() or not bool((logical == expected_logical).all()):
        raise PanelStorePublisherError(f"staged {asset} panel has a non-T1430 logical index for {day}")

    physical = pd.to_datetime(frame["trade_time"], errors="coerce")
    cutoff = pd.Timestamp.combine(day, datetime.strptime(PHYSICAL_CUTOFF_TIME, "%H:%M:%S").time())
    if physical.isna().any() or bool((physical > cutoff).any()):
        raise PanelStorePublisherError(
            f"staged {asset} panel violates physical cutoff {PHYSICAL_CUTOFF_TIME} for {day}"
        )

    codes = frame.index.get_level_values("code")
    seq = pd.to_numeric(pd.Series(frame.index.get_level_values("seq")), errors="coerce")
    if seq.isna().any():
        raise PanelStorePublisherError(f"staged {asset} panel has non-numeric seq for {day}")
    counts = pd.Series(codes).value_counts(sort=False)
    if counts.empty or not bool((counts == COUNT_POINTS).all()):
        raise PanelStorePublisherError(
            f"staged {asset} panel does not retain exactly {COUNT_POINTS} rows per code for {day}"
        )
    seq_min = seq.groupby(pd.Series(codes).to_numpy()).min()
    seq_max = seq.groupby(pd.Series(codes).to_numpy()).max()
    if not bool((seq_min == 0).all()) or not bool((seq_max == COUNT_POINTS - 1).all()):
        raise PanelStorePublisherError(f"staged {asset} panel has an invalid seq range for {day}")

    stat = path.stat()
    result = {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": int(stat.st_size),
        "rows": int(len(frame)),
        "codes": int(len(counts)),
        "logical_dt": expected_logical.isoformat(),
        "physical_cutoff": cutoff.isoformat(),
        "physical_min": pd.Timestamp(physical.min()).isoformat(),
        "physical_max": pd.Timestamp(physical.max()).isoformat(),
        "index_names": list(frame.index.names),
    }
    del frame
    return result


def _source_evidence_is_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return _canonical_json_bytes(before) == _canonical_json_bytes(after)


def _commit_day(
    plan: PanelStorePlan,
    *,
    stage_root: Path,
    day: date,
    run_id: str,
) -> dict[str, Any]:
    staged_assets = {
        asset: _verify_panel_file(_panel_path(stage_root, day, asset), day=day, asset=asset)
        for asset in ASSETS
    }
    expected_source = plan.source_by_day[day.isoformat()]
    current_source = _datahub_day_evidence(day)
    if not _source_evidence_is_unchanged(expected_source, current_source):
        raise PanelStorePublisherError(f"DataHub source evidence changed while staging {day}; refusing publication")

    destinations = {asset: _panel_path(plan.target_root, day, asset) for asset in ASSETS}
    existing = [path.as_posix() for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing to replace already-published panel files: " + ", ".join(existing))

    moved: list[Path] = []
    try:
        for asset in ASSETS:
            source = _panel_path(stage_root, day, asset)
            destination = destinations[asset]
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append(destination)
            staged_assets[asset]["path"] = destination.as_posix()

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "published",
            "run_id": run_id,
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "trade_day": day.isoformat(),
            "contract_sha256": plan.contract_sha256,
            "contract_path": _contract_path(plan.target_root).as_posix(),
            "datahub_source": current_source,
            "assets": staged_assets,
        }
        manifest_path = _day_manifest_path(plan.target_root, day)
        _write_json_atomic(manifest_path, manifest)
        manifest_sha = _sha256_file(manifest_path)
        done_path = _day_done_path(plan.target_root, day)
        _write_json_atomic(
            done_path,
            {
                "schema_version": DONE_SCHEMA,
                "ready": True,
                "run_id": run_id,
                "trade_day": day.isoformat(),
                "manifest_path": manifest_path.as_posix(),
                "manifest_sha256": manifest_sha,
                "contract_sha256": plan.contract_sha256,
                "assets": {asset: staged_assets[asset]["sha256"] for asset in ASSETS},
            },
        )
        return {
            "trade_day": day.isoformat(),
            "manifest_path": manifest_path.as_posix(),
            "done_path": done_path.as_posix(),
            "assets": staged_assets,
        }
    except Exception:
        # The done marker is the visibility boundary.  Best-effort cleanup only
        # touches parquet files this invocation just moved from its own staging
        # directory, never a pre-existing published file.
        if not _day_done_path(plan.target_root, day).exists():
            manifest_path = _day_manifest_path(plan.target_root, day)
            if manifest_path.exists():
                manifest_path.unlink()
            for path in reversed(moved):
                if path.exists():
                    path.unlink()
        raise


def _cleanup_staging(stage_root: Path, *, target_root: Path) -> None:
    staging_parent = target_root / ".staging"
    if stage_root.exists() and _is_strict_child(stage_root, staging_parent):
        shutil.rmtree(stage_root)
    if staging_parent.exists() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()


def execute(plan: PanelStorePlan) -> dict[str, Any]:
    """Publish one fresh public/scratch PanelStore plan.

    This function never invokes FactorStore, scoring, a live scheduler, or a
    database.  It writes only below ``plan.target_root``.
    """

    if plan.target_kind == "public" and not _same_path(plan.target_root, PUBLIC_PANEL_ROOT):
        raise PanelStorePublisherError("public execution may write only the exact canonical PanelStore root")
    if plan.target_kind == "scratch" and not _is_strict_child(plan.target_root, RESEARCH_SCRATCH_PARENT):
        raise PanelStorePublisherError("scratch execution target escaped the approved research_scratch parent")

    run_id = f"panel_store_{datetime.now():%Y%m%dT%H%M%S}_{uuid4().hex[:12]}"
    lock_path = _acquire_lock(plan.target_root, run_id=run_id)
    stage_root = plan.target_root / ".staging" / run_id
    try:
        _assert_existing_contract_compatible(plan)
        _assert_days_fresh(plan)
        contract_payload = {**plan.contract, "contract_sha256": plan.contract_sha256}
        contract_path = _contract_path(plan.target_root)
        if not contract_path.exists():
            _write_json_atomic(contract_path, contract_payload)
        _build_staging(plan, stage_root=stage_root)
        published = [_commit_day(plan, stage_root=stage_root, day=day, run_id=run_id) for day in plan.days]
        batch_manifest = {
            "schema_version": "cbond_on_panel_store_batch/v1",
            "status": "published",
            "run_id": run_id,
            "published_at": datetime.now().isoformat(timespec="seconds"),
            "target_kind": plan.target_kind,
            "target_root": plan.target_root.as_posix(),
            "contract_sha256": plan.contract_sha256,
            "calendar": plan.calendar,
            "days": published,
            "workers": plan.workers,
        }
        batch_path = plan.target_root / "batches" / f"{run_id}.json"
        _write_json_atomic(batch_path, batch_manifest)
        batch_done_path = plan.target_root / "batches" / f"{run_id}.done"
        _write_json_atomic(
            batch_done_path,
            {
                "schema_version": DONE_SCHEMA,
                "ready": True,
                "run_id": run_id,
                "batch_manifest": batch_path.as_posix(),
                "batch_manifest_sha256": _sha256_file(batch_path),
                "contract_sha256": plan.contract_sha256,
                "calendar_days_sha256": plan.calendar["days_sha256"],
                "days_published": len(published),
            },
        )
        return {
            "run_id": run_id,
            "target_root": plan.target_root.as_posix(),
            "target_kind": plan.target_kind,
            "days_published": len(published),
            "batch_manifest": batch_path.as_posix(),
            "batch_done": batch_done_path.as_posix(),
            "contract": contract_path.as_posix(),
        }
    finally:
        _cleanup_staging(stage_root, target_root=plan.target_root)
        _release_lock(lock_path)


def _plan_payload(plan: PanelStorePlan, *, execute_requested: bool) -> dict[str, Any]:
    return {
        "publisher": "T1430_shared_panel_store",
        "execute_requested": bool(execute_requested),
        "writes_only": plan.target_root.as_posix(),
        "target_kind": plan.target_kind,
        "target_root": plan.target_root.as_posix(),
        "range": {"start": plan.start.isoformat(), "end": plan.end.isoformat()},
        "days": len(plan.days),
        "workers": plan.workers,
        "contract": plan.contract,
        "contract_sha256": plan.contract_sha256,
        "calendar": {
            "source": plan.calendar["source"],
            "days_sha256": plan.calendar["days_sha256"],
            "raw_clean_publish_parity": True,
        },
        "side_effects_forbidden": ["FactorStore", "model", "live scheduler", "live DB", "trade list"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish the shared CBOND_ON T1430/T1429 PanelStore")
    parser.add_argument("--start", help="inclusive score day YYYY-MM-DD; defaults to approved full-history start")
    parser.add_argument("--end", help="inclusive score day YYYY-MM-DD; defaults to approved full-history end")
    parser.add_argument("--workers", type=int, default=1, help="explicit panel-build workers; default=1")
    parser.add_argument("--execute", action="store_true", help="perform the already-preflighted panel publication")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--public-root",
        action="store_true",
        help="with --execute, write only the canonical D:/cbond_on/panel_data root",
    )
    target.add_argument(
        "--scratch-root",
        help="with --execute, publish a fresh bounded smoke root below D:/cbond_on/research_scratch",
    )
    args = parser.parse_args(argv)
    plan = preflight(
        start_text=args.start,
        end_text=args.end,
        workers=args.workers,
        execute=bool(args.execute),
        public_root=bool(args.public_root),
        scratch_root=args.scratch_root,
    )
    print(json.dumps(_plan_payload(plan, execute_requested=bool(args.execute)), ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    result = execute(plan)
    print(json.dumps({"result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
