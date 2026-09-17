"""Run the isolated R88 factor-set and LGBM hyperparameter study.

The tool admits only the canonical ``experiment`` factor table through its
published manifest and commit bundles.  The historical R88 scratch root is
migration/audit provenance only; it is not a normal model input.  Every study
config is created below a new `D:/cbond_on/research_scratch` run and invokes
the existing generic score and strict strategy-backtest entrypoints.  It never
loads a live config, writes a production result root, accesses the trade DB,
or restarts a scheduler.

Examples
--------
Prepare and validate the full study inputs::

    py -3 -B harness/tools/run_r88_joint_factor_lgbm_study.py --mode prepare

Run a short isolated smoke::

    py -3 -B harness/tools/run_r88_joint_factor_lgbm_study.py `
      --mode run --run-label smoke_202505 --candidate r88_full88_balanced_warm `
      --start 2025-05-05 --end 2025-05-16

Run one full candidate after the smoke::

    py -3 -B harness/tools/run_r88_joint_factor_lgbm_study.py `
      --mode run --candidate r88_full88_balanced_warm

The primary ranking is restricted to warm-start candidates.  Strict nested
selection is deliberately a cold-refit research arm: its physical feature set
can change each day and it would be misleading to retain earlier trees.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.common.config_utils import load_json_like
from cbond_on.core.trading_days import list_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.infra.data.panel import _iter_existing_snapshot_days
from cbond_on.infra.factors.r88_experiment_table import (
    R88ExperimentTableBinding,
    admit_r88_experiment_table,
)
from cbond_on.infra.model.impl.lgbm.trainer import build_dataset, build_tradable_code_map
from cbond_on.infra.model.neutralization import build_neutralizer
from cbond_on.infra.model.score_io import load_scores_by_date


SCRATCH_PARENT = Path("D:/cbond_on/research_scratch")
CANONICAL_FACTOR_STORE_ROOT = Path("D:/cbond_on/factor_store")
CANONICAL_STUDY_ID = "r88_joint_factor_lgbm_experiment_table_20260828"
EXPERIMENT_TABLE_ID = "experiment"
EXPERIMENT_TABLE_ROOT = CANONICAL_FACTOR_STORE_ROOT / EXPERIMENT_TABLE_ID
EXPERIMENT_TABLE_MANIFEST = EXPERIMENT_TABLE_ROOT / "table_manifest.json"

# The R88 profile has historical materialisation coverage through 2026-07-30.
# The table admission proves the actual published calendar; this range only
# supplies the study's default requested endpoint.
R88_EXPERIMENT_HOLDOUT_END = pd.Timestamp("2026-08-27").date()

STUDY_ID = CANONICAL_STUDY_ID
DEFAULT_STUDY_ROOT = SCRATCH_PARENT / STUDY_ID
FACTOR_ROOT = EXPERIMENT_TABLE_ROOT
BACKFILL_MANIFEST = EXPERIMENT_TABLE_MANIFEST
R88_INPUT_MODE: Literal["canonical_experiment"] = "canonical_experiment"
PROFILE_PATH = PROJECT_ROOT / "cbond_on" / "factor_contracts" / "profiles" / "research_r88_rust88_20260825.json5"
R50_PACK_PATH = PROJECT_ROOT / "cbond_on" / "config" / "factor" / "packs" / "live_screened_no_winsor_50_20260805.json5"
BENCHMARK_CONFIG = PROJECT_ROOT / "cbond_on" / "config" / "benchmark" / "benchmark_config.json5"
RAW_DATA_ROOT = Path("D:/cbond_data_hub/raw_data")
CLEAN_DATA_ROOT = Path("D:/cbond_data_hub/clean_data")
PANEL_ROOT = Path("D:/cbond_on/panel_data")
LABEL_ROOT = Path("D:/cbond_on/label_data")
DEV_END = pd.Timestamp("2025-12-31").date()
DEV_START = pd.Timestamp("2025-01-02").date()
HOLDOUT_START = pd.Timestamp("2026-01-01").date()
HOLDOUT_END = R88_EXPERIMENT_HOLDOUT_END
DEFAULT_END = HOLDOUT_END
EVIDENCE_LEVEL = "exploratory_conditioned_on_preexisting_r88_screen"
ROLLING_WINDOW_DAYS_INCLUDING_SCORE_DAY = 60
PRIOR_HISTORY_DAYS = ROLLING_WINDOW_DAYS_INCLUDING_SCORE_DAY - 1
FIT_TRAIN_DAYS = 41
FIT_VALIDATION_DAYS = 18


@dataclass(frozen=True)
class R88StudyInput:
    """The immutable canonical experiment-table contract for one process."""

    mode: Literal["canonical_experiment"]
    factor_root: Path
    manifest_path: Path
    study_id: str


_R88_BINDING: R88ExperimentTableBinding | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_scratch(path: Path) -> Path:
    resolved = path.resolve()
    parent = SCRATCH_PARENT.resolve()
    if resolved != parent and parent not in resolved.parents:
        raise ValueError(f"study root must be a child of {SCRATCH_PARENT}, got {path}")
    return resolved


def _require_run_child(path: Path | str, run_root: Path, *, field: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = run_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"{field} must stay below the immutable run root: {path}")
    return resolved


def _same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _is_strict_child(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return path.resolve(strict=False) != parent.resolve(strict=False)


def _study_id_or_default(value: str | None, *, default: str) -> str:
    study_id = str(value or "").strip() or default
    if not study_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in study_id):
        raise ValueError("--study-id must contain only letters, digits, '_' or '-'")
    return study_id


def _configure_runtime(
    *,
    factor_root_text: str | None,
    manifest_text: str | None,
    study_id_text: str | None,
) -> R88StudyInput:
    """Bind the process to the sole normal R88 input: canonical experiment.

    Direct R88 ``factor_data`` roots deliberately stopped being a study input
    after migration.  The old root remains owned by migration/audit tools and
    cannot be opted back into a normal score/backtest process through a CLI
    flag.
    """

    global STUDY_ID, DEFAULT_STUDY_ROOT, FACTOR_ROOT, BACKFILL_MANIFEST, _R88_BINDING
    global R88_INPUT_MODE, HOLDOUT_END, DEFAULT_END

    raw_root = str(factor_root_text or "").strip()
    raw_manifest = str(manifest_text or "").strip()
    requested_study_id = str(study_id_text or "").strip()
    if raw_root or raw_manifest:
        raise ValueError(
            "direct R88 factor_data roots are migration/audit provenance only; "
            "normal R88 research reads the canonical experiment factor_table"
        )
    study_id = _study_id_or_default(requested_study_id, default=CANONICAL_STUDY_ID)
    FACTOR_ROOT = EXPERIMENT_TABLE_ROOT
    BACKFILL_MANIFEST = EXPERIMENT_TABLE_MANIFEST
    STUDY_ID = study_id
    DEFAULT_STUDY_ROOT = SCRATCH_PARENT / study_id
    R88_INPUT_MODE = "canonical_experiment"
    HOLDOUT_END = R88_EXPERIMENT_HOLDOUT_END
    DEFAULT_END = HOLDOUT_END
    _R88_BINDING = None
    return R88StudyInput(
        mode=R88_INPUT_MODE,
        factor_root=FACTOR_ROOT,
        manifest_path=BACKFILL_MANIFEST,
        study_id=STUDY_ID,
    )


def _run_root_path(study_root: Path, run_label: str) -> Path:
    label = str(run_label).strip()
    if not label or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in label):
        raise ValueError("run_label must contain only letters, digits, '_' or '-'")
    return _require_scratch(study_root / "runs" / label)


def _run_root(study_root: Path, run_label: str) -> Path:
    root = _run_root_path(study_root, run_label)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _existing_label_days(*, start: date, end: date) -> list[date]:
    """List local 14:42 label partitions without inventing a parallel calendar."""

    days: list[date] = []
    for path in LABEL_ROOT.glob("*/*.parquet"):
        stem = path.stem
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except ValueError:
            continue
        if start <= day <= end:
            days.append(day)
    return sorted(set(days))


def _require_exact_study_calendar(*, factor_days: Sequence[date], start: date, end: date) -> dict[str, Any]:
    """Fail closed unless the rolling chain has exact factor, label, and calendar coverage."""

    score_days = list_trading_days_from_raw(
        RAW_DATA_ROOT,
        start,
        end,
        kind="snapshot",
        asset="cbond",
    )
    if not score_days:
        raise RuntimeError("R88 study score calendar is empty")
    prior_days = prev_trading_days_from_raw(
        RAW_DATA_ROOT,
        start,
        PRIOR_HISTORY_DAYS,
        kind="snapshot",
        asset="cbond",
    )
    if len(prior_days) != PRIOR_HISTORY_DAYS:
        raise RuntimeError(
            "R88 study lacks the exact 59 prior score-history days required for the first warm-start fit"
        )
    expected_inputs = [*prior_days, *score_days]
    if expected_inputs != sorted(expected_inputs) or len(expected_inputs) != len(set(expected_inputs)):
        raise RuntimeError("R88 study history and score calendars overlap or are unordered")
    available = set(factor_days)
    missing_factor = [day for day in expected_inputs if day not in available]
    if missing_factor:
        raise RuntimeError(
            "R88 study canonical experiment factor coverage is incomplete: "
            + ", ".join(day.isoformat() for day in missing_factor[:10])
        )
    labels = _existing_label_days(start=prior_days[0], end=end)
    missing_label = [day for day in expected_inputs if day not in set(labels)]
    if missing_label:
        raise RuntimeError(
            "R88 study label coverage is incomplete: "
            + ", ".join(day.isoformat() for day in missing_label[:10])
        )
    return {
        "score_days": [day.isoformat() for day in score_days],
        "prior_history_days": [day.isoformat() for day in prior_days],
        "input_days_sha256": _hash_json([day.isoformat() for day in expected_inputs]),
        "score_days_sha256": _hash_json([day.isoformat() for day in score_days]),
        "label_days_sha256": _hash_json([day.isoformat() for day in labels]),
    }


def _objective_windows(plan: Mapping[str, Any]) -> tuple[tuple[date, date], tuple[date, date]]:
    primary = plan.get("primary_objective")
    if not isinstance(primary, Mapping):
        raise RuntimeError("study plan lacks primary_objective")
    development = primary.get("development")
    holdout = primary.get("holdout")
    if not isinstance(development, Mapping) or not isinstance(holdout, Mapping):
        raise RuntimeError("study plan primary objective windows are malformed")
    try:
        dev = (pd.Timestamp(development["start"]).date(), pd.Timestamp(development["end"]).date())
        report = (pd.Timestamp(holdout["start"]).date(), pd.Timestamp(holdout["end"]).date())
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("study plan primary objective windows contain invalid dates") from exc
    if dev != (DEV_START, DEV_END) or report[0] != HOLDOUT_START or report[1] < report[0]:
        raise RuntimeError("study plan primary objective windows differ from the fixed R88 contract")
    return dev, report


def _assert_empty_new_run_root(run_root: Path) -> None:
    if not run_root.exists():
        return
    if any(run_root.iterdir()):
        raise FileExistsError(
            f"run label already exists and is immutable: {run_root}; choose a fresh --run-label"
        )


def _integrity_path(run_root: Path) -> Path:
    return run_root / "study_integrity.json"


def _write_study_integrity(run_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "study_plan.json": run_root / "study_plan.json",
        "input_admission.json": _require_run_child(plan["input_admission"], run_root, field="input_admission"),
        "paths_config": _require_run_child(plan["paths_config"], run_root, field="paths_config"),
    }
    for row in plan.get("candidates", []):
        candidate = str(row["candidate"])
        for key in ("model_config", "model_score_config", "backtest_config", "feature_manifest"):
            paths[f"{candidate}/{key}"] = _require_run_child(row[key], run_root, field=key)
    for top_k in (25, 35, 50):
        selection_root = run_root / "frozen_feature_sets" / f"pre2025_icir{top_k}"
        paths[f"pre2025_icir{top_k}_selection_manifest"] = selection_root / "selection_manifest.json"
        paths[f"pre2025_icir{top_k}_selection_stats"] = selection_root / "selection_stats.csv"
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise RuntimeError(f"cannot freeze incomplete study plan; missing {missing}")
    integrity = {
        "schema": "r88_joint_lgbm_study_integrity/v1",
        "created_at_utc": _now(),
        "study_id": STUDY_ID,
        "run_root": str(run_root),
        "files_sha256": {name: _sha256(path) for name, path in sorted(paths.items())},
    }
    _write_json(_integrity_path(run_root), integrity)
    return integrity


def _verify_study_integrity(run_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = _integrity_path(run_root)
    if not path.is_file():
        raise RuntimeError(f"study integrity manifest missing: {path}")
    integrity = _read_json(path)
    if integrity.get("study_id") != STUDY_ID or not _same_path(integrity.get("run_root", ""), run_root):
        raise RuntimeError("study integrity manifest does not belong to this run root")
    expected = integrity.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("study integrity manifest has no file hashes")
    for name, expected_sha in expected.items():
        if name == "study_plan.json":
            file_path = run_root / name
        elif name == "input_admission.json":
            file_path = _require_run_child(plan["input_admission"], run_root, field="input_admission")
        elif name == "paths_config":
            file_path = _require_run_child(plan["paths_config"], run_root, field="paths_config")
        elif name.startswith("pre2025_icir") and name.endswith("_selection_manifest"):
            top_k = name.removeprefix("pre2025_icir").removesuffix("_selection_manifest")
            file_path = run_root / "frozen_feature_sets" / f"pre2025_icir{top_k}" / "selection_manifest.json"
        elif name.startswith("pre2025_icir") and name.endswith("_selection_stats"):
            top_k = name.removeprefix("pre2025_icir").removesuffix("_selection_stats")
            file_path = run_root / "frozen_feature_sets" / f"pre2025_icir{top_k}" / "selection_stats.csv"
        else:
            candidate, key = str(name).split("/", 1)
            row = _candidate_plan(plan, candidate)
            file_path = _require_run_child(row[key], run_root, field=key)
        if not file_path.is_file() or _sha256(file_path) != str(expected_sha):
            raise RuntimeError(f"immutable study artifact changed or is missing: {name}")
    return integrity


def _factor_family_map(profile: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for spec in profile.get("factor_specs", []):
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        params = spec.get("params", {})
        family = params.get("family") if isinstance(params, dict) else None
        result[name] = str(family or spec.get("factor") or name).strip()
    return result


def _load_current_r88_contract(
    manifest: Mapping[str, Any],
    *,
    factor_count: int,
) -> list[date]:
    """Admit only the fresh, all-88 current-contract FactorStore format.

    This branch deliberately rejects every legacy merge field.  Supplying a
    new --factor-root must never turn an old R50-derived result into a current
    contract simply because both happen to expose 88 columns.
    """

    if manifest.get("execution_status") != CURRENT_EXECUTION_STATUS:
        raise RuntimeError(
            "full-current R88 manifest must be "
            f"{CURRENT_EXECUTION_STATUS!r}, got {manifest.get('execution_status')!r}"
        )
    execution = manifest.get("execution", {})
    if not isinstance(execution, Mapping):
        raise RuntimeError("full-current R88 manifest execution must be an object")
    if str(execution.get("mode", "")).strip() != CURRENT_EXECUTION_MODE:
        raise RuntimeError("full-current R88 manifest execution mode is not the current direct recompute")
    if not _same_path(execution.get("factor_root", ""), FACTOR_ROOT):
        raise RuntimeError("full-current R88 execution factor_root does not match --factor-root")
    if int(execution.get("factor_count", 0)) != factor_count:
        raise RuntimeError("full-current R88 execution factor_count does not match the profile")
    if execution.get("no_frozen_factorstore_read") is not True:
        raise RuntimeError("full-current R88 execution must prove no frozen FactorStore read")
    if execution.get("full_calendar_completion") is not True:
        raise RuntimeError("full-current R88 execution did not certify full calendar completion")
    if str(execution.get("calendar_source", "")).strip() != CURRENT_CALENDAR_SOURCE:
        raise RuntimeError("full-current R88 execution calendar source is not the approved DataHub contract")
    forbidden_execution = ("frozen_r50_factor_root", "r38_factor_root", "r88_factor_root")
    present_execution = [key for key in forbidden_execution if key in execution]
    if present_execution:
        raise RuntimeError(
            "full-current R88 execution must not retain legacy frozen merge fields: "
            + ", ".join(present_execution)
        )

    calendar_contract = execution.get("calendar_contract", {})
    if not isinstance(calendar_contract, Mapping):
        raise RuntimeError("full-current R88 execution calendar_contract must be an object")
    if str(calendar_contract.get("source", "")).strip() != CURRENT_CALENDAR_SOURCE:
        raise RuntimeError("full-current R88 calendar_contract source is not the approved DataHub contract")
    executable_range = calendar_contract.get("executable_range")
    if not isinstance(executable_range, Mapping) or {
        key: str(executable_range.get(key, "")).strip() for key in CURRENT_EXECUTABLE_RANGE
    } != CURRENT_EXECUTABLE_RANGE:
        raise RuntimeError("full-current R88 execution calendar range drift")
    days_sha = str(calendar_contract.get("days_sha256", "")).strip().lower()
    calendar_sha = str(execution.get("calendar_sha256", "")).strip().lower()
    if len(days_sha) != 64 or days_sha != calendar_sha:
        raise RuntimeError("full-current R88 execution calendar hash is absent or inconsistent")

    completion = manifest.get("completion", {})
    if not isinstance(completion, Mapping) or not bool(completion.get("full_completion", False)):
        raise RuntimeError("full-current R88 manifest does not certify full completion")
    if str(completion.get("definition", "")).strip() != CURRENT_COMPLETION_DEFINITION:
        raise RuntimeError("full-current R88 manifest completion definition drift")
    if completion.get("blocking_reasons") != []:
        raise RuntimeError("full-current R88 manifest completion has blocking reasons")
    forbidden_completion = ("expected_frozen_r50_days", "frozen_r50_factor_root")
    present_completion = [key for key in forbidden_completion if key in completion]
    if present_completion:
        raise RuntimeError(
            "full-current R88 completion must not retain legacy frozen merge fields: "
            + ", ".join(present_completion)
        )

    execution_days = _canonical_day_list(execution.get("expected_calendar_days"), field="execution.expected_calendar_days")
    source_calendar_lists = {
        name: _canonical_day_list(calendar_contract.get(name), field=f"execution.calendar_contract.{name}")
        for name in ("raw_cbond_snapshot_days", "clean_cbond_snapshot_days", "clean_stock_snapshot_days")
    }
    completion_lists = {
        name: _canonical_day_list(completion.get(name), field=f"completion.{name}")
        for name in ("expected_calendar_days", "execution_days", "r88_factor_store_days")
    }
    execution_rows = execution.get("days", [])
    if not isinstance(execution_rows, list):
        raise RuntimeError("full-current R88 execution day ledger must be a list")
    ledger_days = _canonical_day_list(
        [row.get("day") if isinstance(row, Mapping) else None for row in execution_rows],
        field="execution.days",
    )
    if (
        len(execution_days) != CURRENT_CALENDAR_DAY_COUNT
        or execution_days[0].isoformat() != CURRENT_EXECUTABLE_RANGE["start"]
        or execution_days[-1].isoformat() != CURRENT_EXECUTABLE_RANGE["end"]
        or any(values != execution_days for values in source_calendar_lists.values())
        or ledger_days != execution_days
        or any(values != execution_days for values in completion_lists.values())
    ):
        raise RuntimeError("full-current R88 completion calendar coverage is not exact")
    if _hash_json([day.isoformat() for day in execution_days]) != calendar_sha:
        raise RuntimeError("full-current R88 execution calendar hash does not match its date list")
    if list(_current_datahub_calendar()) != execution_days:
        raise RuntimeError("full-current R88 execution calendar does not match current DataHub inputs")
    return execution_days


def _load_r88_contract() -> tuple[dict[str, Any], dict[str, Any], list[str], list[date]]:
    """Resolve the only normal R88 input: the canonical experiment table."""

    global _R88_BINDING, FACTOR_ROOT, BACKFILL_MANIFEST
    profile = load_json_like(PROFILE_PATH)
    if not isinstance(profile, dict):
        raise RuntimeError("R88 profile must be an object")
    paths_cfg = {
        "factor_table": {
            "table_id": EXPERIMENT_TABLE_ID,
            "root": str(CANONICAL_FACTOR_STORE_ROOT),
        }
    }
    try:
        binding = admit_r88_experiment_table(
            paths_cfg,
            profile_path=PROFILE_PATH,
            profile=profile,
        )
    except Exception as exc:
        raise RuntimeError("R88 canonical experiment-table admission failed") from exc
    _R88_BINDING = binding
    FACTOR_ROOT = binding.table_root
    BACKFILL_MANIFEST = binding.table_manifest_path
    factors = list(binding.contract.factor_ids)
    if factors != [str(value) for value in profile.get("factors", [])]:
        raise RuntimeError("canonical experiment factor contract differs from the ordered R88 profile")
    manifest = binding.to_evidence()
    manifest["schema_version"] = "r88_canonical_experiment_input/v1"
    manifest["research_only"] = True
    manifest["model_training_ready"] = True
    manifest["factors"] = factors
    return profile, manifest, factors, list(binding.days)


def _factor_path(day: date) -> Path:
    _load_r88_contract()
    if _R88_BINDING is None:  # pragma: no cover - admission either binds or raises.
        raise RuntimeError("R88 canonical experiment-table reader is not bound")
    return _R88_BINDING.reader.day_path(day)


def _validate_inputs(study_root: Path) -> dict[str, Any]:
    profile, manifest, factors, days = _load_r88_contract()
    if _R88_BINDING is None:  # pragma: no cover - _load_r88_contract binds or raises.
        raise RuntimeError("R88 canonical experiment-table reader is not bound")
    for path in (RAW_DATA_ROOT, CLEAN_DATA_ROOT, LABEL_ROOT, BENCHMARK_CONFIG):
        if not path.exists():
            raise FileNotFoundError(f"required research input is missing: {path}")

    invalid_days: list[dict[str, Any]] = []
    schema_failures: list[dict[str, Any]] = []
    total_rows = 0
    eligible_rows = 0
    for index, day in enumerate(days, start=1):
        # CanonicalFactorTableReader validates table identity, the day
        # parquet/manifest/.done bundle, contract and frame hashes before it
        # yields a dataframe.  There is intentionally no direct parquet read.
        frame = _R88_BINDING.reader.read_day(day)
        missing = [factor for factor in factors if factor not in frame.columns]
        if missing:
            schema_failures.append({"day": str(day), "missing": missing})
            continue
        values = frame[factors].to_numpy(dtype=float, copy=False)
        if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["dt", "code"]:
            raise RuntimeError(f"R88 index contract mismatch for {day}")
        row_available = np.isfinite(values).sum(axis=1)
        current_eligible = int((row_available >= 66).sum())
        total_rows += int(len(frame))
        eligible_rows += current_eligible
        if current_eligible == 0:
            invalid_days.append(
                {
                    "day": str(day),
                    "rows": int(len(frame)),
                    "max_available_factors": int(row_available.max()) if len(row_available) else 0,
                }
            )
        if index % 100 == 0 or index == len(days):
            print(f"[r88-admission] factor coverage {index}/{len(days)}", flush=True)
    if schema_failures:
        raise RuntimeError(f"R88 schema failures: {schema_failures[:3]}")

    current_specs_sha = _hash_json(profile.get("factor_specs", []))
    result = {
        "schema": "r88_joint_lgbm_input_admission/v1",
        "created_at_utc": _now(),
        "study_id": STUDY_ID,
        "research_only": True,
        "factor_input": {
            "mode": R88_INPUT_MODE,
            **manifest,
        },
        "factor_root": str(FACTOR_ROOT),
        "factor_root_manifest": str(BACKFILL_MANIFEST),
        "manifest_sha256": _sha256(BACKFILL_MANIFEST),
        "profile_path": str(PROFILE_PATH),
        "profile_sha256_current": _sha256(PROFILE_PATH),
        "profile_sha256_recorded_at_materialization": manifest["source_provenance"]["historical_profile"]["profile_sha256"],
        "profile_file_hash_changed_since_materialization": manifest["source_provenance"][
            "profile_file_hash_changed_since_materialization"
        ],
        "factor_specs_sha256": current_specs_sha,
        "profile_factor_sha256": _hash_json(factors),
        "factor_count": len(factors),
        "day_count": len(days),
        "range": {"start": str(days[0]), "end": str(days[-1])},
        "availability_gate": {"raw_factor_count": 88, "min_available_factors": 66},
        "coverage": {
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "eligible_fraction": float(eligible_rows / total_rows) if total_rows else float("nan"),
            "zero_eligible_days": invalid_days,
        },
        "read_only_inputs": {
            "raw_data_root": str(RAW_DATA_ROOT),
            "clean_data_root": str(CLEAN_DATA_ROOT),
            "panel_data_root": {
                "path": str(PANEL_ROOT),
                "exists": bool(PANEL_ROOT.exists()),
                "required_by_current_exposure_contract": False,
                "reason": "the frozen cbond_style_5_tminus1 exposure contract reads only raw daily_base",
            },
            "label_data_root": str(LABEL_ROOT),
        },
        "benchmark": {
            "path": str(BENCHMARK_CONFIG),
            "sha256": _sha256(BENCHMARK_CONFIG),
            "expected_method": "strict_official_prev_close_split",
        },
        "contract": {
            "factor_time": "14:30",
            "label_time": "14:42",
            "rolling_window_days_including_score_day": ROLLING_WINDOW_DAYS_INCLUDING_SCORE_DAY,
            "prior_history_days": PRIOR_HISTORY_DAYS,
            "fit_train_days": FIT_TRAIN_DAYS,
            "fit_validation_days": FIT_VALIDATION_DAYS,
            "daily_refit": True,
            "universe": "strict_tminus1_o_0005",
            "strategy": "strategy01_topk_turnover",
        },
    }
    _write_json(study_root / "input_admission.json", result)
    return result


def _base_lgbm_params() -> dict[str, Any]:
    return {
        "objective": "regression",
        "verbosity": -1,
        "device": "cpu",
        "n_estimators": 600,
        "learning_rate": 0.02,
        "num_leaves": 16,
        "max_depth": 6,
        "min_data_in_leaf": 60,
        "feature_fraction": 1.0,
        "feature_fraction_bynode": 1.0,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "reg_alpha": 0.2,
        "reg_lambda": 2.0,
        "random_state": 20260826,
        "n_jobs": 1,
        "log_eval_period": 1000000,
    }


def _regularized_lgbm_params() -> dict[str, Any]:
    params = _base_lgbm_params()
    params.update(
        {
            "num_leaves": 8,
            "max_depth": 5,
            "min_data_in_leaf": 100,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        }
    )
    return params


def _lgbm_profiles() -> dict[str, dict[str, Any]]:
    """Predeclared compact hyperparameter grid for the development search."""

    profiles: dict[str, dict[str, Any]] = {
        "p1_shallow_strongreg": _base_lgbm_params(),
        "p2_balanced": _base_lgbm_params(),
        "p3_deep_regularized": _base_lgbm_params(),
        "p4_shrinkage": _base_lgbm_params(),
        "p5_light_regularization": _base_lgbm_params(),
        "p6_lower_bagging": _base_lgbm_params(),
    }
    profiles["p1_shallow_strongreg"].update(
        {"num_leaves": 8, "max_depth": 4, "min_data_in_leaf": 100, "reg_alpha": 0.5, "reg_lambda": 5.0}
    )
    profiles["p3_deep_regularized"].update(
        {
            "num_leaves": 31,
            "max_depth": 8,
            "min_data_in_leaf": 80,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "reg_alpha": 0.5,
            "reg_lambda": 5.0,
        }
    )
    profiles["p4_shrinkage"].update(
        {
            "min_data_in_leaf": 100,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
        }
    )
    profiles["p5_light_regularization"].update(
        {"min_data_in_leaf": 40, "reg_alpha": 0.05, "reg_lambda": 1.0}
    )
    profiles["p6_lower_bagging"].update({"bagging_fraction": 0.7})
    return profiles


def _static_icir_feature_set(
    *,
    profile: dict[str, Any],
    factors: list[str],
    days: list[date],
    output_root: Path,
    top_k: int,
) -> dict[str, Any]:
    """Freeze a pre-OOS feature set from 2024 fitting observations only."""

    selection_start = pd.Timestamp("2024-04-04").date()
    selection_end = pd.Timestamp("2024-12-31").date()
    selection_days = [day for day in days if selection_start <= day <= selection_end]
    if len(selection_days) < 60:
        raise RuntimeError("insufficient R88 days for the pre-2025 static factor-selection window")
    _load_r88_contract()
    if _R88_BINDING is None:  # pragma: no cover - canonical admission binds or raises.
        raise RuntimeError("R88 canonical experiment-table reader is not bound")
    store = _R88_BINDING.reader
    tradable_codes = build_tradable_code_map(
        raw_data_root=RAW_DATA_ROOT,
        days=selection_days,
        buy_twap_col="twap_1442_1457",
        sell_twap_col="twap_0930_0939",
        min_amount=0.0,
        min_volume=0.0,
        twap_table="market_cbond.daily_twap",
        asset="cbond",
    )
    neutralizer = build_neutralizer(
        {
            "enabled": True,
            "method": "ridge",
            "ridge_alpha": 1e-6,
            "min_count": 30,
            "standardize_exposures": True,
            "missing_policy": "keep_original",
            "exposures_file": "models/preprocess/neutralization/exposures/cbond_style_5_tminus1.json5",
        },
        raw_data_root=RAW_DATA_ROOT,
        panel_data_root=PANEL_ROOT,
        neutralization_cache_root=output_root / "neutralization_cache",
    )
    split = build_dataset(
        factor_store=store,
        label_root=LABEL_ROOT,
        days=selection_days,
        factor_cols=factors,
        raw_factor_cols=factors,
        preprocess_factor_cols=factors,
        min_count=30,
        winsor_lower=None,
        winsor_upper=None,
        zscore=True,
        factor_time="14:30",
        label_time="14:42",
        require_label=True,
        tradable_code_map=tradable_codes,
        tradable_strict=True,
        neutralizer=neutralizer,
        missing_values={
            "enabled": True,
            "keep_nan": True,
            "min_available_factors": 66,
            "add_valid_count_features": False,
        },
    )
    if split.x.empty:
        raise RuntimeError("pre-2025 selection has no fitting rows after the fixed R88 admission contract")
    observed_days = pd.to_datetime(split.dt, errors="coerce").dt.date
    valid_days = sorted({day for day in observed_days if pd.notna(day)})
    values_by_feature: dict[str, list[float]] = {factor: [] for factor in factors}
    for day in valid_days:
        mask = observed_days.eq(day).to_numpy()
        if int(mask.sum()) < 30:
            continue
        y = pd.to_numeric(split.y.loc[mask], errors="coerce")
        for factor in factors:
            signal = pd.to_numeric(split.x.loc[mask, factor], errors="coerce")
            joint = pd.DataFrame({"signal": signal, "label": y}).dropna()
            if len(joint) < 3 or joint["signal"].nunique() < 2 or joint["label"].nunique() < 2:
                continue
            value = joint["signal"].corr(joint["label"], method="spearman")
            if np.isfinite(value):
                values_by_feature[factor].append(float(value))
    stats: dict[str, dict[str, float | int]] = {}
    for factor in factors:
        values = np.asarray(values_by_feature[factor], dtype=float)
        count = int(len(values))
        mean = float(np.nanmean(values)) if count else float("nan")
        std = float(np.nanstd(values, ddof=1)) if count > 1 else float("nan")
        score = abs(mean) / (std + 1e-6) if np.isfinite(mean) and np.isfinite(std) else float("-inf")
        stats[factor] = {"ic_days": count, "ic_mean": mean, "ic_std": std, "selection_score": score}
    ranked = sorted(factors, key=lambda factor: (-float(stats[factor]["selection_score"]), factor))
    rank_frame = split.x[factors].apply(pd.to_numeric, errors="coerce")
    ranks = rank_frame.groupby(observed_days, sort=False).rank(method="average", pct=True)
    correlations = ranks.corr(method="pearson", min_periods=3)
    family_map = _factor_family_map(profile)
    selected: list[str] = []
    selected_rank: dict[str, int] = {}
    reasons: dict[str, str] = {}
    family_counts: dict[str, int] = {}
    max_corr: dict[str, float] = {}
    for factor in ranked:
        if int(stats[factor]["ic_days"]) < 40:
            reasons[factor] = "insufficient_ic_days"
            continue
        family = family_map.get(factor, factor)
        if family_counts.get(family, 0) >= 2:
            reasons[factor] = "family_cap"
            continue
        pairs = [
            abs(float(correlations.loc[factor, chosen]))
            for chosen in selected
            if np.isfinite(correlations.loc[factor, chosen])
        ]
        strongest = float(max(pairs)) if pairs else float("nan")
        max_corr[factor] = strongest
        if np.isfinite(strongest) and strongest > 0.8:
            reasons[factor] = "correlation_cap"
            continue
        if len(selected) >= top_k:
            reasons[factor] = "top_k"
            continue
        selected.append(factor)
        selected_rank[factor] = len(selected)
        family_counts[family] = family_counts.get(family, 0) + 1
        reasons[factor] = "selected"
    if len(selected) != top_k:
        raise RuntimeError(f"pre-2025 static selector did not fill {top_k} factors; selected={len(selected)}")
    rows = [
        {
            "factor": factor,
            "family": family_map.get(factor, factor),
            "selected": factor in set(selected),
            "selection_rank": selected_rank.get(factor),
            "selection_reason": reasons.get(factor, "not_selected"),
            "max_abs_corr_with_selected": max_corr.get(factor, float("nan")),
            **stats[factor],
        }
        for factor in ranked
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_root / "selection_stats.csv", index=False)
    manifest = {
        "schema": "r88_pre2025_static_icir/v1",
        "created_at_utc": _now(),
        "research_only": True,
        "selection_window": {"start": str(selection_start), "end": str(selection_end)},
        "observed_feature_days": {"start": str(valid_days[0]), "end": str(valid_days[-1]), "count": len(valid_days)},
        "selection_rule": {
            "score": "abs_mean_rank_ic / rank_ic_std",
            "min_ic_days": 40,
            "top_k": top_k,
            "max_abs_rank_correlation": 0.8,
            "max_per_family": 2,
            "source_split": "pre_2025_fitting_period_only",
        },
        "availability_gate": {"raw_factor_count": 88, "min_available_factors": 66},
        "selected_factors": selected,
        "selected_factors_sha256": _hash_json(selected),
        "candidate_factors_sha256": _hash_json(factors),
        "selection_stats_path": str(output_root / "selection_stats.csv"),
    }
    _write_json(output_root / "selection_manifest.json", manifest)
    return manifest


def _candidate_specs(
    profile: dict[str, Any],
    factors: list[str],
    pre2025_selections: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    r50_pack = load_json_like(R50_PACK_PATH)
    r50 = [str(item.get("name", "")).strip() for item in r50_pack.get("factors", []) if isinstance(item, dict)]
    if len(r50) != 50 or len(set(r50)) != 50 or not set(r50).issubset(factors):
        raise RuntimeError("frozen R50 pack does not resolve to 50 R88 factors")
    factor_sets: dict[str, dict[str, Any]] = {
        "full88": {
            "factor_mode": "fixed_full88",
            "effective_selected_factors": list(factors),
            "description": "all 88 R88 factors",
        },
        "r50": {
            "factor_mode": "fixed_r50_mask",
            "effective_selected_factors": r50,
            "description": "frozen R50 control with the shared R88 admission gate",
        },
    }
    for top_k, selection in sorted(pre2025_selections.items()):
        selected = [str(value) for value in selection.get("selected_factors", [])]
        if len(selected) != top_k or len(set(selected)) != top_k or not set(selected).issubset(factors):
            raise RuntimeError(f"pre-2025 static selector did not resolve to {top_k} valid R88 factors")
        factor_sets[f"icir{top_k}"] = {
            "factor_mode": f"pre_oos_static_icir{top_k}_mask",
            "effective_selected_factors": selected,
            "description": f"fixed {top_k}-factor ICIR/correlation subset frozen from pre-2025 fitting history",
        }

    profiles = _lgbm_profiles()
    output: dict[str, dict[str, Any]] = {}
    for factor_set_id, factor_set in factor_sets.items():
        selected_set = set(factor_set["effective_selected_factors"])
        values = {factor: (1.0 if factor in selected_set else 0.0) for factor in factors}
        for hyperparameter_id, params in profiles.items():
            candidate = f"r88_{factor_set_id}__{hyperparameter_id}"
            output[candidate] = {
                "state_mode": "warm_start",
                "factor_set_id": factor_set_id,
                "hyperparameter_id": hyperparameter_id,
                "factor_mode": factor_set["factor_mode"],
                "factors": list(factors),
                "effective_selected_factors": list(factor_set["effective_selected_factors"]),
                "feature_contribution": {"enabled": True, "default": 1.0, "values": values},
                "lgbm_params": dict(params),
                "description": f"{factor_set['description']}; hyperparameter profile {hyperparameter_id}",
            }
    if len(output) != 30:
        raise RuntimeError(f"expected exactly 30 fixed warm-start candidates, got {len(output)}")
    return output


def _model_config(candidate: str, spec: dict[str, Any], run_root: Path, start: date, end: date) -> dict[str, Any]:
    feature_engineering: dict[str, Any] = {
        "missing_values": {
            "enabled": True,
            "keep_nan": True,
            "min_available_factors": 66,
            "add_valid_count_features": False,
        }
    }
    if "nested_factor_selection" in spec:
        feature_engineering["nested_factor_selection"] = spec["nested_factor_selection"]
    incremental = {
        "enabled": True,
        "skip_existing_scores": False,
        "warm_start": spec["state_mode"] == "warm_start",
        "save_state": spec["state_mode"] == "warm_start",
        # The fixed R88 masks make warm-start semantically valid, but only if
        # a failed inherited fit cannot silently become a cold refit.  The
        # runner permits exactly one cold bootstrap in a fresh research chain.
        "strict_warm_start": dict(
            spec.get(
                "strict_warm_start",
                {
                    "enabled": spec["state_mode"] == "warm_start",
                    "require_initial_checkpoint": False,
                },
            )
        ),
        "state_dir": str(run_root / "runtime" / "results" / "model_state" / candidate),
    }
    cfg: dict[str, Any] = {
        "model_name": candidate,
        "start": str(start),
        "end": str(end),
        "panel_name": "T1430",
        "window_minutes": 15,
        "factor_time": "14:30",
        "label_time": "14:42",
        "factors": list(spec["factors"]),
        "min_count": 30,
        "winsor": {"enabled": False},
        "zscore": True,
        "neutralization": {
            "enabled": True,
            "method": "ridge",
            "ridge_alpha": 1e-6,
            "min_count": 30,
            "standardize_exposures": True,
            "missing_policy": "keep_original",
            "exposures_file": "models/preprocess/neutralization/exposures/cbond_style_5_tminus1.json5",
        },
        "feature_engineering": feature_engineering,
        "tradable_filter": {"enabled": True, "strict": True},
        "rolling": {"enabled": True, "window_days": ROLLING_WINDOW_DAYS_INCLUDING_SCORE_DAY},
        "refit_every_n_days": 1,
        "incremental": incremental,
        "score_overwrite": True,
        "score_dedupe": True,
        "score_only_no_target_label_read": True,
        "score_only_apply_tradable_filter": True,
        "loss_mode": "mse",
        "early_stopping_rounds": 80,
        "early_stopping_metric": "rank_ic",
        "train": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
        "lgbm_params": dict(spec["lgbm_params"]),
        "score_output": str(run_root / "runtime" / "results" / "scores" / candidate),
        "artifact_output_root": str(run_root / "runtime" / "results"),
        "neutralization_cache_root": str(run_root / "runtime" / "results" / "neutralization_cache" / candidate),
        "research_only": {
            "study_id": STUDY_ID,
            "evidence_level": EVIDENCE_LEVEL,
            "factor_profile": str(PROFILE_PATH),
            "factor_table": {
                "table_id": EXPERIMENT_TABLE_ID,
                "root": str(CANONICAL_FACTOR_STORE_ROOT),
                "table_manifest": str(BACKFILL_MANIFEST),
            },
            "state_mode": spec["state_mode"],
            "factor_mode": spec["factor_mode"],
            "factor_set_id": spec["factor_set_id"],
            "hyperparameter_id": spec["hyperparameter_id"],
        },
        "experiment": {
            "id": STUDY_ID,
            "candidate": candidate,
            "research_only": True,
            "description": spec["description"],
            "evidence_level": EVIDENCE_LEVEL,
            "factor_set_id": spec["factor_set_id"],
            "hyperparameter_id": spec["hyperparameter_id"],
        },
    }
    if "feature_contribution" in spec:
        cfg["feature_engineering"]["feature_contribution"] = spec["feature_contribution"]
    return cfg


def _paths_config(run_root: Path) -> dict[str, Any]:
    return {
        "raw_data_root": str(RAW_DATA_ROOT),
        "clean_data_root": str(CLEAN_DATA_ROOT),
        "results_root": str(run_root / "runtime" / "results"),
        "read_only_input_roots": {
            "panel_data_root": str(PANEL_ROOT),
            "label_data_root": str(LABEL_ROOT),
        },
        "factor_table": {
            "table_id": EXPERIMENT_TABLE_ID,
            "root": str(CANONICAL_FACTOR_STORE_ROOT),
        },
    }


def _backtest_config(candidate: str, run_root: Path, start: date, end: date) -> dict[str, Any]:
    return {
        "start": str(start),
        "end": str(end),
        "batch_id": f"{STUDY_ID}_{candidate}",
        "output_root": str(run_root / "runtime" / "results"),
        "score_source": {
            "model_id": candidate,
            "score_root": str(run_root / "runtime" / "results" / "scores" / candidate),
        },
        "strategy_id": "strategy01_topk_turnover",
        "strategy_config_path": "strategies/strategy01/strategy01",
        "execution_lag_trading_days": 0,
        "freeze_signal_universe": False,
        "allowlist": {
            "enabled": True,
            "table": "quant_factor_dev.researcher_xuvb.o_0005",
            "lag_trading_days": 1,
            "asset": "cbond",
            "flag_field": "factor_value",
            "fallback_field": "weight",
            "threshold": 0.0,
        },
        "research_only": True,
    }


def _validate_run_configs(run_root: Path, plan: dict[str, Any], row: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Re-check every executable config before a research subprocess starts."""

    candidate = str(row["candidate"])
    paths_path = _require_run_child(plan["paths_config"], run_root, field="paths_config")
    score_path = _require_run_child(row["model_score_config"], run_root, field="model_score_config")
    model_path = _require_run_child(row["model_config"], run_root, field="model_config")
    backtest_path = _require_run_child(row["backtest_config"], run_root, field="backtest_config")
    feature_path = _require_run_child(row["feature_manifest"], run_root, field="feature_manifest")
    for path in (paths_path, score_path, model_path, backtest_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(f"frozen study config is missing: {path}")

    paths_cfg = load_json_like(paths_path)
    inputs = paths_cfg.get("read_only_input_roots", {})
    expected_results = run_root / "runtime" / "results"
    if not _same_path(paths_cfg.get("results_root", ""), expected_results):
        raise RuntimeError("research paths config has an unexpected results root")
    factor_table = paths_cfg.get("factor_table")
    if (
        not isinstance(inputs, dict)
        or "factor_data_root" in inputs
        or not isinstance(factor_table, dict)
        or factor_table != {"table_id": EXPERIMENT_TABLE_ID, "root": str(CANONICAL_FACTOR_STORE_ROOT)}
    ):
        raise RuntimeError("research paths config is not pinned to the canonical experiment factor table")
    if not _same_path(paths_cfg.get("raw_data_root", ""), RAW_DATA_ROOT) or not _same_path(
        paths_cfg.get("clean_data_root", ""), CLEAN_DATA_ROOT
    ):
        raise RuntimeError("research paths config has unexpected DataHub input roots")
    if not _same_path(inputs.get("label_data_root", ""), LABEL_ROOT):
        raise RuntimeError("research paths config has an unexpected label input root")

    score_cfg = load_json_like(score_path)
    models = score_cfg.get("models", {})
    if (
        not isinstance(models, dict)
        or list(models) != [candidate]
        or str(score_cfg.get("model_id")) != candidate
        or str(models[candidate].get("model_type")) != "lgbm"
        or not _same_path(models[candidate].get("model_config", ""), model_path)
    ):
        raise RuntimeError("research score config does not exactly bind this candidate model")
    if str(row.get("state_mode")) == "warm_start":
        execution = score_cfg.get("execution", {})
        if (
            not isinstance(execution, dict)
            or int(execution.get("refit_every_n_days", 0)) != 1
            or int(execution.get("parallel_shards", 0)) != 1
        ):
            raise RuntimeError(
                "R88 strict warm-start requires one serial daily-refit score execution"
            )

    model_cfg = load_json_like(model_path)
    expected_score = expected_results / "scores" / candidate
    expected_state = expected_results / "model_state" / candidate
    if not bool(model_cfg.get("research_only")) or str(model_cfg.get("model_name")) != candidate:
        raise RuntimeError("model config is not an explicit research-only candidate")
    if not _same_path(model_cfg.get("score_output", ""), expected_score):
        raise RuntimeError("model score output is outside the run root")
    if not _same_path(model_cfg.get("artifact_output_root", ""), expected_results):
        raise RuntimeError("model artifact output is outside the run root")
    if not _same_path(model_cfg.get("neutralization_cache_root", ""), expected_results / "neutralization_cache" / candidate):
        raise RuntimeError("model neutralization cache is outside the run root")
    incremental = model_cfg.get("incremental", {})
    if not isinstance(incremental, dict) or not _same_path(incremental.get("state_dir", ""), expected_state):
        raise RuntimeError("model state output is outside the run root")
    strict_warm_start = incremental.get("strict_warm_start", {})
    strict_requires_initial = (
        strict_warm_start.get("require_initial_checkpoint")
        if isinstance(strict_warm_start, dict)
        else None
    )
    if (
        str(row.get("state_mode")) == "warm_start"
        and (
            not isinstance(strict_warm_start, dict)
            or strict_warm_start.get("enabled") is not True
            or not isinstance(strict_requires_initial, bool)
            or (
                strict_requires_initial
                and not str(strict_warm_start.get("initial_checkpoint_day", "")).strip()
            )
            or bool(incremental.get("skip_existing_scores", True))
            or not bool(incremental.get("warm_start", False))
            or not bool(incremental.get("save_state", False))
            or not bool(model_cfg.get("score_overwrite", False))
            or int(model_cfg.get("refit_every_n_days", 0)) != 1
        )
    ):
        raise RuntimeError(
            "R88 warm-start candidate must use the strict no-cold-fallback research contract"
        )
    research = model_cfg.get("research_only", {})
    expected_research_table = {
        "table_id": EXPERIMENT_TABLE_ID,
        "root": str(CANONICAL_FACTOR_STORE_ROOT),
        "table_manifest": str(BACKFILL_MANIFEST),
    }
    if not isinstance(research, dict) or research.get("factor_table") != expected_research_table:
        raise RuntimeError("model research metadata does not bind the canonical R88 experiment factor table")
    profile, _, factors, _ = _load_r88_contract()
    feature_manifest = _read_json(feature_path)
    selected = [str(value) for value in feature_manifest.get("candidate_spec", {}).get("effective_selected_factors", [])]
    if (
        feature_manifest.get("candidate") != candidate
        or [str(value) for value in feature_manifest.get("factor_ids", [])] != factors
        or [str(value) for value in model_cfg.get("factors", [])] != factors
        or not selected
        or not set(selected).issubset(factors)
    ):
        raise RuntimeError("candidate feature manifest does not exactly bind the R88 model schema")
    feature_engineering = model_cfg.get("feature_engineering", {})
    feature_contribution = feature_engineering.get("feature_contribution", {}) if isinstance(feature_engineering, dict) else {}
    values = feature_contribution.get("values", {}) if isinstance(feature_contribution, dict) else {}
    if set(values) != set(factors):
        raise RuntimeError("fixed factor mask must explicitly cover every R88 feature")
    selected_set = set(selected)
    actual_selected = {factor for factor in factors if float(values[factor]) > 0.0}
    if actual_selected != selected_set or any(float(values[factor]) not in {0.0, 1.0} for factor in factors):
        raise RuntimeError("fixed factor mask differs from its immutable feature manifest")
    params = model_cfg.get("lgbm_params", {})
    if not isinstance(params, dict) or any(abs(float(params.get(key, 1.0)) - 1.0) > 1e-12 for key in ("feature_fraction", "feature_fraction_bynode")):
        raise RuntimeError("fixed factor subsets require feature_fraction=feature_fraction_bynode=1")

    backtest_cfg = load_json_like(backtest_path)
    source = backtest_cfg.get("score_source", {})
    if not bool(backtest_cfg.get("research_only")) or not isinstance(source, dict):
        raise RuntimeError("backtest config is not research-only")
    if not _same_path(backtest_cfg.get("output_root", ""), expected_results):
        raise RuntimeError("backtest output root is outside the run root")
    if not _same_path(source.get("score_root", ""), expected_score) or str(source.get("model_id")) != candidate:
        raise RuntimeError("backtest score source is not this candidate's score root")
    forbidden = {"database_write", "db", "scheduler", "live_config", "live_release"}
    for config_name, config in (("model", model_cfg), ("score", score_cfg), ("backtest", backtest_cfg)):
        present = sorted(forbidden & set(config))
        if present:
            raise RuntimeError(f"research {config_name} config contains forbidden live/database fields: {present}")
    return paths_path, score_path, backtest_path


def prepare(study_root: Path, run_label: str, start: date, end: date) -> dict[str, Any]:
    if start != DEV_START or end != R88_EXPERIMENT_HOLDOUT_END:
        raise ValueError(
            "R88 joint LGBM study has one fixed approved range: "
            f"{DEV_START.isoformat()}..{R88_EXPERIMENT_HOLDOUT_END.isoformat()}; "
            f"got {start.isoformat()}..{end.isoformat()}"
        )
    study_root = _require_scratch(study_root)
    study_root.mkdir(parents=True, exist_ok=True)
    run_root = _run_root_path(study_root, run_label)
    _assert_empty_new_run_root(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    admission = _validate_inputs(run_root)
    profile, input_evidence, factors, days = _load_r88_contract()
    calendar_evidence = _require_exact_study_calendar(factor_days=days, start=start, end=end)
    zero_eligible_days = {
        pd.Timestamp(row["day"]).date()
        for row in admission.get("coverage", {}).get("zero_eligible_days", [])
    }
    blocking = sorted(
        day
        for day in zero_eligible_days
        if start <= day <= end and day >= pd.Timestamp("2024-04-04").date()
    )
    if blocking:
        raise RuntimeError(
            "requested continuous R88 warm-start range crosses zero-eligible input day(s): "
            + ", ".join(str(day) for day in blocking)
            + "; "
            + "publish a new immutable canonical experiment-table contract"
            + " before extending this study"
        )
    pre2025_selections = {
        top_k: _static_icir_feature_set(
            profile=profile,
            factors=factors,
            days=days,
            output_root=run_root / "frozen_feature_sets" / f"pre2025_icir{top_k}",
            top_k=top_k,
        )
        for top_k in (25, 35, 50)
    }
    specs = _candidate_specs(profile, factors, pre2025_selections)
    configs = run_root / "configs"
    # `load_config_file("paths")` applies the isolated read-only-root profile
    # only to a paths config below a `data/` directory, matching the project
    # convention for all paths profiles.
    paths_path = configs / "data" / "paths_r88_joint.json5"
    _write_json(paths_path, _paths_config(run_root))
    plan_candidates: list[dict[str, Any]] = []
    for candidate, spec in specs.items():
        candidate_start = start
        model_path = configs / f"model_{candidate}.json5"
        score_path = configs / f"model_score_{candidate}.json5"
        backtest_path = configs / f"backtest_{candidate}.json5"
        _write_json(model_path, _model_config(candidate, spec, run_root, candidate_start, end))
        _write_json(
            score_path,
            {
                "default_model_id": candidate,
                "model_id": candidate,
                "start": str(candidate_start),
                "end": str(end),
                "execution": {
                    "refit_every_n_days": 1,
                    "train_processes": 1,
                    "parallel_shards": 1,
                    "wandb": {"enabled": False, "mode": "disabled"},
                },
                "models": {candidate: {"model_type": "lgbm", "model_config": str(model_path)}},
            },
        )
        _write_json(backtest_path, _backtest_config(candidate, run_root, candidate_start, end))
        factor_manifest = {
            "study_id": STUDY_ID,
            "candidate": candidate,
            "state_mode": spec["state_mode"],
            "factor_mode": spec["factor_mode"],
            "factor_set_id": spec["factor_set_id"],
            "hyperparameter_id": spec["hyperparameter_id"],
            "factor_ids": list(factors),
            "factor_ids_sha256": _hash_json(factors),
            "candidate_spec": spec,
        }
        _write_json(run_root / "candidate_manifests" / f"{candidate}.json", factor_manifest)
        plan_candidates.append(
            {
                "candidate": candidate,
                "state_mode": spec["state_mode"],
                "factor_mode": spec["factor_mode"],
                "factor_set_id": spec["factor_set_id"],
                "hyperparameter_id": spec["hyperparameter_id"],
                "start": str(candidate_start),
                "end": str(end),
                "model_config": str(model_path),
                "model_score_config": str(score_path),
                "backtest_config": str(backtest_path),
                "feature_manifest": str(run_root / "candidate_manifests" / f"{candidate}.json"),
            }
        )
    plan = {
        "schema": "r88_joint_lgbm_plan/v1",
        "created_at_utc": _now(),
        "study_id": STUDY_ID,
        "research_only": True,
        "factor_input": {
            "mode": R88_INPUT_MODE,
            **input_evidence,
        },
        "run_label": run_label,
        "run_root": str(run_root),
        "date_range": {"start": str(start), "end": str(end)},
        "exact_calendar": calendar_evidence,
        "input_admission": str(run_root / "input_admission.json"),
        "paths_config": str(paths_path),
        "primary_objective": {
            "development": {"start": str(DEV_START), "end": str(DEV_END)},
            "holdout": {"start": str(HOLDOUT_START), "end": str(end)},
            "formula": "0.6 * minmax(oos_sharpe) + 0.4 * minmax(hac_alpha_t)",
            "ranking_population": "warm_start candidates only",
        },
        "search_matrix": {
            "factor_sets": ["full88", "r50", "icir25", "icir35", "icir50"],
            "hyperparameter_profiles": list(_lgbm_profiles()),
            "candidate_count": len(plan_candidates),
            "feature_fraction": 1.0,
            "feature_fraction_bynode": 1.0,
            "rolling_window_days_including_score_day": ROLLING_WINDOW_DAYS_INCLUDING_SCORE_DAY,
            "prior_history_days": PRIOR_HISTORY_DAYS,
            "fit_train_days": FIT_TRAIN_DAYS,
            "fit_validation_days": FIT_VALIDATION_DAYS,
        },
        "warm_start_contract": {
            "mode": "strict_checkpoint_chain",
            "bootstrap": "one_cold_refit_at_first_score_day",
            "post_bootstrap": "require_immediately_previous_refit_checkpoint",
            "failure_policy": "fail_closed",
            "audit": "warm_start_coverage/<candidate>.json",
        },
        "evidence": {
            "level": EVIDENCE_LEVEL,
            "promotion_allowed": False,
            "reason": "R88 screened61 provenance used a later full-period screen; this study is conditional exploratory evidence only",
        },
        "candidates": plan_candidates,
    }
    _write_json(run_root / "study_plan.json", plan)
    _write_study_integrity(run_root, plan)
    return plan


def _load_plan(study_root: Path, run_label: str) -> tuple[Path, dict[str, Any]]:
    run_root = _run_root_path(study_root, run_label)
    plan_path = run_root / "study_plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"study plan does not exist; run --mode prepare first: {plan_path}")
    plan = _read_json(plan_path)
    if plan.get("study_id") != STUDY_ID or not bool(plan.get("research_only", False)):
        raise RuntimeError("study plan is not the expected research-only R88 plan")
    if not _same_path(plan.get("run_root", ""), run_root):
        raise RuntimeError("study plan run_root does not match the requested run label")
    factor_input = plan.get("factor_input")
    if not isinstance(factor_input, dict):
        raise RuntimeError("R88 study plans require an explicit canonical experiment-table input block")
    _, current_input, _, _ = _load_r88_contract()
    if factor_input != {"mode": R88_INPUT_MODE, **current_input}:
        raise RuntimeError("R88 canonical experiment-table input changed after the immutable study plan was created")
    _verify_study_integrity(run_root, plan)
    return run_root, plan


def _candidate_plan(plan: dict[str, Any], candidate: str) -> dict[str, Any]:
    matches = [row for row in plan.get("candidates", []) if row.get("candidate") == candidate]
    if len(matches) != 1:
        known = ", ".join(str(row.get("candidate")) for row in plan.get("candidates", []))
        raise KeyError(f"unknown candidate {candidate!r}; known: {known}")
    return matches[0]


def _research_env(paths_config: str) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("CBOND_ON_RUNTIME_ROOT", "CBOND_ON_RAW_ROOT", "CBOND_ON_CLEAN_ROOT", "CBOND_ON_PATHS_PROFILE"):
        env.pop(key, None)
    env["CBOND_ON_PATHS_CONFIG"] = paths_config
    return env


def _invoke(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}); inspect {log_path}")


def _find_backtest_daily_returns(run_root: Path, candidate: str) -> Path:
    root = run_root / "runtime" / "results" / "backtest"
    matches = sorted(root.glob(f"**/{STUDY_ID}_{candidate}/**/daily_returns.csv"), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"daily_returns.csv not found for {candidate} below {root}")
    return matches[-1]


def _csv_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0", ""}:
        return False
    return None


def _audit_strict_warm_start(
    run_root: Path,
    candidate_row: dict[str, Any],
    candidate: str,
) -> dict[str, Any]:
    """Prove each R88 post-bootstrap refit retained the immediate checkpoint.

    Score coverage alone cannot distinguish a genuine warm chain from a
    runner that quietly retrained one or more days cold.  This audit is only
    applicable to the fixed-mask warm-start research candidates.
    """

    audit_path = run_root / "warm_start_coverage" / f"{candidate}.json"
    if str(candidate_row.get("state_mode")) != "warm_start":
        audit = {
            "schema": "r88_joint_lgbm_strict_warm_start/v1",
            "created_at_utc": _now(),
            "candidate": candidate,
            "status": "not_applicable",
            "reason": "candidate is not a warm-start arm",
        }
        _write_json(audit_path, audit)
        return audit

    model_cfg = load_json_like(Path(candidate_row["model_config"]))
    incremental = model_cfg.get("incremental", {})
    strict_cfg = incremental.get("strict_warm_start", {}) if isinstance(incremental, dict) else {}
    state_dir = Path(incremental.get("state_dir", "")) if isinstance(incremental, dict) else Path()
    output_root = run_root / "runtime" / "results" / "models" / candidate
    rolling_paths = sorted(output_root.glob("**/rolling_metrics.csv"), key=lambda path: path.stat().st_mtime)
    errors: list[str] = []
    if len(rolling_paths) != 1:
        errors.append(f"expected exactly one rolling_metrics.csv, found {len(rolling_paths)}")
        rolling = pd.DataFrame()
        rolling_path: Path | None = rolling_paths[-1] if rolling_paths else None
    else:
        rolling_path = rolling_paths[0]
        rolling = pd.read_csv(rolling_path)
    required_columns = {
        "trade_date",
        "strict_warm_start",
        "warm_start_active",
        "warm_start_required",
        "warm_start_bootstrap",
        "warm_start_checkpoint",
        "warm_start_source_day",
    }
    missing_columns = sorted(required_columns - set(rolling.columns))
    if missing_columns:
        errors.append("rolling_metrics.csv missing columns: " + ", ".join(missing_columns))

    _, _, _, contract_days = _load_r88_contract()
    start = pd.Timestamp(candidate_row["start"]).date()
    end = pd.Timestamp(candidate_row["end"]).date()
    expected_days = [day for day in contract_days if start <= day <= end]
    observed_days: list[date] = []
    if not rolling.empty and "trade_date" in rolling:
        parsed_days = pd.to_datetime(rolling["trade_date"], errors="coerce")
        if parsed_days.isna().any():
            errors.append("rolling_metrics.csv contains invalid trade_date")
        observed_days = [value.date() for value in parsed_days.dropna().tolist()]
        if observed_days != expected_days:
            errors.append(
                "rolling warm-start dates differ from the frozen candidate range: "
                f"expected={len(expected_days)} observed={len(observed_days)}"
            )

    require_initial = bool(strict_cfg.get("require_initial_checkpoint", False)) if isinstance(strict_cfg, dict) else False
    initial_day: date | None = None
    if require_initial:
        try:
            initial_day = pd.Timestamp(strict_cfg.get("initial_checkpoint_day")).date()
        except Exception:
            errors.append("strict warm-start initial checkpoint day is invalid")
    if not isinstance(strict_cfg, dict) or strict_cfg.get("enabled") is not True:
        errors.append("candidate model config does not enable strict warm start")

    if not missing_columns and len(observed_days) == len(expected_days):
        rolling = rolling.copy()
        rolling["trade_date"] = pd.to_datetime(rolling["trade_date"], errors="coerce").dt.date
        for idx, row in rolling.reset_index(drop=True).iterrows():
            day = row["trade_date"]
            enabled = _csv_bool(row["strict_warm_start"])
            active = _csv_bool(row["warm_start_active"])
            required = _csv_bool(row["warm_start_required"])
            bootstrap = _csv_bool(row["warm_start_bootstrap"])
            checkpoint_value = row["warm_start_checkpoint"]
            source_day_value = row["warm_start_source_day"]
            checkpoint = "" if pd.isna(checkpoint_value) else str(checkpoint_value).strip()
            source_day = "" if pd.isna(source_day_value) else str(source_day_value).strip()
            if enabled is not True:
                errors.append(f"{day}: strict_warm_start is not true")
                continue
            if idx == 0 and not require_initial:
                if active is not False or required is not False or bootstrap is not True or checkpoint:
                    errors.append(f"{day}: first strict row is not the declared cold bootstrap")
            else:
                expected_source = initial_day if idx == 0 else expected_days[idx - 1]
                expected_checkpoint = f"{expected_source:%Y-%m-%d}.txt" if expected_source else ""
                if active is not True or required is not True or bootstrap is not False:
                    errors.append(f"{day}: strict warm-start flags are not active/required/non-bootstrap")
                if checkpoint != expected_checkpoint:
                    errors.append(
                        f"{day}: warm-start checkpoint mismatch expected={expected_checkpoint} observed={checkpoint}"
                    )
                if source_day != str(expected_source):
                    errors.append(
                        f"{day}: warm-start source day mismatch expected={expected_source} observed={source_day}"
                    )

    expected_checkpoint_names = [f"{day:%Y-%m-%d}.txt" for day in expected_days]
    if initial_day is not None:
        expected_checkpoint_names = [f"{initial_day:%Y-%m-%d}.txt", *expected_checkpoint_names]
    observed_checkpoint_names = sorted(path.name for path in state_dir.glob("*.txt")) if state_dir.is_dir() else []
    if sorted(expected_checkpoint_names) != observed_checkpoint_names:
        errors.append(
            "strict warm-start checkpoint inventory differs from the exact chain: "
            f"expected={len(expected_checkpoint_names)} observed={len(observed_checkpoint_names)}"
        )
    audit = {
        "schema": "r88_joint_lgbm_strict_warm_start/v1",
        "created_at_utc": _now(),
        "candidate": candidate,
        "configured_range": {"start": str(start), "end": str(end)},
        "expected_score_days": [str(day) for day in expected_days],
        "actual_score_days": [str(day) for day in observed_days],
        "rolling_metrics_path": str(rolling_path) if rolling_path is not None else None,
        "state_dir": str(state_dir),
        "initial_checkpoint_day": str(initial_day) if initial_day else None,
        "expected_checkpoint_names": expected_checkpoint_names,
        "actual_checkpoint_names": observed_checkpoint_names,
        "errors": errors,
        "status": "ok" if not errors else "failed",
    }
    _write_json(audit_path, audit)
    if errors:
        raise RuntimeError("strict warm-start audit failed: " + "; ".join(errors[:3]))
    return audit


def _audit_score_coverage(
    run_root: Path,
    candidate_row: dict[str, Any],
    candidate: str,
) -> dict[str, Any]:
    """Make every unavailable score day explicit before strategy evaluation."""

    _, _, _, contract_days = _load_r88_contract()
    start = pd.Timestamp(candidate_row["start"]).date()
    end = pd.Timestamp(candidate_row["end"]).date()
    expected = [day for day in contract_days if start <= day <= end]
    score_root = run_root / "runtime" / "results" / "scores" / candidate
    score_cache = load_scores_by_date(score_root)
    actual = sorted(score_cache)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    invalid_scores: list[dict[str, Any]] = []
    for day in actual:
        frame = score_cache[day]
        codes = frame.get("code", pd.Series(dtype="object"))
        scores = pd.to_numeric(frame.get("score", pd.Series(dtype=float)), errors="coerce")
        if frame.empty or codes.astype(str).str.strip().eq("").any() or codes.duplicated().any() or not np.isfinite(scores).all():
            invalid_scores.append({"day": str(day), "rows": int(len(frame))})
    first_actual = actual[0] if actual else None
    last_actual = actual[-1] if actual else None
    audit = {
        "schema": "r88_joint_lgbm_score_coverage/v1",
        "created_at_utc": _now(),
        "candidate": candidate,
        "configured_range": {"start": str(start), "end": str(end)},
        "expected_factor_days": int(len(expected)),
        "expected_score_days": [str(day) for day in expected],
        "actual_score_days": int(len(actual)),
        "actual_score_range": {"start": str(first_actual) if first_actual else None, "end": str(last_actual) if last_actual else None},
        "missing_expected_score_days": [str(day) for day in missing],
        "unexpected_score_days": [str(day) for day in unexpected],
        "invalid_score_days": invalid_scores,
    }
    _write_json(run_root / "coverage" / f"{candidate}.json", audit)
    if missing or unexpected or invalid_scores:
        raise RuntimeError(
            "score coverage is not exact; do not backtest an incomplete candidate: "
            f"missing={','.join(str(day) for day in missing)} "
            f"unexpected={','.join(str(day) for day in unexpected)} invalid={invalid_scores}"
        )
    return audit


def _audit_backtest_coverage(run_root: Path, candidate_row: dict[str, Any], candidate: str) -> dict[str, Any]:
    """Require one successful strict strategy return for every expected score day."""

    _, _, _, contract_days = _load_r88_contract()
    start = pd.Timestamp(candidate_row["start"]).date()
    end = pd.Timestamp(candidate_row["end"]).date()
    expected = [day for day in contract_days if start <= day <= end]
    daily_path = _find_backtest_daily_returns(run_root, candidate)
    daily = pd.read_csv(daily_path)
    if "trade_date" not in daily.columns:
        raise KeyError("backtest daily_returns.csv is missing trade_date")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.date
    required = {"day_return", "benchmark_return", "benchmark_method"}
    missing_columns = sorted(required - set(daily.columns))
    if missing_columns:
        raise KeyError(f"backtest daily_returns.csv missing columns: {missing_columns}")
    nonfinite = daily[
        daily["trade_date"].isna()
        | ~np.isfinite(pd.to_numeric(daily["day_return"], errors="coerce"))
        | ~np.isfinite(pd.to_numeric(daily["benchmark_return"], errors="coerce"))
    ]
    observed = daily["trade_date"].tolist()
    duplicate_days = sorted(str(day) for day in daily.loc[daily["trade_date"].duplicated(), "trade_date"].dropna())
    missing_days = sorted(set(expected) - set(observed))
    unexpected_days = sorted(set(observed) - set(expected))
    diagnostics_path = daily_path.parent / "diagnostics.csv"
    bad_diagnostics: list[dict[str, Any]] = []
    if diagnostics_path.is_file():
        diagnostics = pd.read_csv(diagnostics_path)
        diagnostics["trade_date"] = pd.to_datetime(diagnostics["trade_date"], errors="coerce").dt.date
        for _, row in diagnostics.iterrows():
            if row.get("trade_date") in set(expected) and str(row.get("status", "")) != "ok":
                bad_diagnostics.append(
                    {"trade_date": str(row.get("trade_date")), "reason": str(row.get("reason", ""))}
                )
    audit = {
        "schema": "r88_joint_lgbm_backtest_coverage/v1",
        "created_at_utc": _now(),
        "candidate": candidate,
        "expected_execution_days": [str(day) for day in expected],
        "actual_return_days": [str(day) for day in observed],
        "missing_execution_days": [str(day) for day in missing_days],
        "unexpected_return_days": [str(day) for day in unexpected_days],
        "duplicate_return_days": duplicate_days,
        "nonfinite_return_rows": int(len(nonfinite)),
        "non_ok_diagnostics": bad_diagnostics,
        "daily_returns_path": str(daily_path),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path.is_file() else None,
    }
    _write_json(run_root / "backtest_coverage" / f"{candidate}.json", audit)
    if missing_days or unexpected_days or duplicate_days or not nonfinite.empty or bad_diagnostics:
        raise RuntimeError(
            "strict strategy coverage failed: "
            f"missing={len(missing_days)} unexpected={len(unexpected_days)} "
            f"duplicate={len(duplicate_days)} nonfinite={len(nonfinite)} diagnostics={bad_diagnostics[:3]}"
        )
    return audit


def _hac_alpha_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    required = {"trade_date", "day_return", "benchmark_return", "benchmark_method"}
    missing = sorted(required - set(daily.columns))
    if missing:
        raise KeyError(f"daily returns missing required columns: {missing}")
    work = daily[["trade_date", "day_return", "benchmark_return", "benchmark_method"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date
    work["day_return"] = pd.to_numeric(work["day_return"], errors="coerce")
    work["benchmark_return"] = pd.to_numeric(work["benchmark_return"], errors="coerce")
    work = work.dropna(subset=["trade_date", "day_return", "benchmark_return"]).sort_values("trade_date")
    if work["trade_date"].duplicated().any():
        duplicated = sorted(str(day) for day in work.loc[work["trade_date"].duplicated(), "trade_date"])
        raise RuntimeError(f"daily returns contain duplicate trade dates: {duplicated[:5]}")
    if len(work) < 30:
        raise RuntimeError(f"need at least 30 aligned daily returns for HAC alpha, got {len(work)}")
    methods = sorted(set(work["benchmark_method"].astype(str)))
    if methods != ["strict_official_prev_close_split"]:
        raise RuntimeError(f"unexpected benchmark method(s): {methods}")
    strategy = work["day_return"].to_numpy(dtype=float)
    benchmark = work["benchmark_return"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(work), dtype=float), benchmark])
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ strategy)
    residual = strategy - x @ beta
    n_obs = len(work)
    hac_lag = max(0, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
    score = x * residual[:, None]
    omega = score.T @ score
    for lag in range(1, hac_lag + 1):
        weight = 1.0 - lag / (hac_lag + 1.0)
        cross = score[lag:].T @ score[:-lag]
        omega += weight * (cross + cross.T)
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ omega @ bread
    alpha_se = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
    alpha_t = float(beta[0] / alpha_se) if alpha_se > 0 else float("nan")
    alpha_p = float(2.0 * norm.sf(abs(alpha_t))) if np.isfinite(alpha_t) else float("nan")
    mean_return = float(np.mean(strategy))
    std_return = float(np.std(strategy, ddof=1))
    sharpe = float(mean_return / std_return * np.sqrt(252.0)) if std_return > 0 else float("nan")
    nav = np.cumprod(1.0 + strategy)
    drawdown = nav / np.maximum.accumulate(nav) - 1.0
    weekly = work.assign(period=pd.to_datetime(work["trade_date"]).dt.to_period("W-FRI"))["day_return"].groupby(
        work.assign(period=pd.to_datetime(work["trade_date"]).dt.to_period("W-FRI"))["period"]
    ).apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
    monthly = work.assign(period=pd.to_datetime(work["trade_date"]).dt.to_period("M"))["day_return"].groupby(
        work.assign(period=pd.to_datetime(work["trade_date"]).dt.to_period("M"))["period"]
    ).apply(lambda values: float(np.prod(1.0 + values.to_numpy(dtype=float)) - 1.0))
    return {
        "n_obs": int(n_obs),
        "start": str(work["trade_date"].min()),
        "end": str(work["trade_date"].max()),
        "oos_sharpe": sharpe,
        "mean_daily_return": mean_return,
        "annualized_return_simple": mean_return * 252.0,
        "max_drawdown": float(np.min(drawdown)),
        "alpha_daily": float(beta[0]),
        "alpha_daily_bp": float(beta[0] * 10000.0),
        "alpha_annualized_simple": float(beta[0] * 252.0),
        "beta": float(beta[1]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((strategy - np.mean(strategy)) ** 2)),
        "hac_kernel": "bartlett",
        "hac_lag": int(hac_lag),
        "alpha_hac_se": alpha_se,
        "alpha_hac_t": alpha_t,
        "alpha_hac_p_two_sided": alpha_p,
        "weekly_positive_fraction": float((weekly > 0.0).mean()) if len(weekly) else float("nan"),
        "monthly_positive_fraction": float((monthly > 0.0).mean()) if len(monthly) else float("nan"),
        "benchmark_method": methods[0],
        "daily_returns_sha256": _sha256_bytes(work[["trade_date", "day_return", "benchmark_return"]]),
    }


def _maybe_hac_alpha_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    """Return an explicit smoke/partial status rather than inventing HAC data."""

    try:
        return _hac_alpha_metrics(daily)
    except RuntimeError as exc:
        if "need at least 30 aligned daily returns" not in str(exc):
            raise
        return {
            "status": "insufficient_observations_for_hac",
            "n_obs": int(len(daily)),
            "minimum_n_obs": 30,
        }


def _sha256_bytes(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def evaluate_candidate(study_root: Path, run_label: str, candidate: str) -> dict[str, Any]:
    run_root, plan = _load_plan(study_root, run_label)
    row = _candidate_plan(plan, candidate)
    daily_path = _find_backtest_daily_returns(run_root, candidate)
    daily = pd.read_csv(daily_path)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.date
    overall = _maybe_hac_alpha_metrics(daily)
    (development_start, development_end), (holdout_start, holdout_end) = _objective_windows(plan)
    development = _maybe_hac_alpha_metrics(
        daily[(daily["trade_date"] >= development_start) & (daily["trade_date"] <= development_end)].copy()
    )
    holdout = daily[(daily["trade_date"] >= holdout_start) & (daily["trade_date"] <= holdout_end)].copy()
    holdout_metrics = _maybe_hac_alpha_metrics(holdout)
    metrics = {
        "schema": "r88_joint_lgbm_candidate_metrics/v1",
        "created_at_utc": _now(),
        "study_id": STUDY_ID,
        "candidate": candidate,
        "state_mode": row["state_mode"],
        "factor_mode": row["factor_mode"],
        "evidence_level": plan.get("evidence", {}).get("level", EVIDENCE_LEVEL),
        "daily_returns_path": str(daily_path),
        "overall": overall,
        "development": development,
        "holdout_reporting_only": holdout_metrics,
        "reporting_window": {"start": str(holdout_start), "end": str(holdout_end)},
    }
    _write_json(run_root / "metrics" / f"{candidate}.json", metrics)
    return metrics


def run_candidate(study_root: Path, run_label: str, candidate: str) -> dict[str, Any]:
    run_root, plan = _load_plan(study_root, run_label)
    row = _candidate_plan(plan, candidate)
    integrity = _verify_study_integrity(run_root, plan)
    paths_path, score_config_path, backtest_config_path = _validate_run_configs(run_root, plan, row)
    paths_config = str(paths_path)
    env = _research_env(paths_config)
    score_log = run_root / "logs" / f"{candidate}.score.log"
    backtest_log = run_root / "logs" / f"{candidate}.backtest.log"
    status_path = run_root / "status" / f"{candidate}.json"
    started = _now()
    _write_json(
        status_path,
        {
            "candidate": candidate,
            "status": "running",
            "started_at_utc": started,
            "research_only": True,
            "study_integrity_sha256": _sha256(_integrity_path(run_root)),
        },
    )
    try:
        _invoke(
            [
                sys.executable,
                "-B",
                "-m",
                "cbond_on.cli.model_score",
                "--config",
                str(score_config_path),
                "--model-id",
                candidate,
            ],
            env=env,
            log_path=score_log,
        )
        warm_start_coverage = _audit_strict_warm_start(run_root, row, candidate)
        coverage = _audit_score_coverage(run_root, row, candidate)
        _invoke(
            [
                sys.executable,
                "-B",
                "-m",
                "cbond_on.cli.strategy_backtest",
                "--config",
                str(backtest_config_path),
            ],
            env=env,
            log_path=backtest_log,
        )
        backtest_coverage = _audit_backtest_coverage(run_root, row, candidate)
        metrics = evaluate_candidate(study_root, run_label, candidate)
        status = {
            "candidate": candidate,
            "status": "completed",
            "started_at_utc": started,
            "completed_at_utc": _now(),
            "score_log": str(score_log),
            "backtest_log": str(backtest_log),
            "metrics": str(run_root / "metrics" / f"{candidate}.json"),
            "warm_start_coverage": str(run_root / "warm_start_coverage" / f"{candidate}.json"),
            "score_coverage": str(run_root / "coverage" / f"{candidate}.json"),
            "backtest_coverage": str(run_root / "backtest_coverage" / f"{candidate}.json"),
            "study_integrity_sha256": _sha256(_integrity_path(run_root)),
        }
        _write_json(status_path, status)
        return {"status": status, "metrics": metrics}
    except Exception as exc:
        status = {
            "candidate": candidate,
            "status": "failed",
            "started_at_utc": started,
            "failed_at_utc": _now(),
            "error": f"{type(exc).__name__}: {exc}",
            "score_log": str(score_log),
            "backtest_log": str(backtest_log),
            "study_integrity_sha256": _sha256(_integrity_path(run_root)),
        }
        _write_json(status_path, status)
        raise


def rank_candidates(study_root: Path, run_label: str) -> pd.DataFrame:
    run_root, plan = _load_plan(study_root, run_label)
    (development_start, development_end), _ = _objective_windows(plan)
    rows: list[dict[str, Any]] = []
    common_dates: list[str] | None = None
    common_benchmark: list[float] | None = None
    for candidate in plan["candidates"]:
        name = str(candidate["candidate"])
        path = run_root / "metrics" / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"candidate metrics missing: {path}")
        metrics = _read_json(path)
        if str(candidate["state_mode"]) == "warm_start":
            daily = pd.read_csv(metrics["daily_returns_path"])
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.date
            daily = daily[(daily["trade_date"] >= development_start) & (daily["trade_date"] <= development_end)].copy()
            daily = daily.dropna(subset=["trade_date", "day_return", "benchmark_return"]).sort_values("trade_date")
            dates = [str(value) for value in daily["trade_date"].tolist()]
            benchmark = [float(value) for value in pd.to_numeric(daily["benchmark_return"], errors="coerce").tolist()]
            if common_dates is None:
                common_dates, common_benchmark = dates, benchmark
            elif dates != common_dates or benchmark != common_benchmark:
                raise RuntimeError(
                    "candidate development return dates or benchmark series differ; do not rank unaligned runs"
                )
        dev = metrics["development"]
        has_primary_metrics = all(key in dev for key in ("oos_sharpe", "alpha_hac_t", "alpha_hac_p_two_sided"))
        rows.append(
            {
                "candidate": name,
                "state_mode": candidate["state_mode"],
                "factor_mode": candidate["factor_mode"],
                "n_obs": dev.get("n_obs"),
                "oos_sharpe": dev.get("oos_sharpe"),
                "alpha_hac_t": dev.get("alpha_hac_t"),
                "alpha_hac_p_two_sided": dev.get("alpha_hac_p_two_sided"),
                "alpha_daily_bp": dev.get("alpha_daily_bp"),
                "beta": dev.get("beta"),
                "max_drawdown": dev.get("max_drawdown"),
                "weekly_positive_fraction": dev.get("weekly_positive_fraction"),
                "monthly_positive_fraction": dev.get("monthly_positive_fraction"),
                "metric_status": dev.get("status", "ok"),
                "has_primary_metrics": has_primary_metrics,
            }
        )
    frame = pd.DataFrame(rows)
    frame["primary_ranking_eligible"] = frame["state_mode"].eq("warm_start") & frame["has_primary_metrics"]
    eligible = frame["primary_ranking_eligible"]
    if not bool(eligible.any()):
        raise RuntimeError("no warm-start candidate has enough observations for the primary HAC-alpha ranking")
    for source, output in (("oos_sharpe", "sharpe_minmax"), ("alpha_hac_t", "alpha_hac_t_minmax")):
        values = pd.to_numeric(frame.loc[eligible, source], errors="coerce")
        low, high = values.min(), values.max()
        if not np.isfinite(low) or not np.isfinite(high):
            raise RuntimeError(f"non-finite primary objective column: {source}")
        frame[output] = np.nan
        frame.loc[eligible, output] = 0.5 if abs(high - low) <= 1e-12 else (values - low) / (high - low)
    frame["primary_composite"] = np.where(
        frame["primary_ranking_eligible"],
        0.6 * frame["sharpe_minmax"] + 0.4 * frame["alpha_hac_t_minmax"],
        np.nan,
    )
    frame = frame.sort_values(["primary_ranking_eligible", "primary_composite"], ascending=[False, False])
    frame.to_csv(run_root / "metrics" / "candidate_ranking.csv", index=False)
    _write_json(
        run_root / "metrics" / "candidate_ranking_manifest.json",
        {
            "created_at_utc": _now(),
            "study_id": STUDY_ID,
            "development": {"start": str(development_start), "end": str(development_end)},
            "objective": "0.6 * minmax(oos_sharpe) + 0.4 * minmax(hac_alpha_t)",
            "ranking_population": "warm_start candidates only",
            "evidence_level": plan.get("evidence", {}).get("level", EVIDENCE_LEVEL),
            "promotion_allowed": False,
            "aligned_development_dates": common_dates,
            "benchmark_sha256": _hash_json(common_benchmark or []),
        },
    )
    return frame


def run_all(study_root: Path, run_label: str) -> pd.DataFrame:
    """Resume a prepared study serially, then rank only completed aligned arms."""

    run_root, plan = _load_plan(study_root, run_label)
    for candidate_row in plan["candidates"]:
        candidate = str(candidate_row["candidate"])
        status_path = run_root / "status" / f"{candidate}.json"
        if status_path.exists():
            status = _read_json(status_path)
            if status.get("status") == "completed" and status.get("study_integrity_sha256") == _sha256(
                _integrity_path(run_root)
            ):
                print(f"[study] skip completed candidate={candidate}", flush=True)
                continue
        print(f"[study] run candidate={candidate}", flush=True)
        run_candidate(study_root, run_label, candidate)
    return rank_candidates(study_root, run_label)


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "evaluate", "rank", "run-all"), required=True)
    parser.add_argument(
        "--factor-root",
        help="rejected: direct R88 factor_data roots are migration/audit provenance only",
    )
    parser.add_argument(
        "--backfill-manifest",
        help="rejected: normal R88 research admits the canonical experiment table manifest",
    )
    parser.add_argument(
        "--study-id",
        help="optional research-only study identity; input is always canonical experiment",
    )
    parser.add_argument("--study-root")
    parser.add_argument("--run-label", default="full")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--start", default=str(DEV_START))
    parser.add_argument("--end")
    args = parser.parse_args(argv)
    _configure_runtime(
        factor_root_text=args.factor_root,
        manifest_text=args.backfill_manifest,
        study_id_text=args.study_id,
    )
    study_root = _require_scratch(Path(args.study_root or DEFAULT_STUDY_ROOT))
    start, end = _parse_date(args.start), _parse_date(args.end or str(DEFAULT_END))
    if start > end:
        raise ValueError("start must be <= end")
    if args.mode == "prepare":
        plan = prepare(study_root, args.run_label, start, end)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.mode == "run":
        if not args.candidate:
            raise ValueError("--candidate is required for --mode run")
        result = run_candidate(study_root, args.run_label, args.candidate)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.mode == "evaluate":
        if not args.candidate:
            raise ValueError("--candidate is required for --mode evaluate")
        print(json.dumps(evaluate_candidate(study_root, args.run_label, args.candidate), ensure_ascii=False, indent=2))
        return 0
    if args.mode == "run-all":
        frame = run_all(study_root, args.run_label)
        print(frame.to_string(index=False))
        return 0
    frame = rank_candidates(study_root, args.run_label)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
