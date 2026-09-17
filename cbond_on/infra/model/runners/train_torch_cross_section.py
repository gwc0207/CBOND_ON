from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.trading_days import list_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors.factor_table_resolution import CanonicalFactorTableReader, build_factor_reader
from cbond_on.infra.factors.r88_experiment_table import admit_r88_experiment_table
from cbond_on.infra.model.impl.lgbm.trainer import (
    SplitData,
    _build_day_group_indices,
    _iter_existing_label_days,
    _mean_pearson_ic_by_groups,
    _mean_rank_ic_by_groups,
    build_dataset,
    build_tradable_code_map,
)
from cbond_on.infra.model.impl.torch_cross_section import build_cross_section_model
from cbond_on.infra.model.neutralization import build_neutralizer
from cbond_on.infra.model.preprocess_config import parse_winsor_bounds
from cbond_on.infra.model.score_io import write_scores_by_date, write_scores_for_day_atomic


_PROTECTED_ROOTS = (
    Path("D:/cbond_on/results/live"),
    Path("D:/cbond_on/results/model_state"),
    Path("D:/cbond_on/results/backtest"),
)

# The live50 path remains the historical default, including its checkpoint
# fingerprint.  R88 is deliberately a separate, research-only contract: it
# enters through the canonical experiment table and never inherits a live
# profile or a direct scattered FactorStore root.
_LIVE50_FACTOR_CONTRACT_REF = "factor_contracts/profiles/live50_rust50_20260806.json5"
_R88_FACTOR_CONTRACT_REF = "factor_contracts/profiles/research_r88_rust88_20260825.json5"
_R88_ADMISSION_PROFILE = "research_r88_rust88_20260825"
_R88_RESEARCH_CONFIG_PROFILE = "research_r88_rust88_20260825"
_R88_FACTOR_COUNT = 88
_R88_RESEARCH_SCRATCH_ROOT = Path("D:/cbond_on/research_scratch")
_R88_CANONICAL_TABLE_ID = "experiment"
_R88_ALLOWED_MIN_COVERAGE_FRACTIONS = (0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
_R88_BACKFILL_MANIFEST_SCHEMA = "r88_factor_backfill_manifest/v1"
_R88_BACKFILL_COMPLETION_DEFINITION = "requested_approved_range_and_exact_frozen_r50_day_coverage"
_R88_APPROVED_BACKFILL_RANGE = {"start": "2024-01-01", "end": "2026-07-30"}
_R88_FROZEN_R50_FACTOR_ROOT = (
    _R88_RESEARCH_SCRATCH_ROOT
    / "rust50_unified_final_20260806_160508"
    / "fullchain_preseed_r4"
    / "runtime_r2"
    / "factor_data"
)
_R88_R38_EXECUTION_MODE = "r38_only_plus_frozen_r50_merge"


@dataclass(frozen=True)
class CrossSectionSplit:
    x: np.ndarray
    y: np.ndarray
    dt: np.ndarray
    code: np.ndarray
    groups: list[np.ndarray]


@dataclass(frozen=True)
class CheckpointRecord:
    """A provenance-checked daily warm-start checkpoint.

    Checkpoints are the only durable model artifacts produced by the original
    interrupted r3 run.  Keeping their provenance in one explicit structure
    lets a resume reconstruct the corresponding score day without retraining
    or mutating the state chain.
    """

    path: Path
    score_day: date
    model_state: dict[str, torch.Tensor]
    max_train_label_day: date
    max_validation_label_day: date
    previous_checkpoint: Path | None


def _load_model_config(path: Path | None) -> dict:
    if path is None:
        raise ValueError("torch cross-section model config path is required")
    return load_config_file(str(path))


def _set_deterministic_seed(*, seed: int, deterministic: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _seed_for_score_day(*, base_seed: int, score_day: date) -> int:
    return (int(base_seed) + int(score_day.strftime("%Y%m%d"))) % (2**31 - 1)


def _require_research_contract(cfg: dict, *, results_root: Path, score_output: Path, state_dir: Path) -> None:
    rolling = dict(cfg.get("rolling", {}))
    incremental = dict(cfg.get("incremental", {}))
    if not bool(rolling.get("enabled", False)):
        raise ValueError("torch cross-section research requires rolling.enabled=true")
    if int(cfg.get("refit_every_n_days", 1)) != 1:
        raise ValueError("torch cross-section research requires refit_every_n_days=1")
    if bool(incremental.get("skip_existing_scores", True)):
        raise ValueError("torch cross-section research requires incremental.skip_existing_scores=false")
    if not bool(incremental.get("warm_start", False)):
        raise ValueError("torch cross-section research requires incremental.warm_start=true")
    if not bool(incremental.get("save_state", False)):
        raise ValueError("torch cross-section research requires incremental.save_state=true")
    if not bool(cfg.get("score_only_no_target_label_read", False)):
        raise ValueError("torch cross-section research requires score_only_no_target_label_read=true")
    if not bool(cfg.get("score_only_apply_tradable_filter", False)):
        raise ValueError("torch cross-section research requires score_only_apply_tradable_filter=true")
    if not bool(dict(cfg.get("tradable_filter", {})).get("strict", False)):
        raise ValueError("torch cross-section research requires tradable_filter.strict=true")
    for value in (results_root, score_output, state_dir):
        resolved = value.resolve(strict=False)
        if any(str(resolved).lower().startswith(str(root).lower()) for root in _PROTECTED_ROOTS):
            raise ValueError(f"torch cross-section research output must not use production root: {resolved}")


def _config_fingerprint(
    *,
    architecture: str,
    factors: list[str],
    model_input_factors: Sequence[str] | None = None,
    model_params: dict,
    input_missingness: dict,
    objective: dict,
    neutralization: dict,
    contract: dict,
) -> str:
    payload = {
        "runner_contract_version": "torch_cross_section_r3_v4_strict_effective_coverage",
        "architecture": str(architecture),
        "factors": list(factors),
        "model_params": dict(model_params),
        "input_missingness": dict(input_missingness),
        "objective": dict(objective),
        "neutralization": dict(neutralization),
        "contract": dict(contract),
    }
    # Keep the historical full50 fingerprint byte-for-byte compatible with
    # r3/v5 checkpoints.  A subset is a new research contract and must be
    # explicit in its own fingerprint.
    if model_input_factors is not None and list(model_input_factors) != list(factors):
        payload["model_input_factors"] = list(model_input_factors)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_factor_contract(*, contract_ref: str, factors: list[str]) -> dict:
    """Pin model input to the immutable Rust-50 admission profile."""

    expected_ref = _LIVE50_FACTOR_CONTRACT_REF
    if str(contract_ref).replace("\\", "/") != expected_ref:
        raise ValueError(f"torch cross-section r3 requires factor_contract={expected_ref}")
    path = PROJECT_ROOT / "cbond_on" / expected_ref
    if not path.exists():
        raise FileNotFoundError(f"frozen live50 factor contract is missing: {path}")
    import json5

    loaded = json5.loads(path.read_text(encoding="utf-8"))
    if [str(item) for item in loaded.get("factors", [])] != factors:
        raise ValueError("torch cross-section factors do not exactly match frozen live50 contract order")
    digest = str(loaded.get("specs_sha256", ""))
    expected_digest = "458af9bcfeb58ab4b344f4daec9d637b7a599fb572574c5bca142b30ce8d15b0"
    if digest != expected_digest:
        raise ValueError("unexpected frozen live50 factor-contract fingerprint")
    return {"path": str(path), "admission_profile": str(loaded.get("admission_profile", "")), "specs_sha256": digest}


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _path_is_within(*, value: str | Path, root: str | Path) -> bool:
    """Compare lexical paths without requiring the research store to exist yet."""

    candidate = str(_resolved_path(value)).replace("\\", "/").rstrip("/").lower()
    boundary = str(_resolved_path(root)).replace("\\", "/").rstrip("/").lower()
    return candidate == boundary or candidate.startswith(f"{boundary}/")


def _canonical_r88_specs_sha256(raw_specs: object) -> tuple[list[str], str]:
    """Validate the explicit R88 FactorSpec payload and return its digest."""

    if not isinstance(raw_specs, list):
        raise ValueError("R88 factor contract must declare factor_specs as an ordered list")
    canonical_specs: list[dict[str, Any]] = []
    names: list[str] = []
    for idx, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"R88 factor_specs[{idx}] must be an object")
        name = str(raw_spec.get("name", "")).strip()
        factor = str(raw_spec.get("factor", "")).strip()
        params = raw_spec.get("params")
        if not name or not factor or not isinstance(params, Mapping):
            raise ValueError(f"R88 factor_specs[{idx}] must contain name, factor, and params object")
        output_col = raw_spec.get("output_col")
        rust_contract_id = raw_spec.get("rust_contract_id")
        canonical_specs.append(
            {
                "name": name,
                "factor": factor,
                "params": dict(params),
                "output_col": None if output_col is None else str(output_col),
                "rust_contract_id": None if rust_contract_id is None else str(rust_contract_id),
            }
        )
        names.append(name)
    if len(names) != _R88_FACTOR_COUNT or len(set(names)) != _R88_FACTOR_COUNT:
        raise ValueError(f"R88 factor_specs must contain exactly {_R88_FACTOR_COUNT} unique ordered factors")
    encoded = json.dumps(
        canonical_specs,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return names, hashlib.sha256(encoded).hexdigest()


def _require_r88_factor_contract(
    *,
    contract_ref: str,
    factors: list[str],
    panel_name: str,
    factor_time: str,
    label_time: str,
) -> dict:
    """Pin R88 to its explicit research-only Rust admission profile.

    The profile is intentionally not a live profile.  The runner validates the
    ordered feature universe and Rust execution declaration before opening the
    FactorStore, while the R88 build/preflight owns formula-level Rust parity.
    """

    if str(contract_ref).replace("\\", "/") != _R88_FACTOR_CONTRACT_REF:
        raise ValueError(f"torch cross-section R88 requires factor_contract={_R88_FACTOR_CONTRACT_REF}")
    path = PROJECT_ROOT / "cbond_on" / _R88_FACTOR_CONTRACT_REF
    if not path.exists():
        raise FileNotFoundError(f"frozen R88 research factor contract is missing: {path}")
    import json5

    loaded = json5.loads(path.read_text(encoding="utf-8"))
    if not bool(loaded.get("research_only", False)):
        raise ValueError("R88 factor contract must declare research_only=true")
    if str(loaded.get("admission_profile", "")).strip() != _R88_ADMISSION_PROFILE:
        raise ValueError("unexpected R88 factor-contract admission profile")
    if str(loaded.get("execution_policy", "")).strip().lower() != "rust_first":
        raise ValueError("R88 factor contract requires execution_policy='rust_first'")
    compute = loaded.get("compute")
    if not isinstance(compute, Mapping):
        raise ValueError("R88 factor contract must declare compute settings")
    if str(compute.get("engine", "")).strip().lower() != "rust":
        raise ValueError("R88 factor contract requires compute.engine='rust'")
    if str(compute.get("execution_policy", "")).strip().lower() != "rust_first":
        raise ValueError("R88 factor contract requires compute.execution_policy='rust_first'")
    profile_factors = [str(item) for item in loaded.get("factors", [])]
    if len(profile_factors) != _R88_FACTOR_COUNT or len(set(profile_factors)) != _R88_FACTOR_COUNT:
        raise ValueError(f"R88 factor contract must contain exactly {_R88_FACTOR_COUNT} unique ordered factors")
    if profile_factors != factors:
        raise ValueError("torch cross-section factors do not exactly match frozen R88 research contract order")
    spec_names, canonical_digest = _canonical_r88_specs_sha256(loaded.get("factor_specs"))
    if spec_names != profile_factors:
        raise ValueError("R88 factor_specs names do not exactly match the ordered factors contract")
    digest = str(loaded.get("specs_sha256", "")).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("R88 factor contract must provide a SHA-256 specs_sha256")
    if digest != canonical_digest:
        raise ValueError("R88 factor-contract specs_sha256 does not match its canonical factor_specs payload")
    time_contract = loaded.get("time_contract")
    if not isinstance(time_contract, Mapping):
        raise ValueError("R88 factor contract must declare a time_contract object")
    expected_time_contract = {
        "panel_name": str(panel_name),
        "factor_time": str(factor_time),
        "label_time": str(label_time),
    }
    actual_time_contract = {key: str(time_contract.get(key, "")).strip() for key in expected_time_contract}
    if actual_time_contract != expected_time_contract:
        raise ValueError(
            "R88 factor-contract time contract drift: "
            f"expected={expected_time_contract} actual={actual_time_contract}"
        )
    return {
        "path": str(path),
        "admission_profile": _R88_ADMISSION_PROFILE,
        "specs_sha256": digest,
        "research_only": True,
        "execution_status": str(loaded.get("execution_status", "")).strip(),
    }


def _require_r88_backfill_readiness(
    *,
    factor_root: Path,
    factor_contract: Mapping[str, Any],
    factors: Sequence[str],
    panel_name: str,
    factor_time: str,
    label_time: str,
    raw_root: Path,
    clean_root: Path,
) -> dict[str, Any]:
    """Require a completed, pinned Rust88 FactorStore before model input I/O."""

    if str(factor_contract.get("execution_status", "")).strip() != "eligible_for_fresh_rust_backfill":
        raise RuntimeError(
            "R88 factor contract is not eligible for model training; "
            "complete the exact Rust-contract and fresh-backfill gates first"
        )
    manifest_path = _resolved_path(factor_root) / "r88_backfill_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"R88 completed Rust backfill manifest is required: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read R88 completed Rust backfill manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise RuntimeError("R88 completed Rust backfill manifest must be an object")
    if str(manifest.get("schema_version", "")).strip() != _R88_BACKFILL_MANIFEST_SCHEMA:
        raise RuntimeError("unexpected R88 completed Rust backfill manifest schema")
    if manifest.get("research_only") is not True:
        raise RuntimeError("R88 completed Rust backfill manifest must declare research_only=true")
    if str(manifest.get("execution_status", "")).strip() != "completed_rust88_backfill":
        raise RuntimeError("R88 completed Rust backfill manifest is not completed_rust88_backfill")
    if manifest.get("model_training_ready") is not True:
        raise RuntimeError("R88 completed Rust backfill manifest is not model_training_ready")

    # A status and readiness flag alone are not sufficient: old smoke
    # manifests could be marked complete despite containing only one or a few
    # days.  Model admission is tied to the completion proof emitted by the
    # dedicated R38-only backfiller and re-checked against its durable stores.
    completion = manifest.get("completion")
    if not isinstance(completion, Mapping):
        raise RuntimeError("R88 completed Rust backfill manifest is missing completion evidence")
    if completion.get("full_completion") is not True:
        raise RuntimeError("R88 completed Rust backfill manifest completion.full_completion must be true")
    if str(completion.get("definition", "")).strip() != _R88_BACKFILL_COMPLETION_DEFINITION:
        raise RuntimeError("R88 completed Rust backfill manifest completion definition drift")

    def _require_approved_range(value: object, *, field: str) -> None:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"R88 completed Rust backfill manifest {field} must be an object")
        actual = {key: str(value.get(key, "")).strip() for key in _R88_APPROVED_BACKFILL_RANGE}
        if set(value) != set(_R88_APPROVED_BACKFILL_RANGE) or actual != _R88_APPROVED_BACKFILL_RANGE:
            raise RuntimeError(
                f"R88 completed Rust backfill manifest {field} must equal the approved R88 range"
            )

    _require_approved_range(manifest.get("range_contract"), field="range_contract")
    _require_approved_range(completion.get("approved_range"), field="completion.approved_range")
    _require_approved_range(completion.get("profile_range"), field="completion.profile_range")
    _require_approved_range(completion.get("requested_range"), field="completion.requested_range")
    blocking_reasons = completion.get("blocking_reasons")
    if not isinstance(blocking_reasons, list) or blocking_reasons:
        raise RuntimeError("R88 completed Rust backfill manifest completion has blocking reasons")

    def _require_day_list(value: object, *, field: str) -> list[date]:
        if not isinstance(value, list) or not value:
            raise RuntimeError(f"R88 completed Rust backfill manifest {field} must be a non-empty day list")
        parsed: list[date] = []
        for position, raw_day in enumerate(value):
            if not isinstance(raw_day, str):
                raise RuntimeError(
                    f"R88 completed Rust backfill manifest {field}[{position}] must be an ISO date string"
                )
            try:
                parsed_day = date.fromisoformat(raw_day)
            except ValueError as exc:
                raise RuntimeError(
                    f"R88 completed Rust backfill manifest {field}[{position}] is not an ISO date"
                ) from exc
            if raw_day != parsed_day.isoformat():
                raise RuntimeError(
                    f"R88 completed Rust backfill manifest {field}[{position}] is not canonical"
                )
            parsed.append(parsed_day)
        if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
            raise RuntimeError(f"R88 completed Rust backfill manifest {field} must be sorted and unique")
        return parsed

    expected_frozen_days = _require_day_list(
        completion.get("expected_frozen_r50_days"),
        field="completion.expected_frozen_r50_days",
    )
    execution_days = _require_day_list(
        completion.get("execution_days"),
        field="completion.execution_days",
    )
    durable_r88_days = _require_day_list(
        completion.get("r88_factor_store_days"),
        field="completion.r88_factor_store_days",
    )
    if execution_days != expected_frozen_days or durable_r88_days != expected_frozen_days:
        raise RuntimeError("R88 completed Rust backfill manifest completion coverage is not exact")

    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise RuntimeError("R88 completed Rust backfill manifest is missing execution provenance")
    if str(execution.get("mode", "")).strip() != _R88_R38_EXECUTION_MODE:
        raise RuntimeError("R88 completed Rust backfill manifest execution mode drift")
    frozen_root_raw = str(completion.get("frozen_r50_factor_root", "")).strip()
    if not frozen_root_raw:
        raise RuntimeError("R88 completed Rust backfill manifest is missing frozen R50 provenance")
    frozen_root = _resolved_path(frozen_root_raw)
    if frozen_root != _resolved_path(_R88_FROZEN_R50_FACTOR_ROOT):
        raise RuntimeError("R88 completed Rust backfill manifest frozen R50 FactorStore root drift")
    execution_frozen_root = str(execution.get("frozen_r50_factor_root", "")).strip()
    if not execution_frozen_root:
        raise RuntimeError("R88 completed Rust backfill manifest execution is missing frozen R50 provenance")
    if _resolved_path(execution_frozen_root) != frozen_root:
        raise RuntimeError("R88 completed Rust backfill manifest execution frozen R50 provenance drift")
    execution_r88_root = str(execution.get("r88_factor_root", "")).strip()
    if not execution_r88_root:
        raise RuntimeError("R88 completed Rust backfill manifest execution is missing R88 FactorStore provenance")
    if _resolved_path(execution_r88_root) != _resolved_path(factor_root):
        raise RuntimeError("R88 completed Rust backfill manifest execution R88 FactorStore root drift")

    execution_rows = execution.get("days")
    if not isinstance(execution_rows, list) or len(execution_rows) != len(execution_days):
        raise RuntimeError("R88 completed Rust backfill manifest execution day ledger drift")
    ledger_days = _require_day_list(
        [row.get("day") if isinstance(row, Mapping) else None for row in execution_rows],
        field="execution.days",
    )
    if ledger_days != expected_frozen_days:
        raise RuntimeError("R88 completed Rust backfill manifest execution day coverage drift")

    def _store_days(root: Path, *, field: str, restrict_to_approved_range: bool) -> list[date]:
        base = _resolved_path(root) / "factors" / str(panel_name)
        if not base.is_dir():
            raise RuntimeError(f"R88 completed Rust backfill manifest {field} is missing panel data: {base}")
        found: list[date] = []
        for path in base.glob("*/*.parquet"):
            try:
                parsed_day = datetime.strptime(path.stem, "%Y%m%d").date()
            except ValueError:
                continue
            if restrict_to_approved_range and not (
                date.fromisoformat(_R88_APPROVED_BACKFILL_RANGE["start"])
                <= parsed_day
                <= date.fromisoformat(_R88_APPROVED_BACKFILL_RANGE["end"])
            ):
                continue
            found.append(parsed_day)
        found.sort()
        if len(set(found)) != len(found):
            raise RuntimeError(f"R88 completed Rust backfill manifest {field} has duplicate score-day files")
        return found

    actual_frozen_days = _store_days(
        frozen_root,
        field="frozen R50 FactorStore",
        restrict_to_approved_range=True,
    )
    actual_r88_days = _store_days(
        _resolved_path(factor_root),
        field="R88 FactorStore",
        restrict_to_approved_range=False,
    )
    if actual_frozen_days != expected_frozen_days:
        raise RuntimeError("R88 completed Rust backfill manifest immutable frozen R50 coverage drift")
    if actual_r88_days != expected_frozen_days:
        raise RuntimeError("R88 completed Rust backfill manifest durable R88 FactorStore coverage drift")

    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise RuntimeError("R88 completed Rust backfill manifest is missing profile provenance")
    if str(profile.get("path", "")).replace("\\", "/") != f"cbond_on/{_R88_FACTOR_CONTRACT_REF}":
        raise RuntimeError("R88 completed Rust backfill manifest profile path drift")
    if str(profile.get("admission_profile", "")).strip() != _R88_ADMISSION_PROFILE:
        raise RuntimeError("R88 completed Rust backfill manifest admission profile drift")
    if str(profile.get("specs_sha256", "")).strip().lower() != str(factor_contract["specs_sha256"]):
        raise RuntimeError("R88 completed Rust backfill manifest factor-contract digest drift")
    if _resolved_path(str(manifest.get("factor_root", ""))) != _resolved_path(factor_root):
        raise RuntimeError("R88 completed Rust backfill manifest FactorStore root drift")
    manifest_factors = [str(value) for value in manifest.get("factors", [])]
    if int(manifest.get("factor_count", -1)) != _R88_FACTOR_COUNT or manifest_factors != list(factors):
        raise RuntimeError("R88 completed Rust backfill manifest ordered factor contract drift")
    expected_time_contract = {
        "panel_name": str(panel_name),
        "factor_time": str(factor_time),
        "label_time": str(label_time),
    }
    time_contract = manifest.get("time_contract")
    if not isinstance(time_contract, Mapping) or {
        key: str(time_contract.get(key, "")).strip() for key in expected_time_contract
    } != expected_time_contract:
        raise RuntimeError("R88 completed Rust backfill manifest time contract drift")
    source_inputs = manifest.get("source_inputs")
    if not isinstance(source_inputs, Mapping):
        raise RuntimeError("R88 completed Rust backfill manifest is missing source input provenance")
    if _resolved_path(str(source_inputs.get("raw_data_root", ""))) != _resolved_path(raw_root):
        raise RuntimeError("R88 completed Rust backfill manifest raw input root drift")
    if _resolved_path(str(source_inputs.get("clean_data_root", ""))) != _resolved_path(clean_root):
        raise RuntimeError("R88 completed Rust backfill manifest clean input root drift")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "execution_status": "completed_rust88_backfill",
        "model_training_ready": True,
    }


def _require_r88_canonical_experiment_readiness(
    *,
    paths_cfg: Mapping[str, Any],
    factor_root: Path,
    factor_contract: Mapping[str, Any],
    factors: Sequence[str],
    panel_name: str,
    factor_time: str,
    label_time: str,
) -> dict[str, Any]:
    """Admit R88 only through the canonical experiment-table reader.

    The old R88 backfill manifest remains migration/audit provenance attached
    to canonical day bundles.  It is deliberately not opened as a runtime
    FactorStore input here.
    """

    profile_path = Path(str(factor_contract.get("path", ""))).expanduser()
    if not profile_path.is_file():
        raise FileNotFoundError(f"R88 factor contract is missing: {profile_path}")
    import json5

    profile = json5.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, Mapping):
        raise RuntimeError("R88 factor contract must be a JSON object")
    binding = admit_r88_experiment_table(
        paths_cfg,
        profile_path=profile_path,
        profile=profile,
    )
    if _resolved_path(binding.table_root) != _resolved_path(factor_root):
        raise RuntimeError("R88 canonical experiment resolver root differs from paths factor_data_root")
    if list(binding.contract.factor_ids) != list(factors):
        raise RuntimeError("R88 canonical experiment contract differs from the configured factor order")
    time_contract = profile.get("time_contract")
    expected_time_contract = {
        "panel_name": str(panel_name),
        "factor_time": str(factor_time),
        "label_time": str(label_time),
    }
    if not isinstance(time_contract, Mapping) or {
        key: str(time_contract.get(key, "")).strip() for key in expected_time_contract
    } != expected_time_contract:
        raise RuntimeError("R88 canonical experiment profile time contract drift")
    return binding.to_evidence()


def _r88_min_available_factors(*, fraction: object, factor_count: int) -> int:
    """Resolve the explicit R88 availability gate without a hidden fallback.

    A 75% floor (66 of 88 factors) keeps the expanded research panel usable
    during a fresh historical backfill while rejecting sparse rows.  The small
    discrete set permits stricter research variants but prevents an arbitrary
    fractional setting from silently weakening admission.
    """

    if isinstance(fraction, bool):
        raise ValueError("R88 research_only.min_available_fraction must be a numeric approved fraction")
    try:
        parsed = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("R88 research_only.min_available_fraction is required") from exc
    if not any(np.isclose(parsed, allowed, rtol=0.0, atol=1e-12) for allowed in _R88_ALLOWED_MIN_COVERAGE_FRACTIONS):
        allowed_text = ", ".join(f"{value:.2f}" for value in _R88_ALLOWED_MIN_COVERAGE_FRACTIONS)
        raise ValueError(
            "R88 research_only.min_available_fraction must be one of "
            f"[{allowed_text}]"
        )
    return int(math.ceil(int(factor_count) * parsed))


def _require_r88_research_profile(
    cfg: dict,
    *,
    factor_root: Path,
    results_root: Path,
    score_output: Path,
    state_dir: Path,
    neutralization_cache_root: Path,
    factors: Sequence[str],
    missing_values: Mapping[str, Any],
    paths_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject every implicit or live-adjacent R88 runtime configuration."""

    research = cfg.get("research_only")
    if not isinstance(research, Mapping):
        raise ValueError("R88 requires an explicit top-level research_only object")
    research_cfg = dict(research)
    if str(research_cfg.get("profile", "")).strip() != _R88_RESEARCH_CONFIG_PROFILE:
        raise ValueError(f"R88 research_only.profile must be {_R88_RESEARCH_CONFIG_PROFILE}")
    experiment = cfg.get("experiment")
    if not isinstance(experiment, Mapping) or not bool(experiment.get("research_only", False)):
        raise ValueError("R88 requires experiment.research_only=true")
    resolved_factor_root = _resolved_path(factor_root)
    if paths_cfg is None:
        # Audit/test compatibility only.  The normal ``main`` path always
        # supplies paths_cfg and therefore takes the canonical-table branch.
        configured_root_raw = str(research_cfg.get("factor_store_root", "")).strip()
        if not configured_root_raw:
            raise ValueError("R88 research_only.factor_store_root is required for legacy audit-only validation")
        if _resolved_path(configured_root_raw) != resolved_factor_root:
            raise ValueError("R88 legacy audit factor_store_root differs from factor_root")
        if not _path_is_within(value=resolved_factor_root, root=_R88_RESEARCH_SCRATCH_ROOT):
            raise ValueError(f"R88 legacy audit FactorStore must be below research scratch: {_R88_RESEARCH_SCRATCH_ROOT}")
        if not resolved_factor_root.is_dir():
            raise FileNotFoundError(f"R88 legacy audit FactorStore root does not exist: {resolved_factor_root}")
        table_evidence: dict[str, Any] = {"factor_store_root": str(resolved_factor_root), "audit_only": True}
    else:
        if "factor_store_root" in research_cfg:
            raise ValueError("R88 research_only.factor_store_root is retired; declare canonical factor_table instead")
        configured_table = research_cfg.get("factor_table")
        paths_table = paths_cfg.get("factor_table")
        expected_table = {
            "table_id": _R88_CANONICAL_TABLE_ID,
            "root": str((paths_table or {}).get("root", "")),
        }
        if not isinstance(configured_table, Mapping) or dict(configured_table) != expected_table:
            raise ValueError("R88 research_only.factor_table must exactly match paths.factor_table experiment identity")
        if not isinstance(paths_table, Mapping) or str(paths_table.get("table_id", "")).strip() != _R88_CANONICAL_TABLE_ID:
            raise ValueError("R88 paths must declare factor_table.table_id='experiment'")
        if "factor_data_root" in dict(paths_cfg.get("read_only_input_roots", {})):
            raise ValueError("R88 canonical paths must not declare a direct read_only_input_roots.factor_data_root")
        expected_factor_root = _resolved_path(Path(str(paths_table.get("root"))) / _R88_CANONICAL_TABLE_ID)
        if resolved_factor_root != expected_factor_root:
            raise ValueError("R88 resolved factor_data_root does not equal the canonical experiment table root")
        if not resolved_factor_root.is_dir():
            raise FileNotFoundError(f"R88 canonical experiment table root does not exist: {resolved_factor_root}")
        table_evidence = {"factor_table": dict(configured_table), "resolved_factor_table_root": str(resolved_factor_root)}
    for label, value in (
        ("results_root", results_root),
        ("score_output", score_output),
        ("state_dir", state_dir),
        ("neutralization_cache_root", neutralization_cache_root),
    ):
        if not _path_is_within(value=value, root=_R88_RESEARCH_SCRATCH_ROOT):
            raise ValueError(f"R88 {label} must be below isolated research scratch root: {_R88_RESEARCH_SCRATCH_ROOT}")
    if len(factors) != _R88_FACTOR_COUNT or len(set(factors)) != _R88_FACTOR_COUNT:
        raise ValueError(f"R88 requires exactly {_R88_FACTOR_COUNT} unique ordered factors")
    expected_min_available = _r88_min_available_factors(
        fraction=research_cfg.get("min_available_fraction"),
        factor_count=len(factors),
    )
    if not bool(missing_values.get("enabled", False)) or not bool(missing_values.get("keep_nan", False)):
        raise ValueError("R88 requires enabled missing-value handling with keep_nan=true")
    if bool(missing_values.get("add_valid_count_features", False)) or missing_values.get("valid_count_features"):
        raise ValueError("R88 requires no valid-count feature column")
    try:
        actual_min_available = int(missing_values.get("min_available_factors"))
    except (TypeError, ValueError) as exc:
        raise ValueError("R88 requires explicit missing_values.min_available_factors") from exc
    if actual_min_available != expected_min_available:
        raise ValueError(
            "R88 missing_values.min_available_factors must equal ceil(88 * "
            f"research_only.min_available_fraction)={expected_min_available}; got {actual_min_available}"
        )
    return {
        "profile": _R88_RESEARCH_CONFIG_PROFILE,
        **table_evidence,
        "min_available_fraction": float(research_cfg["min_available_fraction"]),
        "min_available_factors": expected_min_available,
    }


def _resolve_model_input_factors(
    *,
    full_factors: Sequence[str],
    configured_subset: object,
    contract_label: str = "frozen live50",
) -> list[str]:
    """Resolve a canonical ordered model-input subset of a frozen contract.

    The full contract remains responsible for all preprocessing and the
    >=27-factor admission gate.  This helper governs only the final columns
    seen by a research model, so a factor subset never changes the universe.
    """

    full = [str(value) for value in full_factors]
    if configured_subset is None:
        return list(full)
    if not isinstance(configured_subset, (list, tuple)):
        raise ValueError("model_input_factors must be an ordered list when provided")
    subset = [str(value) for value in configured_subset]
    if not subset:
        raise ValueError("model_input_factors must not be empty")
    if len(subset) != len(set(subset)):
        raise ValueError("model_input_factors must not contain duplicates")
    positions = {factor: idx for idx, factor in enumerate(full)}
    unknown = [factor for factor in subset if factor not in positions]
    if unknown:
        raise ValueError(f"model_input_factors are outside {contract_label}: {unknown}")
    indices = [positions[factor] for factor in subset]
    if indices != sorted(indices):
        raise ValueError(f"model_input_factors must preserve {contract_label} order")
    return subset


def _checkpoint_path(state_dir: Path, score_day: date) -> Path:
    return state_dir / f"{score_day:%Y-%m-%d}.pt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_checkpoint_manifest(*, state_dir: Path, model_id: str) -> tuple[dict[str, Any], str]:
    """Return the canonical byte-level inventory for an immutable resume source."""

    try:
        paths = sorted(state_dir.glob("*.pt"), key=lambda item: item.name)
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r3 resume source state: {state_dir}") from exc
    if not paths:
        raise RuntimeError(f"r3 resume source has no checkpoints: {state_dir}")
    manifest: dict[str, Any] = {
        "format": "r3_v5_source_checkpoint_manifest_v1",
        "model_id": str(model_id),
        "checkpoint_suffix": ".pt",
        "chain_start": paths[0].stem,
        "chain_end": paths[-1].stem,
        "checkpoint_count": len(paths),
        "checkpoints": [
            {
                "day": path.stem,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in paths
        ],
    }
    canonical = json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return manifest, hashlib.sha256(canonical).hexdigest()


def _parse_checkpoint_day(value: object) -> date | None:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _warm_start_chain_paths(
    state_dir: Path,
    successful_score_days: Sequence[date],
) -> tuple[Path | None, Path | None]:
    """Return the prior state and its recorded parent from the success chain.

    ``targets`` contains calendar dates before a 60-day window becomes
    scoreable.  It must never be used to infer checkpoint ancestry: the first
    successful refit may be much later than ``targets[0]``.  The only valid
    predecessor is therefore the immediately preceding *successful* score.
    """

    successful = list(successful_score_days)
    if successful != sorted(successful) or len(successful) != len(set(successful)):
        raise RuntimeError("successful r3 checkpoint chain is not strictly increasing")
    if not successful:
        return None, None
    previous = _checkpoint_path(state_dir, successful[-1])
    predecessor = _checkpoint_path(state_dir, successful[-2]) if len(successful) >= 2 else None
    return previous, predecessor


def _require_fresh_research_outputs(*, state_dir: Path, score_output: Path) -> None:
    """Reject resume/overwrite rather than mixing a broken rolling chain."""

    for label, path in (("state_dir", state_dir), ("score_output", score_output)):
        if not path.exists():
            continue
        if not path.is_dir():
            raise RuntimeError(f"torch cross-section fresh run requires directory path for {label}: {path}")
        try:
            existing = next(path.iterdir(), None)
        except OSError as exc:
            raise RuntimeError(f"cannot inspect r3 {label}: {path}") from exc
        if existing is not None:
            raise RuntimeError(
                f"torch cross-section fresh run refuses existing {label}: {path}; "
                "use a new isolated research root rather than resuming or overwriting"
            )


def _require_expected_state_inventory(*, state_dir: Path, successful_score_days: Sequence[date]) -> None:
    """Detect an injected, stale, partial, or missing state before warm start."""

    expected = {_checkpoint_path(state_dir, day) for day in successful_score_days}
    try:
        actual = set(state_dir.iterdir()) if state_dir.exists() else set()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r3 state inventory: {state_dir}") from exc
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise RuntimeError(
            "strict r3 warm-start state inventory mismatch: "
            f"missing={missing} unexpected={unexpected}"
        )


def _audit_resume_source_prefix(
    *,
    source_state_dir: Path,
    model_id: str,
    expected_fingerprint: str,
    calendar_days: Sequence[date],
    chain_start: date,
    expected_manifest_sha256: str,
    label_cutoff: date | None,
) -> tuple[list[CheckpointRecord], dict[str, Any], str]:
    """Verify a byte-pinned, uninterrupted source prefix before it is read.

    The old r3_v4 state root is never a write target.  This gate proves both
    its file inventory and its causal warm-start ancestry before a v5 research
    continuation can reconstruct a score or consume its terminal state.
    """

    source_state_dir = source_state_dir.resolve(strict=False)
    if not source_state_dir.is_dir():
        raise RuntimeError(f"r3 resume source state_dir is not a directory: {source_state_dir}")
    manifest, actual_manifest_sha256 = _source_checkpoint_manifest(state_dir=source_state_dir, model_id=model_id)
    if not expected_manifest_sha256 or actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "r3 resume source checkpoint manifest hash mismatch: "
            f"expected={expected_manifest_sha256!r} actual={actual_manifest_sha256}"
        )
    try:
        actual_names = {path.name for path in source_state_dir.iterdir()}
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r3 resume source inventory: {source_state_dir}") from exc
    expected_names = {str(item["file"]) for item in manifest["checkpoints"]}
    if actual_names != expected_names:
        raise RuntimeError(
            "r3 resume source state inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)} unexpected={sorted(actual_names - expected_names)}"
        )
    source_days = [_parse_checkpoint_day(item["day"]) for item in manifest["checkpoints"]]
    if any(day is None for day in source_days):
        raise RuntimeError("r3 resume source checkpoint name is not an ISO trade date")
    days = [day for day in source_days if day is not None]
    if not days or days[0] != chain_start:
        raise RuntimeError(f"r3 resume source prefix must start at {chain_start}, got {days[:1]}")
    positions = {day: idx for idx, day in enumerate(calendar_days)}
    if any(day not in positions for day in days):
        raise RuntimeError("r3 resume source checkpoint is outside requested natural trading calendar")
    start_pos = positions[chain_start]
    expected_days = list(calendar_days[start_pos : start_pos + len(days)])
    if days != expected_days:
        raise RuntimeError(
            "r3 resume source checkpoints are not a continuous natural-trading-day prefix: "
            f"expected_end={expected_days[-1] if expected_days else None} actual_end={days[-1]}"
        )
    records: list[CheckpointRecord] = []
    for day in days:
        record = _read_checkpoint_record(
            checkpoint_path=_checkpoint_path(source_state_dir, day),
            expected_fingerprint=expected_fingerprint,
            label_cutoff=label_cutoff,
            expected_checkpoint_predecessor=records[-1].path if records else None,
        )
        records.append(record)
    return records, manifest, actual_manifest_sha256


def _load_score_file_exact(*, score_path: Path, score_day: date) -> pd.DataFrame:
    """Read one resume score file without legacy dedupe/drop-invalid behavior."""

    if not score_path.is_file():
        raise RuntimeError(f"r3 resume score file is missing: {score_path}")
    try:
        frame = pd.read_csv(score_path)
    except Exception as exc:
        raise RuntimeError(f"cannot read r3 resume score file: {score_path}") from exc
    required = {"trade_date", "code", "score"}
    if set(frame.columns) != required:
        raise RuntimeError(f"r3 resume score file schema mismatch: {score_path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["code"] = frame["code"].astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame.empty or set(frame["trade_date"]) != {score_day}:
        raise RuntimeError(f"r3 resume score day mismatch: {score_path}")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise RuntimeError(f"r3 resume score is non-finite: {score_path}")
    if frame.duplicated(subset=["trade_date", "code"]).any() or frame["score"].nunique() <= 1:
        raise RuntimeError(f"r3 resume score audit failed: {score_path}")
    return frame.loc[:, ["trade_date", "code", "score"]]


def _validate_daily_commit_semantics(
    *,
    checkpoint_origin: object,
    score_provenance: object,
    history: object,
    train_days: object,
    val_days: object,
    source: Path,
) -> None:
    """Reject impossible origin/provenance combinations before they persist."""

    valid_pairs = {
        ("source_checkpoint_v4", "reconstructed_from_checkpoint_current_inputs"),
        ("continuation_checkpoint_v5", "trained_continuation_v5"),
        ("continuation_checkpoint_v5", "reconstructed_from_checkpoint_current_inputs"),
    }
    pair = (checkpoint_origin, score_provenance)
    if pair not in valid_pairs:
        raise RuntimeError(f"r3 daily commit origin/provenance mismatch: {source}")
    if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
        raise RuntimeError(f"r3 daily commit training history is malformed: {source}")
    for key, value in (("train_days", train_days), ("val_days", val_days)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise RuntimeError(f"r3 daily commit {key} is malformed: {source}")
    if score_provenance == "reconstructed_from_checkpoint_current_inputs":
        if history or train_days is not None or val_days is not None:
            raise RuntimeError(f"r3 reconstructed commit contains training evidence: {source}")
    elif train_days is None or val_days is None or train_days <= 0 or val_days <= 0:
        raise RuntimeError(f"r3 trained continuation commit lacks split evidence: {source}")


def _verify_daily_commit(
    *,
    commit_path: Path,
    score_day: date,
    score_path: Path,
    checkpoint_path: Path,
    contract_fingerprint: str,
) -> dict[str, Any]:
    try:
        payload = json.loads(commit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read r3 daily commit: {commit_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "r3_v5_daily_commit_v1":
        raise RuntimeError(f"r3 daily commit format mismatch: {commit_path}")
    # ``json.loads`` accepts the non-standard NaN/Infinity constants.  Commits
    # are intended to be portable, canonical JSON records, so reject those
    # values even if a hand-written or damaged file happens to parse.
    try:
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"r3 daily commit contains non-JSON value: {commit_path}") from exc
    if payload.get("score_day") != str(score_day) or payload.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError(f"r3 daily commit contract mismatch: {commit_path}")
    if payload.get("score_path") != str(score_path) or payload.get("checkpoint_path") != str(checkpoint_path):
        raise RuntimeError(f"r3 daily commit artifact path mismatch: {commit_path}")
    frame = _load_score_file_exact(score_path=score_path, score_day=score_day)
    if payload.get("score_file_sha256") != _sha256_file(score_path):
        raise RuntimeError(f"r3 daily commit score file hash mismatch: {commit_path}")
    if payload.get("score_frame_sha256") != _score_frame_digest(frame):
        raise RuntimeError(f"r3 daily commit score frame hash mismatch: {commit_path}")
    if payload.get("checkpoint_sha256") != _sha256_file(checkpoint_path):
        raise RuntimeError(f"r3 daily commit checkpoint hash mismatch: {commit_path}")
    codes = frame["code"].astype(str).sort_values(kind="stable")
    if payload.get("score_rows") != int(len(frame)):
        raise RuntimeError(f"r3 daily commit score row count mismatch: {commit_path}")
    if payload.get("score_code_sha256") != hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest():
        raise RuntimeError(f"r3 daily commit score code hash mismatch: {commit_path}")
    if "warm_start_from" not in payload or not isinstance(payload.get("training_history"), list):
        raise RuntimeError(f"r3 daily commit provenance fields are missing: {commit_path}")
    _validate_daily_commit_semantics(
        checkpoint_origin=payload.get("checkpoint_origin"),
        score_provenance=payload.get("score_provenance"),
        history=payload["training_history"],
        train_days=payload.get("train_days"),
        val_days=payload.get("val_days"),
        source=commit_path,
    )
    for key in ("max_train_label_day", "max_validation_label_day"):
        parsed = _parse_checkpoint_day(payload.get(key))
        if parsed is None or parsed >= score_day:
            raise RuntimeError(f"r3 daily commit label boundary mismatch: {commit_path}")
    return payload


def _verify_commit_checkpoint_link(
    *,
    commit_payload: dict[str, Any],
    record: CheckpointRecord,
    expected_checkpoint_origin: str,
    allowed_score_provenance: set[str],
) -> None:
    """Bind a verified commit to the exact checkpoint ancestry used for it."""

    if commit_payload.get("checkpoint_origin") != expected_checkpoint_origin:
        raise RuntimeError(
            "r3 daily commit checkpoint origin does not match its verified state: "
            f"{record.path}"
        )
    if commit_payload.get("score_provenance") not in allowed_score_provenance:
        raise RuntimeError(
            "r3 daily commit score provenance does not match its verified state: "
            f"{record.path}"
        )
    if (
        _parse_checkpoint_day(commit_payload.get("max_train_label_day")) != record.max_train_label_day
        or _parse_checkpoint_day(commit_payload.get("max_validation_label_day")) != record.max_validation_label_day
    ):
        raise RuntimeError(f"r3 daily commit label provenance mismatch: {record.path}")
    observed_parent = commit_payload.get("warm_start_from")
    if record.previous_checkpoint is None:
        if observed_parent not in (None, ""):
            raise RuntimeError(f"r3 daily commit unexpectedly has a warm-start parent: {record.path}")
        return
    if not isinstance(observed_parent, str) or not observed_parent:
        raise RuntimeError(f"r3 daily commit is missing its warm-start parent: {record.path}")
    try:
        observed_path = Path(observed_parent).resolve(strict=False)
    except OSError as exc:
        raise RuntimeError(f"r3 daily commit warm-start parent is unreadable: {record.path}") from exc
    if observed_path != record.previous_checkpoint.resolve(strict=False):
        raise RuntimeError(f"r3 daily commit warm-start parent mismatch: {record.path}")


def _append_committed_resume_rows(
    *,
    commit_payload: dict[str, Any],
    record: CheckpointRecord,
    frame: pd.DataFrame,
    allowlist_rows: int,
    history_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
) -> None:
    """Restore summary rows from a previously verified daily commit.

    A daily commit is the durable boundary.  On re-entry, retain committed
    scores and training history rather than creating a summary that describes
    only the days trained after the latest process restart.
    """

    warm_start_from = record.previous_checkpoint.name if record.previous_checkpoint else None
    for row in commit_payload["training_history"]:
        history_rows.append({**row, "score_day": record.score_day, "warm_start_from": warm_start_from})
    rolling_rows.append(
        {
            "score_day": record.score_day,
            # Source v4 checkpoints do not contain the original split counts.
            # New v5 commits do, and old v5 commits remain explicitly null.
            "train_days": commit_payload.get("train_days"),
            "val_days": commit_payload.get("val_days"),
            "max_train_label_day": record.max_train_label_day,
            "max_validation_label_day": record.max_validation_label_day,
            "warm_start_from": warm_start_from,
            "score_rows": len(frame),
            "allowlist_rows": int(allowlist_rows),
            "contract_fingerprint": commit_payload["contract_fingerprint"],
            "checkpoint_origin": commit_payload["checkpoint_origin"],
            "score_provenance": commit_payload["score_provenance"],
        }
    )


def _durably_commit_score_day(
    *,
    score_output: Path,
    commit_root: Path,
    score_day: date,
    score_frame: pd.DataFrame,
    checkpoint_path: Path,
    checkpoint_origin: str,
    warm_start_from: Path | None,
    max_train_label_day: date,
    max_validation_label_day: date,
    contract_fingerprint: str,
    score_provenance: str,
    history: list[dict[str, Any]],
    train_days: int | None = None,
    val_days: int | None = None,
) -> dict[str, Any]:
    """Make one day recoverable across checkpoint -> score -> commit crashes."""

    score_path = score_output / f"{score_day:%Y-%m}" / f"{score_day:%Y-%m-%d}.csv"
    commit_path = _commit_path(commit_root, score_day)
    _validate_daily_commit_semantics(
        checkpoint_origin=checkpoint_origin,
        score_provenance=score_provenance,
        history=history,
        train_days=train_days,
        val_days=val_days,
        source=commit_path,
    )
    if commit_path.exists():
        payload = _verify_daily_commit(
            commit_path=commit_path,
            score_day=score_day,
            score_path=score_path,
            checkpoint_path=checkpoint_path,
            contract_fingerprint=contract_fingerprint,
        )
        existing = _load_score_file_exact(score_path=score_path, score_day=score_day)
        if _score_frame_digest(existing) != _score_frame_digest(score_frame):
            raise RuntimeError(f"r3 existing committed score differs from reconstructed data: {score_path}")
        return payload
    if score_path.exists():
        existing = _load_score_file_exact(score_path=score_path, score_day=score_day)
        if _score_frame_digest(existing) != _score_frame_digest(score_frame):
            raise RuntimeError(f"r3 resume score exists but differs from reconstructed data: {score_path}")
    else:
        score_path = write_scores_for_day_atomic(score_output, score_frame, score_day=score_day)
    _write_daily_commit_atomic(
        commit_root=commit_root,
        score_day=score_day,
        score_path=score_path,
        score_frame=score_frame,
        checkpoint_path=checkpoint_path,
        checkpoint_origin=checkpoint_origin,
        warm_start_from=warm_start_from,
        max_train_label_day=max_train_label_day,
        max_validation_label_day=max_validation_label_day,
        contract_fingerprint=contract_fingerprint,
        score_provenance=score_provenance,
        history=history,
        train_days=train_days,
        val_days=val_days,
    )
    return _verify_daily_commit(
        commit_path=commit_path,
        score_day=score_day,
        score_path=score_path,
        checkpoint_path=checkpoint_path,
        contract_fingerprint=contract_fingerprint,
    )


def _model_from_checkpoint(
    *,
    architecture: str,
    n_features: int,
    model_params: dict,
    model_state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    model = build_cross_section_model(architecture, n_features=n_features, model_params=model_params).to(torch.device("cpu"))
    model.load_state_dict(model_state, strict=True)
    return model


def _reconstruct_checkpoint_score_frame(
    *,
    record: CheckpointRecord,
    architecture: str,
    model_params: dict,
    store: FactorStore,
    label_root: Path,
    factors: list[str],
    model_input_factors: Sequence[str] | None = None,
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    factor_time: str,
    label_time: str,
    tradable_code_map: dict[date, set[str]],
    neutralizer: Any,
    missing_values: dict,
    input_missingness: dict,
) -> tuple[pd.DataFrame, int]:
    """Score a saved final state without training or reading its target label.

    This is the only legal repair path for a source checkpoint or a target
    continuation checkpoint that survived an interruption before its daily
    score/commit was made durable.
    """

    test_data = _build_score_day_split(
        store=store,
        label_root=label_root,
        score_day=record.score_day,
        factors=factors,
        model_input_factors=model_input_factors,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        factor_time=factor_time,
        label_time=label_time,
        tradable_code_map=tradable_code_map,
        neutralizer=neutralizer,
        missing_values=missing_values,
        input_missingness=input_missingness,
    )
    restored_model = _model_from_checkpoint(
        architecture=architecture,
        n_features=test_data.x.shape[1],
        model_params=model_params,
        model_state=record.model_state,
    )
    return (
        _score_frame(
            score_day=record.score_day,
            split=test_data,
            prediction=_predict_split(restored_model, test_data, device=torch.device("cpu")),
        ),
        len(tradable_code_map.get(record.score_day, set())),
    )


def _build_score_day_split(
    *,
    store: FactorStore,
    label_root: Path,
    score_day: date,
    factors: list[str],
    model_input_factors: Sequence[str] | None = None,
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    factor_time: str,
    label_time: str,
    tradable_code_map: dict[date, set[str]],
    neutralizer,
    missing_values: dict,
    input_missingness: dict,
) -> CrossSectionSplit:
    test_data = _build_data(
        store=store,
        label_root=label_root,
        days=[score_day],
        factors=factors,
        model_input_factors=model_input_factors,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=True,
        factor_time=factor_time,
        label_time=label_time,
        require_label=False,
        tradable_code_map=tradable_code_map,
        neutralizer=neutralizer,
        missing_values=missing_values,
        input_missingness=input_missingness,
    )
    if not test_data.groups:
        raise RuntimeError(f"score-day allowlist/factor intersection is empty: {score_day}")
    score_day_groups = _observed_group_days(test_data)
    if score_day_groups != {score_day}:
        raise RuntimeError(
            f"score-day factor/group date mismatch for {score_day}: "
            f"observed={[str(day) for day in sorted(score_day_groups)]}"
        )
    allowed_codes = tradable_code_map.get(score_day)
    if not allowed_codes or not set(test_data.code.astype(str)).issubset(allowed_codes):
        raise RuntimeError(f"score-day codes escaped strict T-1 o_0005 allowlist: {score_day}")
    return test_data


def _resume_config(cfg: dict) -> dict[str, Any]:
    raw = cfg.get("resume", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("torch cross-section resume config must be an object")
    return dict(raw)


def _require_fresh_resume_outputs(*, state_dir: Path, score_output: Path, commit_root: Path, manifest_path: Path) -> None:
    """A new continuation starts with no target artifacts whatsoever."""

    for label, path in (
        ("state_dir", state_dir),
        ("score_output", score_output),
        ("commit_root", commit_root),
        ("resume_manifest", manifest_path),
    ):
        if not path.exists():
            continue
        if path.is_file() and label == "resume_manifest":
            raise RuntimeError(f"r3 fresh resume refuses existing {label}: {path}")
        if not path.is_dir():
            raise RuntimeError(f"r3 fresh resume requires directory path for {label}: {path}")
        try:
            existing = next(path.iterdir(), None)
        except OSError as exc:
            raise RuntimeError(f"cannot inspect r3 fresh resume {label}: {path}") from exc
        if existing is not None:
            raise RuntimeError(f"r3 fresh resume refuses existing {label}: {path}")


def _resume_target_paths(*, cfg: dict, results_root: Path, model_name: str) -> tuple[Path, Path, Path, Path]:
    resume = _resume_config(cfg)
    target_root_raw = resume.get("target_root")
    if not bool(resume.get("enabled", False)) or not target_root_raw:
        raise ValueError("torch cross-section verified resume requires resume.enabled=true and resume.target_root")
    target_root = Path(str(target_root_raw)).expanduser()
    resolved_root = target_root.resolve(strict=False)
    if any(str(resolved_root).lower().startswith(str(root).lower()) for root in _PROTECTED_ROOTS):
        raise ValueError(f"torch cross-section resume target must not use production root: {resolved_root}")
    # Outputs deliberately derive only from the explicit continuation root, so
    # a source v4 config cannot accidentally be reused as a write destination.
    return (
        target_root / "model_state" / model_name,
        target_root / "scores" / model_name,
        target_root / "daily_commits" / model_name,
        target_root / "manifests" / f"{model_name}.resume.json",
    )


def _existing_resume_commit_days(*, commit_root: Path) -> list[date]:
    if not commit_root.exists():
        return []
    if not commit_root.is_dir():
        raise RuntimeError(f"r3 resume commit root is not a directory: {commit_root}")
    paths = sorted(commit_root.rglob("*.json"))
    all_files = {path for path in commit_root.rglob("*") if path.is_file()}
    if all_files != set(paths):
        raise RuntimeError(f"r3 resume commit root contains non-JSON artifact: {commit_root}")
    days: list[date] = []
    for path in paths:
        parsed = _parse_checkpoint_day(path.stem)
        relative = path.relative_to(commit_root)
        if (
            parsed is None
            or len(relative.parts) != 2
            or path.name != f"{parsed:%Y-%m-%d}.json"
            or path.parent.name != f"{parsed:%Y-%m}"
        ):
            raise RuntimeError(f"r3 resume commit has non-date name: {path}")
        days.append(parsed)
    if days != sorted(days) or len(days) != len(set(days)):
        raise RuntimeError(f"r3 resume commit days are not unique/increasing: {commit_root}")
    return days


def _resume_target_checkpoint_files(*, state_dir: Path) -> dict[date, Path]:
    """Return only the canonical continuation checkpoints, rejecting debris.

    A crash may leave a target checkpoint after its atomic save but before its
    corresponding score/commit.  It is recoverable only if the inventory is
    otherwise exact; temporary files, nested files, or alternate names make
    the target ambiguous and must stop the resume rather than be ignored.
    """

    if not state_dir.exists():
        return {}
    if not state_dir.is_dir():
        raise RuntimeError(f"r3 resume target state_dir is not a directory: {state_dir}")
    try:
        entries = sorted(state_dir.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r3 resume target state_dir: {state_dir}") from exc
    files: dict[date, Path] = {}
    for path in entries:
        parsed = _parse_checkpoint_day(path.stem)
        if (
            not path.is_file()
            or path.suffix.lower() != ".pt"
            or parsed is None
            or path.name != f"{parsed:%Y-%m-%d}.pt"
        ):
            raise RuntimeError(f"r3 resume target state contains unexpected artifact: {path}")
        if parsed in files:
            raise RuntimeError(f"r3 resume target state has duplicate day: {parsed}")
        files[parsed] = path
    return files


def _resume_target_score_files(*, score_output: Path) -> dict[date, Path]:
    """Return canonical daily score files and reject any partial write debris."""

    if not score_output.exists():
        return {}
    if not score_output.is_dir():
        raise RuntimeError(f"r3 resume score output is not a directory: {score_output}")
    try:
        paths = sorted((path for path in score_output.rglob("*") if path.is_file()), key=lambda item: str(item))
    except OSError as exc:
        raise RuntimeError(f"cannot inspect r3 resume score output: {score_output}") from exc
    files: dict[date, Path] = {}
    for path in paths:
        relative = path.relative_to(score_output)
        parsed = _parse_checkpoint_day(path.stem)
        if (
            path.suffix.lower() != ".csv"
            or parsed is None
            or len(relative.parts) != 2
            or path.name != f"{parsed:%Y-%m-%d}.csv"
            or path.parent.name != f"{parsed:%Y-%m}"
        ):
            raise RuntimeError(f"r3 resume score output contains unexpected artifact: {path}")
        if parsed in files:
            raise RuntimeError(f"r3 resume score output has duplicate day: {parsed}")
        files[parsed] = path
    return files


def _find_resume_orphan_checkpoint(
    *,
    state_dir: Path,
    score_output: Path,
    committed_continuation_days: Sequence[date],
    committed_score_days: Sequence[date],
    source_terminal: Path,
    source_terminal_day: date,
    calendar_days: Sequence[date],
    expected_fingerprint: str,
    label_cutoff: date | None,
) -> CheckpointRecord | None:
    """Find one legally recoverable checkpoint/score-without-commit day.

    The durable order is checkpoint -> score CSV -> commit JSON.  A process
    interrupted in either of the first two gaps may leave a *single* next-day
    target checkpoint, optionally with the score file already present.  The
    checkpoint is validated against its immediate predecessor and returned to
    the caller for score-only reconstruction; this helper never trains.  Any
    other shape is ambiguous and therefore rejected fail-closed.
    """

    continuation = list(committed_continuation_days)
    score_days = list(committed_score_days)
    if continuation != sorted(continuation) or len(continuation) != len(set(continuation)):
        raise RuntimeError("r3 resume continuation commit days are not strictly increasing")
    if score_days != sorted(score_days) or len(score_days) != len(set(score_days)):
        raise RuntimeError("r3 resume committed score days are not strictly increasing")

    state_files = _resume_target_checkpoint_files(state_dir=state_dir)
    score_files = _resume_target_score_files(score_output=score_output)
    expected_state = set(continuation)
    expected_score = set(score_days)
    actual_state = set(state_files)
    actual_score = set(score_files)
    missing_state = sorted(expected_state - actual_state)
    missing_score = sorted(expected_score - actual_score)
    if missing_state or missing_score:
        raise RuntimeError(
            "r3 resume committed artifact is missing: "
            f"state={[str(day) for day in missing_state]} score={[str(day) for day in missing_score]}"
        )

    extra_state = actual_state - expected_state
    extra_score = actual_score - expected_score
    if not extra_state and not extra_score:
        return None

    positions = {day: index for index, day in enumerate(calendar_days)}
    parent_day = continuation[-1] if continuation else source_terminal_day
    if parent_day not in positions or positions[parent_day] + 1 >= len(calendar_days):
        raise RuntimeError("r3 resume has artifact after the requested natural trading calendar")
    orphan_day = calendar_days[positions[parent_day] + 1]
    if extra_state != {orphan_day}:
        raise RuntimeError(
            "r3 resume orphan checkpoint is not the single next natural trading day: "
            f"expected={orphan_day} actual={[str(day) for day in sorted(extra_state)]}"
        )
    if extra_score - {orphan_day}:
        raise RuntimeError(
            "r3 resume orphan score is not the single next natural trading day: "
            f"expected={orphan_day} actual={[str(day) for day in sorted(extra_score)]}"
        )
    if extra_score and extra_score != {orphan_day}:
        raise RuntimeError(f"r3 resume orphan score inventory is invalid: {score_output}")

    predecessor = _checkpoint_path(state_dir, continuation[-1]) if continuation else source_terminal
    return _read_checkpoint_record(
        checkpoint_path=state_files[orphan_day],
        expected_fingerprint=expected_fingerprint,
        label_cutoff=label_cutoff,
        expected_checkpoint_predecessor=predecessor,
    )


def _require_existing_resume_contract(
    *,
    manifest_path: Path,
    model_name: str,
    contract_fingerprint: str,
    source_manifest_sha256: str,
    expected_resume_payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot read r3 resume manifest: {manifest_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "r3_v5_verified_resume_v1":
        raise RuntimeError(f"r3 resume manifest format mismatch: {manifest_path}")
    if payload.get("model_id") != model_name or payload.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError(f"r3 resume manifest contract mismatch: {manifest_path}")
    if payload.get("source_checkpoint_manifest_sha256") != source_manifest_sha256:
        raise RuntimeError(f"r3 resume manifest source fingerprint mismatch: {manifest_path}")
    # The target root is a single, immutable continuation identity.  Accepting
    # the same source under a different requested interval or source location
    # would silently mix a second experiment into already durable artifacts.
    for key in (
        "requested_start",
        "requested_end",
        "source_state_dir",
        "source_terminal_checkpoint",
        "score_reconstruction_provenance",
    ):
        if payload.get(key) != expected_resume_payload.get(key):
            raise RuntimeError(f"r3 resume manifest {key} mismatch: {manifest_path}")
    return payload


def _load_warm_start_state(
    *,
    checkpoint_path: Path,
    expected_fingerprint: str,
    score_day: date,
    label_cutoff: date | None,
    device: torch.device,
    expected_checkpoint_predecessor: Path | None,
) -> dict[str, torch.Tensor] | None:
    checkpoint_day = _parse_checkpoint_day(checkpoint_path.stem)
    if checkpoint_day is None or checkpoint_day >= score_day:
        return None
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
        return None
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return None
    if _parse_checkpoint_day(payload.get("score_day")) != checkpoint_day:
        return None
    if _parse_checkpoint_day(metadata.get("score_day")) != checkpoint_day:
        return None
    recorded_predecessor = metadata.get("previous_checkpoint")
    if expected_checkpoint_predecessor is None:
        if recorded_predecessor not in (None, ""):
            return None
    else:
        if not recorded_predecessor:
            return None
        try:
            if Path(str(recorded_predecessor)).resolve(strict=False) != expected_checkpoint_predecessor.resolve(strict=False):
                return None
        except OSError:
            return None
    for name in ("max_train_label_day", "max_validation_label_day"):
        used_day = _parse_checkpoint_day(metadata.get(name))
        if used_day is None or used_day >= checkpoint_day or used_day >= score_day:
            return None
        if label_cutoff is not None and used_day > label_cutoff:
            return None
    state = payload.get("model_state")
    return state if isinstance(state, dict) else None


def _read_checkpoint_record(
    *,
    checkpoint_path: Path,
    expected_fingerprint: str,
    label_cutoff: date | None,
    expected_checkpoint_predecessor: Path | None,
) -> CheckpointRecord:
    """Load a checkpoint only after applying the same causal provenance gates.

    Unlike ``_load_warm_start_state``, resume needs the metadata for its
    durable commit record.  This helper deliberately fails closed instead of
    returning an ambiguous partial payload.
    """

    checkpoint_day = _parse_checkpoint_day(checkpoint_path.stem)
    if checkpoint_day is None:
        raise RuntimeError(f"invalid r3 checkpoint day for resume: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"cannot load r3 resume checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
        raise RuntimeError(f"r3 resume checkpoint fingerprint mismatch: {checkpoint_path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"r3 resume checkpoint metadata is missing: {checkpoint_path}")
    if _parse_checkpoint_day(payload.get("score_day")) != checkpoint_day:
        raise RuntimeError(f"r3 resume checkpoint payload score_day mismatch: {checkpoint_path}")
    if _parse_checkpoint_day(metadata.get("score_day")) != checkpoint_day:
        raise RuntimeError(f"r3 resume checkpoint metadata score_day mismatch: {checkpoint_path}")
    raw_parent = metadata.get("previous_checkpoint")
    if expected_checkpoint_predecessor is None:
        if raw_parent not in (None, ""):
            raise RuntimeError(f"r3 resume checkpoint unexpectedly has a parent: {checkpoint_path}")
        parent = None
    else:
        if not raw_parent:
            raise RuntimeError(f"r3 resume checkpoint parent is missing: {checkpoint_path}")
        try:
            parent = Path(str(raw_parent)).resolve(strict=False)
        except OSError as exc:
            raise RuntimeError(f"r3 resume checkpoint parent is unreadable: {checkpoint_path}") from exc
        if parent != expected_checkpoint_predecessor.resolve(strict=False):
            raise RuntimeError(f"r3 resume checkpoint parent mismatch: {checkpoint_path}")
    train_day = _parse_checkpoint_day(metadata.get("max_train_label_day"))
    validation_day = _parse_checkpoint_day(metadata.get("max_validation_label_day"))
    for name, used_day in (("max_train_label_day", train_day), ("max_validation_label_day", validation_day)):
        if used_day is None or used_day >= checkpoint_day:
            raise RuntimeError(f"r3 resume checkpoint has invalid {name}: {checkpoint_path}")
        if label_cutoff is not None and used_day > label_cutoff:
            raise RuntimeError(f"r3 resume checkpoint exceeds label_cutoff: {checkpoint_path}")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError(f"r3 resume checkpoint model_state is missing: {checkpoint_path}")
    return CheckpointRecord(
        path=checkpoint_path,
        score_day=checkpoint_day,
        model_state=state,
        max_train_label_day=train_day,
        max_validation_label_day=validation_day,
        previous_checkpoint=parent,
    )


def _save_warm_start_state(
    *,
    checkpoint_path: Path,
    model: torch.nn.Module,
    fingerprint: str,
    score_day: date,
    max_train_label_day: date,
    max_validation_label_day: date,
    previous_checkpoint: Path | None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.stem}.{random.randrange(1 << 30):08x}.tmp")
    torch.save(
        {
            "format_version": 1,
            "score_day": str(score_day),
            "fingerprint": fingerprint,
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": {
                "score_day": str(score_day),
                "max_train_label_day": str(max_train_label_day),
                "max_validation_label_day": str(max_validation_label_day),
                "previous_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
            },
        },
        tmp_path,
    )
    tmp_path.replace(checkpoint_path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a small r3 research provenance record with replace semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{random.randrange(1 << 30):08x}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _score_frame(*, score_day: date, split: CrossSectionSplit, prediction: np.ndarray) -> pd.DataFrame:
    if len(prediction) != len(split.code):
        raise RuntimeError(f"score prediction/code length mismatch: {score_day}")
    if not np.isfinite(prediction).all() or np.unique(prediction).size <= 1:
        raise RuntimeError(f"score audit failed (non-finite or constant): {score_day}")
    frame = pd.DataFrame({"trade_date": score_day, "code": split.code.astype(str), "score": prediction})
    if frame.duplicated(subset=["trade_date", "code"]).any():
        raise RuntimeError(f"score audit failed (duplicate code): {score_day}")
    return frame


def _score_frame_digest(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, ["trade_date", "code", "score"]].copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"]).dt.strftime("%Y-%m-%d")
    ordered["code"] = ordered["code"].astype(str)
    ordered["score"] = ordered["score"].astype(np.float32)
    ordered = ordered.sort_values(["trade_date", "code"], kind="stable")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _commit_path(commit_root: Path, score_day: date) -> Path:
    return commit_root / f"{score_day:%Y-%m}" / f"{score_day:%Y-%m-%d}.json"


def _write_daily_commit_atomic(
    *,
    commit_root: Path,
    score_day: date,
    score_path: Path,
    score_frame: pd.DataFrame,
    checkpoint_path: Path,
    checkpoint_origin: str,
    warm_start_from: Path | None,
    max_train_label_day: date,
    max_validation_label_day: date,
    contract_fingerprint: str,
    score_provenance: str,
    history: list[dict[str, Any]],
    train_days: int | None = None,
    val_days: int | None = None,
) -> Path:
    commit_path = _commit_path(commit_root, score_day)
    if commit_path.exists():
        raise FileExistsError(f"r3 daily commit refuses overwrite: {commit_path}")
    if not score_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(f"r3 daily commit missing durable artifact: {score_day}")
    _validate_daily_commit_semantics(
        checkpoint_origin=checkpoint_origin,
        score_provenance=score_provenance,
        history=history,
        train_days=train_days,
        val_days=val_days,
        source=commit_path,
    )
    codes = score_frame["code"].astype(str).sort_values(kind="stable")
    payload: dict[str, Any] = {
        "format": "r3_v5_daily_commit_v1",
        "score_day": str(score_day),
        "score_path": str(score_path),
        "score_file_sha256": _sha256_file(score_path),
        "score_frame_sha256": _score_frame_digest(score_frame),
        "score_rows": int(len(score_frame)),
        "score_code_sha256": hashlib.sha256("\n".join(codes).encode("utf-8")).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_origin": str(checkpoint_origin),
        "warm_start_from": str(warm_start_from) if warm_start_from else None,
        "max_train_label_day": str(max_train_label_day),
        "max_validation_label_day": str(max_validation_label_day),
        "contract_fingerprint": str(contract_fingerprint),
        "score_provenance": str(score_provenance),
        "training_history": list(history),
        "train_days": int(train_days) if train_days is not None else None,
        "val_days": int(val_days) if val_days is not None else None,
    }
    _atomic_write_json(commit_path, payload)
    return commit_path


def _split_rolling_train_validation(days: Sequence[date], train_ratio: float) -> tuple[list[date], list[date]]:
    pool = list(days)
    if len(pool) < 2:
        return pool, []
    n_train = max(1, min(len(pool) - 1, int(len(pool) * float(train_ratio))))
    return pool[:n_train], pool[n_train:]


def _to_cross_section_split(data: SplitData) -> CrossSectionSplit:
    x = data.x.to_numpy(dtype=np.float32, copy=True)
    y = pd.to_numeric(data.y, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    dt = pd.to_datetime(data.dt, errors="coerce").to_numpy()
    code = data.code.astype(str).to_numpy(dtype=object)
    if len(x) != len(y) or len(x) != len(dt) or len(x) != len(code):
        raise ValueError("cross-section split arrays have mismatched lengths")
    groups = _build_day_group_indices(pd.Series(dt))
    return CrossSectionSplit(x=x, y=y, dt=dt, code=code, groups=groups)


def _model_input(x: np.ndarray, *, fill_value: float, add_missing_mask: bool) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(values)
    filled = np.where(finite, values, float(fill_value)).astype(np.float32, copy=False)
    if add_missing_mask:
        return np.concatenate([filled, (~finite).astype(np.float32)], axis=1)
    return filled


def _listnet_loss(prediction: torch.Tensor, target: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if prediction.ndim != 1 or target.ndim != 1 or len(prediction) != len(target):
        raise ValueError("ListNet expects matching one-dimensional daily tensors")
    if len(prediction) < 2:
        raise ValueError("ListNet requires at least two bonds in a daily batch")
    # Labels are raw intraday returns (usually around 1e-2).  Feeding those
    # levels directly into ListNet would make the target softmax nearly
    # uniform, effectively erasing the day-local ranking signal.  This is a
    # target representation for the loss only: every completed day's labels
    # are centred/scaled using that completed day alone; no score-day label is
    # read and the label definition itself remains unchanged.
    centered = target - target.mean()
    std = centered.std(unbiased=False).clamp_min(1e-6)
    target_z = torch.clamp(centered / std, -6.0, 6.0)
    scale = max(1e-3, float(temperature))
    target_distribution = torch.softmax(target_z / scale, dim=0)
    return -(target_distribution * F.log_softmax(prediction / scale, dim=0)).sum()


def _mean_daily_listnet_loss(
    *,
    model: torch.nn.Module,
    split: CrossSectionSplit,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    if not split.groups:
        raise ValueError("cross-sectional loss requires at least one full daily group")
    losses: list[torch.Tensor] = []
    for indices in split.groups:
        x = torch.from_numpy(split.x[indices]).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(split.y[indices]).to(device=device, dtype=torch.float32)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("model input and labels must be finite after admission/fill")
        pred = model(x, torch.ones(len(indices), dtype=torch.bool, device=device))
        losses.append(_listnet_loss(pred, y, temperature=temperature))
    return torch.stack(losses).mean()


def _predict_split(model: torch.nn.Module, split: CrossSectionSplit, *, device: torch.device) -> np.ndarray:
    if len(split.x) == 0:
        return np.empty(0, dtype=np.float32)
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for indices in split.groups:
            x = torch.from_numpy(split.x[indices]).to(device=device, dtype=torch.float32)
            out = model(x, torch.ones(len(indices), dtype=torch.bool, device=device))
            pieces.append(out.detach().cpu().numpy().astype(np.float32, copy=False))
    predicted = np.empty(len(split.x), dtype=np.float32)
    for indices, piece in zip(split.groups, pieces, strict=True):
        predicted[indices] = piece
    return predicted


def _daily_metrics(split: CrossSectionSplit, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(prediction)),
        "ic": float(_mean_pearson_ic_by_groups(split.y, prediction, split.groups)),
        "rank_ic": float(_mean_rank_ic_by_groups(split.y, prediction, split.groups)),
    }


def _train_one_model(
    *,
    architecture: str,
    n_features: int,
    model_params: dict,
    train_data: CrossSectionSplit,
    val_data: CrossSectionSplit,
    train_cfg: dict,
    temperature: float,
    initial_state: dict[str, torch.Tensor] | None,
    seed: int,
) -> tuple[torch.nn.Module, list[dict]]:
    device = torch.device("cuda" if str(train_cfg.get("device", "cpu")).lower() in {"cuda", "gpu"} and torch.cuda.is_available() else "cpu")
    _set_deterministic_seed(seed=seed, deterministic=bool(train_cfg.get("deterministic", True)))
    model = build_cross_section_model(architecture, n_features=n_features, model_params=model_params).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = max(1, int(train_cfg.get("epochs", train_cfg.get("num_epochs", 20))))
    patience = max(1, int(train_cfg.get("early_stopping_patience", 4)))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("-inf")
    stale = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = _mean_daily_listnet_loss(model=model, split=train_data, device=device, temperature=temperature)
        if not torch.isfinite(train_loss):
            raise RuntimeError("non-finite cross-sectional ListNet loss")
        train_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        val_pred = _predict_split(model, val_data, device=device) if val_data.groups else np.empty(0, dtype=np.float32)
        val_metrics = _daily_metrics(val_data, val_pred) if val_data.groups else {"ic": float("nan"), "rank_ic": float("nan"), "count": 0}
        score = float(val_metrics["rank_ic"])
        history.append({"epoch": epoch, "train_loss": float(train_loss.detach().cpu()), "val_rank_ic": score, "val_ic": float(val_metrics["ic"])})
        if np.isfinite(score) and score > best_score + min_delta:
            best_score = score
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _factor_day_exists(store: FactorStore | CanonicalFactorTableReader, day: date) -> bool:
    try:
        return not store.read_day(day).empty
    except Exception:
        return False


def _build_data(
    *,
    store: FactorStore | CanonicalFactorTableReader,
    label_root: Path,
    days: Sequence[date],
    factors: list[str],
    model_input_factors: Sequence[str] | None = None,
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    factor_time: str,
    label_time: str,
    require_label: bool,
    tradable_code_map: dict[date, set[str]],
    neutralizer,
    missing_values: dict,
    input_missingness: dict,
) -> CrossSectionSplit:
    input_factors = list(model_input_factors) if model_input_factors is not None else list(factors)
    # ``build_dataset`` owns the admission and preprocessing contract.  It must
    # see every frozen contract column: the availability gate is measured over
    # the full contract, and z-score / T-1 Ridge neutralization are
    # intentionally applied before a research-only model slice is selected.
    # Passing only ``input_factors`` here would leave the other contract
    # columns absent while asking the preprocessor to transform them.
    raw = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=list(days),
        factor_cols=list(factors),
        raw_factor_cols=list(factors),
        preprocess_factor_cols=list(factors),
        min_count=int(min_count),
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=bool(zscore),
        factor_time=factor_time,
        label_time=label_time,
        require_label=require_label,
        tradable_code_map=tradable_code_map,
        tradable_strict=True,
        neutralizer=neutralizer,
        missing_values=missing_values,
        read_label_when_not_required=False,
        apply_tradable_filter_when_label_not_required=True,
    )
    missing_input_columns = [factor for factor in input_factors if factor not in raw.x.columns]
    if missing_input_columns:
        raise RuntimeError(
            "full50 preprocessing did not return configured model input columns: "
            f"{missing_input_columns}"
        )
    # Select only after all shared contract transformations have completed. The
    # scalar model sees the requested subset, while admission remains full.
    raw = SplitData(
        x=raw.x.loc[:, input_factors].copy(),
        y=raw.y,
        dt=raw.dt,
        code=raw.code,
        sample_weight=raw.sample_weight,
    )
    split = _to_cross_section_split(raw)
    if not require_label and len(split.x) == 0:
        return split
    transformed = _model_input(
        split.x,
        fill_value=float(input_missingness.get("fill_value", 0.0)),
        add_missing_mask=bool(input_missingness.get("add_missing_mask", True)),
    )
    return CrossSectionSplit(x=transformed, y=split.y, dt=split.dt, code=split.code, groups=split.groups)


def _observed_group_days(split: CrossSectionSplit) -> set[date]:
    parsed = pd.to_datetime(pd.Series(split.dt), errors="coerce")
    return {value.date() for value in parsed.dropna()}


def _missing_group_days(split: CrossSectionSplit, expected_days: Sequence[date]) -> set[date]:
    return set(expected_days) - _observed_group_days(split)


def _strict_history_window_error(
    historical_days: Sequence[date],
    *,
    expected_history_days: int,
    label_cutoff: date | None,
) -> str | None:
    """Reject truncated label history instead of silently training a bootstrap.

    The rolling contract is defined on the natural trading calendar.  A
    score-only smoke may cap readable labels, but that cap must still include
    every one of the prior natural history days.  Filtering the window down to
    the label cutoff would otherwise turn a 60-day test into an undocumented
    short-history experiment.
    """

    days = list(historical_days)
    if len(days) != int(expected_history_days):
        return (
            "natural history window has unexpected size: "
            f"expected={expected_history_days} actual={len(days)}"
        )
    if not days:
        return "natural history window is empty"
    if label_cutoff is not None and max(days) > label_cutoff:
        return (
            "label_cutoff does not cover the complete natural history window: "
            f"cutoff={label_cutoff} history_end={max(days)}"
        )
    return None


def main(
    *,
    config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    execution: dict | None = None,
    resume_interrupted: bool = False,
) -> None:
    paths_cfg = load_config_file("paths")
    cfg = _load_model_config(Path(config_path) if config_path else None)
    execution_cfg = dict(execution or {})
    model_name = str(cfg.get("model_name", "torch_cross_section"))
    architecture = str(cfg.get("architecture", "deepsets"))
    factors = [str(x) for x in cfg.get("factors", [])]
    r88_requested = cfg.get("research_only") is not None
    if r88_requested:
        if not isinstance(cfg.get("research_only"), Mapping):
            raise ValueError("R88 requires an explicit top-level research_only object")
        if len(factors) != _R88_FACTOR_COUNT or len(set(factors)) != _R88_FACTOR_COUNT:
            raise ValueError(f"R88 requires exactly the frozen ordered {_R88_FACTOR_COUNT} factors")
    elif len(factors) != 50 or len(set(factors)) != 50:
        raise ValueError("torch cross-section r3 requires exactly the frozen ordered 50 factors")
    model_input_factors = _resolve_model_input_factors(
        full_factors=factors,
        configured_subset=cfg.get("model_input_factors"),
        contract_label="frozen R88 research contract" if r88_requested else "frozen live50",
    )
    desired_start = parse_date(start or cfg.get("start"))
    desired_end = parse_date(end or cfg.get("end"))
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    if desired_start > desired_end:
        raise ValueError("start date must be <= end date")
    rolling_cfg = dict(cfg.get("rolling", {}))
    window_days = int(rolling_cfg.get("window_days", 60))
    refit_every = int(execution_cfg.get("refit_every_n_days", cfg.get("refit_every_n_days", 1)))
    if refit_every != 1:
        raise ValueError("torch cross-section research requires execution refit_every_n_days=1")
    raw_root = Path(paths_cfg["raw_data_root"])
    clean_root = Path(paths_cfg["clean_data_root"])
    panel_root = Path(paths_cfg["panel_data_root"])
    label_root = Path(paths_cfg["label_data_root"])
    factor_root = Path(paths_cfg["factor_data_root"])
    results_root = resolve_output_path(cfg.get("results_root"), default_path=paths_cfg["results_root"], results_root=paths_cfg["results_root"])
    configured_score_output = resolve_output_path(cfg.get("score_output"), default_path=results_root / "scores" / model_name, results_root=results_root)
    configured_state_dir = resolve_output_path(dict(cfg.get("incremental", {})).get("state_dir"), default_path=results_root / "model_state" / model_name, results_root=results_root)
    resume_cfg = _resume_config(cfg)
    resume_enabled = bool(resume_interrupted or execution_cfg.get("resume_interrupted", False))
    if resume_enabled:
        state_dir, score_output, commit_root, resume_manifest_path = _resume_target_paths(
            cfg=cfg,
            results_root=results_root,
            model_name=model_name,
        )
    else:
        state_dir, score_output = configured_state_dir, configured_score_output
        commit_root = results_root / "daily_commits" / model_name
        resume_manifest_path = results_root / "manifests" / f"{model_name}.resume.json"
    _require_research_contract(cfg, results_root=results_root, score_output=score_output, state_dir=state_dir)
    if int(execution_cfg.get("parallel_shards", 1)) != 1:
        raise ValueError("torch cross-section research requires parallel_shards=1 for a contiguous warm-start chain")
    if int(execution_cfg.get("train_processes", 1)) != 1:
        raise ValueError("torch cross-section research requires train_processes=1")
    panel_name = str(cfg.get("panel_name", "T1430"))
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:42"))
    min_count = int(cfg.get("min_count", 30))
    winsor_lower, winsor_upper = parse_winsor_bounds(cfg.get("winsor", {}))
    if winsor_lower is not None or winsor_upper is not None:
        raise ValueError("torch cross-section r3 requires winsor disabled")
    if not bool(cfg.get("zscore", True)):
        raise ValueError("torch cross-section r3 requires daily z-score")
    missing_values = dict(dict(cfg.get("feature_engineering", {})).get("missing_values", {}))
    input_missingness = dict(cfg.get("model_input_missingness", {}))
    if not bool(input_missingness.get("add_missing_mask", True)):
        raise ValueError("torch cross-section r3 requires an internal missingness mask")
    neutralization_cache_root = (
        Path(str(resume_cfg["target_root"])) / "neutralization_cache" / model_name
        if resume_enabled
        else resolve_output_path(
            cfg.get("neutralization_cache_root"),
            default_path=results_root / "neutralization_cache" / model_name,
            results_root=results_root,
        )
    )
    r88_research_profile: dict[str, Any] | None = None
    r88_experiment_table: dict[str, Any] | None = None
    factor_contract: dict | None = None
    if r88_requested:
        r88_research_profile = _require_r88_research_profile(
            cfg,
            paths_cfg=paths_cfg,
            factor_root=factor_root,
            results_root=results_root,
            score_output=score_output,
            state_dir=state_dir,
            neutralization_cache_root=neutralization_cache_root,
            factors=factors,
            missing_values=missing_values,
        )
        factor_contract = _require_r88_factor_contract(
            contract_ref=str(cfg.get("factor_contract", "")),
            factors=factors,
            panel_name=panel_name,
            factor_time=factor_time,
            label_time=label_time,
        )
        r88_experiment_table = _require_r88_canonical_experiment_readiness(
            paths_cfg=paths_cfg,
            factor_root=factor_root,
            factor_contract=factor_contract,
            factors=factors,
            panel_name=panel_name,
            factor_time=factor_time,
            label_time=label_time,
        )
    else:
        if int(missing_values.get("min_available_factors", -1)) != 27 or bool(missing_values.get("add_valid_count_features", False)):
            raise ValueError("torch cross-section r3 requires live50 missing-value admission (27, no valid-count column)")
        table = paths_cfg.get("factor_table")
        if not isinstance(table, Mapping) or str(table.get("table_id", "")).strip() != "live":
            raise ValueError("torch cross-section live50 requires factor_table.table_id='live'")
        expected_factor_root = _resolved_path(Path(str(table.get("root", ""))) / "live")
        if _resolved_path(factor_root) != expected_factor_root:
            raise ValueError(f"torch cross-section r3 requires the declared live50 factor root: {expected_factor_root}")
    neutralizer = build_neutralizer(cfg.get("neutralization"), raw_data_root=raw_root, panel_data_root=panel_root, neutralization_cache_root=neutralization_cache_root)
    if neutralizer is None or not neutralizer.enabled:
        raise ValueError("torch cross-section r3 requires T-1 style-5 ridge neutralization")
    neutralization_summary = neutralizer.summary()
    if (
        neutralization_summary.get("method") != "ridge"
        or not np.isclose(float(neutralization_summary.get("ridge_alpha", float("nan"))), 1e-6)
        or int(neutralization_summary.get("min_count", -1)) != 30
    ):
        raise ValueError("torch cross-section r3 neutralization contract drift")
    objective = dict(cfg.get("objective", {}))
    if str(objective.get("name", "")).lower() != "listnet" or str(objective.get("day_weighting", "")).lower() != "equal":
        raise ValueError("torch cross-section r3 requires equal-day ListNet objective")
    lookback = prev_trading_days_from_raw(raw_root, desired_start, window_days, kind="snapshot", asset="cbond")
    scan_start = lookback[0] if lookback else desired_start
    calendar_days = list_trading_days_from_raw(raw_root, scan_start, desired_end, kind="snapshot", asset="cbond")
    if not calendar_days:
        raise RuntimeError("no trading calendar days for cross-sectional scorer")
    store = build_factor_reader(
        paths_cfg,
        panel_name=panel_name,
        window_minutes=window_minutes,
    )
    label_days = set(_iter_existing_label_days(label_root, scan_start, desired_end))
    if cutoff_day is not None:
        label_days = {day for day in label_days if day <= cutoff_day}
    # Retain the complete requested trading calendar.  Filtering target days
    # for factor availability would silently bridge a daily warm-start chain
    # around a missing score day.
    targets = [day for day in calendar_days if desired_start <= day <= desired_end]
    if not targets:
        raise RuntimeError("no requested trading days for cross-sectional scorer")
    tradable_map = build_tradable_code_map(
        raw_data_root=raw_root,
        days=calendar_days,
        buy_twap_col="twap_1442_1457",
        sell_twap_col="twap_0930_0939",
    )
    if not tradable_map:
        raise RuntimeError("T-1 o_0005 allowlist map is empty")
    train_cfg = dict(cfg.get("train", {}))
    model_params = dict(cfg.get("model_params", {}))
    if factor_contract is None:
        factor_contract = _validate_factor_contract(
            contract_ref=str(cfg.get("factor_contract", "")),
            factors=factors,
        )
    fingerprint_contract = {
        "factor_contract": factor_contract,
        "factor_root": str(factor_root),
        "factor_time": factor_time,
        "label_time": label_time,
        "min_count": min_count,
        "window_days": window_days,
        "allowlist": "tminus1_o_0005_strict",
    }
    if r88_research_profile is not None:
        fingerprint_contract["research_profile"] = r88_research_profile
        fingerprint_contract["r88_experiment_table"] = r88_experiment_table
    fingerprint = _config_fingerprint(
        architecture=architecture,
        factors=factors,
        model_input_factors=model_input_factors,
        model_params=model_params,
        input_missingness=input_missingness,
        objective=objective,
        neutralization=neutralization_summary,
        contract=fingerprint_contract,
    )
    resume_reentry = False
    if resume_enabled:
        existing_resume_artifacts = any(path.exists() for path in (state_dir, score_output, commit_root, resume_manifest_path))
        if existing_resume_artifacts:
            resume_reentry = True
        else:
            _require_fresh_resume_outputs(
                state_dir=state_dir,
                score_output=score_output,
                commit_root=commit_root,
                manifest_path=resume_manifest_path,
            )
        out_dir = Path(str(resume_cfg["target_root"])) / "models" / model_name / f"{desired_start:%Y-%m-%d}_{desired_end:%Y-%m-%d}"
    else:
        _require_fresh_research_outputs(state_dir=state_dir, score_output=score_output)
        out_dir = results_root / "models" / model_name / f"{desired_start:%Y-%m-%d}_{desired_end:%Y-%m-%d}" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=resume_reentry or not resume_enabled)
    state_dir.mkdir(parents=True, exist_ok=resume_reentry or not resume_enabled)
    base_seed = int(train_cfg.get("seed", 20260812))
    history_rows: list[dict] = []
    rolling_rows: list[dict] = []
    all_scores: list[pd.DataFrame] = []
    positions = {day: idx for idx, day in enumerate(calendar_days)}
    successful_score_days: list[date] = []
    source_records: list[CheckpointRecord] = []
    source_terminal: Path | None = None
    source_terminal_day: date | None = None
    existing_continuation_days: list[date] = []
    if resume_enabled:
        source_dir_raw = str(resume_cfg.get("source_state_dir", "")).strip()
        chain_start_raw = str(resume_cfg.get("chain_start", "")).strip()
        source_hash = str(resume_cfg.get("expected_source_checkpoint_manifest_sha256", "")).strip()
        if not source_dir_raw or not chain_start_raw:
            raise ValueError("r3 verified resume requires source_state_dir and chain_start")
        source_records, source_manifest, source_manifest_sha256 = _audit_resume_source_prefix(
            source_state_dir=Path(source_dir_raw),
            model_id=model_name,
            expected_fingerprint=fingerprint,
            calendar_days=calendar_days,
            chain_start=parse_date(chain_start_raw),
            expected_manifest_sha256=source_hash,
            label_cutoff=cutoff_day,
        )
        source_terminal = source_records[-1].path
        source_terminal_day = source_records[-1].score_day
        resume_payload: dict[str, Any] = {
            "format": "r3_v5_verified_resume_v1",
            "model_id": model_name,
            "requested_start": str(desired_start),
            "requested_end": str(desired_end),
            "contract_fingerprint": fingerprint,
            "source_state_dir": str(Path(source_dir_raw).resolve(strict=False)),
            "source_checkpoint_manifest_sha256": source_manifest_sha256,
            "source_checkpoint_manifest": source_manifest,
            "source_terminal_checkpoint": str(source_terminal),
            "score_reconstruction_provenance": "reconstructed_from_checkpoint_current_inputs",
        }
        if resume_reentry:
            _require_existing_resume_contract(
                manifest_path=resume_manifest_path,
                model_name=model_name,
                contract_fingerprint=fingerprint,
                source_manifest_sha256=source_manifest_sha256,
                expected_resume_payload=resume_payload,
            )
        else:
            _atomic_write_json(resume_manifest_path, resume_payload)
        # Reconstruct every old score solely from its saved final model state.
        # No call to _train_one_model is allowed for this immutable prefix.
        for record in source_records:
            frame, allowlist_rows = _reconstruct_checkpoint_score_frame(
                record=record,
                architecture=architecture,
                model_params=model_params,
                store=store,
                label_root=label_root,
                factors=factors,
                model_input_factors=model_input_factors,
                min_count=min_count,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                factor_time=factor_time,
                label_time=label_time,
                tradable_code_map=tradable_map,
                neutralizer=neutralizer,
                missing_values=missing_values,
                input_missingness=input_missingness,
            )
            commit_payload = _durably_commit_score_day(
                score_output=score_output,
                commit_root=commit_root,
                score_day=record.score_day,
                score_frame=frame,
                checkpoint_path=record.path,
                checkpoint_origin="source_checkpoint_v4",
                warm_start_from=record.previous_checkpoint,
                max_train_label_day=record.max_train_label_day,
                max_validation_label_day=record.max_validation_label_day,
                contract_fingerprint=fingerprint,
                score_provenance="reconstructed_from_checkpoint_current_inputs",
                history=[],
            )
            _verify_commit_checkpoint_link(
                commit_payload=commit_payload,
                record=record,
                expected_checkpoint_origin="source_checkpoint_v4",
                allowed_score_provenance={"reconstructed_from_checkpoint_current_inputs"},
            )
            all_scores.append(frame)
            _append_committed_resume_rows(
                commit_payload=commit_payload,
                record=record,
                frame=frame,
                allowlist_rows=allowlist_rows,
                history_rows=history_rows,
                rolling_rows=rolling_rows,
            )
        successful_score_days = [record.score_day for record in source_records]
        existing_commit_days = _existing_resume_commit_days(commit_root=commit_root)
        expected_prefix_days = [record.score_day for record in source_records]
        if existing_commit_days and existing_commit_days[: len(expected_prefix_days)] != expected_prefix_days:
            raise RuntimeError("r3 resume committed days do not begin with the immutable source prefix")
        for record in source_records:
            commit_payload = _verify_daily_commit(
                commit_path=_commit_path(commit_root, record.score_day),
                score_day=record.score_day,
                score_path=score_output / f"{record.score_day:%Y-%m}" / f"{record.score_day:%Y-%m-%d}.csv",
                checkpoint_path=record.path,
                contract_fingerprint=fingerprint,
            )
            _verify_commit_checkpoint_link(
                commit_payload=commit_payload,
                record=record,
                expected_checkpoint_origin="source_checkpoint_v4",
                allowed_score_provenance={"reconstructed_from_checkpoint_current_inputs"},
            )
        continuation_existing_days = existing_commit_days[len(expected_prefix_days) :]
        if continuation_existing_days:
            expected_continuation = list(
                calendar_days[
                    positions[expected_prefix_days[-1]] + 1 : positions[expected_prefix_days[-1]] + 1 + len(continuation_existing_days)
                ]
            )
            if continuation_existing_days != expected_continuation:
                raise RuntimeError("r3 resume continuation commits are not a contiguous trading-day prefix")
            prior_path: Path | None = source_terminal
            for day in continuation_existing_days:
                checkpoint_path = _checkpoint_path(state_dir, day)
                commit_payload = _verify_daily_commit(
                    commit_path=_commit_path(commit_root, day),
                    score_day=day,
                    score_path=score_output / f"{day:%Y-%m}" / f"{day:%Y-%m-%d}.csv",
                    checkpoint_path=checkpoint_path,
                    contract_fingerprint=fingerprint,
                )
                record = _read_checkpoint_record(
                    checkpoint_path=checkpoint_path,
                    expected_fingerprint=fingerprint,
                    label_cutoff=cutoff_day,
                    expected_checkpoint_predecessor=prior_path,
                )
                _verify_commit_checkpoint_link(
                    commit_payload=commit_payload,
                    record=record,
                    expected_checkpoint_origin="continuation_checkpoint_v5",
                    allowed_score_provenance={
                        "trained_continuation_v5",
                        "reconstructed_from_checkpoint_current_inputs",
                    },
                )
                frame = _load_score_file_exact(
                    score_path=score_output / f"{day:%Y-%m}" / f"{day:%Y-%m-%d}.csv",
                    score_day=day,
                )
                all_scores.append(frame)
                _append_committed_resume_rows(
                    commit_payload=commit_payload,
                    record=record,
                    frame=frame,
                    allowlist_rows=len(tradable_map.get(day, set())),
                    history_rows=history_rows,
                    rolling_rows=rolling_rows,
                )
                prior_path = checkpoint_path
            successful_score_days.extend(continuation_existing_days)
            existing_continuation_days = list(continuation_existing_days)
        # If the last process died after atomically saving a continuation
        # checkpoint (and perhaps its score CSV) but before its daily commit,
        # repair exactly that one verified next day.  It is deliberately a
        # score-only operation: retraining would make an interruption alter
        # the warm-start state chain.
        if source_terminal is None or source_terminal_day is None:
            raise RuntimeError("r3 verified resume is missing its source terminal checkpoint")
        orphan_record = _find_resume_orphan_checkpoint(
            state_dir=state_dir,
            score_output=score_output,
            committed_continuation_days=continuation_existing_days,
            committed_score_days=existing_commit_days,
            source_terminal=source_terminal,
            source_terminal_day=source_terminal_day,
            calendar_days=calendar_days,
            expected_fingerprint=fingerprint,
            label_cutoff=cutoff_day,
        )
        if orphan_record is not None:
            frame, allowlist_rows = _reconstruct_checkpoint_score_frame(
                record=orphan_record,
                architecture=architecture,
                model_params=model_params,
                store=store,
                label_root=label_root,
                factors=factors,
                min_count=min_count,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                factor_time=factor_time,
                label_time=label_time,
                tradable_code_map=tradable_map,
                neutralizer=neutralizer,
                missing_values=missing_values,
                input_missingness=input_missingness,
            )
            commit_payload = _durably_commit_score_day(
                score_output=score_output,
                commit_root=commit_root,
                score_day=orphan_record.score_day,
                score_frame=frame,
                checkpoint_path=orphan_record.path,
                checkpoint_origin="continuation_checkpoint_v5",
                warm_start_from=orphan_record.previous_checkpoint,
                max_train_label_day=orphan_record.max_train_label_day,
                max_validation_label_day=orphan_record.max_validation_label_day,
                contract_fingerprint=fingerprint,
                score_provenance="reconstructed_from_checkpoint_current_inputs",
                history=[],
            )
            _verify_commit_checkpoint_link(
                commit_payload=commit_payload,
                record=orphan_record,
                expected_checkpoint_origin="continuation_checkpoint_v5",
                allowed_score_provenance={"reconstructed_from_checkpoint_current_inputs"},
            )
            all_scores.append(frame)
            _append_committed_resume_rows(
                commit_payload=commit_payload,
                record=orphan_record,
                frame=frame,
                allowlist_rows=allowlist_rows,
                history_rows=history_rows,
                rolling_rows=rolling_rows,
            )
            successful_score_days.append(orphan_record.score_day)
            existing_continuation_days.append(orphan_record.score_day)
    for seq, score_day in enumerate(targets, start=1):
        if resume_enabled and score_day in successful_score_days:
            continue
        pos = positions[score_day]
        if pos < window_days - 1:
            if successful_score_days:
                raise RuntimeError(f"strict r3 rolling chain lost its full lookback after first score: {score_day}")
            continue
        window = calendar_days[pos - window_days + 1 : pos + 1]
        historical_days = window[:-1]
        history_window_error = _strict_history_window_error(
            historical_days,
            expected_history_days=window_days - 1,
            label_cutoff=cutoff_day,
        )
        if history_window_error:
            if successful_score_days:
                raise RuntimeError(
                    f"strict r3 input gap after {successful_score_days[-1]} before {score_day}: "
                    f"{history_window_error}"
                )
            print(f"[cross-section] pre-chain skip score_day={score_day}: {history_window_error}")
            continue
        # A score-only smoke can cap label reads, but only after the guard
        # above proves that the cap covers all 59 natural history days.
        required_label_days = list(historical_days)
        missing_labels = sorted(set(required_label_days) - label_days)
        if missing_labels:
            message = f"missing historical labels: {[str(day) for day in missing_labels]}"
            if successful_score_days:
                raise RuntimeError(f"strict r3 input gap after {successful_score_days[-1]} before {score_day}: {message}")
            print(f"[cross-section] pre-chain skip score_day={score_day}: {message}")
            continue
        train_days, val_days = _split_rolling_train_validation(
            required_label_days,
            float(train_cfg.get("train_ratio", 0.7)),
        )
        if not train_days or not val_days:
            if successful_score_days:
                raise RuntimeError(f"strict rolling chain has empty train/validation day split: {score_day}")
            print(f"[cross-section] pre-chain skip score_day={score_day}: empty train/validation split")
            continue
        train_data = _build_data(
            store=store, label_root=label_root, days=train_days, factors=factors, min_count=min_count,
            model_input_factors=model_input_factors,
            winsor_lower=winsor_lower, winsor_upper=winsor_upper, zscore=True, factor_time=factor_time,
            label_time=label_time, require_label=True, tradable_code_map=tradable_map, neutralizer=neutralizer,
            missing_values=missing_values, input_missingness=input_missingness,
        )
        val_data = _build_data(
            store=store, label_root=label_root, days=val_days, factors=factors, min_count=min_count,
            model_input_factors=model_input_factors,
            winsor_lower=winsor_lower, winsor_upper=winsor_upper, zscore=True, factor_time=factor_time,
            label_time=label_time, require_label=True, tradable_code_map=tradable_map, neutralizer=neutralizer,
            missing_values=missing_values, input_missingness=input_missingness,
        )
        train_missing = _missing_group_days(train_data, train_days)
        val_missing = _missing_group_days(val_data, val_days)
        if train_missing or val_missing:
            message = (
                "missing admitted daily groups: "
                f"train={[str(day) for day in sorted(train_missing)]} "
                f"validation={[str(day) for day in sorted(val_missing)]}"
            )
            if successful_score_days:
                raise RuntimeError(f"strict r3 input gap after {successful_score_days[-1]} before {score_day}: {message}")
            print(f"[cross-section] pre-chain skip score_day={score_day}: {message}")
            continue
        # Calendar days before the first scoreable 60-day window naturally
        # have no state.  Cold-start exactly once at the first *successful*
        # refit.  Thereafter every score must consume the immediately prior
        # successful state; no scan-back to an older checkpoint is permitted.
        continuation_days = [day for day in successful_score_days if source_terminal_day is None or day > source_terminal_day]
        _require_expected_state_inventory(state_dir=state_dir, successful_score_days=continuation_days)
        if continuation_days:
            previous = _checkpoint_path(state_dir, continuation_days[-1])
            predecessor_of_previous = (
                _checkpoint_path(state_dir, continuation_days[-2])
                if len(continuation_days) >= 2
                else source_terminal
            )
        elif resume_enabled:
            previous = source_terminal
            predecessor_of_previous = (
                source_records[-2].path if len(source_records) >= 2 else None
            )
        else:
            previous, predecessor_of_previous = _warm_start_chain_paths(state_dir, successful_score_days)
        if previous is None:
            initial_state = None
        else:
            if not previous.exists():
                raise RuntimeError(f"strict warm-start chain is broken before {score_day}: missing {previous.name}")
            initial_state = _load_warm_start_state(
                checkpoint_path=previous,
                expected_fingerprint=fingerprint,
                score_day=score_day,
                label_cutoff=cutoff_day,
                device=torch.device("cpu"),
                expected_checkpoint_predecessor=predecessor_of_previous,
            )
            if initial_state is None:
                raise RuntimeError(f"strict warm-start checkpoint failed provenance validation: {previous}")
        model, history = _train_one_model(
            architecture=architecture, n_features=train_data.x.shape[1], model_params=model_params,
            train_data=train_data, val_data=val_data, train_cfg=train_cfg,
            temperature=float(objective.get("temperature", 1.0)), initial_state=initial_state,
            seed=_seed_for_score_day(base_seed=base_seed, score_day=score_day),
        )
        test_data = _build_score_day_split(
            store=store,
            label_root=label_root,
            score_day=score_day,
            factors=factors,
            model_input_factors=model_input_factors,
            min_count=min_count,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            factor_time=factor_time,
            label_time=label_time,
            tradable_code_map=tradable_map,
            neutralizer=neutralizer,
            missing_values=missing_values,
            input_missingness=input_missingness,
        )
        allowed_codes = tradable_map[score_day]
        prediction = _predict_split(model, test_data, device=torch.device("cpu"))
        score_frame = _score_frame(score_day=score_day, split=test_data, prediction=prediction)
        all_scores.append(score_frame)
        checkpoint_path = _checkpoint_path(state_dir, score_day)
        if checkpoint_path.exists():
            raise RuntimeError(f"strict r3 runner refuses to overwrite checkpoint: {checkpoint_path}")
        _save_warm_start_state(
            checkpoint_path=checkpoint_path, model=model, fingerprint=fingerprint,
            score_day=score_day, max_train_label_day=max(train_days), max_validation_label_day=max(val_days),
            previous_checkpoint=previous,
        )
        if resume_enabled:
            _durably_commit_score_day(
                score_output=score_output,
                commit_root=commit_root,
                score_day=score_day,
                score_frame=score_frame,
                checkpoint_path=checkpoint_path,
                checkpoint_origin="continuation_checkpoint_v5",
                warm_start_from=previous,
                max_train_label_day=max(train_days),
                max_validation_label_day=max(val_days),
                contract_fingerprint=fingerprint,
                score_provenance="trained_continuation_v5",
                history=history,
                train_days=len(train_days),
                val_days=len(val_days),
            )
        for row in history:
            history_rows.append({"score_day": score_day, "warm_start_from": previous.name if initial_state is not None and previous else None, **row})
        rolling_rows.append({
            "score_day": score_day, "train_days": len(train_days), "val_days": len(val_days),
            "max_train_label_day": max(train_days), "max_validation_label_day": max(val_days),
            "warm_start_from": previous.name if initial_state is not None and previous else None,
            "score_rows": len(prediction), "allowlist_rows": len(allowed_codes), "contract_fingerprint": fingerprint,
            "checkpoint_origin": "continuation_checkpoint_v5" if resume_enabled else "fresh_checkpoint",
            "score_provenance": "trained_continuation_v5" if resume_enabled else "fresh_run",
        })
        successful_score_days.append(score_day)
        print(f"[cross-section] {seq}/{len(targets)} score_day={score_day} rows={len(prediction)} warm_start={initial_state is not None}")
    if not all_scores:
        raise RuntimeError("torch cross-section scorer produced no scores")
    scores = pd.concat(all_scores, ignore_index=True)
    if not resume_enabled:
        write_scores_by_date(
            score_output,
            scores,
            overwrite=bool(cfg.get("score_overwrite", False)),
            dedupe=bool(cfg.get("score_dedupe", True)),
        )
    pd.DataFrame(history_rows).to_csv(out_dir / "training_history.csv", index=False)
    pd.DataFrame(rolling_rows).to_csv(out_dir / "rolling_audit.csv", index=False)
    summary = {
        "score_days": int(scores["trade_date"].nunique()),
        "score_rows": int(len(scores)),
        "all_equal_days": int(scores.groupby("trade_date")["score"].nunique().le(1).sum()),
        "contract_fingerprint": fingerprint,
        "score_output": str(score_output),
        "state_dir": str(state_dir),
    }
    (out_dir / "score_eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved rolling: {out_dir}")
    print(f"saved scores: {score_output}")


if __name__ == "__main__":
    main()
