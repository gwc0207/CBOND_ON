"""Fail-closed, research-only launcher for the R88 research universe.

``R88`` means exactly ``legacy27 + screened61``.  The profile is deliberately
separate from live factor packs: it is a provenance and admission contract for
an isolated FactorStore, not permission to touch a live path or to train a
model.  The command defaults to a no-write preflight.

The launcher validates every exact Rust contract before it loads DataHub data,
creates a scratch directory, imports a research batch module, or calls a
FactorStore writer.  Its legacy path computes only the R38 research additions
and merges them with a frozen R50 research FactorStore.  The
``--full-current-r88`` path is separate: it computes all 88 exact contracts
from DataHub clean snapshots through the same in-memory T1430 panel construction
used by the active live factor runtime.  It never reads the frozen FactorStore
or creates persistent panel data.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from cbond_on.config.loader import load_config_file, parse_date  # noqa: E402
from cbond_on.core.trading_days import list_trading_days_from_raw  # noqa: E402
from cbond_on.domain.factors.spec import (  # noqa: E402
    FactorSpec,
    build_factor_col,
    infer_factor_context_requirements,
)
from cbond_on.infra.factors.daily_context import (  # noqa: E402
    index_daily_table,
    load_daily_context_for_day,
    resolve_daily_source_specs,
)
from cbond_on.infra.factors.pipeline import (  # noqa: E402
    _index_daily_table,
    _read_bond_stock_map_day,
    run_factor_pipeline,
)
from cbond_on.infra.data.panel import _iter_existing_snapshot_days  # noqa: E402
from cbond_on.domain.factors.storage import FactorStore  # noqa: E402
from cbond_on.infra.factors.canonical_writer import (  # noqa: E402
    CanonicalFactorTableWriter,
    current_catalog_contract,
    issue_experiment_publisher_authority,
)
from cbond_on.infra.factors.canonical_store import (  # noqa: E402
    CanonicalFactorStore,
    CanonicalFactorStoreError,
)
from cbond_on.infra.factors.rust_backend import (  # noqa: E402
    build_factor_frame_rust,
    validate_rust_first_contracts,
)
from cbond_on.schemas.config.shared import validate_paths_config  # noqa: E402
from cbond_on.workflows.research.factor_batch import (  # noqa: E402
    R88_RESEARCH_ADMISSION_PROFILE,
    R88_RESEARCH_FACTOR_MODULE_MAP,
    R88_RESEARCH_FACTOR_MODULES,
    prepare_factor_modules,
    validate_r88_research_module_contract,
)


_PROFILE_PATH = (
    _REPO_ROOT
    / "cbond_on"
    / "factor_contracts"
    / "profiles"
    / "research_r88_rust88_20260825.json5"
)
_PROFILE_ID = R88_RESEARCH_ADMISSION_PROFILE
_PROFILE_SCHEMA = "research_r88_factor_profile/v1"
_BACKFILL_MANIFEST_NAME = "r88_backfill_manifest.json"
_BACKFILL_MANIFEST_SCHEMA = "r88_factor_backfill_manifest/v1"
_FULL_BACKFILL_STATUS = "completed_rust88_backfill"
_PARTIAL_BACKFILL_STATUS = "partial_rust88_backfill"
_FULL_CURRENT_RECOMPUTE_STATUS = "completed_rust88_current_recompute"
_PARTIAL_CURRENT_RECOMPUTE_STATUS = "partial_rust88_current_recompute"
_SOURCE_PATHS_PROFILE = "data/paths_factor_mining_20260802"
_PANEL_CONFIG_NAME = "panel"
_RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_CANONICAL_FACTOR_STORE_ROOT = Path(r"D:/cbond_on/factor_store")
_CANONICAL_EXPERIMENT_TABLE_ID = "experiment"
_CANONICAL_EXPERIMENT_TABLE_ROOT = _CANONICAL_FACTOR_STORE_ROOT / _CANONICAL_EXPERIMENT_TABLE_ID
_DATAHUB_RAW_ROOT = Path(r"D:/cbond_data_hub/raw_data")
_DATAHUB_CLEAN_ROOT = Path(r"D:/cbond_data_hub/clean_data")
_DATAHUB_MANIFEST_ROOT = Path(r"D:/cbond_data_hub/manifests")
_PUBLIC_PANEL_STORE_ROOT = Path(r"D:/cbond_on/panel_data")
_PANEL_STORE_NAME = "T1430"
_PANEL_STORE_ASSETS = ("cbond", "stock")
_PANEL_STORE_CONTRACT_SCHEMA = "cbond_on_panel_store_contract/v1"
_PANEL_STORE_MANIFEST_SCHEMA = "cbond_on_panel_store_manifest/v1"
_PANEL_STORE_DONE_SCHEMA = "cbond_on_panel_store_done/v1"
_PANEL_STORE_LOGICAL_TIME = "14:30"
_PANEL_STORE_PHYSICAL_CUTOFF = "14:29:00"
_PANEL_STORE_COUNT_POINTS = 5000
_PANEL_STORE_MAX_LOOKBACK_DAYS = 4
_APPROVED_START = date(2024, 1, 1)
_APPROVED_END = date(2026, 7, 30)
# The immutable profile envelope begins on 2024-01-01.  The first complete
# historical T1430 chain used by this research contract is 2024-01-03.  This
# is deliberately a *current DataHub* calendar contract, not an inference from
# the legacy frozen R50 store.
_FULL_CURRENT_EXECUTABLE_START = date(2024, 1, 3)
# The frozen profile's provenance range remains through 2026-07-30.  A new
# clean-direct materialisation is a distinct generation and may extend that
# immutable factor identity through the last label-ready 2026 score day.  Do
# not change ``_APPROVED_END``: the legacy R38/frozen-R50 route stays frozen.
_FULL_CURRENT_EXECUTABLE_END = date(2026, 8, 27)
_FULL_CURRENT_EXPECTED_DAY_COUNT = 642
_FULL_CURRENT_CALENDAR_SOURCE = "datahub_clean_cbond_stock_snapshot_with_raw_calendar_crosscheck"
_FULL_CURRENT_DEFAULT_GENERATION_ID = "r88_clean_direct_20260828"
_FULL_CURRENT_GENERATION_MIGRATION_ID = "r88_clean_direct_20260828_full_642d"
_ELIGIBLE_STATUS = "eligible_for_fresh_rust_backfill"
_BLOCKED_STATUS = "blocked_pending_exact_rust_contracts"
_R88_RUST_SITE_ENV = "CBOND_ON_R88_RUST_SITE"
# This is an acceptance-captured scratch copy, never the active live50 root.
# It is read-only input to the R38-only research mode and must be proven before
# every execution.  Keeping it under research_scratch prevents a shortcut that
# would make a running live FactorStore an implicit R88 source.
_FROZEN_R50_FACTOR_ROOT = (
    _RESEARCH_SCRATCH_PARENT
    / "rust50_unified_final_20260806_160508"
    / "fullchain_preseed_r4"
    / "runtime_r2"
    / "factor_data"
)
_R38_FACTOR_STORE_DIR = "r38_factor_data"
_R38_EXECUTION_MODE = "r38_only_plus_frozen_r50_merge"
_R38_BOND_SNAPSHOT_COLUMNS = (
    "code",
    "trade_time",
    "pre_close",
    "last",
    "open",
    "volume",
    "amount",
    "num_trades",
    "high_limited",
    "low_limited",
    "ask_price1",
    "ask_volume1",
    "bid_price1",
    "bid_volume1",
    "ask_price2",
    "ask_volume2",
    "bid_price2",
    "bid_volume2",
    "ask_price3",
    "ask_volume3",
    "bid_price3",
    "bid_volume3",
    "ask_price4",
    "ask_volume4",
    "bid_price4",
    "bid_volume4",
    "ask_price5",
    "ask_volume5",
    "bid_price5",
    "bid_volume5",
)
_R38_STOCK_SNAPSHOT_COLUMNS = (
    "code",
    "trade_time",
    "pre_close",
    "last",
    "open",
    "ask_price1",
    "ask_volume1",
    "bid_price1",
    "bid_volume1",
)

_SOURCE_ROOTS = {
    "raw_data_root": _DATAHUB_RAW_ROOT,
    "clean_data_root": _DATAHUB_CLEAN_ROOT,
    "cleaned_data_root": _DATAHUB_CLEAN_ROOT,
}
_DERIVED_ROOTS = OrderedDict(
    (
        ("panel_data_root", Path("panel_data")),
        ("label_data_root", Path("label_data")),
        ("factor_data_root", Path("factor_data")),
        ("ads_root", Path("ads")),
        ("results_root", Path("results")),
        ("model_root", Path("results") / "models"),
        ("score_root", Path("results") / "scores"),
        ("logs_root", Path("logs")),
    )
)
_FULL_CURRENT_DERIVED_ROOTS = OrderedDict(
    (
        ("results_root", Path("results")),
    )
)
_PRODUCTION_ROOTS = (
    Path(r"D:/cbond_on/factor_data"),
    Path(r"D:/cbond_on/panel_data"),
    Path(r"D:/cbond_on/label_data"),
    Path(r"D:/cbond_on/results"),
    Path(r"D:/cbond_on/logs"),
    Path(r"D:/cbond_on/model_state"),
)


class R88ContractError(ValueError):
    """The immutable research profile is malformed or stale."""


class R88RustContractIncomplete(RuntimeError):
    """One or more R88 specs lacks an exact Rust instance contract."""


@dataclass(frozen=True)
class PreparedR88Backfill:
    """A complete no-write plan.  It becomes executable only after admission."""

    profile_path: Path
    profile: dict[str, Any]
    specs: tuple[FactorSpec, ...]
    frozen_r50_specs: tuple[FactorSpec, ...]
    r38_specs: tuple[FactorSpec, ...]
    scratch_root: Path
    paths_cfg: dict[str, Any]
    factor_cfg: dict[str, Any]
    start: date
    end: date
    # ``None`` is valid only for the retired R38/frozen-R50 audit route.  A
    # current clean-direct recompute must declare an internal canonical
    # experiment generation so it can never overwrite the active historical
    # experiment table.
    experiment_generation_id: str | None = None


@dataclass(frozen=True)
class R38ContextLoaders:
    daily_requirements: tuple[Any, ...]
    daily_source_specs: Mapping[str, Any]
    daily_source_indexes: Mapping[str, Any]
    requires_map: bool
    map_index: Any


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _path_text(path: str | Path) -> str:
    return _resolved(path).as_posix()


def _is_strict_child(path: str | Path, parent: str | Path) -> bool:
    candidate = _resolved(path)
    root = _resolved(parent)
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return candidate != root


def _configure_isolated_rust_site(site_text: str | Path | None) -> Path:
    """Make the R88 runner load only a scratch-built Rust wheel.

    A research backfill must never replace or accidentally import the local
    repository's ``cbond_on_rust`` package, because that package is shared by
    the live process.  The caller supplies an extracted wheel/site below the
    research scratch parent; placing it first on ``sys.path`` affects only
    this dedicated research process.
    """

    raw = str(site_text or "").strip()
    if not raw:
        raise R88ContractError(
            "R88 executable backfill requires --rust-site (or "
            f"{_R88_RUST_SITE_ENV}) pointing to an isolated scratch wheel site"
        )
    site = _resolved(raw)
    if not _is_strict_child(site, _RESEARCH_SCRATCH_PARENT):
        raise R88ContractError(
            "R88 --rust-site must resolve strictly below research scratch: "
            f"{_resolved(_RESEARCH_SCRATCH_PARENT).as_posix()}"
        )
    package = site / "cbond_on_rust"
    if not package.is_dir():
        raise FileNotFoundError(f"R88 isolated Rust site has no cbond_on_rust package: {package}")
    for module_name in tuple(sys.modules):
        if module_name == "cbond_on_rust" or module_name.startswith("cbond_on_rust."):
            del sys.modules[module_name]
    site_text_normalized = str(site)
    sys.path = [entry for entry in sys.path if str(_resolved(entry)) != site_text_normalized]
    sys.path.insert(0, site_text_normalized)
    importlib.invalidate_caches()
    return site


def _assert_loaded_isolated_rust_site(site: Path) -> None:
    """Prove the loaded extension belongs to the requested scratch wheel."""

    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    module_path = _resolved(str(getattr(module, "__file__", "")))
    extension_path = _resolved(str(getattr(extension, "__file__", "")))
    if not _is_strict_child(module_path, site) or not _is_strict_child(extension_path, site):
        raise RuntimeError(
            "R88 backfill did not load the isolated scratch Rust wheel: "
            f"module={module_path} extension={extension_path} site={site}"
        )


def _loaded_rust_execution_evidence() -> dict[str, Any]:
    """Capture the exact isolated extension that executed this process."""

    module = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    module_path = _resolved(str(getattr(module, "__file__", "")))
    extension_path = _resolved(str(getattr(extension, "__file__", "")))
    capabilities = module.factor_capabilities()
    payload = json.dumps(
        capabilities,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "package_path": module_path.as_posix(),
        "extension_path": extension_path.as_posix(),
        "extension_sha256": _sha256(extension_path),
        "capabilities_sha256": hashlib.sha256(payload).hexdigest(),
        "contract_count": len(capabilities.get("factor_contracts", [])),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    """Match the public PanelStore publisher's canonical JSON hash contract."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: str | Path, *, field: str) -> dict[str, Any]:
    resolved = _resolved(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} is missing: {resolved.as_posix()}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise R88ContractError(f"{field} is not valid JSON: {resolved.as_posix()}") from exc
    if not isinstance(payload, Mapping):
        raise R88ContractError(f"{field} must be a JSON object: {resolved.as_posix()}")
    return dict(payload)


def _canonical_spec_payload(raw_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Preserve the same exact instance fields checked by the Rust adapter."""

    payload: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_specs):
        if not isinstance(raw, Mapping):
            raise R88ContractError(f"profile.factor_specs[{position}] must be an object")
        name = str(raw.get("name", "")).strip()
        factor = str(raw.get("factor", "")).strip()
        if not name or not factor:
            raise R88ContractError(
                f"profile.factor_specs[{position}] requires non-empty name and factor"
            )
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise R88ContractError(f"profile.factor_specs[{position}].params must be an object")
        output_col = raw.get("output_col")
        if output_col is not None:
            output_col = str(output_col).strip() or None
        rust_contract_id = raw.get("rust_contract_id")
        if rust_contract_id is not None:
            rust_contract_id = str(rust_contract_id).strip() or None
        payload.append(
            {
                "name": name,
                "factor": factor,
                "params": dict(params),
                "output_col": output_col,
                "rust_contract_id": rust_contract_id,
            }
        )
    return payload


def _specs_sha256(payload: Sequence[Mapping[str, Any]]) -> str:
    try:
        canonical = json.dumps(
            list(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise R88ContractError(
            "profile.factor_specs must contain finite JSON-compatible values"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_profile(path: str | Path = _PROFILE_PATH) -> tuple[Path, dict[str, Any]]:
    resolved = _resolved(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"R88 research profile is missing: {resolved.as_posix()}")
    import json5

    with resolved.open("r", encoding="utf-8") as handle:
        profile = json5.load(handle) or {}
    if not isinstance(profile, dict):
        raise R88ContractError("R88 research profile must be an object")
    return resolved, dict(profile)


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if value != expected:
        raise R88ContractError(f"R88 profile {field} must equal {expected!r}, got {value!r}")


def _validate_profile_structure(profile: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the frozen R88 identity without opening DataHub or FactorStore paths."""

    _require_exact(profile.get("schema_version"), _PROFILE_SCHEMA, field="schema_version")
    _require_exact(profile.get("research_only"), True, field="research_only")
    _require_exact(profile.get("admission_profile"), _PROFILE_ID, field="admission_profile")
    _require_exact(profile.get("execution_policy"), "rust_first", field="execution_policy")
    _require_exact(profile.get("factor_count"), 88, field="factor_count")
    _require_exact(profile.get("source_counts"), {"legacy27": 27, "screened61": 61}, field="source_counts")
    _require_exact(
        profile.get("time_contract"),
        {"panel_name": "T1430", "factor_time": "14:30", "label_time": "14:42"},
        field="time_contract",
    )
    _require_exact(
        profile.get("range_contract"),
        {"start": _APPROVED_START.isoformat(), "end": _APPROVED_END.isoformat()},
        field="range_contract",
    )
    _require_exact(profile.get("model_training_ready"), False, field="model_training_ready")

    compute = profile.get("compute")
    if not isinstance(compute, Mapping):
        raise R88ContractError("R88 profile compute must be an object")
    _require_exact(compute.get("engine"), "rust", field="compute.engine")
    _require_exact(compute.get("execution_policy"), "rust_first", field="compute.execution_policy")
    _require_exact(compute.get("backend"), "cpu", field="compute.backend")

    raw_names = profile.get("factors")
    raw_specs = profile.get("factor_specs")
    if not isinstance(raw_names, list) or not isinstance(raw_specs, list):
        raise R88ContractError("R88 profile factors and factor_specs must be lists")
    names = [str(value).strip() for value in raw_names]
    if len(names) != 88 or any(not name for name in names):
        raise R88ContractError("R88 profile factors must contain exactly 88 non-empty names")
    if len(set(names)) != len(names):
        raise R88ContractError("R88 profile factors must not contain duplicate names")
    if len(raw_specs) != 88:
        raise R88ContractError("R88 profile factor_specs must contain exactly 88 entries")

    payload = _canonical_spec_payload(raw_specs)
    spec_names = [str(item["name"]) for item in payload]
    if names != spec_names:
        raise R88ContractError(
            "R88 profile factors must exactly equal the ordered factor_specs names"
        )
    actual_digest = _specs_sha256(payload)
    expected_digest = str(profile.get("specs_sha256", "")).strip().lower()
    if len(expected_digest) != 64 or any(ch not in "0123456789abcdef" for ch in expected_digest):
        raise R88ContractError("R88 profile specs_sha256 must be a lowercase SHA-256 hex digest")
    if actual_digest != expected_digest:
        raise R88ContractError(
            "R88 profile specs_sha256 mismatch: "
            f"expected={expected_digest!r}, actual={actual_digest!r}"
        )

    # Profile JSON records the exact reviewed module list and key mapping, but
    # cannot extend the executable surface.  The research workflow compares it
    # against its own static R88 allowlist before either Rust admission or
    # DataHub/scratch access.
    try:
        validate_r88_research_module_contract(
            declared_modules=profile.get("research_factor_modules"),
            module_map=profile.get("research_kernel_modules"),
            factor_keys=(item["factor"] for item in payload),
        )
    except (TypeError, ValueError) as exc:
        raise R88ContractError(f"R88 research module admission is invalid: {exc}") from exc

    pending = [str(item["name"]) for item in payload if not item.get("rust_contract_id")]
    declared_pending = profile.get("pending_rust_contract_factors")
    if not isinstance(declared_pending, list):
        raise R88ContractError("R88 profile pending_rust_contract_factors must be a list")
    if pending != [str(value).strip() for value in declared_pending]:
        raise R88ContractError(
            "R88 profile pending_rust_contract_factors does not exactly match factor_specs"
        )
    status = str(profile.get("execution_status", "")).strip()
    if pending:
        _require_exact(status, _BLOCKED_STATUS, field="execution_status while contracts are pending")
        _require_exact(
            profile.get("next_execution_status"),
            _ELIGIBLE_STATUS,
            field="next_execution_status",
        )
    elif status != _ELIGIBLE_STATUS:
        raise R88ContractError(
            "a complete R88 profile must use "
            f"execution_status={_ELIGIBLE_STATUS!r}, got {status!r}"
        )
    return payload, pending


def _verify_provenance(profile: Mapping[str, Any]) -> None:
    """Verify immutable source-list hashes only after the profile is structurally valid."""

    provenance = profile.get("provenance")
    if not isinstance(provenance, Mapping):
        raise R88ContractError("R88 profile provenance must be an object")
    legacy = provenance.get("legacy27_pack")
    screened = provenance.get("screened61")
    if not isinstance(legacy, Mapping) or not isinstance(screened, Mapping):
        raise R88ContractError("R88 profile provenance must contain legacy27_pack and screened61")
    legacy_path = _resolved(_REPO_ROOT / str(legacy.get("path", "")))
    if not legacy_path.is_file() or _sha256(legacy_path) != str(legacy.get("sha256", "")).lower():
        raise R88ContractError("R88 legacy27 provenance hash does not match the frozen pack")
    accepted_path = _resolved(str(screened.get("accepted_factors_csv", "")))
    manifest_path = _resolved(str(screened.get("screen_manifest", "")))
    if not accepted_path.is_file() or _sha256(accepted_path) != str(
        screened.get("accepted_factors_sha256", "")
    ).lower():
        raise R88ContractError("R88 screened61 accepted_factors provenance hash mismatch")
    if not manifest_path.is_file() or _sha256(manifest_path) != str(
        screened.get("screen_manifest_sha256", "")
    ).lower():
        raise R88ContractError("R88 screened61 screen_manifest provenance hash mismatch")
    _require_exact(screened.get("selected_count"), 61, field="provenance.screened61.selected_count")
    _require_exact(screened.get("score_start"), "2025-01-02", field="provenance.screened61.score_start")
    _require_exact(screened.get("score_end"), "2026-07-30", field="provenance.screened61.score_end")


def _to_specs(payload: Sequence[Mapping[str, Any]]) -> tuple[FactorSpec, ...]:
    return tuple(
        FactorSpec(
            name=str(item["name"]),
            factor=str(item["factor"]),
            params=dict(item["params"]),
            output_col=item.get("output_col"),
            rust_contract_id=item.get("rust_contract_id"),
        )
        for item in payload
    )


def _split_frozen_r50_and_r38_specs(
    specs: Sequence[FactorSpec],
) -> tuple[tuple[FactorSpec, ...], tuple[FactorSpec, ...]]:
    """Split the frozen 88 profile by exact instance-contract namespace.

    The profile is the only authority for the split.  In particular, do not
    infer it from a mutable live factor pack or a FactorStore header.
    """

    r50 = tuple(
        spec
        for spec in specs
        if str(spec.rust_contract_id or "").startswith("live50_r5/")
    )
    r38 = tuple(
        spec
        for spec in specs
        if str(spec.rust_contract_id or "").startswith("research_r88_20260825/")
    )
    r50_cols = tuple(build_factor_col(spec) for spec in r50)
    r38_cols = tuple(build_factor_col(spec) for spec in r38)
    all_cols = tuple(build_factor_col(spec) for spec in specs)
    if (
        len(r50) != 50
        or len(r38) != 38
        or len(set(r50_cols)) != 50
        or len(set(r38_cols)) != 38
        or set(r50_cols).intersection(r38_cols)
        or set(r50_cols).union(r38_cols) != set(all_cols)
    ):
        raise R88ContractError(
            "R88 profile must split exactly into disjoint frozen R50 and new R38 factor columns"
        )
    return r50, r38


def _factor_store_for(root: str | Path) -> FactorStore:
    return FactorStore(_resolved(root), panel_name="T1430")


def _require_factor_index(frame: pd.DataFrame, *, day: date, field: str) -> None:
    if not isinstance(frame.index, pd.MultiIndex) or tuple(frame.index.names) != ("dt", "code"):
        raise R88ContractError(f"{field} must have a unique (dt, code) MultiIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise R88ContractError(f"{field} index must be unique and sorted")
    dates = pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce").normalize()
    if dates.isna().any() or not (dates == pd.Timestamp(day)).all():
        raise R88ContractError(f"{field} index must contain only score-day {day.isoformat()} labels")
    codes = frame.index.get_level_values("code").astype(str)
    if not len(codes) or any(not code.strip() for code in codes):
        raise R88ContractError(f"{field} index must contain non-empty codes")


def _frame_index_sha256(frame: pd.DataFrame) -> str:
    index_frame = frame.index.to_frame(index=False)
    canonical = index_frame.astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_frozen_r50_day(
    root: str | Path,
    day: date,
    *,
    expected_columns: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and certify one immutable R50 score-day without touching it."""

    store = _factor_store_for(root)
    path = store.day_path(day)
    if not path.is_file():
        raise R88ContractError(f"frozen R50 FactorStore is missing {day.isoformat()}: {path.as_posix()}")
    frame = store.read_day(day)
    _require_factor_index(frame, day=day, field="frozen R50 FactorStore")
    actual_columns = tuple(str(column) for column in frame.columns)
    expected_set = set(expected_columns)
    if len(actual_columns) != 50 or set(actual_columns) != expected_set:
        missing = sorted(expected_set.difference(actual_columns))
        unexpected = sorted(set(actual_columns).difference(expected_set))
        raise R88ContractError(
            "frozen R50 FactorStore columns do not equal the exact profile R50 set: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frame, {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "columns": list(actual_columns),
        "index_sha256": _frame_index_sha256(frame),
    }


def _merge_exact_r50_r38(
    r50: pd.DataFrame,
    r38: pd.DataFrame,
    *,
    day: date,
    r50_columns: Sequence[str],
    r38_columns: Sequence[str],
    output_columns: Sequence[str],
) -> pd.DataFrame:
    """Strict index-preserving concat; deliberately never an outer join."""

    _require_factor_index(r50, day=day, field="frozen R50 FactorStore")
    _require_factor_index(r38, day=day, field="new R38 FactorStore")
    if not r50.index.equals(r38.index):
        raise R88ContractError("R38 output index must exactly equal the frozen R50 FactorStore index")
    actual_r50 = tuple(str(column) for column in r50.columns)
    actual_r38 = tuple(str(column) for column in r38.columns)
    if set(actual_r50) != set(r50_columns) or len(actual_r50) != len(r50_columns):
        raise R88ContractError("frozen R50 columns changed during exact merge")
    if set(actual_r38) != set(r38_columns) or len(actual_r38) != len(r38_columns):
        raise R88ContractError("new R38 columns do not equal the exact research R38 set")
    if set(actual_r50).intersection(actual_r38):
        raise R88ContractError("R50/R38 merge has overlapping factor columns")
    expected_output = tuple(str(column) for column in output_columns)
    if set(actual_r50).union(actual_r38) != set(expected_output):
        raise R88ContractError("R50/R38 merge columns do not equal the exact R88 profile")
    merged = pd.concat([r50, r38], axis=1).reindex(columns=expected_output)
    if not merged.index.equals(r50.index) or tuple(merged.columns) != expected_output:
        raise R88ContractError("R88 exact concat lost the certified R50 index or profile column order")
    return merged


def _snapshot_day_path(cleaned_root: str | Path, *, asset: str, day: date) -> Path:
    return _resolved(cleaned_root) / "snapshot" / asset / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"


def _strict_current_day_panel(
    path: str | Path,
    *,
    day: date,
    eligible_index: pd.MultiIndex,
    columns: Sequence[str],
    require_all_codes: bool = True,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Create a current-day, all-visible strict-14:29 panel for exact R50 keys.

    It intentionally avoids the generic `count_points` builder: R38 contains
    whole-path formulas, while the frozen R50 index is the eligibility oracle.
    """

    if not isinstance(eligible_index, pd.MultiIndex) or tuple(eligible_index.names) != ("dt", "code"):
        raise R88ContractError("R50 eligibility oracle must be a (dt, code) MultiIndex")
    requested = tuple(dict.fromkeys(str(column) for column in columns))
    required = {"code", "trade_time"}
    if not required.issubset(requested):
        raise R88ContractError("R38 snapshot projection must retain code and trade_time")
    snapshot = _resolved(path)
    if not snapshot.is_file():
        raise FileNotFoundError(f"R38 snapshot is missing: {snapshot.as_posix()}")
    raw = pd.read_parquet(snapshot, columns=list(requested))
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise R88ContractError(f"R38 snapshot missing required columns: {missing}")
    # `format="mixed"` keeps a strict sub-microsecond boundary from being
    # coerced to NaT merely because an earlier source row had second precision.
    trade_time = pd.to_datetime(raw["trade_time"], errors="coerce", format="mixed")
    trade_time_us = trade_time.dt.floor("us")
    cutoff = pd.Timestamp.combine(day, datetime.min.time()).replace(hour=14, minute=29)
    clock_minutes = trade_time_us.dt.hour * 60 + trade_time_us.dt.minute
    same_day = trade_time_us.dt.normalize().eq(pd.Timestamp(day))
    strict_visible = trade_time_us.le(cutoff)
    not_lunch = (clock_minutes < (11 * 60 + 30)) | (clock_minutes >= (13 * 60))
    eligible_codes = eligible_index.get_level_values("code").astype(str).str.strip().str.upper()
    label_by_code = pd.Series(
        pd.to_datetime(eligible_index.get_level_values("dt"), errors="coerce").to_numpy(),
        index=eligible_codes,
    )
    if label_by_code.index.has_duplicates:
        raise R88ContractError("R50 eligibility oracle has duplicate codes for one score day")
    frame = raw.loc[same_day & strict_visible & not_lunch].copy()
    frame["code"] = frame["code"].astype(str).str.strip().str.upper()
    frame = frame[frame["code"].isin(label_by_code.index)]
    if not frame.empty:
        frame["trade_time"] = trade_time.loc[frame.index]
        frame = frame.sort_values(["code", "trade_time"], kind="mergesort")
    observed = set(frame["code"]) if not frame.empty else set()
    missing_codes = sorted(set(label_by_code.index).difference(observed))
    if missing_codes and require_all_codes:
        raise R88ContractError(
            "R38 current-day panel is missing frozen R50 eligible code(s): " + ", ".join(missing_codes[:8])
        )
    if frame.empty:
        if not allow_empty:
            raise R88ContractError(f"R38 strict current-day panel is empty for {day.isoformat()}")
        frame = pd.DataFrame(columns=[*requested, "dt", "seq"])
        frame = frame.set_index(["dt", "code", "seq"])
    else:
        frame["dt"] = frame["code"].map(label_by_code)
        frame["seq"] = frame.groupby("code", sort=False).cumcount().astype("int64")
        frame = frame.set_index(["dt", "code", "seq"]).sort_index()
    frame.attrs["__build_day__"] = day.isoformat()
    return frame


def _stock_index_for_codes(index: pd.MultiIndex, codes: Sequence[str]) -> pd.MultiIndex:
    labels = pd.to_datetime(index.get_level_values("dt"), errors="coerce")
    if labels.isna().any() or labels.nunique() != 1:
        raise R88ContractError("R50 eligibility oracle must have one valid score-day label")
    normalized = sorted({str(code).strip().upper() for code in codes if str(code).strip()})
    if not normalized:
        return pd.MultiIndex.from_arrays([[], []], names=["dt", "code"])
    return pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([labels[0]] * len(normalized)), pd.Index(normalized)],
        names=["dt", "code"],
    )


def _normalize_stock_code(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text or text in {"NAN", "NONE", "<NA>"}:
        return ""
    if "." in text:
        return text
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) != 6:
        return text
    if digits[0] in {"5", "6", "9"}:
        return f"{digits}.SH"
    if digits[0] in {"4", "8"}:
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def _instrument_key(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.split(".", 1)[0]


def _strict_prior_base_stock_codes(
    daily_base: pd.DataFrame | None,
    *,
    day: date,
    bond_codes: Sequence[str],
) -> set[str]:
    """Return exact strict-prior base mappings needed by joint R38 signals."""

    if daily_base is None or daily_base.empty:
        return set()
    needed = {_instrument_key(code) for code in bond_codes if _instrument_key(code)}
    if not {"trade_date", "code", "stock_code"}.issubset(daily_base.columns):
        return set()
    frame = daily_base.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["_instrument_key"] = frame["code"].map(_instrument_key)
    prior = frame[(frame["trade_date"] < day) & frame["_instrument_key"].isin(needed)]
    if prior.empty:
        return set()
    anchor = prior["trade_date"].max()
    return {
        _normalize_stock_code(value)
        for value in prior.loc[prior["trade_date"].eq(anchor), "stock_code"]
        if _normalize_stock_code(value)
    }


def _write_new_factor_day(store: FactorStore, day: date, frame: pd.DataFrame) -> Path:
    """Write a previously absent research day through a local atomic replace."""

    path = store.day_path(day)
    if path.exists():
        raise FileExistsError(f"R38-only runner refuses to overwrite FactorStore day: {path.as_posix()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"R38-only runner found stale temporary factor file: {temporary.as_posix()}")
    frame.to_parquet(temporary, index=True)
    os.replace(temporary, path)
    return path


def _iter_factor_store_days(root: str | Path, *, field: str) -> list[date]:
    """List every dated FactorStore file below one exact root.

    This deliberately looks at the concrete FactorStore rather than an
    execution counter.  Completion admission must be tied to durable files,
    so a prematurely stopped run cannot claim that its planned days finished.
    """

    base = _resolved(root) / "factors" / "T1430"
    if not base.is_dir():
        raise R88ContractError(f"{field} is missing T1430: {base.as_posix()}")
    out: list[date] = []
    for path in base.glob("*/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        out.append(day)
    out.sort()
    if len(set(out)) != len(out):
        raise R88ContractError(f"{field} has duplicate score-day files")
    return out


def _iter_frozen_r50_days(root: str | Path, *, start: date, end: date) -> list[date]:
    out = [
        day
        for day in _iter_factor_store_days(root, field="frozen R50 FactorStore root")
        if start <= day <= end
    ]
    if not out:
        raise R88ContractError("frozen R50 FactorStore has no days in requested R88 range")
    return out


def _days_sha256(days: Sequence[date]) -> str:
    """Hash an ordered score-day calendar without accepting an unordered set."""

    payload = [day.isoformat() for day in days]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _full_current_datahub_calendar(
    *,
    raw_data_root: Path,
    cleaned_data_root: Path,
    start: date,
    end: date,
) -> tuple[list[date], dict[str, Any]]:
    """Resolve the current-contract score calendar from DataHub only.

    The full-current route must not use frozen R50 even as a calendar oracle.
    It therefore requires agreement among the raw trading calendar and the
    clean cbond/stock snapshot inventories.  It is a calendar and daily
    context gate only: factor computation builds a clean-direct panel in
    memory for each score day.  This helper is read-only and is deliberately
    called before a fresh scratch root is created.
    """

    raw_days = list_trading_days_from_raw(
        raw_data_root,
        start,
        end,
        kind="snapshot",
        asset="cbond",
    )
    raw_stock_days = list_trading_days_from_raw(
        raw_data_root,
        start,
        end,
        kind="snapshot",
        asset="stock",
    )
    clean_cbond_days = _iter_existing_snapshot_days(
        cleaned_data_root,
        start,
        end,
        asset="cbond",
    )
    clean_stock_days = _iter_existing_snapshot_days(
        cleaned_data_root,
        start,
        end,
        asset="stock",
    )
    if not raw_days:
        raise R88ContractError("full-current R88 DataHub raw trading calendar is empty")
    if raw_days != clean_cbond_days:
        raise R88ContractError(
            "full-current R88 raw and clean cbond snapshot calendars differ: "
            f"raw_only={[day.isoformat() for day in sorted(set(raw_days).difference(clean_cbond_days))[:5]]} "
            f"clean_only={[day.isoformat() for day in sorted(set(clean_cbond_days).difference(raw_days))[:5]]}"
        )
    if raw_days != raw_stock_days:
        raise R88ContractError(
            "full-current R88 raw cbond and raw stock snapshot calendars differ: "
            f"cbond_only={[day.isoformat() for day in sorted(set(raw_days).difference(raw_stock_days))[:5]]} "
            f"stock_only={[day.isoformat() for day in sorted(set(raw_stock_days).difference(raw_days))[:5]]}"
        )
    if raw_days != clean_stock_days:
        raise R88ContractError(
            "full-current R88 raw and clean stock snapshot calendars differ: "
            f"raw_only={[day.isoformat() for day in sorted(set(raw_days).difference(clean_stock_days))[:5]]} "
            f"clean_only={[day.isoformat() for day in sorted(set(clean_stock_days).difference(raw_days))[:5]]}"
        )
    return raw_days, {
        "source": _FULL_CURRENT_CALENDAR_SOURCE,
        "requested_range": {"start": start.isoformat(), "end": end.isoformat()},
        "raw_cbond_snapshot_days": [day.isoformat() for day in raw_days],
        "raw_stock_snapshot_days": [day.isoformat() for day in raw_stock_days],
        "clean_cbond_snapshot_days": [day.isoformat() for day in clean_cbond_days],
        "clean_stock_snapshot_days": [day.isoformat() for day in clean_stock_days],
        "days_sha256": _days_sha256(raw_days),
    }


def _verify_direct_datahub_evidence(*, expected_days: Sequence[date]) -> dict[str, Any]:
    """Prove each clean-direct input day is published before construction.

    The retired persistent PanelStore used to carry this evidence indirectly.
    The clean-direct route records DataHub's clean manifest and publish commit
    directly, while retaining the exact cbond/stock readiness gate.
    """

    days = list(expected_days)
    if not days or days != sorted(days) or len(days) != len(set(days)):
        raise R88ContractError("clean-direct DataHub evidence requires a non-empty ordered calendar")
    rows: list[dict[str, str]] = []
    for day in days:
        clean_path = _DATAHUB_MANIFEST_ROOT / "clean" / f"{day:%Y-%m-%d}.json"
        done_path = _DATAHUB_MANIFEST_ROOT / "publish" / f"{day:%Y-%m-%d}.done"
        clean = _read_json_object(clean_path, field="DataHub clean manifest")
        done = _read_json_object(done_path, field="DataHub publish done marker")
        if str(clean.get("trade_day", "")).strip() != day.isoformat():
            raise R88ContractError(f"DataHub clean manifest trade_day mismatch for {day.isoformat()}")
        if str(done.get("trade_day", "")).strip() != day.isoformat() or done.get("ready") is not True:
            raise R88ContractError(f"DataHub publish done is not ready for {day.isoformat()}")
        run_id = str(clean.get("run_id", "")).strip()
        if not run_id or str(done.get("run_id", "")).strip() != run_id:
            raise R88ContractError(f"DataHub clean/publish run_id mismatch for {day.isoformat()}")
        if str(clean.get("status", "")).strip().lower() != "success":
            raise R88ContractError(f"DataHub clean manifest is not successful for {day.isoformat()}")
        assets = clean.get("assets_status")
        if not isinstance(assets, Mapping) or any(
            str(assets.get(asset, "")).strip().lower() != "success"
            for asset in _PANEL_STORE_ASSETS
        ):
            raise R88ContractError(f"DataHub clean manifest asset status is incomplete for {day.isoformat()}")
        rows.append(
            {
                "day": day.isoformat(),
                "run_id": run_id,
                "clean_manifest": clean_path.as_posix(),
                "clean_manifest_sha256": _sha256(clean_path),
                "publish_done": done_path.as_posix(),
                "publish_done_sha256": _sha256(done_path),
            }
        )
    return {
        "mode": "clean_direct",
        "manifest_root": _DATAHUB_MANIFEST_ROOT.as_posix(),
        "days_sha256": _days_sha256(days),
        "days": rows,
    }


def _panel_store_contract_path(root: Path) -> Path:
    return root / "contracts" / f"{_PANEL_STORE_NAME}.json"


def _panel_store_manifest_path(root: Path, day: date) -> Path:
    return root / "manifests" / _PANEL_STORE_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.json"


def _panel_store_done_path(root: Path, day: date) -> Path:
    return root / "publish" / _PANEL_STORE_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.done"


def _panel_store_asset_path(root: Path, day: date, asset: str) -> Path:
    return root / "panels" / asset / _PANEL_STORE_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"


def _iter_panel_store_asset_days(
    root: Path,
    *,
    asset: str,
    start: date,
    end: date,
) -> list[date]:
    base = root / "panels" / asset / _PANEL_STORE_NAME
    if not base.is_dir():
        raise FileNotFoundError(f"published PanelStore {asset} directory is missing: {base.as_posix()}")
    days: list[date] = []
    for path in base.glob("*/*.parquet"):
        try:
            day = datetime.strptime(path.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if start <= day <= end:
            days.append(day)
    days.sort()
    if len(days) != len(set(days)):
        raise R88ContractError(f"published PanelStore {asset} has duplicate score-day files")
    return days


def _require_path_equal(actual: object, expected: Path, *, field: str) -> None:
    text = str(actual or "").strip()
    if not text or _resolved(text) != _resolved(expected):
        raise R88ContractError(
            f"{field} must equal {expected.as_posix()}, got {text or '<empty>'}"
        )


def _verify_published_panel_contract(root: Path) -> dict[str, Any]:
    """Validate the immutable public T1430 contract before reading its days."""

    path = _panel_store_contract_path(root)
    contract = _read_json_object(path, field="published PanelStore contract")
    declared_hash = str(contract.get("contract_sha256", "")).strip().lower()
    canonical = dict(contract)
    canonical.pop("contract_sha256", None)
    actual_hash = _canonical_json_sha256(canonical)
    if declared_hash != actual_hash:
        raise R88ContractError("published PanelStore contract_sha256 does not match its canonical payload")
    required = {
        "schema_version": _PANEL_STORE_CONTRACT_SCHEMA,
        "panel_name": _PANEL_STORE_NAME,
        "logical_panel_time": _PANEL_STORE_LOGICAL_TIME,
        "physical_cutoff_time": _PANEL_STORE_PHYSICAL_CUTOFF,
        "lead_minutes": 1,
        "panel_mode": "snapshot_sequence",
        "assets": list(_PANEL_STORE_ASSETS),
        "count_points": _PANEL_STORE_COUNT_POINTS,
        "max_lookback_days": _PANEL_STORE_MAX_LOOKBACK_DAYS,
        "schedule_windows": [{"start": "14:30", "end": "14:30"}],
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise R88ContractError(
                f"published PanelStore contract {field} must equal {expected!r}, got {contract.get(field)!r}"
            )
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256(path),
        "contract_sha256": declared_hash,
    }


def _verify_panel_datahub_source(source: Mapping[str, Any], *, day: date) -> dict[str, Any]:
    """Read-only verification of the publisher's DataHub release evidence."""

    if str(source.get("trade_day", "")).strip() != day.isoformat():
        raise R88ContractError(f"published PanelStore datahub_source trade_day mismatch for {day.isoformat()}")
    run_id = str(source.get("datahub_run_id", "")).strip()
    if not run_id:
        raise R88ContractError(f"published PanelStore datahub_source is missing datahub_run_id for {day.isoformat()}")
    clean = source.get("clean_manifest")
    done = source.get("publish_done")
    if not isinstance(clean, Mapping) or not isinstance(done, Mapping):
        raise R88ContractError(f"published PanelStore datahub_source lacks clean manifest/done evidence for {day.isoformat()}")
    clean_path = _DATAHUB_MANIFEST_ROOT / "clean" / f"{day:%Y-%m-%d}.json"
    done_path = _DATAHUB_MANIFEST_ROOT / "publish" / f"{day:%Y-%m-%d}.done"
    _require_path_equal(clean.get("path"), clean_path, field="datahub_source.clean_manifest.path")
    _require_path_equal(done.get("path"), done_path, field="datahub_source.publish_done.path")
    clean_payload = _read_json_object(clean_path, field="DataHub clean manifest")
    done_payload = _read_json_object(done_path, field="DataHub publish done marker")
    if str(clean.get("sha256", "")).strip().lower() != _sha256(clean_path):
        raise R88ContractError(f"published PanelStore clean manifest hash drifted for {day.isoformat()}")
    if str(done.get("sha256", "")).strip().lower() != _sha256(done_path):
        raise R88ContractError(f"published PanelStore publish done hash drifted for {day.isoformat()}")
    if str(clean_payload.get("trade_day", "")).strip() != day.isoformat():
        raise R88ContractError(f"DataHub clean manifest trade_day mismatch for {day.isoformat()}")
    if str(done_payload.get("trade_day", "")).strip() != day.isoformat():
        raise R88ContractError(f"DataHub publish done trade_day mismatch for {day.isoformat()}")
    if str(clean_payload.get("run_id", "")).strip() != run_id or str(done_payload.get("run_id", "")).strip() != run_id:
        raise R88ContractError(f"DataHub manifest/done run_id mismatch for {day.isoformat()}")
    if str(clean_payload.get("status", "")).strip().lower() != "success":
        raise R88ContractError(f"DataHub clean manifest is not successful for {day.isoformat()}")
    if done_payload.get("ready") is not True:
        raise R88ContractError(f"DataHub publish done marker is not ready for {day.isoformat()}")
    assets_status = clean_payload.get("assets_status")
    if not isinstance(assets_status, Mapping) or any(
        str(assets_status.get(asset, "")).strip().lower() != "success"
        for asset in _PANEL_STORE_ASSETS
    ):
        raise R88ContractError(f"DataHub clean manifest asset status is incomplete for {day.isoformat()}")
    return {
        "datahub_run_id": run_id,
        "clean_manifest_sha256": _sha256(clean_path),
        "publish_done_sha256": _sha256(done_path),
    }


def _verify_published_panel_day(
    root: Path,
    *,
    day: date,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one complete cbond+stock published day without writing the store."""

    manifest_path = _panel_store_manifest_path(root, day)
    done_path = _panel_store_done_path(root, day)
    manifest = _read_json_object(manifest_path, field="published PanelStore day manifest")
    done = _read_json_object(done_path, field="published PanelStore done marker")
    contract_hash = str(contract["contract_sha256"])
    if str(manifest.get("schema_version", "")).strip() != _PANEL_STORE_MANIFEST_SCHEMA:
        raise R88ContractError(f"published PanelStore manifest schema mismatch for {day.isoformat()}")
    if str(manifest.get("status", "")).strip().lower() != "published":
        raise R88ContractError(f"published PanelStore manifest is not published for {day.isoformat()}")
    if str(manifest.get("trade_day", "")).strip() != day.isoformat():
        raise R88ContractError(f"published PanelStore manifest trade_day mismatch for {day.isoformat()}")
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id:
        raise R88ContractError(f"published PanelStore manifest is missing run_id for {day.isoformat()}")
    if str(manifest.get("contract_sha256", "")).strip().lower() != contract_hash:
        raise R88ContractError(f"published PanelStore manifest contract hash mismatch for {day.isoformat()}")
    _require_path_equal(
        manifest.get("contract_path"),
        _panel_store_contract_path(root),
        field="published PanelStore manifest contract_path",
    )
    if str(done.get("schema_version", "")).strip() != _PANEL_STORE_DONE_SCHEMA or done.get("ready") is not True:
        raise R88ContractError(f"published PanelStore done marker is not ready for {day.isoformat()}")
    if str(done.get("trade_day", "")).strip() != day.isoformat() or str(done.get("run_id", "")).strip() != run_id:
        raise R88ContractError(f"published PanelStore manifest/done identity mismatch for {day.isoformat()}")
    if str(done.get("contract_sha256", "")).strip().lower() != contract_hash:
        raise R88ContractError(f"published PanelStore done contract hash mismatch for {day.isoformat()}")
    _require_path_equal(done.get("manifest_path"), manifest_path, field="published PanelStore done manifest_path")
    manifest_file_hash = _sha256(manifest_path)
    if str(done.get("manifest_sha256", "")).strip().lower() != manifest_file_hash:
        raise R88ContractError(f"published PanelStore done manifest hash mismatch for {day.isoformat()}")

    assets = manifest.get("assets")
    done_assets = done.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(_PANEL_STORE_ASSETS):
        raise R88ContractError(f"published PanelStore manifest assets are incomplete for {day.isoformat()}")
    if not isinstance(done_assets, Mapping) or set(done_assets) != set(_PANEL_STORE_ASSETS):
        raise R88ContractError(f"published PanelStore done assets are incomplete for {day.isoformat()}")
    expected_logical = pd.Timestamp(day).replace(hour=14, minute=30)
    expected_cutoff = pd.Timestamp(day).replace(hour=14, minute=29)
    asset_evidence: dict[str, dict[str, Any]] = {}
    for asset in _PANEL_STORE_ASSETS:
        raw = assets.get(asset)
        if not isinstance(raw, Mapping):
            raise R88ContractError(f"published PanelStore manifest {asset} entry is invalid for {day.isoformat()}")
        path = _panel_store_asset_path(root, day, asset)
        _require_path_equal(raw.get("path"), path, field=f"published PanelStore {asset} path")
        if not path.is_file():
            raise FileNotFoundError(f"published PanelStore {asset} panel is missing: {path.as_posix()}")
        actual_hash = _sha256(path)
        if str(raw.get("sha256", "")).strip().lower() != actual_hash:
            raise R88ContractError(f"published PanelStore {asset} parquet hash mismatch for {day.isoformat()}")
        if str(done_assets.get(asset, "")).strip().lower() != actual_hash:
            raise R88ContractError(f"published PanelStore done {asset} hash mismatch for {day.isoformat()}")
        try:
            declared_bytes = int(raw.get("bytes", -1))
            declared_rows = int(raw.get("rows", 0))
            declared_codes = int(raw.get("codes", 0))
        except (TypeError, ValueError) as exc:
            raise R88ContractError(
                f"published PanelStore {asset} numeric metadata is invalid for {day.isoformat()}"
            ) from exc
        if declared_bytes != int(path.stat().st_size):
            raise R88ContractError(f"published PanelStore {asset} byte count mismatch for {day.isoformat()}")
        if declared_rows <= 0 or declared_codes <= 0:
            raise R88ContractError(f"published PanelStore {asset} metadata is empty for {day.isoformat()}")
        if list(raw.get("index_names", [])) != ["dt", "code", "seq"]:
            raise R88ContractError(f"published PanelStore {asset} index contract mismatch for {day.isoformat()}")
        try:
            logical_dt = pd.Timestamp(raw.get("logical_dt"))
            physical_cutoff = pd.Timestamp(raw.get("physical_cutoff"))
            physical_min = pd.Timestamp(raw.get("physical_min"))
            physical_max = pd.Timestamp(raw.get("physical_max"))
        except (TypeError, ValueError) as exc:
            raise R88ContractError(
                f"published PanelStore {asset} time metadata is invalid for {day.isoformat()}"
            ) from exc
        if logical_dt != expected_logical:
            raise R88ContractError(f"published PanelStore {asset} logical time mismatch for {day.isoformat()}")
        if physical_cutoff != expected_cutoff:
            raise R88ContractError(f"published PanelStore {asset} cutoff mismatch for {day.isoformat()}")
        if pd.isna(physical_min) or pd.isna(physical_max) or physical_min > physical_max or physical_max > expected_cutoff:
            raise R88ContractError(f"published PanelStore {asset} physical range is invalid for {day.isoformat()}")
        asset_evidence[asset] = {
            "path": path.as_posix(),
            "sha256": actual_hash,
            "bytes": int(path.stat().st_size),
            "rows": declared_rows,
            "codes": declared_codes,
        }
    source = manifest.get("datahub_source")
    if not isinstance(source, Mapping):
        raise R88ContractError(f"published PanelStore datahub_source is invalid for {day.isoformat()}")
    return {
        "day": day.isoformat(),
        "run_id": run_id,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": manifest_file_hash,
        "done_path": done_path.as_posix(),
        "assets": asset_evidence,
        "datahub_source": _verify_panel_datahub_source(source, day=day),
    }


def _verify_published_panel_store(root: str | Path, *, expected_days: Sequence[date]) -> dict[str, Any]:
    """Fail closed unless every requested day is a committed public panel bundle."""

    panel_root = _resolved(root)
    normalized_days = list(expected_days)
    if not normalized_days or normalized_days != sorted(normalized_days) or len(set(normalized_days)) != len(normalized_days):
        raise R88ContractError("published PanelStore verification requires a non-empty, unique, ordered calendar")
    if not panel_root.is_dir():
        raise FileNotFoundError(f"published PanelStore root is missing: {panel_root.as_posix()}")
    contract = _verify_published_panel_contract(panel_root)
    for asset in _PANEL_STORE_ASSETS:
        actual_days = _iter_panel_store_asset_days(
            panel_root,
            asset=asset,
            start=normalized_days[0],
            end=normalized_days[-1],
        )
        if actual_days != normalized_days:
            raise R88ContractError(
                f"published PanelStore {asset} parquet calendar does not equal the requested calendar"
            )
    rows = [_verify_published_panel_day(panel_root, day=day, contract=contract) for day in normalized_days]
    return {
        "root": panel_root.as_posix(),
        "contract": contract,
        "days_sha256": _days_sha256(normalized_days),
        "days": rows,
    }


def _require_rust_admission(profile: Mapping[str, Any], specs: Sequence[FactorSpec], pending: Sequence[str]) -> None:
    """Fail before source-path resolution, panel reads, or scratch writes."""

    if pending:
        preview = ", ".join(pending[:8])
        suffix = "" if len(pending) <= 8 else f", ... (+{len(pending) - 8} more)"
        raise R88RustContractIncomplete(
            "R88 Rust-first preflight blocked before DataHub/factor-store I/O: "
            f"{len(pending)} exact rust_contract_id values are missing: {preview}{suffix}"
        )
    _require_exact(profile.get("execution_status"), _ELIGIBLE_STATUS, field="execution_status")
    _require_exact(profile.get("model_training_ready"), False, field="model_training_ready")
    validate_rust_first_contracts(specs)


def _assert_scratch_root(value: str | Path, *, execute: bool) -> Path:
    scratch_root = _resolved(value)
    scratch_parent = _resolved(_RESEARCH_SCRATCH_PARENT)
    if not _is_strict_child(scratch_root, scratch_parent):
        raise ValueError(
            "--scratch-root must resolve strictly below "
            f"{scratch_parent.as_posix()}, got {scratch_root.as_posix()}"
        )
    for production_root in _PRODUCTION_ROOTS:
        if scratch_root == _resolved(production_root) or _is_strict_child(scratch_root, production_root):
            raise ValueError(f"--scratch-root must never be a production root: {scratch_root.as_posix()}")
    if execute and scratch_root.exists():
        raise FileExistsError(
            "--execute requires a previously absent scratch root; refusing to reuse "
            f"{scratch_root.as_posix()}"
        )
    return scratch_root


def _assert_equal_path(actual: str | Path, expected: str | Path, *, field: str) -> None:
    actual_path = _resolved(actual)
    expected_path = _resolved(expected)
    if actual_path != expected_path:
        raise RuntimeError(
            f"unsafe {field}: expected {expected_path.as_posix()}, got {actual_path.as_posix()}"
        )


def _full_current_generation_paths(generation_id: str | None):
    """Resolve one *inactive* internal experiment generation write target.

    The normal experiment resolver deliberately follows only the active
    pointer.  A clean-direct R88 rebuild instead has to be explicit: it writes
    one newly created internal generation, validates it, and leaves activation
    to a later, separately authorised operation.
    """

    resolved_id = str(generation_id or _FULL_CURRENT_DEFAULT_GENERATION_ID).strip()
    store = CanonicalFactorStore(_CANONICAL_FACTOR_STORE_ROOT)
    paths = store.experiment_generation_paths(resolved_id)
    if paths.is_legacy_flat:
        raise R88ContractError("full-current R88 may not write the legacy flat experiment generation")
    return paths


def _load_datahub_only_paths() -> dict[str, Any]:
    paths_cfg = validate_paths_config(load_config_file(_SOURCE_PATHS_PROFILE))
    for field, expected in _SOURCE_ROOTS.items():
        if field not in paths_cfg:
            raise RuntimeError(f"source paths profile did not resolve required {field}")
        _assert_equal_path(paths_cfg[field], expected, field=field)
    return dict(paths_cfg)


def _assert_strict_t1429_panel_contract(panel_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Prove the shared clean-direct panel excludes physical 14:29+ rows.

    The R88 profile labels the score at 14:30 but newly ported intraday
    signals are admissible only through the 14:29 snapshot.  The standard
    panel builder obtains that boundary by subtracting one minute from the
    declared 14:30 window.  Validate it explicitly rather than assuming a
    mutable global panel config still has that property.
    """

    panel = dict(panel_config) if panel_config is not None else dict(load_config_file(_PANEL_CONFIG_NAME))
    if str(panel.get("panel_name", "")).strip() != "T1430":
        raise R88ContractError("R88 requires panel_config.panel_name='T1430'")
    if int(panel.get("lead_minutes", -1)) != 1:
        raise R88ContractError("R88 requires panel_config.lead_minutes=1 for strict physical 14:29")
    schedule = panel.get("schedule")
    if not isinstance(schedule, Mapping) or str(schedule.get("mode", "")).strip() != "custom_windows":
        raise R88ContractError("R88 requires a custom_windows T1430 panel schedule")
    windows = schedule.get("windows")
    if not isinstance(windows, list) or not any(
        isinstance(window, Mapping)
        and str(window.get("start", "")).strip() == "14:30"
        and str(window.get("end", "")).strip() == "14:30"
        for window in windows
    ):
        raise R88ContractError("R88 requires a 14:30-labelled panel window with one-minute lead")
    if str(panel.get("panel_mode", "")).strip() != "snapshot_sequence":
        raise R88ContractError("R88 requires panel_config.panel_mode='snapshot_sequence'")
    if [str(asset).strip().lower() for asset in panel.get("assets", [])] != list(_PANEL_STORE_ASSETS):
        raise R88ContractError("R88 requires panel_config.assets=['cbond', 'stock']")
    if int(panel.get("count_points", -1)) != _PANEL_STORE_COUNT_POINTS:
        raise R88ContractError(f"R88 requires panel_config.count_points={_PANEL_STORE_COUNT_POINTS}")
    if int(panel.get("max_lookback_days", -1)) != _PANEL_STORE_MAX_LOOKBACK_DAYS:
        raise R88ContractError(f"R88 requires panel_config.max_lookback_days={_PANEL_STORE_MAX_LOOKBACK_DAYS}")
    return {
        "panel_name": "T1430",
        "labelled_factor_time": "14:30",
        "physical_cutoff": "14:29:00.999999",
        "lead_minutes": 1,
    }


def _redirect_derived_paths(
    source_paths: Mapping[str, Any],
    scratch_root: Path,
    *,
    full_current_r88: bool = False,
    experiment_generation_id: str | None = None,
) -> dict[str, Any]:
    """Bind the requested R88 mode to its allowed input/output roots.

    Full-current R88 constructs a strict T1430 panel in memory from DataHub
    clean snapshots and allows only FactorStore/result outputs below scratch.
    The legacy R38-only route retains its pre-existing isolated derived-path
    boundary.
    """

    redirected = (
        {field: source_paths[field] for field in _SOURCE_ROOTS}
        if full_current_r88
        else dict(source_paths)
    )
    derived_roots = _FULL_CURRENT_DERIVED_ROOTS if full_current_r88 else _DERIVED_ROOTS
    if full_current_r88:
        # ``run_factor_pipeline`` retains a positional panel-root argument for
        # API compatibility.  In clean-direct mode it is never read, so bind
        # it to the already-audited DataHub clean root rather than a cache.
        redirected["panel_data_root"] = redirected["cleaned_data_root"]
        generation = _full_current_generation_paths(experiment_generation_id)
        # The full-current R88 publisher targets a newly named internal
        # generation, never the active historical experiment data.  It is
        # formal canonical storage, not a scratch FactorStore, and therefore
        # must not be reintroduced into the derived-root loop below.
        redirected["factor_data_root"] = _path_text(generation.generation_root)
        redirected["factor_table"] = {
            "table_id": _CANONICAL_EXPERIMENT_TABLE_ID,
            "root": _path_text(_CANONICAL_FACTOR_STORE_ROOT),
            "generation_id": generation.generation_id,
        }
    for field, relative in derived_roots.items():
        redirected[field] = _path_text(scratch_root / relative)
    for field, expected in _SOURCE_ROOTS.items():
        _assert_equal_path(redirected[field], expected, field=field)
    if full_current_r88:
        _assert_equal_path(
            redirected["panel_data_root"],
            _DATAHUB_CLEAN_ROOT,
            field="clean-direct panel compatibility root",
        )
        _assert_equal_path(
            redirected["factor_data_root"],
            _full_current_generation_paths(experiment_generation_id).generation_root,
            field="staged canonical experiment generation factor_data_root",
        )
    for field, relative in derived_roots.items():
        expected = scratch_root / relative
        _assert_equal_path(redirected[field], expected, field=field)
        if not _is_strict_child(redirected[field], scratch_root):
            raise RuntimeError(f"derived root escaped scratch: {field}={redirected[field]}")
    return redirected


def _requested_range(start_text: str | None, end_text: str | None) -> tuple[date, date]:
    start = parse_date(start_text) if start_text else _APPROVED_START
    end = parse_date(end_text) if end_text else _APPROVED_END
    if start < _APPROVED_START or end > _APPROVED_END or end < start:
        raise ValueError(
            "R88 range must be contained in "
            f"{_APPROVED_START.isoformat()}..{_APPROVED_END.isoformat()}"
        )
    return start, end


def _requested_full_current_range(start_text: str | None, end_text: str | None) -> tuple[date, date]:
    """Resolve the clean-direct current R88 range."""

    start = parse_date(start_text) if start_text else _FULL_CURRENT_EXECUTABLE_START
    end = parse_date(end_text) if end_text else _FULL_CURRENT_EXECUTABLE_END
    if start < _FULL_CURRENT_EXECUTABLE_START or end > _FULL_CURRENT_EXECUTABLE_END or end < start:
        raise ValueError(
            "full-current R88 range must be contained in "
            f"{_FULL_CURRENT_EXECUTABLE_START.isoformat()}..{_FULL_CURRENT_EXECUTABLE_END.isoformat()}"
        )
    return start, end


def _factor_cfg(
    profile: Mapping[str, Any],
    payload: Sequence[Mapping[str, Any]],
    start: date,
    end: date,
    *,
    full_current_r88: bool = False,
) -> dict[str, Any]:
    """Build an in-memory research config; it never inherits a live pack."""

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "panel_name": "T1430",
        "factor_time": "14:30",
        "label_time": "14:42",
        "research_only": True,
        "refresh": False,
        "overwrite": False,
        "workers": 1,
        "factor_workers": 1,
        # Both R88 routes construct strict T1430 panels in memory from
        # DataHub clean snapshots.  This preserves the active live factor
        # source mode and never materialises a PanelStore parquet cache.
        "panel_source": {"mode": "clean_direct"},
        "compute": dict(profile["compute"]),
        "context": {"mode": "auto"},
        "backtest_enabled": False,
        "backtest": {"workers": 1, "batch_context": False},
        "screening": {"enabled": False, "copy_reports": False, "walk_forward_strategy": {"enabled": False}},
        "bad_factor_report": {"enabled": False},
        "factor_files": [],
        "factors": [dict(item) for item in payload],
        "disabled_factors_file": "factor/guards/factor_disabled_factors_empty.json",
        # Keep the profile evidence intact through the workflow boundary.  The
        # workflow will compare both fields to its static R88 allowlist before
        # importing any research metadata module.
        "research_factor_admission_profile": _PROFILE_ID,
        "research_factor_modules": list(profile["research_factor_modules"]),
        "research_factor_module_map": dict(profile["research_kernel_modules"]),
    }


def preflight(
    *,
    scratch_root_text: str,
    profile_path: str | Path = _PROFILE_PATH,
    start_text: str | None = None,
    end_text: str | None = None,
    rust_site_text: str | Path | None = None,
    execute: bool = False,
    verify_provenance: bool = True,
    full_current_r88: bool = False,
    experiment_generation_id: str | None = None,
) -> PreparedR88Backfill:
    """Prepare an R88 run without any derived output side effect.

    Rust admission deliberately occurs before DataHub profile resolution.  It
    keeps the current incomplete 38-factor state from turning into a partial
    Python/legacy-store build merely because a caller supplied a writable path.
    """

    path, profile = _load_profile(profile_path)
    payload, pending = _validate_profile_structure(profile)
    specs = _to_specs(payload)
    # The incomplete profile stops above this point.  Once it becomes
    # eligible, bind this process to a freshly built scratch wheel *before*
    # the Rust capability handshake, panel reads, or output creation.
    isolated_rust_site: Path | None = None
    if not pending:
        isolated_rust_site = _configure_isolated_rust_site(
            rust_site_text if rust_site_text is not None else os.environ.get(_R88_RUST_SITE_ENV)
        )
    _require_rust_admission(profile, specs, pending)
    frozen_r50_specs, r38_specs = _split_frozen_r50_and_r38_specs(specs)
    if isolated_rust_site is not None:
        _assert_loaded_isolated_rust_site(isolated_rust_site)
    # Both research paths use the same strict in-memory T1430 construction as
    # the active live factor runtime.  Validate the PIT boundary before any
    # DataHub access or scratch/output activity.
    panel_build_cfg = dict(load_config_file(_PANEL_CONFIG_NAME))
    panel_contract = _assert_strict_t1429_panel_contract(panel_build_cfg)
    if verify_provenance:
        _verify_provenance(profile)
    scratch_root = _assert_scratch_root(scratch_root_text, execute=execute)
    start, end = (
        _requested_full_current_range(start_text, end_text)
        if full_current_r88
        else _requested_range(start_text, end_text)
    )
    resolved_generation_id = None
    if full_current_r88:
        resolved_generation_id = _full_current_generation_paths(experiment_generation_id).generation_id
    paths_cfg = _redirect_derived_paths(
        _load_datahub_only_paths(),
        scratch_root,
        full_current_r88=full_current_r88,
        experiment_generation_id=resolved_generation_id,
    )
    cfg = _factor_cfg(profile, payload, start, end, full_current_r88=full_current_r88)
    cfg["panel_build_config"] = panel_build_cfg
    cfg["panel_contract"] = panel_contract
    return PreparedR88Backfill(
        profile_path=path,
        profile=profile,
        specs=specs,
        frozen_r50_specs=frozen_r50_specs,
        r38_specs=r38_specs,
        scratch_root=scratch_root,
        paths_cfg=paths_cfg,
        factor_cfg=cfg,
        start=start,
        end=end,
        experiment_generation_id=resolved_generation_id,
    )


def _preflight_summary(
    prepared: PreparedR88Backfill,
    *,
    execute: bool,
    full_current_r88: bool = False,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "research_only": True,
        "execute_requested": execute,
        "profile": {
            "path": prepared.profile_path.as_posix(),
            "admission_profile": prepared.profile["admission_profile"],
            "specs_sha256": prepared.profile["specs_sha256"],
        },
        "research_module_admission": {
            "modules": list(R88_RESEARCH_FACTOR_MODULES),
            "factor_key_to_module": dict(R88_RESEARCH_FACTOR_MODULE_MAP),
        },
        "factor_count": len(prepared.specs),
        "factors": [spec.name for spec in prepared.specs],
        "time_contract": prepared.profile["time_contract"],
        "range_contract": {"start": prepared.start.isoformat(), "end": prepared.end.isoformat()},
        "source_inputs": {field: prepared.paths_cfg[field] for field in _SOURCE_ROOTS},
        "clean_direct_panel_input": (
            prepared.paths_cfg["cleaned_data_root"] if full_current_r88 else None
        ),
        "canonical_factor_table": prepared.paths_cfg.get("factor_table") if full_current_r88 else None,
        "derived_write_roots": {
            field: prepared.paths_cfg[field]
            for field in (_FULL_CURRENT_DERIVED_ROOTS if full_current_r88 else _DERIVED_ROOTS)
        },
        "scratch_root_exists_before_execution": prepared.scratch_root.exists(),
        "reports_disabled": {"backtest": True, "screening": True, "bad_factor_report": True},
    }
    if full_current_r88:
        generation = _full_current_generation_paths(prepared.experiment_generation_id)
        summary.update(
            {
                "execution_mode": "full_current_rust88_recompute",
                "no_frozen_factorstore_read": True,
                "canonical_experiment_table": _CANONICAL_EXPERIMENT_TABLE_ROOT.as_posix(),
                "experiment_generation": {
                    "generation_id": generation.generation_id,
                    "root": generation.generation_root.as_posix(),
                    "activation": "not_requested",
                },
                "panel_source": "clean_direct_in_memory",
                "full_current_datahub_calendar_contract": {
                    "start": _FULL_CURRENT_EXECUTABLE_START.isoformat(),
                    "end": _FULL_CURRENT_EXECUTABLE_END.isoformat(),
                    "source": _FULL_CURRENT_CALENDAR_SOURCE,
                },
            }
        )
    else:
        summary.update(
            {
                "execution_mode": _R38_EXECUTION_MODE,
                "frozen_r50": {
                    "factor_root": _resolved(_FROZEN_R50_FACTOR_ROOT).as_posix(),
                    "factor_count": len(prepared.frozen_r50_specs),
                    "factors": [spec.name for spec in prepared.frozen_r50_specs],
                },
                "new_r38": {
                    "factor_root": (prepared.scratch_root / _R38_FACTOR_STORE_DIR).as_posix(),
                    "factor_count": len(prepared.r38_specs),
                    "factors": [spec.name for spec in prepared.r38_specs],
                },
            }
        )
    return summary


def _prepare_r38_context_loaders(
    *,
    prepared: PreparedR88Backfill,
    raw_data_root: Path,
) -> R38ContextLoaders:
    requirements = infer_factor_context_requirements(prepared.r38_specs)
    source_specs = resolve_daily_source_specs(requirements.daily_requirements)
    return R38ContextLoaders(
        daily_requirements=tuple(requirements.daily_requirements),
        daily_source_specs=source_specs,
        daily_source_indexes={
            source: index_daily_table(raw_data_root, source_spec.table)
            for source, source_spec in source_specs.items()
        },
        requires_map=bool(requirements.bond_stock_map_required),
        map_index=(
            _index_daily_table(raw_data_root, "market_cbond.daily_base")
            if requirements.bond_stock_map_required
            else None
        ),
    )


def _load_r38_daily_and_map_context(
    *,
    day: date,
    raw_data_root: Path,
    loaders: R38ContextLoaders,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Reuse normal daily/map loaders for the R38-only runner.

    The specialised optimisation applies solely to score-day panel construction;
    daily source lookbacks and map semantics remain the ordinary Rust-first
    context contract.
    """

    daily_data = load_daily_context_for_day(
        day,
        raw_data_root=raw_data_root,
        requirements=loaders.daily_requirements,
        source_specs=loaders.daily_source_specs,
        source_indexes=loaders.daily_source_indexes,
    )
    map_frame: pd.DataFrame | None = None
    if loaders.requires_map:
        map_frame = _read_bond_stock_map_day(
            day=day,
            raw_data_root=raw_data_root,
            table="market_cbond.daily_base",
            table_index=loaders.map_index,
        )
        if map_frame.empty:
            raise R88ContractError(f"R38 requires a non-empty bond-stock map on {day.isoformat()}")
    return daily_data, map_frame


def _mapped_stock_codes_for_r38(
    *,
    r50_index: pd.MultiIndex,
    day: date,
    daily_data: Mapping[str, pd.DataFrame],
    bond_stock_map: pd.DataFrame | None,
) -> set[str]:
    bond_codes = r50_index.get_level_values("code").astype(str).str.strip().str.upper()
    out = _strict_prior_base_stock_codes(
        daily_data.get("market_cbond.daily_base"),
        day=day,
        bond_codes=bond_codes,
    )
    if bond_stock_map is not None and not bond_stock_map.empty:
        map_frame = bond_stock_map.copy()
        if {"code", "stock_code"}.issubset(map_frame.columns):
            map_frame["_instrument_key"] = map_frame["code"].map(_instrument_key)
            needed = {_instrument_key(code) for code in bond_codes if _instrument_key(code)}
            out.update(
                _normalize_stock_code(value)
                for value in map_frame.loc[map_frame["_instrument_key"].isin(needed), "stock_code"]
                if _normalize_stock_code(value)
            )
    if not out:
        raise R88ContractError("R38 joint/ITR stock universe is empty after exact map union")
    return out


def _assert_frozen_r50_root() -> Path:
    root = _resolved(_FROZEN_R50_FACTOR_ROOT)
    if not _is_strict_child(root, _RESEARCH_SCRATCH_PARENT):
        raise R88ContractError("frozen R50 FactorStore must live strictly below research_scratch")
    if not root.is_dir():
        raise FileNotFoundError(f"frozen R50 FactorStore root is missing: {root.as_posix()}")
    return root


def _execute_r38_only_backfill(prepared: PreparedR88Backfill) -> tuple[Path, dict[str, Any]]:
    """Run only new R38 kernels and exactly concatenate them with frozen R50.

    This function is deliberately reachable only after `--execute`; it never
    writes to the frozen source or to live FactorStore paths.
    """

    frozen_root = _assert_frozen_r50_root()
    r50_columns = tuple(build_factor_col(spec) for spec in prepared.frozen_r50_specs)
    r38_columns = tuple(build_factor_col(spec) for spec in prepared.r38_specs)
    output_columns = tuple(build_factor_col(spec) for spec in prepared.specs)
    days = _iter_frozen_r50_days(frozen_root, start=prepared.start, end=prepared.end)
    raw_data_root = _resolved(prepared.paths_cfg["raw_data_root"])
    cleaned_data_root = _resolved(prepared.paths_cfg["cleaned_data_root"])
    if not raw_data_root.is_dir() or not cleaned_data_root.is_dir():
        raise RuntimeError("R38-only source DataHub roots are unavailable")

    prepare_factor_modules(prepared.factor_cfg)
    validate_rust_first_contracts(prepared.r38_specs)
    context_loaders = _prepare_r38_context_loaders(
        prepared=prepared,
        raw_data_root=raw_data_root,
    )

    r38_root = prepared.scratch_root / _R38_FACTOR_STORE_DIR
    final_root = _resolved(prepared.paths_cfg["factor_data_root"])
    result_root = _resolved(prepared.paths_cfg["results_root"]) / "r88_r38_only_backfill"
    for target in (r38_root, final_root, result_root):
        if target.exists():
            raise FileExistsError(f"R38-only execution requires fresh derived root: {target.as_posix()}")
        if not _is_strict_child(target, prepared.scratch_root):
            raise RuntimeError(f"R38-only derived root escaped scratch: {target.as_posix()}")

    prepared.scratch_root.mkdir(parents=True, exist_ok=False)
    r38_store = _factor_store_for(r38_root)
    final_store = _factor_store_for(final_root)
    rows: list[dict[str, Any]] = []
    total_days = len(days)
    for ordinal, day in enumerate(days, start=1):
        day_started = perf_counter()
        print(
            f"[r88-backfill] {ordinal}/{total_days} score_day={day.isoformat()} starting",
            flush=True,
        )
        r50, r50_provenance = _read_frozen_r50_day(
            frozen_root,
            day,
            expected_columns=r50_columns,
        )
        bond_panel = _strict_current_day_panel(
            _snapshot_day_path(cleaned_data_root, asset="cbond", day=day),
            day=day,
            eligible_index=r50.index,
            columns=_R38_BOND_SNAPSHOT_COLUMNS,
            require_all_codes=False,
            allow_empty=True,
        )
        observed_codes = set(bond_panel.index.get_level_values("code").astype(str))
        source_missing_codes = sorted(
            set(r50.index.get_level_values("code").astype(str)).difference(observed_codes)
        )
        observed_r50 = r50.loc[r50.index.get_level_values("code").isin(observed_codes)]
        stock_codes: set[str] = set()
        stock_panel_rows = 0
        if observed_r50.empty:
            r38 = pd.DataFrame(index=r50.index, columns=r38_columns, dtype="float64")
        else:
            daily_data, bond_stock_map = _load_r38_daily_and_map_context(
                day=day,
                raw_data_root=raw_data_root,
                loaders=context_loaders,
            )
            stock_codes = _mapped_stock_codes_for_r38(
                r50_index=observed_r50.index,
                day=day,
                daily_data=daily_data,
                bond_stock_map=bond_stock_map,
            )
            stock_panel = _strict_current_day_panel(
                _snapshot_day_path(cleaned_data_root, asset="stock", day=day),
                day=day,
                eligible_index=_stock_index_for_codes(observed_r50.index, sorted(stock_codes)),
                columns=_R38_STOCK_SNAPSHOT_COLUMNS,
                require_all_codes=False,
            )
            if stock_panel.empty:
                raise R88ContractError(f"R38 mapped stock panel is empty on {day.isoformat()}")
            stock_panel_rows = int(len(stock_panel))
            observed_r38 = build_factor_frame_rust(
                bond_panel,
                prepared.r38_specs,
                stock_panel=stock_panel,
                bond_stock_map=bond_stock_map,
                daily_data=daily_data,
                compute_backend_params={
                    "execution_policy": "rust_first",
                    "__compute_backend__": {"execution_policy": "rust_first", "backend": "cpu"},
                },
            )
            if not observed_r38.index.equals(observed_r50.index):
                raise R88ContractError(
                    "observed R38 output index must exactly equal observed frozen R50 index"
                )
            r38 = observed_r38.reindex(r50.index)
        merged = _merge_exact_r50_r38(
            r50,
            r38,
            day=day,
            r50_columns=r50_columns,
            r38_columns=r38_columns,
            output_columns=output_columns,
        )
        r38_path = _write_new_factor_day(r38_store, day, r38)
        final_path = _write_new_factor_day(final_store, day, merged)
        rows.append(
            {
                "day": day.isoformat(),
                "frozen_r50": r50_provenance,
                "bond_panel_rows": int(len(bond_panel)),
                "observed_bond_code_count": int(len(observed_codes)),
                "source_missing_bond_code_count": int(len(source_missing_codes)),
                "source_missing_bond_codes": source_missing_codes,
                "stock_panel_rows": stock_panel_rows,
                "mapped_stock_codes": int(len(stock_codes)),
                "r38_path": r38_path.as_posix(),
                "r38_sha256": _sha256(r38_path),
                "r38_rows": int(len(r38)),
                "r38_index_sha256": _frame_index_sha256(r38),
                "r88_path": final_path.as_posix(),
                "r88_sha256": _sha256(final_path),
                "r88_rows": int(len(merged)),
                "r88_index_sha256": _frame_index_sha256(merged),
            }
        )
        print(
            f"[r88-backfill] {ordinal}/{total_days} score_day={day.isoformat()} "
            f"completed rows={len(merged)} elapsed_seconds={perf_counter() - day_started:.2f}",
            flush=True,
        )

    result_root.mkdir(parents=True, exist_ok=False)
    execution = {
        "mode": _R38_EXECUTION_MODE,
        "frozen_r50_factor_root": frozen_root.as_posix(),
        "r38_factor_root": r38_root.as_posix(),
        "r88_factor_root": final_root.as_posix(),
        "r50_factor_count": len(r50_columns),
        "r38_factor_count": len(r38_columns),
        "r88_factor_count": len(output_columns),
        "days": rows,
    }
    (result_root / "r38_execution_manifest.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_root, execution


def _full_current_execution_ledger(
    *,
    prepared: PreparedR88Backfill,
    factor_store: CanonicalFactorTableWriter,
    expected_days: Sequence[date],
) -> tuple[list[dict[str, Any]], list[date]]:
    """Collect exact current-contract output evidence from canonical commits."""

    expected_columns = tuple(build_factor_col(spec) for spec in prepared.specs)
    rows: list[dict[str, Any]] = []
    days: list[date] = []
    for day in expected_days:
        frame = factor_store.read_day(day)
        if frame.empty:
            raise R88ContractError(f"canonical experiment output is absent on {day.isoformat()}")
        _require_factor_index(frame, day=day, field="full-current R88 output")
        actual_columns = tuple(str(column) for column in frame.columns)
        if actual_columns != expected_columns:
            raise R88ContractError(
                "full-current R88 output columns/order differ from the frozen profile on "
                f"{day.isoformat()}"
            )
        path = factor_store.day_path(day)
        if not path.is_file():
            raise FileNotFoundError(f"canonical experiment output path missing: {path.as_posix()}")
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        finite_mask = np.isfinite(values)
        finite = finite_mask.sum(axis=1)
        rows.append(
            {
                "day": day.isoformat(),
                "r88_path": path.as_posix(),
                "r88_sha256": _sha256(path),
                "r88_rows": int(len(frame)),
                "r88_index_sha256": _frame_index_sha256(frame),
                "min_available_factor_count": int(finite.min()) if len(finite) else 0,
                "median_available_factor_count": float(np.median(finite)) if len(finite) else 0.0,
                "max_available_factor_count": int(finite.max()) if len(finite) else 0,
                "mean_available_factor_count": float(finite.mean()) if len(finite) else 0.0,
                "eligible_66of88_row_count": int((finite >= 66).sum()),
                "inf_cell_count": int(np.isinf(values).sum()),
                "all_nan_columns": [
                    str(column) for column, has_finite in zip(numeric.columns, finite_mask.any(axis=0)) if not has_finite
                ],
            }
        )
        days.append(day)
    return rows, days


def _execute_full_current_r88_recompute(
    prepared: PreparedR88Backfill,
    *,
    workers: int,
    factor_workers: int,
) -> tuple[Path, dict[str, Any]]:
    """Compute all 88 profile contracts in a new Rust-first research root.

    Unlike the historical R38-only path, this mode does not read, merge, or
    modify the frozen R50 FactorStore. It establishes a new current-contract
    R88 data version and is therefore research-only by construction.
    """

    generation = _full_current_generation_paths(prepared.experiment_generation_id)
    factor_root = _resolved(prepared.paths_cfg["factor_data_root"])
    result_root = _resolved(prepared.paths_cfg["results_root"]) / "r88_full_current_recompute"
    _assert_equal_path(
        factor_root,
        generation.generation_root,
        field="full-current staged canonical experiment generation root",
    )
    _assert_equal_path(
        _resolved(prepared.paths_cfg["results_root"]),
        prepared.scratch_root / "results",
        field="full-current R88 results root",
    )
    if result_root.exists():
        raise FileExistsError(f"full-current R88 recompute requires fresh result root: {result_root.as_posix()}")
    if not _is_strict_child(result_root, prepared.scratch_root):
        raise RuntimeError(f"full-current R88 result root escaped scratch: {result_root.as_posix()}")

    raw_data_root = _resolved(prepared.paths_cfg["raw_data_root"])
    cleaned_data_root = _resolved(prepared.paths_cfg["cleaned_data_root"])
    panel_data_root = _resolved(prepared.paths_cfg["panel_data_root"])
    if not raw_data_root.is_dir() or not cleaned_data_root.is_dir():
        raise RuntimeError("full-current R88 recompute source DataHub roots are unavailable")
    _assert_equal_path(
        panel_data_root,
        cleaned_data_root,
        field="full-current clean-direct panel compatibility root",
    )
    panel_source = prepared.factor_cfg.get("panel_source")
    if not isinstance(panel_source, Mapping) or str(panel_source.get("mode", "")).strip().lower() != "clean_direct":
        raise R88ContractError("full-current R88 requires panel_source.mode='clean_direct'")
    expected_days, calendar_contract = _full_current_datahub_calendar(
        raw_data_root=raw_data_root,
        cleaned_data_root=cleaned_data_root,
        start=prepared.start,
        end=prepared.end,
    )
    # Verify published upstream evidence directly before creating a scratch
    # output root.  No persistent PanelStore is consulted by this route.
    direct_datahub = _verify_direct_datahub_evidence(expected_days=expected_days)
    raw_panel_build_cfg = prepared.factor_cfg.get("panel_build_config")
    if not isinstance(raw_panel_build_cfg, Mapping):
        raise R88ContractError("full-current R88 has no frozen clean-direct panel_build_config")
    panel_build_cfg = dict(raw_panel_build_cfg)
    panel_contract = _assert_strict_t1429_panel_contract(panel_build_cfg)

    prepare_factor_modules(prepared.factor_cfg)
    validate_rust_first_contracts(prepared.specs)
    contract = current_catalog_contract([spec.name for spec in prepared.specs])
    canonical = CanonicalFactorStore(_CANONICAL_FACTOR_STORE_ROOT)
    active_generation_before = canonical.active_experiment_generation_id()
    source_evidence = {
        "source": "r88_full_current_rust88_recompute",
        "experiment_generation_id": generation.generation_id,
        "active_generation_before": active_generation_before,
        "profile": {
            "path": "cbond_on/factor_contracts/profiles/research_r88_rust88_20260825.json5",
            "admission_profile": _PROFILE_ID,
            "profile_sha256": _sha256(prepared.profile_path),
            "specs_sha256": prepared.profile["specs_sha256"],
        },
        "time_contract": dict(prepared.profile["time_contract"]),
        "calendar_sha256": _days_sha256(expected_days),
        "clean_direct_panel": {
            "mode": "clean_direct",
            "cleaned_data_root": cleaned_data_root.as_posix(),
            "panel_config_sha256": _canonical_json_sha256(panel_build_cfg),
            "panel_contract": panel_contract,
            "datahub_evidence_sha256": _canonical_json_sha256(direct_datahub),
        },
    }
    # The generation is created only after every no-write input/contract check
    # above has succeeded.  It is deliberately not active, and no code in this
    # launcher calls ``activate_generation``.
    prepared.scratch_root.mkdir(parents=True, exist_ok=False)
    created_generation = canonical.create_stage_generation(
        generation.generation_id,
        source_evidence=source_evidence,
    )
    _assert_equal_path(
        created_generation.generation_root,
        factor_root,
        field="created R88 experiment generation root",
    )
    writer = CanonicalFactorTableWriter(
        _CANONICAL_FACTOR_STORE_ROOT,
        _CANONICAL_EXPERIMENT_TABLE_ID,
        contract,
        source_evidence,
        authority=issue_experiment_publisher_authority(
            root=_CANONICAL_FACTOR_STORE_ROOT,
            research_only=True,
            generation_id=generation.generation_id,
        ),
        generation_id=generation.generation_id,
    )
    started = perf_counter()
    result = run_factor_pipeline(
        panel_data_root,
        factor_root,
        prepared.start,
        prepared.end,
        panel_name="T1430",
        refresh=False,
        overwrite=False,
        workers=max(1, int(workers)),
        factor_workers=max(1, int(factor_workers)),
        raw_data_root=raw_data_root,
        # The factor pipeline builds one strict T1430 panel per day in memory
        # from the same DataHub clean snapshots used by active live factors.
        cleaned_data_root=cleaned_data_root,
        context_cfg=prepared.factor_cfg.get("context"),
        compute_cfg=prepared.factor_cfg.get("compute"),
        panel_source_cfg=prepared.factor_cfg.get("panel_source"),
        panel_build_cfg=panel_build_cfg,
        factor_store=writer,
        specs=prepared.specs,
    )
    rows, actual_days = _full_current_execution_ledger(
        prepared=prepared,
        factor_store=writer,
        expected_days=expected_days,
    )
    calendar_complete = actual_days == expected_days
    zero_eligible_days = [
        str(row.get("day", ""))
        for row in rows
        if int(row.get("eligible_66of88_row_count", 0)) <= 0
    ]
    all_nan_day_columns = {
        str(row.get("day", "")): list(row.get("all_nan_columns", []))
        for row in rows
        if list(row.get("all_nan_columns", []))
    }
    inf_rows = [str(row.get("day", "")) for row in rows if int(row.get("inf_cell_count", 0)) != 0]
    finalization_blockers: list[str] = []
    if not calendar_complete:
        finalization_blockers.append("execution calendar does not equal the verified DataHub calendar")
    if inf_rows:
        finalization_blockers.append("infinite factor cells on " + ", ".join(inf_rows[:8]))
    # Natural daily-lookback warm-up can yield zero 66-of-88 rows or all-NaN
    # factor columns near the beginning of a historical generation.  These are
    # immutable quality facts, not a reason to destroy a complete generation;
    # downstream rolling admission remains responsible for enforcing the 66/88
    # gate on its actual scoring range.
    quality_warnings = {
        "zero_eligible_66of88_days": zero_eligible_days,
        "all_nan_columns_by_day": all_nan_day_columns,
    }
    full_completion = not finalization_blockers
    generation_lifecycle: dict[str, Any]
    if full_completion:
        coverage = {
            "calendar_source": _FULL_CURRENT_CALENDAR_SOURCE,
            "score_day_start": expected_days[0].isoformat(),
            "score_day_end": expected_days[-1].isoformat(),
            "expected_day_count": len(expected_days),
            "expected_days_sha256": _days_sha256(expected_days),
            "factor_count": len(prepared.specs),
            "contract_list_sha256": contract.sha256,
            "panel_source": "clean_direct",
        }
        canonical.record_experiment_generation_attestation(
            generation.generation_id,
            {
                "migration_id": _FULL_CURRENT_GENERATION_MIGRATION_ID,
                "scope": "full",
                "expected_coverage": coverage,
                "source": "r88_full_current_rust88_recompute",
                "calendar_contract_sha256": _canonical_json_sha256(calendar_contract),
                "direct_datahub_evidence_sha256": _canonical_json_sha256(direct_datahub),
            },
        )
        canonical.mark_experiment_generation_verified(
            generation.generation_id,
            migration_id=_FULL_CURRENT_GENERATION_MIGRATION_ID,
            coverage=coverage,
        )
        canonical.finalize_stage_generation(
            generation.generation_id,
            expected_days=expected_days,
            verification={
                "schema_version": "r88_clean_direct_generation_verification/v1",
                "generation_id": generation.generation_id,
                "coverage": coverage,
                "execution_days_sha256": _days_sha256(actual_days),
                "no_frozen_factorstore_read": True,
                "activation": "not_requested",
                "quality_warnings": quality_warnings,
            },
        )
        active_generation_after = canonical.active_experiment_generation_id()
        if active_generation_after != active_generation_before:
            raise R88ContractError(
                "R88 generation execution must not change the active experiment generation pointer: "
                f"before={active_generation_before!r} after={active_generation_after!r}"
            )
        finalized_paths = canonical.experiment_generation_paths(generation.generation_id)
        generation_lifecycle = {
            "generation_id": generation.generation_id,
            "root": finalized_paths.generation_root.as_posix(),
            "manifest": finalized_paths.manifest_path.as_posix(),
            "done": finalized_paths.done_path.as_posix() if finalized_paths.done_path is not None else None,
            "status": "finalized_inactive",
            "active_generation_before": active_generation_before,
            "active_generation_after": active_generation_after,
            "quality_warnings": quality_warnings,
        }
    else:
        generation_lifecycle = {
            "generation_id": generation.generation_id,
            "root": factor_root.as_posix(),
            "status": "staging_quality_or_coverage_failed",
            "activation": "not_requested",
            "finalization_blockers": finalization_blockers,
            "quality_warnings": quality_warnings,
        }
    execution = {
        "mode": "full_current_rust88_recompute",
        "factor_root": factor_root.as_posix(),
        "factor_table": {
            "table_id": _CANONICAL_EXPERIMENT_TABLE_ID,
            "root": _CANONICAL_FACTOR_STORE_ROOT.as_posix(),
            "contract_list_sha256": contract.sha256,
            "generation_id": generation.generation_id,
        },
        "experiment_generation": generation_lifecycle,
        "factor_count": len(prepared.specs),
        "no_frozen_factorstore_read": True,
        "rust_execution": _loaded_rust_execution_evidence(),
        "workers": max(1, int(workers)),
        "factor_workers": max(1, int(factor_workers)),
        "pipeline_written": int(result.written),
        "pipeline_skipped": int(result.skipped),
        "elapsed_seconds": perf_counter() - started,
        "calendar_source": _FULL_CURRENT_CALENDAR_SOURCE,
        "calendar_contract": calendar_contract,
        "calendar_sha256": calendar_contract["days_sha256"],
        "clean_direct_panel": {
            "mode": "clean_direct",
            "cleaned_data_root": cleaned_data_root.as_posix(),
            "panel_config_sha256": _canonical_json_sha256(panel_build_cfg),
            "panel_contract": panel_contract,
            "datahub_evidence": direct_datahub,
        },
        "expected_calendar_days": [day.isoformat() for day in expected_days],
        "days": rows,
        "full_calendar_completion": bool(calendar_complete),
        "finalization_blockers": finalization_blockers,
        "quality_warnings": quality_warnings,
    }
    result_root.mkdir(parents=True, exist_ok=False)
    (result_root / "full_current_execution_manifest.json").write_text(
        json.dumps(execution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result_root, execution


def _assert_successful_output(out_root: str | Path, prepared: PreparedR88Backfill) -> Path:
    resolved = _resolved(out_root)
    expected_results = _resolved(prepared.paths_cfg["results_root"])
    if not _is_strict_child(resolved, expected_results) or not resolved.is_dir():
        raise RuntimeError("R88 R38-only runner returned no result under the approved scratch results root")
    factor_root = _resolved(prepared.paths_cfg["factor_data_root"])
    if prepared.paths_cfg.get("factor_table") is not None:
        generation = _full_current_generation_paths(prepared.experiment_generation_id)
        _assert_equal_path(
            factor_root,
            generation.generation_root,
            field="full-current staged canonical experiment generation root",
        )
    elif not factor_root.is_dir():
        raise RuntimeError("R88 R38-only runner did not produce the approved scratch FactorStore")
    return resolved


def _execution_day_list(execution: Mapping[str, Any] | None) -> tuple[list[date], list[str]]:
    """Parse the durable per-day execution ledger without accepting guesses."""

    if not isinstance(execution, Mapping):
        return [], ["execution ledger is missing"]
    rows = execution.get("days")
    if not isinstance(rows, list):
        return [], ["execution ledger days must be a list"]
    days: list[date] = []
    reasons: list[str] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            reasons.append(f"execution ledger row {position} is not an object")
            continue
        raw_day = row.get("day")
        try:
            parsed = date.fromisoformat(str(raw_day))
        except (TypeError, ValueError):
            reasons.append(f"execution ledger row {position} has invalid day={raw_day!r}")
            continue
        days.append(parsed)
    if len(set(days)) != len(days):
        reasons.append("execution ledger has duplicate score days")
    return days, reasons


def _verify_canonical_experiment_days(
    days: Sequence[date],
    *,
    generation_id: str,
) -> list[date]:
    """Read each finalized R88 generation day through canonical validation only."""

    store = CanonicalFactorStore(_CANONICAL_FACTOR_STORE_ROOT)
    store.require_experiment_generation_ready(generation_id)
    observed: list[date] = []
    for day in days:
        frame = store.read_day(
            _CANONICAL_EXPERIMENT_TABLE_ID,
            day,
            generation_id=generation_id,
        )
        if frame.empty:
            raise R88ContractError(f"canonical experiment generation yielded an empty R88 day: {day.isoformat()}")
        observed.append(day)
    return observed


def _completion_evidence(
    *,
    prepared: PreparedR88Backfill,
    execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prove whether this exact run is a full approved R88 backfill.

    A smoke or interval backfill is useful research evidence, but it is not a
    model-input admission.  The legacy route admits only an exact frozen-R50
    calendar.  The full-current route admits only its independently verified
    DataHub calendar; it never treats frozen R50 as an oracle.
    """

    approved_range = {
        "start": _APPROVED_START.isoformat(),
        "end": _APPROVED_END.isoformat(),
    }
    requested_range = {
        "start": prepared.start.isoformat(),
        "end": prepared.end.isoformat(),
    }
    declared_range_raw = prepared.profile.get("range_contract")
    declared_range = (
        {
            "start": str(declared_range_raw.get("start", "")).strip(),
            "end": str(declared_range_raw.get("end", "")).strip(),
        }
        if isinstance(declared_range_raw, Mapping)
        else None
    )
    reasons: list[str] = []
    if declared_range != approved_range:
        reasons.append("profile approved range does not equal the R88 approved range")
    mode = str((execution or {}).get("mode", "")).strip()
    if mode == "full_current_rust88_recompute":
        generation_id = str(
            ((execution or {}).get("factor_table") or {}).get("generation_id", "")
            if isinstance((execution or {}).get("factor_table"), Mapping)
            else ""
        ).strip()
        expected_generation_id = str(prepared.experiment_generation_id or "").strip()
        if not generation_id:
            reasons.append("full-current execution is missing experiment generation_id")
        elif generation_id != expected_generation_id:
            reasons.append("full-current execution generation_id differs from the prepared generation")
        current_range = {
            "start": _FULL_CURRENT_EXECUTABLE_START.isoformat(),
            "end": _FULL_CURRENT_EXECUTABLE_END.isoformat(),
        }
        if requested_range != current_range:
            reasons.append("requested range is not the full current-contract R88 executable range")
        expected_raw = (execution or {}).get("expected_calendar_days")
        try:
            expected_days = [date.fromisoformat(str(value)) for value in expected_raw or []]
        except (TypeError, ValueError):
            expected_days = []
            reasons.append("full-current execution has an invalid expected_calendar_days value")
        execution_days, execution_errors = _execution_day_list(execution)
        reasons.extend(execution_errors)
        if not bool((execution or {}).get("no_frozen_factorstore_read", False)):
            reasons.append("full-current execution did not certify no frozen FactorStore read")
        forbidden_frozen_fields = (
            "frozen_r50_factor_root",
            "r38_factor_root",
            "r88_factor_root",
            "r50_factor_count",
            "r38_factor_count",
        )
        present_frozen_fields = [field for field in forbidden_frozen_fields if field in (execution or {})]
        if present_frozen_fields:
            reasons.append(
                "full-current execution contains legacy frozen-R50 provenance fields: "
                + ", ".join(present_frozen_fields)
            )
        calendar_contract = (execution or {}).get("calendar_contract")
        if not isinstance(calendar_contract, Mapping):
            reasons.append("full-current execution is missing DataHub calendar_contract")
            calendar_days: list[date] = []
        else:
            try:
                calendar_days = [
                    date.fromisoformat(str(value))
                    for value in calendar_contract.get("raw_cbond_snapshot_days", [])
                ]
            except (TypeError, ValueError):
                calendar_days = []
                reasons.append("full-current calendar_contract has invalid raw_cbond_snapshot_days")
            for field in (
                "raw_stock_snapshot_days",
                "clean_cbond_snapshot_days",
                "clean_stock_snapshot_days",
            ):
                try:
                    values = [date.fromisoformat(str(value)) for value in calendar_contract.get(field, [])]
                except (TypeError, ValueError):
                    values = []
                    reasons.append(f"full-current calendar_contract has invalid {field}")
                if values != calendar_days:
                    reasons.append(f"full-current DataHub calendar mismatch for {field}")
            if str(calendar_contract.get("source", "")).strip() != _FULL_CURRENT_CALENDAR_SOURCE:
                reasons.append("full-current calendar_contract source is not the approved DataHub contract")
            if str(calendar_contract.get("days_sha256", "")).lower() != _days_sha256(calendar_days):
                reasons.append("full-current calendar_contract day hash mismatch")
        if expected_days != calendar_days:
            reasons.append("full-current execution expected calendar differs from DataHub calendar_contract")
        if (
            len(calendar_days) != _FULL_CURRENT_EXPECTED_DAY_COUNT
            or not calendar_days
            or calendar_days[0] != _FULL_CURRENT_EXECUTABLE_START
            or calendar_days[-1] != _FULL_CURRENT_EXECUTABLE_END
        ):
            reasons.append(
                "full-current DataHub calendar does not equal the frozen executable contract "
                f"{_FULL_CURRENT_EXECUTABLE_START.isoformat()}..{_FULL_CURRENT_EXECUTABLE_END.isoformat()} "
                f"({_FULL_CURRENT_EXPECTED_DAY_COUNT} days)"
            )
        if str((execution or {}).get("calendar_source", "")).strip() != _FULL_CURRENT_CALENDAR_SOURCE:
            reasons.append("full-current execution calendar_source is not the approved DataHub contract")
        if str((execution or {}).get("calendar_sha256", "")).lower() != _days_sha256(expected_days):
            reasons.append("full-current execution calendar_sha256 mismatch")
        clean_direct_panel = (execution or {}).get("clean_direct_panel")
        if not isinstance(clean_direct_panel, Mapping):
            reasons.append("full-current execution is missing clean-direct panel evidence")
            evidence_days: list[date] = []
        else:
            if str(clean_direct_panel.get("mode", "")).strip() != "clean_direct":
                reasons.append("full-current panel evidence mode is not clean_direct")
            expected_clean_root = _resolved(
                prepared.paths_cfg.get("cleaned_data_root", _DATAHUB_CLEAN_ROOT)
            )
            if _resolved(str(clean_direct_panel.get("cleaned_data_root", ""))) != expected_clean_root:
                reasons.append("full-current clean-direct panel evidence root drifted")
            contract = clean_direct_panel.get("panel_contract")
            if not isinstance(contract, Mapping):
                reasons.append("full-current clean-direct panel evidence lacks a panel contract")
            elif (
                contract.get("panel_name") != "T1430"
                or contract.get("physical_cutoff") != "14:29:00.999999"
                or int(contract.get("lead_minutes", -1)) != 1
            ):
                reasons.append("full-current clean-direct panel contract differs from strict T1429")
            evidence = clean_direct_panel.get("datahub_evidence")
            if not isinstance(evidence, Mapping):
                reasons.append("full-current clean-direct panel lacks DataHub publish evidence")
                evidence_days = []
            else:
                try:
                    evidence_days = [
                        date.fromisoformat(str(item.get("day", "")))
                        for item in evidence.get("days", [])
                    ]
                except (AttributeError, TypeError, ValueError):
                    evidence_days = []
                    reasons.append("full-current clean-direct DataHub evidence has invalid day rows")
                if str(evidence.get("mode", "")).strip() != "clean_direct":
                    reasons.append("full-current DataHub evidence mode is not clean_direct")
                if str(evidence.get("days_sha256", "")).lower() != _days_sha256(expected_days):
                    reasons.append("full-current clean-direct DataHub evidence day hash mismatch")
                if evidence_days != expected_days:
                    reasons.append("full-current clean-direct DataHub evidence days do not match the declared calendar")
        lifecycle = (execution or {}).get("experiment_generation")
        if not isinstance(lifecycle, Mapping):
            reasons.append("full-current execution is missing experiment generation lifecycle evidence")
        else:
            if lifecycle.get("generation_id") != generation_id:
                reasons.append("full-current lifecycle generation_id differs from factor-table evidence")
            if lifecycle.get("status") != "finalized_inactive":
                reasons.append("full-current experiment generation is not finalized inactive")
            if lifecycle.get("active_generation_before") != lifecycle.get("active_generation_after"):
                reasons.append("full-current execution changed the active experiment generation pointer")
        try:
            r88_store_days = _verify_canonical_experiment_days(
                expected_days,
                generation_id=generation_id,
            )
        except (OSError, R88ContractError, CanonicalFactorStoreError) as exc:
            r88_store_days = []
            reasons.append(f"cannot establish full-current canonical experiment coverage: {exc}")
        if not expected_days:
            reasons.append("full-current execution is missing expected_calendar_days")
        if execution_days != expected_days:
            reasons.append("full-current execution ledger days do not match the declared calendar")
        if r88_store_days != expected_days:
            reasons.append("full-current R88 FactorStore days do not match the declared calendar")
        if not bool((execution or {}).get("full_calendar_completion", False)):
            reasons.append("full-current execution did not certify calendar completion")
        return {
            "definition": "full_current_rust88_profile_recompute_exact_datahub_calendar",
            "approved_range": approved_range,
            "executable_range": current_range,
            "profile_range": declared_range,
            "requested_range": requested_range,
            "calendar_source": (execution or {}).get("calendar_source"),
            "panel_source": "clean_direct",
            "cleaned_data_root": (
                clean_direct_panel.get("cleaned_data_root")
                if isinstance(clean_direct_panel, Mapping)
                else None
            ),
            "expected_calendar_days": [day.isoformat() for day in expected_days],
            "execution_days": [day.isoformat() for day in execution_days],
            "r88_factor_store_days": [day.isoformat() for day in r88_store_days],
            "full_completion": not reasons,
            "blocking_reasons": reasons,
        }

    if requested_range != approved_range:
        reasons.append("requested range is not the full approved R88 range")

    execution_days, execution_errors = _execution_day_list(execution)
    reasons.extend(execution_errors)
    expected_frozen_days: list[date] = []
    r88_store_days: list[date] = []
    frozen_root_text: str | None = None
    try:
        frozen_root = _assert_frozen_r50_root()
        frozen_root_text = frozen_root.as_posix()
        expected_frozen_days = _iter_frozen_r50_days(
            frozen_root,
            start=_APPROVED_START,
            end=_APPROVED_END,
        )
    except (OSError, R88ContractError) as exc:
        reasons.append(f"cannot establish frozen R50 coverage: {exc}")

    if not isinstance(execution, Mapping):
        reasons.append("execution provenance is missing")
    else:
        recorded_frozen_root = str(execution.get("frozen_r50_factor_root", "")).strip()
        if not frozen_root_text or not recorded_frozen_root:
            reasons.append("execution frozen R50 root provenance is missing")
        elif _resolved(recorded_frozen_root) != _resolved(frozen_root_text):
            reasons.append("execution frozen R50 root provenance drifted")
        recorded_r88_root = str(execution.get("r88_factor_root", "")).strip()
        expected_r88_root = _resolved(prepared.paths_cfg["factor_data_root"])
        if not recorded_r88_root:
            reasons.append("execution R88 FactorStore root provenance is missing")
        elif _resolved(recorded_r88_root) != expected_r88_root:
            reasons.append("execution R88 FactorStore root provenance drifted")

    factor_root = _resolved(prepared.paths_cfg["factor_data_root"])
    try:
        r88_store_days = _iter_factor_store_days(
            factor_root,
            field="R88 scratch FactorStore",
        )
    except (OSError, R88ContractError) as exc:
        reasons.append(f"cannot establish durable R88 coverage: {exc}")

    if execution_days != expected_frozen_days:
        reasons.append("execution ledger days do not exactly equal frozen R50 approved-range days")
    if r88_store_days != expected_frozen_days:
        reasons.append("durable R88 FactorStore days do not exactly equal frozen R50 approved-range days")

    return {
        "definition": "requested_approved_range_and_exact_frozen_r50_day_coverage",
        "approved_range": approved_range,
        "profile_range": declared_range,
        "requested_range": requested_range,
        "frozen_r50_factor_root": frozen_root_text,
        "expected_frozen_r50_days": [day.isoformat() for day in expected_frozen_days],
        "execution_days": [day.isoformat() for day in execution_days],
        "r88_factor_store_days": [day.isoformat() for day in r88_store_days],
        "full_completion": not reasons,
        "blocking_reasons": reasons,
    }


def _write_backfill_manifest(
    *,
    prepared: PreparedR88Backfill,
    result_root: Path,
    execution: Mapping[str, Any] | None = None,
) -> Path:
    factor_root = _resolved(prepared.paths_cfg["factor_data_root"])
    execution_mode = str((execution or {}).get("mode", "")).strip()
    # Canonical table roots are intentionally allowed to contain only their
    # table commit layout.  The run-specific R88 ledger is research evidence,
    # so it belongs in the scratch result root rather than beside table data.
    manifest_path = (
        _resolved(result_root) / _BACKFILL_MANIFEST_NAME
        if execution_mode == "full_current_rust88_recompute"
        else factor_root / _BACKFILL_MANIFEST_NAME
    )
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite R88 backfill manifest: {manifest_path.as_posix()}")
    completion = _completion_evidence(prepared=prepared, execution=execution)
    full_completion = bool(completion["full_completion"])
    if execution_mode == "full_current_rust88_recompute":
        status = _FULL_CURRENT_RECOMPUTE_STATUS if full_completion else _PARTIAL_CURRENT_RECOMPUTE_STATUS
    else:
        status = _FULL_BACKFILL_STATUS if full_completion else _PARTIAL_BACKFILL_STATUS
    manifest = {
        "schema_version": _BACKFILL_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "execution_status": status,
        "model_training_ready": full_completion,
        "profile": {
            "path": "cbond_on/factor_contracts/profiles/research_r88_rust88_20260825.json5",
            "admission_profile": _PROFILE_ID,
            "specs_sha256": prepared.profile["specs_sha256"],
            "profile_sha256": _sha256(prepared.profile_path),
        },
        "research_module_admission": {
            "modules": list(R88_RESEARCH_FACTOR_MODULES),
            "factor_key_to_module": dict(R88_RESEARCH_FACTOR_MODULE_MAP),
        },
        "execution": dict(execution or {"mode": _R38_EXECUTION_MODE}),
        "completion": completion,
        "factor_root": factor_root.as_posix(),
        "factor_count": len(prepared.specs),
        "factors": [spec.name for spec in prepared.specs],
        "time_contract": dict(prepared.profile["time_contract"]),
        "range_contract": {"start": prepared.start.isoformat(), "end": prepared.end.isoformat()},
        "source_inputs": {field: prepared.paths_cfg[field] for field in _SOURCE_ROOTS},
        "result_root": result_root.as_posix(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True, help="fresh child of D:/cbond_on/research_scratch")
    parser.add_argument("--profile", default=str(_PROFILE_PATH), help="R88 research-only profile")
    parser.add_argument("--start", help="optional start, no earlier than 2024-01-01")
    parser.add_argument("--end", help="optional end; full-current clean-direct mode permits through 2026-08-27")
    parser.add_argument(
        "--rust-site",
        help="extracted isolated cbond_on_rust wheel site below D:/cbond_on/research_scratch",
    )
    parser.add_argument("--execute", action="store_true", help="run only after the Rust admission preflight")
    parser.add_argument(
        "--full-current-r88",
        action="store_true",
        help=(
            "research-only: recompute all 88 profile contracts from DataHub clean snapshots through "
            "an in-memory strict T1430 panel with the isolated current R88 Rust wheel; do not inherit "
            "the frozen R50 FactorStore"
        ),
    )
    parser.add_argument(
        "--experiment-generation",
        help=(
            "required for --execute --full-current-r88: a new inactive internal experiment generation ID; "
            f"default for no-write preflight is {_FULL_CURRENT_DEFAULT_GENERATION_ID!r}"
        ),
    )
    parser.add_argument("--workers", type=int, default=1, help="score-day workers for --full-current-r88")
    parser.add_argument(
        "--factor-workers",
        type=int,
        default=1,
        help="per-day factor workers for --full-current-r88",
    )
    parser.add_argument(
        "--skip-provenance-verify",
        action="store_true",
        help="test-only escape hatch; never use this for a research run",
    )
    args = parser.parse_args(argv)
    if args.execute and not args.full_current_r88:
        parser.error(
            "the legacy R38/frozen-R50 writer is migration/audit-only; "
            "normal R88 execution must use --full-current-r88 and the canonical experiment table"
        )
    if args.execute and args.full_current_r88 and not str(args.experiment_generation or "").strip():
        parser.error("--execute --full-current-r88 requires --experiment-generation; it must not overwrite active experiment data")
    prepared = preflight(
        scratch_root_text=args.scratch_root,
        profile_path=args.profile,
        start_text=args.start,
        end_text=args.end,
        rust_site_text=args.rust_site,
        execute=bool(args.execute),
        verify_provenance=not bool(args.skip_provenance_verify),
        full_current_r88=bool(args.full_current_r88),
        experiment_generation_id=args.experiment_generation,
    )
    print(
        json.dumps(
            _preflight_summary(
                prepared,
                execute=bool(args.execute),
                full_current_r88=bool(args.full_current_r88),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute:
        return 0
    if args.full_current_r88:
        result_root, execution = _execute_full_current_r88_recompute(
            prepared,
            workers=max(1, int(args.workers)),
            factor_workers=max(1, int(args.factor_workers)),
        )
    else:
        result_root, execution = _execute_r38_only_backfill(prepared)
    result_root = _assert_successful_output(result_root, prepared)
    manifest_path = _write_backfill_manifest(
        prepared=prepared,
        result_root=result_root,
        execution=execution,
    )
    print(json.dumps({"research_only": True, "manifest": manifest_path.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
