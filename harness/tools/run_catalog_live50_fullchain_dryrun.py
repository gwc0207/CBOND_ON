"""Run the catalog-admitted live50 chain in a fresh, no-DB scratch process.

This is an operational verifier, not a scheduler entry point.  It deliberately
does not modify a live config, a scheduler, or the production live50
FactorStore permit.  The parent process only prepares a fresh child of
``D:/cbond_on/research_scratch``.  The child process temporarily replaces two
references *inside its own* ``live_runtime`` module:

* ``load_config_file('live')`` returns a dynamically generated no-DB config;
* ``run_factor_build`` writes the exact catalog-admitted Rust-50 factor stage
  to the child FactorStore without asking for the production-store permit.

All other config reads continue through the normal loader and point to config
copies below the scratch root.  In particular, the normal live chain still
builds labels, scores all configured switch sources, selects a trade list, and
records timings, but cannot write a production DB or a production output root.

Examples
--------
Read-only plan (does not create the scratch root)::

    py harness/tools/run_catalog_live50_fullchain_dryrun.py ^
      --score-day 2026-08-25 ^
      --target-day 2026-08-26 ^
      --scratch-root D:/cbond_on/research_scratch/catalog_live50_fullchain_20260826_r1 ^
      --live-config live/live_panel_store_candidate_20260827 ^
      --preflight

Execute the same plan in a new child process::

    py harness/tools/run_catalog_live50_fullchain_dryrun.py ^
      --score-day 2026-08-25 ^
      --target-day 2026-08-26 ^
      --scratch-root D:/cbond_on/research_scratch/catalog_live50_fullchain_20260826_r1 ^
      --execute
"""

from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs  # noqa: E402
from cbond_on.core.config import load_config_file, parse_date, resolve_output_path  # noqa: E402
from cbond_on.core.trading_days import prev_trading_days_from_raw  # noqa: E402
from cbond_on.domain.factors.storage import FactorStore  # noqa: E402
from cbond_on.infra.factors.factor_table_resolution import build_factor_reader  # noqa: E402
from cbond_on.infra.factors.pipeline import run_factor_pipeline  # noqa: E402
from cbond_on.infra.live.factor_admission import (  # noqa: E402
    LIVE50_COLUMNS,
    LIVE50_RELEASE_ID,
    prepare_live50_factor_admission,
)


SCHEMA_VERSION = "catalog_live50_fullchain_dryrun/v1"
RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
DRYRUN_LIVE_CONFIG = "live/live_50_dryrun_20260805"
ACTIVE_LIVE_CONFIG = "live"
PLAN_FILE_NAME = "fullchain_plan.json"
RESULT_FILE_NAME = "fullchain_result.json"
MANIFEST_FILE_NAME = "fullchain_manifest.json"


class CatalogLive50FullchainDryrunError(RuntimeError):
    """Raised before a dry-run can escape its explicit scratch boundary."""


def _resolved(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _is_strict_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"JSON object required: {path}")
    return raw


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Copying recursive merge used only to construct a scratch config."""

    out = copy.deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(out.get(key), dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(dict(out[key]), dict(value))
        else:
            out[key] = copy.deepcopy(value)
    return out


def _platform_path(value: object, *, default_path: Path, results_root: str | Path) -> Path:
    return resolve_output_path(value, default_path=default_path, results_root=results_root)


def _validate_scratch_root(path: str | Path) -> Path:
    root = _resolved(path)
    parent = _resolved(RESEARCH_SCRATCH_PARENT)
    if not _is_strict_child(root, parent):
        raise CatalogLive50FullchainDryrunError(
            "full-chain dry-run scratch root must resolve strictly below "
            f"{parent.as_posix()}, got {root.as_posix()}"
        )
    if root.exists():
        raise FileExistsError(
            "full-chain dry-run requires a fresh scratch root; refusing to merge/reuse "
            f"{root.as_posix()}"
        )
    return root


def _validate_catalog_live50_factor_config(
    factor_cfg: Mapping[str, Any],
    *,
    source: str,
) -> None:
    specs = build_signal_specs(dict(factor_cfg))
    admission = prepare_live50_factor_admission(dict(factor_cfg), specs=specs)
    columns = tuple(str(spec.output_col or spec.name) for spec in specs)
    if admission is None or admission.release_id != LIVE50_RELEASE_ID or columns != LIVE50_COLUMNS:
        raise CatalogLive50FullchainDryrunError(
            f"{source} is not the exact catalog-admitted Rust-50 contract"
        )


def _load_active_live_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Read current live metadata only; no runtime configuration is applied."""

    active_live = dict(load_config_file(ACTIVE_LIVE_CONFIG))
    runtime = active_live.get("runtime")
    factor = active_live.get("factor")
    if not isinstance(runtime, Mapping) or not isinstance(factor, Mapping):
        raise CatalogLive50FullchainDryrunError("active live config requires runtime and factor objects")
    paths_ref = str(runtime.get("paths_config", "")).strip()
    factor_ref = str(factor.get("config", "")).strip()
    if not paths_ref or not factor_ref:
        raise CatalogLive50FullchainDryrunError("active live runtime.paths_config/factor.config must be non-empty")
    paths_cfg = dict(load_config_file(paths_ref))
    factor_cfg = dict(load_config_file(factor_ref))
    _validate_catalog_live50_factor_config(factor_cfg, source="active live factor config")
    return active_live, paths_cfg, factor_cfg, factor_ref


def _assert_no_db_write(live_cfg: Mapping[str, Any]) -> None:
    output = live_cfg.get("output")
    if not isinstance(output, Mapping) or bool(output.get("db_write", True)):
        raise CatalogLive50FullchainDryrunError("full-chain dry-run requires output.db_write=false")


def _validate_explicit_candidate_live_config(
    live_config_ref: str,
    live_cfg: Mapping[str, Any],
) -> str:
    """Accept a caller-selected no-DB candidate, never the active selector."""

    ref = str(live_config_ref or "").strip()
    if not ref or ref == ACTIVE_LIVE_CONFIG:
        raise CatalogLive50FullchainDryrunError(
            "--live-config must name a non-active candidate config"
        )
    candidate = live_cfg.get("candidate")
    if not isinstance(candidate, Mapping):
        raise CatalogLive50FullchainDryrunError(
            "--live-config requires an explicit candidate marker"
        )
    candidate_mode = str(candidate.get("mode", "")).strip()
    if candidate_mode == "disabled":
        raise CatalogLive50FullchainDryrunError(
            "--live-config names a disabled candidate; use the established clean-direct no-DB dry-run instead"
        )
    if candidate_mode not in {"published_panel_store", "canonical_live_factor_table"}:
        raise CatalogLive50FullchainDryrunError(
            "--live-config candidate.mode must be published_panel_store or canonical_live_factor_table"
        )
    if candidate.get("scheduler_selectable") is not False:
        raise CatalogLive50FullchainDryrunError(
            "--live-config candidate must set scheduler_selectable=false"
        )
    factor = live_cfg.get("factor")
    if not isinstance(factor, Mapping):
        raise CatalogLive50FullchainDryrunError("candidate live config.factor must be an object")
    factor_ref = str(factor.get("config", "")).strip()
    if not factor_ref:
        raise CatalogLive50FullchainDryrunError("candidate live config.factor.config must be non-empty")
    factor_cfg = dict(load_config_file(factor_ref))
    if candidate_mode == "published_panel_store":
        panel_store = factor_cfg.get("panel_store")
        panel_source = factor_cfg.get("panel_source")
        if (
            not isinstance(panel_store, Mapping)
            or str(panel_store.get("mode", "")).strip() != "published_read_only"
            or not isinstance(panel_source, Mapping)
            or str(panel_source.get("mode", "")).strip() != "cached_panel"
        ):
            raise CatalogLive50FullchainDryrunError(
                "published-panel candidate factor config must use published_read_only cached_panel"
            )
    _validate_catalog_live50_factor_config(factor_cfg, source="candidate live factor config")
    return factor_ref


def _collect_model_score_refs(live_cfg: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []

    def add(value: object) -> None:
        text = str(value or "").strip()
        if text and text not in refs:
            refs.append(text)

    model_score = live_cfg.get("model_score")
    if not isinstance(model_score, Mapping):
        raise CatalogLive50FullchainDryrunError("dry-run live config.model_score must be an object")
    add(model_score.get("config"))

    switch = live_cfg.get("model_switch")
    if not isinstance(switch, Mapping):
        return refs
    groups: list[object] = [switch.get("challenger")]
    raw_challengers = switch.get("challengers")
    if isinstance(raw_challengers, list):
        groups.extend(raw_challengers)
    for raw in groups:
        if not isinstance(raw, Mapping):
            continue
        add(raw.get("config"))
        sources = raw.get("sources")
        if isinstance(sources, list):
            for source in sources:
                if isinstance(source, Mapping):
                    add(source.get("config"))
    if not refs:
        raise CatalogLive50FullchainDryrunError("dry-run live config has no model-score configs")
    return refs


def _replace_model_score_refs(value: object, replacements: Mapping[str, str]) -> object:
    if isinstance(value, list):
        return [_replace_model_score_refs(item, replacements) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key == "config" and isinstance(item, str) and item in replacements:
            out[key] = replacements[item]
        else:
            out[key] = _replace_model_score_refs(item, replacements)
    return out


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _source_switch_path(active: Mapping[str, Any], *, dotted: str, fallback: object) -> object:
    """Use the current live input path when the dry-run shape has the same role.

    The dry-run model ids intentionally differ from production ids, but state
    and return histories are compatible source evidence.  If a newly added
    group has no active peer, its dry-run path remains the explicit fallback
    and normal preflight reports it as unavailable.
    """

    node: object = active
    try:
        for component in dotted.split("."):
            if component.endswith("]") and "[" in component:
                key, raw_index = component[:-1].split("[", 1)
                if key:
                    node = node[key]  # type: ignore[index]
                node = node[int(raw_index)]  # type: ignore[index]
            else:
                node = node[component]  # type: ignore[index]
    except (KeyError, IndexError, TypeError, ValueError):
        return fallback
    return node


def _rewrite_switch_paths(
    switch_cfg: Mapping[str, Any],
    *,
    active_switch_cfg: Mapping[str, Any],
    scratch_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Redirect all mutable model-switch paths and return their seed plan."""

    seed_requests: list[dict[str, str]] = []

    def walk(value: object, dotted: str) -> object:
        if isinstance(value, list):
            return [walk(item, f"{dotted}[{index}]") for index, item in enumerate(value)]
        if not isinstance(value, Mapping):
            return copy.deepcopy(value)
        out: dict[str, Any] = {}
        for key, raw in value.items():
            here = f"{dotted}.{key}" if dotted else key
            if key in {"state_feature_path", "return_path"}:
                source_raw = _source_switch_path(active_switch_cfg, dotted=here, fallback=raw)
                source = _platform_path(
                    source_raw,
                    default_path=Path("missing") / _safe_name(here),
                    results_root="D:/cbond_on/results",
                )
                suffix = source.suffix or ".csv"
                destination = scratch_root / "runtime" / "seed" / "model_switch" / f"{_safe_name(here)}{suffix}"
                out[key] = str(destination)
                seed_requests.append(
                    {"kind": key, "role": here, "source": str(source), "destination": str(destination)}
                )
            elif key == "score_output":
                model_id = str(value.get("model_id") or _safe_name(here)).strip()
                out[key] = str(scratch_root / "runtime" / "results" / "scores" / "live" / model_id)
            else:
                out[key] = walk(raw, here)
        return out

    return dict(walk(switch_cfg, "")), seed_requests


def _build_scratch_model_configs(
    *,
    dryrun_live_cfg: Mapping[str, Any],
    scratch_root: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], int]:
    """Materialize resolved score/model configs so all their writes stay scratch."""

    score_replacements: dict[str, str] = {}
    config_files: dict[str, dict[str, Any]] = {}
    max_window_days = 1
    for index, score_ref in enumerate(_collect_model_score_refs(dryrun_live_cfg), start=1):
        source_score_cfg = dict(load_config_file(score_ref))
        models = source_score_cfg.get("models")
        if not isinstance(models, Mapping) or len(models) != 1:
            raise CatalogLive50FullchainDryrunError(
                f"dry-run score config must have exactly one model: {score_ref}"
            )
        model_id, model_entry_raw = next(iter(models.items()))
        if not isinstance(model_entry_raw, Mapping):
            raise CatalogLive50FullchainDryrunError(f"invalid model entry in {score_ref}")
        model_entry = dict(model_entry_raw)
        source_model_ref = str(model_entry.get("model_config", "")).strip()
        if not source_model_ref:
            raise CatalogLive50FullchainDryrunError(f"score config missing model_config: {score_ref}")
        model_cfg = dict(load_config_file(source_model_ref))
        model_name = str(model_cfg.get("model_name") or model_id).strip()
        if not model_name:
            raise CatalogLive50FullchainDryrunError(f"model name missing: {source_model_ref}")
        max_window_days = max(
            max_window_days,
            int(dict(model_cfg.get("rolling", {})).get("window_days", 1) or 1),
        )
        model_cfg = _deep_merge(
            model_cfg,
            {
                "score_output": str(scratch_root / "runtime" / "results" / "scores" / "live" / model_name),
                "incremental": {
                    "state_dir": str(scratch_root / "runtime" / "results" / "model_state" / model_name),
                    "skip_existing_scores": False,
                },
                "neutralization_cache_root": str(
                    scratch_root / "runtime" / "neutralization_cache" / model_name
                ),
                "experiment": {"research_only": True},
            },
        )
        model_path = scratch_root / "config" / "models" / f"{index:02d}_{_safe_name(model_name)}.json"
        config_files[str(model_path)] = model_cfg
        model_entry["model_config"] = str(model_path)
        score_cfg = copy.deepcopy(source_score_cfg)
        score_cfg["models"] = {str(model_id): model_entry}
        score_cfg["model_id"] = str(model_id)
        score_cfg["default_model_id"] = str(model_id)
        score_path = scratch_root / "config" / "model_scores" / f"{index:02d}_{_safe_name(str(model_id))}.json"
        config_files[str(score_path)] = score_cfg
        score_replacements[score_ref] = str(score_path)
    return score_replacements, config_files, max_window_days


def _factor_seed_requests(
    *,
    source_paths: Mapping[str, Any],
    score_day: date,
    seed_days: int,
    scratch_root: Path,
) -> list[dict[str, str]]:
    raw_root = str(source_paths.get("raw_data_root", "")).strip()
    factor_root = str(source_paths.get("factor_data_root", "")).strip()
    label_root = str(source_paths.get("label_data_root", "")).strip()
    if not raw_root or not factor_root or not label_root:
        raise CatalogLive50FullchainDryrunError(
            "active paths profile must define raw_data_root, factor_data_root, and label_data_root"
        )
    history = prev_trading_days_from_raw(raw_root, score_day, max(1, int(seed_days)), kind="snapshot", asset="cbond")
    if len(history) < max(1, int(seed_days)):
        raise CatalogLive50FullchainDryrunError(
            f"insufficient trading history for dry-run seed: need={seed_days}, found={len(history)}"
        )
    requests: list[dict[str, str]] = []
    source_store = build_factor_reader(source_paths, panel_name="T1430")
    target_store = FactorStore(scratch_root / "runtime" / "factor_data", panel_name="T1430")
    for day in history:
        # Validate a canonical source bundle before its physical parquet is
        # copied into an isolated scratch replay. The legacy branch keeps the
        # historical preflight contract of checking only file existence/hash.
        if source_paths.get("factor_table") is not None:
            source_store.read_day(day)
        requests.append(
            {
                "kind": "factor_history",
                "role": day.isoformat(),
                "source": str(source_store.day_path(day)),
                "destination": str(target_store.day_path(day)),
            }
        )
        month = f"{day.year:04d}-{day.month:02d}"
        label = Path(label_root) / month / f"{day:%Y%m%d}.parquet"
        target = scratch_root / "runtime" / "label_data" / month / f"{day:%Y%m%d}.parquet"
        requests.append(
            {"kind": "label_history", "role": day.isoformat(), "source": str(label), "destination": str(target)}
        )
    return requests


def _source_regime_seed_requests(config_files: Mapping[str, Mapping[str, Any]], scratch_root: Path) -> list[dict[str, str]]:
    """Copy only mutable/needed model-side CSV inputs; leave immutable raw data read-only."""

    requests: list[dict[str, str]] = []
    for path_text, model_cfg in config_files.items():
        feature = model_cfg.get("feature_engineering")
        regime = dict(feature.get("regime", {})) if isinstance(feature, Mapping) else {}
        raw = regime.get("source_path")
        if raw in (None, ""):
            continue
        source = _platform_path(raw, default_path=Path("missing") / "regime.csv", results_root="D:/cbond_on/results")
        destination = scratch_root / "runtime" / "seed" / "regime" / f"{_safe_name(Path(path_text).stem)}{source.suffix or '.csv'}"
        # Rewrite the already materialized model config in place; it is scratch-only.
        feature_copy = dict(feature or {})
        regime_copy = dict(regime)
        regime_copy["source_path"] = str(destination)
        feature_copy["regime"] = regime_copy
        model_cfg["feature_engineering"] = feature_copy
        requests.append(
            {"kind": "regime_source", "role": Path(path_text).stem, "source": str(source), "destination": str(destination)}
        )
    return requests


def preflight(
    *,
    score_day: date | str,
    target_day: date | str,
    scratch_root: str | Path,
    live_config: str | None = None,
) -> dict[str, Any]:
    """Return the exact scratch plan without creating files or a child process."""

    root = _validate_scratch_root(scratch_root)
    score = parse_date(score_day)
    target = parse_date(target_day)
    active_live, active_paths, _factor_cfg, factor_ref = _load_active_live_inputs()
    selected_live_config = str(live_config or DRYRUN_LIVE_CONFIG).strip()
    dryrun_base = dict(load_config_file(selected_live_config))
    source_paths = active_paths
    if live_config is not None:
        factor_ref = _validate_explicit_candidate_live_config(
            selected_live_config,
            dryrun_base,
        )
        candidate_runtime = dryrun_base.get("runtime")
        if not isinstance(candidate_runtime, Mapping):
            raise CatalogLive50FullchainDryrunError("candidate live config.runtime must be an object")
        candidate_paths_ref = str(candidate_runtime.get("paths_config", "")).strip()
        if not candidate_paths_ref:
            raise CatalogLive50FullchainDryrunError("candidate live config.runtime.paths_config is required")
        source_paths = dict(load_config_file(candidate_paths_ref))
    _assert_no_db_write(dryrun_base)
    source_paths_ref = (
        str(dict(dryrun_base.get("runtime", {})).get("paths_config", "")).strip()
        if live_config is not None
        else str(dict(active_live.get("runtime", {})).get("paths_config", "")).strip()
    )
    scratch_paths_path = root / "config" / "paths_config.json"
    runtime_root = root / "runtime"
    results_root = runtime_root / "results"
    # This explicit audit-only profile is consumed only by the isolated child
    # below. It retains a temporary FactorStore solely for no-DB replay; every
    # normal factor reader/writer remains bound to the three canonical tables.
    scratch_paths = {
        "lifecycle": {
            "status": "audit_only",
            "reason": "no_db_ephemeral_factor_stage",
            "normal_consumer": False,
        },
        "raw_data_root": str(source_paths["raw_data_root"]),
        "clean_data_root": str(source_paths.get("cleaned_data_root") or source_paths.get("clean_data_root")),
        "cleaned_data_root": str(source_paths.get("cleaned_data_root") or source_paths.get("clean_data_root")),
        "panel_data_root": str(source_paths["panel_data_root"]),
        "label_data_root": str(runtime_root / "label_data"),
        "factor_data_root": str(runtime_root / "factor_data"),
        "ads_root": str(runtime_root / "ads"),
        "results_root": str(results_root),
        "model_root": str(results_root / "models"),
        "score_root": str(results_root / "scores"),
        "logs_root": str(runtime_root / "logs"),
        "read_only_input_roots": {
            "panel_data_root": str(source_paths["panel_data_root"]),
            "label_data_root": str(runtime_root / "label_data"),
            "factor_data_root": str(runtime_root / "factor_data"),
        },
    }
    score_replacements, config_files, window_days = _build_scratch_model_configs(
        dryrun_live_cfg=dryrun_base,
        scratch_root=root,
    )
    active_switch = active_live.get("model_switch")
    active_switch = dict(active_switch) if isinstance(active_switch, Mapping) else {}
    dryrun_switch = dryrun_base.get("model_switch")
    dryrun_switch = dict(dryrun_switch) if isinstance(dryrun_switch, Mapping) else {}
    rewritten_switch, switch_seeds = _rewrite_switch_paths(
        dryrun_switch,
        active_switch_cfg=active_switch,
        scratch_root=root,
    )
    dynamic_live = _deep_merge(
        dryrun_base,
        {
            "runtime": {
                "paths_config": str(scratch_paths_path),
                "factor_route_mode": "no_db_ephemeral_factor_stage",
            },
            "factor": {"config": factor_ref},
            "model_switch": rewritten_switch,
            "output": {"db_write": False, "db_table": None, "db_backend": None},
        },
    )
    dynamic_live = dict(_replace_model_score_refs(dynamic_live, score_replacements))
    _assert_no_db_write(dynamic_live)
    regime_seeds = _source_regime_seed_requests(config_files, root)
    seeds = [
        *_factor_seed_requests(
            source_paths=source_paths,
            score_day=score,
            seed_days=window_days,
            scratch_root=root,
        ),
        *switch_seeds,
        *regime_seeds,
    ]
    production_factor_root = _resolved(str(source_paths["factor_data_root"]))
    scratch_factor_root = _resolved(str(scratch_paths["read_only_input_roots"]["factor_data_root"]))
    if scratch_factor_root == production_factor_root:
        raise CatalogLive50FullchainDryrunError("scratch factor root must not equal the production live50 FactorStore")
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "preflight",
        "research_only": True,
        "score_day": score.isoformat(),
        "target_day": target.isoformat(),
        "scratch_root": str(root),
        "admission": {"release_id": LIVE50_RELEASE_ID, "factor_count": len(LIVE50_COLUMNS), "columns": list(LIVE50_COLUMNS)},
        "source": {
            "active_live_config": ACTIVE_LIVE_CONFIG,
            "active_paths_config": source_paths_ref,
            "dryrun_live_config": selected_live_config,
            "factor_config": factor_ref,
            "production_factor_root": str(production_factor_root),
        },
        "scratch_paths": scratch_paths,
        "dynamic_live_config": dynamic_live,
        "dynamic_config_files": config_files,
        "seed_requests": seeds,
        "seed_history_days": int(window_days),
        "side_effect_boundary": {
            "production_live_config": "read_only",
            "scheduler": "not_called",
            "production_permit": "not_modified",
            "database_write": False,
            "child_patches": ["live_runtime.load_config_file('live')", "live_runtime.run_factor_build"],
            "all_writes_below": str(root),
        },
    }


def _validate_seed_request(request: Mapping[str, Any]) -> tuple[Path, Path]:
    source = _resolved(str(request.get("source", "")))
    destination = _resolved(str(request.get("destination", "")))
    root = _resolved(str(request.get("scratch_root", ""))) if request.get("scratch_root") else None
    if root is not None and not _is_strict_child(destination, root):
        raise CatalogLive50FullchainDryrunError(f"seed destination escapes scratch root: {destination}")
    if not source.is_file():
        raise FileNotFoundError(f"required dry-run seed is missing: {source}")
    return source, destination


def _seed_inputs(plan: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_requests = plan.get("seed_requests")
    if not isinstance(raw_requests, list):
        raise CatalogLive50FullchainDryrunError("plan seed_requests must be a list")
    for raw in raw_requests:
        if not isinstance(raw, Mapping):
            raise CatalogLive50FullchainDryrunError("plan seed request must be an object")
        request = {**dict(raw), "scratch_root": str(root)}
        source, destination = _validate_seed_request(request)
        if not _is_strict_child(destination, root):
            raise CatalogLive50FullchainDryrunError(f"seed destination escapes scratch root: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = _sha256_file(source)
        destination_hash = _sha256_file(destination)
        if source_hash != destination_hash:
            raise CatalogLive50FullchainDryrunError(f"seed hash mismatch: {source} -> {destination}")
        records.append(
            {
                "kind": raw.get("kind"),
                "role": raw.get("role"),
                "source": str(source),
                "destination": str(destination),
                "bytes": int(destination.stat().st_size),
                "sha256": destination_hash,
            }
        )
    return records


def _write_dynamic_config_files(plan: Mapping[str, Any], *, root: Path) -> list[dict[str, Any]]:
    config_files = plan.get("dynamic_config_files")
    if not isinstance(config_files, Mapping):
        raise CatalogLive50FullchainDryrunError("plan dynamic_config_files must be an object")
    records: list[dict[str, Any]] = []
    for path_text, payload in config_files.items():
        path = _resolved(str(path_text))
        if not _is_strict_child(path, root):
            raise CatalogLive50FullchainDryrunError(f"dynamic config escapes scratch root: {path}")
        if not isinstance(payload, Mapping):
            raise CatalogLive50FullchainDryrunError(f"dynamic config payload must be an object: {path}")
        _write_json(path, dict(payload))
        records.append({"path": str(path), "sha256": _sha256_file(path)})
    paths_payload = plan.get("scratch_paths")
    if not isinstance(paths_payload, Mapping):
        raise CatalogLive50FullchainDryrunError("plan scratch_paths must be an object")
    paths_path = root / "config" / "paths_config.json"
    _write_json(paths_path, dict(paths_payload))
    records.append({"path": str(paths_path), "sha256": _sha256_file(paths_path)})
    return records


def _run_scratch_factor_build(*, scratch_root: Path, **kwargs: Any) -> dict[str, Any]:
    """Child-only factor-build replacement; it never calls the production permit.

    ``run_factor_pipeline`` still enforces Rust-first and the immutable catalog
    admission.  Its regular write guard accepts a non-production root with no
    permit, so this wrapper cannot mint or alter a production capability.
    """

    cfg_raw = kwargs.get("cfg")
    if not isinstance(cfg_raw, Mapping):
        raise CatalogLive50FullchainDryrunError("child factor build requires a factor config")
    factor_cfg = dict(cfg_raw)
    specs = build_signal_specs(factor_cfg)
    admission = prepare_live50_factor_admission(factor_cfg, specs=specs)
    columns = tuple(str(spec.output_col or spec.name) for spec in specs)
    if admission is None or admission.release_id != LIVE50_RELEASE_ID or columns != LIVE50_COLUMNS:
        raise CatalogLive50FullchainDryrunError("child factor build refused a non-catalog live50 contract")
    paths_cfg = dict(load_config_file("paths"))
    factor_root = _resolved(str(paths_cfg["factor_data_root"]))
    if not _is_strict_child(factor_root, scratch_root):
        raise CatalogLive50FullchainDryrunError("child factor build factor root is not scratch-only")
    start = parse_date(kwargs.get("start") or factor_cfg.get("start"))
    end = parse_date(kwargs.get("end") or factor_cfg.get("end"))
    panel_cfg = dict(load_config_file("panel"))
    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise CatalogLive50FullchainDryrunError("catalog live50 factor config needs panel_name")
    result = run_factor_pipeline(
        paths_cfg["panel_data_root"],
        factor_root,
        start,
        end,
        panel_name=panel_name,
        refresh=bool(kwargs.get("refresh", factor_cfg.get("refresh", False))),
        overwrite=bool(kwargs.get("overwrite", factor_cfg.get("overwrite", False))),
        workers=int(factor_cfg.get("workers", 1)),
        factor_workers=int(factor_cfg.get("factor_workers", 1)),
        raw_data_root=paths_cfg.get("raw_data_root"),
        cleaned_data_root=paths_cfg.get("cleaned_data_root") or paths_cfg.get("clean_data_root"),
        context_cfg=factor_cfg.get("context"),
        compute_cfg=factor_cfg.get("compute"),
        panel_source_cfg=factor_cfg.get("panel_source"),
        panel_build_cfg=panel_cfg,
        # No production permit: this is an explicitly bound scratch-only
        # no-DB staging leaf, never a normal factor-result table.
        allow_ephemeral_factor_store=True,
        specs=specs,
    )
    store = FactorStore(factor_root, panel_name=panel_name)
    for day in (start, end):
        frame = store.read_day(day)
        if frame.empty or tuple(str(column) for column in frame.columns) != LIVE50_COLUMNS:
            raise CatalogLive50FullchainDryrunError("scratch factor output differs from ordered catalog live50")
    return {"start": start, "end": end, "written": int(result.written), "skipped": int(result.skipped)}


def _child_execute(plan_path: str | Path) -> int:
    """Execute inside a new Python process; no global process state is reused."""

    plan = _read_json(_resolved(plan_path))
    root = _resolved(str(plan["scratch_root"]))
    if not _is_strict_child(root, _resolved(RESEARCH_SCRATCH_PARENT)) or not root.exists():
        raise CatalogLive50FullchainDryrunError("child plan is not under a created research scratch root")
    live_cfg = plan.get("dynamic_live_config")
    if not isinstance(live_cfg, Mapping):
        raise CatalogLive50FullchainDryrunError("child plan dynamic_live_config must be an object")
    _assert_no_db_write(live_cfg)

    # An inherited scheduler profile must never override the child scratch file.
    for env_name in ("CBOND_ON_PATHS_CONFIG", "CBOND_ON_RUNTIME_ROOT", "CBOND_ON_RAW_ROOT", "CBOND_ON_CLEAN_ROOT"):
        os.environ.pop(env_name, None)
    prior_ephemeral_gate = os.environ.get("CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE")
    os.environ["CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE"] = "1"

    from cbond_on.app.usecases import live_runtime

    original_load = live_runtime.load_config_file
    original_factor_build = live_runtime.run_factor_build
    stage_stamps: list[tuple[str, float]] = []
    started = perf_counter()

    def child_load_config_file(name: str | Path) -> dict[str, Any]:
        # This is intentionally the only config-loader patch.  Every nested
        # model config is a normal, absolute scratch JSON file.
        if str(name).replace("\\", "/").strip() == "live":
            return copy.deepcopy(dict(live_cfg))
        return original_load(name)

    def stage_reporter(stage: str) -> None:
        stage_stamps.append((str(stage), perf_counter()))

    live_runtime.load_config_file = child_load_config_file
    live_runtime.run_factor_build = lambda **kwargs: _run_scratch_factor_build(scratch_root=root, **kwargs)
    try:
        output_dir = live_runtime.run_once(
            start=str(plan["score_day"]),
            target=str(plan["target_day"]),
            mode="catalog_live50_fullchain_dryrun",
            stage_reporter=stage_reporter,
        )
        output_dir = _resolved(output_dir)
        if not _is_strict_child(output_dir, root):
            raise CatalogLive50FullchainDryrunError("child live runtime output escaped scratch root")
        trade_list = output_dir / "trade_list.csv"
        if not trade_list.is_file():
            raise CatalogLive50FullchainDryrunError("child live runtime did not write a scratch trade_list.csv")
        finished = perf_counter()
        stage_seconds: dict[str, float] = {}
        timeline = [("start", started), *stage_stamps, ("finish", finished)]
        for (name, begin), (_, end) in zip(timeline, timeline[1:]):
            stage_seconds[name] = round(max(0.0, end - begin), 6)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "research_only": True,
            "database_write": False,
            "score_day": plan["score_day"],
            "target_day": plan["target_day"],
            "output_dir": str(output_dir),
            "trade_list": str(trade_list),
            "trade_list_sha256": _sha256_file(trade_list),
            "wall_seconds": round(finished - started, 6),
            "stage_seconds": stage_seconds,
            "stages": [name for name, _ in stage_stamps],
            "child_patches": ["live_runtime.load_config_file('live')", "live_runtime.run_factor_build"],
        }
        _write_json(root / RESULT_FILE_NAME, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        live_runtime.load_config_file = original_load
        live_runtime.run_factor_build = original_factor_build
        if prior_ephemeral_gate is None:
            os.environ.pop("CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE", None)
        else:
            os.environ["CBOND_ON_ALLOW_NO_DB_AUDIT_FACTORSTORE"] = prior_ephemeral_gate


def execute(
    *,
    score_day: date | str,
    target_day: date | str,
    scratch_root: str | Path,
    live_config: str | None = None,
) -> dict[str, Any]:
    """Seed a fresh scratch root and invoke the full chain in a child process."""

    plan = preflight(
        score_day=score_day,
        target_day=target_day,
        scratch_root=scratch_root,
        live_config=live_config,
    )
    root = _resolved(str(plan["scratch_root"]))
    root.mkdir(parents=True, exist_ok=False)
    config_records = _write_dynamic_config_files(plan, root=root)
    seed_records: list[dict[str, Any]] = []
    process: subprocess.CompletedProcess[str] | None = None
    try:
        seed_records = _seed_inputs(plan, root=root)
        executable_plan = dict(plan)
        executable_plan["mode"] = "execute"
        executable_plan["prepared_at_utc"] = datetime.now(timezone.utc).isoformat()
        executable_plan["dynamic_config_records"] = config_records
        executable_plan["seed_records"] = seed_records
        plan_path = root / PLAN_FILE_NAME
        _write_json(plan_path, executable_plan)
        process = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--_child-plan", str(plan_path)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        result_path = root / RESULT_FILE_NAME
        child_result = _read_json(result_path) if result_path.is_file() else {}
        manifest = {
            **executable_plan,
            "status": "completed" if process.returncode == 0 and child_result.get("status") == "completed" else "failed",
            "child": {
                "returncode": int(process.returncode),
                "stdout": process.stdout,
                "stderr": process.stderr,
                "result": child_result,
            },
        }
        _write_json(root / MANIFEST_FILE_NAME, manifest)
        if manifest["status"] != "completed":
            raise CatalogLive50FullchainDryrunError(
                f"full-chain child failed: returncode={process.returncode}; manifest={root / MANIFEST_FILE_NAME}"
            )
        return manifest
    except Exception as exc:
        failure = {
            **plan,
            "mode": "execute",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "dynamic_config_records": config_records,
            "seed_records": seed_records,
        }
        if process is not None:
            failure["child"] = {"returncode": int(process.returncode), "stdout": process.stdout, "stderr": process.stderr}
        _write_json(root / MANIFEST_FILE_NAME, failure)
        raise


def compact_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    child = manifest.get("child") if isinstance(manifest.get("child"), Mapping) else {}
    result = child.get("result") if isinstance(child.get("result"), Mapping) else {}
    admission = manifest.get("admission") if isinstance(manifest.get("admission"), Mapping) else {}
    return {
        "status": manifest.get("status", "preflight_ready"),
        "mode": manifest.get("mode"),
        "research_only": True,
        "score_day": manifest.get("score_day"),
        "target_day": manifest.get("target_day"),
        "release_id": admission.get("release_id"),
        "factor_count": admission.get("factor_count"),
        "seed_files": len(manifest.get("seed_requests", [])),
        "scratch_root": manifest.get("scratch_root"),
        "wall_seconds": result.get("wall_seconds"),
        "stage_seconds": result.get("stage_seconds", {}),
        "manifest_path": str(_resolved(str(manifest.get("scratch_root", "."))) / MANIFEST_FILE_NAME)
        if manifest.get("mode") == "execute"
        else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-day", help="score day in YYYY-MM-DD")
    parser.add_argument("--target-day", help="target/trade-list day in YYYY-MM-DD")
    parser.add_argument("--scratch-root", help="fresh child of D:/cbond_on/research_scratch")
    parser.add_argument(
        "--live-config",
        help=(
            "explicit non-active candidate live config; default keeps the established "
            "live50 dry-run configuration"
        ),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight", action="store_true", help="validate and print a no-write plan")
    actions.add_argument("--execute", action="store_true", help="seed scratch and run a child full-chain dry-run")
    parser.add_argument("--print-full-manifest", action="store_true")
    parser.add_argument("--_child-plan", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._child_plan:
        return _child_execute(args._child_plan)
    if not args.score_day or not args.target_day or not args.scratch_root:
        parser.error("--score-day, --target-day, and --scratch-root are required")
    if not args.preflight and not args.execute:
        parser.error("choose exactly one of --preflight or --execute")
    manifest = (
        preflight(
            score_day=args.score_day,
            target_day=args.target_day,
            scratch_root=args.scratch_root,
            live_config=args.live_config,
        )
        if args.preflight
        else execute(
            score_day=args.score_day,
            target_day=args.target_day,
            scratch_root=args.scratch_root,
            live_config=args.live_config,
        )
    )
    print(json.dumps(manifest if args.print_full_manifest else compact_summary(manifest), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
