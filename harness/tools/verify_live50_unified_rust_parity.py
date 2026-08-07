"""Scratch-only verifier for the unified live-50 Rust public API.

This tool deliberately does *not* invoke ``run_factor_pipeline`` or
``run_once``.  Those normal production entrypoints persist a FactorStore and
other live-chain artifacts.  Instead it:

1. loads the real score-day panel/context through the same read-only loaders;
2. freezes those in-memory inputs in a newly-created research-scratch folder;
3. runs the historical 27-Rust + 23-Python hybrid in one child interpreter;
4. runs the candidate unified public ``compute_factor_frame`` API from a
   separately installed scratch wheel in another child interpreter; and
5. compares both frames and the persisted hybrid golden output exactly.

The two child interpreters are intentional.  On Windows a loaded ``.pyd``
cannot be reliably unloaded/replaced inside one interpreter, so a one-process
comparison could silently exercise the wrong extension binary.

All generated files remain below ``D:/cbond_on/research_scratch``.  The
verifier never imports the scheduler, never connects to the trade DB, and
never calls a live output writer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SELF = Path(__file__).resolve()
_RESEARCH_ROOT = Path(r"D:/cbond_on/research_scratch").resolve()
_DEFAULT_GOLDEN = Path(
    r"D:/cbond_on/factor_data_live50_20260805/factors/T1430/2026-08/20260806.parquet"
)
_DEFAULT_LIVE_OUTPUT = Path(r"D:/cbond_on/results/live/2026-08-07/trade_list.csv")
_DEFAULT_SCHEDULER_STATE = Path(r"D:/cbond_on/results/live/scheduler/state.json")
_CONFIG_ENV_VARS = (
    "CBOND_ON_PATHS_CONFIG",
    "CBOND_ON_PATHS_PROFILE",
    "CBOND_ON_RUNTIME_ROOT",
    "CBOND_ON_RAW_ROOT",
    "CBOND_ON_CLEAN_ROOT",
)


def _parse_day(value: str) -> date:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise argparse.ArgumentTypeError(f"invalid date: {value!r}")
    return parsed.date()


def _normal_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validate_scratch_root(path: Path, *, must_not_exist: bool = False) -> Path:
    root = _normal_path(path)
    if root == _RESEARCH_ROOT or not _is_below(root, _RESEARCH_ROOT):
        raise ValueError(
            "--scratch-root must be a new child of "
            f"{_RESEARCH_ROOT}; got {root}"
        )
    if must_not_exist and root.exists():
        raise FileExistsError(
            f"scratch root already exists (refusing overwrite): {root}"
        )
    return root


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _fingerprint(path: Path) -> dict[str, Any]:
    path = _normal_path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "is_file": path.is_file(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if path.is_file():
        payload["sha256"] = _sha256_file(path)
    return payload


def _repo_extension_paths() -> list[Path]:
    package = _REPO_ROOT / "cbond_on_rust"
    return sorted(package.glob("cbond_on_rust*.pyd"))


def _default_protected_paths(golden: Path) -> list[Path]:
    return [
        *_repo_extension_paths(),
        golden,
        _DEFAULT_LIVE_OUTPUT,
        _DEFAULT_SCHEDULER_STATE,
    ]


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = _normal_path(path)
        key = os.path.normcase(str(resolved))
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def _fingerprint_paths(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {str(path): _fingerprint(path) for path in _dedupe_paths(paths)}


def _scheduler_state_semantic_projection(path: Path) -> dict[str, Any]:
    """Fingerprint scheduler state while preserving its documented heartbeat.

    The normal scheduler rewrites only ``heartbeat`` while idle.  A long
    read-only parity replay must still fail if *any* operational state changes,
    but treating that volatile liveness field as a production mutation makes a
    correct replay spuriously fail.  Keep raw before/after fingerprints for
    audit and compare a canonical JSON projection with only that one field
    excluded.
    """

    path = _normal_path(path)
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if not isinstance(decoded, dict):
        return {
            "path": str(path),
            "available": False,
            "error_type": "TypeError",
            "error": "scheduler state JSON must be an object",
        }
    heartbeat = decoded.pop("heartbeat", None)
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": str(path),
        "available": True,
        "heartbeat": heartbeat,
        "semantic_sha256_without_heartbeat": hashlib.sha256(canonical).hexdigest().upper(),
    }


def _protected_semantic_snapshots(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Capture semantic guards only for protected paths with known volatility."""

    scheduler_state = _normal_path(_DEFAULT_SCHEDULER_STATE)
    return {
        str(path): _scheduler_state_semantic_projection(path)
        for path in _dedupe_paths(paths)
        if _normal_path(path) == scheduler_state
    }


def _protected_unchanged(
    protected_before: dict[str, Any] | None,
    protected_after: dict[str, Any],
    semantic_before: dict[str, dict[str, Any]] | None,
    semantic_after: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Strictly compare protected artifacts, except scheduler heartbeat only."""

    if protected_before is None:
        return True, {}
    scheduler_state = _normal_path(_DEFAULT_SCHEDULER_STATE)
    guards: dict[str, dict[str, Any]] = {}
    passed = True
    for raw_path, before in protected_before.items():
        after = protected_after.get(raw_path)
        if _normal_path(Path(raw_path)) != scheduler_state:
            item_passed = before == after
            guards[raw_path] = {"mode": "raw_fingerprint", "passed": item_passed}
        else:
            before_projection = (semantic_before or {}).get(raw_path)
            after_projection = semantic_after.get(raw_path)
            item_passed = bool(
                before_projection
                and after_projection
                and before_projection.get("available")
                and after_projection.get("available")
                and before_projection.get("semantic_sha256_without_heartbeat")
                == after_projection.get("semantic_sha256_without_heartbeat")
            )
            guards[raw_path] = {
                "mode": "scheduler_state_without_heartbeat",
                "passed": item_passed,
                "before": before_projection,
                "after": after_projection,
                "raw_fingerprint_changed": before != after,
            }
        passed = passed and item_passed
    return passed, guards


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _index_dtypes(frame: pd.DataFrame) -> list[str]:
    index = frame.index
    if isinstance(index, pd.MultiIndex):
        return [str(level.dtype) for level in index.levels]
    return [str(index.dtype)]


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
        "dtypes": {str(col): str(dtype) for col, dtype in frame.dtypes.items()},
        "index_names": [str(name) if name is not None else None for name in frame.index.names],
        "index_dtypes": _index_dtypes(frame),
        "index_has_duplicates": bool(frame.index.has_duplicates),
        "index_monotonic_increasing": bool(frame.index.is_monotonic_increasing),
    }


def _factor_output_columns(specs: list[Any]) -> list[str]:
    from cbond_on.domain.factors.spec import build_factor_col

    return [build_factor_col(spec) for spec in specs]


def _load_factor_specs(config_name: str) -> tuple[dict[str, Any], list[Any], Path]:
    from cbond_on.core.config import load_config_file, resolve_config_file_path
    from cbond_on.infra.factors.quality import load_factor_specs_from_cfg

    cfg = dict(load_config_file(config_name))
    specs = list(load_factor_specs_from_cfg(cfg))
    return cfg, specs, resolve_config_file_path(config_name)


def _assert_unconfigured_paths_environment() -> None:
    inherited = {
        name: value
        for name in _CONFIG_ENV_VARS
        if (value := str(os.environ.get(name, "")).strip())
    }
    if inherited:
        details = ", ".join(f"{key}={value}" for key, value in inherited.items())
        raise RuntimeError(
            "refusing inherited CBOND_ON path/profile override in exact parity capture: "
            + details
        )


def _capture_real_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Use the normal factor loaders, but stop before any compute/store write."""

    _assert_unconfigured_paths_environment()
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from cbond_on.core.config import load_config_file, resolve_config_file_path
    from cbond_on.infra.factors import pipeline
    from cbond_on.infra.factors.daily_context import (
        index_daily_table,
        load_daily_context_for_day,
        resolve_daily_source_specs,
    )
    from cbond_on.infra.live.factor_admission import prepare_live50_factor_admission

    factor_cfg, specs, factor_cfg_path = _load_factor_specs(args.factor_config)
    # The admission import is part of the production factor runtime.  It is
    # necessary to obtain the same frozen registered 23 kernels/spec contract,
    # but does not compute or persist any factor values.
    prepare_live50_factor_admission(factor_cfg, specs=specs)

    output_columns = _factor_output_columns(specs)
    if len(output_columns) != 50 or len(set(output_columns)) != 50:
        raise RuntimeError(
            "live50 verifier requires exactly 50 unique factor outputs; got "
            f"{len(output_columns)}"
        )
    legacy_cfg, legacy_specs, legacy_cfg_path = _load_factor_specs(args.legacy_pack)
    del legacy_cfg
    legacy_columns = _factor_output_columns(legacy_specs)
    if len(legacy_columns) != 27 or not set(legacy_columns).issubset(output_columns):
        raise RuntimeError(
            "legacy pack must be the live 27-column subset of the live50 pack; "
            f"got legacy={len(legacy_columns)}"
        )
    python_columns = [column for column in output_columns if column not in set(legacy_columns)]
    if len(python_columns) != 23:
        raise RuntimeError(
            "live50 split must be 27 legacy + 23 admitted columns; got "
            f"new={len(python_columns)}"
        )

    paths_cfg = dict(load_config_file(args.paths_config))
    paths_cfg_path = resolve_config_file_path(args.paths_config)
    panel_cfg = dict(load_config_file(args.panel_config))
    panel_cfg_path = resolve_config_file_path(args.panel_config)
    raw_root = Path(paths_cfg["raw_data_root"])
    clean_root = Path(paths_cfg.get("cleaned_data_root") or paths_cfg["clean_data_root"])
    panel_root = Path(paths_cfg["panel_data_root"])
    if not raw_root.is_dir() or not clean_root.is_dir():
        raise FileNotFoundError(
            f"live input roots are unavailable: raw={raw_root}, clean={clean_root}"
        )

    panel_name = str(factor_cfg.get("panel_name", "")).strip()
    if not panel_name:
        raise ValueError("live50 factor config has no panel_name")
    source_runtime = pipeline._build_panel_source_runtime(
        panel_source_cfg=factor_cfg.get("panel_source"),
        panel_build_cfg=panel_cfg,
        cleaned_data_root=clean_root,
    )
    context_cfg = pipeline._build_context_config(factor_cfg.get("context"), specs=specs)
    score_day = args.score_day
    panel_load = pipeline._load_factor_panel(
        panel_root,
        score_day,
        window_minutes=15,
        panel_name=panel_name,
        asset="cbond",
        panel_source=source_runtime,
    )
    panel = panel_load.panel
    if panel is None or panel.empty:
        raise RuntimeError(f"missing/empty cbond T1430 panel for {score_day}")
    panel = panel.copy()
    panel.attrs["__build_day__"] = score_day.isoformat()

    stock_panel: pd.DataFrame | None = None
    stock_message: str | None = None
    if bool(context_cfg.get("stock_enabled", False)):
        stock_load = pipeline._load_factor_panel(
            panel_root,
            score_day,
            window_minutes=15,
            panel_name=panel_name,
            asset="stock",
            panel_source=source_runtime,
        )
        stock_message = stock_load.message
        if stock_load.panel is None or stock_load.panel.empty:
            if bool(context_cfg.get("stock_strict", False)):
                raise RuntimeError(f"missing/empty stock panel for strict live50 context on {score_day}")
        else:
            stock_panel = stock_load.panel.copy()
            stock_panel.attrs["__build_day__"] = score_day.isoformat()

    bond_stock_map: pd.DataFrame | None = None
    map_index = None
    if bool(context_cfg.get("map_enabled", False)):
        map_index = pipeline._index_daily_table(raw_root, str(context_cfg.get("map_table")))
        map_frame = pipeline._read_bond_stock_map_day(
            day=score_day,
            raw_data_root=raw_root,
            table=str(context_cfg.get("map_table")),
            table_index=map_index,
        )
        if map_frame.empty and bool(context_cfg.get("map_strict", False)):
            raise RuntimeError(f"missing/empty strict bond-stock map for {score_day}")
        if not map_frame.empty:
            bond_stock_map = map_frame.copy()

    daily_data: dict[str, pd.DataFrame] = {}
    if bool(context_cfg.get("daily_enabled", False)):
        requirements = list(context_cfg.get("daily_requirements", ()))
        source_specs = resolve_daily_source_specs(
            requirements,
            daily_cfg=context_cfg.get("daily_cfg"),
        )
        source_indexes = {
            source: index_daily_table(raw_root, source_spec.table)
            for source, source_spec in source_specs.items()
        }
        daily_data = load_daily_context_for_day(
            score_day,
            raw_data_root=raw_root,
            requirements=requirements,
            source_specs=source_specs,
            source_indexes=source_indexes,
        )
        if bool(context_cfg.get("daily_strict", False)):
            for requirement in requirements:
                frame = daily_data.get(str(requirement.source))
                if frame is None:
                    raise RuntimeError(f"missing strict daily context source {requirement.source}")
                if not frame.empty:
                    missing = [column for column in requirement.columns if column not in frame.columns]
                    if missing:
                        raise RuntimeError(
                            f"daily context {requirement.source} missing columns {missing}"
                        )

    payload = {
        "schema_version": 1,
        "score_day": score_day.isoformat(),
        "specs": specs,
        "output_columns": output_columns,
        "legacy_columns": legacy_columns,
        "python_columns": python_columns,
        "panel": panel,
        "stock_panel": stock_panel,
        "bond_stock_map": bond_stock_map,
        "daily_data": daily_data,
    }
    return {
        "payload": payload,
        "capture": {
            "score_day": score_day.isoformat(),
            "factor_config": str(factor_cfg_path),
            "factor_config_sha256": _sha256_file(factor_cfg_path),
            "legacy_pack": str(legacy_cfg_path),
            "legacy_pack_sha256": _sha256_file(legacy_cfg_path),
            "paths_config": str(paths_cfg_path),
            "paths_config_sha256": _sha256_file(paths_cfg_path),
            "panel_config": str(panel_cfg_path),
            "panel_config_sha256": _sha256_file(panel_cfg_path),
            "raw_data_root": str(raw_root),
            "clean_data_root": str(clean_root),
            "panel_source_mode": source_runtime.mode,
            "panel_load_message": panel_load.message,
            "stock_panel_load_message": stock_message,
            "context": {
                "stock_enabled": bool(context_cfg.get("stock_enabled", False)),
                "map_enabled": bool(context_cfg.get("map_enabled", False)),
                "daily_enabled": bool(context_cfg.get("daily_enabled", False)),
                "daily_sources": list(map(str, daily_data)),
            },
            "factor_columns": output_columns,
            "legacy_rust_columns": legacy_columns,
            "historical_python_columns": python_columns,
            "panel": _frame_summary(panel),
            "stock_panel": _frame_summary(stock_panel)
            if stock_panel is not None
            else None,
            "bond_stock_map": _frame_summary(bond_stock_map)
            if bond_stock_map is not None
            else None,
            "daily_data": {source: _frame_summary(frame) for source, frame in daily_data.items()},
        },
    }


def _input_path(root: Path) -> Path:
    return root / "captured_live50_inputs.pkl"


def _capture_path(root: Path) -> Path:
    return root / "capture.json"


def _result_path(root: Path, mode: str) -> Path:
    return root / f"{mode}_result.pkl"


def _result_meta_path(root: Path, mode: str) -> Path:
    return root / f"{mode}_result.json"


def _run_capture(args: argparse.Namespace, root: Path) -> int:
    payload_path = _input_path(root)
    meta_path = _capture_path(root)
    if payload_path.exists() or meta_path.exists():
        raise FileExistsError("capture artifact already exists; refusing overwrite")
    loaded = _capture_real_inputs(args)
    with payload_path.open("wb") as handle:
        pickle.dump(loaded["payload"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    capture = dict(loaded["capture"])
    capture["input_snapshot"] = _fingerprint(payload_path)
    _write_json(meta_path, capture)
    print(json.dumps({"mode": "capture", "capture": capture}, ensure_ascii=False))
    return 0


def _load_payload(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    path = _input_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"capture input missing: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError("unrecognized captured live50 input snapshot")
    if str(payload.get("score_day")) != args.score_day.isoformat():
        raise RuntimeError(
            "capture score day does not match requested score day: "
            f"capture={payload.get('score_day')}, requested={args.score_day}"
        )
    expected = list(payload.get("output_columns") or [])
    if len(expected) != 50 or len(set(expected)) != 50:
        raise RuntimeError("capture does not carry an exact 50-column contract")
    return payload


def _extension_info(*, expected_root: Path) -> dict[str, Any]:
    package = importlib.import_module("cbond_on_rust")
    extension = importlib.import_module("cbond_on_rust.cbond_on_rust")
    extension_path = Path(extension.__file__).resolve()
    if not _is_below(extension_path, expected_root):
        raise RuntimeError(
            "loaded cbond_on_rust extension is outside the required root: "
            f"loaded={extension_path}, expected_below={expected_root}"
        )
    return {
        "package_path": str(Path(package.__file__).resolve()),
        "extension_path": str(extension_path),
        "extension_sha256": _sha256_file(extension_path),
        "has_compute_factor_frame": bool(hasattr(package, "compute_factor_frame")),
        "has_factor_capabilities": bool(hasattr(package, "factor_capabilities")),
        "has_typed_kernel_api": bool(hasattr(package, "compute_typed_factor_frame")),
    }


def _load_scratch_extension(site: Path) -> dict[str, Any]:
    site = _normal_path(site)
    package_dir = site / "cbond_on_rust"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"--scratch-site must contain an installed cbond_on_rust package: {site}"
        )
    for name in tuple(sys.modules):
        if name == "cbond_on_rust" or name.startswith("cbond_on_rust."):
            del sys.modules[name]
    sys.path.insert(0, str(site))
    importlib.invalidate_caches()
    info = _extension_info(expected_root=site)
    package = importlib.import_module("cbond_on_rust")
    if not bool(info["has_compute_factor_frame"]):
        raise RuntimeError("scratch extension has no public compute_factor_frame API")
    if not bool(info["has_factor_capabilities"]):
        raise RuntimeError("scratch extension has no factor_capabilities() API")
    capabilities = package.factor_capabilities()
    if not isinstance(capabilities, dict):
        raise RuntimeError("scratch factor_capabilities() did not return a dict")
    info["capabilities"] = capabilities
    return info


def _write_frame_result(
    *,
    root: Path,
    mode: str,
    frame: pd.DataFrame,
    elapsed_s: float,
    extension: dict[str, Any],
) -> None:
    output = _result_path(root, mode)
    meta = _result_meta_path(root, mode)
    if output.exists() or meta.exists():
        raise FileExistsError(f"{mode} result already exists; refusing overwrite")
    frame.to_pickle(output)
    _write_json(
        meta,
        {
            "mode": mode,
            "elapsed_s": elapsed_s,
            "result": _fingerprint(output),
            "frame": _frame_summary(frame),
            "extension": extension,
        },
    )


def _run_hybrid(args: argparse.Namespace, root: Path) -> int:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    payload = _load_payload(root, args)
    extension = _extension_info(expected_root=_REPO_ROOT / "cbond_on_rust")
    if args.scratch_site and _is_below(Path(extension["extension_path"]), _normal_path(args.scratch_site)):
        raise RuntimeError("historical hybrid child unexpectedly imported the scratch extension")

    from cbond_on.infra.factors.rust_python_hybrid import (
        RustPythonHybridPlan,
        build_factor_frame_rust_python_hybrid,
    )
    from cbond_on.core.config import load_config_file
    from cbond_on.infra.live.factor_admission import prepare_live50_factor_admission

    # The reference half of the historical hybrid still evaluates the admitted
    # 23 through their Python kernels.  A fresh child has an empty
    # FactorRegistry, so explicitly recreate the same admission boundary before
    # calling ``build_factor_frame`` through the hybrid adapter.
    reference_factor_cfg = dict(load_config_file(args.factor_config))
    prepare_live50_factor_admission(reference_factor_cfg, specs=payload["specs"])

    plan = RustPythonHybridPlan(
        rust_columns=tuple(payload["legacy_columns"]),
        python_columns=tuple(payload["python_columns"]),
        output_columns=tuple(payload["output_columns"]),
    )
    started = perf_counter()
    frame = build_factor_frame_rust_python_hybrid(
        payload["panel"],
        payload["specs"],
        plan=plan,
        stock_panel=payload.get("stock_panel"),
        bond_stock_map=payload.get("bond_stock_map"),
        daily_data=payload.get("daily_data"),
        workers=1,
        compute_backend_params={
            "__compute_backend__": {"reference_mode": "isolated_parity_reference"}
        },
        reference_mode="isolated_parity_reference",
    )
    elapsed = perf_counter() - started
    _write_frame_result(
        root=root,
        mode="hybrid",
        frame=frame,
        elapsed_s=elapsed,
        extension=extension,
    )
    print(json.dumps({"mode": "hybrid", "rows": len(frame), "elapsed_s": elapsed}))
    return 0


def _run_unified(args: argparse.Namespace, root: Path) -> int:
    if args.scratch_site is None:
        raise ValueError("--scratch-site is required for --mode unified")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    payload = _load_payload(root, args)
    extension = _load_scratch_extension(args.scratch_site)

    from cbond_on.infra.factors.rust_backend import build_factor_frame_rust

    started = perf_counter()
    frame = build_factor_frame_rust(
        payload["panel"],
        payload["specs"],
        stock_panel=payload.get("stock_panel"),
        bond_stock_map=payload.get("bond_stock_map"),
        daily_data=payload.get("daily_data"),
        compute_backend_params={
            "__compute_backend__": {
                "execution_policy": "rust_first",
                "engine_requested": "rust",
                "engine_active": "rust",
                "plan_log_summary": True,
            }
        },
    )
    elapsed = perf_counter() - started
    _write_frame_result(
        root=root,
        mode="unified",
        frame=frame,
        elapsed_s=elapsed,
        extension=extension,
    )
    print(json.dumps({"mode": "unified", "rows": len(frame), "elapsed_s": elapsed}))
    return 0


def _key_preview(index: pd.Index, limit: int = 5) -> list[str]:
    preview = []
    for item in list(index[:limit]):
        preview.append(str(item))
    return preview


def _compare_indices(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, Any]:
    expected_index = expected.index
    actual_index = actual.index
    names_equal = list(expected_index.names) == list(actual_index.names)
    dtypes_equal = _index_dtypes(expected) == _index_dtypes(actual)
    exact_order_equal = bool(expected_index.equals(actual_index))
    try:
        expected_only = expected_index.difference(actual_index)
        actual_only = actual_index.difference(expected_index)
        expected_only_count = int(len(expected_only))
        actual_only_count = int(len(actual_only))
        expected_only_preview = _key_preview(expected_only)
        actual_only_preview = _key_preview(actual_only)
    except Exception as exc:  # defensive evidence only
        expected_only_count = actual_only_count = -1
        expected_only_preview = actual_only_preview = [f"index-difference-error:{type(exc).__name__}"]
    return {
        "expected_rows": int(len(expected_index)),
        "actual_rows": int(len(actual_index)),
        "names_equal": names_equal,
        "expected_names": [str(item) if item is not None else None for item in expected_index.names],
        "actual_names": [str(item) if item is not None else None for item in actual_index.names],
        "dtypes_equal": dtypes_equal,
        "expected_dtypes": _index_dtypes(expected),
        "actual_dtypes": _index_dtypes(actual),
        "exact_order_equal": exact_order_equal,
        "expected_has_duplicates": bool(expected_index.has_duplicates),
        "actual_has_duplicates": bool(actual_index.has_duplicates),
        "expected_only_count": expected_only_count,
        "actual_only_count": actual_only_count,
        "expected_only_preview": expected_only_preview,
        "actual_only_preview": actual_only_preview,
        "passed": bool(
            names_equal
            and dtypes_equal
            and exact_order_equal
            and not expected_index.has_duplicates
            and not actual_index.has_duplicates
        ),
    }


def _float_column_evidence(expected: pd.Series, actual: pd.Series) -> dict[str, Any]:
    expected_dtype = str(expected.dtype)
    actual_dtype = str(actual.dtype)
    dtype_equal = expected_dtype == actual_dtype
    evidence: dict[str, Any] = {
        "expected_dtype": expected_dtype,
        "actual_dtype": actual_dtype,
        "dtype_equal": dtype_equal,
    }
    if expected_dtype != "float64" or actual_dtype != "float64":
        evidence.update(
            {
                "nan_mask_mismatch": -1,
                "inf_mask_mismatch": -1,
                "finite_mask_mismatch": -1,
                "finite_count": -1,
                "finite_uint64_mismatch": -1,
                "negative_zero_expected": -1,
                "negative_zero_actual": -1,
                "negative_zero_mismatch": -1,
                "passed": False,
                "reason": "all live50 factor outputs must be float64",
            }
        )
        return evidence

    left = expected.to_numpy(dtype=np.float64, copy=False)
    right = actual.to_numpy(dtype=np.float64, copy=False)
    left_nan = np.isnan(left)
    right_nan = np.isnan(right)
    left_inf = np.isinf(left)
    right_inf = np.isinf(right)
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    paired_finite = left_finite & right_finite
    paired_zero = paired_finite & (left == 0.0) & (right == 0.0)
    finite_bits_mismatch = int(
        np.count_nonzero(
            left[paired_finite].view(np.uint64) != right[paired_finite].view(np.uint64)
        )
    )
    negative_zero_mismatch = int(
        np.count_nonzero(np.signbit(left[paired_zero]) != np.signbit(right[paired_zero]))
    )
    nan_mask_mismatch = int(np.count_nonzero(left_nan != right_nan))
    inf_mask_mismatch = int(np.count_nonzero(left_inf != right_inf))
    finite_mask_mismatch = int(np.count_nonzero(left_finite != right_finite))
    # For non-NaN values this additionally proves the sign of any infinities.
    paired_non_nan = ~left_nan & ~right_nan
    non_nan_bits_mismatch = int(
        np.count_nonzero(
            left[paired_non_nan].view(np.uint64) != right[paired_non_nan].view(np.uint64)
        )
    )
    evidence.update(
        {
            "nan_mask_mismatch": nan_mask_mismatch,
            "inf_mask_mismatch": inf_mask_mismatch,
            "finite_mask_mismatch": finite_mask_mismatch,
            "finite_count": int(np.count_nonzero(paired_finite)),
            "finite_uint64_mismatch": finite_bits_mismatch,
            "non_nan_uint64_mismatch": non_nan_bits_mismatch,
            "negative_zero_expected": int(np.count_nonzero((left == 0.0) & np.signbit(left))),
            "negative_zero_actual": int(np.count_nonzero((right == 0.0) & np.signbit(right))),
            "negative_zero_mismatch": negative_zero_mismatch,
            "passed": bool(
                dtype_equal
                and nan_mask_mismatch == 0
                and inf_mask_mismatch == 0
                and finite_mask_mismatch == 0
                and finite_bits_mismatch == 0
                and non_nan_bits_mismatch == 0
                and negative_zero_mismatch == 0
            ),
        }
    )
    return evidence


def _compare_frames(name: str, expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, Any]:
    index = _compare_indices(expected, actual)
    expected_columns = list(map(str, expected.columns))
    actual_columns = list(map(str, actual.columns))
    columns_equal = expected_columns == actual_columns
    missing_columns = [column for column in expected_columns if column not in actual.columns]
    extra_columns = [column for column in actual_columns if column not in expected.columns]
    shared_columns = [column for column in expected_columns if column in actual.columns]
    per_column = {
        column: _float_column_evidence(expected[column], actual[column])
        for column in shared_columns
    }
    passed = bool(
        index["passed"]
        and columns_equal
        and not missing_columns
        and not extra_columns
        and len(shared_columns) == len(expected_columns)
        and all(item["passed"] for item in per_column.values())
    )
    return {
        "name": name,
        "expected": _frame_summary(expected),
        "actual": _frame_summary(actual),
        "index": index,
        "columns": {
            "exact_order_equal": columns_equal,
            "expected": expected_columns,
            "actual": actual_columns,
            "missing": missing_columns,
            "extra": extra_columns,
        },
        "per_column": per_column,
        "passed": passed,
    }


def _load_result_frame(root: Path, mode: str) -> pd.DataFrame:
    path = _result_path(root, mode)
    if not path.is_file():
        raise FileNotFoundError(f"missing {mode} result: {path}")
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{mode} result is not a DataFrame")
    return frame


def _load_result_meta(root: Path, mode: str) -> dict[str, Any]:
    path = _result_meta_path(root, mode)
    if not path.is_file():
        raise FileNotFoundError(f"missing {mode} metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_compare(
    args: argparse.Namespace,
    root: Path,
    *,
    protected_before: dict[str, Any] | None = None,
    protected_semantic_before: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    golden_path = _normal_path(args.golden_factor)
    if not golden_path.is_file():
        raise FileNotFoundError(f"golden live50 FactorStore day missing: {golden_path}")
    capture = json.loads(_capture_path(root).read_text(encoding="utf-8"))
    expected_columns = list(capture["factor_columns"])
    golden = pd.read_parquet(golden_path)
    hybrid = _load_result_frame(root, "hybrid")
    unified = _load_result_frame(root, "unified")
    golden_schema = _frame_summary(golden)
    golden_contract_ok = list(golden.columns) == expected_columns and len(golden.columns) == 50
    comparisons = {
        "hybrid_vs_persisted_golden": _compare_frames(
            "historical hybrid replay vs persisted 2026-08-06 golden", golden, hybrid
        ),
        "unified_vs_hybrid": _compare_frames(
            "candidate unified Rust public API vs historical hybrid replay", hybrid, unified
        ),
        "unified_vs_persisted_golden": _compare_frames(
            "candidate unified Rust public API vs persisted 2026-08-06 golden", golden, unified
        ),
    }
    protected_after = _fingerprint_paths(
        [Path(item) for item in (protected_before or {}).keys()]
    )
    protected_semantic_after = _protected_semantic_snapshots(
        Path(item) for item in (protected_before or {}).keys()
    )
    protected_unchanged, protected_guards = _protected_unchanged(
        protected_before,
        protected_after,
        protected_semantic_before,
        protected_semantic_after,
    )
    payload = {
        "tool": "verify_live50_unified_rust_parity",
        "schema_version": 1,
        "read_only_inputs": True,
        "db_write": False,
        "scheduler_called": False,
        "live_output_writer_called": False,
        "score_day": args.score_day.isoformat(),
        "scratch_root": str(root),
        "golden_factor": _fingerprint(golden_path),
        "golden_contract": {
            "expected_columns": expected_columns,
            "golden_schema": golden_schema,
            "passed": golden_contract_ok,
        },
        "capture": capture,
        "hybrid": _load_result_meta(root, "hybrid"),
        "unified": _load_result_meta(root, "unified"),
        "comparisons": comparisons,
        "protected_before": protected_before,
        "protected_after": protected_after,
        "protected_semantic_guards": protected_guards,
        "protected_unchanged": protected_unchanged,
    }
    payload["candidate_parity_passed"] = bool(comparisons["unified_vs_hybrid"]["passed"])
    payload["historical_replay_passed"] = bool(
        comparisons["hybrid_vs_persisted_golden"]["passed"]
        and comparisons["unified_vs_persisted_golden"]["passed"]
    )
    payload["passed"] = bool(
        golden_contract_ok
        and payload["candidate_parity_passed"]
        and payload["historical_replay_passed"]
        and protected_unchanged
    )
    _write_json(root / "verification.json", payload)
    return payload


def _run_child(args: argparse.Namespace, mode: str, root: Path) -> None:
    command = [
        sys.executable,
        "-B",
        str(_SELF),
        "--mode",
        mode,
        "--scratch-root",
        str(root),
        "--score-day",
        args.score_day.isoformat(),
        "--golden-factor",
        str(args.golden_factor),
        "--factor-config",
        str(args.factor_config),
        "--legacy-pack",
        str(args.legacy_pack),
        "--paths-config",
        str(args.paths_config),
        "--panel-config",
        str(args.panel_config),
    ]
    if args.scratch_site is not None:
        command.extend(["--scratch-site", str(args.scratch_site)])
    child_env = os.environ.copy()
    for name in _CONFIG_ENV_VARS:
        child_env.pop(name, None)
    # Do not inherit a caller's candidate-wheel PYTHONPATH.  The unified child
    # inserts exactly --scratch-site itself; the hybrid child must see the
    # repository production extension only.
    child_env.pop("PYTHONPATH", None)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=child_env,
        text=True,
        capture_output=True,
        check=False,
    )
    (root / f"{mode}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (root / f"{mode}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} child failed with exit={completed.returncode}; "
            f"see {root / f'{mode}.stdout.log'} and {root / f'{mode}.stderr.log'}"
        )


def _run_verify(args: argparse.Namespace) -> int:
    if args.scratch_site is None:
        raise ValueError("--scratch-site is required for --mode verify")
    root = _validate_scratch_root(args.scratch_root, must_not_exist=True)
    scratch_site = _normal_path(args.scratch_site)
    if not scratch_site.is_dir():
        raise FileNotFoundError(f"scratch wheel site is missing: {scratch_site}")
    root.mkdir(parents=True, exist_ok=False)
    protected = _default_protected_paths(_normal_path(args.golden_factor))
    protected.extend(Path(value) for value in args.protect)
    protected_before = _fingerprint_paths(protected)
    protected_semantic_before = _protected_semantic_snapshots(protected)
    try:
        _run_child(args, "capture", root)
        _run_child(args, "hybrid", root)
        _run_child(args, "unified", root)
        report = _run_compare(
            args,
            root,
            protected_before=protected_before,
            protected_semantic_before=protected_semantic_before,
        )
    except Exception as exc:
        failure = {
            "tool": "verify_live50_unified_rust_parity",
            "schema_version": 1,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "score_day": args.score_day.isoformat(),
            "scratch_root": str(root),
            "protected_before": protected_before,
            "protected_after": _fingerprint_paths(protected),
            "protected_semantic_guards": {
                str(path): {
                    "mode": "scheduler_state_without_heartbeat",
                    "before": protected_semantic_before.get(str(path)),
                    "after": _scheduler_state_semantic_projection(path),
                }
                for path in _dedupe_paths(protected)
                if _normal_path(path) == _normal_path(_DEFAULT_SCHEDULER_STATE)
            },
        }
        _write_json(root / "verification.json", failure)
        raise
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "candidate_parity_passed": report["candidate_parity_passed"],
                "historical_replay_passed": report["historical_replay_passed"],
                "evidence": str(root / "verification.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("verify", "capture", "hybrid", "unified", "compare"),
        default="verify",
        help="verify orchestrates the three isolated child phases; other modes are internal/debug phases.",
    )
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument(
        "--scratch-site",
        type=Path,
        help="new scratch wheel install site containing cbond_on_rust; required for verify/unified",
    )
    parser.add_argument("--score-day", type=_parse_day, default=date(2026, 8, 6))
    parser.add_argument("--golden-factor", type=Path, default=_DEFAULT_GOLDEN)
    parser.add_argument(
        "--factor-config", default="live/live_factors_50_20260805", help="frozen live50 factor config"
    )
    parser.add_argument(
        "--legacy-pack",
        default="factor/packs/live_screened_no_winsor_27.json5",
        help="historical Rust-27 pack",
    )
    parser.add_argument("--paths-config", default="data/paths_live50_20260805")
    parser.add_argument("--panel-config", default="panel")
    parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help="additional production artifact to fingerprint before/after; repeatable",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "verify":
        return _run_verify(args)
    root = _validate_scratch_root(args.scratch_root)
    if not root.is_dir():
        raise FileNotFoundError(f"scratch root does not exist for child mode: {root}")
    if args.mode == "capture":
        return _run_capture(args, root)
    if args.mode == "hybrid":
        return _run_hybrid(args, root)
    if args.mode == "unified":
        return _run_unified(args, root)
    if args.mode == "compare":
        report = _run_compare(args, root)
        print(json.dumps({"passed": report["passed"], "evidence": str(root / "verification.json")}))
        return 0 if report["passed"] else 2
    raise AssertionError(f"unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
