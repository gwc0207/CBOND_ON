"""Safely expand and run the isolated 2026-08 factor-mining catalogue.

This research launcher is the only intended entrypoint for
``factor/research/factor_mining_20260802``.  It is deliberately conservative:

* it binds every derived output root to the dedicated scratch runtime;
* it removes inherited path-profile overrides in its own process;
* it refuses production factor/results paths before a build begins;
* it defaults to a no-write static preflight; and
* it keeps factor-batch backtests, screening, and bad-factor reports disabled.

The actual IC screen is a separate, strict read-only-on-inputs stage in
``harness/tools/factor_mining_screen.py``.  This launcher never writes a DB,
touches live configuration, or changes the scheduler.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


# Direct ``py harness/tools/...`` execution does not automatically put the
# repository root on sys.path.  This is a harness tool, not a cbond_on/run
# entrypoint, so keep its import bootstrap local and explicit.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from cbond_on.bootstrap.research import load_factor_batch_inputs  # noqa: E402
from cbond_on.core.config import parse_date  # noqa: E402
from cbond_on.core.registry import FactorRegistry, RegistryError  # noqa: E402
from cbond_on.domain.factors.defs.research_factor_mining_catalog_v1 import (  # noqa: E402
    CATALOG_VERSION,
    CatalogEntry,
    factor_mining_catalog,
)
from cbond_on.workflows.research.factor_batch import run as run_factor_batch  # noqa: E402
from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs  # noqa: E402


_CONFIG_NAME = "factor/research/factor_mining_20260802"
_PATHS_CONFIG_NAME = "data/paths_factor_mining_20260802"
_PATHS_CONFIG_PATH = _REPO_ROOT / "cbond_on" / "config" / "data" / "paths_factor_mining_20260802_config.json5"
_CONFIG_PATH = _REPO_ROOT / "cbond_on" / "config" / "factor" / "research" / "factor_mining_20260802_config.json5"
_CATALOGUE_PATH = _REPO_ROOT / "cbond_on" / "domain" / "factors" / "defs" / "research_factor_mining_catalog_v1.py"
_SCRATCH_RUNTIME_ROOT = Path(r"D:/cbond_on/research_scratch/factor_mining_20260802_strict_pit_v3")
_DATAHUB_RAW_ROOT = Path(r"D:/cbond_data_hub/raw_data")
_DATAHUB_CLEAN_ROOT = Path(r"D:/cbond_data_hub/clean_data")
_PATH_OVERRIDE_ENV = (
    "CBOND_ON_RUNTIME_ROOT",
    "CBOND_ON_PATHS_PROFILE",
    "CBOND_ON_RAW_ROOT",
    "CBOND_ON_CLEAN_ROOT",
    "CBOND_ON_DATA_ROOT",
    "CBOND_ON_PATHS_CONFIG",
)


def _resolved(path: str | Path) -> Path:
    """Produce a lexical absolute path without requiring it to exist."""

    return Path(path).expanduser().resolve(strict=False)


def _path_text(path: str | Path) -> str:
    return _resolved(path).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_isolated_paths() -> None:
    """Discard inherited path overrides and pin this process to scratch only."""

    for name in _PATH_OVERRIDE_ENV:
        os.environ.pop(name, None)
    os.environ["CBOND_ON_PATHS_CONFIG"] = str(_PATHS_CONFIG_PATH)


def _assert_equal_path(actual: str | Path, expected: str | Path, *, field: str) -> None:
    actual_path = _resolved(actual)
    expected_path = _resolved(expected)
    if actual_path != expected_path:
        raise RuntimeError(
            f"unsafe {field}: expected {expected_path.as_posix()}, got {actual_path.as_posix()}"
        )


def _assert_safe_paths(paths_cfg: dict[str, Any]) -> None:
    """Fail closed if any factor-batch derived path escaped the scratch runtime."""

    expected = {
        "panel_data_root": _SCRATCH_RUNTIME_ROOT / "panel_data",
        "label_data_root": _SCRATCH_RUNTIME_ROOT / "label_data",
        "factor_data_root": _SCRATCH_RUNTIME_ROOT / "factor_data",
        "results_root": _SCRATCH_RUNTIME_ROOT / "results",
        "model_root": _SCRATCH_RUNTIME_ROOT / "results" / "models",
        "score_root": _SCRATCH_RUNTIME_ROOT / "results" / "scores",
        "logs_root": _SCRATCH_RUNTIME_ROOT / "logs",
        "raw_data_root": _DATAHUB_RAW_ROOT,
        "clean_data_root": _DATAHUB_CLEAN_ROOT,
        "cleaned_data_root": _DATAHUB_CLEAN_ROOT,
    }
    for field, expected_path in expected.items():
        if field not in paths_cfg:
            raise RuntimeError(f"paths profile did not resolve required {field}")
        _assert_equal_path(paths_cfg[field], expected_path, field=field)

    production_factor_root = _resolved(r"D:/cbond_on/factor_data")
    factor_root = _resolved(paths_cfg["factor_data_root"])
    if factor_root == production_factor_root:
        raise RuntimeError("refusing production D:/cbond_on/factor_data as factor output")


def _catalogue_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("factor_mining_catalog")
    if not isinstance(raw, dict):
        raise ValueError("factor_mining_config.factor_mining_catalog must be an object")
    return raw


def _family_mapping(entries: Iterable[CatalogEntry]) -> dict[str, list[str]]:
    families: OrderedDict[str, list[str]] = OrderedDict()
    signals: set[str] = set()
    for entry in entries:
        family = str(entry.family).strip()
        signal = str(entry.signal).strip()
        kernel = str(entry.kernel).strip()
        if not family or not signal or not kernel:
            raise ValueError(f"catalogue contains an incomplete entry: {entry!r}")
        if signal in signals:
            raise ValueError(f"catalogue contains duplicate signal: {signal}")
        signals.add(signal)
        families.setdefault(family, []).append(signal)
    if not families:
        raise ValueError("factor-mining catalogue is empty")
    return dict(families)


def _assert_catalogue(cfg: dict[str, Any]) -> tuple[tuple[CatalogEntry, ...], dict[str, list[str]]]:
    metadata = _catalogue_metadata(cfg)
    expected_module = "cbond_on.domain.factors.defs.research_factor_mining_catalog_v1"
    if str(metadata.get("module", "")).strip() != expected_module:
        raise ValueError("factor-mining catalogue module does not match the approved research module")
    if str(metadata.get("version", "")).strip() != CATALOG_VERSION:
        raise ValueError(
            f"factor-mining catalogue version mismatch: config={metadata.get('version')!r}, code={CATALOG_VERSION!r}"
        )

    entries = tuple(factor_mining_catalog())
    families = _family_mapping(entries)
    expected_signals = int(metadata.get("expected_signal_count", 0))
    expected_families = int(metadata.get("expected_family_count", 0))
    if len(entries) != expected_signals:
        raise ValueError(f"catalogue signal count mismatch: expected {expected_signals}, got {len(entries)}")
    if len(families) != expected_families:
        raise ValueError(f"catalogue family count mismatch: expected {expected_families}, got {len(families)}")

    missing_kernels: list[str] = []
    for kernel in sorted({entry.kernel for entry in entries}):
        try:
            FactorRegistry.get(kernel)
        except RegistryError:
            missing_kernels.append(kernel)
    if missing_kernels:
        raise RuntimeError(f"factor-mining catalogue kernels are not registered: {missing_kernels}")
    return entries, families


def _assert_build_config(cfg: dict[str, Any]) -> None:
    """Protect this launcher from future config/module drift."""

    if str(cfg.get("start", "")) != "2025-01-01":
        raise ValueError("factor-mining build start must remain 2025-01-01")
    if str(cfg.get("panel_name", "")) != "T1430":
        raise ValueError("factor-mining panel_name must remain T1430")
    if str(cfg.get("factor_time", "")) != "14:30":
        raise ValueError("factor-mining factor_time must remain 14:30")
    if str(cfg.get("label_time", "")) != "14:42":
        raise ValueError("factor-mining label_time must remain 14:42")
    if bool(cfg.get("refresh", False)) or bool(cfg.get("overwrite", False)):
        raise ValueError("factor-mining config must not refresh or overwrite existing scratch output")
    if int(cfg.get("workers", 1)) != 1 or int(cfg.get("factor_workers", 1)) != 1:
        raise ValueError("factor-mining config must keep workers and factor_workers at 1")

    panel_source = cfg.get("panel_source")
    if not isinstance(panel_source, dict) or str(panel_source.get("mode", "")).lower() != "clean_direct":
        raise ValueError("factor-mining panel_source.mode must be clean_direct")
    compute = cfg.get("compute")
    if not isinstance(compute, dict) or str(compute.get("engine", "")).lower() != "python":
        raise ValueError("factor-mining compute.engine must be python")
    if not bool(compute.get("allow_python_engine", False)):
        raise ValueError("factor-mining compute.allow_python_engine must be true")
    if bool(cfg.get("backtest_enabled", True)):
        raise ValueError("factor-mining batch backtest must stay disabled")
    screening = cfg.get("screening")
    if not isinstance(screening, dict) or bool(screening.get("enabled", True)):
        raise ValueError("factor-mining batch screening must stay disabled")
    walk_forward = screening.get("walk_forward_strategy", {})
    if not isinstance(walk_forward, dict) or bool(walk_forward.get("enabled", True)):
        raise ValueError("factor-mining walk-forward screening must stay disabled")
    bad_report = cfg.get("bad_factor_report")
    if not isinstance(bad_report, dict) or bool(bad_report.get("enabled", True)):
        raise ValueError("factor-mining bad-factor report must stay disabled")
    if cfg.get("factor_files") != [] or cfg.get("factors") != []:
        raise ValueError("factor-mining static config must not inherit any factor pack or inline factor")
    if str(cfg.get("disabled_factors_file", "")).strip() != "factor/guards/factor_disabled_factors_empty.json":
        raise ValueError("factor-mining must use its explicit empty research disabled-factor guard")


def _expanded_config(cfg: dict[str, Any], entries: tuple[CatalogEntry, ...]) -> dict[str, Any]:
    expanded = dict(cfg)
    expanded["factors"] = [
        {
            "name": entry.signal,
            "factor": entry.kernel,
            "params": {
                "signal": entry.signal,
                "family": entry.family,
                "catalog_version": CATALOG_VERSION,
            },
        }
        for entry in entries
    ]
    return expanded


def _requested_range(cfg: dict[str, Any], start_text: str | None, end_text: str | None) -> tuple[date, date]:
    configured_start = parse_date(cfg["start"])
    configured_end = parse_date(cfg["end"])
    start = parse_date(start_text) if start_text else configured_start
    end = parse_date(end_text) if end_text else configured_end
    if start < configured_start:
        raise ValueError(f"--start cannot precede approved research start {configured_start.isoformat()}")
    if end > configured_end:
        raise ValueError(f"--end cannot exceed configured research end {configured_end.isoformat()}")
    if end < start:
        raise ValueError("--end must be on or after --start")
    return start, end


def _preflight(*, start_text: str | None, end_text: str | None) -> tuple[dict[str, Any], dict[str, Any], tuple[CatalogEntry, ...], dict[str, list[str]], date, date]:
    if not _PATHS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"scratch paths profile missing: {_PATHS_CONFIG_PATH}")
    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(f"factor-mining config missing: {_CONFIG_PATH}")
    if not _CATALOGUE_PATH.is_file():
        raise FileNotFoundError(f"factor-mining catalogue missing: {_CATALOGUE_PATH}")

    _configure_isolated_paths()
    cfg, paths_cfg = load_factor_batch_inputs(_CONFIG_NAME, _PATHS_CONFIG_NAME)
    _assert_safe_paths(paths_cfg)
    _assert_build_config(cfg)
    entries, families = _assert_catalogue(cfg)
    expanded = _expanded_config(cfg, entries)
    specs = build_signal_specs(expanded)
    if len(specs) != len(entries):
        raise RuntimeError(f"expanded spec count mismatch: catalogue={len(entries)}, specs={len(specs)}")
    if {spec.name for spec in specs} != {entry.signal for entry in entries}:
        raise RuntimeError("expanded specs do not exactly match catalogue signals")
    start, end = _requested_range(expanded, start_text, end_text)
    expanded["start"] = start.isoformat()
    expanded["end"] = end.isoformat()
    return expanded, paths_cfg, entries, families, start, end


def _preflight_summary(
    *,
    cfg: dict[str, Any],
    paths_cfg: dict[str, Any],
    entries: tuple[CatalogEntry, ...],
    families: dict[str, list[str]],
    start: date,
    end: date,
    execute: bool,
) -> dict[str, Any]:
    return {
        "research_only": True,
        "execute_requested": execute,
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "panel": cfg["panel_name"],
        "factor_time": cfg["factor_time"],
        "engine": cfg["compute"]["engine"],
        "panel_source": cfg["panel_source"]["mode"],
        "backtest_enabled": cfg["backtest_enabled"],
        "screening_enabled": cfg["screening"]["enabled"],
        "bad_factor_report_enabled": cfg["bad_factor_report"]["enabled"],
        "catalogue": {
            "version": CATALOG_VERSION,
            "signals": len(entries),
            "families": len(families),
        },
        "paths": {
            key: paths_cfg[key]
            for key in (
                "raw_data_root",
                "clean_data_root",
                "factor_data_root",
                "results_root",
            )
        },
    }


def _write_run_evidence(
    *,
    out_root: Path,
    paths_cfg: dict[str, Any],
    entries: tuple[CatalogEntry, ...],
    families: dict[str, list[str]],
    start: date,
    end: date,
) -> None:
    """Save immutable-enough catalog evidence next to the scratch build result."""

    family_path = out_root / "factor_mining_family_catalog.json"
    manifest_path = out_root / "factor_mining_run_manifest.json"
    for path in (family_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing run evidence: {path}")
    family_path.write_text(
        json.dumps({"families": families}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "research_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": str(Path(__file__).resolve()),
        "config": {
            "path": str(_CONFIG_PATH),
            "sha256": _sha256(_CONFIG_PATH),
        },
        "paths_profile": {
            "path": str(_PATHS_CONFIG_PATH),
            "sha256": _sha256(_PATHS_CONFIG_PATH),
        },
        "catalogue": {
            "module": "cbond_on.domain.factors.defs.research_factor_mining_catalog_v1",
            "path": str(_CATALOGUE_PATH),
            "sha256": _sha256(_CATALOGUE_PATH),
            "version": CATALOG_VERSION,
            "signal_count": len(entries),
            "family_count": len(families),
            "entries": [
                {
                    "family": entry.family,
                    "signal": entry.signal,
                    "kernel": entry.kernel,
                    "hypothesis": entry.hypothesis,
                }
                for entry in entries
            ],
        },
        "requested_date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "resolved_paths": {
            key: paths_cfg[key]
            for key in (
                "raw_data_root",
                "clean_data_root",
                "panel_data_root",
                "label_data_root",
                "factor_data_root",
                "results_root",
            )
        },
        "factor_batch_reports_disabled": {
            "backtest": True,
            "screening": True,
            "bad_factor_report": True,
        },
        "family_catalog_path": str(family_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="perform the factor build after the isolated static preflight; default is no-write preflight",
    )
    parser.add_argument("--start", help="optional smoke/retry start, not earlier than 2025-01-01")
    parser.add_argument("--end", help="optional smoke/retry end, not later than the approved config end")
    args = parser.parse_args(argv)

    cfg, paths_cfg, entries, families, start, end = _preflight(
        start_text=args.start,
        end_text=args.end,
    )
    print(json.dumps(
        _preflight_summary(
            cfg=cfg,
            paths_cfg=paths_cfg,
            entries=entries,
            families=families,
            start=start,
            end=end,
            execute=bool(args.execute),
        ),
        ensure_ascii=False,
        indent=2,
    ))
    if not args.execute:
        return 0

    out_root = Path(run_factor_batch(cfg, paths_cfg=paths_cfg))
    _write_run_evidence(
        out_root=out_root,
        paths_cfg=paths_cfg,
        entries=entries,
        families=families,
        start=start,
        end=end,
    )
    print(json.dumps({"research_only": True, "out_root": str(out_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
