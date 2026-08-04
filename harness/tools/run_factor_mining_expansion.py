"""Run an explicit, scratch-isolated research factor-expansion catalogue.

This is deliberately separate from ``run_factor_mining_20260802.py``.  It is
for a *new* research catalogue only and never uses the production factor
store, results roots, live configuration, database, or scheduler.

The command defaults to a no-write preflight.  A build requires all of:

* an explicit importable research catalogue module;
* an explicit, previously absent scratch root below
  ``D:/cbond_on/research_scratch``; and
* the explicit ``--execute`` switch.

The source data profile is the existing DataHub-only factor-mining profile.
All derived paths are redirected in memory, so this launcher never edits a
configuration file in order to run an expansion batch.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable, Iterator, Mapping


# Direct ``py harness/tools/...`` execution does not automatically put the
# repository root on sys.path.  This is a harness tool rather than a run
# entrypoint, so retain that bootstrap locally and explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from cbond_on.app.usecases.factor_batch_runtime import build_signal_specs  # noqa: E402
from cbond_on.bootstrap.research import load_factor_batch_inputs  # noqa: E402
from cbond_on.core.config import parse_date  # noqa: E402
from cbond_on.core.registry import FactorRegistry, RegistryError  # noqa: E402
from cbond_on.workflows.research.factor_batch import run as run_factor_batch  # noqa: E402


_CONFIG_NAME = "factor/research/factor_mining_20260802"
_PATHS_CONFIG_NAME = "data/paths_factor_mining_20260802"
_CONFIG_PATH = _REPO_ROOT / "cbond_on" / "config" / "factor" / "research" / "factor_mining_20260802_config.json5"
_PATHS_CONFIG_PATH = _REPO_ROOT / "cbond_on" / "config" / "data" / "paths_factor_mining_20260802_config.json5"
_RESEARCH_MODULE_DIR = _REPO_ROOT / "cbond_on" / "domain" / "factors" / "defs"
_RESEARCH_MODULE_PREFIX = "cbond_on.domain.factors.defs.research_"
_RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_DATAHUB_RAW_ROOT = Path(r"D:/cbond_data_hub/raw_data")
_DATAHUB_CLEAN_ROOT = Path(r"D:/cbond_data_hub/clean_data")
_APPROVED_START = date(2025, 1, 1)
_APPROVED_END = date(2026, 7, 30)

_PATH_OVERRIDE_ENV = (
    "CBOND_ON_RUNTIME_ROOT",
    "CBOND_ON_PATHS_PROFILE",
    "CBOND_ON_RAW_ROOT",
    "CBOND_ON_CLEAN_ROOT",
    "CBOND_ON_DATA_ROOT",
    "CBOND_ON_PATHS_CONFIG",
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
_PRODUCTION_ROOTS = (
    Path(r"D:/cbond_on/factor_data"),
    Path(r"D:/cbond_on/panel_data"),
    Path(r"D:/cbond_on/label_data"),
    Path(r"D:/cbond_on/results"),
    Path(r"D:/cbond_on/logs"),
    Path(r"D:/cbond_on/model_state"),
)


@dataclass(frozen=True)
class CatalogEntry:
    """Normalized generic research-catalogue entry."""

    family: str
    signal: str
    kernel: str
    hypothesis: str


@dataclass(frozen=True)
class PreparedRun:
    """Validated no-write expansion build plan."""

    catalog_module: str
    catalog_path: Path
    catalog_version: str | None
    cfg: dict[str, Any]
    paths_cfg: dict[str, Any]
    entries: tuple[CatalogEntry, ...]
    families: dict[str, list[str]]
    scratch_root: Path
    start: date
    end: date


def _resolved(path: str | Path) -> Path:
    """Return an absolute lexical path without requiring it to exist."""

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


def _assert_equal_path(actual: str | Path, expected: str | Path, *, field: str) -> None:
    actual_path = _resolved(actual)
    expected_path = _resolved(expected)
    if actual_path != expected_path:
        raise RuntimeError(
            f"unsafe {field}: expected {expected_path.as_posix()}, got {actual_path.as_posix()}"
        )


def _assert_scratch_root(value: str | Path, *, execute: bool) -> Path:
    """Validate a caller-owned, empty-on-execution research scratch root."""

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


@contextmanager
def _pinned_source_profile() -> Iterator[None]:
    """Pin the current process to the fixed DataHub-only source profile.

    The batch runtime loads the generic ``paths`` profile again while it runs,
    so the pin must cover both preflight and ``run_factor_batch``.  Restoring
    the caller's environment on exit keeps unit tests and embedded use safe.
    """

    missing = object()
    previous = {name: os.environ.get(name, missing) for name in _PATH_OVERRIDE_ENV}
    for name in _PATH_OVERRIDE_ENV:
        os.environ.pop(name, None)
    os.environ["CBOND_ON_PATHS_CONFIG"] = str(_PATHS_CONFIG_PATH)
    try:
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(old_value)


def _assert_frozen_build_config(cfg: Mapping[str, Any]) -> None:
    """Fail closed if the generic launcher would inherit an unsafe batch mode."""

    if parse_date(cfg.get("start")) != _APPROVED_START:
        raise ValueError(f"factor-mining config start must remain {_APPROVED_START.isoformat()}")
    if parse_date(cfg.get("end")) != _APPROVED_END:
        raise ValueError(f"factor-mining config end must remain {_APPROVED_END.isoformat()}")
    if str(cfg.get("panel_name", "")).strip() != "T1430":
        raise ValueError("factor-mining panel_name must remain T1430")
    if str(cfg.get("factor_time", "")).strip() != "14:30":
        raise ValueError("factor-mining factor_time must remain 14:30")
    if str(cfg.get("label_time", "")).strip() != "14:42":
        raise ValueError("factor-mining label_time must remain 14:42")
    if bool(cfg.get("refresh", False)) or bool(cfg.get("overwrite", False)):
        raise ValueError("factor-mining expansion config must not refresh or overwrite")
    if int(cfg.get("workers", 1)) != 1 or int(cfg.get("factor_workers", 1)) != 1:
        raise ValueError("factor-mining expansion config must keep workers and factor_workers at 1")

    panel_source = cfg.get("panel_source")
    if not isinstance(panel_source, Mapping) or str(panel_source.get("mode", "")).lower() != "clean_direct":
        raise ValueError("factor-mining expansion panel_source.mode must be clean_direct")
    compute = cfg.get("compute")
    if not isinstance(compute, Mapping) or str(compute.get("engine", "")).lower() != "python":
        raise ValueError("factor-mining expansion compute.engine must be python")
    if not bool(compute.get("allow_python_engine", False)):
        raise ValueError("factor-mining expansion compute.allow_python_engine must be true")

    if bool(cfg.get("backtest_enabled", True)):
        raise ValueError("factor-mining expansion batch backtest must stay disabled")
    screening = cfg.get("screening")
    if not isinstance(screening, Mapping) or bool(screening.get("enabled", True)):
        raise ValueError("factor-mining expansion screening must stay disabled")
    if bool(screening.get("copy_reports", False)):
        raise ValueError("factor-mining expansion screening must not copy reports")
    walk_forward = screening.get("walk_forward_strategy", {})
    if not isinstance(walk_forward, Mapping) or bool(walk_forward.get("enabled", True)):
        raise ValueError("factor-mining expansion walk-forward screening must stay disabled")
    bad_factor_report = cfg.get("bad_factor_report")
    if not isinstance(bad_factor_report, Mapping) or bool(bad_factor_report.get("enabled", True)):
        raise ValueError("factor-mining expansion bad-factor report must stay disabled")
    if cfg.get("factor_files") != [] or cfg.get("factors") != []:
        raise ValueError("factor-mining expansion static config must not inherit a factor pack or inline factors")
    if str(cfg.get("disabled_factors_file", "")).strip() != "factor/guards/factor_disabled_factors_empty.json":
        raise ValueError("factor-mining expansion must use the explicit empty research disabled-factor guard")


def _assert_source_paths(paths_cfg: Mapping[str, Any]) -> None:
    """Prove inputs are exactly the fixed DataHub-only profile roots."""

    for field, expected in _SOURCE_ROOTS.items():
        if field not in paths_cfg:
            raise RuntimeError(f"source paths profile did not resolve required {field}")
        _assert_equal_path(paths_cfg[field], expected, field=field)


def _redirect_derived_paths(source_paths_cfg: Mapping[str, Any], scratch_root: Path) -> dict[str, Any]:
    """Keep sources fixed and redirect every known derived root in memory."""

    _assert_source_paths(source_paths_cfg)
    redirected = dict(source_paths_cfg)
    for field, relative_path in _DERIVED_ROOTS.items():
        redirected[field] = _path_text(scratch_root / relative_path)
    _assert_only_safe_paths(redirected, scratch_root=scratch_root)
    return redirected


def _assert_only_safe_paths(paths_cfg: Mapping[str, Any], *, scratch_root: Path) -> None:
    """Reject an unknown or escaped runtime root before a build can start."""

    for field, expected in _SOURCE_ROOTS.items():
        if field not in paths_cfg:
            raise RuntimeError(f"paths profile did not resolve required source {field}")
        _assert_equal_path(paths_cfg[field], expected, field=field)

    for field, relative_path in _DERIVED_ROOTS.items():
        if field not in paths_cfg:
            raise RuntimeError(f"paths profile did not resolve required derived root {field}")
        expected = scratch_root / relative_path
        _assert_equal_path(paths_cfg[field], expected, field=field)
        if not _is_strict_child(paths_cfg[field], scratch_root):
            raise RuntimeError(f"derived root escaped scratch: {field}={paths_cfg[field]}")

    known_root_fields = set(_SOURCE_ROOTS) | set(_DERIVED_ROOTS)
    unexpected_root_fields = sorted(
        field for field in paths_cfg if field.endswith("_root") and field not in known_root_fields
    )
    if unexpected_root_fields:
        raise RuntimeError(
            "paths profile exposes an unclassified root; refuse to infer its write safety: "
            f"{unexpected_root_fields}"
        )


def _require_research_module_name(module_name: str) -> str:
    normalized = str(module_name).strip()
    if not normalized.startswith(_RESEARCH_MODULE_PREFIX):
        raise ValueError(
            "--catalog-module must be an explicit research registration below "
            f"{_RESEARCH_MODULE_PREFIX}"
        )
    if normalized.rsplit(".", 1)[-1].startswith("_"):
        raise ValueError("--catalog-module may not name a private module")
    return normalized


def _catalog_version(module: ModuleType) -> str | None:
    for attribute in ("EXPANSION_VERSION", "CATALOG_VERSION", "__version__"):
        raw = getattr(module, attribute, None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _catalog_entry_field(raw_entry: object, *, field: str, position: int) -> str:
    raw_value: object
    if isinstance(raw_entry, Mapping):
        raw_value = raw_entry.get(field)
    else:
        raw_value = getattr(raw_entry, field, None)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"catalogue entry {position} has no non-empty {field!r}: {raw_entry!r}")
    return raw_value.strip()


def _normalise_catalogue(module_name: str) -> tuple[Path, str | None, tuple[CatalogEntry, ...], dict[str, list[str]]]:
    """Import registrations explicitly, then validate the generic catalogue contract."""

    module_name = _require_research_module_name(module_name)
    module = importlib.import_module(module_name)
    module_file_text = getattr(module, "__file__", None)
    if not isinstance(module_file_text, str) or not module_file_text:
        raise ValueError(f"research catalogue module has no source file: {module_name}")
    module_path = _resolved(module_file_text)
    if not _is_strict_child(module_path, _RESEARCH_MODULE_DIR):
        raise ValueError(
            "research catalogue module must resolve under the repository factor definitions directory: "
            f"{module_path.as_posix()}"
        )

    factory = getattr(module, "factor_mining_catalog", None)
    if not callable(factory):
        raise ValueError(f"research catalogue module must expose callable factor_mining_catalog(): {module_name}")
    raw_entries = factory()
    if isinstance(raw_entries, (str, bytes, Mapping)):
        raise ValueError("factor_mining_catalog() must return an iterable of entry objects, not a scalar/mapping")
    try:
        entries_source = tuple(raw_entries)
    except TypeError as exc:
        raise ValueError("factor_mining_catalog() must return an iterable of entry objects") from exc
    if not entries_source:
        raise ValueError("research factor-mining catalogue is empty")

    entries: list[CatalogEntry] = []
    families: OrderedDict[str, list[str]] = OrderedDict()
    seen_signals: set[str] = set()
    canonical_family_names: dict[str, str] = {}
    missing_kernels: set[str] = set()
    for position, raw_entry in enumerate(entries_source):
        family = _catalog_entry_field(raw_entry, field="family", position=position)
        signal = _catalog_entry_field(raw_entry, field="signal", position=position)
        kernel = _catalog_entry_field(raw_entry, field="kernel", position=position)
        hypothesis = _catalog_entry_field(raw_entry, field="hypothesis", position=position)
        if signal in seen_signals:
            raise ValueError(f"research catalogue has duplicate signal: {signal}")
        seen_signals.add(signal)

        canonical_family = family.casefold()
        prior_family = canonical_family_names.setdefault(canonical_family, family)
        if prior_family != family:
            raise ValueError(
                "research catalogue has ambiguous family spelling after case folding: "
                f"{prior_family!r} versus {family!r}"
            )
        families.setdefault(family, []).append(signal)
        entries.append(CatalogEntry(family=family, signal=signal, kernel=kernel, hypothesis=hypothesis))
        try:
            FactorRegistry.get(kernel)
        except RegistryError:
            missing_kernels.add(kernel)

    if missing_kernels:
        raise RuntimeError(
            "research catalogue kernels were not registered by the explicit module import: "
            f"{sorted(missing_kernels)}"
        )
    if not families:
        raise ValueError("research factor-mining catalogue has no families")
    return module_path, _catalog_version(module), tuple(entries), dict(families)


def _requested_range(start_text: str | None, end_text: str | None) -> tuple[date, date]:
    start = parse_date(start_text) if start_text else _APPROVED_START
    end = parse_date(end_text) if end_text else _APPROVED_END
    if start < _APPROVED_START:
        raise ValueError(f"--start cannot precede {_APPROVED_START.isoformat()}")
    if end > _APPROVED_END:
        raise ValueError(f"--end cannot exceed {_APPROVED_END.isoformat()}")
    if end < start:
        raise ValueError("--end must be on or after --start")
    return start, end


def _expanded_config(
    base_cfg: Mapping[str, Any],
    *,
    entries: Iterable[CatalogEntry],
    start: date,
    end: date,
) -> dict[str, Any]:
    """Expand an empty frozen config into in-memory factor specs only."""

    expanded = dict(base_cfg)
    # This static config's original v3 catalogue metadata is intentionally not
    # meaningful for an explicit second-wave module.
    expanded.pop("factor_mining_catalog", None)
    expanded["start"] = start.isoformat()
    expanded["end"] = end.isoformat()
    expanded["backtest_enabled"] = False
    screening = dict(expanded["screening"])
    screening["enabled"] = False
    screening["copy_reports"] = False
    walk_forward = dict(screening["walk_forward_strategy"])
    walk_forward["enabled"] = False
    screening["walk_forward_strategy"] = walk_forward
    expanded["screening"] = screening
    bad_factor_report = dict(expanded["bad_factor_report"])
    bad_factor_report["enabled"] = False
    expanded["bad_factor_report"] = bad_factor_report
    expanded["factor_files"] = []
    expanded["factors"] = [
        {
            "name": entry.signal,
            "factor": entry.kernel,
            "params": {
                "signal": entry.signal,
                "family": entry.family,
            },
        }
        for entry in entries
    ]
    return expanded


def _preflight(
    *,
    catalog_module: str,
    scratch_root_text: str,
    start_text: str | None,
    end_text: str | None,
    execute: bool,
) -> PreparedRun:
    """Validate a complete generic expansion plan without creating files."""

    if not _CONFIG_PATH.is_file():
        raise FileNotFoundError(f"factor-mining config missing: {_CONFIG_PATH}")
    if not _PATHS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"DataHub-only paths profile missing: {_PATHS_CONFIG_PATH}")
    scratch_root = _assert_scratch_root(scratch_root_text, execute=execute)
    base_cfg, source_paths_cfg = load_factor_batch_inputs(_CONFIG_NAME, _PATHS_CONFIG_NAME)
    _assert_frozen_build_config(base_cfg)
    paths_cfg = _redirect_derived_paths(source_paths_cfg, scratch_root)
    catalog_path, catalog_version, entries, families = _normalise_catalogue(catalog_module)
    start, end = _requested_range(start_text, end_text)
    cfg = _expanded_config(
        base_cfg,
        entries=entries,
        start=start,
        end=end,
    )
    specs = build_signal_specs(cfg)
    expected_signals = {entry.signal for entry in entries}
    if len(specs) != len(entries) or {spec.name for spec in specs} != expected_signals:
        raise RuntimeError("expanded factor specs do not exactly match the validated research catalogue")
    return PreparedRun(
        catalog_module=catalog_module,
        catalog_path=catalog_path,
        catalog_version=catalog_version,
        cfg=cfg,
        paths_cfg=paths_cfg,
        entries=entries,
        families=families,
        scratch_root=scratch_root,
        start=start,
        end=end,
    )


def _preflight_summary(prepared: PreparedRun, *, execute: bool) -> dict[str, Any]:
    return {
        "research_only": True,
        "execute_requested": execute,
        "catalog_module": prepared.catalog_module,
        "catalog_version": prepared.catalog_version,
        "catalogue": {
            "signals": len(prepared.entries),
            "families": len(prepared.families),
            "kernels": sorted({entry.kernel for entry in prepared.entries}),
        },
        "date_range": {"start": prepared.start.isoformat(), "end": prepared.end.isoformat()},
        "panel": prepared.cfg["panel_name"],
        "factor_time": prepared.cfg["factor_time"],
        "label_time": prepared.cfg["label_time"],
        "engine": prepared.cfg["compute"]["engine"],
        "panel_source": prepared.cfg["panel_source"]["mode"],
        "reports_disabled": {
            "backtest": not prepared.cfg["backtest_enabled"],
            "screening": not prepared.cfg["screening"]["enabled"],
            "walk_forward_screening": not prepared.cfg["screening"]["walk_forward_strategy"]["enabled"],
            "bad_factor_report": not prepared.cfg["bad_factor_report"]["enabled"],
        },
        "scratch_root": _path_text(prepared.scratch_root),
        "scratch_root_exists_before_execution": prepared.scratch_root.exists(),
        "source_inputs": {field: prepared.paths_cfg[field] for field in _SOURCE_ROOTS},
        "derived_write_roots": {field: prepared.paths_cfg[field] for field in _DERIVED_ROOTS},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_successful_output_root(out_root: str | Path, *, prepared: PreparedRun) -> Path:
    resolved_out_root = _resolved(out_root)
    expected_results_root = _resolved(prepared.paths_cfg["results_root"])
    if not _is_strict_child(resolved_out_root, expected_results_root):
        raise RuntimeError(
            "factor batch returned an output outside the approved scratch results root: "
            f"{resolved_out_root.as_posix()}"
        )
    if not resolved_out_root.is_dir():
        raise RuntimeError(f"factor batch did not produce its declared output directory: {resolved_out_root.as_posix()}")
    return resolved_out_root


def _write_run_evidence(*, out_root: Path, prepared: PreparedRun) -> None:
    """Write catalog/manifest only after the batch has returned successfully."""

    family_path = out_root / "factor_mining_family_catalog.json"
    manifest_path = out_root / "factor_mining_run_manifest.json"
    for path in (family_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing run evidence: {path}")

    family_document = {"families": prepared.families}
    manifest = {
        "research_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "base_config": {"path": str(_CONFIG_PATH), "sha256": _sha256(_CONFIG_PATH)},
        "source_paths_profile": {"path": str(_PATHS_CONFIG_PATH), "sha256": _sha256(_PATHS_CONFIG_PATH)},
        "catalogue": {
            "module": prepared.catalog_module,
            "path": str(prepared.catalog_path),
            "sha256": _sha256(prepared.catalog_path),
            "version": prepared.catalog_version,
            "signal_count": len(prepared.entries),
            "family_count": len(prepared.families),
            "entries": [
                {
                    "family": entry.family,
                    "signal": entry.signal,
                    "kernel": entry.kernel,
                    "hypothesis": entry.hypothesis,
                }
                for entry in prepared.entries
            ],
        },
        "requested_date_range": {"start": prepared.start.isoformat(), "end": prepared.end.isoformat()},
        "source_inputs": {field: prepared.paths_cfg[field] for field in _SOURCE_ROOTS},
        "derived_write_roots": {field: prepared.paths_cfg[field] for field in _DERIVED_ROOTS},
        "factor_batch_reports_disabled": {
            "backtest": True,
            "screening": True,
            "walk_forward_screening": True,
            "bad_factor_report": True,
        },
        "family_catalog_path": str(family_path),
    }
    family_path.write_text(json.dumps(family_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-module",
        required=True,
        help="explicit research module exposing factor_mining_catalog() registrations",
    )
    parser.add_argument(
        "--scratch-root",
        required=True,
        help="new derived-output root strictly below D:/cbond_on/research_scratch",
    )
    parser.add_argument("--start", help="optional start, constrained to 2025-01-01 or later")
    parser.add_argument("--end", help="optional end, constrained to 2026-07-30 or earlier")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run the factor build after no-write preflight; defaults to no-write preflight",
    )
    args = parser.parse_args(argv)

    with _pinned_source_profile():
        prepared = _preflight(
            catalog_module=args.catalog_module,
            scratch_root_text=args.scratch_root,
            start_text=args.start,
            end_text=args.end,
            execute=bool(args.execute),
        )
        print(json.dumps(_preflight_summary(prepared, execute=bool(args.execute)), ensure_ascii=False, indent=2))
        if not args.execute:
            return 0

        out_root = _assert_successful_output_root(
            run_factor_batch(prepared.cfg, paths_cfg=prepared.paths_cfg),
            prepared=prepared,
        )
        _write_run_evidence(out_root=out_root, prepared=prepared)
        print(json.dumps({"research_only": True, "out_root": str(out_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
