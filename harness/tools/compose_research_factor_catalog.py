"""Compose an auditable research-only factor family catalogue.

The composer is deliberately separate from factor build and screening.  It
merges one already-vetted v3 family JSON with explicit research catalogue
modules that expose factor_mining_catalog().  The default is a no-write plan.
Only --execute writes, and it can write only a new child under the dedicated
research scratch root.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


# This tool's default mode is a no-write plan.  Keep imports of the requested
# research catalogues from creating incidental ``__pycache__`` artifacts when
# a caller only asks to inspect the proposed composition.
sys.dont_write_bytecode = True


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_SCRATCH_ROOT = Path(r"D:/cbond_on/research_scratch")
ALLOWED_MODULE_PREFIX = "cbond_on.domain.factors.operators.research_"
_RESEARCH_OPERATORS_ROOT = (_REPO_ROOT / "cbond_on" / "domain" / "factors" / "operators").resolve()
_CHILD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_SCHEMA_VERSION = "research_factor_catalog_composer_v1"


@dataclass(frozen=True)
class SourceSummary:
    kind: str
    identifier: str
    path: Path
    sha256: str
    family_count: int
    signal_count: int


@dataclass(frozen=True)
class CatalogCompositionPlan:
    catalog: OrderedDict[str, list[str]]
    vetted_v3: SourceSummary
    modules: tuple[SourceSummary, ...]

    @property
    def family_count(self) -> int:
        return len(self.catalog)

    @property
    def signal_count(self) -> int:
        return sum(len(signals) for signals in self.catalog.values())


@dataclass(frozen=True)
class ComposeResult:
    executed: bool
    output_dir: Path
    catalog_path: Path
    manifest_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _signal_count(catalog: OrderedDict[str, list[str]]) -> int:
    return sum(len(signals) for signals in catalog.values())


def _nonempty_name(value: object, *, role: str, source: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{source} {role} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{source} {role} must be non-empty")
    return text


def _validate_family_mapping(
    raw: object,
    *,
    source: str,
) -> OrderedDict[str, list[str]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{source} must be a non-empty family->signals JSON object")
    catalog: OrderedDict[str, list[str]] = OrderedDict()
    signal_owner: dict[str, str] = {}
    for family_raw, signals_raw in raw.items():
        family = _nonempty_name(family_raw, role="family", source=source)
        if family in catalog:
            raise ValueError(f"{source} has duplicate family: {family}")
        if not isinstance(signals_raw, list) or not signals_raw:
            raise ValueError(f"{source} family {family} must contain a non-empty signal list")
        signals: list[str] = []
        for signal_raw in signals_raw:
            signal = _nonempty_name(signal_raw, role="signal", source=source)
            prior = signal_owner.get(signal)
            if prior is not None:
                raise ValueError(
                    f"{source} signal ambiguity: {signal} belongs to both {prior} and {family}"
                )
            signal_owner[signal] = family
            signals.append(signal)
        catalog[family] = signals
    return catalog


def load_vetted_v3_family_catalog(path: Path) -> tuple[OrderedDict[str, list[str]], SourceSummary]:
    """Load one immutable v3 vetted family mapping without rewriting it."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix.lower() != ".json":
        raise ValueError(f"vetted v3 family catalog must be an existing .json file: {resolved}")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"vetted v3 family catalog is not valid JSON: {resolved}") from exc
    # The historical strict-PIT v3 artifact is deliberately wrapped as
    # ``{\"families\": {...}}``.  Test fixtures and hand-authored vetted
    # subsets may use the inner mapping directly.  Accept exactly these two
    # schemas; do not silently treat catalog metadata as a factor family.
    if isinstance(raw, dict) and "families" in raw:
        if set(raw) != {"families"}:
            raise ValueError(
                f"vetted v3 catalog wrapper must contain only 'families': {resolved}"
            )
        raw = raw["families"]
    catalog = _validate_family_mapping(raw, source=f"vetted v3 catalog {resolved}")
    summary = SourceSummary(
        kind="vetted_v3_family_json",
        identifier=str(resolved),
        path=resolved,
        sha256=_sha256_file(resolved),
        family_count=len(catalog),
        signal_count=_signal_count(catalog),
    )
    return catalog, summary


def _validate_module_path(module_name: str) -> Path:
    if not module_name.startswith(ALLOWED_MODULE_PREFIX):
        raise ValueError(
            f"research module must start with {ALLOWED_MODULE_PREFIX!r}: {module_name!r}"
        )
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        raise ValueError(f"research module cannot be resolved: {module_name!r}")
    path = Path(spec.origin).resolve(strict=True)
    try:
        path.relative_to(_RESEARCH_OPERATORS_ROOT)
    except ValueError as exc:
        raise ValueError(f"research module escaped operators root: {module_name!r} -> {path}") from exc
    if path.suffix != ".py" or not path.name.startswith("research_"):
        raise ValueError(f"research module must resolve to a research_ .py file: {path}")
    return path


def _entries_to_catalog(
    entries: Iterable[object],
    *,
    source: str,
) -> OrderedDict[str, list[str]]:
    catalog: OrderedDict[str, list[str]] = OrderedDict()
    signal_owner: dict[str, str] = {}
    count = 0
    for entry in entries:
        family = _nonempty_name(getattr(entry, "family", None), role="family", source=source)
        signal = _nonempty_name(getattr(entry, "signal", None), role="signal", source=source)
        prior = signal_owner.get(signal)
        if prior is not None:
            raise ValueError(
                f"{source} signal ambiguity: {signal} belongs to both {prior} and {family}"
            )
        signal_owner[signal] = family
        catalog.setdefault(family, []).append(signal)
        count += 1
    if count == 0:
        raise ValueError(f"{source} factor_mining_catalog() returned no entries")
    return catalog


def load_research_module_catalog(module_name: str) -> tuple[OrderedDict[str, list[str]], SourceSummary]:
    """Load a repeatable explicit research catalogue from an allowed module."""

    path = _validate_module_path(module_name)
    module = importlib.import_module(module_name)
    factory = getattr(module, "factor_mining_catalog", None)
    if not callable(factory):
        raise ValueError(f"{module_name!r} does not expose callable factor_mining_catalog()")
    try:
        first_entries = tuple(factory())
        second_entries = tuple(factory())
    except Exception as exc:
        raise ValueError(f"{module_name!r} factor_mining_catalog() failed") from exc
    first = _entries_to_catalog(first_entries, source=f"module {module_name}")
    second = _entries_to_catalog(second_entries, source=f"module {module_name}")
    if first != second:
        raise ValueError(f"{module_name!r} factor_mining_catalog() is not repeatable")
    summary = SourceSummary(
        kind="research_catalog_module",
        identifier=module_name,
        path=path,
        sha256=_sha256_file(path),
        family_count=len(first),
        signal_count=_signal_count(first),
    )
    return first, summary


def build_composition_plan(
    *,
    vetted_v3_catalog: Path,
    module_names: Sequence[str],
) -> CatalogCompositionPlan:
    """Combine sources in the explicit order while rejecting every ambiguity."""

    normalized_modules = [str(name).strip() for name in module_names if str(name).strip()]
    if not normalized_modules:
        raise ValueError("at least one explicit research module is required")
    if len(set(normalized_modules)) != len(normalized_modules):
        raise ValueError("duplicate research module in composition request")

    vetted, vetted_summary = load_vetted_v3_family_catalog(vetted_v3_catalog)
    combined: OrderedDict[str, list[str]] = OrderedDict(
        (family, list(signals)) for family, signals in vetted.items()
    )
    family_owner: dict[str, str] = {family: "vetted_v3" for family in combined}
    signal_owner: dict[str, str] = {
        signal: family for family, signals in combined.items() for signal in signals
    }
    summaries: list[SourceSummary] = []

    for module_name in normalized_modules:
        module_catalog, summary = load_research_module_catalog(module_name)
        for family, signals in module_catalog.items():
            if family in family_owner:
                raise ValueError(
                    f"family ambiguity: {family} appears in {family_owner[family]} and {module_name}"
                )
            for signal in signals:
                previous = signal_owner.get(signal)
                if previous is not None:
                    raise ValueError(
                        f"signal ambiguity: {signal} appears in {previous} and {module_name}"
                    )
            combined[family] = list(signals)
            family_owner[family] = module_name
            for signal in signals:
                signal_owner[signal] = family
        summaries.append(summary)

    return CatalogCompositionPlan(
        catalog=combined,
        vetted_v3=vetted_summary,
        modules=tuple(summaries),
    )


def _validate_child_name(output_name: str) -> str:
    name = str(output_name).strip()
    if not _CHILD_NAME_RE.fullmatch(name) or name in {".", ".."} or ".." in name:
        raise ValueError(f"invalid fresh scratch child name: {output_name!r}")
    return name


def _planned_target(scratch_root: Path, output_name: str) -> Path:
    root = scratch_root.resolve(strict=False)
    return root / _validate_child_name(output_name)


def _fresh_target(scratch_root: Path, output_name: str) -> Path:
    root = scratch_root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"research scratch root is not a directory: {root}")
    target = (root / _validate_child_name(output_name)).resolve(strict=False)
    if target.parent != root:
        raise ValueError(f"output escaped research scratch root: {target}")
    if target.exists():
        raise FileExistsError(f"fresh research scratch child already exists: {target}")
    return target


def _source_manifest(summary: SourceSummary) -> dict[str, object]:
    return {
        "kind": summary.kind,
        "identifier": summary.identifier,
        "path": str(summary.path),
        "sha256": summary.sha256,
        "family_count": summary.family_count,
        "signal_count": summary.signal_count,
    }


def build_manifest(plan: CatalogCompositionPlan) -> dict[str, object]:
    """Return an auditable source and output manifest without writing it."""

    catalog_bytes = _canonical_json_bytes(plan.catalog)
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "vetted_v3": _source_manifest(plan.vetted_v3),
        "research_modules": [_source_manifest(summary) for summary in plan.modules],
        "combined": {
            "family_count": plan.family_count,
            "signal_count": plan.signal_count,
            "family_catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        },
        "write_contract": {
            "default_mode": "dry_run",
            "execute_flag_required": True,
            "fresh_child_required": True,
            "scratch_root": str(DEFAULT_SCRATCH_ROOT),
        },
    }


def write_composition(
    plan: CatalogCompositionPlan,
    *,
    output_name: str,
    execute: bool = False,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
) -> ComposeResult:
    """Dry-run by default; execution creates exactly one fresh scratch child."""

    target = _planned_target(scratch_root, output_name)
    catalog_path = target / "family_catalog.json"
    manifest_path = target / "catalog_manifest.json"
    if not execute:
        return ComposeResult(
            executed=False,
            output_dir=target,
            catalog_path=catalog_path,
            manifest_path=manifest_path,
        )

    target = _fresh_target(scratch_root, output_name)
    catalog_path = target / "family_catalog.json"
    manifest_path = target / "catalog_manifest.json"
    catalog_bytes = _canonical_json_bytes(plan.catalog)
    manifest_bytes = _canonical_json_bytes(build_manifest(plan))
    target.mkdir()
    catalog_path.write_bytes(catalog_bytes)
    manifest_path.write_bytes(manifest_bytes)
    return ComposeResult(
        executed=True,
        output_dir=target,
        catalog_path=catalog_path,
        manifest_path=manifest_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vetted-v3-catalog", required=True, help="Existing vetted family->signals JSON")
    parser.add_argument(
        "--module",
        dest="modules",
        action="append",
        required=True,
        help=f"Explicit module under {ALLOWED_MODULE_PREFIX}",
    )
    parser.add_argument("--output-name", required=True, help="Fresh child name below research_scratch")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write family_catalog.json and catalog_manifest.json to a fresh scratch child",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_composition_plan(
        vetted_v3_catalog=Path(args.vetted_v3_catalog),
        module_names=args.modules,
    )
    result = write_composition(
        plan,
        output_name=str(args.output_name),
        execute=bool(args.execute),
    )
    report: dict[str, Any] = {
        "executed": result.executed,
        "output_dir": str(result.output_dir),
        "family_catalog": str(result.catalog_path),
        "manifest": str(result.manifest_path),
        "family_count": plan.family_count,
        "signal_count": plan.signal_count,
        "sources": {
            "vetted_v3": _source_manifest(plan.vetted_v3),
            "research_modules": [_source_manifest(summary) for summary in plan.modules],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
