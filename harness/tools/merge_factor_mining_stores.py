"""Strictly merge scratch-only T1430 FactorStores for one unified IC screen.

This utility is intentionally a data-only research tool.  It reads two or
more immutable FactorStore roots using the canonical
``factors/T1430/YYYY-MM/YYYYMMDD.parquet`` layout and writes a *new* merged
FactorStore only below ``D:/cbond_on/research_scratch``.  It does not load
runtime configuration, build factors, access a database, or interact with
live scheduling.

The default is a complete read-only preflight.  ``--execute`` is required to
write an output, and the requested output root must not already exist.

Different immutable research builds can legitimately expose a different
number of valid T1430 rows on a historical day (for example after a DataHub
historical backfill).  A merge therefore uses the deterministic *outer union*
of the validated ``(dt, code)`` indexes for that day and retains an absent
source value as ``NaN``.  It never substitutes an intersection or fabricates a
value.  The downstream screen remains responsible for applying the frozen
T-1 ``o_0005`` universe and for measuring each factor's actual coverage.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PANEL_NAME = "T1430"
_RESEARCH_SCRATCH_PARENT = Path(r"D:/cbond_on/research_scratch")
_PRODUCTION_ROOTS = (
    Path(r"D:/cbond_on/factor_data"),
    Path(r"D:/cbond_on/panel_data"),
    Path(r"D:/cbond_on/label_data"),
    Path(r"D:/cbond_on/results"),
    Path(r"D:/cbond_on/logs"),
    Path(r"D:/cbond_on/model_state"),
)


@dataclass(frozen=True)
class DayFile:
    """Validated immutable input-file metadata retained through execution."""

    day: date
    path: Path
    sha256: str
    bytes: int
    row_count: int
    columns: tuple[str, ...]
    index: pd.MultiIndex = field(repr=False, compare=False)


@dataclass(frozen=True)
class SourceStore:
    """One fully scanned scratch FactorStore root."""

    root: Path
    factor_dir: Path
    days: tuple[date, ...]
    columns: tuple[str, ...]
    day_files: Mapping[date, DayFile]


@dataclass(frozen=True)
class FamilyCatalog:
    """A validated supplied catalog whose signals exactly match output fields."""

    source_path: Path
    sha256: str
    source_text: str
    family_to_signals: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class MergePlan:
    """Complete preflight result; no files are created while constructing it."""

    sources: tuple[SourceStore, ...]
    selected_columns_by_source: tuple[tuple[str, ...], ...]
    output_root: Path
    days: tuple[date, ...]
    output_indexes: Mapping[date, pd.MultiIndex] = field(repr=False, compare=False)
    output_columns: tuple[str, ...]
    family_catalog: FamilyCatalog | None


def _resolved(path: str | Path) -> Path:
    """Produce an absolute lexical path without requiring it to exist."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_scratch_root(value: str | Path, *, role: str, must_exist: bool) -> Path:
    """Reject production or ambiguous roots before any Parquet access/write."""

    root = _resolved(value)
    scratch_parent = _resolved(_RESEARCH_SCRATCH_PARENT)
    if not _is_strict_child(root, scratch_parent):
        raise ValueError(
            f"{role} must resolve strictly below {scratch_parent.as_posix()}, got {root.as_posix()}"
        )
    for production_root in _PRODUCTION_ROOTS:
        resolved_production_root = _resolved(production_root)
        if root == resolved_production_root or _is_strict_child(root, resolved_production_root):
            raise ValueError(f"{role} must never be a production root: {root.as_posix()}")
    if must_exist and not root.is_dir():
        raise FileNotFoundError(f"{role} does not exist as a directory: {root.as_posix()}")
    return root


def _validate_roots(input_root_values: Sequence[str], output_root_value: str) -> tuple[tuple[Path, ...], Path]:
    if len(input_root_values) < 2:
        raise ValueError("at least two --input-root values are required")
    input_roots = tuple(
        _assert_scratch_root(value, role=f"input root {position + 1}", must_exist=True)
        for position, value in enumerate(input_root_values)
    )
    if len(set(input_roots)) != len(input_roots):
        raise ValueError("input roots must be distinct after path resolution")

    output_root = _assert_scratch_root(output_root_value, role="output root", must_exist=False)
    if output_root.exists():
        raise FileExistsError(f"output root already exists; output reuse is forbidden: {output_root.as_posix()}")
    for input_root in input_roots:
        if output_root == input_root or _is_strict_child(output_root, input_root):
            raise ValueError("output root must not be inside an input root; inputs are read-only")
    return input_roots, output_root


def _factor_dir(root: Path) -> Path:
    factor_dir = root / "factors" / _PANEL_NAME
    if not factor_dir.is_dir():
        raise FileNotFoundError(f"FactorStore is missing factors/{_PANEL_NAME}: {factor_dir.as_posix()}")
    return factor_dir


def _discover_day_paths(*, factor_dir: Path, source_root: Path) -> dict[date, Path]:
    """Accept only the canonical two-level month/day FactorStore layout."""

    files = sorted((path for path in factor_dir.rglob("*") if path.is_file()), key=lambda path: path.as_posix())
    if not files:
        raise ValueError(f"FactorStore has no files below {factor_dir.as_posix()}")
    day_paths: dict[date, Path] = {}
    for path in files:
        relative = path.relative_to(factor_dir)
        if len(relative.parts) != 2 or path.suffix.lower() != ".parquet":
            raise ValueError(
                "FactorStore contains a noncanonical file; expected YYYY-MM/YYYYMMDD.parquet: "
                f"{path.as_posix()}"
            )
        month_text, filename = relative.parts
        try:
            parsed_day = datetime.strptime(Path(filename).stem, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError(f"FactorStore day filename is not YYYYMMDD.parquet: {path.as_posix()}") from exc
        expected_month = f"{parsed_day.year:04d}-{parsed_day.month:02d}"
        if month_text != expected_month:
            raise ValueError(
                "FactorStore month/day layout mismatch: "
                f"{path.as_posix()} should be under {expected_month}"
            )
        if parsed_day in day_paths:
            raise ValueError(
                f"FactorStore has duplicate day files for {parsed_day.isoformat()}: "
                f"{day_paths[parsed_day].as_posix()} and {path.as_posix()}"
            )
        day_paths[parsed_day] = path
    if not day_paths:
        raise ValueError(f"FactorStore has no canonical day files below {source_root.as_posix()}")
    return dict(sorted(day_paths.items()))


def _validated_frame(*, frame: pd.DataFrame, day: date, path: Path) -> tuple[tuple[str, ...], pd.MultiIndex]:
    """Check the exact FactorStore day-frame contract before any merge."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"FactorStore day is not a DataFrame: {path.as_posix()}")
    if frame.empty:
        raise ValueError(f"FactorStore day frame is empty: {path.as_posix()}")
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError(f"FactorStore day index must be a two-level (dt, code) MultiIndex: {path.as_posix()}")
    if list(frame.index.names) != ["dt", "code"]:
        raise ValueError(
            "FactorStore day index names must be exactly ['dt', 'code']: "
            f"{path.as_posix()} has {list(frame.index.names)!r}"
        )
    if frame.index.has_duplicates:
        raise ValueError(f"FactorStore day index contains duplicate (dt, code) rows: {path.as_posix()}")

    dt_values = pd.to_datetime(frame.index.get_level_values("dt"), errors="coerce")
    if pd.isna(dt_values).any():
        raise ValueError(f"FactorStore day index has invalid dt values: {path.as_posix()}")
    observed_days = set(pd.DatetimeIndex(dt_values).date)
    if observed_days != {day}:
        raise ValueError(
            "FactorStore file day and index dt date disagree: "
            f"file={day.isoformat()} index_days={sorted(item.isoformat() for item in observed_days)} "
            f"path={path.as_posix()}"
        )
    code_values = frame.index.get_level_values("code")
    if any(not isinstance(code, str) or not code.strip() for code in code_values):
        raise ValueError(f"FactorStore day index has blank/non-string codes: {path.as_posix()}")

    columns = tuple(frame.columns.tolist())
    if not columns:
        raise ValueError(f"FactorStore day has no factor columns: {path.as_posix()}")
    if any(not isinstance(column, str) or not column.strip() for column in columns):
        raise ValueError(f"FactorStore factor columns must be non-empty strings: {path.as_posix()}")
    if len(set(columns)) != len(columns):
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        raise ValueError(f"FactorStore day has duplicate factor columns {duplicates[:10]}: {path.as_posix()}")

    infinite_columns: list[str] = []
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            infinite_columns.append(column)
    if infinite_columns:
        raise ValueError(
            "FactorStore day has infinite factor values; source inputs must be finite or NaN before merge: "
            f"columns={infinite_columns[:10]} path={path.as_posix()}"
        )
    return columns, frame.index


def _scan_source_store(root: Path) -> SourceStore:
    """Read and validate every input file without mutating the source root."""

    factor_dir = _factor_dir(root)
    day_paths = _discover_day_paths(factor_dir=factor_dir, source_root=root)
    source_columns: tuple[str, ...] | None = None
    day_files: dict[date, DayFile] = {}
    for day, path in day_paths.items():
        frame = pd.read_parquet(path)
        columns, index = _validated_frame(frame=frame, day=day, path=path)
        if source_columns is None:
            source_columns = columns
        elif columns != source_columns:
            raise ValueError(
                "FactorStore has a factor-column schema drift within one source root: "
                f"day={day.isoformat()} path={path.as_posix()}"
            )
        stat = path.stat()
        day_files[day] = DayFile(
            day=day,
            path=path,
            sha256=_sha256(path),
            bytes=int(stat.st_size),
            row_count=len(frame),
            columns=columns,
            index=index,
        )
    if source_columns is None:
        raise AssertionError("non-empty day path map did not produce source columns")
    return SourceStore(
        root=root,
        factor_dir=factor_dir,
        days=tuple(day_files),
        columns=source_columns,
        day_files=day_files,
    )


def _assert_same_calendar(sources: Sequence[SourceStore]) -> tuple[date, ...]:
    reference_days = set(sources[0].days)
    for source in sources[1:]:
        candidate_days = set(source.days)
        if candidate_days != reference_days:
            missing = sorted(reference_days - candidate_days)
            extra = sorted(candidate_days - reference_days)
            raise ValueError(
                "FactorStore calendar mismatch against first input: "
                f"root={source.root.as_posix()} "
                f"missing={[day.isoformat() for day in missing[:10]]} "
                f"extra={[day.isoformat() for day in extra[:10]]}"
            )
    return tuple(sorted(reference_days))


def _assert_disjoint_columns(sources: Sequence[SourceStore]) -> tuple[str, ...]:
    seen: set[str] = set()
    columns: list[str] = []
    for source in sources:
        overlap = sorted(seen.intersection(source.columns))
        if overlap:
            raise ValueError(
                "duplicate factor columns across input FactorStores are forbidden: "
                f"root={source.root.as_posix()} overlap={overlap[:20]}"
            )
        seen.update(source.columns)
        columns.extend(source.columns)
    return tuple(columns)


def _build_outer_union_indexes(
    sources: Sequence[SourceStore],
    days: Iterable[date],
) -> dict[date, pd.MultiIndex]:
    """Build each deterministic output index without weakening source validation.

    Every source file has already proved that its own ``(dt, code)`` index is
    unique and belongs to the filename date.  Cross-root equality is neither a
    PIT guarantee nor a valid reason to remove an allowlisted code.  Keep the
    full union instead, with per-source absence represented as ``NaN`` at
    execution time.
    """

    output_indexes: dict[date, pd.MultiIndex] = {}
    for day in days:
        union_index = sources[0].day_files[day].index
        for source in sources[1:]:
            union_index = union_index.union(source.day_files[day].index, sort=False)
        union_index = union_index.sort_values()
        if not isinstance(union_index, pd.MultiIndex) or union_index.has_duplicates:
            raise AssertionError(f"validated FactorStore union is not a unique MultiIndex: {day.isoformat()}")
        if list(union_index.names) != ["dt", "code"]:
            raise AssertionError(
                "validated FactorStore union index names drifted: "
                f"day={day.isoformat()} names={list(union_index.names)!r}"
            )
        output_indexes[day] = union_index
    return output_indexes


def _parse_family_catalog(path_value: str | Path, *, available_columns: Sequence[str]) -> FamilyCatalog:
    """Validate a supplied catalog as an explicit output-selection contract.

    The catalog may intentionally select a vetted subset of a larger source
    FactorStore.  After the selection is applied, its signals must therefore
    exactly equal the output columns.  This makes it possible to merge the
    strict v3 store's vetted candidates with later expansion stores without
    copying its deliberately excluded candidates into the unified screen.
    """

    source_path = _resolved(path_value)
    if not source_path.is_file():
        raise FileNotFoundError(f"family catalog does not exist: {source_path.as_posix()}")
    source_text = source_path.read_text(encoding="utf-8")
    try:
        raw: object = json.loads(source_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"family catalog is not valid JSON: {source_path.as_posix()}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("family catalog must be a JSON object")
    payload: object = raw.get("families") if set(raw) == {"families"} else raw
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("family catalog is empty or malformed")

    family_to_signals: OrderedDict[str, tuple[str, ...]] = OrderedDict()
    signal_to_family: dict[str, str] = {}
    values = list(payload.values())
    if all(isinstance(value, list) for value in values):
        for family_value, signals_value in payload.items():
            family = str(family_value).strip()
            if not family or not signals_value:
                raise ValueError("family catalog has an empty family name or family signal list")
            signals = tuple(str(signal).strip() for signal in signals_value)
            if any(not signal for signal in signals):
                raise ValueError(f"family catalog has an empty signal in family {family!r}")
            family_to_signals[family] = signals
            for signal in signals:
                prior = signal_to_family.setdefault(signal, family)
                if prior != family:
                    raise ValueError(
                        f"family catalog maps signal {signal!r} to multiple families: {prior!r}, {family!r}"
                    )
    elif all(isinstance(value, str) for value in values):
        by_family: OrderedDict[str, list[str]] = OrderedDict()
        for signal_value, family_value in payload.items():
            signal = str(signal_value).strip()
            family = str(family_value).strip()
            if not signal or not family:
                raise ValueError("factor-to-family catalog entries must be non-empty strings")
            prior = signal_to_family.setdefault(signal, family)
            if prior != family:
                raise ValueError(
                    f"family catalog maps signal {signal!r} to multiple families: {prior!r}, {family!r}"
                )
            by_family.setdefault(family, []).append(signal)
        family_to_signals = OrderedDict((family, tuple(signals)) for family, signals in by_family.items())
    else:
        raise ValueError("family catalog must map family->list[signal] or signal->family")

    all_catalog_signals = [signal for signals in family_to_signals.values() for signal in signals]
    if len(set(all_catalog_signals)) != len(all_catalog_signals):
        duplicates = sorted({signal for signal in all_catalog_signals if all_catalog_signals.count(signal) > 1})
        raise ValueError(f"family catalog has duplicate signals: {duplicates[:20]}")
    output_set = set(available_columns)
    catalog_set = set(all_catalog_signals)
    if not catalog_set.issubset(output_set):
        extra = sorted(catalog_set - output_set)
        raise ValueError(
            "family catalog has signals absent from the input FactorStores: "
            f"extra={extra[:20]}"
        )
    return FamilyCatalog(
        source_path=source_path,
        sha256=_sha256(source_path),
        source_text=source_text,
        family_to_signals=family_to_signals,
    )


def preflight_merge(
    *,
    input_roots: Sequence[str],
    output_root: str,
    family_catalog: str | None = None,
) -> MergePlan:
    """Return a fully validated no-write plan for a FactorStore merge."""

    validated_input_roots, validated_output_root = _validate_roots(input_roots, output_root)
    sources = tuple(_scan_source_store(root) for root in validated_input_roots)
    days = _assert_same_calendar(sources)
    all_input_columns = _assert_disjoint_columns(sources)
    output_indexes = _build_outer_union_indexes(sources, days)
    catalog = _parse_family_catalog(family_catalog, available_columns=all_input_columns) if family_catalog else None
    selected_signal_set = (
        {signal for signals in catalog.family_to_signals.values() for signal in signals}
        if catalog is not None
        else set(all_input_columns)
    )
    selected_columns_by_source = tuple(
        tuple(column for column in source.columns if column in selected_signal_set)
        for source in sources
    )
    empty_sources = [
        source.root.as_posix()
        for source, selected_columns in zip(sources, selected_columns_by_source, strict=True)
        if not selected_columns
    ]
    if empty_sources:
        raise ValueError(
            "family catalog would select zero factors from one or more input roots; "
            f"refusing an accidental partial merge: {empty_sources}"
        )
    output_columns = tuple(
        column for selected_columns in selected_columns_by_source for column in selected_columns
    )
    if catalog is not None and set(output_columns) != selected_signal_set:
        raise AssertionError("validated family catalog did not exactly define selected output columns")
    return MergePlan(
        sources=sources,
        selected_columns_by_source=selected_columns_by_source,
        output_root=validated_output_root,
        days=days,
        output_indexes=output_indexes,
        output_columns=output_columns,
        family_catalog=catalog,
    )


def _recheck_day_file(day_file: DayFile) -> pd.DataFrame:
    """Protect execution from a source mutation after preflight."""

    if not day_file.path.is_file():
        raise RuntimeError(f"source FactorStore file disappeared after preflight: {day_file.path.as_posix()}")
    if _sha256(day_file.path) != day_file.sha256:
        raise RuntimeError(f"source FactorStore file changed after preflight: {day_file.path.as_posix()}")
    frame = pd.read_parquet(day_file.path)
    columns, index = _validated_frame(frame=frame, day=day_file.day, path=day_file.path)
    if (
        columns != day_file.columns
        or len(frame) != day_file.row_count
        or not index.equals(day_file.index)
    ):
        raise RuntimeError(f"source FactorStore frame changed after preflight: {day_file.path.as_posix()}")
    return frame


def _write_factor_source_audit(*, root: Path, plan: MergePlan) -> Path:
    rows: list[dict[str, object]] = []
    for source_position, (source, selected_columns) in enumerate(
        zip(plan.sources, plan.selected_columns_by_source, strict=True),
        start=1,
    ):
        for column in selected_columns:
            rows.append(
                {
                    "factor": column,
                    "source_position": source_position,
                    "source_root": _path_text(source.root),
                }
            )
    audit_path = root / "factor_source_audit.csv"
    pd.DataFrame(rows, columns=["factor", "source_position", "source_root"]).to_csv(
        audit_path,
        index=False,
        encoding="utf-8",
    )
    return audit_path


def _source_manifest_entry(
    source: SourceStore,
    *,
    position: int,
    selected_columns: Sequence[str],
) -> dict[str, object]:
    return {
        "position": position,
        "root": _path_text(source.root),
        "panel_name": _PANEL_NAME,
        "day_count": len(source.days),
        "column_count": len(source.columns),
        "columns": list(source.columns),
        "selected_column_count": len(selected_columns),
        "selected_columns": list(selected_columns),
        "files": [
            {
                "trade_day": day.isoformat(),
                "relative_path": day_file.path.relative_to(source.root).as_posix(),
                "sha256": day_file.sha256,
                "bytes": day_file.bytes,
                "row_count": day_file.row_count,
                "column_count": len(day_file.columns),
            }
            for day, day_file in source.day_files.items()
        ],
    }


def _write_manifest(
    *,
    staging_root: Path,
    plan: MergePlan,
    output_files: Sequence[dict[str, object]],
    family_catalog_path: Path | None,
    audit_path: Path,
) -> Path:
    manifest_path = staging_root / "factor_mining_store_merge_manifest.json"
    manifest = {
        "research_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher": str(Path(__file__).resolve()),
        "panel_name": _PANEL_NAME,
        "output_root": _path_text(plan.output_root),
        "day_count": len(plan.days),
        "days": [day.isoformat() for day in plan.days],
        "output_column_count": len(plan.output_columns),
        "output_columns": list(plan.output_columns),
        "index_alignment": {
            "mode": "per_day_outer_union_missing_source_values_are_nan",
            "output_index_order": "sorted_dt_code",
            "source_index_contract": "unique_two_level_dt_code_per_file",
        },
        "output_files": list(output_files),
        "sources": [
            _source_manifest_entry(source, position=position, selected_columns=selected_columns)
            for position, (source, selected_columns) in enumerate(
                zip(plan.sources, plan.selected_columns_by_source, strict=True),
                start=1,
            )
        ],
        "family_catalog": {
            "supplied": plan.family_catalog is not None,
            "source_path": _path_text(plan.family_catalog.source_path) if plan.family_catalog else None,
            "sha256": plan.family_catalog.sha256 if plan.family_catalog else None,
            "copied_to": family_catalog_path.name if family_catalog_path else None,
            "audit_mapping": audit_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def execute_merge(plan: MergePlan) -> Path:
    """Write a fully validated merge through a scratch staging root only.

    A failed write intentionally leaves its uniquely named staging root in
    place for diagnosis; the requested output root remains absent and is never
    reused or overwritten by a retry.
    """

    if plan.output_root.exists():
        raise FileExistsError(f"output root appeared after preflight: {plan.output_root.as_posix()}")
    staging_root = plan.output_root.parent / f".{plan.output_root.name}.merge_staging_{uuid4().hex}"
    if staging_root.exists():
        raise FileExistsError(f"unexpected staging root already exists: {staging_root.as_posix()}")
    if not _is_strict_child(staging_root, _RESEARCH_SCRATCH_PARENT):
        raise RuntimeError(f"staging root escaped research scratch: {staging_root.as_posix()}")
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    output_files: list[dict[str, object]] = []
    try:
        for day in plan.days:
            target_index = plan.output_indexes[day]
            frames = [
                _recheck_day_file(source.day_files[day])
                .loc[:, list(selected_columns)]
                .reindex(target_index)
                for source, selected_columns in zip(plan.sources, plan.selected_columns_by_source, strict=True)
            ]
            if any(not frame.index.equals(target_index) for frame in frames):
                raise RuntimeError(f"source index alignment drifted during merge for {day.isoformat()}")
            merged = pd.concat(frames, axis=1)
            if not merged.index.equals(target_index):
                raise RuntimeError(f"merged output index drifted for {day.isoformat()}")
            if tuple(merged.columns.tolist()) != plan.output_columns:
                raise RuntimeError(f"merged output column contract drifted for {day.isoformat()}")
            if merged.columns.duplicated().any():
                raise RuntimeError(f"merged output has duplicate factor columns for {day.isoformat()}")
            output_path = staging_root / "factors" / _PANEL_NAME / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(output_path, index=True)
            output_files.append(
                {
                    "trade_day": day.isoformat(),
                    "relative_path": output_path.relative_to(staging_root).as_posix(),
                    "sha256": _sha256(output_path),
                    "bytes": int(output_path.stat().st_size),
                    "row_count": len(merged),
                    "column_count": len(merged.columns),
                    "source_row_counts": [
                        int(source.day_files[day].row_count) for source in plan.sources
                    ],
                    "source_missing_rows": [
                        int(len(merged) - source.day_files[day].row_count) for source in plan.sources
                    ],
                }
            )

        audit_path = _write_factor_source_audit(root=staging_root, plan=plan)
        family_catalog_path: Path | None = None
        if plan.family_catalog is not None:
            family_catalog_path = staging_root / "factor_mining_family_catalog.json"
            family_catalog_path.write_text(plan.family_catalog.source_text, encoding="utf-8")
        _write_manifest(
            staging_root=staging_root,
            plan=plan,
            output_files=output_files,
            family_catalog_path=family_catalog_path,
            audit_path=audit_path,
        )
        if plan.output_root.exists():
            raise FileExistsError(f"output root appeared during merge: {plan.output_root.as_posix()}")
        staging_root.replace(plan.output_root)
    except Exception:
        # Do not remove diagnostic staging data automatically.  It was created
        # by this invocation, is scratch-only, and is never mistaken for the
        # requested output root because its name is intentionally distinct.
        raise
    return plan.output_root


def _preflight_summary(plan: MergePlan, *, execute: bool) -> dict[str, object]:
    return {
        "research_only": True,
        "execute_requested": execute,
        "panel_name": _PANEL_NAME,
        "output_root": _path_text(plan.output_root),
        "output_root_exists_before_execution": plan.output_root.exists(),
        "calendar": {
            "day_count": len(plan.days),
            "first": plan.days[0].isoformat(),
            "last": plan.days[-1].isoformat(),
        },
        "output_column_count": len(plan.output_columns),
        "index_alignment": {
            "mode": "per_day_outer_union_missing_source_values_are_nan",
            "minimum_output_rows": min(len(plan.output_indexes[day]) for day in plan.days),
            "maximum_output_rows": max(len(plan.output_indexes[day]) for day in plan.days),
        },
        "sources": [
            {
                "root": _path_text(source.root),
                "day_count": len(source.days),
                "column_count": len(source.columns),
                "selected_column_count": len(selected_columns),
                "file_count": len(source.day_files),
            }
            for source, selected_columns in zip(plan.sources, plan.selected_columns_by_source, strict=True)
        ],
        "family_catalog": {
            "supplied": plan.family_catalog is not None,
            "mode": "validated_selection_and_copy" if plan.family_catalog else "factor_source_audit_only",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="repeat for each read-only scratch FactorStore root",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="new, previously absent merged FactorStore root under D:/cbond_on/research_scratch",
    )
    parser.add_argument(
        "--family-catalog",
        help="optional combined family JSON; it explicitly selects and is copied with the merged factor columns",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write the merge after complete read-only preflight; default is preflight only",
    )
    args = parser.parse_args(argv)

    plan = preflight_merge(
        input_roots=list(args.input_root),
        output_root=args.output_root,
        family_catalog=args.family_catalog,
    )
    print(json.dumps(_preflight_summary(plan, execute=bool(args.execute)), ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    output_root = execute_merge(plan)
    print(json.dumps({"research_only": True, "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
