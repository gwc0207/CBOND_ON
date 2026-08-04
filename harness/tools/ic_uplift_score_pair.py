"""Audit and evaluate one isolated model score stream against a fixed baseline.

The ``audit`` stage reads score CSVs only and freezes a score-day/code-universe
artifact.  The separate ``evaluate`` stage refuses to run without that frozen
artifact, then opens same-score-day 14:42 labels.  It is intended for a single
pre-registered research candidate, not model or strategy selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SHA256_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Hash one immutable audit input without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_SHA256_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_code_set_sha256(codes: Sequence[str] | set[str]) -> str:
    """Hash an allowlist independent of source-file row order."""
    payload = "\n".join(sorted({str(code) for code in codes})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"audit input is missing: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _same_file_evidence(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return (
        str(left.get("path")) == str(right.get("path"))
        and int(left.get("bytes", -1)) == int(right.get("bytes", -1))
        and str(left.get("sha256")) == str(right.get("sha256"))
    )


def _verify_file_evidence(
    evidence: Mapping[str, object],
    *,
    context: str,
    path: Path | None = None,
) -> None:
    """Fail closed when an audited byte stream is missing or has changed."""
    source = path or Path(str(evidence.get("path", "")))
    expected_hash = str(evidence.get("sha256", ""))
    try:
        expected_bytes = int(evidence["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"integrity evidence is incomplete for {context}") from exc
    if not expected_hash or not source.is_file():
        raise ValueError(f"integrity verification failed for {context}: source missing")
    actual = _file_evidence(source)
    if (
        int(actual["bytes"]) != expected_bytes
        or str(actual["sha256"]) != expected_hash
    ):
        raise ValueError(f"integrity verification failed for {context}: SHA-256 drift")


def _artifact_evidence(output_root: Path, path: Path) -> dict[str, object]:
    try:
        artifact = path.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"audit artifact is outside output root: {path}") from exc
    evidence = _file_evidence(path)
    return {
        "artifact": artifact.as_posix(),
        "bytes": evidence["bytes"],
        "sha256": evidence["sha256"],
    }


def _verify_artifact_evidence(
    output_root: Path,
    evidence: Mapping[str, object],
    *,
    context: str,
) -> Path:
    artifact = Path(str(evidence.get("artifact", "")))
    if not artifact.parts or artifact.is_absolute() or ".." in artifact.parts:
        raise ValueError(f"integrity evidence has an unsafe artifact path for {context}")
    path = output_root / artifact
    _verify_file_evidence(evidence, context=context, path=path)
    return path


def _day_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"score root missing: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.csv")):
        day = path.stem
        if not _DAY_RE.match(day):
            raise ValueError(f"unexpected score filename: {path}")
        if day in files:
            raise ValueError(f"duplicate score day {day}: {files[day]} and {path}")
        files[day] = path
    if not files:
        raise FileNotFoundError(f"no daily score CSVs under: {root}")
    return files


def _read_score(path: Path, expected_day: str, column: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    actual_days = set(frame["trade_date"].dropna().astype(str))
    if actual_days != {expected_day}:
        raise ValueError(f"score date mismatch: file={path}, values={sorted(actual_days)}")
    frame["code"] = frame["code"].astype(str)
    frame[column] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame[["code", column]].dropna()
    if frame["code"].duplicated().any():
        raise ValueError(f"duplicate code in score file: {path}")
    if frame.empty or not np.isfinite(frame[column].to_numpy(dtype=float)).all():
        raise ValueError(f"no finite scores in: {path}")
    return frame


def _read_score_with_evidence(
    path: Path,
    expected_day: str,
    column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Bind the parsed score values to stable source bytes during audit."""
    before = _file_evidence(path)
    frame = _read_score(path, expected_day, column)
    after = _file_evidence(path)
    if not _same_file_evidence(before, after):
        raise RuntimeError(f"score source changed while audit was reading it: {path}")
    return frame, after


def _pool_snapshot_path(
    *,
    raw_data_root: Path,
    pool_table: str,
    pool_day: str,
) -> Path:
    try:
        day = date.fromisoformat(pool_day)
    except ValueError as exc:
        raise ValueError(f"invalid resolved T-1 pool day: {pool_day!r}") from exc
    return (
        raw_data_root
        / pool_table.replace(".", "__")
        / f"{day.year:04d}-{day.month:02d}"
        / f"{day:%Y%m%d}.parquet"
    )


def _canonical_pool_codes_frame(codes_by_day: Mapping[str, set[str]]) -> pd.DataFrame:
    rows = [
        {"score_day": str(day), "code": str(code)}
        for day in sorted(codes_by_day)
        for code in sorted(codes_by_day[day])
    ]
    frame = pd.DataFrame(rows, columns=["score_day", "code"])
    if frame.empty or frame.duplicated(["score_day", "code"]).any():
        raise ValueError("fixed pool code artifact must contain unique non-empty score-day/code rows")
    return frame


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_fixed_pool_audit(
    fixed_pool_audit: pd.DataFrame,
    codes_by_day: Mapping[str, set[str]],
) -> None:
    required = {
        "score_day",
        "pool_day_expected",
        "pool_day_used",
        "pool_codes",
        "fallback_no_filter",
        "allowlist_codes_sha256",
        "pool_file_path",
        "pool_file_bytes",
        "pool_file_sha256",
    }
    missing = sorted(required.difference(fixed_pool_audit.columns))
    if missing:
        raise ValueError(f"fixed pool audit missing integrity columns: {missing}")
    if fixed_pool_audit["score_day"].astype(str).duplicated().any():
        raise ValueError("fixed pool audit has duplicate score days")
    audit_days = set(fixed_pool_audit["score_day"].astype(str))
    if audit_days != set(codes_by_day):
        raise ValueError("fixed pool audit days do not match canonical allowlist days")
    by_day = fixed_pool_audit.copy()
    by_day["score_day"] = by_day["score_day"].astype(str)
    by_day = by_day.set_index("score_day")
    for day, codes in codes_by_day.items():
        row = by_day.loc[str(day)]
        if _truthy(row["fallback_no_filter"]):
            raise ValueError(f"fixed pool audit recorded a no-filter fallback: {day}")
        if int(row["pool_codes"]) != len(codes):
            raise ValueError(f"fixed pool code count drift for score day: {day}")
        if str(row["allowlist_codes_sha256"]) != _canonical_code_set_sha256(codes):
            raise ValueError(f"fixed pool canonical code hash drift for score day: {day}")
        evidence = {
            "path": row["pool_file_path"],
            "bytes": row["pool_file_bytes"],
            "sha256": row["pool_file_sha256"],
        }
        _verify_file_evidence(evidence, context=f"T-1 pool parquet for score_day={day}")


def _fixed_pool_codes_by_day(
    *,
    raw_data_root: str | Path,
    score_days: Sequence[str],
) -> tuple[dict[str, set[str]], pd.DataFrame, dict[str, object]]:
    """Load the existing causal T-1 pool without opening any score-day label.

    This is deliberately an *evaluation-universe* filter, not a new mask.  It
    lets a score-only candidate be compared with an older score history whose
    files contain rows that downstream execution would already discard through
    the same upstream ``o_0005`` contract.
    """
    # The documented invocation is ``py harness/tools/...py``.  In that mode
    # Python adds the tool directory, rather than the repository root, to
    # ``sys.path``.  Make the optional repository import work in both that
    # direct-script form and module/test execution.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from cbond_on.infra.universe.pool_filter import (
        load_upstream_pool_config,
        resolve_pool_codes_for_trade_day,
    )

    root = Path(raw_data_root)
    cfg = load_upstream_pool_config()
    codes_by_day: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    for day_text in score_days:
        trade_day = date.fromisoformat(day_text)
        codes, info = resolve_pool_codes_for_trade_day(
            raw_data_root=root,
            trade_day=trade_day,
            pool_cfg=cfg,
        )
        if codes is None or bool(info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "fixed pool unavailable; refusing to compare with a relaxed universe: "
                f"score_day={day_text} expected_pool_day={info.get('pool_day_expected')} "
                f"reason={info.get('fallback_reason')}"
            )
        normalized = {str(code) for code in codes}
        if not normalized:
            raise RuntimeError(f"fixed pool resolved empty: score_day={day_text}")
        pool_day_used = str(info.get("pool_day_used") or "")
        pool_path = _pool_snapshot_path(
            raw_data_root=root,
            pool_table=cfg.pool_table,
            pool_day=pool_day_used,
        )
        pool_evidence = _file_evidence(pool_path)
        codes_by_day[day_text] = normalized
        rows.append(
            {
                "score_day": day_text,
                "pool_day_expected": str(info.get("pool_day_expected") or ""),
                "pool_day_used": pool_day_used,
                "pool_codes": int(len(normalized)),
                "fallback_no_filter": bool(info.get("fallback_no_filter", False)),
                "allowlist_codes_sha256": _canonical_code_set_sha256(normalized),
                "pool_file_path": pool_evidence["path"],
                "pool_file_bytes": pool_evidence["bytes"],
                "pool_file_sha256": pool_evidence["sha256"],
            }
        )
    return (
        codes_by_day,
        pd.DataFrame(rows),
        {
            "enabled": True,
            "kind": "existing_tminus1_o0005_allowlist",
            "raw_data_root": str(root),
            "pool_config": asdict(cfg),
            "artifact": "fixed_pool_audit.csv",
            "codes_artifact": "fixed_pool_codes.csv",
            "hash_algorithm": "sha256",
            "label_access": "none",
        },
    )


def _read_label(label_root: Path, day: str) -> pd.DataFrame | None:
    path = label_root / day[:7] / f"{day.replace('-', '')}.parquet"
    if not path.is_file():
        return None
    label = pd.read_parquet(path, columns=["code", "trade_time", "y"])
    trade_time = pd.to_datetime(label["trade_time"], errors="coerce")
    label = label.loc[
        (trade_time.dt.hour == 14) & (trade_time.dt.minute == 42),
        ["code", "y"],
    ].copy()
    label["code"] = label["code"].astype(str)
    label["y"] = pd.to_numeric(label["y"], errors="coerce")
    label = label.dropna()
    if label["code"].duplicated().any():
        raise ValueError(f"duplicate code in 14:42 label: {path}")
    return label if not label.empty else None


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _corr(frame: pd.DataFrame, score_col: str, method: str) -> float | None:
    subset = frame[[score_col, "y"]].dropna()
    if len(subset) < 30 or subset[score_col].nunique() < 2 or subset["y"].nunique() < 2:
        return None
    return _finite(subset[score_col].corr(subset["y"], method=method))


def _summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (prediction, partition), group in daily.groupby(["prediction", "partition"], sort=True):
        row: dict[str, object] = {
            "prediction": prediction,
            "partition": partition,
            "valid_days": int(len(group)),
            "first_score_day": str(group["score_day"].min()),
            "last_score_day": str(group["score_day"].max()),
            "avg_n": float(group["n"].mean()),
        }
        for metric in ("pearson_ic", "rank_ic", "top20_mean_y"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = _finite(values.mean()) if not values.empty else None
            row[f"t_{metric}"] = (
                _finite(values.mean() / values.std(ddof=1) * math.sqrt(len(values)))
                if len(values) > 1 and values.std(ddof=1) > 0.0
                else None
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["partition", "prediction"])


def _paired(daily: pd.DataFrame, baseline_name: str, candidate_name: str) -> pd.DataFrame:
    base = daily.loc[daily["prediction"] == baseline_name].set_index(["score_day", "partition"])
    candidate = daily.loc[daily["prediction"] == candidate_name].set_index(["score_day", "partition"])
    joined = candidate.join(
        base[["pearson_ic", "rank_ic", "top20_mean_y"]],
        how="inner",
        lsuffix="__candidate",
        rsuffix="__baseline",
    )
    rows: list[dict[str, object]] = []
    for partition, group in joined.groupby(level="partition", sort=True):
        for metric in ("pearson_ic", "rank_ic", "top20_mean_y"):
            delta = group[f"{metric}__candidate"] - group[f"{metric}__baseline"]
            delta = pd.to_numeric(delta, errors="coerce").dropna()
            rows.append(
                {
                    "partition": partition,
                    "metric": metric,
                    "shared_days": int(len(delta)),
                    "mean_delta": _finite(delta.mean()) if not delta.empty else None,
                    "t_delta": (
                        _finite(delta.mean() / delta.std(ddof=1) * math.sqrt(len(delta)))
                        if len(delta) > 1 and delta.std(ddof=1) > 0.0
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["partition", "metric"])


def audit(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if not (args.start <= args.validation_start <= args.end):
        raise ValueError("require start <= validation-start <= end")
    baseline_files = _day_files(Path(args.baseline_score_root))
    candidate_files = _day_files(Path(args.candidate_score_root))
    all_days = sorted(
        day
        for day in set(baseline_files).union(candidate_files)
        if args.start <= day <= args.end
    )
    if not all_days:
        raise RuntimeError("no score dates in the requested audit range")
    fixed_pool_raw_root = getattr(args, "fixed_pool_raw_root", None)
    fixed_pool_codes: dict[str, set[str]] | None = None
    fixed_pool_audit = pd.DataFrame()
    fixed_pool_codes_frame: pd.DataFrame | None = None
    universe_filter: dict[str, object] = {"enabled": False}
    if fixed_pool_raw_root:
        fixed_pool_codes, fixed_pool_audit, universe_filter = _fixed_pool_codes_by_day(
            raw_data_root=fixed_pool_raw_root,
            score_days=all_days,
        )
        _validate_fixed_pool_audit(fixed_pool_audit, fixed_pool_codes)
        fixed_pool_codes_frame = _canonical_pool_codes_frame(fixed_pool_codes)
    rows: list[dict[str, object]] = []
    frozen_pair_rows: list[pd.DataFrame] = []
    baseline_source_files: dict[str, dict[str, object]] = {}
    candidate_source_files: dict[str, dict[str, object]] = {}
    for day in all_days:
        baseline_path = baseline_files.get(day)
        candidate_path = candidate_files.get(day)
        row: dict[str, object] = {
            "score_day": day,
            "baseline_present": baseline_path is not None,
            "candidate_present": candidate_path is not None,
            "baseline_source_path": str(baseline_path) if baseline_path is not None else "",
            "baseline_source_bytes": 0,
            "baseline_source_sha256": "",
            "candidate_source_path": str(candidate_path) if candidate_path is not None else "",
            "candidate_source_bytes": 0,
            "candidate_source_sha256": "",
            "baseline_raw_codes": 0,
            "candidate_raw_codes": 0,
            "fixed_pool_codes": 0,
            "baseline_codes": 0,
            "candidate_codes": 0,
            "common_codes": 0,
            "same_code_universe": False,
            "candidate_contains_baseline_codes": False,
            "status": "missing_baseline" if baseline_path is None else "missing_candidate",
        }
        baseline: pd.DataFrame | None = None
        candidate: pd.DataFrame | None = None
        if baseline_path is not None:
            baseline, baseline_evidence = _read_score_with_evidence(
                baseline_path,
                day,
                "baseline_score",
            )
            baseline_source_files[day] = baseline_evidence
            row.update(
                {
                    "baseline_source_bytes": baseline_evidence["bytes"],
                    "baseline_source_sha256": baseline_evidence["sha256"],
                }
            )
        if candidate_path is not None:
            candidate, candidate_evidence = _read_score_with_evidence(
                candidate_path,
                day,
                "candidate_score",
            )
            candidate_source_files[day] = candidate_evidence
            row.update(
                {
                    "candidate_source_bytes": candidate_evidence["bytes"],
                    "candidate_source_sha256": candidate_evidence["sha256"],
                }
            )
        if baseline_path is not None and candidate_path is not None:
            assert baseline is not None
            assert candidate is not None
            row["baseline_raw_codes"] = int(len(baseline))
            row["candidate_raw_codes"] = int(len(candidate))
            if fixed_pool_codes is not None:
                allowed_codes = fixed_pool_codes[day]
                row["fixed_pool_codes"] = int(len(allowed_codes))
                baseline = baseline.loc[baseline["code"].isin(allowed_codes)].copy()
                candidate = candidate.loc[candidate["code"].isin(allowed_codes)].copy()
            baseline_codes = set(baseline["code"])
            candidate_codes = set(candidate["code"])
            row.update(
                {
                    "baseline_codes": len(baseline_codes),
                    "candidate_codes": len(candidate_codes),
                    "common_codes": len(baseline_codes.intersection(candidate_codes)),
                    "same_code_universe": baseline_codes == candidate_codes,
                    "candidate_contains_baseline_codes": baseline_codes.issubset(candidate_codes),
                    "status": (
                        "ok_exact_universe"
                        if baseline_codes == candidate_codes
                        else (
                            "ok_candidate_superset"
                            if baseline_codes.issubset(candidate_codes)
                            else "candidate_missing_baseline_codes"
                        )
                    ),
                }
            )
            frozen_pair = baseline.merge(
                candidate,
                on="code",
                how="inner",
                validate="one_to_one",
            )
            if not frozen_pair.empty:
                frozen_pair.insert(0, "score_day", day)
                frozen_pair_rows.append(frozen_pair)
        rows.append(row)
    audit_frame = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=False)
    audit_path = output_root / "score_universe_audit.csv"
    audit_frame.to_csv(audit_path, index=False, encoding="utf-8")
    fixed_pool_audit_path: Path | None = None
    fixed_pool_codes_path: Path | None = None
    if fixed_pool_codes is not None:
        assert fixed_pool_codes_frame is not None
        fixed_pool_audit_path = output_root / "fixed_pool_audit.csv"
        fixed_pool_codes_path = output_root / "fixed_pool_codes.csv"
        fixed_pool_audit.to_csv(fixed_pool_audit_path, index=False, encoding="utf-8")
        fixed_pool_codes_frame.to_csv(fixed_pool_codes_path, index=False, encoding="utf-8")
    frozen_pairs = (
        pd.concat(frozen_pair_rows, ignore_index=True)
        if frozen_pair_rows
        else pd.DataFrame(columns=["score_day", "code", "baseline_score", "candidate_score"])
    )
    frozen_pairs = frozen_pairs.sort_values(["score_day", "code"], kind="mergesort").reset_index(drop=True)
    frozen_pairs_path = output_root / "frozen_pair_scores.csv"
    frozen_pairs.to_csv(frozen_pairs_path, index=False, encoding="utf-8")
    integrity_artifacts: dict[str, dict[str, object]] = {
        "score_universe_audit": _artifact_evidence(output_root, audit_path),
        "frozen_pair_scores": _artifact_evidence(output_root, frozen_pairs_path),
    }
    if fixed_pool_audit_path is not None and fixed_pool_codes_path is not None:
        integrity_artifacts["fixed_pool_audit"] = _artifact_evidence(output_root, fixed_pool_audit_path)
        integrity_artifacts["fixed_pool_codes"] = _artifact_evidence(output_root, fixed_pool_codes_path)
    manifest = {
        "schema_version": 2,
        "stage": "score_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "research-only frozen pair-score universe audit; no labels read",
        "label_access": "none in audit stage",
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline_score_root": str(Path(args.baseline_score_root)),
        "candidate_score_root": str(Path(args.candidate_score_root)),
        "baseline_files": {day: str(path) for day, path in baseline_files.items() if args.start <= day <= args.end},
        "candidate_files": {day: str(path) for day, path in candidate_files.items() if args.start <= day <= args.end},
        "start": args.start,
        "end": args.end,
        "validation_start": args.validation_start,
        "universe_filter": universe_filter,
        "frozen_score_pairs": {
            "artifact": "frozen_pair_scores.csv",
            "rows": int(len(frozen_pairs)),
            "columns": ["score_day", "code", "baseline_score", "candidate_score"],
            "bytes": integrity_artifacts["frozen_pair_scores"]["bytes"],
            "sha256": integrity_artifacts["frozen_pair_scores"]["sha256"],
            "purpose": "exact score-day/code/value pairs frozen before label access",
        },
        "integrity": {
            "hash_algorithm": "sha256",
            "source_score_files": {
                "baseline": baseline_source_files,
                "candidate": candidate_source_files,
            },
            "artifacts": integrity_artifacts,
            "evaluation_contract": (
                "evaluate must re-hash every source score, fixed-pool input/artifact, "
                "and frozen pair artifact before opening any label"
            ),
        },
        "audit_summary": {
            "days_union": int(len(audit_frame)),
            "days_exact_code_universe": int((audit_frame["status"] == "ok_exact_universe").sum()),
            "days_candidate_superset": int((audit_frame["status"] == "ok_candidate_superset").sum()),
            "days_missing_baseline_coverage": int(
                (~audit_frame["candidate_contains_baseline_codes"]).sum()
            ),
        },
        "next_stage": "evaluate may open labels only after this artifact exists",
    }
    (output_root / "score_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_root


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"integrity evidence is missing or malformed for {context}")
    return {str(key): item for key, item in value.items()}


def _validate_source_score_integrity(manifest: Mapping[str, object]) -> None:
    integrity = _require_mapping(manifest.get("integrity"), context="score audit")
    sources = _require_mapping(integrity.get("source_score_files"), context="source scores")
    for stream, manifest_key in (("baseline", "baseline_files"), ("candidate", "candidate_files")):
        expected_paths = _require_mapping(manifest.get(manifest_key), context=f"{stream} source paths")
        evidence_by_day = _require_mapping(sources.get(stream), context=f"{stream} source scores")
        if set(evidence_by_day) != set(expected_paths):
            raise ValueError(f"integrity verification failed for {stream} source scores: day coverage drift")
        for day, raw_path in expected_paths.items():
            evidence = _require_mapping(evidence_by_day[day], context=f"{stream} score day={day}")
            if str(evidence.get("path", "")) != str(raw_path):
                raise ValueError(
                    f"integrity verification failed for {stream} source scores: path drift on {day}"
                )
            _verify_file_evidence(evidence, context=f"{stream} source score day={day}")


def _validate_fixed_pool_integrity(
    output_root: Path,
    *,
    manifest: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> None:
    universe_filter = _require_mapping(manifest.get("universe_filter"), context="universe filter")
    if not bool(universe_filter.get("enabled", False)):
        return
    audit_evidence = _require_mapping(artifacts.get("fixed_pool_audit"), context="fixed pool audit")
    codes_evidence = _require_mapping(artifacts.get("fixed_pool_codes"), context="fixed pool codes")
    audit_path = _verify_artifact_evidence(
        output_root,
        audit_evidence,
        context="fixed pool audit artifact",
    )
    codes_path = _verify_artifact_evidence(
        output_root,
        codes_evidence,
        context="fixed pool code artifact",
    )
    code_frame = pd.read_csv(codes_path, usecols=["score_day", "code"])
    if code_frame.empty or code_frame[["score_day", "code"]].isna().any().any():
        raise ValueError("fixed pool code artifact is empty or invalid")
    code_frame["score_day"] = code_frame["score_day"].astype(str)
    code_frame["code"] = code_frame["code"].astype(str)
    if code_frame.duplicated(["score_day", "code"]).any():
        raise ValueError("fixed pool code artifact has duplicate score-day/code rows")
    codes_by_day = {
        str(day): set(group["code"].tolist())
        for day, group in code_frame.groupby("score_day", sort=True)
    }
    _validate_fixed_pool_audit(pd.read_csv(audit_path), codes_by_day)


def _validate_audit_integrity_before_label_access(
    output_root: Path,
    manifest: Mapping[str, object],
) -> None:
    try:
        schema_version = int(manifest.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("score audit manifest has an invalid schema version") from exc
    if schema_version < 2:
        raise ValueError(
            "score audit manifest lacks the required SHA-256 integrity contract; rerun audit before evaluation"
        )
    integrity = _require_mapping(manifest.get("integrity"), context="score audit")
    if str(integrity.get("hash_algorithm", "")).lower() != "sha256":
        raise ValueError("score audit manifest does not declare SHA-256 integrity")
    artifacts = _require_mapping(integrity.get("artifacts"), context="score audit artifacts")
    _validate_source_score_integrity(manifest)
    _verify_artifact_evidence(
        output_root,
        _require_mapping(artifacts.get("score_universe_audit"), context="score universe audit"),
        context="score universe audit artifact",
    )
    _verify_artifact_evidence(
        output_root,
        _require_mapping(artifacts.get("frozen_pair_scores"), context="frozen pair scores"),
        context="frozen pair score artifact",
    )
    _validate_fixed_pool_integrity(output_root, manifest=manifest, artifacts=artifacts)


def evaluate(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    manifest_path = output_root / "score_audit_manifest.json"
    audit_path = output_root / "score_universe_audit.csv"
    frozen_pairs_path = output_root / "frozen_pair_scores.csv"
    if not manifest_path.is_file() or not audit_path.is_file() or not frozen_pairs_path.is_file():
        raise FileNotFoundError("score audit artifacts missing; run audit before opening labels")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # No score, pool, or frozen-pair drift may be accepted after this point.
    # This check intentionally precedes both parsing the evaluation inputs and
    # the first call that can open a same-score-day 14:42 label.
    _validate_audit_integrity_before_label_access(output_root, manifest)
    audit_frame = pd.read_csv(audit_path)
    baseline_covered = (
        audit_frame["candidate_contains_baseline_codes"].astype(str).str.lower().eq("true")
    )
    invalid = audit_frame.loc[~baseline_covered]
    if not invalid.empty:
        raise ValueError(
            "candidate score pair omits baseline codes; "
            f"invalid_days={invalid['score_day'].astype(str).tolist()[:5]}"
        )
    baseline_name = str(manifest["baseline_name"])
    candidate_name = str(manifest["candidate_name"])
    frozen_pairs = pd.read_csv(
        frozen_pairs_path,
        usecols=["score_day", "code", "baseline_score", "candidate_score"],
    )
    frozen_pairs["score_day"] = frozen_pairs["score_day"].astype(str)
    frozen_pairs["code"] = frozen_pairs["code"].astype(str)
    if frozen_pairs.duplicated(["score_day", "code"]).any():
        raise ValueError("duplicate score-day/code in frozen score-pair artifact")
    label_root = Path(args.label_root)
    rows: list[dict[str, object]] = []
    skipped_no_label: list[str] = []
    skipped_invalid: dict[str, str] = {}
    for day in audit_frame["score_day"].astype(str):
        scores = frozen_pairs.loc[
            frozen_pairs["score_day"] == day,
            ["code", "baseline_score", "candidate_score"],
        ].copy()
        if scores.empty:
            skipped_invalid[day] = "no frozen paired scores"
            continue
        label = _read_label(label_root, day)
        if label is None:
            skipped_no_label.append(day)
            continue
        merged = scores.merge(label, on="code", how="inner", validate="one_to_one")
        if len(merged) < 30:
            skipped_invalid[day] = f"only {len(merged)} label-aligned rows"
            continue
        partition = "development" if day < str(manifest["validation_start"]) else "validation"
        for prediction, score_col in ((baseline_name, "baseline_score"), (candidate_name, "candidate_score")):
            top = merged.nlargest(20, score_col)["y"]
            rows.append(
                {
                    "score_day": day,
                    "partition": partition,
                    "prediction": prediction,
                    "n": int(len(merged)),
                    "pearson_ic": _corr(merged, score_col, "pearson"),
                    "rank_ic": _corr(merged, score_col, "spearman"),
                    "top20_mean_y": _finite(top.mean()),
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise RuntimeError("evaluation produced no score/label-aligned daily metrics")
    summary = _summary(daily)
    paired = _paired(daily, baseline_name, candidate_name)
    daily.to_csv(output_root / "daily_metrics.csv", index=False, encoding="utf-8")
    summary.to_csv(output_root / "summary_metrics.csv", index=False, encoding="utf-8")
    paired.to_csv(output_root / "paired_vs_baseline.csv", index=False, encoding="utf-8")
    evaluation_manifest = {
        "schema_version": 1,
        "stage": "evaluate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_access": "same-score-day 14:42 labels only after frozen score audit",
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "time_range": {
            "start": manifest["start"],
            "end": manifest["end"],
            "validation_start": manifest["validation_start"],
        },
        "fixed_universe": (
            "audit required candidate coverage of every baseline score code after any "
            "recorded existing-universe filter; candidate-only extra score codes are "
            "retained in score artifacts but excluded from the baseline-paired metric"
        ),
        "universe_filter": manifest.get("universe_filter", {"enabled": False}),
        "score_input": "frozen_pair_scores.csv",
        "skipped_no_label": skipped_no_label,
        "skipped_invalid": skipped_invalid,
        "outputs": ["daily_metrics.csv", "summary_metrics.csv", "paired_vs_baseline.csv"],
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--baseline-score-root", required=True)
    audit_parser.add_argument("--candidate-score-root", required=True)
    audit_parser.add_argument("--output-root", required=True)
    audit_parser.add_argument("--start", required=True)
    audit_parser.add_argument("--end", required=True)
    audit_parser.add_argument("--validation-start", required=True)
    audit_parser.add_argument("--baseline-name", default="regsim")
    audit_parser.add_argument("--candidate-name", default="candidate")
    audit_parser.add_argument(
        "--fixed-pool-raw-root",
        default=None,
        help=(
            "apply the existing causal T-1 o_0005 pool to both score streams before "
            "freezing their pair universe; no score-day labels are read"
        ),
    )
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--output-root", required=True)
    evaluate_parser.add_argument("--label-root", default=r"D:\cbond_on\label_data")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = audit(args) if args.command == "audit" else evaluate(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
