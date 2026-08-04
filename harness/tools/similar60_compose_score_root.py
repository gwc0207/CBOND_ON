"""Compose a research-only score root with exact Regsim fallback days.

The generic LGBM runner can fall back from a Similar60 selector to ordinary
rolling training when its strict state history is incomplete.  A rolling score
is not a Similar60 observation.  This tool creates a *separate* score root in
which every such fallback day is copied byte-for-byte from the frozen Regsim
baseline.  It deliberately never edits either input root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import re
from typing import Sequence

import pandas as pd


_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _daily_score_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"score root missing: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.glob("*/*.csv")):
        day = path.stem
        if not _DAY_RE.fullmatch(day):
            raise ValueError(f"unexpected score filename: {path}")
        expected_month = day[:7]
        if path.parent.name != expected_month:
            raise ValueError(
                "daily score file must be under its YYYY-MM directory: "
                f"{path}"
            )
        if day in files:
            raise ValueError(f"duplicate score day {day}: {files[day]} and {path}")
        files[day] = path
    if not files:
        raise FileNotFoundError(f"no daily score CSVs under: {root}")
    return files


def _fallback_days(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"rolling similar-day audit missing: {path}")
    frame = pd.read_csv(path, usecols=["target_day", "reason"])
    if frame.empty:
        return set()
    frame["target_day"] = frame["target_day"].astype(str)
    if not frame["target_day"].map(lambda value: bool(_DAY_RE.fullmatch(value))).all():
        invalid = frame.loc[
            ~frame["target_day"].map(lambda value: bool(_DAY_RE.fullmatch(value))),
            "target_day",
        ].head(5).tolist()
        raise ValueError(f"invalid target_day in rolling similar-day audit: {invalid}")
    reasons = frame["reason"].fillna("").astype(str).str.strip().str.lower()
    return set(frame.loc[reasons.ne("ok"), "target_day"])


def compose_score_root(
    *,
    candidate_score_root: str | Path,
    baseline_score_root: str | Path,
    rolling_similar_days_path: str | Path,
    output_root: str | Path,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    """Create a bounded composed score root and a source/destination hash audit.

    ``start`` / ``end`` select an inclusive score-day range.  Every candidate
    day within the range must also exist in the baseline root; this makes the
    later pair audit fail closed instead of silently comparing an unpaired
    candidate day.  Any fallback day must have been emitted by the candidate
    runner too, so an absent score cannot be masked as a legitimate fallback.
    """

    candidate_root = Path(candidate_score_root)
    baseline_root = Path(baseline_score_root)
    similar_audit = Path(rolling_similar_days_path)
    target = Path(output_root)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite composed score root: {target}")
    if start is not None and not _DAY_RE.fullmatch(start):
        raise ValueError(f"start must be YYYY-MM-DD: {start}")
    if end is not None and not _DAY_RE.fullmatch(end):
        raise ValueError(f"end must be YYYY-MM-DD: {end}")
    if start is not None and end is not None and start > end:
        raise ValueError("start must be <= end")

    candidate_files = _daily_score_files(candidate_root)
    baseline_files = _daily_score_files(baseline_root)
    fallback_days = _fallback_days(similar_audit)
    selected_days = [
        day
        for day in sorted(candidate_files)
        if (start is None or day >= start) and (end is None or day <= end)
    ]
    if not selected_days:
        raise RuntimeError("no candidate score days in the requested compose range")
    unknown_fallback = sorted(fallback_days.difference(candidate_files))
    if unknown_fallback:
        raise ValueError(
            "rolling similar-day fallback has no candidate score file: "
            f"{unknown_fallback[:5]}"
        )
    fallback_days = fallback_days.intersection(selected_days)
    missing_baseline = [day for day in selected_days if day not in baseline_files]
    if missing_baseline:
        raise FileNotFoundError(
            "baseline score is missing for candidate compose days: "
            f"{missing_baseline[:5]}"
        )

    target.mkdir(parents=True, exist_ok=False)
    # Do not recursively delete an explicit user-supplied output path if a
    # later integrity check fails.  The sentinel makes an interrupted root
    # visibly invalid; successful completion removes it only after writing the
    # final manifest.
    incomplete = target / ".compose_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    records: list[dict[str, object]] = []
    try:
        for day in selected_days:
            candidate_path = candidate_files[day]
            baseline_path = baseline_files[day]
            is_fallback = day in fallback_days
            source_path = baseline_path if is_fallback else candidate_path
            destination = target / day[:7] / f"{day}.csv"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination)
            source_hash = _sha256(source_path)
            destination_hash = _sha256(destination)
            if source_hash != destination_hash:
                raise RuntimeError(f"non-identical score copy for {day}")
            records.append(
                {
                    "score_day": day,
                    "selection": "baseline_exact_fallback" if is_fallback else "candidate_similar60",
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": _sha256(candidate_path),
                    "baseline_path": str(baseline_path),
                    "baseline_sha256": _sha256(baseline_path),
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "destination_path": str(destination),
                    "destination_sha256": destination_hash,
                    "bytes": int(destination.stat().st_size),
                }
            )
        audit = pd.DataFrame(records)
        audit.to_csv(target / "compose_audit.csv", index=False, encoding="utf-8")
        audit_hash = _sha256(target / "compose_audit.csv")
        manifest = {
            "schema_version": 1,
            "purpose": "research-only Similar60 score composition; no input score roots modified",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_score_root": str(candidate_root),
            "baseline_score_root": str(baseline_root),
            "rolling_similar_days_path": str(similar_audit),
            "rolling_similar_days_sha256": _sha256(similar_audit),
            "range": {"start": start, "end": end},
            "score_days": int(len(selected_days)),
            "fallback_days": sorted(fallback_days),
            "fallback_count": int(len(fallback_days)),
            "compose_audit": {"path": "compose_audit.csv", "sha256": audit_hash},
            "invariant": (
                "Every baseline_exact_fallback row has an output CSV whose SHA-256 equals "
                "the frozen baseline input file; all other days equal their candidate input."
            ),
        }
        (target / "compose_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        incomplete.unlink()
    except Exception:
        # A failed compose must never look like a usable root.  Its explicitly
        # created sentinel remains, and no source root has been changed.
        raise
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-score-root", required=True)
    parser.add_argument("--baseline-score-root", required=True)
    parser.add_argument("--rolling-similar-days", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = compose_score_root(
        candidate_score_root=args.candidate_score_root,
        baseline_score_root=args.baseline_score_root,
        rolling_similar_days_path=args.rolling_similar_days,
        output_root=args.output_root,
        start=args.start,
        end=args.end,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
