"""Research-only A0 preflight for the frozen R88 P6/P3/Regsim trio.

This tool deliberately stops before score fusion or a generic backtest.  It
freezes and audits the already-produced score and return inputs required for a
later, separately authorised single-Top20 rank-fusion experiment.

It never trains or scores a model, computes factors, reads/writes a live
artifact, calls the scheduler, writes a database, or changes a repository
configuration.  All output is contained in a fresh research-scratch run root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
R88_STUDY_ROOT = Path(
    r"D:\cbond_on\research_scratch\r88_joint_factor_lgbm_20260828_r1"
    r"\runs\rolling_20250102_20260827"
)
R88_SCORE_ROOT = R88_STUDY_ROOT / "runtime" / "results" / "scores"
R88_BACKTEST_ROOT = R88_STUDY_ROOT / "runtime" / "results" / "backtest"
REGSIM_SCORE_ROOT = Path(
    r"D:\cbond_on\results\scores\live"
    r"\lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_50_20260805"
)
REGSIM_RETURN_PATH = Path(
    r"D:\cbond_on\results\analysis\model_switch_scoreopt_live_20260805_50f"
    r"\return_history\Challenger_Regsim.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\r88_trio_score_fusion_20260908_r1")

MODELS = {
    "P6": "r88_icir50__p6_lower_bagging",
    "P3": "r88_full88__p3_deep_regularized",
    "Regsim": "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_50_20260805",
}
R88_RETURN_FRAGMENTS = {
    "P6": "r88_icir50__p6_lower_bagging",
    "P3": "r88_full88__p3_deep_regularized",
}
REQUIRED_SCORE_COLUMNS = ("trade_date", "code", "score")
REQUIRED_RETURN_COLUMNS = ("trade_date", "day_return", "benchmark_return")


class PreflightError(RuntimeError):
    """Raised when frozen R88 trio inputs do not satisfy the A0 contract."""


@dataclass(frozen=True)
class ScoreRootAudit:
    model: str
    root: Path
    files: dict[pd.Timestamp, Path]


def _json_default(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_fresh_output_root(output_root: Path, run_name: str) -> Path:
    resolved_root = output_root.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved_root.relative_to(allowed)
    except ValueError as exc:
        raise PreflightError(f"output root must stay under {allowed}: {resolved_root}") from exc
    if not run_name or Path(run_name).name != run_name:
        raise PreflightError("run name must be exactly one non-empty directory leaf")
    run_root = resolved_root / run_name
    if run_root.exists():
        raise PreflightError(f"refusing to overwrite existing run root: {run_root}")
    return run_root


def _read_score_date(path: Path) -> pd.Timestamp:
    try:
        date_value = pd.Timestamp(path.stem).normalize()
    except ValueError as exc:
        raise PreflightError(f"invalid score file date in path: {path}") from exc
    if str(date_value.date()) != path.stem:
        raise PreflightError(f"score filename must be YYYY-MM-DD: {path}")
    return date_value


def score_file_map(root: Path, *, model: str) -> dict[pd.Timestamp, Path]:
    if not root.is_dir():
        raise PreflightError(f"missing {model} score root: {root}")
    result: dict[pd.Timestamp, Path] = {}
    for path in sorted(root.rglob("*.csv")):
        score_day = _read_score_date(path)
        if score_day in result:
            raise PreflightError(f"duplicate {model} score date {score_day.date()}: {path}")
        result[score_day] = path
    if not result:
        raise PreflightError(f"no {model} score files beneath {root}")
    return result


def read_score_codes(path: Path, *, expected_day: pd.Timestamp) -> set[str]:
    try:
        frame = pd.read_csv(path, usecols=list(REQUIRED_SCORE_COLUMNS))
    except ValueError as exc:
        raise PreflightError(f"{path} lacks required score columns {REQUIRED_SCORE_COLUMNS}") from exc
    if frame.empty:
        raise PreflightError(f"empty score file: {path}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if frame["trade_date"].isna().any() or not frame["trade_date"].eq(expected_day).all():
        raise PreflightError(f"score date payload does not match filename: {path}")
    if frame["code"].eq("").any() or frame["code"].duplicated().any():
        raise PreflightError(f"invalid or duplicate score code in {path}")
    if not frame["score"].map(math.isfinite).all():
        raise PreflightError(f"non-finite score in {path}")
    return set(frame["code"].tolist())


def locate_r88_return_path(fragment: str) -> Path:
    candidates = sorted(R88_BACKTEST_ROOT.rglob(f"*{fragment}*/**/daily_returns.csv"))
    if len(candidates) != 1:
        raise PreflightError(f"expected exactly one R88 daily_returns path for {fragment}, found {len(candidates)}")
    return candidates[0]


def read_returns(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise PreflightError(f"missing {label} return file: {path}")
    try:
        frame = pd.read_csv(path, usecols=list(REQUIRED_RETURN_COLUMNS))
    except ValueError as exc:
        raise PreflightError(f"{label} returns lack {REQUIRED_RETURN_COLUMNS}: {path}") from exc
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    for column in ("day_return", "benchmark_return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.isna().any().any() or frame["trade_date"].duplicated().any():
        raise PreflightError(f"invalid dates or values in {label} returns: {path}")
    return frame.sort_values("trade_date").reset_index(drop=True)


def _score_root_path(model: str) -> Path:
    if model == "Regsim":
        return REGSIM_SCORE_ROOT
    return R88_SCORE_ROOT / MODELS[model]


def audit_score_universes(score_roots: dict[str, ScoreRootAudit]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    calendars = {model: set(audit.files) for model, audit in score_roots.items()}
    common_days = sorted(set.intersection(*calendars.values()))
    if not common_days:
        raise PreflightError("the three model score calendars have no common date")
    missing_by_model = {
        model: [str(day.date()) for day in sorted(set.union(*calendars.values()) - dates)]
        for model, dates in calendars.items()
    }
    rows: list[dict[str, object]] = []
    exact_shared_days = 0
    for day in common_days:
        code_sets = {
            model: read_score_codes(audit.files[day], expected_day=day)
            for model, audit in score_roots.items()
        }
        intersection = set.intersection(*code_sets.values())
        union = set.union(*code_sets.values())
        exact_shared = len({frozenset(codes) for codes in code_sets.values()}) == 1
        exact_shared_days += int(exact_shared)
        rows.append(
            {
                "score_day": str(day.date()),
                **{f"{model.lower()}_score_count": len(codes) for model, codes in code_sets.items()},
                "intersection_count": len(intersection),
                "union_count": len(union),
                "intersection_to_union_ratio": len(intersection) / len(union),
                "identical_score_universe": exact_shared,
            }
        )
    daily = pd.DataFrame(rows)
    summary = {
        "common_score_day_count": len(common_days),
        "common_score_start": str(common_days[0].date()),
        "common_score_end": str(common_days[-1].date()),
        "identical_score_universe_days": exact_shared_days,
        "identical_score_universe_ratio": exact_shared_days / len(common_days),
        "mean_intersection_count": float(daily["intersection_count"].mean()),
        "mean_union_count": float(daily["union_count"].mean()),
        "mean_intersection_to_union_ratio": float(daily["intersection_to_union_ratio"].mean()),
        "fusion_readiness": "BLOCKED_UNTIL_MISSING_SCORE_POLICY_IS_FROZEN"
        if exact_shared_days != len(common_days)
        else "READY_FOR_IDENTICAL_UNIVERSE_RANK_FUSION",
    }
    return daily, summary, missing_by_model


def audit_return_alignment(return_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Any]]:
    returns = {model: read_returns(path, label=model) for model, path in return_paths.items()}
    indexed = {
        model: frame.set_index("trade_date")[["day_return", "benchmark_return"]].rename(
            columns={"day_return": f"{model.lower()}_day_return", "benchmark_return": f"{model.lower()}_benchmark_return"}
        )
        for model, frame in returns.items()
    }
    combined = pd.concat(indexed.values(), axis=1, join="inner").sort_index()
    if combined.empty:
        raise PreflightError("the three return histories have no common date")
    calendar_union = set.union(*(set(frame["trade_date"]) for frame in returns.values()))
    common_days = set(combined.index)
    missing = {
        model: [str(day.date()) for day in sorted(calendar_union - set(frame["trade_date"]))]
        for model, frame in returns.items()
    }
    benchmark_delta = (combined["p6_benchmark_return"] - combined["regsim_benchmark_return"]).abs()
    summary = {
        "common_return_day_count": int(len(combined)),
        "common_return_start": str(combined.index.min().date()),
        "common_return_end": str(combined.index.max().date()),
        "benchmark_mismatch_vs_regsim_days_gt_1e_12": int((benchmark_delta > 1e-12).sum()),
        "benchmark_max_abs_difference_vs_regsim": float(benchmark_delta.max()),
        "relative_comparison_policy": "raw_day_return_only_on_common_dates",
        "generic_regsim_identity": "PENDING_A1_GENERIC_RUNTIME_REPLAY",
    }
    combined = combined.reset_index().rename(columns={"trade_date": "score_day"})
    return combined, {"summary": summary, "missing_by_model": missing, "common_days": [str(day.date()) for day in sorted(common_days)]}


def build_input_manifest(score_roots: dict[str, ScoreRootAudit], return_paths: dict[str, Path]) -> dict[str, Any]:
    score_files = {
        model: [
            {"score_day": str(day.date()), "path": str(path), "sha256": sha256(path)}
            for day, path in sorted(audit.files.items())
        ]
        for model, audit in score_roots.items()
    }
    return {
        "schema": "r88_trio_score_fusion_a0_input_manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "live_runtime_called": False,
        "generic_backtest_called": False,
        "database_writes": False,
        "scheduler_called": False,
        "candidate_membership": MODELS,
        "score_files": score_files,
        "return_files": {
            model: {"path": str(path), "sha256": sha256(path)} for model, path in return_paths.items()
        },
        "r88_study_integrity": {
            "path": str(R88_STUDY_ROOT / "study_integrity.json"),
            "sha256": sha256(R88_STUDY_ROOT / "study_integrity.json"),
        },
    }


def run_a0_preflight(*, output_root: Path, run_name: str) -> Path:
    run_root = _assert_fresh_output_root(output_root, run_name)
    score_roots = {
        model: ScoreRootAudit(model=model, root=_score_root_path(model), files=score_file_map(_score_root_path(model), model=model))
        for model in MODELS
    }
    return_paths = {
        "P6": locate_r88_return_path(R88_RETURN_FRAGMENTS["P6"]),
        "P3": locate_r88_return_path(R88_RETURN_FRAGMENTS["P3"]),
        "Regsim": REGSIM_RETURN_PATH,
    }
    score_daily, score_summary, score_missing = audit_score_universes(score_roots)
    return_daily, return_audit = audit_return_alignment(return_paths)
    manifest = build_input_manifest(score_roots, return_paths)
    contract = {
        "schema": "r88_trio_score_fusion_contract/v1",
        "status": "A0_PREFLIGHT_COMPLETE_STAGE_B_NOT_AUTHORIZED",
        "research_question": "Can a frozen P6/P3/Regsim single-Top20 score fusion improve risk-adjusted performance without changing the existing trading contract?",
        "candidate_membership": MODELS,
        "execution_contract": {
            "universe": "existing o_0005 and all existing masks; intersection of score files is prohibited as a replacement universe",
            "strategy": "existing strategy01 Top20, 5 percent max single-name weight, full turnover",
            "execution": "existing strict cycle buy twap_1442_1457 and next-trading-day sell twap_0930_0939",
            "costs": "current unified fees config resolved only by a later generic identity replay",
        },
        "score_contract": {
            "normalization": "within-day percentile rank before weighted fusion",
            "required_before_stage_b": [
                "freeze an explicit missing-score policy that preserves the existing o_0005 universe",
                "complete generic Regsim raw-score identity replay",
                "owner authorization for a fresh research-only B-stage score-fusion run",
            ],
            "prohibited": [
                "silent score-universe intersection",
                "live configuration changes",
                "database writes",
                "scheduler operations",
                "model training or rescoring",
                "BaseGap, Champion, LCB, confidence threshold, veto, or hard-selector logic",
            ],
        },
        "evaluation_contract": {
            "development": "2025-01-02 through 2025-12-31, descriptive/design only because candidate selection already used this era",
            "reporting": "2026-01-05 through 2026-08-27, reporting only and never used to retune a later candidate",
            "relative_regsim": "raw day_return on exact common dates only until a unified generic backtest produces one shared benchmark",
            "promotion": "not allowed; any future promotion requires a separately frozen forward-shadow plan",
        },
        "a0_results": {"score_universe": score_summary, "return_alignment": return_audit["summary"]},
    }
    run_root.mkdir(parents=True, exist_ok=False)
    score_daily.to_csv(run_root / "a0_daily_score_universe.csv", index=False)
    return_daily.to_csv(run_root / "a0_common_return_alignment.csv", index=False)
    (run_root / "a0_score_calendar_missing_by_model.json").write_text(
        json.dumps(score_missing, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    (run_root / "a0_return_calendar_audit.json").write_text(
        json.dumps(return_audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    (run_root / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    (run_root / "study_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    status = {
        "status": "completed_a0_preflight",
        "run_root": str(run_root),
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "generic_backtest_called": False,
        "model_training_called": False,
        "model_scoring_called": False,
        "scheduler_called": False,
        "next_gate": "owner must approve an explicit missing-score policy and B-stage generic identity/fusion run",
    }
    (run_root / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="a0_preflight_20260908_r1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = run_a0_preflight(output_root=args.output_root, run_name=str(args.run_name))
    print(json.dumps({"status": "completed_a0_preflight", "run_root": str(run_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
