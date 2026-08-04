from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.trading_days import list_available_trading_days_from_raw
from cbond_on.infra.live.post_close_materials import inspect_post_close_readiness
from liveLaunch.attempt_journal import read_attempt_events


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _report_root(*, live_cfg: dict[str, Any], paths_cfg: dict[str, Any]) -> Path:
    cfg = dict(live_cfg.get("post_close_readiness", {}))
    return resolve_output_path(
        cfg.get("report_root"),
        default_path=Path(paths_cfg["results_root"]) / "ops" / "post_close_readiness",
        results_root=paths_cfg["results_root"],
    )


def _write_report(*, root: Path, score_day: date, report: dict[str, Any]) -> Path:
    checked_at = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = root / f"{score_day:%Y-%m-%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{checked_at}_{os.getpid()}.json"
    temp_path = out_path.with_suffix(".tmp")
    report["report_path"] = str(out_path)
    temp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp_path, out_path)
    return out_path


def _exit_code(disposition: str) -> int:
    if disposition in {"SKIPPED_HEALTHY", "SKIPPED_DISABLED", "SKIPPED_NON_TRADING_DAY", "SKIPPED_NO_INCIDENT", "READY_REPAIRED"}:
        return 0
    if disposition == "REPAIR_REQUIRED":
        return 2
    return 3


def _same_day_runtime_evidence(
    *,
    scheduler_state: dict[str, Any],
    attempt_events: list[dict[str, Any]],
    log_text: str,
    score_day: date,
) -> dict[str, Any]:
    day_text = str(score_day)
    state_today = str(scheduler_state.get("today", "")).strip()[:10] == day_text
    matching_events = [event for event in attempt_events if str(event.get("score_day", "")).strip()[:10] == day_text]
    run_lines = [line for line in log_text.splitlines() if "[run]" in line.lower()]
    failed_events = [
        event
        for event in matching_events
        if event.get("event") == "attempt_finished" and str(event.get("result", "")).strip().lower() == "failed"
    ]
    started_ids = {str(event.get("attempt_id", "")) for event in matching_events if event.get("event") == "attempt_started"}
    finished_ids = {str(event.get("attempt_id", "")) for event in matching_events if event.get("event") == "attempt_finished"}
    unfinished_ids = sorted(item for item in started_ids if item and item not in finished_ids)
    failed_log_lines = [line for line in run_lines if "[run] failed" in line.lower()]
    state_status = str(scheduler_state.get("status", "")).strip().lower()
    unresolved_state = state_today and state_status in {"failed", "running_live", "waiting_cutoff", "already_running"}
    return {
        "state_today": state_today,
        "state_status": state_status,
        "same_day_attempt_event_count": int(len(matching_events)),
        "legacy_run_line_count": int(len(run_lines)),
        "activity_seen": bool(state_today or matching_events or run_lines),
        "failed_attempt_count": int(len(failed_events)),
        "unfinished_attempt_ids": unfinished_ids,
        "failed_log_line_count": int(len(failed_log_lines)),
        "incident_hint": bool(failed_events or unfinished_ids or failed_log_lines or unresolved_state),
    }


def _operator_report(
    *,
    score_day: date,
    check_id: str,
    summary: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "score_day": str(score_day),
        "read_only": True,
        "disposition": "OPERATOR_DECISION",
        "checks": [
            {
                "id": check_id,
                "status": "failed",
                "severity": "operator",
                "summary": summary,
                "evidence": evidence,
            }
        ],
    }


def run(*, score_day: date | None = None) -> tuple[dict[str, Any], Path | None]:
    """Execute one post-close reconciliation without invoking the live pipeline."""
    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    post_close_cfg = dict(live_cfg.get("post_close_readiness", {}))
    asof = score_day or datetime.now().date()
    report_root = _report_root(live_cfg=live_cfg, paths_cfg=paths_cfg)
    if not bool(post_close_cfg.get("enabled", True)):
        return {
            "schema_version": 1,
            "score_day": str(asof),
            "read_only": True,
            "disposition": "SKIPPED_DISABLED",
            "checks": [],
        }, None

    live_root = Path(paths_cfg["results_root"]) / "live"
    scheduler_dir = live_root / "scheduler"
    state = _read_json(scheduler_dir / "state.json")
    scheduler_log = live_root / f"{asof:%Y-%m-%d}" / "logs" / f"live_scheduler_{asof:%Y-%m-%d}.log"
    log_text = scheduler_log.read_text(encoding="utf-8", errors="replace") if scheduler_log.exists() else ""
    events = read_attempt_events(scheduler_dir=scheduler_dir, score_day=asof)
    runtime_evidence = _same_day_runtime_evidence(
        scheduler_state=state,
        attempt_events=events,
        log_text=log_text,
        score_day=asof,
    )
    try:
        open_days = set(
            list_available_trading_days_from_raw(
                paths_cfg["raw_data_root"],
                kind="snapshot",
                asset="cbond",
            )
        )
    except Exception as exc:
        if not runtime_evidence["incident_hint"]:
            return {
                "schema_version": 1,
                "score_day": str(asof),
                "read_only": True,
                "disposition": "SKIPPED_NO_INCIDENT",
                "checks": [],
            }, None
        report = _operator_report(
            score_day=asof,
            check_id="trading_calendar",
            summary=f"cannot resolve raw trading-calendar context: {type(exc).__name__}",
            evidence={"error": str(exc), "scheduler_precheck": runtime_evidence},
        )
        report["attempt_journal_path"] = str(scheduler_dir / "attempts" / f"{asof:%Y-%m-%d}.jsonl")
        report["scheduler_log_path"] = str(scheduler_log)
        report_path = _write_report(root=report_root, score_day=asof, report=report)
        return report, report_path
    if asof not in open_days:
        if not runtime_evidence["incident_hint"]:
            return {
                "schema_version": 1,
                "score_day": str(asof),
                "read_only": True,
                "disposition": "SKIPPED_NON_TRADING_DAY",
                "checks": [],
            }, None
        report = _operator_report(
            score_day=asof,
            check_id="trading_calendar",
            summary="same-day failure or unresolved scheduler evidence exists but raw calendar does not classify the day as trading",
            evidence={"scheduler_precheck": runtime_evidence, "open_day_count": int(len(open_days))},
        )
        report["attempt_journal_path"] = str(scheduler_dir / "attempts" / f"{asof:%Y-%m-%d}.jsonl")
        report["scheduler_log_path"] = str(scheduler_log)
        report_path = _write_report(root=report_root, score_day=asof, report=report)
        return report, report_path
    try:
        report = inspect_post_close_readiness(
            live_cfg=live_cfg,
            paths_cfg=paths_cfg,
            score_day=asof,
            scheduler_state=state,
            attempt_events=events,
            scheduler_log_text=log_text,
        )
    except Exception as exc:
        report = _operator_report(
            score_day=asof,
            check_id="checker_execution",
            summary=f"checker execution failed: {type(exc).__name__}",
            evidence={"error": str(exc), "scheduler_precheck": runtime_evidence},
        )
    report["attempt_journal_path"] = str(scheduler_dir / "attempts" / f"{asof:%Y-%m-%d}.jsonl")
    report["scheduler_log_path"] = str(scheduler_log)
    if report["disposition"] == "SKIPPED_HEALTHY":
        return report, None
    report_path = _write_report(root=report_root, score_day=asof, report=report)
    return report, report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only CBOND_ON post-close readiness checker")
    parser.add_argument("--asof", help="score day in YYYY-MM-DD; defaults to local today")
    args = parser.parse_args(argv)
    try:
        score_day = parse_date(args.asof) if args.asof else None
        report, _ = run(score_day=score_day)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "read_only": True,
            "disposition": "OPERATOR_DECISION",
            "checks": [
                {
                    "id": "checker_execution",
                    "status": "failed",
                    "severity": "operator",
                    "summary": f"checker execution failed: {type(exc).__name__}",
                    "evidence": {"error": str(exc)},
                }
            ],
        }
        print(json.dumps(failure, ensure_ascii=False, default=str))
        return 3
    print(json.dumps({"disposition": report["disposition"], "report_path": report.get("report_path", "")}, ensure_ascii=False))
    return _exit_code(str(report["disposition"]))


if __name__ == "__main__":
    raise SystemExit(main())
