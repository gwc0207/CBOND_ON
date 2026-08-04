from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ATTEMPT_JOURNAL_SCHEMA_VERSION = 1


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable, non-secret fingerprint for an effective live config."""
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_attempt_id(*, now: datetime, pid: int) -> str:
    return f"{now:%Y%m%dT%H%M%S}_{int(pid)}_{uuid4().hex[:8]}"


def journal_path(scheduler_dir: str | Path, score_day: date) -> Path:
    return Path(scheduler_dir) / "attempts" / f"{score_day:%Y-%m-%d}.jsonl"


def append_attempt_event(
    *,
    scheduler_dir: str | Path,
    score_day: date,
    event: dict[str, Any],
) -> Path:
    """Append one durable, immutable scheduler-attempt event.

    The scheduler state file remains a current-status snapshot. This journal is
    intentionally separate so a later successful repair cannot erase evidence
    of an earlier failed attempt.
    """
    path = journal_path(scheduler_dir, score_day)
    record = {
        "schema_version": ATTEMPT_JOURNAL_SCHEMA_VERSION,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        **dict(event),
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def read_attempt_events(*, scheduler_dir: str | Path, score_day: date) -> list[dict[str, Any]]:
    path = journal_path(scheduler_dir, score_day)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(dict(value))
    return events
