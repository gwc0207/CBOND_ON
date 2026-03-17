from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_time
from cbond_on.data.io import read_trading_calendar
from cbond_on.services.live.live_service import run_once

WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HEARTBEAT_LOG_INTERVAL_SECONDS = 300


def _day_live_dir(live_root: Path, day: date) -> Path:
    day_dir = live_root / f"{day:%Y-%m-%d}"
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def _day_logs_dir(live_root: Path, day: date) -> Path:
    logs_dir = _day_live_dir(live_root, day) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _next_trading_day(raw_root: str, run_day: date) -> date:
    cal = read_trading_calendar(raw_root)
    if cal.empty or "calendar_date" not in cal.columns:
        return run_day + timedelta(days=1)
    work = cal.copy()
    work["calendar_date"] = pd.to_datetime(work["calendar_date"], errors="coerce").dt.date
    if "is_open" in work.columns:
        work = work[work["is_open"].astype(bool)]
    days = sorted(d for d in work["calendar_date"].dropna().unique().tolist() if d > run_day)
    if not days:
        return run_day + timedelta(days=1)
    return days[0]


def _resolve_target_day(raw_root: str, today: date, schedule_cfg: dict) -> date:
    policy = str(schedule_cfg.get("target_policy", "next_trading_day_after_cutoff")).strip().lower()
    if policy in ("today", "current", "run_day"):
        return today
    if policy in ("next_trading_day_after_cutoff", "next_trading_day", "next"):
        return _next_trading_day(raw_root, today)
    raise ValueError(
        "unsupported schedule.target_policy="
        f"{policy}, expected one of ['today', 'next_trading_day_after_cutoff']"
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _drop_run_fields(state: dict) -> dict:
    cleaned = dict(state)
    for key in [
        "log_path",
        "run_started_at",
        "run_finished_at",
        "last_return_code",
        "out_dir",
    ]:
        cleaned.pop(key, None)
    return cleaned


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            creationflags=WIN_NO_WINDOW,
        )
        txt = (out.stdout or "").lower()
        return str(pid) in txt and "no tasks are running" not in txt
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _append_heartbeat_log(live_root: Path, now: datetime, status: str, today: date, target: date) -> None:
    log_path = _day_logs_dir(live_root, now.date()) / f"live_scheduler_{now:%Y-%m-%d}.log"
    line = (
        f"{now:%Y-%m-%d %H:%M:%S} [heartbeat] "
        f"status={status} today={today} target={target}\n"
    )
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line)


def main() -> None:
    paths_cfg = load_config_file("paths")
    raw_root = str(paths_cfg["raw_data_root"])
    results_root = Path(paths_cfg["results_root"])
    live_root = results_root / "live"
    sched_dir = live_root / "scheduler"
    state_path = sched_dir / "state.json"
    pid_path = sched_dir / "pid.json"
    sched_dir.mkdir(parents=True, exist_ok=True)

    old_pid = int(_read_json(pid_path).get("pid", 0) or 0)
    if old_pid and old_pid != os.getpid() and _is_pid_alive(old_pid):
        _write_json(
            state_path,
            {
                "status": "already_running",
                "pid": old_pid,
                "now": datetime.now().isoformat(timespec="seconds"),
                "heartbeat": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return

    _write_json(
        pid_path,
        {
            "pid": os.getpid(),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    hb_last_emit: dict[str, datetime] = {
        "waiting_cutoff": datetime.min,
        "idle_after_run": datetime.min,
        "running_live": datetime.min,
    }

    while True:
        live_cfg = load_config_file("live")
        schedule_cfg = dict(live_cfg.get("schedule", {}))
        cutoff = parse_time(str(schedule_cfg.get("cutoff_time", "14:30")))

        now = datetime.now()
        today = now.date()
        target = _resolve_target_day(raw_root, today, schedule_cfg)
        st = _read_json(state_path)
        last_target_run = st.get("last_target_run")

        if now.time() < cutoff:
            base = _drop_run_fields(st)
            _write_json(
                state_path,
                {
                    **base,
                    "status": "waiting_cutoff",
                    "today": str(today),
                    "target": str(target),
                    "cutoff_time": str(cutoff),
                    "heartbeat": now.isoformat(timespec="seconds"),
                },
            )
            if (
                now - hb_last_emit["waiting_cutoff"]
            ).total_seconds() >= HEARTBEAT_LOG_INTERVAL_SECONDS:
                _append_heartbeat_log(live_root, now, "waiting_cutoff", today, target)
                hb_last_emit["waiting_cutoff"] = now
            time.sleep(30)
            continue

        if last_target_run == str(target):
            base = _drop_run_fields(st)
            _write_json(
                state_path,
                {
                    **base,
                    "status": "idle_after_run",
                    "today": str(today),
                    "target": str(target),
                    "heartbeat": now.isoformat(timespec="seconds"),
                },
            )
            if (
                now - hb_last_emit["idle_after_run"]
            ).total_seconds() >= HEARTBEAT_LOG_INTERVAL_SECONDS:
                _append_heartbeat_log(live_root, now, "idle_after_run", today, target)
                hb_last_emit["idle_after_run"] = now
            time.sleep(30)
            continue

        log_path = _day_logs_dir(live_root, now.date()) / f"live_scheduler_{now:%Y-%m-%d}.log"
        _write_json(
            state_path,
            {
                **st,
                "status": "running_live",
                "today": str(today),
                "target": str(target),
                "log_path": str(log_path),
                "run_started_at": datetime.now().isoformat(timespec="seconds"),
                "heartbeat": datetime.now().isoformat(timespec="seconds"),
            },
        )

        rc = 0
        out_dir = ""
        with log_path.open("a", encoding="utf-8") as fp:
            with redirect_stdout(fp), redirect_stderr(fp):
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [run] start target={target}")
                try:
                    out_path = run_once(start=target, target=target, mode="scheduler")
                    out_dir = str(out_path)
                    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [run] success out={out_dir}")
                except Exception as exc:
                    rc = 1
                    print(
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} [run] failed "
                        f"{type(exc).__name__}: {exc}"
                    )
                    traceback.print_exc()

        now_done = datetime.now()
        done = {
            **_read_json(state_path),
            "run_finished_at": now_done.isoformat(timespec="seconds"),
            "last_return_code": int(rc),
            "last_target_attempt": str(target),
            "heartbeat": now_done.isoformat(timespec="seconds"),
            "out_dir": out_dir,
        }
        if rc == 0:
            done["last_target_run"] = str(target)
            done["status"] = "success"
        else:
            done["status"] = "failed"
        _write_json(state_path, done)
        _append_heartbeat_log(live_root, now_done, done["status"], today, target)
        hb_last_emit["running_live"] = now_done
        time.sleep(10)


if __name__ == "__main__":
    main()
