from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_time
from cbond_on.data.io import read_trading_calendar

WIN_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


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
        "cmd",
        "log_path",
        "worker_pid",
        "run_started_at",
        "run_finished_at",
        "last_return_code",
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


def main() -> None:
    paths_cfg = load_config_file("paths")
    raw_root = str(paths_cfg["raw_data_root"])
    results_root = Path(paths_cfg["results_root"])
    sched_dir = results_root / "live" / "scheduler"
    logs_dir = sched_dir / "logs"
    state_path = sched_dir / "state.json"
    pid_path = sched_dir / "pid.json"
    logs_dir.mkdir(parents=True, exist_ok=True)

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

    while True:
        live_cfg = load_config_file("live")
        cutoff = parse_time(str(live_cfg.get("cutoff_time", "14:30")))
        now = datetime.now()
        today = now.date()
        target = _next_trading_day(raw_root, today)
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
            time.sleep(30)
            continue

        log_path = logs_dir / f"live_{target:%Y%m%d}_{datetime.now():%H%M%S}.log"
        cmd = [
            sys.executable,
            "-m",
            "cbond_on.run.live",
            "--start",
            str(target),
            "--target",
            str(target),
        ]
        env = os.environ.copy()
        env["TQDM_DISABLE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        _write_json(
            state_path,
            {
                **st,
                "status": "running_live",
                "today": str(today),
                "target": str(target),
                "cmd": cmd,
                "log_path": str(log_path),
                "run_started_at": datetime.now().isoformat(timespec="seconds"),
            },
        )

        with log_path.open("w", encoding="utf-8") as fp:
            flags = WIN_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            while True:
                rc = proc.poll()
                st = _read_json(state_path)
                _write_json(
                    state_path,
                    {
                        **st,
                        "status": "running_live",
                        "today": str(today),
                        "target": str(target),
                        "cmd": cmd,
                        "log_path": str(log_path),
                        "run_started_at": st.get(
                            "run_started_at",
                            datetime.now().isoformat(timespec="seconds"),
                        ),
                        "heartbeat": datetime.now().isoformat(timespec="seconds"),
                        "worker_pid": int(proc.pid),
                    },
                )
                if rc is not None:
                    break
                time.sleep(5)

        st = _read_json(state_path)
        done = {
            **st,
            "run_finished_at": datetime.now().isoformat(timespec="seconds"),
            "last_return_code": int(rc if rc is not None else -1),
            "last_target_attempt": str(target),
            "heartbeat": datetime.now().isoformat(timespec="seconds"),
        }
        if rc == 0:
            done["last_target_run"] = str(target)
            done["status"] = "success"
        else:
            done["status"] = "failed"
        _write_json(state_path, done)
        time.sleep(10)


if __name__ == "__main__":
    main()
