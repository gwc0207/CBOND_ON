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
from cbond_on.run import sync_data

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


def _append_heartbeat_log(live_root: Path, now: datetime, status: str, today: date, target: date) -> None:
    log_path = _day_logs_dir(live_root, now.date()) / f"live_scheduler_{now:%Y-%m-%d}.log"
    line = (
        f"{now:%Y-%m-%d %H:%M:%S} [heartbeat] "
        f"status={status} today={today} target={target}\n"
    )
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line)


def _append_scheduler_log(live_root: Path, now: datetime, tag: str, message: str) -> None:
    log_path = _day_logs_dir(live_root, now.date()) / f"live_scheduler_{now:%Y-%m-%d}.log"
    line = f"{now:%Y-%m-%d %H:%M:%S} [{tag}] {message}\n"
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line)


def _run_db_backfill(
    *,
    raw_root: str,
    live_root: Path,
    now: datetime,
    lookback_days: int,
    reason: str,
) -> bool:
    lookback_days = max(1, int(lookback_days))
    start = (now.date() - timedelta(days=lookback_days))
    end = now.date()
    try:
        sync_cfg = load_config_file("raw_data")
        mode = str(sync_cfg.get("mode", "both")).lower()

        if mode in ("db", "both"):
            db_cfg = dict(sync_cfg.get("db", {}))
            if not db_cfg.get("sync_tables"):
                db_cfg["sync_tables"] = [
                    "metadata.trading_calendar",
                    "market_cbond.daily_price",
                    "market_cbond.daily_twap",
                    "market_cbond.daily_vwap",
                    "market_cbond.daily_deriv",
                    "market_cbond.daily_base",
                    "market_cbond.daily_rating",
                ]
            db_cfg["start"] = str(start)
            db_cfg["end"] = str(end)
            db_cfg["refresh"] = False
            db_cfg["overwrite"] = False
            _append_scheduler_log(
                live_root,
                now,
                "pre_sync",
                f"{reason} db_sync start={start} end={end}",
            )
            sync_data._sync_db(raw_root, db_cfg)
            _append_scheduler_log(live_root, now, "pre_sync", f"{reason} db_sync done")

        if mode in ("nfs", "both"):
            nfs_cfg = dict(sync_cfg.get("nfs", {}))
            nfs_cfg["start"] = str(start)
            nfs_cfg["end"] = str(end)
            nfs_cfg["refresh"] = False
            nfs_cfg["overwrite"] = False
            _append_scheduler_log(
                live_root,
                now,
                "pre_sync",
                f"{reason} nfs_sync start={start} end={end}",
            )
            sync_data._sync_nfs(raw_root, nfs_cfg)
            _append_scheduler_log(live_root, now, "pre_sync", f"{reason} nfs_sync done")
        elif mode == "ftp":
            ftp_cfg = dict(sync_cfg.get("ftp", {}))
            ftp_cfg["start"] = str(start)
            ftp_cfg["end"] = str(end)
            ftp_cfg["refresh"] = False
            ftp_cfg["overwrite"] = False
            _append_scheduler_log(
                live_root,
                now,
                "pre_sync",
                f"{reason} ftp_sync start={start} end={end}",
            )
            sync_data._sync_ftp(raw_root, ftp_cfg)
            _append_scheduler_log(live_root, now, "pre_sync", f"{reason} ftp_sync done")
        return True
    except Exception as exc:  # pragma: no cover
        _append_scheduler_log(
            live_root,
            now,
            "pre_sync",
            f"{reason} pre_sync failed: {type(exc).__name__}: {exc}",
        )
        return False


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
    }

    while True:
        live_cfg = load_config_file("live")
        cutoff = parse_time(str(live_cfg.get("cutoff_time", "14:30")))
        morning_sync_enabled = bool(live_cfg.get("morning_sync_enabled", True))
        pre_run_sync_enabled = bool(live_cfg.get("pre_run_sync_enabled", True))
        morning_sync_time = parse_time(str(live_cfg.get("morning_sync_time", "09:30")))
        pre_sync_lookback_days = int(live_cfg.get("pre_sync_lookback_days", 7))
        now = datetime.now()
        today = now.date()
        target = _next_trading_day(raw_root, today)
        st = _read_json(state_path)
        last_target_run = st.get("last_target_run")
        last_morning_sync_day = st.get("last_morning_sync_day")

        if (
            morning_sync_enabled
            and now.time() >= morning_sync_time
            and last_morning_sync_day != str(today)
        ):
            ok = _run_db_backfill(
                raw_root=raw_root,
                live_root=live_root,
                now=now,
                lookback_days=pre_sync_lookback_days,
                reason="morning",
            )
            st = _read_json(state_path)
            st["last_morning_sync_day"] = str(today)
            st["last_morning_sync_at"] = now.isoformat(timespec="seconds")
            st["last_morning_sync_ok"] = bool(ok)
            st["heartbeat"] = now.isoformat(timespec="seconds")
            _write_json(state_path, st)

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

        if pre_run_sync_enabled:
            ok = _run_db_backfill(
                raw_root=raw_root,
                live_root=live_root,
                now=now,
                lookback_days=pre_sync_lookback_days,
                reason="pre_run",
            )
            st = _read_json(state_path)
            st["last_pre_run_sync_target"] = str(target)
            st["last_pre_run_sync_at"] = now.isoformat(timespec="seconds")
            st["last_pre_run_sync_ok"] = bool(ok)
            st["heartbeat"] = now.isoformat(timespec="seconds")
            _write_json(state_path, st)

        log_path = _day_logs_dir(live_root, now.date()) / f"live_scheduler_{now:%Y-%m-%d}.log"
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

        with log_path.open("a", encoding="utf-8") as fp:
            flags = WIN_NO_WINDOW if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=fp,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
            last_hb_log = datetime.min
            while True:
                rc = proc.poll()
                st = _read_json(state_path)
                now_run = datetime.now()
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
                            now_run.isoformat(timespec="seconds"),
                        ),
                        "heartbeat": now_run.isoformat(timespec="seconds"),
                        "worker_pid": int(proc.pid),
                    },
                )
                if (now_run - last_hb_log).total_seconds() >= HEARTBEAT_LOG_INTERVAL_SECONDS:
                    _append_heartbeat_log(live_root, now_run, "running_live", today, target)
                    last_hb_log = now_run
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
        _append_heartbeat_log(live_root, datetime.now(), done["status"], today, target)
        time.sleep(10)


if __name__ == "__main__":
    main()
