from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, parse_time
from cbond_on.data.io import read_trading_calendar


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Schedule live run for next trading day at cutoff time")
    parser.add_argument("--run-date", help="scheduler run date (YYYY-MM-DD), default=today")
    parser.add_argument("--no-wait", action="store_true", help="run immediately without waiting cutoff")
    args = parser.parse_args(argv)

    paths_cfg = load_config_file("paths")
    live_cfg = load_config_file("live")
    raw_root = str(paths_cfg["raw_data_root"])

    run_day = parse_date(args.run_date) if args.run_date else date.today()
    cutoff_time = parse_time(str(live_cfg.get("cutoff_time", "14:30")))
    target = _next_trading_day(raw_root, run_day)

    now = datetime.now()
    trigger_dt = datetime.combine(run_day, cutoff_time)
    if not args.no_wait and now < trigger_dt:
        wait_sec = int((trigger_dt - now).total_seconds())
        print(
            f"[scheduler] run_day={run_day} cutoff={cutoff_time} target={target} "
            f"waiting {wait_sec}s"
        )
        time.sleep(wait_sec)
    else:
        print(
            f"[scheduler] run_day={run_day} cutoff={cutoff_time} target={target} "
            f"(run immediately)"
        )

    cmd = [
        sys.executable,
        "-m",
        "cbond_on.run.live",
        "--start",
        str(target),
        "--target",
        str(target),
    ]
    print(f"[scheduler] launch: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

