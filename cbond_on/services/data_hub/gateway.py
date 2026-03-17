from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from cbond_on.core.config import load_config_file, parse_date


def runtime_from_live(live_cfg: dict | None = None) -> dict[str, str]:
    cfg = dict((live_cfg or load_config_file("live")).get("data_hub", {}))
    return {
        "python_exe": str(cfg.get("python_exe") or sys.executable),
        "module": str(cfg.get("module") or "cbond_data_hub"),
        "project_root": str(cfg.get("project_root") or "C:/Users/BaiYang/CBOND_DATA_HUB"),
        "pg_config_path": str(cfg.get("pg_config_path") or "").strip(),
    }


def _parse_json_output(text: str) -> dict[str, Any]:
    payload = str(text or "").strip()
    if not payload:
        return {}
    try:
        obj = json.loads(payload)
        return dict(obj) if isinstance(obj, dict) else {"value": obj}
    except Exception:
        pass
    start = payload.find("{")
    end = payload.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(payload[start : end + 1])
        return dict(obj) if isinstance(obj, dict) else {"value": obj}
    raise ValueError(f"failed to parse Data Hub JSON output: {payload[:400]}")


def run_cli(
    args: list[str],
    *,
    runtime: dict[str, str] | None = None,
    expect_json: bool = True,
) -> dict[str, Any]:
    rt = dict(runtime or runtime_from_live())
    if not Path(rt["project_root"]).exists():
        raise FileNotFoundError(f"data hub project_root not found: {rt['project_root']}")
    cmd = [rt["python_exe"], "-m", rt["module"], *args]
    proc = subprocess.run(
        cmd,
        cwd=rt["project_root"],
        capture_output=True,
        text=True,
    )
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"data hub command failed rc={proc.returncode}: {' '.join(cmd)}")
    if not expect_json:
        return {}
    return _parse_json_output(stdout)


def sync_history(
    *,
    start: str | date,
    end: str | date,
    refresh: bool = False,
    overwrite: bool = False,
    mode: str | None = None,
    raw_cfg: dict | None = None,
    paths_cfg: dict | None = None,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = dict(raw_cfg or load_config_file("raw_data"))
    paths = dict(paths_cfg or load_config_file("paths"))
    rt = dict(runtime or runtime_from_live())
    args = [
        "raw",
        "sync-history",
        "--raw-root",
        str(paths["raw_data_root"]),
        "--start",
        str(parse_date(start)),
        "--end",
        str(parse_date(end)),
        "--mode",
        str(mode or raw.get("mode", "both")).strip().lower(),
    ]
    if refresh:
        args.append("--refresh")
    if overwrite:
        args.append("--overwrite")
    db_tables = list(dict(raw.get("db", {})).get("sync_tables", []) or [])
    if db_tables:
        args.extend(["--sync-tables", ",".join(str(x) for x in db_tables)])
    nfs_cfg = dict(raw.get("nfs", {}))
    nfs_root = str(nfs_cfg.get("nfs_root", "")).strip()
    if nfs_root:
        args.extend(["--nfs-root", nfs_root])
    nfs_base_dir = str(nfs_cfg.get("base_dir", "")).strip()
    if nfs_base_dir:
        args.extend(["--nfs-base-dir", nfs_base_dir])
    if rt.get("pg_config_path"):
        args.extend(["--pg-config-path", str(rt["pg_config_path"])])
    return run_cli(args, runtime=rt, expect_json=True)


def build_clean(
    *,
    start: str | date,
    end: str | date,
    refresh: bool = False,
    overwrite: bool = False,
    kline_enabled: bool | None = None,
    cleaned_cfg: dict | None = None,
    data_cfg: dict | None = None,
    paths_cfg: dict | None = None,
    runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    cleaned = dict(cleaned_cfg or load_config_file("cleaned_data"))
    data = dict(data_cfg or load_config_file("live").get("data", {}))
    paths = dict(paths_cfg or load_config_file("paths"))
    snapshot = dict(cleaned.get("snapshot", {}))

    enabled = bool(data.get("kline_enabled", cleaned.get("kline_enabled", True)))
    if kline_enabled is not None:
        enabled = bool(kline_enabled)

    args = [
        "clean",
        "build",
        "--raw-root",
        str(paths["raw_data_root"]),
        "--clean-root",
        str(paths.get("cleaned_data_root") or paths.get("clean_data_root")),
        "--start",
        str(parse_date(start)),
        "--end",
        str(parse_date(end)),
        "--allowed-phases",
        ",".join(str(x) for x in snapshot.get("allowed_phases", ["T", "T0"])),
        "--price-field",
        str(snapshot.get("price_field", "last")),
        "--kline-enabled" if enabled else "--no-kline-enabled",
        "--filter-trading-phase"
        if bool(snapshot.get("filter_trading_phase", True))
        else "--no-filter-trading-phase",
        "--drop-no-trade"
        if bool(snapshot.get("drop_no_trade", True))
        else "--no-drop-no-trade",
        "--use-prev-snapshot"
        if bool(snapshot.get("use_prev_snapshot", True))
        else "--no-use-prev-snapshot",
    ]
    if refresh:
        args.append("--refresh")
    if overwrite:
        args.append("--overwrite")
    return run_cli(args, runtime=runtime, expect_json=True)
