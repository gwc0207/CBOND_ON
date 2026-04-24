from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from cbond_on.core.config import load_config_file, resolve_output_path


def _default_data_hub_project_root() -> Path:
    env_root = str(os.getenv("CBOND_DATA_HUB_PROJECT_ROOT", "")).strip()
    if env_root:
        return Path(env_root).expanduser()
    if os.name == "nt":
        return Path("C:/Users/BaiYang/CBOND_DATA_HUB")
    return Path.home() / "CBOND_DATA_HUB"


def runtime_from_live(live_cfg: dict | None = None) -> dict[str, str]:
    cfg = dict((live_cfg or load_config_file("live")).get("data_hub", {}))
    project_root = resolve_output_path(
        cfg.get("project_root"),
        default_path=_default_data_hub_project_root(),
    )
    return {
        "python_exe": str(cfg.get("python_exe") or sys.executable),
        "module": str(cfg.get("module") or "cbond_data_hub"),
        "project_root": str(project_root),
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

