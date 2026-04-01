from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_CONFIG_EXTS: tuple[str, ...] = (".json5", ".yaml", ".yml", ".json")
_PATHS_PROFILE_PRINTED = False
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _normalize_key(name: str | Path) -> str:
    text = str(name).strip().replace("\\", "/")
    if not text:
        raise ValueError("config name must not be empty")
    return text


def _with_config_suffix(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in _CONFIG_EXTS:
        return name
    return name if name.endswith("_config") else f"{name}_config"


def resolve_config_file_path(name: str | Path) -> Path:
    raw = _normalize_key(name)
    direct = Path(raw)
    if direct.exists():
        return direct

    if direct.suffix.lower() in _CONFIG_EXTS:
        candidate = CONFIG_DIR / raw
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"config not found: {name}")

    base = _with_config_suffix(raw)
    for ext in _CONFIG_EXTS:
        candidate = CONFIG_DIR / f"{base}{ext}"
        if candidate.exists():
            return candidate

    if "/" in base:
        for ext in _CONFIG_EXTS:
            suffix = f"{base}{ext}"
            matches = [
                p
                for p in CONFIG_DIR.rglob(f"{Path(base).name}{ext}")
                if p.relative_to(CONFIG_DIR).as_posix().endswith(suffix)
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                rels = ", ".join(sorted(p.relative_to(CONFIG_DIR).as_posix() for p in matches))
                raise FileNotFoundError(f"ambiguous config key '{name}', matches: {rels}")

    fallback: list[Path] = []
    for ext in _CONFIG_EXTS:
        matches = sorted(CONFIG_DIR.rglob(f"{Path(base).name}{ext}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            fallback.extend(matches)
    if fallback:
        rels = ", ".join(sorted(p.relative_to(CONFIG_DIR).as_posix() for p in fallback))
        raise FileNotFoundError(f"ambiguous config name '{name}', matches: {rels}")
    raise FileNotFoundError(f"config not found: {name}")


def _env_text(name: str) -> str:
    return str(os.getenv(name, "")).strip()


def _to_text_path(path: Path | str) -> str:
    return Path(path).expanduser().as_posix()


def _path_exists(value: str | Path | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().exists()
    except Exception:
        return False


def _extract_results_relative(raw_path: str) -> str | None:
    norm = raw_path.replace("\\", "/")
    low = norm.lower()
    token = "/results/"
    idx = low.find(token)
    if idx < 0:
        return None
    rel = norm[idx + len(token):].strip("/")
    return rel or None


def resolve_output_path(
    value: str | Path | None,
    *,
    default_path: str | Path,
    results_root: str | Path | None = None,
) -> Path:
    default = Path(default_path).expanduser()
    if value is None:
        return default
    raw = str(value).strip()
    if not raw:
        return default

    is_windows_style = bool(_WINDOWS_ABS_RE.match(raw))
    is_posix_style = raw.startswith("/")
    running_on_windows = sys.platform.startswith("win")

    # Compatible absolute path for current platform.
    if (running_on_windows and is_windows_style) or (not running_on_windows and is_posix_style):
        return Path(raw).expanduser()

    # Incompatible absolute path: map by /results/... suffix when possible.
    if (running_on_windows and is_posix_style) or (not running_on_windows and is_windows_style):
        rel = _extract_results_relative(raw)
        if rel:
            base = Path(results_root).expanduser() if results_root is not None else default.parent
            mapped = base / Path(rel)
            print(f"path remap: {raw} -> {mapped}")
            return mapped
        print(f"path remap fallback: {raw} -> {default}")
        return default

    # Relative path.
    return Path(raw).expanduser()


def _detect_data_hub_mount() -> Path | None:
    env_data_root = _env_text("CBOND_ON_DATA_ROOT")
    if env_data_root:
        root = Path(env_data_root).expanduser()
        if root.exists() and (root / "raw_data").exists() and (root / "clean_data").exists():
            return root
    for candidate in ("/mnt/cbond_data_hub_ro", "/mnt/cbond_data_hub"):
        root = Path(candidate)
        if root.exists() and (root / "raw_data").exists() and (root / "clean_data").exists():
            return root
    if sys.platform.startswith("win"):
        win_root = Path("D:/cbond_data_hub")
        if win_root.exists() and (win_root / "raw_data").exists() and (win_root / "clean_data").exists():
            return win_root
    return None


def _infer_runtime_root(cfg: dict[str, Any]) -> Path:
    env_runtime_root = _env_text("CBOND_ON_RUNTIME_ROOT")
    if env_runtime_root:
        return Path(env_runtime_root).expanduser()

    current_results = str(cfg.get("results_root", "")).strip()
    profile = _env_text("CBOND_ON_PATHS_PROFILE").lower()
    explicit_paths_cfg = bool(_env_text("CBOND_ON_PATHS_CONFIG"))
    if explicit_paths_cfg and current_results:
        return Path(current_results).expanduser().parent
    if profile in {"server", "linux"} and current_results:
        return Path(current_results).expanduser().parent
    if _path_exists(current_results):
        return Path(current_results).expanduser().parent

    if sys.platform.startswith("win"):
        win_runtime = Path("D:/cbond_on")
        if win_runtime.exists():
            return win_runtime
        return Path.home() / "cbond_on_runtime"
    return Path.home() / "cbond_on_runtime"


def _resolve_paths_profile_path(default_path: Path) -> Path:
    custom = _env_text("CBOND_ON_PATHS_CONFIG")
    if custom:
        custom_path = Path(custom).expanduser()
        if custom_path.exists():
            return custom_path
        return resolve_config_file_path(custom)

    profile = _env_text("CBOND_ON_PATHS_PROFILE").lower()
    if profile in {"server", "linux"}:
        return resolve_config_file_path("data/paths_server")
    if profile in {"local", "windows", "default"}:
        return default_path
    return default_path


def _apply_runtime_paths_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)

    env_raw = _env_text("CBOND_ON_RAW_ROOT")
    env_clean = _env_text("CBOND_ON_CLEAN_ROOT")
    profile = _env_text("CBOND_ON_PATHS_PROFILE").lower()
    explicit_paths_cfg = bool(_env_text("CBOND_ON_PATHS_CONFIG"))
    data_hub_root = _detect_data_hub_mount()

    raw_current = str(out.get("raw_data_root", "")).strip()
    clean_current = str(out.get("clean_data_root", "")).strip()

    strict_profile = profile in {"server", "linux", "local", "windows", "default"} or explicit_paths_cfg

    if env_raw:
        raw_root = Path(env_raw).expanduser()
    elif strict_profile and raw_current:
        raw_root = Path(raw_current).expanduser()
    elif _path_exists(raw_current):
        raw_root = Path(raw_current).expanduser()
    elif data_hub_root is not None:
        raw_root = data_hub_root / "raw_data"
    else:
        raw_root = Path(raw_current).expanduser() if raw_current else Path("raw_data")

    if env_clean:
        clean_root = Path(env_clean).expanduser()
    elif strict_profile and clean_current:
        clean_root = Path(clean_current).expanduser()
    elif _path_exists(clean_current):
        clean_root = Path(clean_current).expanduser()
    elif data_hub_root is not None:
        clean_root = data_hub_root / "clean_data"
    else:
        clean_root = Path(clean_current).expanduser() if clean_current else Path("clean_data")

    runtime_root = _infer_runtime_root(out)
    results_root = runtime_root / "results"

    out["raw_data_root"] = _to_text_path(raw_root)
    out["clean_data_root"] = _to_text_path(clean_root)
    out["cleaned_data_root"] = _to_text_path(clean_root)
    out["panel_data_root"] = _to_text_path(runtime_root / "panel_data")
    out["label_data_root"] = _to_text_path(runtime_root / "label_data")
    out["factor_data_root"] = _to_text_path(runtime_root / "factor_data")
    out["ads_root"] = _to_text_path(runtime_root / "ads")
    out["results_root"] = _to_text_path(results_root)
    out["model_root"] = _to_text_path(results_root / "models")
    out["score_root"] = _to_text_path(results_root / "scores")
    out["logs_root"] = _to_text_path(runtime_root / "logs")

    global _PATHS_PROFILE_PRINTED
    if not _PATHS_PROFILE_PRINTED:
        _PATHS_PROFILE_PRINTED = True
        print(
            "paths profile:",
            f"runtime={out['results_root']}",
            f"raw={out['raw_data_root']}",
            f"clean={out['clean_data_root']}",
        )
    return out


def load_config_file(name: str) -> dict[str, Any]:
    path = resolve_config_file_path(name)
    if path.stem == "paths_config":
        path = _resolve_paths_profile_path(path)
    suffix = path.suffix.lower()
    data: dict[str, Any]
    if suffix == ".json5":
        import json5

        with path.open("r", encoding="utf-8") as handle:
            data = json5.load(handle) or {}
    elif suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

    if path.parent.name == "data" and path.stem.startswith("paths"):
        return _apply_runtime_paths_profile(data)
    return data


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def parse_time(value: str | time) -> time:
    if isinstance(value, time):
        return value
    parts = str(value).split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid time value: {value}")
    return time(int(parts[0]), int(parts[1]))
