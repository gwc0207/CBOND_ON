from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
_CONFIG_EXTS: tuple[str, ...] = (".json5", ".yaml", ".yml", ".json")


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


def load_config_file(name: str) -> dict[str, Any]:
    path = resolve_config_file_path(name)
    suffix = path.suffix.lower()
    if suffix == ".json5":
        import json5

        with path.open("r", encoding="utf-8") as handle:
            return json5.load(handle) or {}
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
