from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_config_file(name: str) -> dict[str, Any]:
    base = name if name.endswith("_config") else f"{name}_config"
    json5_path = CONFIG_DIR / f"{base}.json5"
    yaml_path = CONFIG_DIR / f"{base}.yaml"
    json_path = CONFIG_DIR / f"{base}.json"
    if json5_path.exists():
        import json5

        with json5_path.open("r", encoding="utf-8") as handle:
            return json5.load(handle) or {}
    if yaml_path.exists():
        import yaml

        with yaml_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with json_path.open("r", encoding="utf-8") as handle:
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
