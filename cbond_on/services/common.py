from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cbond_on.core.config import resolve_config_file_path


def resolve_config_path(name_or_path: str) -> Path:
    return resolve_config_file_path(name_or_path)


def load_json_like(path: Path) -> dict[str, Any]:
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
        return json.load(handle) or {}

