from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ArtifactStore(Protocol):
    def ensure_dir(self, path: Path) -> Path: ...

    def write_json(self, path: Path, payload: dict[str, Any]) -> Path: ...

