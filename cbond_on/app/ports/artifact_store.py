from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactStorePort(Protocol):
    def ensure_dir(self, path: Path) -> Path: ...

