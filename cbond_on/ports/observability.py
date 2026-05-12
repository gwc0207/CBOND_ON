from __future__ import annotations

from typing import Any, Protocol

from cbond_on.schemas.run_metadata import RunMetadata


class RunRecorder(Protocol):
    def start(self, metadata: RunMetadata) -> None: ...

    def log_step(self, step: str, payload: dict[str, Any] | None = None) -> None: ...

    def finish(self, metadata: RunMetadata) -> None: ...

    def fail(self, metadata: RunMetadata, error: BaseException) -> None: ...

