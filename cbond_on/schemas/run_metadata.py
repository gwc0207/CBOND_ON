from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    workflow_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    git_sha: str | None = None
    config_hash: str | None = None
    factor_contract_hash: str | None = None
    model_ref: str | None = None
    artifact_paths: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

