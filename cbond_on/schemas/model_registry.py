from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    version: str
    alias: str | None
    artifact_path: str
    status: str
    train_run_id: str | None = None
    feature_contract_hash: str | None = None
    config_hash: str | None = None

