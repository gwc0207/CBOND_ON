from __future__ import annotations

from typing import Any

from cbond_on.app.pipelines.train_score_pipeline import execute as run_train_score_pipeline


def apply_execution_overrides(
    model_cfg: dict[str, Any],
    *,
    refit_every_n_days: int | None = None,
    train_processes: int | None = None,
    parallel_shards: int | None = None,
    parallel_shard_index: int | None = None,
) -> dict[str, Any]:
    cfg = dict(model_cfg)
    execution_cfg = dict(cfg.get("execution", {}))
    if refit_every_n_days is not None:
        execution_cfg["refit_every_n_days"] = int(refit_every_n_days)
    if train_processes is not None:
        execution_cfg["train_processes"] = int(train_processes)
    if parallel_shards is not None:
        execution_cfg["parallel_shards"] = int(parallel_shards)
    if parallel_shard_index is not None:
        execution_cfg["parallel_shard_index"] = int(parallel_shard_index)

    train_processes_eff = int(execution_cfg.get("train_processes", 1))
    if parallel_shards is None and train_processes_eff > 1:
        execution_cfg["parallel_shards"] = int(train_processes_eff)
    if parallel_shard_index is None:
        execution_cfg.pop("parallel_shard_index", None)
    cfg["execution"] = execution_cfg
    return cfg


def run(
    model_cfg: dict[str, Any],
    *,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict:
    return run_train_score_pipeline(
        model_cfg,
        model_id=model_id or model_cfg.get("model_id") or model_cfg.get("default_model_id"),
        start=start or model_cfg.get("start"),
        end=end or model_cfg.get("end"),
        label_cutoff=label_cutoff or model_cfg.get("label_cutoff"),
    )

