from __future__ import annotations

from datetime import date

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.services.model.adapters import build_adapter


def run(
    *,
    model_id: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
    label_cutoff: str | date | None = None,
    cfg: dict | None = None,
) -> dict:
    score_cfg = dict(cfg or load_config_file("model_score"))
    model_id = str(model_id or score_cfg.get("model_id") or score_cfg.get("default_model_id", "")).strip()
    if not model_id:
        raise ValueError("model_score config missing model_id/default_model_id")

    models = score_cfg.get("models", {})
    if model_id not in models:
        raise KeyError(f"model_id not configured: {model_id}")
    model_entry = dict(models[model_id])
    model_type = str(model_entry["model_type"])
    model_config_key = str(model_entry.get("model_config", "")).strip()
    if not model_config_key:
        raise ValueError(f"model entry missing model_config: {model_id}")

    model_config_path = resolve_config_path(model_config_key)
    model_cfg = load_json_like(model_config_path)

    start_day = parse_date(start or score_cfg.get("start") or model_cfg.get("start"))
    end_day = parse_date(end or score_cfg.get("end") or model_cfg.get("end"))
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    execution_cfg = score_cfg.get("execution")
    if execution_cfg is None:
        execution_cfg = {}
    if not isinstance(execution_cfg, dict):
        raise ValueError("model_score.execution must be an object")
    execution_cfg = dict(execution_cfg)

    adapter = build_adapter(model_type, model_config_path=model_config_path)
    artifact = adapter.fit(
        start=str(start_day),
        end=str(end_day),
        label_cutoff=str(cutoff_day) if cutoff_day else None,
        execution=execution_cfg,
    )
    adapter.predict(
        start=str(start_day),
        end=str(end_day),
        artifact=artifact,
        label_cutoff=str(cutoff_day) if cutoff_day else None,
        execution=execution_cfg,
    )
    return {
        "model_id": model_id,
        "model_type": model_type,
        "model_config_path": str(model_config_path),
        "score_output": model_cfg.get("score_output"),
        "start": start_day,
        "end": end_day,
    }

