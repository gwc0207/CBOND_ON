from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from cbond_on.infra.model.tuning.space import (
    estimate_total_combinations,
    sanitize_wandb_param_name,
)


def _to_loggable_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            v = float(value)
        except Exception:
            return None
        if pd.isna(v):
            return None
        return int(value) if isinstance(value, int) else v
    try:
        v = float(value)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v


def run_wandb_sweep_trials(
    *,
    score_cfg: dict[str, Any],
    model_id: str,
    start: date,
    end: date,
    execution_override: dict[str, Any],
    out_dir: Path,
    tuning_cfg: dict[str, Any],
    space: dict[str, list[Any]],
    objective_metric: str,
    execute_trial_fn: Callable[[int, int, str, dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any] | None]],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "search_type=wandb_sweep requires wandb. Install with `pip install wandb`."
        ) from exc

    wandb_cfg = dict(tuning_cfg.get("wandb", {}))
    method = str(wandb_cfg.get("method", "random")).strip().lower() or "random"
    max_trials = int(tuning_cfg.get("max_trials", 20))
    total_combos = estimate_total_combinations(space)
    default_count = total_combos if method == "grid" else max_trials
    if default_count <= 0:
        default_count = total_combos
    agent_count = int(wandb_cfg.get("count", default_count))
    if agent_count <= 0:
        agent_count = default_count

    project = str(wandb_cfg.get("project", "cbond_on_tuning")).strip() or "cbond_on_tuning"
    entity_raw = str(wandb_cfg.get("entity", "")).strip()
    entity = entity_raw or None
    sweep_name = str(wandb_cfg.get("name", "")).strip()
    if not sweep_name:
        sweep_name = f"{model_id}_{start:%Y%m%d}_{end:%Y%m%d}"
    group = str(wandb_cfg.get("group", "")).strip() or f"{model_id}_sweep"
    job_type = str(wandb_cfg.get("job_type", "tuning")).strip() or "tuning"
    tags_cfg = wandb_cfg.get("tags", [])
    tags = [str(t).strip() for t in tags_cfg] if isinstance(tags_cfg, (list, tuple)) else []
    tags = [t for t in tags if t]
    mode = str(wandb_cfg.get("mode", "online")).strip().lower() or "online"
    disable_inner = bool(wandb_cfg.get("disable_inner_model_wandb", True))

    used_alias: set[str] = set()
    alias_to_path: dict[str, str] = {}
    sweep_params: dict[str, dict[str, Any]] = {}
    for path, values in space.items():
        alias = sanitize_wandb_param_name(path, used_alias)
        alias_to_path[alias] = path
        sweep_params[alias] = {"values": values}

    sweep_config: dict[str, Any] = {
        "name": sweep_name,
        "method": method,
        "metric": {
            "name": "objective_value",
            "goal": "maximize",
        },
        "parameters": sweep_params,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    print(
        "[model_eval] wandb sweep created:",
        f"sweep_id={sweep_id}",
        f"project={project}",
        f"count={agent_count}",
        f"method={method}",
    )

    trial_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_objective = float("-inf")
    state = {"idx": 0}

    run_execution_override = dict(execution_override)
    if disable_inner:
        inner_wandb = dict(run_execution_override.get("wandb", {}))
        inner_wandb["enabled"] = False
        run_execution_override["wandb"] = inner_wandb

    def _run_trial() -> None:
        nonlocal best_objective, best_trial
        run_kwargs: dict[str, Any] = {
            "group": group,
            "job_type": job_type,
            "reinit": True,
        }
        if tags:
            run_kwargs["tags"] = tags
        if mode in {"online", "offline", "disabled"}:
            run_kwargs["mode"] = mode
        run = wandb.init(**run_kwargs)
        if run is None:
            raise RuntimeError("wandb.init returned None in sweep agent")

        config_obj = getattr(run, "config", {})
        if hasattr(config_obj, "as_dict"):
            raw_cfg = dict(config_obj.as_dict())
        else:
            raw_cfg = dict(config_obj)

        params_patch: dict[str, Any] = {}
        for alias, path in alias_to_path.items():
            if alias in raw_cfg:
                params_patch[path] = raw_cfg[alias]

        state["idx"] += 1
        trial_id = f"trial_{state['idx']:03d}_{run.id}"
        row, candidate = execute_trial_fn(
            state["idx"],
            agent_count,
            trial_id,
            params_patch,
            run_execution_override,
        )
        trial_rows.append(row)

        log_payload = {
            "objective_value": float(row.get("objective_value", float("-inf"))),
            "status_ok": 1 if str(row.get("status")) == "ok" else 0,
        }
        eval_payload: dict[str, Any] = {}
        for key, value in row.items():
            if key.startswith("param."):
                log_payload[key.replace("param.", "param/")] = value
                continue
            if key in {"trial_id", "status", "objective_metric", "objective_value", "error"}:
                continue
            number = _to_loggable_number(value)
            if number is not None:
                eval_payload[f"eval/{key}"] = number
        log_payload.update(eval_payload)
        run.log(log_payload)

        run.summary["trial_id"] = trial_id
        run.summary["status"] = row.get("status")
        run.summary["status_ok"] = 1 if str(row.get("status")) == "ok" else 0
        run.summary["objective_metric"] = objective_metric
        run.summary["objective_value"] = row.get("objective_value")
        for key, value in eval_payload.items():
            run.summary[key] = value
        if row.get("error") is not None:
            run.summary["error"] = str(row.get("error"))

        if candidate is not None and float(candidate.get("objective_value", float("-inf"))) > best_objective:
            best_objective = float(candidate["objective_value"])
            best_trial = candidate
            run.summary["is_best_so_far"] = 1
        run.finish()

    wandb.agent(sweep_id, function=_run_trial, count=agent_count, project=project, entity=entity)

    trial_df = pd.DataFrame(trial_rows)
    if not trial_df.empty:
        trial_df.to_csv(out_dir / "trial_summary.csv", index=False)
    sweep_meta = {
        "sweep_id": sweep_id,
        "project": project,
        "entity": entity,
        "method": method,
        "count": int(agent_count),
        "objective_metric": objective_metric,
        "disable_inner_model_wandb": bool(disable_inner),
    }
    (out_dir / "wandb_sweep.json").write_text(
        json.dumps(sweep_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if best_trial is not None:
        best_trial_with_sweep = dict(best_trial)
        best_trial_with_sweep["wandb_sweep_id"] = sweep_id
        (out_dir / "best_trial.json").write_text(
            json.dumps(best_trial_with_sweep, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return trial_df, best_trial
