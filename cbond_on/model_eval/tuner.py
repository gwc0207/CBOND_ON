from __future__ import annotations

import copy
import itertools
import json
import random
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.model_eval.evaluator import (
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
    objective_from_summary,
)
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.services.model.model_score_service import run as run_model_score


def _set_by_dot_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [p for p in str(path).strip().split(".") if p]
    if not parts:
        raise ValueError("empty parameter path")
    cur: dict[str, Any] = target
    for key in parts[:-1]:
        node = cur.get(key)
        if not isinstance(node, dict):
            node = {}
            cur[key] = node
        cur = node
    cur[parts[-1]] = value


def _normalize_space(space: dict[str, Any]) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for key, values in (space or {}).items():
        if isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [values]
        if not vals:
            raise ValueError(f"parameter space values empty for: {key}")
        out[str(key)] = vals
    if not out:
        raise ValueError("tuning.parameter_space is empty")
    return out


def _sanitize_wandb_param_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    if not base:
        base = "param"
    if base[0].isdigit():
        base = f"p_{base}"
    alias = base
    idx = 2
    while alias in used:
        alias = f"{base}_{idx}"
        idx += 1
    used.add(alias)
    return alias


def _estimate_total_combinations(space: dict[str, list[Any]]) -> int:
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


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


def _execute_single_trial(
    *,
    idx: int,
    total: int,
    trial_id: str,
    params_patch: dict[str, Any],
    score_cfg: dict[str, Any],
    model_id: str,
    model_type: str,
    base_model_cfg: dict[str, Any],
    start: date,
    end: date,
    label_cutoff: date | None,
    execution_override: dict[str, Any],
    label_root: str | Path,
    trials_root: Path,
    objective_metric: str,
    higher_is_better: bool,
    bins: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    trial_dir = trials_root / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_score_output = trial_dir / "scores"

    trial_model_cfg = copy.deepcopy(base_model_cfg)
    trial_model_cfg["start"] = str(start)
    trial_model_cfg["end"] = str(end)
    trial_model_cfg["model_name"] = f"{base_model_cfg.get('model_name', model_id)}__{trial_id}"
    trial_model_cfg["score_output"] = str(trial_score_output.as_posix())
    trial_model_cfg["score_overwrite"] = True
    incremental = dict(trial_model_cfg.get("incremental", {}))
    incremental["enabled"] = True
    incremental["skip_existing_scores"] = False
    incremental["warm_start"] = False
    incremental["save_state"] = False
    trial_model_cfg["incremental"] = incremental

    for key, value in params_patch.items():
        _set_by_dot_path(trial_model_cfg, key, value)

    trial_model_cfg_path = trial_dir / "model_config.json"
    trial_model_cfg_path.write_text(
        json.dumps(trial_model_cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trial_score_cfg = copy.deepcopy(score_cfg)
    trial_score_cfg["model_id"] = trial_id
    trial_score_cfg["default_model_id"] = trial_id
    trial_score_cfg["models"] = {
        trial_id: {
            "model_type": model_type,
            "model_config": str(trial_model_cfg_path.as_posix()),
        }
    }
    trial_score_cfg["execution"] = dict(execution_override)

    try:
        print(f"[model_eval] tuning {idx}/{total} {trial_id} start")
        run_model_score(
            model_id=trial_id,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            cfg=trial_score_cfg,
        )

        scores = load_scores_frame(trial_score_output)
        merged = merge_score_with_label(
            scores=scores,
            label_root=label_root,
            factor_time=str(trial_model_cfg.get("factor_time", "14:30")),
            label_time=str(trial_model_cfg.get("label_time", "14:42")),
            start=start,
            end=end,
        )
        eval_result = evaluate_merged_scores(merged, bins=bins)
        objective_value = objective_from_summary(
            eval_result.summary,
            metric=objective_metric,
            higher_is_better=higher_is_better,
        )

        (trial_dir / "params.json").write_text(
            json.dumps(params_patch, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (trial_dir / "evaluation_summary.json").write_text(
            json.dumps(eval_result.summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if not eval_result.daily.empty:
            eval_result.daily.to_csv(trial_dir / "evaluation_daily.csv", index=False)

        row = {
            "trial_id": trial_id,
            "status": "ok",
            "objective_metric": objective_metric,
            "objective_value": objective_value,
            **eval_result.summary,
        }
        for key, value in params_patch.items():
            row[f"param.{key}"] = value

        candidate = {
            "trial_id": trial_id,
            "objective_value": objective_value,
            "params": params_patch,
            "summary": eval_result.summary,
            "model_config_path": str(trial_model_cfg_path.as_posix()),
            "score_output": str(trial_score_output.as_posix()),
        }
        print(f"[model_eval] tuning {idx}/{total} {trial_id} done objective={objective_value:.6f}")
        return row, candidate
    except Exception as exc:
        row = {
            "trial_id": trial_id,
            "status": "failed",
            "objective_metric": objective_metric,
            "objective_value": float("-inf"),
            "error": f"{type(exc).__name__}: {exc}",
        }
        for key, value in params_patch.items():
            row[f"param.{key}"] = value
        print(f"[model_eval] tuning {idx}/{total} {trial_id} failed: {type(exc).__name__}: {exc}")
        return row, None


def _run_wandb_sweep_trials(
    *,
    score_cfg: dict[str, Any],
    model_id: str,
    model_type: str,
    base_model_cfg: dict[str, Any],
    start: date,
    end: date,
    label_cutoff: date | None,
    execution_override: dict[str, Any],
    label_root: str | Path,
    out_dir: Path,
    tuning_cfg: dict[str, Any],
    space: dict[str, list[Any]],
    objective_metric: str,
    higher_is_better: bool,
    bins: int,
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
    total_combos = _estimate_total_combinations(space)
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
        alias = _sanitize_wandb_param_name(path, used_alias)
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

    trials_root = out_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
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
        row, candidate = _execute_single_trial(
            idx=state["idx"],
            total=agent_count,
            trial_id=trial_id,
            params_patch=params_patch,
            score_cfg=score_cfg,
            model_id=model_id,
            model_type=model_type,
            base_model_cfg=base_model_cfg,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution_override=run_execution_override,
            label_root=label_root,
            trials_root=trials_root,
            objective_metric=objective_metric,
            higher_is_better=higher_is_better,
            bins=bins,
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


def _build_trials(
    *,
    space: dict[str, list[Any]],
    search_type: str,
    max_trials: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    keys = list(space.keys())
    products = [space[k] for k in keys]
    all_trials = [dict(zip(keys, combo)) for combo in itertools.product(*products)]
    if not all_trials:
        return []

    if max_trials <= 0:
        max_trials = len(all_trials)

    mode = str(search_type or "grid").strip().lower()
    if mode == "random":
        rng = random.Random(int(random_seed))
        if max_trials >= len(all_trials):
            return all_trials
        return rng.sample(all_trials, k=max_trials)

    return all_trials[:max_trials]


def run_hyperparameter_tuning(
    *,
    score_cfg: dict[str, Any],
    model_id: str,
    start: date,
    end: date,
    label_cutoff: date | None,
    execution_override: dict[str, Any],
    label_root: str | Path,
    out_dir: Path,
    tuning_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"model_id not configured for tuning: {model_id}")
    base_entry = dict(models[model_id])
    model_type = str(base_entry.get("model_type", "")).strip()
    model_config_key = str(base_entry.get("model_config", "")).strip()
    if not model_type or not model_config_key:
        raise ValueError(f"invalid model entry for tuning: {model_id}")

    base_model_cfg_path = resolve_config_path(model_config_key)
    base_model_cfg = load_json_like(base_model_cfg_path)

    search_type = str(tuning_cfg.get("search_type", "grid")).strip().lower()
    max_trials = int(tuning_cfg.get("max_trials", 20))
    random_seed = int(tuning_cfg.get("random_seed", 42))
    space = _normalize_space(dict(tuning_cfg.get("parameter_space", {})))

    objective_cfg = dict(eval_cfg.get("objective", {}))
    objective_metric = str(objective_cfg.get("metric", "rank_ic_mean"))
    higher_is_better = bool(objective_cfg.get("higher_is_better", True))
    bins = int(eval_cfg.get("bins", base_model_cfg.get("bins", 5)))

    if search_type in {"wandb", "wandb_sweep", "sweep"}:
        return _run_wandb_sweep_trials(
            score_cfg=score_cfg,
            model_id=model_id,
            model_type=model_type,
            base_model_cfg=base_model_cfg,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution_override=execution_override,
            label_root=label_root,
            out_dir=out_dir,
            tuning_cfg=tuning_cfg,
            space=space,
            objective_metric=objective_metric,
            higher_is_better=higher_is_better,
            bins=bins,
        )

    trial_params_list = _build_trials(
        space=space,
        search_type=search_type,
        max_trials=max_trials,
        random_seed=random_seed,
    )
    if not trial_params_list:
        raise RuntimeError("no tuning trials generated")

    trial_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_objective = float("-inf")

    trials_root = out_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    for idx, params_patch in enumerate(trial_params_list, start=1):
        trial_id = f"trial_{idx:03d}"
        row, candidate = _execute_single_trial(
            idx=idx,
            total=len(trial_params_list),
            trial_id=trial_id,
            params_patch=params_patch,
            score_cfg=score_cfg,
            model_id=model_id,
            model_type=model_type,
            base_model_cfg=base_model_cfg,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution_override=execution_override,
            label_root=label_root,
            trials_root=trials_root,
            objective_metric=objective_metric,
            higher_is_better=higher_is_better,
            bins=bins,
        )
        trial_rows.append(row)
        if candidate is not None and float(candidate.get("objective_value", float("-inf"))) > best_objective:
            best_objective = float(candidate["objective_value"])
            best_trial = candidate

    trial_df = pd.DataFrame(trial_rows)
    if not trial_df.empty:
        trial_df.to_csv(out_dir / "trial_summary.csv", index=False)
    if best_trial is not None:
        (out_dir / "best_trial.json").write_text(
            json.dumps(best_trial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return trial_df, best_trial
