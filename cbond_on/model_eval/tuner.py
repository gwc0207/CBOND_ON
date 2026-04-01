from __future__ import annotations

import copy
import itertools
import json
import random
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

    search_type = str(tuning_cfg.get("search_type", "grid"))
    max_trials = int(tuning_cfg.get("max_trials", 20))
    random_seed = int(tuning_cfg.get("random_seed", 42))
    space = _normalize_space(dict(tuning_cfg.get("parameter_space", {})))
    trial_params_list = _build_trials(
        space=space,
        search_type=search_type,
        max_trials=max_trials,
        random_seed=random_seed,
    )
    if not trial_params_list:
        raise RuntimeError("no tuning trials generated")

    objective_cfg = dict(eval_cfg.get("objective", {}))
    objective_metric = str(objective_cfg.get("metric", "rank_ic_mean"))
    higher_is_better = bool(objective_cfg.get("higher_is_better", True))
    bins = int(eval_cfg.get("bins", base_model_cfg.get("bins", 5)))

    trial_rows: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_objective = float("-inf")

    trials_root = out_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    for idx, params_patch in enumerate(trial_params_list, start=1):
        trial_id = f"trial_{idx:03d}"
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
            print(f"[model_eval] tuning {idx}/{len(trial_params_list)} {trial_id} start")
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
            trial_rows.append(row)

            if objective_value > best_objective:
                best_objective = objective_value
                best_trial = {
                    "trial_id": trial_id,
                    "objective_value": objective_value,
                    "params": params_patch,
                    "summary": eval_result.summary,
                    "model_config_path": str(trial_model_cfg_path.as_posix()),
                    "score_output": str(trial_score_output.as_posix()),
                }
            print(f"[model_eval] tuning {idx}/{len(trial_params_list)} {trial_id} done objective={objective_value:.6f}")
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
            trial_rows.append(row)
            print(f"[model_eval] tuning {idx}/{len(trial_params_list)} {trial_id} failed: {type(exc).__name__}: {exc}")

    trial_df = pd.DataFrame(trial_rows)
    if not trial_df.empty:
        trial_df.to_csv(out_dir / "trial_summary.csv", index=False)
    if best_trial is not None:
        (out_dir / "best_trial.json").write_text(
            json.dumps(best_trial, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return trial_df, best_trial
