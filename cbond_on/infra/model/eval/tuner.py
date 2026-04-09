from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.infra.model.tuning.space import build_trials, normalize_space
from cbond_on.infra.model.tuning.trial_executor import execute_single_trial
from cbond_on.infra.model.tuning.wandb_sweep import run_wandb_sweep_trials


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
    space = normalize_space(dict(tuning_cfg.get("parameter_space", {})))

    objective_cfg = dict(eval_cfg.get("objective", {}))
    objective_metric = str(objective_cfg.get("metric", "rank_ic_mean"))
    higher_is_better = bool(objective_cfg.get("higher_is_better", True))
    bins = int(eval_cfg.get("bins", base_model_cfg.get("bins", 5)))

    trials_root = out_dir / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)

    def _execute_trial(
        idx: int,
        total: int,
        trial_id: str,
        params_patch: dict[str, Any],
        exec_cfg: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return execute_single_trial(
            idx=idx,
            total=total,
            trial_id=trial_id,
            params_patch=params_patch,
            score_cfg=score_cfg,
            model_id=model_id,
            model_type=model_type,
            base_model_cfg=base_model_cfg,
            start=start,
            end=end,
            label_cutoff=label_cutoff,
            execution_override=exec_cfg,
            label_root=label_root,
            trials_root=trials_root,
            objective_metric=objective_metric,
            higher_is_better=higher_is_better,
            bins=bins,
            run_model_score_fn=run_model_score,
        )

    if search_type in {"wandb", "wandb_sweep", "sweep"}:
        return run_wandb_sweep_trials(
            score_cfg=score_cfg,
            model_id=model_id,
            start=start,
            end=end,
            execution_override=execution_override,
            out_dir=out_dir,
            tuning_cfg=tuning_cfg,
            space=space,
            objective_metric=objective_metric,
            execute_trial_fn=_execute_trial,
        )

    trial_params_list = build_trials(
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

    for idx, params_patch in enumerate(trial_params_list, start=1):
        trial_id = f"trial_{idx:03d}"
        row, candidate = _execute_trial(
            idx=idx,
            total=len(trial_params_list),
            trial_id=trial_id,
            params_patch=params_patch,
            exec_cfg=execution_override,
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
