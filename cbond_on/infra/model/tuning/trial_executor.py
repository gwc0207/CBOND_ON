from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from cbond_on.infra.model.tuning.space import set_by_dot_path
from cbond_on.infra.model.eval.evaluator import (
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
    objective_from_summary,
)


def execute_single_trial(
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
    run_model_score_fn: Callable[..., dict],
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
        set_by_dot_path(trial_model_cfg, key, value)

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
        run_model_score_fn(
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

