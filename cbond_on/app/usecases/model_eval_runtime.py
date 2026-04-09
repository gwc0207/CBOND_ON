from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.core.config import load_config_file, parse_date
from cbond_on.infra.model.eval_artifacts import load_best_trial_daily, write_config_snapshot
from cbond_on.infra.model.eval_config import (
    build_output_dir,
    load_eval_config,
    load_score_config,
    resolve_single_model,
)
from cbond_on.infra.model.single_eval import run_single_eval
from cbond_on.infra.model.eval.reporter import write_report_bundle
from cbond_on.infra.model.eval.tuner import run_hyperparameter_tuning


def run(
    *,
    cfg: dict[str, Any] | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict[str, Any]:
    eval_cfg = load_eval_config(cfg)
    paths_cfg = load_config_file("paths")

    score_config_key = str(eval_cfg.get("model_score_config", "score/model_score"))
    score_cfg = load_score_config(score_config_key)

    chosen_model_id = str(
        model_id
        or eval_cfg.get("model_id")
        or score_cfg.get("model_id")
        or score_cfg.get("default_model_id")
        or ""
    ).strip()
    if not chosen_model_id:
        raise ValueError("model_eval missing model_id")

    model_entry, model_cfg = resolve_single_model(score_cfg, chosen_model_id)

    start_day = parse_date(start or eval_cfg.get("start") or score_cfg.get("start") or model_cfg.get("start"))
    end_day = parse_date(end or eval_cfg.get("end") or score_cfg.get("end") or model_cfg.get("end"))
    cutoff_raw = label_cutoff or eval_cfg.get("label_cutoff")
    cutoff_day = parse_date(cutoff_raw) if cutoff_raw else None
    if start_day > end_day:
        raise ValueError("start must be <= end")

    experiment_name = str(eval_cfg.get("experiment_name", "model_eval")).strip() or "model_eval"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = build_output_dir(paths_cfg, experiment_name=experiment_name, timestamp=ts)
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_override = dict(score_cfg.get("execution", {}))
    execution_override.update(dict(eval_cfg.get("execution", {})))

    run_model_before_eval = bool(eval_cfg.get("run_model_before_eval", True))
    eval_block = dict(eval_cfg.get("eval", {}))
    objective_metric = str(dict(eval_block.get("objective", {})).get("metric", "rank_ic_mean"))

    tuning_cfg = dict(eval_cfg.get("tuning", {}))
    tuning_enabled = bool(tuning_cfg.get("enabled", False))

    config_snapshot = {
        "model_eval": eval_cfg,
        "model_score_config_key": score_config_key,
        "model_score": score_cfg,
        "model_id": chosen_model_id,
        "model_entry": model_entry,
        "model_config": model_cfg,
        "start": str(start_day),
        "end": str(end_day),
        "label_cutoff": str(cutoff_day) if cutoff_day else None,
    }
    write_config_snapshot(out_dir, config_snapshot)

    if tuning_enabled:
        print(f"[model_eval] tuning enabled model_id={chosen_model_id}")
        trial_df, best_trial = run_hyperparameter_tuning(
            score_cfg=score_cfg,
            model_id=chosen_model_id,
            start=start_day,
            end=end_day,
            label_cutoff=cutoff_day,
            execution_override=execution_override,
            label_root=paths_cfg["label_data_root"],
            out_dir=out_dir,
            tuning_cfg=tuning_cfg,
            eval_cfg=eval_block,
        )
        summary = {
            "mode": "tuning",
            "model_id": chosen_model_id,
            "trials": int(len(trial_df)),
            "ok_trials": int((trial_df.get("status") == "ok").sum()) if not trial_df.empty else 0,
            "best_trial_id": best_trial.get("trial_id") if best_trial else None,
            "best_objective_value": best_trial.get("objective_value") if best_trial else None,
            "objective_metric": objective_metric,
        }
        daily = load_best_trial_daily(out_dir, best_trial)
        write_report_bundle(
            out_dir=out_dir,
            experiment_name=experiment_name,
            config_snapshot=config_snapshot,
            summary=summary,
            daily=daily,
            trials=trial_df,
            objective_key="objective_value",
        )
        return {
            "mode": "tuning",
            "out_dir": str(out_dir),
            "best_trial": best_trial,
        }

    score_cfg_for_run = copy.deepcopy(score_cfg)
    score_cfg_for_run["execution"] = execution_override
    if run_model_before_eval:
        print(f"[model_eval] running model before eval model_id={chosen_model_id}")
        run_model_score(
            model_id=chosen_model_id,
            start=start_day,
            end=end_day,
            label_cutoff=cutoff_day,
            cfg=score_cfg_for_run,
        )

    _, refreshed_model_cfg = resolve_single_model(score_cfg, chosen_model_id)
    summary, daily = run_single_eval(
        refreshed_model_cfg=refreshed_model_cfg,
        chosen_model_id=chosen_model_id,
        eval_block=eval_block,
        paths_cfg=paths_cfg,
        start_day=start_day,
        end_day=end_day,
    )
    write_report_bundle(
        out_dir=out_dir,
        experiment_name=experiment_name,
        config_snapshot=config_snapshot,
        summary=summary,
        daily=daily,
        trials=None,
    )
    return {
        "mode": "single_eval",
        "out_dir": str(out_dir),
        "summary": summary,
    }

