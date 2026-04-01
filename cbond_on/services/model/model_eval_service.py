from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.model_eval.evaluator import (
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
)
from cbond_on.model_eval.reporter import write_report_bundle
from cbond_on.model_eval.tuner import run_hyperparameter_tuning
from cbond_on.services.common import load_json_like, resolve_config_path
from cbond_on.services.model.model_score_service import run as run_model_score


def _load_eval_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if cfg is not None:
        return dict(cfg)
    return dict(load_config_file("score/model_eval"))


def _load_score_config(path_key: str) -> dict[str, Any]:
    path = resolve_config_path(path_key)
    return dict(load_json_like(path))


def _resolve_single_model(score_cfg: dict[str, Any], model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"model_id not found in score config: {model_id}")
    model_entry = dict(models[model_id])
    model_cfg_key = str(model_entry.get("model_config", "")).strip()
    if not model_cfg_key:
        raise ValueError(f"model entry missing model_config: {model_id}")
    model_cfg = dict(load_json_like(resolve_config_path(model_cfg_key)))
    return model_entry, model_cfg


def run(
    *,
    cfg: dict[str, Any] | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> dict[str, Any]:
    eval_cfg = _load_eval_config(cfg)
    paths_cfg = load_config_file("paths")

    score_config_key = str(eval_cfg.get("model_score_config", "score/model_score"))
    score_cfg = _load_score_config(score_config_key)

    chosen_model_id = str(
        model_id
        or eval_cfg.get("model_id")
        or score_cfg.get("model_id")
        or score_cfg.get("default_model_id")
        or ""
    ).strip()
    if not chosen_model_id:
        raise ValueError("model_eval missing model_id")

    model_entry, model_cfg = _resolve_single_model(score_cfg, chosen_model_id)

    start_day = parse_date(start or eval_cfg.get("start") or score_cfg.get("start") or model_cfg.get("start"))
    end_day = parse_date(end or eval_cfg.get("end") or score_cfg.get("end") or model_cfg.get("end"))
    cutoff_raw = label_cutoff or eval_cfg.get("label_cutoff")
    cutoff_day = parse_date(cutoff_raw) if cutoff_raw else None
    if start_day > end_day:
        raise ValueError("start must be <= end")

    experiment_name = str(eval_cfg.get("experiment_name", "model_eval")).strip() or "model_eval"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(paths_cfg["results_root"])
        / "model_eval"
        / experiment_name
        / ts
    )
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
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

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
        daily = pd.DataFrame()
        if best_trial is not None:
            best_daily_path = out_dir / "trials" / best_trial["trial_id"] / "evaluation_daily.csv"
            if best_daily_path.exists():
                daily = pd.read_csv(best_daily_path)
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

    _, refreshed_model_cfg = _resolve_single_model(score_cfg, chosen_model_id)
    score_output_raw = refreshed_model_cfg.get("score_output")
    if not score_output_raw:
        raise ValueError(f"model score_output missing for model_id={chosen_model_id}")
    score_output = resolve_output_path(
        score_output_raw,
        default_path=Path(paths_cfg["results_root"]) / "scores" / str(refreshed_model_cfg.get("model_name", chosen_model_id)),
        results_root=paths_cfg["results_root"],
    )

    bins = int(eval_block.get("bins", refreshed_model_cfg.get("bins", 5)))
    factor_time = str(refreshed_model_cfg.get("factor_time", "14:30"))
    label_time = str(refreshed_model_cfg.get("label_time", "14:42"))

    scores = load_scores_frame(score_output)
    merged = merge_score_with_label(
        scores=scores,
        label_root=paths_cfg["label_data_root"],
        factor_time=factor_time,
        label_time=label_time,
        start=start_day,
        end=end_day,
    )
    eval_result = evaluate_merged_scores(merged, bins=bins)
    summary = {
        "mode": "single_eval",
        "model_id": chosen_model_id,
        "score_output": str(score_output),
        **eval_result.summary,
    }
    write_report_bundle(
        out_dir=out_dir,
        experiment_name=experiment_name,
        config_snapshot=config_snapshot,
        summary=summary,
        daily=eval_result.daily,
        trials=None,
    )
    return {
        "mode": "single_eval",
        "out_dir": str(out_dir),
        "summary": summary,
    }
