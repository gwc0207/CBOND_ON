from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from cbond_on.core.config import resolve_output_path
from cbond_on.infra.model.eval.evaluator import (
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
)


def run_single_eval(
    *,
    refreshed_model_cfg: dict[str, Any],
    chosen_model_id: str,
    eval_block: dict[str, Any],
    paths_cfg: dict[str, Any],
    start_day: date,
    end_day: date,
) -> tuple[dict[str, Any], Any]:
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
    return summary, eval_result.daily

