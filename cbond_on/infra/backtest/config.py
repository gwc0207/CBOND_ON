from __future__ import annotations

from pathlib import Path

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import resolve_output_path


def resolve_score_path(cfg: dict, paths_cfg: dict) -> Path:
    score_source = dict(cfg.get("score_source", {}))
    score_root = score_source.get("score_root")
    model_id = score_source.get("model_id")
    if score_root:
        return resolve_output_path(
            score_root,
            default_path=Path(paths_cfg["results_root"]) / "scores" / str(model_id or "model_score"),
            results_root=paths_cfg["results_root"],
        )
    if model_id:
        return Path(paths_cfg["results_root"]) / "scores" / str(model_id)
    raise ValueError("backtest_config.score_source requires score_root or model_id")


def load_strategy_config(path_text: str | None, inline: dict | None = None) -> dict:
    if isinstance(inline, dict):
        return dict(inline)
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)
