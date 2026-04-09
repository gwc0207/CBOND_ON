from __future__ import annotations

from typing import Any, Callable

from cbond_on.core.config import load_config_file
from cbond_on.core.utils import progress
from cbond_on.app.pipelines.panel_pipeline import execute as run_panel_pipeline
from cbond_on.app.pipelines.label_pipeline import execute as run_label_pipeline
from cbond_on.app.pipelines.factor_pipeline import execute as run_factor_pipeline
from cbond_on.app.pipelines.train_score_pipeline import execute as run_train_score_pipeline
from cbond_on.app.pipelines.backtest_pipeline import execute as run_backtest_pipeline


def _stage_switches(cfg: dict[str, Any], stage: str) -> dict[str, bool]:
    raw = cfg.get(stage, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"pipeline_all.{stage} must be an object")
    out: dict[str, bool] = {}
    if "refresh" in raw:
        out["refresh"] = bool(raw.get("refresh"))
    if "overwrite" in raw:
        out["overwrite"] = bool(raw.get("overwrite"))
    return out


def _apply_common_window(stage_cfg: dict[str, Any], *, start: Any, end: Any) -> None:
    if start is not None:
        stage_cfg["start"] = start
    if end is not None:
        stage_cfg["end"] = end


def _build_runtime_configs(pipeline_cfg: dict[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
    panel_cfg = dict(load_config_file("panel"))
    label_cfg = dict(load_config_file("label"))
    factor_cfg = dict(load_config_file("factor"))
    model_cfg = dict(load_config_file("model_score"))
    bt_cfg = dict(load_config_file("backtest"))

    common_start = pipeline_cfg.get("start")
    common_end = pipeline_cfg.get("end")
    if common_start is None or common_end is None:
        raise KeyError("pipeline_all requires both top-level start and end")

    for cfg in (panel_cfg, label_cfg, factor_cfg, model_cfg, bt_cfg):
        _apply_common_window(cfg, start=common_start, end=common_end)

    for stage_name, stage_cfg in (
        ("panel", panel_cfg),
        ("label", label_cfg),
        ("factor", factor_cfg),
        ("model_score", model_cfg),
        ("backtest", bt_cfg),
    ):
        for key, value in _stage_switches(pipeline_cfg, stage_name).items():
            stage_cfg[key] = value

    model_id = pipeline_cfg.get("model_id")
    if model_id:
        model_cfg["model_id"] = str(model_id)
        score_source = bt_cfg.get("score_source")
        if not isinstance(score_source, dict):
            score_source = {}
        score_source = dict(score_source)
        score_source["model_id"] = str(model_id)
        bt_cfg["score_source"] = score_source

    strategy_id = pipeline_cfg.get("strategy_id")
    if strategy_id:
        bt_cfg["strategy_id"] = str(strategy_id)

    return panel_cfg, label_cfg, factor_cfg, model_cfg, bt_cfg


def execute(*, config_name: str = "pipeline_all") -> None:
    pipeline_cfg = dict(load_config_file(config_name))
    panel_cfg, label_cfg, factor_cfg, model_cfg, bt_cfg = _build_runtime_configs(pipeline_cfg)

    stages: list[tuple[str, Callable[[], Any]]] = [
        (
            "panel",
            lambda: run_panel_pipeline(panel_cfg),
        ),
        (
            "label",
            lambda: run_label_pipeline(label_cfg, panel_cfg=panel_cfg),
        ),
        (
            "factor",
            lambda: run_factor_pipeline(factor_cfg),
        ),
        (
            "model_score",
            lambda: run_train_score_pipeline(
                model_cfg,
                model_id=model_cfg.get("model_id") or model_cfg.get("default_model_id"),
                start=model_cfg.get("start"),
                end=model_cfg.get("end"),
                label_cutoff=model_cfg.get("label_cutoff"),
            ),
        ),
        (
            "backtest",
            lambda: run_backtest_pipeline(bt_cfg),
        ),
    ]

    for stage_name, stage_runner in progress(
        stages,
        desc="pipeline_all",
        unit="stage",
        total=len(stages),
    ):
        print(f"[pipeline_all] start {stage_name}")
        stage_runner()
        print(f"[pipeline_all] done {stage_name}")

