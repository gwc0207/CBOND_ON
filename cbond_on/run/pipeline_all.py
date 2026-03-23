from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.backtest.backtest_service import run as run_backtest
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.model.model_score_service import run as run_model_score


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


def main(*, config_name: str = "pipeline_all") -> None:
    pipeline_cfg = dict(load_config_file(config_name))
    panel_cfg, label_cfg, factor_cfg, model_cfg, bt_cfg = _build_runtime_configs(pipeline_cfg)

    run_panel(
        start=parse_date(panel_cfg.get("start")),
        end=parse_date(panel_cfg.get("end")),
        refresh=bool(panel_cfg.get("refresh", False)),
        overwrite=bool(panel_cfg.get("overwrite", False)),
        cfg=panel_cfg,
    )
    run_label(
        start=parse_date(label_cfg.get("start")),
        end=parse_date(label_cfg.get("end")),
        refresh=bool(label_cfg.get("refresh", False)),
        overwrite=bool(label_cfg.get("overwrite", False)),
        cfg=label_cfg,
        panel_cfg=panel_cfg,
    )
    run_factor_build(
        start=parse_date(factor_cfg.get("start")),
        end=parse_date(factor_cfg.get("end")),
        refresh=bool(factor_cfg.get("refresh", False)),
        overwrite=bool(factor_cfg.get("overwrite", False)),
        cfg=factor_cfg,
    )
    run_model_score(
        model_id=model_cfg.get("model_id") or model_cfg.get("default_model_id"),
        start=model_cfg.get("start"),
        end=model_cfg.get("end"),
        label_cutoff=model_cfg.get("label_cutoff"),
        cfg=model_cfg,
    )
    run_backtest(
        start=parse_date(bt_cfg.get("start")),
        end=parse_date(bt_cfg.get("end")),
        cfg=bt_cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full pipeline using one pipeline_all config.")
    parser.add_argument(
        "--config",
        default="pipeline_all",
        help="config key or config path for pipeline_all (default: pipeline_all)",
    )
    args = parser.parse_args()
    main(config_name=args.config)
