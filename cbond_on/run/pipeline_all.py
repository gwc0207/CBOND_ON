from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.backtest.backtest_service import run as run_backtest
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.model.model_score_service import run as run_model_score


def _require_section(cfg: dict, key: str) -> dict:
    section = cfg.get(key)
    if not isinstance(section, dict):
        raise KeyError(f"pipeline_all config missing section: {key}")
    return dict(section)


def main(*, config_name: str = "pipeline_all") -> None:
    pipeline_cfg = load_config_file(config_name)
    panel_cfg = _require_section(pipeline_cfg, "panel")
    label_cfg = _require_section(pipeline_cfg, "label")
    factor_cfg = _require_section(pipeline_cfg, "factor")
    model_cfg = _require_section(pipeline_cfg, "model_score")
    bt_cfg = _require_section(pipeline_cfg, "backtest")

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
