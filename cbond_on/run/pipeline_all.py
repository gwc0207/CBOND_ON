from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.services.backtest.backtest_service import run as run_backtest
from cbond_on.services.data.clean_service import run as run_clean
from cbond_on.services.data.label_service import run as run_label
from cbond_on.services.data.panel_service import run as run_panel
from cbond_on.services.data.raw_service import run as run_raw
from cbond_on.services.factor.factor_build_service import run as run_factor_build
from cbond_on.services.model.model_score_service import run as run_model_score


def main() -> None:
    raw_cfg = load_config_file("raw_data")
    clean_cfg = load_config_file("cleaned_data")
    panel_cfg = load_config_file("panel")
    label_cfg = load_config_file("label")
    factor_cfg = load_config_file("factor")
    model_cfg = load_config_file("model_score")
    bt_cfg = load_config_file("backtest")

    run_raw(
        start=parse_date(raw_cfg.get("start")),
        end=parse_date(raw_cfg.get("end")),
        refresh=bool(raw_cfg.get("refresh", False)),
        overwrite=bool(raw_cfg.get("overwrite", False)),
        cfg=raw_cfg,
    )
    run_clean(
        start=parse_date(clean_cfg.get("start")),
        end=parse_date(clean_cfg.get("end")),
        refresh=bool(clean_cfg.get("refresh", False)),
        overwrite=bool(clean_cfg.get("overwrite", False)),
        cfg=clean_cfg,
    )
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
    main()
