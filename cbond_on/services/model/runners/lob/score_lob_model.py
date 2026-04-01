from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.models.impl.lob.score_builder import ScoreConfig, build_scores


def main() -> None:
    paths_cfg = load_config_file("paths")
    ds_cfg = load_config_file("models/lob/dataset")
    bt_cfg = load_config_file("backtest")
    model_cfg = load_config_file("models/lob/model")

    start = parse_date(bt_cfg["start"])
    end = parse_date(bt_cfg["end"])

    clean_root = Path(paths_cfg["clean_data_root"])
    output_dir = clean_root / str(ds_cfg.get("output_dir", "LOBDS"))
    results_root = paths_cfg["results_root"]
    weights_path = resolve_output_path(
        model_cfg.get("weights_path"),
        default_path=Path(results_root) / "models" / "lob_st_default" / "model.pt",
        results_root=results_root,
    )
    score_path = resolve_output_path(
        model_cfg.get("score_output"),
        default_path=Path(results_root) / "scores" / "lob_st_default",
        results_root=results_root,
    )

    train_cfg = model_cfg.get("train", {})
    score_cfg = ScoreConfig(
        device=str(train_cfg.get("device", "cpu")),
        batch_size=int(train_cfg.get("batch_size", 16)),
    )
    params = model_cfg.get("params", {})

    build_scores(
        dataset_root=output_dir,
        weights_path=weights_path,
        start=start,
        end=end,
        model_params=params,
        score_cfg=score_cfg,
        output_path=score_path,
    )
    print(f"saved scores: {score_path}")


if __name__ == "__main__":
    main()
