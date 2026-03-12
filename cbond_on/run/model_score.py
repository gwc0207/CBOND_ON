from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file


def main(
    *,
    model_type: str | None = None,
    model_config: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> None:
    cfg = {}
    try:
        cfg = load_config_file("model_score")
    except FileNotFoundError:
        cfg = {}
    model_type = str(model_type or cfg.get("model_type", "linear")).lower()
    model_config = str(model_config or cfg.get("model_config", "models/linear/model"))
    start = start or cfg.get("start")
    end = end or cfg.get("end")
    label_cutoff = label_cutoff or cfg.get("label_cutoff")

    if model_type == "linear":
        from cbond_on.run.linear import train_linear
        cfg_path = Path(model_config)
        if model_config.endswith(".json5") and cfg_path.exists():
            sys.argv = [sys.argv[0], str(cfg_path)]
        else:
            sys.argv = [sys.argv[0]]
        train_linear.main(start=start, end=end)
        return
    if model_type == "lgbm":
        from cbond_on.run.lgbm import train_lgbm
        cfg_path = Path(model_config)
        if model_config.endswith(".json5") and cfg_path.exists():
            sys.argv = [sys.argv[0], str(cfg_path)]
        else:
            sys.argv = [sys.argv[0]]
        train_lgbm.main(start=start, end=end, label_cutoff=label_cutoff)
        return
    if model_type == "lgbm_ranker":
        from cbond_on.run.lgbm_ranker import train_lgbm_ranker
        cfg_path = Path(model_config)
        if model_config.endswith(".json5") and cfg_path.exists():
            sys.argv = [sys.argv[0], str(cfg_path)]
        else:
            sys.argv = [sys.argv[0]]
        train_lgbm_ranker.main(start=start, end=end, label_cutoff=label_cutoff)
        return

    raise ValueError(f"unsupported model_type: {model_type}")


if __name__ == "__main__":
    main()
