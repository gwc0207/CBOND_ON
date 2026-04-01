from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file
from cbond_on.services.model.model_eval_service import run as run_model_eval


def main(
    *,
    config_name: str | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> None:
    cfg = load_config_file(config_name or "score/model_eval")
    result = run_model_eval(
        cfg=cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation and hyperparameter tuning")
    parser.add_argument("--config", default="score/model_eval")
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--label-cutoff")
    args = parser.parse_args()
    main(
        config_name=args.config,
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        label_cutoff=args.label_cutoff,
    )
