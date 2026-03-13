from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file
from cbond_on.services.model.model_score_service import run as run_model_score


def main( 
    *,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> None:
    cfg = load_config_file("model_score")
    result = run_model_score(
        model_id=model_id or cfg.get("model_id") or cfg.get("default_model_id"),
        start=start or cfg.get("start"),
        end=end or cfg.get("end"),
        label_cutoff=label_cutoff or cfg.get("label_cutoff"),
        cfg=cfg,
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build model scores from services layer")
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--label-cutoff")
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        label_cutoff=args.label_cutoff,
    )

