from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.interfaces.cli.model_score import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build model scores from services layer")
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--label-cutoff")
    parser.add_argument("--refit-every-n-days", type=int)
    parser.add_argument("--train-processes", type=int)
    parser.add_argument("--parallel-shards", type=int)
    parser.add_argument("--parallel-shard-index", type=int)
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        label_cutoff=args.label_cutoff,
        refit_every_n_days=args.refit_every_n_days,
        train_processes=args.train_processes,
        parallel_shards=args.parallel_shards,
        parallel_shard_index=args.parallel_shard_index,
    )

