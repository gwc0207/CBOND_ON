from __future__ import annotations

import argparse
import subprocess
import sys

from cbond_on.bootstrap.research import load_model_score_config
from cbond_on.workflows.research.model_score import apply_execution_overrides, run


def main(
    *,
    config_name: str | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    refit_every_n_days: int | None = None,
    train_processes: int | None = None,
    parallel_shards: int | None = None,
    parallel_shard_index: int | None = None,
) -> None:
    cfg = load_model_score_config(config_name or "model_score")
    cfg = apply_execution_overrides(
        cfg,
        refit_every_n_days=refit_every_n_days,
        train_processes=train_processes,
        parallel_shards=parallel_shards,
        parallel_shard_index=parallel_shard_index,
    )
    execution_cfg = dict(cfg.get("execution", {}))

    effective_shards = int(execution_cfg.get("parallel_shards", 1))
    if parallel_shard_index is None and effective_shards > 1:
        base_cmd = [sys.executable, "-m", "cbond_on.cli.model_score"]
        if config_name:
            base_cmd += ["--config", str(config_name)]
        if model_id:
            base_cmd += ["--model-id", str(model_id)]
        if start:
            base_cmd += ["--start", str(start)]
        if end:
            base_cmd += ["--end", str(end)]
        if label_cutoff:
            base_cmd += ["--label-cutoff", str(label_cutoff)]
        if refit_every_n_days is not None:
            base_cmd += ["--refit-every-n-days", str(int(refit_every_n_days))]
        if train_processes is not None:
            base_cmd += ["--train-processes", str(int(train_processes))]

        procs: list[subprocess.Popen] = []
        for shard_idx in range(effective_shards):
            cmd = base_cmd + [
                "--parallel-shards",
                str(effective_shards),
                "--parallel-shard-index",
                str(shard_idx),
            ]
            print(f"[model_score] launch shard {shard_idx + 1}/{effective_shards}: {' '.join(cmd)}")
            procs.append(subprocess.Popen(cmd))
        exit_codes = [p.wait() for p in procs]
        failed = [i for i, code in enumerate(exit_codes) if code != 0]
        if failed:
            raise RuntimeError(f"model_score shard failed: {failed}, codes={exit_codes}")
        print({"model_id": model_id or cfg.get("model_id"), "parallel_shards": effective_shards, "status": "ok"})
        return

    result = run(
        cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )
    print(result)


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build model scores from application layer")
    parser.add_argument("--config", default="model_score")
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--label-cutoff")
    parser.add_argument("--refit-every-n-days", type=int)
    parser.add_argument("--train-processes", type=int)
    parser.add_argument("--parallel-shards", type=int)
    parser.add_argument("--parallel-shard-index", type=int)
    args = parser.parse_args(argv)
    main(
        config_name=args.config,
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        label_cutoff=args.label_cutoff,
        refit_every_n_days=args.refit_every_n_days,
        train_processes=args.train_processes,
        parallel_shards=args.parallel_shards,
        parallel_shard_index=args.parallel_shard_index,
    )


if __name__ == "__main__":
    cli_main()
