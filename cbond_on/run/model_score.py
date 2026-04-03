from __future__ import annotations

import argparse
import subprocess
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
    refit_every_n_days: int | None = None,
    train_processes: int | None = None,
    parallel_shards: int | None = None,
    parallel_shard_index: int | None = None,
) -> None:
    cfg = load_config_file("model_score")
    execution_cfg = dict(cfg.get("execution", {}))
    if refit_every_n_days is not None:
        execution_cfg["refit_every_n_days"] = int(refit_every_n_days)
    if train_processes is not None:
        execution_cfg["train_processes"] = int(train_processes)
    if parallel_shards is not None:
        execution_cfg["parallel_shards"] = int(parallel_shards)
    if parallel_shard_index is not None:
        execution_cfg["parallel_shard_index"] = int(parallel_shard_index)
    train_processes_eff = int(execution_cfg.get("train_processes", 1))
    if parallel_shards is None and train_processes_eff > 1:
        execution_cfg["parallel_shards"] = int(train_processes_eff)
    if parallel_shard_index is None:
        execution_cfg.pop("parallel_shard_index", None)
    cfg["execution"] = execution_cfg

    effective_shards = int(execution_cfg.get("parallel_shards", 1))
    if parallel_shard_index is None and effective_shards > 1:
        script_path = Path(__file__).resolve()
        base_cmd = [sys.executable, str(script_path)]
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
                "--parallel-shards", str(effective_shards),
                "--parallel-shard-index", str(shard_idx),
            ]
            print(f"[model_score] launch shard {shard_idx + 1}/{effective_shards}: {' '.join(cmd)}")
            procs.append(subprocess.Popen(cmd))
        exit_codes = [p.wait() for p in procs]
        failed = [i for i, code in enumerate(exit_codes) if code != 0]
        if failed:
            raise RuntimeError(f"model_score shard failed: {failed}, codes={exit_codes}")
        print({"model_id": model_id or cfg.get("model_id"), "parallel_shards": effective_shards, "status": "ok"})
        return

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

