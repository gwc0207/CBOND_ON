from __future__ import annotations

import argparse

from cbond_on.bootstrap.research import load_model_eval_config
from cbond_on.workflows.research.model_eval import run


def main(
    *,
    config_name: str | None = None,
    model_id: str | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
) -> None:
    cfg = load_model_eval_config(config_name or "score/model_eval")
    result = run(
        cfg,
        model_id=model_id,
        start=start,
        end=end,
        label_cutoff=label_cutoff,
    )
    print(result)


def cli_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run model evaluation and hyperparameter tuning")
    parser.add_argument("--config", default="score/model_eval")
    parser.add_argument("--model-id")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--label-cutoff")
    args = parser.parse_args(argv)
    main(
        config_name=args.config,
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        label_cutoff=args.label_cutoff,
    )


if __name__ == "__main__":
    cli_main()
