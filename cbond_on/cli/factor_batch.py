from __future__ import annotations

from cbond_on.bootstrap.research import load_factor_batch_inputs
from cbond_on.domain.factors import defs  # noqa: F401
from cbond_on.workflows.research.factor_batch import run


def main() -> None:
    cfg, paths_cfg = load_factor_batch_inputs()
    out_root = run(cfg, paths_cfg=paths_cfg)
    print({"out_root": str(out_root)})


if __name__ == "__main__":
    main()

