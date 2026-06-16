from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from cbond_on.config.loader import load_config_file
from cbond_on.workflows.research.factor_batch import run


def main() -> None:
    cfg = deepcopy(load_config_file("factor"))
    cfg["factor_files"] = ["factor/ai_factory/seed/ai_factor_factory_seed.json5"]
    paths_cfg = deepcopy(load_config_file("paths"))
    paths_cfg["results_root"] = "D:/cbond_on/results/ai_factor_factory/wave80_20260608"

    print("WAVE80_FULL_START", datetime.now().isoformat(timespec="seconds"), flush=True)
    print(
        "CONFIG",
        {
            "start": cfg.get("start"),
            "end": cfg.get("end"),
            "workers": cfg.get("workers"),
            "factor_workers": cfg.get("factor_workers"),
            "backtest_workers": (cfg.get("backtest") or {}).get("workers"),
            "factor_files": cfg.get("factor_files"),
        },
        flush=True,
    )
    out = run(cfg, paths_cfg=paths_cfg)
    print("WAVE80_FULL_OUT", out, flush=True)
    print("WAVE80_FULL_DONE", datetime.now().isoformat(timespec="seconds"), flush=True)


if __name__ == "__main__":
    main()
