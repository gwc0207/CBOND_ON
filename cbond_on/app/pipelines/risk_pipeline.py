from __future__ import annotations

from pathlib import Path
from typing import Any

from cbond_on.app.usecases.risk_runtime import run as run_risk


def execute(
    cfg: dict[str, Any],
    *,
    paths_cfg: dict[str, Any],
    start: str | None = None,
    end: str | None = None,
    strategy_positions_path: str | Path | None = None,
    benchmark_positions_path: str | Path | None = None,
    output_root: str | Path | None = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    return run_risk(
        cfg=cfg,
        paths_cfg=paths_cfg,
        start=start,
        end=end,
        strategy_positions_path=strategy_positions_path,
        benchmark_positions_path=benchmark_positions_path,
        output_root=output_root,
        write_outputs=write_outputs,
    )
