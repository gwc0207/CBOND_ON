from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.app.usecases.model_score_runtime import run
from cbond_on.config.loader import load_config_file


if __name__ == "__main__":
    cfg = load_config_file("score/model/model_score_old19_research")
    result = run(
        cfg=cfg,
        model_id="lgbm_live_old19_research_202501",
        start="2025-01-01",
        end="2026-06-04",
    )
    print(result)
