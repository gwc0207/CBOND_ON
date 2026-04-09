from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_config_snapshot(out_dir: Path, snapshot: dict[str, Any]) -> None:
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_best_trial_daily(out_dir: Path, best_trial: dict[str, Any] | None) -> pd.DataFrame:
    daily = pd.DataFrame()
    if best_trial is None:
        return daily
    best_daily_path = out_dir / "trials" / str(best_trial["trial_id"]) / "evaluation_daily.csv"
    if best_daily_path.exists():
        daily = pd.read_csv(best_daily_path)
    return daily
