from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def resolve_score_df_for_target(
    score_cache: dict[date, pd.DataFrame],
    score_day: date,
    score_path: Path,
) -> pd.DataFrame:
    score_df = score_cache.get(score_day, pd.DataFrame())
    if score_df is None or score_df.empty:
        raise ValueError(f"no scores for {score_day} in {score_path}")
    return score_df
