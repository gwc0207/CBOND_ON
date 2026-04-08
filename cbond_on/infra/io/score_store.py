from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.models.score_io import load_scores_by_date, write_scores_by_date


def load_score_frames(root: Path) -> dict[date, pd.DataFrame]:
    return load_scores_by_date(root)


def write_score_frames(
    root: Path,
    frame: pd.DataFrame,
    *,
    overwrite: bool = False,
    dedupe: bool = True,
) -> None:
    write_scores_by_date(root, frame, overwrite=overwrite, dedupe=dedupe)

