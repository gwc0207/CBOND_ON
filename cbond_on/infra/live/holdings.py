from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_previous_holdings(results_root: Path, day: date) -> pd.DataFrame:
    if not results_root.exists():
        return pd.DataFrame(columns=["code", "weight"])
    dirs = sorted([p for p in results_root.iterdir() if p.is_dir() and p.name < f"{day:%Y-%m-%d}"])
    if not dirs:
        return pd.DataFrame(columns=["code", "weight"])
    latest = dirs[-1] / "trade_list.csv"
    if not latest.exists():
        return pd.DataFrame(columns=["code", "weight"])
    try:
        df = pd.read_csv(latest)
    except Exception:
        return pd.DataFrame(columns=["code", "weight"])
    if "code" not in df.columns:
        return pd.DataFrame(columns=["code", "weight"])
    if "weight" not in df.columns:
        df["weight"] = 0.0
    return df[["code", "weight"]].copy()
