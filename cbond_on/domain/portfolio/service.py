from __future__ import annotations

import pandas as pd


def normalize_weights(
    picks: pd.DataFrame,
    *,
    weight_col: str = "weight",
) -> pd.DataFrame:
    if picks is None or picks.empty:
        return pd.DataFrame(columns=["code", "score", "weight", "rank"])

    out = picks.copy()
    if weight_col not in out.columns:
        out[weight_col] = 0.0
    out[weight_col] = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(out[weight_col].sum())
    if total <= 0.0:
        out[weight_col] = 1.0 / len(out)
    else:
        out[weight_col] = out[weight_col] / total
    return out


def to_prev_positions(picks: pd.DataFrame, *, code_col: str = "code", weight_col: str = "weight") -> pd.DataFrame:
    if picks is None or picks.empty:
        return pd.DataFrame(columns=["code", "weight"])
    if code_col not in picks.columns:
        return pd.DataFrame(columns=["code", "weight"])
    out = picks[[code_col]].copy()
    if weight_col in picks.columns:
        out["weight"] = pd.to_numeric(picks[weight_col], errors="coerce").fillna(0.0)
    else:
        out["weight"] = 0.0
    out = out.rename(columns={code_col: "code"})
    out["code"] = out["code"].astype(str)
    return out[["code", "weight"]]

