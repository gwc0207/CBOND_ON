from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.models.impl.lgbm.trainer import _read_label_day, evaluate_metrics
from cbond_on.models.score_io import load_scores_by_date


@dataclass
class EvaluationResult:
    merged: pd.DataFrame
    daily: pd.DataFrame
    summary: dict[str, Any]


def load_scores_frame(score_output: str | Path) -> pd.DataFrame:
    cache = load_scores_by_date(score_output)
    frames: list[pd.DataFrame] = []
    for day, group in sorted(cache.items(), key=lambda kv: kv[0]):
        if group is None or group.empty:
            continue
        block = group.copy()
        block["trade_date"] = day
        frames.append(block[["trade_date", "code", "score"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "code", "score"])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out["code"] = out["code"].astype(str)
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out = out.dropna(subset=["trade_date", "code", "score"])
    return out


def merge_score_with_label(
    *,
    scores: pd.DataFrame,
    label_root: str | Path,
    factor_time: str,
    label_time: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    if scores.empty:
        return pd.DataFrame(columns=["trade_date", "dt", "code", "score", "y"])
    frames: list[pd.DataFrame] = []
    label_root_path = Path(label_root)
    for trade_day in sorted(set(scores["trade_date"])):
        if start is not None and trade_day < start:
            continue
        if end is not None and trade_day > end:
            continue
        label_df = _read_label_day(
            label_root_path,
            trade_day,
            factor_time=factor_time,
            label_time=label_time,
        )
        if label_df.empty or "dt" not in label_df.columns:
            continue
        label_df = label_df[["dt", "code", "y"]].dropna(subset=["dt", "code", "y"]).copy()
        if label_df.empty:
            continue
        label_df["code"] = label_df["code"].astype(str)

        score_df = scores.loc[scores["trade_date"] == trade_day, ["trade_date", "code", "score"]].copy()
        score_df["code"] = score_df["code"].astype(str)
        score_df["dt"] = pd.to_datetime(score_df["trade_date"], errors="coerce")
        score_df["dt"] = score_df["dt"] + pd.to_timedelta(f"{factor_time}:00")

        merged = score_df.merge(label_df, on=["dt", "code"], how="inner")
        if merged.empty:
            continue
        frames.append(merged[["trade_date", "dt", "code", "score", "y"]])
    if not frames:
        return pd.DataFrame(columns=["trade_date", "dt", "code", "score", "y"])
    out = pd.concat(frames, ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.date
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna(subset=["trade_date", "dt", "code", "score", "y"])
    return out


def evaluate_merged_scores(
    merged: pd.DataFrame,
    *,
    bins: int = 5,
) -> EvaluationResult:
    if merged.empty:
        summary = {
            "days": 0,
            "samples": 0,
            "rank_ic_mean": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_ir": float("nan"),
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
            "dir_mean": float("nan"),
            "r2_mean": float("nan"),
            "mse_mean": float("nan"),
        }
        return EvaluationResult(
            merged=merged,
            daily=pd.DataFrame(),
            summary=summary,
        )

    day_rows: list[dict[str, Any]] = []
    for trade_day, group in merged.groupby("trade_date", sort=True):
        metrics = evaluate_metrics(
            x=group[["score"]],
            y=group["y"],
            dt=group["dt"],
            pred=group["score"].to_numpy(),
            bins=bins,
        )
        day_rows.append(
            {
                "trade_date": trade_day,
                "count": int(len(group)),
                "mse": metrics["mse"],
                "r2": metrics["r2"],
                "dir": metrics["dir"],
                "ic": metrics["ic_mean"],
                "ic_ir": metrics["ic_ir"],
                "rank_ic": metrics["rank_ic_mean"],
                "rank_ic_ir": metrics["rank_ic_ir"],
            }
        )

    daily = pd.DataFrame(day_rows).sort_values("trade_date").reset_index(drop=True)
    rank_ic_std = float(pd.to_numeric(daily["rank_ic"], errors="coerce").std(ddof=0))
    ic_std = float(pd.to_numeric(daily["ic"], errors="coerce").std(ddof=0))
    rank_ic_mean = float(pd.to_numeric(daily["rank_ic"], errors="coerce").mean())
    ic_mean = float(pd.to_numeric(daily["ic"], errors="coerce").mean())
    summary = {
        "days": int(len(daily)),
        "samples": int(len(merged)),
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "rank_ic_ir": (rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else float("nan"),
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": (ic_mean / ic_std) if ic_std > 0 else float("nan"),
        "dir_mean": float(pd.to_numeric(daily["dir"], errors="coerce").mean()),
        "r2_mean": float(pd.to_numeric(daily["r2"], errors="coerce").mean()),
        "mse_mean": float(pd.to_numeric(daily["mse"], errors="coerce").mean()),
    }
    return EvaluationResult(merged=merged, daily=daily, summary=summary)


def objective_from_summary(
    summary: dict[str, Any],
    *,
    metric: str,
    higher_is_better: bool,
) -> float:
    raw = summary.get(metric)
    try:
        value = float(raw)
    except Exception:
        return float("-inf")
    if not np.isfinite(value):
        return float("-inf")
    return value if higher_is_better else -value
