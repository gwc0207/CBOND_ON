from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.infra.model.impl.lgbm.trainer import _read_label_day, evaluate_metrics
from cbond_on.infra.model.score_io import load_scores_by_date


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


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _annualized_sharpe(returns: pd.Series, annualization: float) -> float:
    ret = pd.to_numeric(returns, errors="coerce").dropna()
    if ret.empty:
        return float("nan")
    std = float(ret.std(ddof=0))
    if std <= 0:
        return float("nan")
    return float(ret.mean() / std * np.sqrt(annualization))


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    ret = pd.to_numeric(returns, errors="coerce").dropna()
    if ret.empty:
        return float("nan")
    nav = (1.0 + ret).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(drawdown.min())


def _rolling_sharpe(returns: pd.Series, *, window: int, min_periods: int, annualization: float) -> pd.Series:
    ret = pd.to_numeric(returns, errors="coerce")
    if ret.empty:
        return pd.Series(dtype=float)
    window = max(1, int(window))
    min_periods = max(1, min(int(min_periods), window))
    rolling_mean = ret.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = ret.rolling(window=window, min_periods=min_periods).std(ddof=0)
    out = rolling_mean / rolling_std.replace(0.0, np.nan) * np.sqrt(annualization)
    return out.replace([np.inf, -np.inf], np.nan)


def _top_bin_return(group: pd.DataFrame, *, bins: int) -> tuple[float, int]:
    block = group[["score", "y"]].copy()
    block["score"] = pd.to_numeric(block["score"], errors="coerce")
    block["y"] = pd.to_numeric(block["y"], errors="coerce")
    block = block.dropna(subset=["score", "y"])
    if block.empty:
        return float("nan"), 0

    n_bins = max(1, min(int(bins), int(len(block))))
    if n_bins <= 1:
        selected = block
    else:
        ranked = block["score"].rank(method="first")
        try:
            bin_id = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            return float("nan"), 0
        if bin_id is None:
            return float("nan"), 0
        block = block.assign(_bin=pd.to_numeric(bin_id, errors="coerce"))
        if block["_bin"].isna().all():
            return float("nan"), 0
        selected = block.loc[block["_bin"] == block["_bin"].max()]
    if selected.empty:
        return float("nan"), 0
    return float(selected["y"].mean()), int(len(selected))


def evaluate_merged_scores(
    merged: pd.DataFrame,
    *,
    bins: int = 5,
    stability_window: int = 40,
    stability_min_periods: int = 20,
    annualization: float = 252.0,
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
            "top_bin_ret_mean": float("nan"),
            "top_bin_ret_std": float("nan"),
            "top_bin_sharpe": float("nan"),
            "top_bin_win_rate": float("nan"),
            "top_bin_max_drawdown": float("nan"),
            "top_bin_rolling_sharpe_mean": float("nan"),
            "top_bin_rolling_sharpe_positive_ratio": float("nan"),
            "sharpe_stability_score": float("nan"),
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
        top_bin_ret, top_bin_count = _top_bin_return(group, bins=bins)
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
                "top_bin_return": top_bin_ret,
                "top_bin_count": top_bin_count,
            }
        )

    daily = pd.DataFrame(day_rows).sort_values("trade_date").reset_index(drop=True)
    rank_ic_std = float(pd.to_numeric(daily["rank_ic"], errors="coerce").std(ddof=0))
    ic_std = float(pd.to_numeric(daily["ic"], errors="coerce").std(ddof=0))
    rank_ic_mean = float(pd.to_numeric(daily["rank_ic"], errors="coerce").mean())
    ic_mean = float(pd.to_numeric(daily["ic"], errors="coerce").mean())
    top_bin_ret = pd.to_numeric(daily["top_bin_return"], errors="coerce")
    top_bin_ret_clean = top_bin_ret.dropna()
    top_bin_rolling_sharpe = _rolling_sharpe(
        top_bin_ret,
        window=stability_window,
        min_periods=stability_min_periods,
        annualization=annualization,
    )
    rolling_clean = top_bin_rolling_sharpe.dropna()
    top_bin_sharpe = _annualized_sharpe(top_bin_ret, annualization=annualization)
    rolling_positive_ratio = (
        float((rolling_clean > 0).mean()) if not rolling_clean.empty else float("nan")
    )
    sharpe_stability_score = (
        top_bin_sharpe * rolling_positive_ratio
        if np.isfinite(top_bin_sharpe) and np.isfinite(rolling_positive_ratio)
        else float("nan")
    )
    daily["top_bin_rolling_sharpe"] = top_bin_rolling_sharpe
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
        "top_bin_ret_mean": _safe_float(top_bin_ret_clean.mean()),
        "top_bin_ret_std": _safe_float(top_bin_ret_clean.std(ddof=0)),
        "top_bin_sharpe": top_bin_sharpe,
        "top_bin_win_rate": _safe_float((top_bin_ret_clean > 0).mean()) if not top_bin_ret_clean.empty else float("nan"),
        "top_bin_max_drawdown": _max_drawdown_from_returns(top_bin_ret),
        "top_bin_rolling_sharpe_mean": _safe_float(rolling_clean.mean()),
        "top_bin_rolling_sharpe_positive_ratio": rolling_positive_ratio,
        "sharpe_stability_score": sharpe_stability_score,
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

