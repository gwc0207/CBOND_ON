from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_guard_stats(pred: Any, *, equal_tol: float = 1e-12) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(pred), errors="coerce")
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n <= 0:
        return {
            "score_count": 0,
            "score_min": float("nan"),
            "score_max": float("nan"),
            "score_range": float("nan"),
            "score_std": float("nan"),
            "score_unique_count": 0,
            "score_pos_ratio": float("nan"),
            "score_neg_ratio": float("nan"),
            "score_zero_ratio": float("nan"),
            "score_same_sign": False,
            "score_all_equal": False,
        }

    score_min = float(np.min(arr))
    score_max = float(np.max(arr))
    score_range = float(score_max - score_min)
    score_std = float(np.std(arr, ddof=0))
    score_unique_count = int(np.unique(arr).size)

    pos = int(np.sum(arr > 0))
    neg = int(np.sum(arr < 0))
    zero = n - pos - neg
    pos_ratio = float(pos / n)
    neg_ratio = float(neg / n)
    zero_ratio = float(zero / n)

    return {
        "score_count": n,
        "score_min": score_min,
        "score_max": score_max,
        "score_range": score_range,
        "score_std": score_std,
        "score_unique_count": score_unique_count,
        "score_pos_ratio": pos_ratio,
        "score_neg_ratio": neg_ratio,
        "score_zero_ratio": zero_ratio,
        "score_same_sign": bool(pos == 0 or neg == 0),
        "score_all_equal": bool(score_range <= float(equal_tol)),
    }


def score_guard_flags(stats: dict[str, Any], *, warn_same_sign: bool = False) -> list[str]:
    flags: list[str] = []
    if bool(stats.get("score_all_equal", False)):
        flags.append("all_equal")
    if bool(warn_same_sign) and bool(stats.get("score_same_sign", False)):
        flags.append("same_sign")
    return flags


def score_guard_bin_stats(
    pred: Any,
    *,
    target_bins: int = 20,
    min_count: int = 20,
) -> dict[str, Any]:
    bins_target = max(2, int(target_bins))
    min_samples = max(1, int(min_count))
    series = pd.to_numeric(pd.Series(pred), errors="coerce")
    arr = series.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    result = {
        "score_bin_target": int(bins_target),
        "score_bin_min_samples": int(min_samples),
        "score_bin_count": int(0),
        "score_bin_guard_checked": bool(False),
        "score_bin_insufficient": bool(False),
    }
    if n < min_samples:
        return result

    s = pd.Series(arr)
    labels: pd.Series
    try:
        ranks = s.rank(method="average", pct=True)
        labels = pd.qcut(ranks, q=bins_target, labels=False, duplicates="drop")
    except Exception:
        try:
            labels = pd.qcut(s, q=bins_target, labels=False, duplicates="drop")
        except Exception:
            labels = pd.Series(dtype=float)

    actual_bins = int(pd.to_numeric(pd.Series(labels), errors="coerce").dropna().nunique())
    result["score_bin_count"] = int(actual_bins)
    result["score_bin_guard_checked"] = bool(True)
    result["score_bin_insufficient"] = bool(actual_bins < bins_target)
    return result
