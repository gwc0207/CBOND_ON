"""Read-only replay of a frozen statistical challenger-vs-Base abstention gate.

This is intentionally a research-only replacement diagnostic for the existing
long-history Robust branch.  It does not read the Ridge predictions.  On the
same low-confidence Base branch where the current replay has a valid Robust
diagnostic, it may select a challenger only when a fixed, historical pairwise
statistical gate passes.  Otherwise it abstains and keeps Base.

All state and model-return inputs are read-only.  The only writes are the
research artifacts under the configurable experiments output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf


ANALYSIS_ROOT = Path(r"D:\cbond_on\results\analysis")
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_stat_pairwise_gate_20260728")
STATE_PATH = ANALYSIS_ROOT / "model_switch_t1430_path_full_20260721" / "t1430_market_state_features_path_full.csv"
CURRENT_PATH = ANALYSIS_ROOT / "model_switch_robust_strict_veto_20260728" / "run_20260728_152318" / "daily_current.csv"
RETURN_PATHS = {
    "Regsim": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Challenger_Regsim.csv",
    "Ensemble": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Challenger_Ensemble.csv",
    "HL20": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Champion_HL20.csv",
}
ID_TO_NAME = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": "Regsim",
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": "Ensemble",
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": "HL20",
}
NAME_TO_ID = {value: key for key, value in ID_TO_NAME.items()}
MODELS = ["Regsim", "Ensemble", "HL20"]

# This is the existing path_full_t1430 44-feature research state, split before
# the experiment into two semantic blocks.  Each block contributes exactly half
# of the normalized squared Mahalanobis distance.
PREFIXES = [
    "full0935_1430",
    "seg0935_1000",
    "seg1000_1030",
    "seg1030_1100",
    "seg1100_1130",
    "seg1300_1330",
    "seg1330_1400",
    "seg1400_1430",
]
FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in PREFIXES
    for suffix in ["mean", "std", "iqr", "pos_ratio", "tail_spread"]
] + ["trend_accel_median", "trend_accel_mean", "dispersion_accel", "tail_balance_full"]
TREND_PARTICIPATION_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in PREFIXES
    for suffix in ["mean", "pos_ratio"]
] + ["trend_accel_median", "trend_accel_mean"]
VOL_TAIL_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in PREFIXES
    for suffix in ["std", "iqr", "tail_spread"]
] + ["dispersion_accel", "tail_balance_full"]

# Frozen before execution.  There is deliberately no parameter grid or
# outcome-driven choice in this tool:
#
#   d = sqrt(0.5 * d_trend_participation**2 + 0.5 * d_vol_tail**2)
#
# where each block distance is a rolling-z-score, Ledoit-Wolf shrinkage
# Mahalanobis distance normalized by sqrt(block feature count), so the two
# semantic blocks have equal rather than dimension-proportional influence.
# The 60 nearest dates are equal weighted.  For each challenger c versus Base
# b, LCB(c-b) = trim20(c-b) - winsor10/90_std(c-b) / sqrt(60).  A challenger
# can override Base only when max LCB > 5bp and the current Kth radius is no
# worse than the 75th percentile of Kth radii from *previous ready days*.
LOOKBACK = 360
MIN_PERIODS = 120
NEAREST_K = 60
MIN_ESS = 55.0
DENSITY_QUANTILE = 0.75
TRIM_PROPORTION = 0.20
WINSOR_LO = 0.10
WINSOR_HI = 0.90
LCB_ADVANTAGE_MIN = 0.0005  # 5bp: match the existing Base confidence margin.
VALID_ROBUST_REASONS = {"score_best", "margin_default"}

if len(FEATURES) != 44 or len(TREND_PARTICIPATION_FEATURES) != 18 or len(VOL_TAIL_FEATURES) != 26:
    raise RuntimeError("unexpected path_full_t1430 feature-block definition")
if set(TREND_PARTICIPATION_FEATURES).intersection(VOL_TAIL_FEATURES):
    raise RuntimeError("feature blocks must not overlap")
if set(TREND_PARTICIPATION_FEATURES).union(VOL_TAIL_FEATURES) != set(FEATURES):
    raise RuntimeError("feature blocks must cover the state exactly")


def sha256(path: Path) -> str:
    """Return an input fingerprint without changing that input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trimmed_mean(values: np.ndarray, proportion: float = TRIM_PROPORTION) -> float:
    values = np.sort(np.asarray(values, dtype=float)[np.isfinite(values)])
    if not len(values):
        return float("nan")
    cut = int(math.floor(len(values) * proportion))
    if 2 * cut >= len(values):
        return float(values.mean())
    return float(values[cut : len(values) - cut].mean())


def robust_pairwise_lcb(values: np.ndarray) -> dict[str, float]:
    """Return the fixed project-style robust lower confidence bound.

    The center is a 20%-trimmed mean of daily challenger-minus-Base returns.
    Its uncertainty is the 10/90-winsorized sample standard deviation divided
    by sqrt(K).  This is deliberately the project's ``trim20_lcb10`` form,
    applied directly to the pairwise return difference rather than to separate
    model returns.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n < 2:
        return {
            "n": n,
            "trimmed_mean": float("nan"),
            "winsor_std": float("nan"),
            "standard_error": float("nan"),
            "lcb_trim20": float("nan"),
        }
    lo, hi = np.quantile(values, [WINSOR_LO, WINSOR_HI])
    winsorized = np.clip(values, lo, hi)
    winsor_std = float(np.std(winsorized, ddof=1))
    standard_error = winsor_std / math.sqrt(n)
    center = trimmed_mean(values)
    return {
        "n": n,
        "trimmed_mean": center,
        "winsor_std": winsor_std,
        "standard_error": standard_error,
        "lcb_trim20": float(center - standard_error),
    }


def selector_metrics(returns: np.ndarray, dates: pd.Series, selected: pd.Series) -> dict:
    values = np.asarray(returns, dtype=float)
    if not len(values):
        return {
            "days": 0,
            "start_score_day": None,
            "end_score_day": None,
            "total_return": float("nan"),
            "annualized_return": float("nan"),
            "annualized_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "switch_count": 0,
        }
    nav = np.cumprod(1.0 + values)
    total = float(nav[-1] - 1.0)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0)) if volatility > 0 else float("nan")
    selected_values = selected.astype(str).to_numpy()
    switches = int((selected_values[1:] != selected_values[:-1]).sum()) if len(selected_values) > 1 else 0
    return {
        "days": int(len(values)),
        "start_score_day": str(dates.iloc[0]),
        "end_score_day": str(dates.iloc[-1]),
        "total_return": total,
        "annualized_return": float((1.0 + total) ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "switch_count": switches,
    }


def conditional_metrics(delta: pd.Series | np.ndarray) -> dict:
    """Paired arithmetic return-difference metrics, preserving zero ties."""

    values = np.asarray(pd.Series(delta).dropna(), dtype=float)
    if not len(values):
        return {
            "n": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "mean_delta_bp": float("nan"),
            "median_delta_bp": float("nan"),
            "sum_delta_bp": float("nan"),
            "paired_t_p_one_sided": float("nan"),
            "sign_p_one_sided": float("nan"),
            "wilcoxon_p_one_sided": float("nan"),
        }
    wins = int((values > 0).sum())
    losses = int((values < 0).sum())
    ties = int((values == 0).sum())
    nonzero = values[values != 0]
    result = {
        "n": int(len(values)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_delta_bp": float(values.mean() * 1e4),
        "median_delta_bp": float(np.median(values) * 1e4),
        "sum_delta_bp": float(values.sum() * 1e4),
        "paired_t_p_one_sided": float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue) if len(values) > 1 else float("nan"),
        "sign_p_one_sided": float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="greater").pvalue) if len(nonzero) else float("nan"),
    }
    try:
        # Wilcoxon is undefined for an all-tie sample.  Passing only nonzero
        # values otherwise mirrors its zero-discarding comparison and avoids a
        # noisy numerical warning in an abstention-heavy replay.
        result["wilcoxon_p_one_sided"] = float(stats.wilcoxon(nonzero, alternative="greater", method="auto").pvalue) if len(nonzero) else float("nan")
    except ValueError:
        result["wilcoxon_p_one_sided"] = float("nan")
    return result


def csv_block(frame: pd.DataFrame) -> list[str]:
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load immutable shadow returns, market states, and current replay rows."""

    returns: pd.DataFrame | None = None
    for name, path in RETURN_PATHS.items():
        frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[name] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", name]].dropna().drop_duplicates("score_day", keep="last")
        returns = frame if returns is None else returns.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if returns is None:
        raise RuntimeError("no model-return histories loaded")

    state = pd.read_csv(STATE_PATH, usecols=["trade_date", *FEATURES])
    state["score_day"] = pd.to_datetime(state["trade_date"], errors="coerce").dt.date
    state = state[["score_day", *FEATURES]].dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    for column in FEATURES:
        state[column] = pd.to_numeric(state[column], errors="coerce")
    all_rows = returns.merge(state, on="score_day", how="left", validate="one_to_one").sort_values("score_day").reset_index(drop=True)

    current = pd.read_csv(CURRENT_PATH)
    current["score_day"] = pd.to_datetime(current["score_day"], errors="coerce").dt.date
    current = current.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    current["current_name"] = current["selected_model_id"].map(ID_TO_NAME)
    current["base_name"] = current["base_model_id"].map(ID_TO_NAME)
    if current[["current_name", "base_name"]].isna().any().any():
        raise RuntimeError("unmapped model identifier in current selector replay")
    aligned = current.merge(all_rows, on="score_day", how="inner", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    for row in aligned.itertuples(index=False):
        if not math.isclose(float(getattr(row, row.current_name)), float(row.selected_return), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"current return mismatch on {row.score_day}")
    return all_rows, aligned


def complete_history(all_rows: pd.DataFrame, score_day: object) -> pd.DataFrame:
    """The last <=360 fully observed dates strictly before ``score_day``."""

    history = all_rows.loc[all_rows["score_day"] < score_day].copy()
    complete = np.isfinite(history[FEATURES].to_numpy(float)).all(axis=1) & np.isfinite(history[MODELS].to_numpy(float)).all(axis=1)
    return history.loc[complete].tail(LOOKBACK).copy()


def _zscore(history: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = history.mean(axis=0)
    std = history.std(axis=0)
    std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
    return (history - mean) / std, (current - mean) / std


def block_balanced_distance(history: pd.DataFrame, current_state: np.ndarray) -> dict[str, object]:
    """Fit historical-only two-block shrinkage Mahalanobis distances.

    Each block is first standardized on the candidate history.  Its squared
    Mahalanobis value is divided by its feature count, then the two normalized
    squared values receive equal 50% weights.  Its Kth radius is later compared
    only with Kth radii recorded on strictly previous ready decision days.
    """

    history_trend, current_trend = _zscore(
        history[TREND_PARTICIPATION_FEATURES].to_numpy(float),
        current_state[[FEATURES.index(column) for column in TREND_PARTICIPATION_FEATURES]],
    )
    history_vol, current_vol = _zscore(
        history[VOL_TAIL_FEATURES].to_numpy(float),
        current_state[[FEATURES.index(column) for column in VOL_TAIL_FEATURES]],
    )
    precision_trend = LedoitWolf().fit(history_trend).precision_
    precision_vol = LedoitWolf().fit(history_vol).precision_

    trend_delta = history_trend - current_trend
    vol_delta = history_vol - current_vol
    trend_sq = np.einsum("ij,jk,ik->i", trend_delta, precision_trend, trend_delta) / history_trend.shape[1]
    vol_sq = np.einsum("ij,jk,ik->i", vol_delta, precision_vol, vol_delta) / history_vol.shape[1]
    distances = np.sqrt(np.maximum(0.5 * trend_sq + 0.5 * vol_sq, 0.0))

    nearest = np.argsort(distances, kind="stable")[:NEAREST_K]
    kth_distance = float(distances[nearest[-1]])
    return {
        "nearest": nearest,
        "nearest_distance": float(distances[nearest[0]]),
        "kth_distance": kth_distance,
    }


def run_gate(all_rows: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Replay the single frozen gate, strictly sequentially by score day."""

    route_eligible = aligned["base_reason"].eq("margin_default") & aligned["robust_reason"].isin(VALID_ROBUST_REASONS)
    records: list[dict] = []
    prior_ready_kth_radii: list[float] = []
    for index, row in aligned.iterrows():
        score_day = row["score_day"]
        base_name = row["base_name"]
        current_name = row["current_name"]
        eligible = bool(route_eligible.iloc[index])
        base_return = float(row[base_name])
        current_return = float(row["selected_return"])
        record: dict[str, object] = {
            "score_day": score_day,
            "route_eligible": eligible,
            "base_name": base_name,
            "base_model_id": NAME_TO_ID[base_name],
            "base_return": base_return,
            "base_reason": row["base_reason"],
            "current_name": current_name,
            "current_model_id": NAME_TO_ID[current_name],
            "current_return": current_return,
            "current_reason": row["reason"],
            "robust_reason_current": row["robust_reason"],
            "base_route_name": base_name if eligible else current_name,
            "base_route_return": base_return if eligible else current_return,
            "history_days": 0,
            "stat_ready": False,
            "distance_pass": False,
            "ess_pass": False,
            "effective_sample_size": float("nan"),
            "nearest_distance": float("nan"),
            "kth_distance": float("nan"),
            "prior_ready_radius_count": int(len(prior_ready_kth_radii)),
            "prior_ready_kth_radius_q75": float("nan"),
            "best_challenger": None,
            "best_lcb_trim20": float("nan"),
            "pairwise_statistics": "{}",
            "override_base": False,
            "abstain": False,
        }

        # Calculate and record a kth radius on every ready score day, including
        # non-eligible days.  The reference list is updated only after today's
        # gate decision, so a radius can never validate itself.
        current_state = row[FEATURES].to_numpy(float)
        history = complete_history(all_rows, score_day)
        record["history_days"] = int(len(history))
        ready = bool(np.isfinite(current_state).all() and len(history) >= MIN_PERIODS)
        record["stat_ready"] = ready
        distance: dict[str, object] | None = None
        if ready:
            if not bool((history["score_day"] < score_day).all()):
                raise RuntimeError(f"look-ahead history detected for {score_day}")
            distance = block_balanced_distance(history, current_state)
            record.update({key: value for key, value in distance.items() if key != "nearest"})
            if prior_ready_kth_radii:
                radius_q75 = float(np.quantile(prior_ready_kth_radii, DENSITY_QUANTILE))
                record["prior_ready_kth_radius_q75"] = radius_q75
                record["distance_pass"] = bool(float(distance["kth_distance"]) <= radius_q75)
            ess = float(NEAREST_K)  # exact, because the frozen weights are 1 / K.
            record["effective_sample_size"] = ess
            record["ess_pass"] = bool(ess >= MIN_ESS)

        if not eligible:
            chosen = current_name
            reason = "outside_existing_robust_intervention_branch"
        elif not ready:
            chosen = base_name
            reason = "abstain_insufficient_history_or_state"
            record["abstain"] = True
        elif not prior_ready_kth_radii:
            chosen = base_name
            reason = "abstain_no_prior_ready_radius_reference"
            record["abstain"] = True
        else:
            if distance is None:
                raise RuntimeError("ready decision is missing its distance bundle")
            pairwise: dict[str, dict[str, float]] = {}
            qualifying: list[tuple[float, str]] = []
            for challenger in MODELS:
                if challenger == base_name:
                    continue
                deltas = history.iloc[distance["nearest"]][challenger].to_numpy(float) - history.iloc[distance["nearest"]][base_name].to_numpy(float)
                estimates = robust_pairwise_lcb(deltas)
                estimates["lcb_pass"] = bool(estimates["lcb_trim20"] > LCB_ADVANTAGE_MIN)
                pairwise[challenger] = estimates
                if record["distance_pass"] and record["ess_pass"] and estimates["lcb_pass"]:
                    qualifying.append((float(estimates["lcb_trim20"]), challenger))
            record["pairwise_statistics"] = json.dumps(pairwise, ensure_ascii=False, sort_keys=True)
            if pairwise:
                best_lcb, best_challenger = max((float(item["lcb_trim20"]), name) for name, item in pairwise.items())
                record["best_challenger"] = best_challenger
                record["best_lcb_trim20"] = best_lcb
            if qualifying:
                _, chosen = max(qualifying)
                record["override_base"] = True
                reason = "stat_pairwise_lcb_override_base"
            else:
                chosen = base_name
                record["abstain"] = True
                if not record["distance_pass"]:
                    reason = "abstain_kth_radius_out_of_support"
                elif not record["ess_pass"]:
                    reason = "abstain_ess_below_min"
                else:
                    reason = "abstain_no_challenger_lcb_above_5bp"

        if ready and distance is not None:
            prior_ready_kth_radii.append(float(distance["kth_distance"]))

        record["selected_name"] = chosen
        record["selected_model_id"] = NAME_TO_ID[chosen]
        record["selected_return"] = float(row[chosen])
        record["gate_reason"] = reason
        record["changed_vs_current"] = bool(chosen != current_name)
        record["changed_vs_base_route"] = bool(chosen != record["base_route_name"])
        record["current_overrode_base"] = bool(eligible and current_name != base_name)
        record["return_delta_vs_current"] = float(record["selected_return"] - current_return)
        record["return_delta_vs_base_route"] = float(record["selected_return"] - float(record["base_route_return"]))
        record["current_delta_vs_base_route"] = float(current_return - float(record["base_route_return"]))
        records.append(record)
    return pd.DataFrame(records)


def contribution_rows(daily: pd.DataFrame) -> pd.DataFrame:
    """Split coverage and abstention effects against both current and Base."""

    conditions = {
        "all_eligible": daily["route_eligible"],
        "coverage_override_base": daily["route_eligible"] & daily["override_base"],
        "abstention_keep_base": daily["route_eligible"] & daily["abstain"],
        "coverage_changed_vs_current": daily["route_eligible"] & daily["override_base"] & daily["changed_vs_current"],
        "abstention_where_current_overrode_base": daily["route_eligible"] & daily["abstain"] & daily["current_overrode_base"],
    }
    comparisons = {
        "gate_minus_base_fallback": "return_delta_vs_base_route",
        "gate_minus_current": "return_delta_vs_current",
        "current_minus_base_fallback": "current_delta_vs_base_route",
    }
    rows: list[dict] = []
    for condition, mask in conditions.items():
        subset = daily.loc[mask]
        for comparison, column in comparisons.items():
            metrics = conditional_metrics(subset[column])
            metrics.update({"condition": condition, "comparison": comparison})
            rows.append(metrics)
    return pd.DataFrame(rows)


def slice_summary(frame: pd.DataFrame, slice_type: str, slice_name: str) -> dict:
    current = selector_metrics(frame["current_return"].to_numpy(float), frame["score_day"], frame["current_name"])
    fallback = selector_metrics(frame["base_route_return"].to_numpy(float), frame["score_day"], frame["base_route_name"])
    gate = selector_metrics(frame["selected_return"].to_numpy(float), frame["score_day"], frame["selected_name"])
    coverage = frame["route_eligible"] & frame["override_base"]
    abstention = frame["route_eligible"] & frame["abstain"]
    changed = frame["changed_vs_current"]
    return {
        "slice_type": slice_type,
        "slice": slice_name,
        "start_score_day": str(frame["score_day"].iloc[0]),
        "end_score_day": str(frame["score_day"].iloc[-1]),
        "days": int(len(frame)),
        "route_eligible_days": int(frame["route_eligible"].sum()),
        "coverage_override_base_days": int(coverage.sum()),
        "abstention_keep_base_days": int(abstention.sum()),
        "changed_days_vs_current": int(changed.sum()),
        "current_total_return": current["total_return"],
        "base_fallback_total_return": fallback["total_return"],
        "gate_total_return": gate["total_return"],
        "gate_total_delta_vs_current": float(gate["total_return"] - current["total_return"]),
        "gate_total_delta_vs_base_fallback": float(gate["total_return"] - fallback["total_return"]),
        "current_sharpe": current["sharpe"],
        "base_fallback_sharpe": fallback["sharpe"],
        "gate_sharpe": gate["sharpe"],
        "gate_sharpe_delta_vs_current": float(gate["sharpe"] - current["sharpe"]),
        "gate_sharpe_delta_vs_base_fallback": float(gate["sharpe"] - fallback["sharpe"]),
        "coverage_mean_gate_vs_base_bp": float(frame.loc[coverage, "return_delta_vs_base_route"].mean() * 1e4) if bool(coverage.any()) else float("nan"),
        "coverage_mean_gate_vs_current_bp": float(frame.loc[coverage, "return_delta_vs_current"].mean() * 1e4) if bool(coverage.any()) else float("nan"),
        "abstention_mean_gate_vs_current_bp": float(frame.loc[abstention, "return_delta_vs_current"].mean() * 1e4) if bool(abstention.any()) else float("nan"),
    }


def make_reports(daily: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = selector_metrics(daily["current_return"].to_numpy(float), daily["score_day"], daily["current_name"])
    fallback = selector_metrics(daily["base_route_return"].to_numpy(float), daily["score_day"], daily["base_route_name"])
    gate = selector_metrics(daily["selected_return"].to_numpy(float), daily["score_day"], daily["selected_name"])
    summary_rows = []
    for name, metrics in [("current_replay", current), ("base_fallback_on_eligible_branch", fallback), ("frozen_stat_pairwise_gate", gate)]:
        row = {"strategy": name, **metrics}
        row["total_return_delta_vs_current"] = float(metrics["total_return"] - current["total_return"])
        row["sharpe_delta_vs_current"] = float(metrics["sharpe"] - current["sharpe"])
        row["total_return_delta_vs_base_fallback"] = float(metrics["total_return"] - fallback["total_return"])
        row["sharpe_delta_vs_base_fallback"] = float(metrics["sharpe"] - fallback["sharpe"])
        row["route_eligible_days"] = int(daily["route_eligible"].sum())
        row["coverage_override_base_days"] = int((daily["route_eligible"] & daily["override_base"]).sum())
        row["abstention_keep_base_days"] = int((daily["route_eligible"] & daily["abstain"]).sum())
        row["changed_days_vs_current"] = int(daily["changed_vs_current"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    contributions = contribution_rows(daily)

    fold_labels = np.empty(len(daily), dtype=object)
    for number, indices in enumerate(np.array_split(np.arange(len(daily)), 3), start=1):
        fold_labels[indices] = f"wf{number}"
    daily["walk_forward_fold"] = fold_labels
    walk_forward = pd.DataFrame(
        [slice_summary(group, "walk_forward_sequential_test", fold) for fold, group in daily.groupby("walk_forward_fold", sort=True)]
    )

    midpoint = daily["score_day"].iloc[len(daily) // 2]
    daily["time_half"] = np.where(daily["score_day"] <= midpoint, "first_half", "second_half")
    time_segments = pd.DataFrame(
        [slice_summary(group, "chronological_half", name) for name, group in daily.groupby("time_half", sort=True)]
    )

    daily["month"] = pd.to_datetime(daily["score_day"]).dt.to_period("M").astype(str)
    monthly = pd.DataFrame(
        [slice_summary(group, "calendar_month", month) for month, group in daily.groupby("month", sort=True)]
    )

    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    contributions.to_csv(output / "coverage_abstention_contributions.csv", index=False, encoding="utf-8-sig")
    walk_forward.to_csv(output / "walk_forward_slices.csv", index=False, encoding="utf-8-sig")
    time_segments.to_csv(output / "time_segment_slices.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "monthly_slices.csv", index=False, encoding="utf-8-sig")
    return summary, contributions, walk_forward, time_segments


def write_markdown(
    output: Path,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    contributions: pd.DataFrame,
    walk_forward: pd.DataFrame,
    time_segments: pd.DataFrame,
    plot_path: str,
) -> None:
    gate = summary.loc[summary["strategy"].eq("frozen_stat_pairwise_gate")].iloc[0]
    coverage = contributions[(contributions["condition"] == "coverage_override_base") & (contributions["comparison"] == "gate_minus_base_fallback")].iloc[0]
    abstention = contributions[(contributions["condition"] == "abstention_keep_base") & (contributions["comparison"] == "gate_minus_current")].iloc[0]
    report = [
        "# Frozen statistical pairwise abstention-gate replay (2026-07-28)",
        "",
        "## Frozen contract",
        "",
        f"- Aligned score days: `{daily.score_day.min()}` to `{daily.score_day.max()}` ({len(daily)} days).",
        "- The current replay's Base, Champion-first, Champion-third, and all branches outside Base-low-confidence/valid-Robust are retained exactly.",
        f"- Eligible replacement branch: Base `margin_default` plus current numeric Robust diagnostic: `{int(daily['route_eligible'].sum())}` days.",
        f"- Historical candidates: last `{LOOKBACK}` complete prior score days only; minimum `{MIN_PERIODS}`; Top-`{NEAREST_K}` equal weight, ESS=`{NEAREST_K}` and required ESS >= `{MIN_ESS:.0f}`.",
        "- State metric: separate Ledoit-Wolf shrinkage Mahalanobis blocks, normalized by block dimension: trend/participation (18 features) and volatility/tail (26 features); normalized squared distances are 50% / 50%.",
        f"- Distance support: current K-th radius must be no larger than the `{DENSITY_QUANTILE:.0%}` quantile of K-th radii from strictly prior ready decision days; the current day never enters its own reference set.",
        f"- Pairwise evidence: for each challenger-minus-Base daily return difference, `trim20_mean - winsor10/90_std / sqrt(60)`; a challenger must have LCB strictly > `{LCB_ADVANTAGE_MIN * 1e4:.1f}bp`.",
        "- If no challenger satisfies every fixed condition, the gate abstains and keeps Base.  It does not consume Ridge predicted values.",
        "- Return construction is standalone model shadow-return composition; it does not transmit continuous positions or incremental switch cost.",
        "",
        "## Full-period result",
        "",
        *csv_block(summary[["strategy", "total_return", "sharpe", "max_drawdown", "switch_count", "total_return_delta_vs_current", "route_eligible_days", "coverage_override_base_days", "abstention_keep_base_days", "changed_days_vs_current"]]),
        "",
        "## Coverage and abstention contributions",
        "",
        f"- Coverage: `{int(coverage['n'])}` Base overrides; gate minus Base on these days averages `{coverage['mean_delta_bp']:.2f}bp` (wins/losses `{int(coverage['wins'])}/{int(coverage['losses'])}`, one-sided paired t p=`{coverage['paired_t_p_one_sided']:.4f}`).",
        f"- Abstention: `{int(abstention['n'])}` eligible days keep Base; Base minus current on these days averages `{abstention['mean_delta_bp']:.2f}bp`. This directly shows what is surrendered or protected by refusing the current Robust selection.",
        "",
        *csv_block(contributions[["condition", "comparison", "n", "wins", "losses", "ties", "mean_delta_bp", "sum_delta_bp", "paired_t_p_one_sided", "sign_p_one_sided", "wilcoxon_p_one_sided"]]),
        "",
        "## Sequential walk-forward and time stability",
        "",
        "- Every row is scored sequentially using only dates strictly earlier than its score day.  The three reported walk-forward test slices are chronological and use the same frozen rule; no fold is used to fit or select a parameter.",
        "",
        *csv_block(walk_forward[["slice", "start_score_day", "end_score_day", "days", "route_eligible_days", "coverage_override_base_days", "abstention_keep_base_days", "gate_total_delta_vs_current", "gate_total_delta_vs_base_fallback", "gate_sharpe_delta_vs_current"]]),
        "",
        *csv_block(time_segments[["slice", "start_score_day", "end_score_day", "days", "route_eligible_days", "coverage_override_base_days", "abstention_keep_base_days", "gate_total_delta_vs_current", "gate_total_delta_vs_base_fallback", "gate_sharpe_delta_vs_current"]]),
        "",
        "## Artifacts",
        "",
        "- `daily_stat_pairwise_gate.csv`: every score day, historical-only distance checks, pairwise LCBs, gate reason, and return deltas.",
        "- `summary_metrics.csv`, `coverage_abstention_contributions.csv`, `walk_forward_slices.csv`, `time_segment_slices.csv`, `monthly_slices.csv`.",
        "- `input_manifest.json` and `nav_compare.png` (or an explicitly recorded plot exception).",
        f"- Plot: `{plot_path}`.",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    output = Path(args.output_root) / f"run_20260728_{datetime.now().strftime('%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    all_rows, aligned = load_inputs()
    daily = run_gate(all_rows, aligned)
    summary, contributions, walk_forward, time_segments = make_reports(daily, output)
    daily.to_csv(output / "daily_stat_pairwise_gate.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "run_kind": "offline_read_only_statistical_pairwise_abstention_gate_replay",
        "baseline": "aligned current model-switch replay",
        "candidate": "frozen statistical challenger-vs-Base gate; Base fallback on abstention",
        "date_window": {"start": str(daily["score_day"].min()), "end": str(daily["score_day"].max()), "days": int(len(daily))},
        "branch_contract": "replace only Base margin_default + valid current Robust diagnostic branch; retain all other current selections",
        "selection_information_cutoff": "all candidate state and realized model-return rows satisfy score_day < decision score_day",
        "model_return_semantics": "standalone model shadow day_return composition; no continuous fused positions or incremental switching costs",
        "fixed_rule": {
            "lookback_complete_days": LOOKBACK,
            "min_periods": MIN_PERIODS,
            "nearest_k": NEAREST_K,
            "weighting": "equal",
            "effective_sample_size": NEAREST_K,
            "min_effective_sample_size": MIN_ESS,
            "distance": "0.5 * normalized trend_participation Mahalanobis squared + 0.5 * normalized vol_tail Mahalanobis squared, each LedoitWolf-shrunk",
            "trend_participation_feature_count": len(TREND_PARTICIPATION_FEATURES),
            "vol_tail_feature_count": len(VOL_TAIL_FEATURES),
            "distance_support": {"prior_ready_kth_radius_quantile": DENSITY_QUANTILE, "comparison": "current_kth_distance <= prior_ready_quantile", "reference_excludes_current_day": True},
            "pairwise_lcb": {"formula": "20pct trimmed mean - 10/90 winsorized sample std / sqrt(60)", "minimum_advantage": LCB_ADVANTAGE_MIN},
            "parameter_search": False,
        },
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in [STATE_PATH, CURRENT_PATH, *RETURN_PATHS.values()]],
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(12, 5.4))
        for name, returns, color, width in [
            ("current", daily["current_return"], "#202020", 2.4),
            ("Base fallback", daily["base_route_return"], "#9c755f", 1.5),
            ("stat pairwise gate", daily["selected_return"], "#4c78a8", 2.0),
        ]:
            axis.plot(pd.to_datetime(daily["score_day"]), np.cumprod(1.0 + returns.to_numpy(float)), label=name, color=color, linewidth=width)
        axis.set_title("Frozen statistical pairwise abstention gate (shadow-return replay)")
        axis.set_ylabel("NAV")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / "nav_compare.png", dpi=160)
        plt.close(fig)
        plot_path = "nav_compare.png"
    except Exception as exc:  # Plotting is evidence only, never calculation input.
        plot_path = f"plot_unavailable: {type(exc).__name__}: {exc}"

    write_markdown(output, daily, summary, contributions, walk_forward, time_segments, plot_path)
    gate = summary.loc[summary["strategy"].eq("frozen_stat_pairwise_gate")].iloc[0]
    print(f"OUTPUT_ROOT {output}")
    print(f"ROUTE_ELIGIBLE_DAYS {int(daily['route_eligible'].sum())}")
    print(f"COVERAGE_OVERRIDE_DAYS {int((daily['route_eligible'] & daily['override_base']).sum())}")
    print(f"ABSTENTION_DAYS {int((daily['route_eligible'] & daily['abstain']).sum())}")
    print(f"TOTAL_RETURN_DELTA_VS_CURRENT {gate['total_return_delta_vs_current']:.8f}")
    print(f"TOTAL_RETURN_DELTA_VS_BASE_FALLBACK {gate['total_return_delta_vs_base_fallback']:.8f}")


if __name__ == "__main__":
    main()
