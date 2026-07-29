"""Read-only replay of a long-window KNN replacement for Robust Ridge.

The script preserves the aligned current Base and Champion routing from the
current-selector replay. It replaces only the long-history Robust decision on
Base-low-confidence dates where current Robust has a numeric diagnostic.
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
LOOKBACK = 360
MIN_PERIODS = 120
MARGIN = 0.0005
KS = [20, 40, 60]
PRIMARY_VARIANT = "mahal_kernel_k40"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trim_mean(values: np.ndarray, proportion: float = 0.20) -> float:
    values = np.sort(np.asarray(values, float)[np.isfinite(values)])
    cut = int(math.floor(len(values) * proportion))
    if not len(values):
        return float("nan")
    return float(values[cut : len(values) - cut].mean()) if 2 * cut < len(values) else float(values.mean())


def trim20_lcb10(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    center = trim_mean(values)
    lo, hi = np.quantile(values, [0.10, 0.90])
    clipped = np.clip(values, lo, hi)
    std = float(np.std(clipped, ddof=1)) if len(clipped) > 1 else 0.0
    return float(center - std / math.sqrt(len(clipped)))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = np.asarray(values, float)[order]
    weights = np.asarray(weights, float)[order]
    weights = weights / weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), quantile, side="left")])


def weighted_trim_mean(values: np.ndarray, weights: np.ndarray, proportion: float = 0.20) -> float:
    order = np.argsort(values)
    values = np.asarray(values, float)[order]
    remaining = np.asarray(weights, float)[order].copy()
    remaining /= remaining.sum()
    for direction in (range(len(remaining)), range(len(remaining) - 1, -1, -1)):
        to_trim = proportion
        for index in direction:
            take = min(float(remaining[index]), to_trim)
            remaining[index] -= take
            to_trim -= take
            if to_trim <= 1e-15:
                break
    mass = float(remaining.sum())
    return float(np.dot(values, remaining) / mass) if mass > 1e-15 else float(np.dot(values, weights))


def weighted_lcb(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    center = weighted_trim_mean(values, weights)
    lo, hi = weighted_quantile(values, weights, 0.10), weighted_quantile(values, weights, 0.90)
    clipped = np.clip(values, lo, hi)
    clipped_mean = float(np.dot(weights, clipped))
    denom = max(1.0 - float(np.dot(weights, weights)), 1e-12)
    variance = float(np.dot(weights, (clipped - clipped_mean) ** 2) / denom)
    ess = float(1.0 / np.dot(weights, weights))
    return float(center - math.sqrt(max(variance, 0.0)) / math.sqrt(ess)), ess, float(weights.max())


def selector_metrics(returns: np.ndarray, dates: pd.Series, selected: pd.Series) -> dict:
    values = np.asarray(returns, float)
    nav = np.cumprod(1.0 + values)
    total = float(nav[-1] - 1.0)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0)) if volatility > 0 else float("nan")
    switches = int((selected.astype(str).to_numpy()[1:] != selected.astype(str).to_numpy()[:-1]).sum()) if len(selected) > 1 else 0
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


def conditional_metrics(delta: pd.Series) -> dict:
    values = delta.dropna().to_numpy(float)
    if not len(values):
        return {"n": 0}
    wins = int((values > 0).sum())
    losses = int((values < 0).sum())
    result = {
        "n": int(len(values)),
        "wins": wins,
        "losses": losses,
        "mean_delta_bp": float(values.mean() * 1e4),
        "median_delta_bp": float(np.median(values) * 1e4),
        "sum_delta_bp": float(values.sum() * 1e4),
        "paired_t_p_two_sided": float(stats.ttest_1samp(values, 0).pvalue) if len(values) > 1 else float("nan"),
        "paired_t_p_one_sided": float(stats.ttest_1samp(values, 0, alternative="greater").pvalue) if len(values) > 1 else float("nan"),
        "sign_p_two_sided": float(stats.binomtest(wins, len(values), 0.5, alternative="two-sided").pvalue),
        "sign_p_one_sided": float(stats.binomtest(wins, len(values), 0.5, alternative="greater").pvalue),
    }
    try:
        result["wilcoxon_p_one_sided"] = float(stats.wilcoxon(values, alternative="greater", method="auto").pvalue)
    except ValueError:
        result["wilcoxon_p_one_sided"] = float("nan")
    return result


def csv_block(frame: pd.DataFrame) -> list[str]:
    """Render a compact table without the optional pandas tabulate dependency."""
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    returns: pd.DataFrame | None = None
    for name, path in RETURN_PATHS.items():
        frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[name] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", name]].dropna().drop_duplicates("score_day", keep="last")
        returns = frame if returns is None else returns.merge(frame, on="score_day", how="inner")
    state = pd.read_csv(STATE_PATH, usecols=["trade_date", *FEATURES])
    state["score_day"] = pd.to_datetime(state["trade_date"], errors="coerce").dt.date
    state = state[["score_day", *FEATURES]].dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    for column in FEATURES:
        state[column] = pd.to_numeric(state[column], errors="coerce")
    all_rows = returns.merge(state, on="score_day", how="left").sort_values("score_day").reset_index(drop=True)

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


def run_variant(
    all_rows: pd.DataFrame,
    aligned: pd.DataFrame,
    route_eligible: pd.Series,
    *,
    variant: str,
    distance_method: str,
    weighting: str,
    nearest_k: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for index, row in aligned.iterrows():
        score_day = row["score_day"]
        base_name = row["base_name"]
        current_name = row["current_name"]
        eligible = bool(route_eligible.iloc[index])
        record = {
            "score_day": score_day,
            "variant": variant,
            "distance_method": distance_method,
            "weighting": weighting,
            "nearest_k": nearest_k,
            "base_name": base_name,
            "base_model_id": NAME_TO_ID[base_name],
            "current_name": current_name,
            "current_model_id": NAME_TO_ID[current_name],
            "current_return": float(row["selected_return"]),
            "route_eligible": eligible,
            "base_reason": row["base_reason"],
            "current_reason": row["reason"],
            "robust_reason_current": row["robust_reason"],
        }
        state_now = row[FEATURES].to_numpy(float)
        history = all_rows.loc[all_rows["score_day"] < score_day].tail(LOOKBACK).copy()
        complete_history = np.isfinite(history[FEATURES].to_numpy(float)).all(axis=1) & np.isfinite(history[MODELS].to_numpy(float)).all(axis=1)
        history = history.loc[complete_history].copy()
        ready = bool(np.isfinite(state_now).all() and len(history) >= MIN_PERIODS)
        record["history_days"] = int(len(history))
        record["knn_ready"] = ready
        if not eligible:
            chosen = current_name
            record.update({
                "selected_name": chosen,
                "selected_model_id": NAME_TO_ID[chosen],
                "selected_return": float(row[chosen]),
                "knn_reason": "outside_existing_robust_intervention_branch",
                "knn_confident": False,
                "score_gap": np.nan,
                "effective_sample_size": np.nan,
                "max_weight": np.nan,
                "nearest_distance": np.nan,
                "kth_distance": np.nan,
                "candidate_scores": "{}",
                "override_base": False,
            })
        elif not ready:
            chosen = base_name
            record.update({
                "selected_name": chosen,
                "selected_model_id": NAME_TO_ID[chosen],
                "selected_return": float(row[chosen]),
                "knn_reason": "insufficient_history_or_feature",
                "knn_confident": False,
                "score_gap": np.nan,
                "effective_sample_size": np.nan,
                "max_weight": np.nan,
                "nearest_distance": np.nan,
                "kth_distance": np.nan,
                "candidate_scores": "{}",
                "override_base": False,
            })
        else:
            features = history[FEATURES].to_numpy(float)
            mean = features.mean(axis=0)
            std = features.std(axis=0)
            std = np.where((~np.isfinite(std)) | (std < 1e-12), 1.0, std)
            history_z = (features - mean) / std
            current_z = (state_now - mean) / std
            delta = history_z - current_z
            if distance_method == "diag":
                distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            elif distance_method == "mahal":
                precision = LedoitWolf().fit(history_z).precision_
                squared = np.einsum("ij,jk,ik->i", delta, precision, delta)
                distances = np.sqrt(np.maximum(squared, 0.0))
            else:
                raise ValueError(f"unsupported distance method {distance_method}")
            nearest = np.argsort(distances)[:nearest_k]
            kth_distance = float(distances[nearest[-1]])
            if weighting == "hard":
                weights = np.full(nearest_k, 1.0 / nearest_k)
            elif weighting == "kernel":
                bandwidth = max(kth_distance, 1e-12)
                weights = np.exp(-0.5 * (distances[nearest] / bandwidth) ** 2)
                weights /= weights.sum()
            else:
                raise ValueError(f"unsupported weighting {weighting}")
            scores: dict[str, float] = {}
            for model in MODELS:
                values = history.iloc[nearest][model].to_numpy(float)
                scores[model] = trim20_lcb10(values) if weighting == "hard" else weighted_lcb(values, weights)[0]
            ranking = sorted(MODELS, key=lambda model: scores[model], reverse=True)
            gap = float(scores[ranking[0]] - scores[ranking[1]])
            confident = bool(gap > MARGIN)
            override = bool(confident and ranking[0] != base_name)
            chosen = ranking[0] if override else base_name
            record.update({
                "selected_name": chosen,
                "selected_model_id": NAME_TO_ID[chosen],
                "selected_return": float(row[chosen]),
                "knn_reason": "knn_override_base" if override else ("knn_agrees_base" if confident else "knn_not_confident_keep_base"),
                "knn_confident": confident,
                "score_gap": gap,
                "effective_sample_size": float(1.0 / np.dot(weights, weights)),
                "max_weight": float(weights.max()),
                "nearest_distance": float(distances[nearest[0]]),
                "kth_distance": kth_distance,
                "candidate_scores": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                "override_base": override,
            })
        record["return_delta_vs_current"] = float(record["selected_return"] - record["current_return"])
        record["return_delta_vs_base"] = float(record["selected_return"] - float(row[base_name])) if eligible else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(ANALYSIS_ROOT / "model_switch_similarity_robust_20260728"))
    args = parser.parse_args()
    output = Path(args.output_root) / f"run_20260728_{datetime.now().strftime('%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)

    all_rows, aligned = load_inputs()
    valid_robust = aligned["robust_reason"].isin(["score_best", "margin_default"])
    route_eligible = aligned["base_reason"].eq("margin_default") & valid_robust
    variants = [(f"diag_hard_k{k}", "diag", "hard", k) for k in KS]
    variants += [(f"mahal_hard_k{k}", "mahal", "hard", k) for k in KS]
    variants += [(f"mahal_kernel_k{k}", "mahal", "kernel", k) for k in KS]

    baseline = selector_metrics(aligned["selected_return"].to_numpy(float), aligned["score_day"], aligned["current_name"])
    daily_by_variant: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []
    condition_rows: list[dict] = []
    monthly_rows: list[dict] = []
    for variant, method, weighting, nearest_k in variants:
        daily = run_variant(all_rows, aligned, route_eligible, variant=variant, distance_method=method, weighting=weighting, nearest_k=nearest_k)
        daily.to_csv(output / f"daily_{variant}.csv", index=False, encoding="utf-8-sig")
        daily_by_variant[variant] = daily
        metrics = selector_metrics(daily["selected_return"].to_numpy(float), daily["score_day"], daily["selected_name"])
        changed_current = daily["selected_name"].ne(daily["current_name"])
        overrides = daily["route_eligible"] & daily["override_base"]
        metrics.update({
            "variant": variant,
            "route_eligible_days": int(daily["route_eligible"].sum()),
            "changed_days_vs_current": int(changed_current.sum()),
            "override_base_days": int(overrides.sum()),
            "total_return_delta_vs_current": float(metrics["total_return"] - baseline["total_return"]),
            "sharpe_delta_vs_current": float(metrics["sharpe"] - baseline["sharpe"]),
        })
        summary_rows.append(metrics)
        for condition, subset, delta_column in [
            ("override_vs_base", daily.loc[overrides], "return_delta_vs_base"),
            ("all_eligible_vs_current", daily.loc[daily["route_eligible"]], "return_delta_vs_current"),
        ]:
            result = conditional_metrics(subset[delta_column])
            result.update({"variant": variant, "condition": condition, "route_eligible_days": int(daily["route_eligible"].sum()), "override_base_days": int(overrides.sum())})
            condition_rows.append(result)
        split_date = daily["score_day"].iloc[len(daily) // 2]
        for slice_name, subset in [("first_half", daily.loc[overrides & (daily["score_day"] <= split_date)]), ("second_half", daily.loc[overrides & (daily["score_day"] > split_date)])]:
            result = conditional_metrics(subset["return_delta_vs_base"])
            result.update({"variant": variant, "condition": slice_name, "route_eligible_days": int(daily["route_eligible"].sum()), "override_base_days": int(len(subset))})
            condition_rows.append(result)
        monthly = daily.copy()
        monthly["month"] = pd.to_datetime(monthly["score_day"]).dt.to_period("M").astype(str)
        for month, group in monthly.groupby("month"):
            changed = group["selected_name"].ne(group["current_name"])
            monthly_rows.append({
                "variant": variant,
                "month": month,
                "days": int(len(group)),
                "route_eligible_days": int(group["route_eligible"].sum()),
                "override_base_days": int((group["route_eligible"] & group["override_base"]).sum()),
                "changed_days_vs_current": int(changed.sum()),
                "current_return": float(np.prod(1.0 + group["current_return"].to_numpy(float)) - 1.0),
                "variant_return": float(np.prod(1.0 + group["selected_return"].to_numpy(float)) - 1.0),
                "return_delta_vs_current": float((np.prod(1.0 + group["selected_return"].to_numpy(float)) - 1.0) - (np.prod(1.0 + group["current_return"].to_numpy(float)) - 1.0)),
            })

    summary = pd.DataFrame(summary_rows).sort_values("total_return_delta_vs_current", ascending=False)
    conditions = pd.DataFrame(condition_rows)
    monthly = pd.DataFrame(monthly_rows)
    summary.to_csv(output / "similarity_robust_summary.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output / "similarity_robust_conditional_metrics.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "similarity_robust_monthly.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "run_kind": "offline_read_only_similarity_robust_replacement",
        "baseline": "aligned current model switch replay; retain current Base and Champion routing",
        "date_window": {"start": str(aligned["score_day"].min()), "end": str(aligned["score_day"].max()), "days": int(len(aligned))},
        "model_return_semantics": "standalone model shadow day_return composition; no continuous fused position or incremental switching cost replay",
        "branch_contract": "replace only Base margin_default dates with a valid current Robust diagnostic; preserve all other current selections",
        "route_eligible_days": int(route_eligible.sum()),
        "knn_config": {"lookback_days": LOOKBACK, "min_periods": MIN_PERIODS, "feature_set": "path_full_t1430", "feature_count": len(FEATURES), "margin": MARGIN, "primary_variant": PRIMARY_VARIANT, "variants": [item[0] for item in variants]},
        "distance_methods": {"diag": "candidate-window z-scored Euclidean", "mahal": "candidate-window z-score plus LedoitWolf shrinkage Mahalanobis"},
        "weighting": {"hard": "equal TopK weights and exact current trim20_lcb10", "kernel": "Gaussian exp(-0.5*(d/d_K)^2) and weighted trim20 LCB"},
        "database_writes": False,
        "live_runtime_called": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in [STATE_PATH, CURRENT_PATH, *RETURN_PATHS.values()]],
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(12, 5.4))
        for name, color, width in [("current", "#202020", 2.4), ("diag_hard_k40", "#4c78a8", 1.4), ("mahal_hard_k40", "#f58518", 1.4), (PRIMARY_VARIANT, "#54a24b", 2.0)]:
            returns = aligned["selected_return"].to_numpy(float) if name == "current" else daily_by_variant[name]["selected_return"].to_numpy(float)
            axis.plot(pd.to_datetime(aligned["score_day"]), np.cumprod(1.0 + returns), label=name, color=color, linewidth=width)
        axis.set_title("Similarity-Robust replacement replay (shadow-return composition)")
        axis.set_ylabel("NAV")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / "nav_compare.png", dpi=160)
        plt.close(fig)
        plot_path = "nav_compare.png"
    except Exception as exc:  # plot is optional evidence, not calculation input
        plot_path = f"plot_unavailable: {type(exc).__name__}: {exc}"

    primary = summary.loc[summary["variant"].eq(PRIMARY_VARIANT)].iloc[0]
    primary_condition = conditions[(conditions["variant"] == PRIMARY_VARIANT) & (conditions["condition"] == "override_vs_base")].iloc[0]
    report = [
        "# Similarity-Robust replacement experiment (2026-07-28)",
        "",
        "## Contract",
        "",
        f"- Aligned score days: `{aligned.score_day.min()}` to `{aligned.score_day.max()}` ({len(aligned)} days).",
        "- Base, Champion-first, Champion-third, and low-confidence routing remain unchanged.",
        "- Only the 360-day Robust diagnostic is replaced offline by a state-similarity Top-K historical-return statistic.",
        f"- Current Base-low-confidence / valid-Robust branch: `{int(route_eligible.sum())}` days.",
        "- Returns are standalone shadow return composition, not continuous fused holdings or incremental model-switch costs.",
        "",
        "## Baseline",
        "",
        f"- Total return `{baseline['total_return']:.2%}`, Sharpe `{baseline['sharpe']:.3f}`, max drawdown `{baseline['max_drawdown']:.2%}`, switches `{baseline['switch_count']}`.",
        "",
        "## Pre-specified primary variant",
        "",
        f"- `{PRIMARY_VARIANT}`: 360-day lookback, K=40, Ledoit-Wolf shrinkage Mahalanobis distance, Gaussian distance weights, weighted trim20 LCB.",
        f"- Full delta versus current: `{primary['total_return_delta_vs_current']:.2%}`; changed days `{int(primary['changed_days_vs_current'])}`; Base overrides `{int(primary['override_base_days'])}`.",
        f"- Override-only delta versus Base: n=`{int(primary_condition['n'])}`, mean=`{primary_condition['mean_delta_bp']:.2f}bp`, wins/losses=`{int(primary_condition['wins'])}/{int(primary_condition['losses'])}`, one-sided t p=`{primary_condition['paired_t_p_one_sided']:.4f}`, sign p=`{primary_condition['sign_p_one_sided']:.4f}`.",
        "",
        "## Variant summary",
        "",
        *csv_block(summary[["variant", "total_return", "sharpe", "max_drawdown", "switch_count", "changed_days_vs_current", "override_base_days", "total_return_delta_vs_current"]]),
        "",
        "## Condition metrics: actual KNN override versus Base",
        "",
        *csv_block(conditions.loc[conditions["condition"].eq("override_vs_base"), ["variant", "n", "wins", "losses", "mean_delta_bp", "median_delta_bp", "paired_t_p_one_sided", "sign_p_one_sided", "wilcoxon_p_one_sided"]]),
        "",
        "## Artifacts",
        "",
        "- `similarity_robust_summary.csv`",
        "- `similarity_robust_conditional_metrics.csv`",
        "- `similarity_robust_monthly.csv`",
        "- `daily_<variant>.csv`",
        f"- `{plot_path}`",
        "- `input_manifest.json`",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"OUTPUT_ROOT {output}")
    print(f"ROUTE_ELIGIBLE_DAYS {int(route_eligible.sum())}")
    print(summary[["variant", "total_return_delta_vs_current", "sharpe_delta_vs_current", "changed_days_vs_current", "override_base_days"]].to_string(index=False))


if __name__ == "__main__":
    main()
