"""Read-only walk-forward replay for Robust Ridge input reconstruction.

This tool deliberately does not import the live selector or alter live state.
It uses the exact artefacts used by the existing selector replay, retains its
Base/Champion routing, and only substitutes the long-window Robust diagnostic
on the already eligible low-Base-confidence branch.

Variants are pre-specified, with identical model-target/routing parameters:

* path44_standard_ridge: current 44 path-full T1430 features and rolling
  mean/std z-scaling.  It is also a reproduction check for the existing Ridge.
* disp7_standard_ridge: the original seven afternoon-dispersion features.
* path44_robust_pca15_whiten_ridge: rolling median/IQR scaling, PCA capped at
  15 components, then PCA whitening before the same Ridge regressions.
* path44_pca15_disagreement_ridge: the same PCA input plus fixed causal
  same-day cross-model score-disagreement features.

All history is strictly before the score day.  Outputs are shadow-return
compositions, not a continuous-position or incremental-switching-cost replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


ANALYSIS_ROOT = Path(r"D:\cbond_on\results\analysis")
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_ridge_rebuild_20260728")
STATE_PATH = ANALYSIS_ROOT / "model_switch_t1430_path_full_20260721" / "t1430_market_state_features_path_full.csv"
CURRENT_PATH = (
    ANALYSIS_ROOT
    / "model_switch_robust_strict_veto_20260728"
    / "run_20260728_152318"
    / "daily_current.csv"
)
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
PAIRWISE = [("Regsim", "Ensemble"), ("Regsim", "HL20"), ("Ensemble", "HL20")]
SCORE_ROOTS = {
    "Regsim": Path(r"D:\cbond_on\results\scores\live\lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708"),
    "Ensemble": Path(r"D:\cbond_on\results\scores\live\ensemble_rankavg_baseline_hl20_labeltop20_20260626"),
    "HL20": Path(r"D:\cbond_on\results\scores\live\lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625"),
}

PATH_PREFIXES = [
    "full0935_1430",
    "seg0935_1000",
    "seg1000_1030",
    "seg1030_1100",
    "seg1100_1130",
    "seg1300_1330",
    "seg1330_1400",
    "seg1400_1430",
]
PATH44_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in PATH_PREFIXES
    for suffix in ["mean", "std", "iqr", "pos_ratio", "tail_spread"]
] + ["trend_accel_median", "trend_accel_mean", "dispersion_accel", "tail_balance_full"]
DISP7_FEATURES = [
    "afternoon1300_1430_std",
    "afternoon1300_1430_iqr",
    "afternoon1300_1430_tail_spread",
    "last30_1330_1430_std",
    "last30_1330_1430_iqr",
    "last30_1330_1430_tail_spread",
    "dispersion_accel",
]
DISAGREEMENT_FEATURES = [
    *[f"disagree_spearman_{left}_{right}" for left, right in PAIRWISE],
    *[f"disagree_top20_jaccard_{left}_{right}" for left, right in PAIRWISE],
    *[f"disagree_score_std_{model}" for model in MODELS],
    *[f"disagree_top20_prev_jaccard_{model}" for model in MODELS],
]

LOOKBACK_DAYS = 360
MIN_PERIODS = 120
ALPHA = 100.0
TARGET_CLIP = 0.0075
MARGIN = 0.0005
PCA_COMPONENTS = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(left: str, right: str) -> str:
    return f"{left}_minus_{right}"


def selector_metrics(returns: pd.Series, dates: pd.Series, selected: pd.Series) -> dict[str, float | int | str]:
    values = returns.to_numpy(dtype=float)
    nav = np.cumprod(1.0 + values)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0)) if volatility > 0.0 else float("nan")
    switches = int((selected.astype(str).to_numpy()[1:] != selected.astype(str).to_numpy()[:-1]).sum()) if len(selected) > 1 else 0
    total_return = float(nav[-1] - 1.0)
    return {
        "days": int(len(values)),
        "start_score_day": str(dates.iloc[0]),
        "end_score_day": str(dates.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252.0 / len(values)) - 1.0),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "switch_count": switches,
    }


def conditional_metrics(delta: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"n": 0}
    wins = int((values > 0.0).sum())
    losses = int((values < 0.0).sum())
    result: dict[str, float | int] = {
        "n": int(len(values)),
        "wins": wins,
        "losses": losses,
        "mean_delta_bp": float(values.mean() * 1e4),
        "median_delta_bp": float(np.median(values) * 1e4),
        "sum_delta_bp": float(values.sum() * 1e4),
        "paired_t_p_two_sided": float(stats.ttest_1samp(values, 0.0).pvalue) if len(values) > 1 else float("nan"),
        "paired_t_p_one_sided": float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue) if len(values) > 1 else float("nan"),
        "sign_p_two_sided": float(stats.binomtest(wins, len(values), 0.5, alternative="two-sided").pvalue),
        "sign_p_one_sided": float(stats.binomtest(wins, len(values), 0.5, alternative="greater").pvalue),
    }
    try:
        result["wilcoxon_p_one_sided"] = float(stats.wilcoxon(values, alternative="greater", method="auto").pvalue)
    except ValueError:
        result["wilcoxon_p_one_sided"] = float("nan")
    return result


def csv_block(frame: pd.DataFrame) -> list[str]:
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def _score_file_path(model: str, score_day: object) -> Path:
    timestamp = pd.Timestamp(score_day)
    return SCORE_ROOTS[model] / timestamp.strftime("%Y-%m") / f"{timestamp:%Y-%m-%d}.csv"


def _load_score_series(model: str, score_day: object) -> pd.Series:
    path = _score_file_path(model, score_day)
    if not path.exists():
        raise FileNotFoundError(f"score archive missing for {model} {score_day}: {path}")
    frame = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    reported_days = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    expected_day = pd.Timestamp(score_day).date()
    if reported_days.isna().any() or not reported_days.eq(expected_day).all():
        bad_days = sorted({str(day) for day in reported_days.dropna().unique()})
        raise ValueError(f"score archive day mismatch for {model} {score_day}: path={path}; reported={bad_days}")
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["code", "score"])
    frame = frame.loc[frame["code"].ne("")]
    series = frame.groupby("code", sort=True)["score"].max()
    if len(series) < 20:
        raise ValueError(f"score archive has fewer than 20 valid codes for {model} {score_day}: {path}")
    return series.astype(float)


def _top20_codes(scores: pd.Series) -> set[str]:
    return set(scores.sort_values(ascending=False, kind="stable").head(20).index.astype(str))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else float("nan")


def build_score_disagreement_features(score_days: pd.Series) -> tuple[pd.DataFrame, dict[str, object]]:
    """Construct fixed, point-in-time model-disagreement features.

    For each score day the only score input is that day's three archived score
    CSVs.  The per-model previous-Top20 overlap uses the previous *aligned
    score day* archive, so it also contains no future information.
    """

    dates = [pd.Timestamp(day).date() for day in score_days.tolist()]
    if len(dates) != len(set(dates)):
        raise ValueError("score-disagreement input contains duplicate score days")
    records: list[dict[str, object]] = []
    previous_top20: dict[str, set[str]] | None = None
    files_seen: dict[str, int] = {model: 0 for model in MODELS}
    for score_day in dates:
        score_by_model = {model: _load_score_series(model, score_day) for model in MODELS}
        files_seen = {model: files_seen[model] + 1 for model in MODELS}
        top20_by_model = {model: _top20_codes(scores) for model, scores in score_by_model.items()}
        record: dict[str, object] = {"score_day": score_day}
        for left, right in PAIRWISE:
            paired = pd.concat([score_by_model[left], score_by_model[right]], axis=1, join="inner").dropna()
            if len(paired) < 20:
                raise ValueError(f"insufficient common score universe for {left}/{right} on {score_day}: {len(paired)}")
            corr = float(stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1]).statistic)
            record[f"disagree_spearman_{left}_{right}"] = corr if math.isfinite(corr) else 0.0
            record[f"disagree_top20_jaccard_{left}_{right}"] = _jaccard(top20_by_model[left], top20_by_model[right])
        for model in MODELS:
            record[f"disagree_score_std_{model}"] = float(np.std(score_by_model[model].to_numpy(dtype=float), ddof=0))
            record[f"disagree_top20_prev_jaccard_{model}"] = (
                np.nan if previous_top20 is None else _jaccard(top20_by_model[model], previous_top20[model])
            )
        records.append(record)
        previous_top20 = top20_by_model
    features = pd.DataFrame(records)
    post_first = features.iloc[1:]
    if post_first[DISAGREEMENT_FEATURES].isna().any().any():
        missing = post_first.loc[post_first[DISAGREEMENT_FEATURES].isna().any(axis=1), "score_day"].tolist()
        raise ValueError(f"unexpected score-disagreement feature NaN after first day: {missing[:10]}")
    meta: dict[str, object] = {
        "status": "available",
        "feature_count": len(DISAGREEMENT_FEATURES),
        "days_requested": len(dates),
        "days_complete": int(features[DISAGREEMENT_FEATURES].iloc[1:].notna().all(axis=1).sum()),
        "first_day_missing_previous_top20": str(dates[0]) if dates else None,
        "score_roots": {model: str(root) for model, root in SCORE_ROOTS.items()},
        "score_files_read_per_model": files_seen,
        "same_day_features": "three pairwise score Spearman correlations, three Top20 Jaccards, and each model score cross-sectional std",
        "prior_day_features": "each model Top20 Jaccard versus previous aligned score day; turnover proxy is 1 - Jaccard",
    }
    return features, meta


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    returns: pd.DataFrame | None = None
    for name, path in RETURN_PATHS.items():
        frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[name] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", name]].dropna().drop_duplicates("score_day", keep="last")
        returns = frame if returns is None else returns.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if returns is None:
        raise RuntimeError("no shadow return inputs loaded")

    required_state_cols = [*PATH44_FEATURES, *[column for column in DISP7_FEATURES if column not in PATH44_FEATURES]]
    state = pd.read_csv(STATE_PATH, usecols=["trade_date", *required_state_cols])
    state["score_day"] = pd.to_datetime(state["trade_date"], errors="coerce").dt.date
    state = state[["score_day", *required_state_cols]].dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    for column in required_state_cols:
        state[column] = pd.to_numeric(state[column], errors="coerce")
    all_rows = returns.merge(state, on="score_day", how="left", validate="one_to_one").sort_values("score_day").reset_index(drop=True)

    current = pd.read_csv(CURRENT_PATH)
    current["score_day"] = pd.to_datetime(current["score_day"], errors="coerce").dt.date
    current = current.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    current["current_name"] = current["selected_model_id"].map(ID_TO_NAME)
    current["base_name"] = current["base_model_id"].map(ID_TO_NAME)
    current["current_robust_name"] = current["robust_model_id"].map(ID_TO_NAME)
    if current[["current_name", "base_name"]].isna().any().any():
        unknown = current.loc[current[["current_name", "base_name"]].isna().any(axis=1), ["score_day", "selected_model_id", "base_model_id"]]
        raise RuntimeError(f"unmapped model identifier in current selector replay:\n{unknown.to_string(index=False)}")
    disagreement, disagreement_meta = build_score_disagreement_features(all_rows["score_day"])
    all_rows = all_rows.merge(disagreement, on="score_day", how="left", validate="one_to_one")
    aligned = current.merge(all_rows, on="score_day", how="inner", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    for row in aligned.itertuples(index=False):
        expected = float(getattr(row, row.current_name))
        if not math.isclose(expected, float(row.selected_return), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"current return mismatch on {row.score_day}: expected={expected}, replay={row.selected_return}")
    return all_rows, aligned, disagreement_meta


def standardize_mean_std(train_x: np.ndarray, current_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0, ddof=0)
    stds = np.where((~np.isfinite(stds)) | (stds < 1e-12), 1.0, stds)
    return (train_x - means) / stds, (current_x - means) / stds, {"transform": "rolling_mean_std_zscore"}


def robust_pca_whiten(train_x: np.ndarray, current_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    medians = np.median(train_x, axis=0)
    q25, q75 = np.quantile(train_x, [0.25, 0.75], axis=0)
    iqrs = q75 - q25
    iqrs = np.where((~np.isfinite(iqrs)) | (iqrs < 1e-12), 1.0, iqrs)
    scaled_train = (train_x - medians) / iqrs
    scaled_current = (current_x - medians) / iqrs
    components = min(PCA_COMPONENTS, scaled_train.shape[0], scaled_train.shape[1])
    if components < 1:
        raise ValueError("PCA requires at least one component")
    estimator = PCA(n_components=components, whiten=True, svd_solver="full")
    transformed_train = estimator.fit_transform(scaled_train)
    transformed_current = estimator.transform(scaled_current)
    return transformed_train, transformed_current, {
        "transform": "rolling_median_iqr_pca_whiten",
        "pca_components": int(components),
        "pca_explained_variance_ratio_sum": float(np.sum(estimator.explained_variance_ratio_)),
    }


Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]]


VARIANTS: tuple[dict[str, object], ...] = (
    {
        "variant": "path44_standard_ridge",
        "feature_cols": PATH44_FEATURES,
        "transform": standardize_mean_std,
        "description": "current 44 path-full features with rolling mean/std z-score",
    },
    {
        "variant": "disp7_standard_ridge",
        "feature_cols": DISP7_FEATURES,
        "transform": standardize_mean_std,
        "description": "seven afternoon-dispersion features with rolling mean/std z-score",
    },
    {
        "variant": "path44_robust_pca15_whiten_ridge",
        "feature_cols": PATH44_FEATURES,
        "transform": robust_pca_whiten,
        "description": "44 path-full features with rolling median/IQR scale, PCA<=15, whiten=True",
    },
    {
        "variant": "path44_pca15_disagreement_ridge",
        "feature_cols": [*PATH44_FEATURES, *DISAGREEMENT_FEATURES],
        "transform": robust_pca_whiten,
        "description": "44 path-full plus 12 fixed causal score-disagreement features; rolling median/IQR scale, PCA<=15, whiten=True",
    },
)


def run_variant(
    all_rows: pd.DataFrame,
    aligned: pd.DataFrame,
    route_eligible: pd.Series,
    *,
    variant: str,
    feature_cols: list[str],
    transform: Transform,
    description: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in aligned.iterrows():
        score_day = row["score_day"]
        base_name = str(row["base_name"])
        current_name = str(row["current_name"])
        eligible = bool(route_eligible.iloc[index])
        current_features = row[feature_cols].to_numpy(dtype=float).reshape(1, -1)
        history = all_rows.loc[all_rows["score_day"] < score_day].tail(LOOKBACK_DAYS).copy()
        valid_history = np.isfinite(history[feature_cols].to_numpy(dtype=float)).all(axis=1) & np.isfinite(history[MODELS].to_numpy(dtype=float)).all(axis=1)
        history = history.loc[valid_history].copy()
        ready = bool(np.isfinite(current_features).all() and len(history) >= MIN_PERIODS)
        record: dict[str, object] = {
            "score_day": score_day,
            "variant": variant,
            "variant_description": description,
            "base_name": base_name,
            "base_model_id": NAME_TO_ID[base_name],
            "current_name": current_name,
            "current_model_id": NAME_TO_ID[current_name],
            "current_return": float(row["selected_return"]),
            "route_eligible": eligible,
            "base_reason": str(row["base_reason"]),
            "current_reason": str(row["reason"]),
            "current_robust_reason": str(row["robust_reason"]),
            "current_robust_name": row["current_robust_name"],
            "current_robust_confident": bool(row["robust_confident"]),
            "current_robust_score_gap": pd.to_numeric(pd.Series([row["robust_score_gap"]]), errors="coerce").iloc[0],
            "history_days": int(len(history)),
            "prediction_ready": ready,
            "feature_count": int(len(feature_cols)),
            "target_clip": TARGET_CLIP,
            "alpha": ALPHA,
            "margin": MARGIN,
        }
        for left, right in PAIRWISE:
            key = pair_key(left, right)
            actual_raw = float(row[left] - row[right])
            record[f"actual_{key}_raw"] = actual_raw
            record[f"actual_{key}_clipped"] = float(np.clip(actual_raw, -TARGET_CLIP, TARGET_CLIP))
            record[f"pred_{key}"] = np.nan

        if ready:
            train_x = history[feature_cols].to_numpy(dtype=float)
            train_returns = history[MODELS].to_numpy(dtype=float)
            transformed_train, transformed_current, transform_meta = transform(train_x, current_features)
            record.update(transform_meta)
            utilities = np.zeros(len(MODELS), dtype=float)
            pairwise_predictions: list[dict[str, object]] = []
            for left_idx, right_idx in ((0, 1), (0, 2), (1, 2)):
                left, right = MODELS[left_idx], MODELS[right_idx]
                key = pair_key(left, right)
                target = np.clip(train_returns[:, left_idx] - train_returns[:, right_idx], -TARGET_CLIP, TARGET_CLIP)
                estimator = Ridge(alpha=ALPHA, fit_intercept=True)
                estimator.fit(transformed_train, target)
                predicted_diff = float(estimator.predict(transformed_current)[0])
                record[f"pred_{key}"] = predicted_diff
                utilities[left_idx] += predicted_diff / len(MODELS)
                utilities[right_idx] -= predicted_diff / len(MODELS)
                pairwise_predictions.append(
                    {
                        "left": left,
                        "right": right,
                        "predicted_return_diff": predicted_diff,
                    }
                )
            ranking_indices = np.argsort(-utilities, kind="stable")
            best_idx, second_idx = int(ranking_indices[0]), int(ranking_indices[1])
            robust_name = MODELS[best_idx]
            robust_gap = float(utilities[best_idx] - utilities[second_idx])
            robust_confident = bool(robust_gap > MARGIN)
            robust_reason = "score_best" if robust_confident else "margin_default"
            record.update(
                {
                    "robust_name": robust_name,
                    "robust_model_id": NAME_TO_ID[robust_name],
                    "robust_score_gap": robust_gap,
                    "robust_confident": robust_confident,
                    "robust_reason": robust_reason,
                    "candidate_scores": json.dumps({model: float(utilities[idx]) for idx, model in enumerate(MODELS)}, sort_keys=True),
                    "pairwise_predictions": json.dumps(pairwise_predictions, sort_keys=True),
                }
            )
        else:
            record.update(
                {
                    "robust_name": None,
                    "robust_model_id": None,
                    "robust_score_gap": np.nan,
                    "robust_confident": False,
                    "robust_reason": "insufficient_history_or_feature",
                    "candidate_scores": "{}",
                    "pairwise_predictions": "[]",
                }
            )

        if not eligible:
            selected_name = current_name
            decision_reason = "outside_existing_robust_intervention_branch"
        elif not ready:
            selected_name = base_name
            decision_reason = "variant_not_ready_keep_base"
        elif bool(record["robust_confident"]):
            selected_name = str(record["robust_name"])
            decision_reason = "variant_robust_override_or_agree_base"
        else:
            selected_name = base_name
            decision_reason = "variant_robust_not_confident_keep_base"

        record.update(
            {
                "selected_name": selected_name,
                "selected_model_id": NAME_TO_ID[selected_name],
                "selected_return": float(row[selected_name]),
                "decision_reason": decision_reason,
                "override_base": bool(eligible and selected_name != base_name),
                "changed_vs_current": bool(selected_name != current_name),
                "return_delta_vs_current": float(row[selected_name] - row[current_name]),
                "return_delta_vs_base": float(row[selected_name] - row[base_name]) if eligible else 0.0,
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def pairwise_oos_metrics(daily: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in PAIRWISE:
        key = pair_key(left, right)
        prediction = pd.to_numeric(daily[f"pred_{key}"], errors="coerce")
        for target_kind in ("raw", "clipped"):
            actual = pd.to_numeric(daily[f"actual_{key}_{target_kind}"], errors="coerce")
            valid = daily["prediction_ready"].astype(bool) & prediction.notna() & actual.notna()
            y_pred = prediction.loc[valid].to_numpy(dtype=float)
            y_true = actual.loc[valid].to_numpy(dtype=float)
            if len(y_true) < 2:
                rows.append({"variant": variant, "pair": key, "target": target_kind, "n": int(len(y_true))})
                continue
            corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan")
            sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / sst) if sst > 0.0 else float("nan")
            nonzero = np.abs(y_true) > 1e-15
            direction = float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))) if nonzero.any() else float("nan")
            rows.append(
                {
                    "variant": variant,
                    "pair": key,
                    "target": target_kind,
                    "n": int(len(y_true)),
                    "corr": corr,
                    "oos_r2": r2,
                    "direction_n": int(nonzero.sum()),
                    "direction_hit_rate": direction,
                    "mae_bp": float(np.mean(np.abs(y_true - y_pred)) * 1e4),
                    "mean_actual_bp": float(np.mean(y_true) * 1e4),
                    "mean_prediction_bp": float(np.mean(y_pred) * 1e4),
                }
            )
    return pd.DataFrame(rows)


def monthly_rows(daily: pd.DataFrame, variant: str) -> list[dict[str, object]]:
    frame = daily.copy()
    frame["month"] = pd.to_datetime(frame["score_day"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month"):
        current_return = float(np.prod(1.0 + group["current_return"].to_numpy(dtype=float)) - 1.0)
        variant_return = float(np.prod(1.0 + group["selected_return"].to_numpy(dtype=float)) - 1.0)
        rows.append(
            {
                "variant": variant,
                "month": month,
                "days": int(len(group)),
                "route_eligible_days": int(group["route_eligible"].sum()),
                "ready_days": int(group["prediction_ready"].sum()),
                "override_base_days": int((group["route_eligible"] & group["override_base"]).sum()),
                "changed_days_vs_current": int(group["changed_vs_current"].sum()),
                "current_return": current_return,
                "variant_return": variant_return,
                "return_delta_vs_current": float(variant_return - current_return),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="parent directory for a timestamped run")
    parser.add_argument("--strict-current-validation", action="store_true", help="fail if current path44 Ridge does not reproduce all eligible current decisions")
    args = parser.parse_args()

    output = Path(args.output_root) / f"run_20260728_{datetime.now().strftime('%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    all_rows, aligned, disagreement_meta = load_inputs()
    valid_robust = aligned["robust_reason"].isin(["score_best", "margin_default"])
    route_eligible = aligned["base_reason"].eq("margin_default") & valid_robust
    baseline = selector_metrics(aligned["selected_return"], aligned["score_day"], aligned["current_name"])

    daily_by_variant: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    pairwise_rows: list[pd.DataFrame] = []
    month_rows: list[dict[str, object]] = []

    for config in VARIANTS:
        variant = str(config["variant"])
        daily = run_variant(
            all_rows,
            aligned,
            route_eligible,
            variant=variant,
            feature_cols=list(config["feature_cols"]),
            transform=config["transform"],  # type: ignore[arg-type]
            description=str(config["description"]),
        )
        daily.to_csv(output / f"daily_{variant}.csv", index=False, encoding="utf-8-sig")
        daily_by_variant[variant] = daily
        metrics = selector_metrics(daily["selected_return"], daily["score_day"], daily["selected_name"])
        changed = daily["changed_vs_current"].astype(bool)
        overrides = daily["route_eligible"].astype(bool) & daily["override_base"].astype(bool)
        metrics.update(
            {
                "variant": variant,
                "feature_count": int(config["feature_cols"].__len__()),
                "route_eligible_days": int(daily["route_eligible"].sum()),
                "prediction_ready_days": int(daily["prediction_ready"].sum()),
                "eligible_ready_days": int((daily["route_eligible"] & daily["prediction_ready"]).sum()),
                "changed_days_vs_current": int(changed.sum()),
                "override_base_days": int(overrides.sum()),
                "total_return_delta_vs_current": float(metrics["total_return"] - baseline["total_return"]),
                "sharpe_delta_vs_current": float(metrics["sharpe"] - baseline["sharpe"]),
            }
        )
        summary_rows.append(metrics)
        for condition, subset, column in (
            ("changed_vs_current", daily.loc[changed], "return_delta_vs_current"),
            ("override_vs_base", daily.loc[overrides], "return_delta_vs_base"),
            ("all_eligible_vs_current", daily.loc[daily["route_eligible"]], "return_delta_vs_current"),
        ):
            result = conditional_metrics(subset[column])
            result.update({"variant": variant, "condition": condition, "route_eligible_days": int(daily["route_eligible"].sum()), "override_base_days": int(overrides.sum())})
            condition_rows.append(result)
        eligible = daily["route_eligible"].astype(bool)
        selected_match = daily["selected_name"].eq(daily["current_name"])
        robust_match = daily["robust_name"].eq(daily["current_robust_name"])
        validation_rows.append(
            {
                "variant": variant,
                "all_decision_matches_current": int(selected_match.sum()),
                "all_decision_mismatches_current": int((~selected_match).sum()),
                "eligible_decision_matches_current": int((selected_match & eligible).sum()),
                "eligible_decision_mismatches_current": int((~selected_match & eligible).sum()),
                "eligible_robust_winner_matches_current": int((robust_match & eligible).sum()),
                "eligible_robust_winner_mismatches_current": int((~robust_match & eligible).sum()),
            }
        )
        pairwise_rows.append(pairwise_oos_metrics(daily, variant))
        month_rows.extend(monthly_rows(daily, variant))

    summary = pd.DataFrame(summary_rows).sort_values("total_return_delta_vs_current", ascending=False)
    conditions = pd.DataFrame(condition_rows)
    validation = pd.DataFrame(validation_rows)
    pairwise = pd.concat(pairwise_rows, ignore_index=True)
    monthly = pd.DataFrame(month_rows)
    summary.to_csv(output / "ridge_replay_summary.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output / "ridge_replay_conditional_metrics.csv", index=False, encoding="utf-8-sig")
    validation.to_csv(output / "ridge_replay_validation.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output / "ridge_replay_pairwise_oos.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(output / "ridge_replay_monthly.csv", index=False, encoding="utf-8-sig")
    all_rows[["score_day", *DISAGREEMENT_FEATURES]].to_csv(output / "score_disagreement_features.csv", index=False, encoding="utf-8-sig")

    current_validation = validation.loc[validation["variant"].eq("path44_standard_ridge")].iloc[0]
    if args.strict_current_validation and int(current_validation["eligible_decision_mismatches_current"]) != 0:
        raise RuntimeError("current 44-feature Ridge did not reproduce the eligible current decisions; see ridge_replay_validation.csv")

    manifest = {
        "run_kind": "offline_read_only_robust_ridge_input_rebuild",
        "baseline": "aligned current selector replay; retain current Base, Champion-first, Champion-third, and all routing outside the current Robust branch",
        "date_window": {"start": str(aligned["score_day"].min()), "end": str(aligned["score_day"].max()), "days": int(len(aligned))},
        "branch_contract": "replace only Base margin_default dates with current numeric Robust diagnostics; preserve all other current selections",
        "route_eligible_days": int(route_eligible.sum()),
        "robust_contract": {"lookback_days": LOOKBACK_DAYS, "min_periods": MIN_PERIODS, "alpha": ALPHA, "target_clip": TARGET_CLIP, "margin": MARGIN, "pairwise_target": "clip(standalone model day_return difference, +/-75bp)"},
        "variants": [{"variant": str(item["variant"]), "description": str(item["description"]), "feature_count": len(item["feature_cols"])} for item in VARIANTS],
        "pairwise_oos_evaluation": "all prediction-ready dates; both raw realized return difference and the clipped training target are reported",
        "model_disagreement_variant": disagreement_meta,
        "model_return_semantics": "standalone shadow day_return composition; no continuous fused holdings or incremental switching-cost replay",
        "database_writes": False,
        "live_runtime_called": False,
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in [STATE_PATH, CURRENT_PATH, *RETURN_PATHS.values()]],
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(12.4, 5.6))
        axis.plot(pd.to_datetime(aligned["score_day"]), np.cumprod(1.0 + aligned["selected_return"].to_numpy(dtype=float)), label="current", color="#202020", linewidth=2.5)
        colors = {
            "path44_standard_ridge": "#4c78a8",
            "disp7_standard_ridge": "#f58518",
            "path44_robust_pca15_whiten_ridge": "#54a24b",
            "path44_pca15_disagreement_ridge": "#e45756",
        }
        for variant, daily in daily_by_variant.items():
            axis.plot(pd.to_datetime(daily["score_day"]), np.cumprod(1.0 + daily["selected_return"].to_numpy(dtype=float)), label=variant, color=colors.get(variant), linewidth=1.5)
        axis.set_title("Robust Ridge input rebuild replay (shadow-return composition)")
        axis.set_ylabel("NAV")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output / "nav_compare.png", dpi=160)
        plt.close(fig)
        plot_path = "nav_compare.png"
    except Exception as exc:  # plot is evidence only, never a calculation input
        plot_path = f"plot_unavailable: {type(exc).__name__}: {exc}"

    raw_pairwise = pairwise.loc[pairwise["target"].eq("raw")].copy()
    report = [
        "# Robust Ridge input rebuild replay (2026-07-28)",
        "",
        "## Contract",
        "",
        f"- Aligned score days: `{aligned.score_day.min()}` to `{aligned.score_day.max()}` (`{len(aligned)}` days).",
        "- Base, Champion-first, Champion-third, and all non-Robust routing remain the existing selector replay values.",
        f"- Only the current Base-low-confidence / numeric-Robust branch is recomputed (`{int(route_eligible.sum())}` days).",
        f"- Every variant uses 360 prior days, min 120, pairwise clipped target `+/-{TARGET_CLIP:.2%}`, Ridge alpha `{ALPHA:.0f}`, and 5bp top-two utility margin.",
        "- Returns are standalone shadow-return composition, not a continuous-position or incremental-switching-cost replay.",
        "",
        "## Current 44-feature reconstruction check",
        "",
        *csv_block(validation.loc[validation["variant"].eq("path44_standard_ridge")]),
        "",
        "A zero eligible-decision mismatch is required before treating the other two variants as aligned comparisons.",
        "",
        "## Pairwise OOS diagnostics (raw realized return differences)",
        "",
        "Metrics use every prediction-ready day, not just the dates whose final routing can change. `oos_r2` is versus the out-of-sample raw pairwise return difference; the CSV additionally includes the clipped-target version.",
        "",
        *csv_block(raw_pairwise[["variant", "pair", "n", "corr", "oos_r2", "direction_hit_rate", "mae_bp"]]),
        "",
        "## Selector result",
        "",
        f"- Existing current replay: total return `{baseline['total_return']:.2%}`, Sharpe `{baseline['sharpe']:.3f}`, max drawdown `{baseline['max_drawdown']:.2%}`, switches `{baseline['switch_count']}`.",
        "",
        *csv_block(summary[["variant", "total_return", "sharpe", "max_drawdown", "switch_count", "changed_days_vs_current", "override_base_days", "total_return_delta_vs_current", "sharpe_delta_vs_current"]]),
        "",
        "## Conditional result: actual changed decisions",
        "",
        *csv_block(conditions.loc[conditions["condition"].eq("changed_vs_current"), ["variant", "n", "wins", "losses", "mean_delta_bp", "median_delta_bp", "paired_t_p_one_sided", "sign_p_one_sided", "wilcoxon_p_one_sided"]]),
        "",
        "## Model-disagreement extension",
        "",
        "- Included as one pre-specified fourth variant: 44 path-full state features plus 12 fixed score-disagreement features, followed by the same rolling median/IQR PCA-15-whiten Ridge.",
        "- For each score day the score features use only that day\'s three archived score CSVs: three pairwise Spearman correlations, three pairwise Top20 Jaccards, and each model\'s cross-sectional score standard deviation.",
        "- The remaining three features are each model\'s Top20 Jaccard versus the preceding aligned score day (turnover proxy = `1 - Jaccard`). The first score day has no prior-day overlap and is therefore unavailable only as a training observation.",
        "",
        "## Artifacts",
        "",
        "- `ridge_replay_summary.csv`",
        "- `ridge_replay_pairwise_oos.csv` (raw and clipped-target OOS diagnostics)",
        "- `ridge_replay_conditional_metrics.csv`",
        "- `ridge_replay_validation.csv`",
        "- `ridge_replay_monthly.csv`",
        "- `score_disagreement_features.csv`",
        "- `daily_<variant>.csv`",
        f"- `{plot_path}`",
        "- `input_manifest.json`",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"OUTPUT_ROOT {output}")
    print(f"ROUTE_ELIGIBLE_DAYS {int(route_eligible.sum())}")
    print("CURRENT44_VALIDATION")
    print(validation.loc[validation["variant"].eq("path44_standard_ridge")].to_string(index=False))
    print("SELECTOR_SUMMARY")
    print(summary[["variant", "total_return_delta_vs_current", "sharpe_delta_vs_current", "changed_days_vs_current", "override_base_days"]].to_string(index=False))
    print("PAIRWISE_OOS_RAW")
    print(raw_pairwise[["variant", "pair", "n", "corr", "oos_r2", "direction_hit_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
