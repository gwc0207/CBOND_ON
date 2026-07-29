"""Research-only coherent two-contrast Ridge replay for model switching.

The current Robust diagnostic fits three independently clipped pairwise Ridge
models for three candidate models.  This tool instead represents each daily
three-model label vector in two orthogonal Helmert contrasts, jointly scales
the complete vector to retain pairwise consistency under the +/-75bp clip, and
fits two independent Ridge regressions.  The inverse transform reconstructs
three utilities whose pairwise differences are always coherent.

The label interface is intentionally separate from this replay:

* --label-csv supplies an external switch-aware panel with columns
  score_day, Regsim, Ensemble, HL20.  It is the only formal label input.
* --evaluation-csv optionally supplies a separate realised-return panel with
  the same schema for selector-return accounting.
* --shadow-current-full-liquidation validates the active `turnover_ratio=1.0`
  strategy boundary and uses the existing standalone shadow returns as the
  current switch-aware-equivalent label/evaluation panel.  This is valid for
  the active full-liquidation strategy only, still research-only, and must not
  be extrapolated to a future partial-turnover strategy.

No live model-switch code, live configuration, database, scheduler, model
state, or live artifact is imported or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ANALYSIS_ROOT = Path(r"D:\cbond_on\results\analysis")
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_coherent_ridge_20260728")
STATE_PATH = ANALYSIS_ROOT / "model_switch_t1430_path_full_20260721" / "t1430_market_state_features_path_full.csv"
CURRENT_PATH = (
    ANALYSIS_ROOT
    / "model_switch_robust_strict_veto_20260728"
    / "run_20260728_152318"
    / "daily_current.csv"
)
SHADOW_RETURN_PATHS = {
    "Regsim": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Challenger_Regsim.csv",
    "Ensemble": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Challenger_Ensemble.csv",
    "HL20": ANALYSIS_ROOT / "model_switch_scoreopt_live_20260709" / "return_history" / "Champion_HL20.csv",
}
STRATEGY_CONFIG_PATH = Path(r"C:\Users\BaiYang\CBOND_ON\cbond_on\cbond_on\config\strategies\strategy01\strategy01_config.json5")
STRATEGY_IMPLEMENTATION_PATH = Path(r"C:\Users\BaiYang\CBOND_ON\cbond_on\cbond_on\domain\strategies\strategy01\strategy01_topk_turnover.py")
LIVE_CONFIG_PATH = Path(r"C:\Users\BaiYang\CBOND_ON\cbond_on\cbond_on\config\live\live_config.json5")

ID_TO_NAME = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": "Regsim",
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": "Ensemble",
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": "HL20",
}
NAME_TO_ID = {value: key for key, value in ID_TO_NAME.items()}
MODELS = ["Regsim", "Ensemble", "HL20"]
PAIRWISE = [("Regsim", "Ensemble"), ("Regsim", "HL20"), ("Ensemble", "HL20")]

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
FEATURES = [
    f"{prefix}_{suffix}"
    for prefix in PATH_PREFIXES
    for suffix in ["mean", "std", "iqr", "pos_ratio", "tail_spread"]
] + ["trend_accel_median", "trend_accel_mean", "dispersion_accel", "tail_balance_full"]

LOOKBACK_DAYS = 360
MIN_PERIODS = 120
ALPHA = 100.0
TARGET_CLIP = 0.0075
MARGIN = 0.0005
SQRT2 = math.sqrt(2.0)
SQRT6 = math.sqrt(6.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_key(left: str, right: str) -> str:
    return f"{left}_minus_{right}"


def csv_block(frame: pd.DataFrame) -> list[str]:
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def selector_metrics(returns: pd.Series, dates: pd.Series, selected: pd.Series) -> dict[str, float | int | str]:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("selector metrics require finite aligned evaluation returns")
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
        "paired_t_p_one_sided": float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue) if len(values) > 1 else float("nan"),
        "sign_p_one_sided": float(stats.binomtest(wins, len(values), 0.5, alternative="greater").pvalue),
    }
    try:
        result["wilcoxon_p_one_sided"] = float(stats.wilcoxon(values, alternative="greater", method="auto").pvalue)
    except ValueError:
        result["wilcoxon_p_one_sided"] = float("nan")
    return result


def load_external_panel(path: Path, *, kind: str) -> pd.DataFrame:
    """Load an external label or evaluation panel without inference.

    Required schema is intentionally minimal and model-name based so a future
    switch-aware label producer is not coupled to this replay implementation.
    """

    if not path.exists():
        raise FileNotFoundError(f"{kind} panel missing: {path}")
    frame = pd.read_csv(path)
    required = {"score_day", *MODELS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{kind} panel missing required columns {missing}: {path}")
    output = frame[["score_day", *MODELS]].copy()
    output["score_day"] = pd.to_datetime(output["score_day"], errors="coerce").dt.date
    for model in MODELS:
        output[model] = pd.to_numeric(output[model], errors="coerce")
    output = output.dropna(subset=["score_day"]).sort_values("score_day")
    if output["score_day"].duplicated().any():
        duplicates = output.loc[output["score_day"].duplicated(keep=False), "score_day"].astype(str).tolist()
        raise ValueError(f"{kind} panel has duplicate score_day values: {duplicates[:10]}")
    return output.reset_index(drop=True)


def validate_current_full_liquidation_contract() -> dict[str, object]:
    """Verify the current strategy contract that makes shadow labels equivalent.

    Under `turnover_ratio=1.0`, strategy01 ignores `prev_positions`; strict
    cycle returns fully sell the selected holdings on the following morning.
    Therefore a selector's return for a score day is exactly the current
    selected model's standalone shadow `day_return`, rather than a path that
    depends on yesterday's selected model holdings.
    """

    from cbond_on.core.config import load_config_file

    config = load_config_file(STRATEGY_CONFIG_PATH)
    live_config = load_config_file(LIVE_CONFIG_PATH)
    live_strategy = live_config.get("strategy", {})
    if not isinstance(live_strategy, dict):
        raise RuntimeError("current live_config strategy block is not a mapping")
    if str(live_strategy.get("strategy_id", "")).strip() != "strategy01_topk_turnover":
        raise RuntimeError(f"current live strategy is not strategy01_topk_turnover: {live_strategy.get('strategy_id')}")
    configured_path = str(live_strategy.get("strategy_config_path", "")).replace("\\", "/")
    if not configured_path.endswith("strategies/strategy01/strategy01"):
        raise RuntimeError(f"current live strategy config path is not strategy01: {configured_path}")
    turnover_ratio = float(config.get("turnover_ratio", float("nan")))
    if not math.isclose(turnover_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            "current standalone shadow labels are not selector-equivalent when "
            f"strategy01 turnover_ratio != 1.0; observed {turnover_ratio}"
        )
    source = STRATEGY_IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    if "turnover_ratio < 1.0" not in source:
        raise RuntimeError("could not verify strategy01 previous-position gate in current source")
    return {
        "turnover_ratio": turnover_ratio,
        "strategy_config_path": str(STRATEGY_CONFIG_PATH),
        "strategy_config_sha256": sha256(STRATEGY_CONFIG_PATH),
        "live_config_path": str(LIVE_CONFIG_PATH),
        "live_config_sha256": sha256(LIVE_CONFIG_PATH),
        "strategy_implementation_path": str(STRATEGY_IMPLEMENTATION_PATH),
        "strategy_implementation_sha256": sha256(STRATEGY_IMPLEMENTATION_PATH),
        "equivalence": "turnover_ratio=1.0 makes strategy01 ignore prev_positions; each strict cycle sells on next-day 09:30-09:39, so score-day model selection equals that model standalone shadow day_return",
    }


def load_shadow_panel() -> pd.DataFrame:
    """Current full-liquidation-equivalent panel from standalone shadow returns."""

    panel: pd.DataFrame | None = None
    for model, path in SHADOW_RETURN_PATHS.items():
        frame = pd.read_csv(path, usecols=["trade_date", "day_return"])
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[model] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", model]].dropna().drop_duplicates("score_day", keep="last")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
    if panel is None:
        raise RuntimeError("no current shadow-return panel loaded")
    return panel.sort_values("score_day").reset_index(drop=True)


def load_current_routing() -> pd.DataFrame:
    current = pd.read_csv(CURRENT_PATH)
    current["score_day"] = pd.to_datetime(current["score_day"], errors="coerce").dt.date
    current = current.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    current["current_name"] = current["selected_model_id"].map(ID_TO_NAME)
    current["base_name"] = current["base_model_id"].map(ID_TO_NAME)
    if current[["current_name", "base_name"]].isna().any().any():
        unknown = current.loc[current[["current_name", "base_name"]].isna().any(axis=1), ["score_day", "selected_model_id", "base_model_id"]]
        raise RuntimeError(f"unmapped model ID in current routing:\n{unknown.to_string(index=False)}")
    return current


def load_state() -> pd.DataFrame:
    state = pd.read_csv(STATE_PATH, usecols=["trade_date", *FEATURES])
    state["score_day"] = pd.to_datetime(state["trade_date"], errors="coerce").dt.date
    state = state[["score_day", *FEATURES]].dropna(subset=["score_day"]).drop_duplicates("score_day", keep="last")
    for feature in FEATURES:
        state[feature] = pd.to_numeric(state[feature], errors="coerce")
    return state.sort_values("score_day").reset_index(drop=True)


def joint_clip_and_contrasts(values: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Return coherent clipped utilities, a common scale, and Helmert targets.

    Scaling the whole centred vector, instead of clipping three pairwise labels
    independently, preserves all transitivity identities.  The maximum
    absolute reconstructed pairwise difference is at most TARGET_CLIP.
    """

    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    pairwise_max = max(abs(centered[0] - centered[1]), abs(centered[0] - centered[2]), abs(centered[1] - centered[2]))
    scale = min(1.0, TARGET_CLIP / pairwise_max) if pairwise_max > 0.0 else 1.0
    utilities = centered * scale
    contrast = np.asarray(
        [
            (utilities[0] - utilities[1]) / SQRT2,
            (utilities[0] + utilities[1] - 2.0 * utilities[2]) / SQRT6,
        ],
        dtype=float,
    )
    return utilities, float(scale), contrast


def inverse_contrasts(contrast: np.ndarray) -> np.ndarray:
    contrast = np.asarray(contrast, dtype=float)
    if contrast.shape != (2,):
        raise ValueError(f"expected exactly two contrast predictions, got {contrast.shape}")
    first, second = float(contrast[0]), float(contrast[1])
    return np.asarray(
        [
            first / SQRT2 + second / SQRT6,
            -first / SQRT2 + second / SQRT6,
            -2.0 * second / SQRT6,
        ],
        dtype=float,
    )


def make_joint_targets(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.copy()
    utilities: list[np.ndarray] = []
    scales: list[float] = []
    contrasts: list[np.ndarray] = []
    valid: list[bool] = []
    for row in output[MODELS].to_numpy(dtype=float):
        if not np.isfinite(row).all():
            utilities.append(np.full(3, np.nan))
            contrasts.append(np.full(2, np.nan))
            scales.append(float("nan"))
            valid.append(False)
            continue
        utility, scale, contrast = joint_clip_and_contrasts(row)
        utilities.append(utility)
        scales.append(scale)
        contrasts.append(contrast)
        valid.append(True)
    utility_array = np.asarray(utilities)
    contrast_array = np.asarray(contrasts)
    output["joint_label_valid"] = valid
    output["joint_label_scale"] = scales
    for index, model in enumerate(MODELS):
        output[f"joint_utility_{model}"] = utility_array[:, index]
    output["contrast_regsim_vs_ensemble"] = contrast_array[:, 0]
    output["contrast_regsim_ensemble_vs_hl20"] = contrast_array[:, 1]
    return output


def normalize_features(train_x: np.ndarray, current_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0, ddof=0)
    stds = np.where((~np.isfinite(stds)) | (stds < 1e-12), 1.0, stds)
    return (train_x - means) / stds, (current_x - means) / stds


def prepare_inputs(
    label_panel: pd.DataFrame,
    evaluation_panel: pd.DataFrame | None,
    *,
    verify_current_shadow_equivalence: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, dict[str, object]]:
    labels = make_joint_targets(label_panel)
    state = load_state()
    all_rows = labels.merge(state, on="score_day", how="left", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    if evaluation_panel is not None:
        evaluation = evaluation_panel.rename(columns={model: f"eval_{model}" for model in MODELS})
        all_rows = all_rows.merge(evaluation, on="score_day", how="left", validate="one_to_one")
    else:
        for model in MODELS:
            all_rows[f"eval_{model}"] = np.nan
    routing = load_current_routing()
    aligned = routing.merge(all_rows, on="score_day", how="inner", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    evaluation_available = bool(aligned[[f"eval_{model}" for model in MODELS]].notna().all(axis=None))
    alignment: dict[str, object] = {"checked": False, "reason": "external labels or no evaluation panel"}
    if verify_current_shadow_equivalence:
        if not evaluation_available:
            raise RuntimeError("current full-liquidation shadow mode requires a complete evaluation panel")
        expected = pd.to_numeric(aligned["selected_return"], errors="coerce").to_numpy(dtype=float)
        actual = np.asarray([float(row[f"eval_{row['current_name']}"]) for _, row in aligned.iterrows()], dtype=float)
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise RuntimeError("current shadow-equivalence check encountered non-finite returns")
        difference = np.abs(actual - expected)
        max_abs_difference = float(difference.max()) if len(difference) else 0.0
        mismatch_count = int((difference > 1e-12).sum())
        if mismatch_count:
            bad = aligned.loc[difference > 1e-12, ["score_day", "current_name", "selected_return"]].head(10)
            raise RuntimeError(
                "standalone shadow returns do not reproduce current routing return on "
                f"{mismatch_count} days; max_abs={max_abs_difference}; examples:\n{bad.to_string(index=False)}"
            )
        alignment = {
            "checked": True,
            "aligned_days": int(len(aligned)),
            "mismatch_days": mismatch_count,
            "max_abs_return_difference": max_abs_difference,
            "tolerance": 1e-12,
        }
    return all_rows, aligned, evaluation_available, alignment


def run_replay(all_rows: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in aligned.iterrows():
        score_day = row["score_day"]
        base_name = str(row["base_name"])
        current_name = str(row["current_name"])
        valid_robust = str(row["robust_reason"]) in {"score_best", "margin_default"}
        route_eligible = bool(str(row["base_reason"]) == "margin_default" and valid_robust)
        current_features = row[FEATURES].to_numpy(dtype=float).reshape(1, -1)
        history = all_rows.loc[all_rows["score_day"] < score_day].tail(LOOKBACK_DAYS).copy()
        valid_history = (
            history["joint_label_valid"].astype(bool).to_numpy()
            & np.isfinite(history[FEATURES].to_numpy(dtype=float)).all(axis=1)
        )
        history = history.loc[valid_history].copy()
        ready = bool(np.isfinite(current_features).all() and len(history) >= MIN_PERIODS)
        record: dict[str, object] = {
            "score_day": score_day,
            "base_name": base_name,
            "base_model_id": NAME_TO_ID[base_name],
            "current_name": current_name,
            "current_model_id": NAME_TO_ID[current_name],
            "route_eligible": route_eligible,
            "base_reason": str(row["base_reason"]),
            "current_reason": str(row["reason"]),
            "current_robust_reason": str(row["robust_reason"]),
            "history_days": int(len(history)),
            "prediction_ready": ready,
            "feature_count": len(FEATURES),
            "target_clip": TARGET_CLIP,
            "alpha": ALPHA,
            "margin": MARGIN,
            "joint_label_scale": row["joint_label_scale"],
            "actual_contrast_regsim_vs_ensemble": row["contrast_regsim_vs_ensemble"],
            "actual_contrast_regsim_ensemble_vs_hl20": row["contrast_regsim_ensemble_vs_hl20"],
        }
        for left, right in PAIRWISE:
            key = pair_key(left, right)
            record[f"actual_{key}_raw"] = float(row[left] - row[right]) if pd.notna(row[left]) and pd.notna(row[right]) else np.nan
            record[f"actual_{key}_joint_clipped"] = float(row[f"joint_utility_{left}"] - row[f"joint_utility_{right}"])
            record[f"pred_{key}"] = np.nan
        if ready:
            train_x = history[FEATURES].to_numpy(dtype=float)
            current_x = current_features
            train_x, current_x = normalize_features(train_x, current_x)
            targets = history[["contrast_regsim_vs_ensemble", "contrast_regsim_ensemble_vs_hl20"]].to_numpy(dtype=float)
            predictions: list[float] = []
            for target_index in range(2):
                estimator = Ridge(alpha=ALPHA, fit_intercept=True)
                estimator.fit(train_x, targets[:, target_index])
                predictions.append(float(estimator.predict(current_x)[0]))
            utilities = inverse_contrasts(np.asarray(predictions, dtype=float))
            if not math.isclose(float(utilities.sum()), 0.0, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("inverse coherent utilities do not sum to zero")
            ranking = np.argsort(-utilities, kind="stable")
            best_idx, second_idx = int(ranking[0]), int(ranking[1])
            robust_name = MODELS[best_idx]
            robust_gap = float(utilities[best_idx] - utilities[second_idx])
            robust_confident = bool(robust_gap > MARGIN)
            record.update(
                {
                    "pred_contrast_regsim_vs_ensemble": predictions[0],
                    "pred_contrast_regsim_ensemble_vs_hl20": predictions[1],
                    "robust_name": robust_name,
                    "robust_model_id": NAME_TO_ID[robust_name],
                    "robust_score_gap": robust_gap,
                    "robust_confident": robust_confident,
                    "robust_reason": "score_best" if robust_confident else "margin_default",
                    "candidate_scores": json.dumps({model: float(utilities[idx]) for idx, model in enumerate(MODELS)}, sort_keys=True),
                }
            )
            for left, right in PAIRWISE:
                record[f"pred_{pair_key(left, right)}"] = float(utilities[MODELS.index(left)] - utilities[MODELS.index(right)])
        else:
            record.update(
                {
                    "pred_contrast_regsim_vs_ensemble": np.nan,
                    "pred_contrast_regsim_ensemble_vs_hl20": np.nan,
                    "robust_name": None,
                    "robust_model_id": None,
                    "robust_score_gap": np.nan,
                    "robust_confident": False,
                    "robust_reason": "insufficient_history_or_feature",
                    "candidate_scores": "{}",
                }
            )
        if not route_eligible:
            selected_name = current_name
            decision_reason = "outside_existing_robust_intervention_branch"
        elif not ready:
            selected_name = base_name
            decision_reason = "coherent_not_ready_keep_base"
        elif bool(record["robust_confident"]):
            selected_name = str(record["robust_name"])
            decision_reason = "coherent_robust_override_or_agree_base"
        else:
            selected_name = base_name
            decision_reason = "coherent_robust_not_confident_keep_base"
        record["selected_name"] = selected_name
        record["selected_model_id"] = NAME_TO_ID[selected_name]
        record["decision_reason"] = decision_reason
        record["override_base"] = bool(route_eligible and selected_name != base_name)
        record["changed_vs_current"] = bool(selected_name != current_name)
        for model in MODELS:
            record[f"eval_{model}"] = row[f"eval_{model}"]
        selected_return = row[f"eval_{selected_name}"]
        current_return = row[f"eval_{current_name}"]
        base_return = row[f"eval_{base_name}"]
        record["selected_return"] = selected_return
        record["current_return"] = current_return
        record["return_delta_vs_current"] = float(selected_return - current_return) if pd.notna(selected_return) and pd.notna(current_return) else np.nan
        record["return_delta_vs_base"] = float(selected_return - base_return) if route_eligible and pd.notna(selected_return) and pd.notna(base_return) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def pairwise_oos_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in PAIRWISE:
        key = pair_key(left, right)
        prediction = pd.to_numeric(daily[f"pred_{key}"], errors="coerce")
        for target_kind in ("raw", "joint_clipped"):
            actual = pd.to_numeric(daily[f"actual_{key}_{target_kind}"], errors="coerce")
            valid = daily["prediction_ready"].astype(bool) & prediction.notna() & actual.notna()
            y_pred = prediction.loc[valid].to_numpy(dtype=float)
            y_true = actual.loc[valid].to_numpy(dtype=float)
            if len(y_true) < 2:
                rows.append({"pair": key, "target": target_kind, "n": int(len(y_true))})
                continue
            corr = float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan")
            sst = float(np.sum((y_true - y_true.mean()) ** 2))
            r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / sst) if sst > 0.0 else float("nan")
            nonzero = np.abs(y_true) > 1e-15
            direction = float(np.mean(np.sign(y_pred[nonzero]) == np.sign(y_true[nonzero]))) if nonzero.any() else float("nan")
            rows.append(
                {
                    "pair": key,
                    "target": target_kind,
                    "n": int(len(y_true)),
                    "corr": corr,
                    "oos_r2": r2,
                    "direction_n": int(nonzero.sum()),
                    "direction_hit_rate": direction,
                    "mae_bp": float(np.mean(np.abs(y_true - y_pred)) * 1e4),
                }
            )
    return pd.DataFrame(rows)


def make_summary(daily: pd.DataFrame, *, evaluation_available: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_eligible = daily["route_eligible"].astype(bool)
    changed = daily["changed_vs_current"].astype(bool)
    overrides = route_eligible & daily["override_base"].astype(bool)
    decision_row = {
        "route_eligible_days": int(route_eligible.sum()),
        "prediction_ready_days": int(daily["prediction_ready"].sum()),
        "eligible_ready_days": int((route_eligible & daily["prediction_ready"].astype(bool)).sum()),
        "changed_days_vs_current": int(changed.sum()),
        "override_base_days": int(overrides.sum()),
    }
    conditions: list[dict[str, object]] = []
    if not evaluation_available:
        return pd.DataFrame([decision_row]), pd.DataFrame(conditions)
    valid = daily[["selected_return", "current_return"]].notna().all(axis=1)
    if not valid.all():
        raise RuntimeError("evaluation panel was declared available but does not cover every aligned replay row")
    current = selector_metrics(daily["current_return"], daily["score_day"], daily["current_name"])
    coherent = selector_metrics(daily["selected_return"], daily["score_day"], daily["selected_name"])
    summary = pd.DataFrame(
        [
            {"strategy": "current_routing", **current, **decision_row, "total_return_delta_vs_current": 0.0, "sharpe_delta_vs_current": 0.0},
            {
                "strategy": "coherent_two_contrast_ridge",
                **coherent,
                **decision_row,
                "total_return_delta_vs_current": float(coherent["total_return"] - current["total_return"]),
                "sharpe_delta_vs_current": float(coherent["sharpe"] - current["sharpe"]),
            },
        ]
    )
    for condition, subset, column in (
        ("changed_vs_current", daily.loc[changed], "return_delta_vs_current"),
        ("override_vs_base", daily.loc[overrides], "return_delta_vs_base"),
        ("all_eligible_vs_current", daily.loc[route_eligible], "return_delta_vs_current"),
    ):
        metrics = conditional_metrics(subset[column])
        metrics.update({"condition": condition, **decision_row})
        conditions.append(metrics)
    return summary, pd.DataFrame(conditions)


def input_source_manifest(
    *,
    label_path: Path | None,
    evaluation_path: Path | None,
    current_full_liquidation_shadow: bool,
    full_liquidation_contract: dict[str, object] | None,
    current_routing_shadow_alignment: dict[str, object],
    all_rows: pd.DataFrame,
    aligned: pd.DataFrame,
) -> dict[str, object]:
    inputs = [{"path": str(path), "sha256": sha256(path)} for path in [STATE_PATH, CURRENT_PATH]]
    if current_full_liquidation_shadow:
        inputs.extend({"path": str(path), "sha256": sha256(path)} for path in SHADOW_RETURN_PATHS.values())
    else:
        if label_path is None:
            raise RuntimeError("formal run manifest requires label path")
        inputs.append({"path": str(label_path), "sha256": sha256(label_path)})
        if evaluation_path is not None:
            inputs.append({"path": str(evaluation_path), "sha256": sha256(evaluation_path)})
    return {
        "run_kind": "current_full_liquidation_shadow_label_coherent_two_contrast_ridge" if current_full_liquidation_shadow else "external_label_coherent_two_contrast_ridge",
        "label_contract": {
            "schema": "CSV columns score_day, Regsim, Ensemble, HL20; rows represent only values available after their score_day selection",
            "training_history": "strictly score_day < current score_day",
            "label_source": "current standalone shadow returns under verified full-liquidation strategy boundary" if current_full_liquidation_shadow else str(label_path),
            "evaluation_source": "same current standalone shadow returns under verified full-liquidation strategy boundary" if current_full_liquidation_shadow else (str(evaluation_path) if evaluation_path else "not supplied"),
        },
        "routing_contract": "preserve current Base, Champion-first, Champion-third, and all non-eligible routing; recompute only Base margin_default plus current numeric Robust branch",
        "coherent_target": {
            "utility_basis": "three daily labels centered cross-sectionally",
            "contrasts": ["(Regsim-Ensemble)/sqrt(2)", "(Regsim+Ensemble-2*HL20)/sqrt(6)"],
            "joint_clip": "multiply complete centered utility vector by min(1, 75bp/max_abs_pairwise_difference); preserves pairwise transitivity",
            "models": "two independent Ridge(alpha=100), reconstructed into three sum-zero utilities",
        },
        "robust_contract": {"features": "path_full_t1430", "feature_count": len(FEATURES), "lookback_days": LOOKBACK_DAYS, "min_periods": MIN_PERIODS, "alpha": ALPHA, "target_clip": TARGET_CLIP, "margin": MARGIN},
        "date_window": {"label_rows": int(len(all_rows)), "aligned_rows": int(len(aligned)), "start": str(aligned["score_day"].min()), "end": str(aligned["score_day"].max())},
        "current_full_liquidation_shadow_equivalence": full_liquidation_contract,
        "current_routing_shadow_return_alignment": current_routing_shadow_alignment,
        "database_writes": False,
        "live_runtime_called": False,
        "inputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--label-csv", help="external switch-aware label CSV: score_day, Regsim, Ensemble, HL20")
    source.add_argument(
        "--shadow-current-full-liquidation",
        action="store_true",
        help="validate active turnover_ratio=1.0 and use current standalone shadow returns as the current strategy-equivalent label/evaluation panel",
    )
    source.add_argument(
        "--shadow-sanity",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--evaluation-csv", help="optional external realised-return CSV with same four-column schema")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="parent directory for timestamped research output")
    args = parser.parse_args()
    current_full_liquidation_shadow = bool(args.shadow_current_full_liquidation or args.shadow_sanity)
    if current_full_liquidation_shadow and args.evaluation_csv:
        raise ValueError("--evaluation-csv is incompatible with current full-liquidation shadow mode; it always uses the same shadow panel")

    label_path = Path(args.label_csv) if args.label_csv else None
    evaluation_path = Path(args.evaluation_csv) if args.evaluation_csv else None
    full_liquidation_contract: dict[str, object] | None = None
    if current_full_liquidation_shadow:
        full_liquidation_contract = validate_current_full_liquidation_contract()
        labels = load_shadow_panel()
        evaluation = labels.copy()
    else:
        if label_path is None:
            raise RuntimeError("external label path is required")
        labels = load_external_panel(label_path, kind="label")
        evaluation = load_external_panel(evaluation_path, kind="evaluation") if evaluation_path is not None else None

    all_rows, aligned, evaluation_available, current_routing_shadow_alignment = prepare_inputs(
        labels,
        evaluation,
        verify_current_shadow_equivalence=current_full_liquidation_shadow,
    )
    if len(aligned) < MIN_PERIODS + 1:
        raise ValueError(f"only {len(aligned)} aligned rows; need more than {MIN_PERIODS} for walk-forward Ridge")
    output = Path(args.output_root) / f"run_20260728_{datetime.now().strftime('%H%M%S')}"
    output.mkdir(parents=True, exist_ok=False)
    daily = run_replay(all_rows, aligned)
    pairwise = pairwise_oos_metrics(daily)
    summary, conditions = make_summary(daily, evaluation_available=evaluation_available)
    daily.to_csv(output / "daily_coherent_two_contrast_ridge.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(output / "pairwise_oos_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output / "conditional_metrics.csv", index=False, encoding="utf-8-sig")
    all_rows[["score_day", "joint_label_valid", "joint_label_scale", *[f"joint_utility_{model}" for model in MODELS], "contrast_regsim_vs_ensemble", "contrast_regsim_ensemble_vs_hl20"]].to_csv(output / "coherent_label_transform.csv", index=False, encoding="utf-8-sig")
    manifest = input_source_manifest(
        label_path=label_path,
        evaluation_path=evaluation_path,
        current_full_liquidation_shadow=current_full_liquidation_shadow,
        full_liquidation_contract=full_liquidation_contract,
        current_routing_shadow_alignment=current_routing_shadow_alignment,
        all_rows=all_rows,
        aligned=aligned,
    )
    (output / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_path = "not_generated_without_evaluation"
    if evaluation_available:
        try:
            import matplotlib.pyplot as plt

            fig, axis = plt.subplots(figsize=(12.4, 5.6))
            axis.plot(pd.to_datetime(daily["score_day"]), np.cumprod(1.0 + daily["current_return"].to_numpy(dtype=float)), label="current routing", color="#202020", linewidth=2.4)
            axis.plot(pd.to_datetime(daily["score_day"]), np.cumprod(1.0 + daily["selected_return"].to_numpy(dtype=float)), label="coherent two-contrast Ridge", color="#4c78a8", linewidth=1.8)
            axis.set_title("Coherent two-contrast Robust Ridge replay")
            axis.set_ylabel("NAV")
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output / "nav_compare.png", dpi=160)
            plt.close(fig)
            plot_path = "nav_compare.png"
        except Exception as exc:  # plot is optional evidence only
            plot_path = f"plot_unavailable: {type(exc).__name__}: {exc}"

    report = [
        "# Coherent two-contrast Robust Ridge replay (research-only)",
        "",
        "## Label status",
        "",
        f"- Run class: `{'current-full-liquidation shadow-label research' if current_full_liquidation_shadow else 'external-label research'}`.",
        f"- Label source: `{'existing standalone shadow returns' if current_full_liquidation_shadow else label_path}`.",
        f"- Evaluation source: `{'existing standalone shadow returns' if current_full_liquidation_shadow else (evaluation_path if evaluation_path else 'not supplied')}`.",
        "- Current-shadow equivalence is valid only because the verified active strategy has `turnover_ratio=1.0`: prior positions are ignored and each strict cycle is fully sold the following morning. It does not extend to a future partial-turnover strategy.",
        "- This remains research-only and is not a live-promotion result.",
        "",
        "## Fixed contract",
        "",
        f"- Aligned score days: `{aligned.score_day.min()}` to `{aligned.score_day.max()}` (`{len(aligned)}` rows).",
        "- Preserve current Base, Champion-first, Champion-third, and every non-eligible route; only current Base-low-confidence / numeric-Robust dates are recomputed.",
        f"- 44 path features; prior 360 labelled days; minimum 120; two independent Ridge models, alpha `{ALPHA:.0f}`; joint pairwise cap `+/-{TARGET_CLIP:.2%}`; 5bp utility-gap decision margin.",
        "- Contrasts are orthogonal Helmert contrasts. Reconstructed utilities sum to zero and their three pairwise differences are algebraically coherent.",
        "",
        "## Pairwise OOS diagnostics",
        "",
        *csv_block(pairwise),
        "",
        "## Decision / evaluation summary",
        "",
        *csv_block(summary),
        "",
        "## Conditional changed-decision metrics",
        "",
        *csv_block(conditions),
        "",
        "## Artifacts",
        "",
        "- `daily_coherent_two_contrast_ridge.csv`",
        "- `coherent_label_transform.csv`",
        "- `pairwise_oos_metrics.csv`",
        "- `summary_metrics.csv`",
        "- `conditional_metrics.csv`",
        "- `input_manifest.json`",
        f"- `{plot_path}`",
    ]
    (output / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"OUTPUT_ROOT {output}")
    print(f"RUN_CLASS {'current_full_liquidation_shadow_label_research' if current_full_liquidation_shadow else 'external_label_research'}")
    print(f"ROUTE_ELIGIBLE_DAYS {int(daily['route_eligible'].sum())}")
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("PAIRWISE_OOS")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
