"""Offline temporal model-switch arbitration research replay.

This tool is deliberately isolated from live execution.  It consumes only a
byte-copied input snapshot, invokes the read-only production Base decision
function to reproduce Base scores, and writes only under the supplied
``results/experiments/model_switch_temporal_arbitration_20260729`` root.

It evaluates a small frozen family of full-feedback temporal forecasters:

* current independent-pair Ridge, but with a direct Base-versus-candidate gate;
* coherent two-contrast Ridge;
* EWMA coherent utilities;
* Fixed-share Hedge with EWMA direct-return evidence;
* two-contrast local-level state space; and
* Dynamic Model Averaging across three local-level process variances.

The action rule is never a temporal top-one/top-two gap.  For Base winner ``b``
and temporal candidate ``r``, an override needs a positive dynamic LCB of the
direct ``r-b`` advantage after combining Base's own ``B_r-B_b`` evidence with
temporal uncertainty calibrated strictly from prior outcomes.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cbond_on.core.config import load_config_file
from cbond_on.infra.live.model_switch import decide_scoreopt_t1430_dispersion
from harness.model_switch_temporal_arbitration import (
    HELMERT,
    CrossArbitration,
    TemporalForecast,
    contrasts_to_utilities,
    cross_arbitrate,
    dma_local_level_forecast,
    ewma_utility_forecast,
    fixed_share_hedge_weights,
    joint_clip_utilities,
    local_level_forecast,
    make_utility_forecast,
    ordered_indices,
    pair_scale_from_utility_covariance,
    pairwise_from_utilities,
    rank_of,
    utilities_to_contrasts,
    utility_covariance_from_contrast_covariance,
)


EXPERIMENT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_temporal_arbitration_20260729")
DEFAULT_INPUT_ROOT = EXPERIMENT_ROOT / "input_snapshot_20260729"

MODELS = ("Regsim", "Ensemble", "HL20")
ID_TO_NAME = {
    "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708": "Regsim",
    "ensemble_rankavg_baseline_hl20_labeltop20_20260626": "Ensemble",
    "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625": "HL20",
}
NAME_TO_ID = {name: model_id for model_id, name in ID_TO_NAME.items()}
RETURN_FILES = {
    "Regsim": "Challenger_Regsim.csv",
    "Ensemble": "Challenger_Ensemble.csv",
    "HL20": "Champion_HL20.csv",
}

PATH_PREFIXES = (
    "full0935_1430",
    "seg0935_1000",
    "seg1000_1030",
    "seg1030_1100",
    "seg1100_1130",
    "seg1300_1330",
    "seg1330_1400",
    "seg1400_1430",
)
FEATURES = tuple(
    f"{prefix}_{suffix}"
    for prefix in PATH_PREFIXES
    for suffix in ("mean", "std", "iqr", "pos_ratio", "tail_spread")
) + ("trend_accel_median", "trend_accel_mean", "dispersion_accel", "tail_balance_full")
PAIR_INDICES = ((0, 1), (0, 2), (1, 2))
VALID_ROBUST_REASONS = {"score_best", "margin_default"}


# This plan is intentionally fixed in source: the first pass is a comparison,
# not a same-sample hyperparameter search.
FROZEN_PLAN: dict[str, Any] = {
    "schema_version": 1,
    "models": list(MODELS),
    "label_contract": {
        "source": "frozen standalone strict shadow day_return histories",
        "justification": "active strategy01 turnover_ratio=1.0; standalone daily shadow return equals selector counterfactual",
        "economic_floor": 0.0,
    },
    "base_reconstruction": {
        "implementation": "cbond_on.infra.live.model_switch.decide_scoreopt_t1430_dispersion",
        "lookback_days": 60,
        "nearest_k": 40,
        "min_periods": 40,
        "score_mode": "trim20_lcb10",
        "margin_diagnostic_only": 0.0005,
    },
    "primary_intervention_envelope": {
        "base_reason": "margin_default",
        "robust_reason_in": sorted(VALID_ROBUST_REASONS),
        "description": "the 239 historical dates where the prior live fusion could numerically consult Robust",
    },
    "cross_arbitration": {
        "rule": "override only if LCB(combined direct expected r-b) > economic_floor",
        "z_value": 1.0,
        "economic_floor": 0.0,
        "scale_floor": 0.0001,
        "base_uncertainty": "winsor10/90 paired-neighbour std / sqrt(n), floor 1bp",
        "temporal_uncertainty": "max(intrinsic forecast scale, prior 120 direct prediction RMSE); Ridge requires prior calibration",
        "calibration_lookback": 120,
        "calibration_min_periods": 60,
    },
    "forecasters": {
        "current_pairwise_ridge_direct": {
            "features": "path_full_t1430_44",
            "lookback_days": 360,
            "min_periods": 120,
            "alpha": 100.0,
            "target_clip": 0.0075,
            "ranking": "legacy independent pairwise utility aggregation",
        },
        "coherent_two_contrast_ridge_direct": {
            "features": "path_full_t1430_44",
            "lookback_days": 360,
            "min_periods": 120,
            "alpha": 100.0,
            "target_clip": 0.0075,
            "ranking": "coherent Helmert utility",
        },
        "ewma_two_contrast": {
            "warmup_days": 120,
            "half_life_days": 60.0,
            "target_clip": 0.0075,
        },
        "fixed_share_hedge": {
            "warmup_days": 120,
            "target_clip": 0.0075,
            "eta": math.sqrt(2.0 * math.log(3.0) / 120.0),
            "share": 1.0 / 60.0,
            "direct_edge_estimator": "EWMA two-contrast, half-life 60",
        },
        "local_level_two_contrast": {
            "warmup_days": 120,
            "process_ratio": 0.02,
            "covariance_shrinkage": 0.25,
            "std_floor": 0.0001,
        },
        "dma_local_level_two_contrast": {
            "warmup_days": 120,
            "process_ratios": [0.0, 0.02, 0.10],
            "forgetting": 0.99,
            "covariance_shrinkage": 0.25,
            "std_floor": 0.0001,
        },
    },
    "known_limitations": [
        "state artifact was built with <=14:30 and is not strict-14:29 PIT certified",
        "the historical window has informed prior research; this is not a final untouched holdout",
        "no partial-turnover or path-dependent switch-cost contract is introduced",
    ],
}


@dataclass(frozen=True)
class BaseContext:
    score_day: date
    base_name: str
    base_reason: str
    base_score_gap: float | None
    scores: np.ndarray | None
    neighbour_returns: np.ndarray | None
    history_days: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: object) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if math.isfinite(output) else None


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _safe_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else f"git_error:{result.stderr.strip()}"


def _assert_experiment_output_root(output_root: Path) -> Path:
    allowed = EXPERIMENT_ROOT.resolve()
    output = output_root.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError(f"output root must stay under {allowed}; received {output}")
    return output


def _read_return_panel(input_root: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    panel: pd.DataFrame | None = None
    paths: dict[str, Path] = {}
    for name in MODELS:
        path = input_root / RETURN_FILES[name]
        if not path.exists():
            raise FileNotFoundError(f"frozen return input missing: {path}")
        frame = pd.read_csv(path)
        required = {"trade_date", "day_return"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"return input {path} missing columns {missing}")
        frame = frame[["trade_date", "day_return"]].copy()
        frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
        frame[name] = pd.to_numeric(frame["day_return"], errors="coerce")
        frame = frame[["score_day", name]].dropna(subset=["score_day", name]).sort_values("score_day")
        if frame["score_day"].duplicated().any():
            duplicates = frame.loc[frame["score_day"].duplicated(keep=False), "score_day"].astype(str).tolist()
            raise ValueError(f"return input {path} has duplicate dates: {duplicates[:10]}")
        panel = frame if panel is None else panel.merge(frame, on="score_day", how="inner", validate="one_to_one")
        paths[name] = path
    if panel is None or panel.empty:
        raise RuntimeError("no complete return panel could be formed")
    return panel.sort_values("score_day").reset_index(drop=True), paths


def _read_state(input_root: Path) -> tuple[pd.DataFrame, Path]:
    path = input_root / "t1430_market_state_features_path_full.csv"
    if not path.exists():
        raise FileNotFoundError(f"frozen state input missing: {path}")
    frame = pd.read_csv(path, usecols=["trade_date", *FEATURES])
    frame["score_day"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame = frame[["score_day", *FEATURES]].dropna(subset=["score_day"]).sort_values("score_day")
    if frame["score_day"].duplicated().any():
        duplicates = frame.loc[frame["score_day"].duplicated(keep=False), "score_day"].astype(str).tolist()
        raise ValueError(f"state input has duplicate dates: {duplicates[:10]}")
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    frame["_state_present"] = True
    return frame.reset_index(drop=True), path


def _read_current(input_root: Path) -> tuple[pd.DataFrame, Path]:
    path = input_root / "daily_current.csv"
    if not path.exists():
        raise FileNotFoundError(f"frozen current-routing input missing: {path}")
    frame = pd.read_csv(path)
    required = {
        "score_day",
        "selected_model_id",
        "selected_return",
        "base_model_id",
        "base_reason",
        "base_score_gap",
        "robust_reason",
        "robust_score_gap",
        "reason",
        "action",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"daily_current missing columns {missing}")
    frame["score_day"] = pd.to_datetime(frame["score_day"], errors="coerce").dt.date
    frame = frame.dropna(subset=["score_day"]).sort_values("score_day").reset_index(drop=True)
    if frame["score_day"].duplicated().any():
        duplicates = frame.loc[frame["score_day"].duplicated(keep=False), "score_day"].astype(str).tolist()
        raise ValueError(f"daily_current has duplicate dates: {duplicates[:10]}")
    frame["current_name"] = frame["selected_model_id"].map(ID_TO_NAME)
    frame["base_name"] = frame["base_model_id"].map(ID_TO_NAME)
    unknown = frame.loc[frame[["current_name", "base_name"]].isna().any(axis=1), ["score_day", "selected_model_id", "base_model_id"]]
    if not unknown.empty:
        raise ValueError(f"daily_current has unmapped model ids:\n{unknown.to_string(index=False)}")
    return frame, path


def _load_frozen_configs(input_root: Path, return_paths: Mapping[str, Path], state_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    live_path = input_root / "live_config.json5"
    strategy_path = input_root / "strategy01_config.json5"
    if not live_path.exists() or not strategy_path.exists():
        raise FileNotFoundError("frozen live/strategy configuration is incomplete")
    live_cfg = load_config_file(live_path)
    strategy_cfg = load_config_file(strategy_path)
    turnover_ratio = _finite_or_none(strategy_cfg.get("turnover_ratio"))
    if turnover_ratio is None or not math.isclose(turnover_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"shadow-return label contract requires turnover_ratio=1.0; observed {turnover_ratio}")
    switch_cfg = copy.deepcopy(live_cfg.get("model_switch", {}))
    if str(switch_cfg.get("mode", "")).strip() != "scoreopt_t1430_fusion_gate":
        raise RuntimeError(f"unexpected frozen live model switch mode: {switch_cfg.get('mode')}")
    switch_cfg["state_feature_path"] = str(state_path)

    groups = [switch_cfg.get("champion"), *(switch_cfg.get("challengers") or [])]
    if len(groups) != 3 or any(not isinstance(group, dict) for group in groups):
        raise RuntimeError("frozen live model switch does not describe exactly three candidate groups")
    observed_names: list[str] = []
    for group in groups:
        assert isinstance(group, dict)
        model_id = str(group.get("model_id", "")).strip()
        name = ID_TO_NAME.get(model_id)
        if name is None:
            raise RuntimeError(f"frozen live config has unexpected model id: {model_id}")
        group["return_path"] = str(return_paths[name])
        observed_names.append(name)
    if set(observed_names) != set(MODELS):
        raise RuntimeError(f"frozen live config model set differs from expected {MODELS}: {observed_names}")
    challenger = switch_cfg.get("challenger")
    if isinstance(challenger, dict):
        model_id = str(challenger.get("model_id", "")).strip()
        if model_id in ID_TO_NAME:
            challenger["return_path"] = str(return_paths[ID_TO_NAME[model_id]])
    return live_cfg, strategy_cfg, switch_cfg


def _base_contexts(
    current: pd.DataFrame,
    *,
    switch_cfg: dict[str, Any],
    strict: bool,
) -> tuple[dict[date, BaseContext], pd.DataFrame, dict[str, Any]]:
    """Rebuild production Base, including all three scores and similar days."""

    contexts: dict[date, BaseContext] = {}
    records: list[dict[str, Any]] = []
    mismatches: list[str] = []
    gap_tolerance = 1e-12
    for row in current.itertuples(index=False):
        score_day = row.score_day
        decision = decide_scoreopt_t1430_dispersion(switch_cfg, score_day=score_day)
        score_map: dict[str, float | None] = {name: None for name in MODELS}
        for detail in decision.candidate_scores or []:
            name = ID_TO_NAME.get(str(detail.get("model_id", "")).strip())
            if name is not None:
                score_map[name] = _finite_or_none(detail.get("score"))
        scores = None
        if all(score_map[name] is not None for name in MODELS):
            scores = np.asarray([float(score_map[name]) for name in MODELS], dtype=float)

        neighbours: list[list[float]] = []
        neighbour_days: list[str] = []
        for similar in decision.similar_days or []:
            day_values: dict[str, float] = {}
            for value in similar.get("model_returns", []):
                name = ID_TO_NAME.get(str(value.get("model_id", "")).strip())
                returned = _finite_or_none(value.get("day_return"))
                if name is not None and returned is not None:
                    day_values[name] = returned
            if len(day_values) == len(MODELS):
                neighbours.append([day_values[name] for name in MODELS])
                neighbour_days.append(str(similar.get("trade_date")))
        neighbour_returns = np.asarray(neighbours, dtype=float) if neighbours else None

        expected_gap = _finite_or_none(row.base_score_gap)
        actual_gap = _finite_or_none(decision.score_diff)
        id_match = str(decision.selected_model_id) == str(row.base_model_id)
        reason_match = str(decision.reason) == str(row.base_reason)
        gap_match = (expected_gap is None and actual_gap is None) or (
            expected_gap is not None and actual_gap is not None and abs(expected_gap - actual_gap) <= gap_tolerance
        )
        if not (id_match and reason_match and gap_match):
            mismatches.append(
                f"{score_day}: expected id/reason/gap={row.base_model_id}/{row.base_reason}/{expected_gap}; "
                f"actual={decision.selected_model_id}/{decision.reason}/{actual_gap}"
            )

        base_name = ID_TO_NAME.get(str(decision.selected_model_id))
        if base_name is None:
            raise RuntimeError(f"production Base returned an unexpected model id on {score_day}: {decision.selected_model_id}")
        contexts[score_day] = BaseContext(
            score_day=score_day,
            base_name=base_name,
            base_reason=str(decision.reason),
            base_score_gap=actual_gap,
            scores=scores,
            neighbour_returns=neighbour_returns,
            history_days=int(decision.history_days),
        )
        record: dict[str, Any] = {
            "score_day": score_day,
            "base_name": base_name,
            "base_model_id": decision.selected_model_id,
            "base_reason": decision.reason,
            "base_score_gap": actual_gap,
            "base_history_days": int(decision.history_days),
            "base_history_end": decision.history_end,
            "base_reproduction_id_match": id_match,
            "base_reproduction_reason_match": reason_match,
            "base_reproduction_gap_match": gap_match,
            "neighbour_count": int(len(neighbours)),
            "neighbour_score_days": json.dumps(neighbour_days, ensure_ascii=False),
        }
        for index, name in enumerate(MODELS):
            record[f"base_score_{name}"] = None if scores is None else float(scores[index])
        records.append(record)

    if mismatches and strict:
        preview = "\n".join(mismatches[:12])
        raise RuntimeError(f"strict Base reproduction failed on {len(mismatches)} days:\n{preview}")
    return contexts, pd.DataFrame(records), {
        "checked_days": int(len(records)),
        "mismatch_days": int(len(mismatches)),
        "mismatches_preview": mismatches[:12],
        "gap_tolerance": gap_tolerance,
    }


def _base_pair_scale(context: BaseContext, *, candidate_index: int, base_index: int) -> tuple[float | None, int]:
    if context.neighbour_returns is None or len(context.neighbour_returns) < 2:
        return None, 0
    delta = context.neighbour_returns[:, candidate_index] - context.neighbour_returns[:, base_index]
    if not np.isfinite(delta).all() or len(delta) < 2:
        return None, 0
    lower, upper = np.quantile(delta, [0.10, 0.90])
    winsorized = np.clip(delta, lower, upper)
    scale = float(np.std(winsorized, ddof=1) / math.sqrt(len(winsorized)))
    return (scale if math.isfinite(scale) else None), int(len(winsorized))


def _feature_history(panel: pd.DataFrame, score_day: date, *, lookback: int) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    current = panel.loc[panel["score_day"] == score_day]
    if current.empty or not bool(current.iloc[-1].get("_state_present", False)):
        return None, None, None, 0
    current_x = current.iloc[-1][list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(current_x).all():
        return None, None, None, 0
    history = panel.loc[(panel["score_day"] < score_day) & panel["_state_present"].fillna(False)].sort_values("score_day").tail(lookback)
    valid = np.isfinite(history[list(FEATURES)].to_numpy(dtype=float)).all(axis=1) & np.isfinite(history[list(MODELS)].to_numpy(dtype=float)).all(axis=1)
    history = history.loc[valid]
    if history.empty:
        return current_x, np.empty((0, len(FEATURES))), np.empty((0, 3)), 0
    return (
        current_x,
        history[list(FEATURES)].to_numpy(dtype=float),
        history[list(MODELS)].to_numpy(dtype=float),
        int(len(history)),
    )


def _normalize_features(train_x: np.ndarray, current_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.mean(train_x, axis=0)
    stds = np.std(train_x, axis=0, ddof=0)
    stds = np.where((~np.isfinite(stds)) | (stds < 1e-12), 1.0, stds)
    return (train_x - means) / stds, (current_x.reshape(1, -1) - means) / stds


def _ridge_forecasts(panel: pd.DataFrame, score_day: date) -> dict[str, TemporalForecast]:
    """Fit the frozen legacy and coherent Ridge variants using only date<t."""

    target_clip = float(FROZEN_PLAN["forecasters"]["current_pairwise_ridge_direct"]["target_clip"])
    lookback = int(FROZEN_PLAN["forecasters"]["current_pairwise_ridge_direct"]["lookback_days"])
    min_periods = int(FROZEN_PLAN["forecasters"]["current_pairwise_ridge_direct"]["min_periods"])
    alpha = float(FROZEN_PLAN["forecasters"]["current_pairwise_ridge_direct"]["alpha"])
    current_x, history_x, history_returns, history_days = _feature_history(panel, score_day, lookback=lookback)
    if current_x is None or history_x is None or history_returns is None or history_days < min_periods:
        return {}
    train_x, prediction_x = _normalize_features(history_x, current_x)

    pair_mean = np.zeros((3, 3), dtype=float)
    legacy_utilities = np.zeros(3, dtype=float)
    for left, right in PAIR_INDICES:
        target = np.clip(history_returns[:, left] - history_returns[:, right], -target_clip, target_clip)
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(train_x, target)
        prediction = float(estimator.predict(prediction_x)[0])
        pair_mean[left, right] = prediction
        pair_mean[right, left] = -prediction
        legacy_utilities[left] += prediction / 3.0
        legacy_utilities[right] -= prediction / 3.0

    utilities_history = np.asarray([joint_clip_utilities(values, target_clip=target_clip)[0] for values in history_returns], dtype=float)
    contrast_targets = utilities_history @ HELMERT.T
    contrast_prediction: list[float] = []
    for target_index in range(2):
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(train_x, contrast_targets[:, target_index])
        contrast_prediction.append(float(estimator.predict(prediction_x)[0]))
    coherent_utilities = contrasts_to_utilities(np.asarray(contrast_prediction, dtype=float))
    return {
        "current_pairwise_ridge_direct": make_utility_forecast(
            "current_pairwise_ridge_direct",
            legacy_utilities,
            pair_mean=pair_mean,
            history_days=history_days,
            ready=True,
            diagnostics={"feature_count": len(FEATURES), "alpha": alpha, "target_clip": target_clip},
        ),
        "coherent_two_contrast_ridge_direct": make_utility_forecast(
            "coherent_two_contrast_ridge_direct",
            coherent_utilities,
            history_days=history_days,
            ready=True,
            diagnostics={"feature_count": len(FEATURES), "alpha": alpha, "target_clip": target_clip},
        ),
    }


def _time_series_forecasts(history_utilities: np.ndarray) -> dict[str, TemporalForecast]:
    """Generate non-feature temporal forecasts from realised history only."""

    plans = FROZEN_PLAN["forecasters"]
    warmup = int(plans["ewma_two_contrast"]["warmup_days"])
    if len(history_utilities) < warmup:
        return {}
    target_clip = float(plans["ewma_two_contrast"]["target_clip"])
    half_life = float(plans["ewma_two_contrast"]["half_life_days"])
    ewma_mean, ewma_covariance, effective_n = ewma_utility_forecast(history_utilities, half_life=half_life)
    ewma_scale = pair_scale_from_utility_covariance(ewma_covariance)
    results: dict[str, TemporalForecast] = {
        "ewma_two_contrast": make_utility_forecast(
            "ewma_two_contrast",
            ewma_mean,
            pair_scale=ewma_scale,
            history_days=len(history_utilities),
            ready=effective_n >= 40.0,
            diagnostics={"half_life_days": half_life, "effective_n": effective_n},
        )
    }

    hedge_plan = plans["fixed_share_hedge"]
    hedge_weights = fixed_share_hedge_weights(
        history_utilities,
        target_clip=float(hedge_plan["target_clip"]),
        eta=float(hedge_plan["eta"]),
        share=float(hedge_plan["share"]),
    )
    results["fixed_share_hedge"] = make_utility_forecast(
        "fixed_share_hedge",
        hedge_weights,
        pair_mean=pairwise_from_utilities(ewma_mean),
        pair_scale=ewma_scale,
        history_days=len(history_utilities),
        ready=effective_n >= 40.0,
        diagnostics={
            "eta": float(hedge_plan["eta"]),
            "share": float(hedge_plan["share"]),
            "weight_Regsim": float(hedge_weights[0]),
            "weight_Ensemble": float(hedge_weights[1]),
            "weight_HL20": float(hedge_weights[2]),
            "direct_edge_effective_n": effective_n,
        },
    )

    history_contrasts = history_utilities @ HELMERT.T
    local_plan = plans["local_level_two_contrast"]
    local_mean, local_covariance, local_diag = local_level_forecast(
        history_contrasts,
        process_ratio=float(local_plan["process_ratio"]),
        shrinkage=float(local_plan["covariance_shrinkage"]),
        std_floor=float(local_plan["std_floor"]),
    )
    local_utilities = contrasts_to_utilities(local_mean)
    local_utility_covariance = utility_covariance_from_contrast_covariance(local_covariance)
    results["local_level_two_contrast"] = make_utility_forecast(
        "local_level_two_contrast",
        local_utilities,
        pair_scale=pair_scale_from_utility_covariance(local_utility_covariance),
        history_days=len(history_utilities),
        ready=True,
        diagnostics=local_diag,
    )

    dma_plan = plans["dma_local_level_two_contrast"]
    dma_mean, dma_covariance, dma_weights, dma_diag = dma_local_level_forecast(
        history_contrasts,
        process_ratios=tuple(float(value) for value in dma_plan["process_ratios"]),
        forgetting=float(dma_plan["forgetting"]),
        shrinkage=float(dma_plan["covariance_shrinkage"]),
        std_floor=float(dma_plan["std_floor"]),
    )
    dma_utilities = contrasts_to_utilities(dma_mean)
    dma_utility_covariance = utility_covariance_from_contrast_covariance(dma_covariance)
    results["dma_local_level_two_contrast"] = make_utility_forecast(
        "dma_local_level_two_contrast",
        dma_utilities,
        pair_scale=pair_scale_from_utility_covariance(dma_utility_covariance),
        history_days=len(history_utilities),
        ready=True,
        diagnostics={
            **dma_diag,
            "weight_q0": float(dma_weights[0]),
            "weight_q002": float(dma_weights[1]),
            "weight_q010": float(dma_weights[2]),
        },
    )
    return results


def _calibration_scale(errors: list[float], *, lookback: int, min_periods: int) -> tuple[float | None, int]:
    recent = np.asarray(errors[-lookback:], dtype=float)
    recent = recent[np.isfinite(recent)]
    if len(recent) < min_periods:
        return None, int(len(recent))
    scale = float(math.sqrt(float(np.mean(recent**2))))
    return (scale if math.isfinite(scale) else None), int(len(recent))


def _pair_label(left: int, right: int) -> str:
    return f"{MODELS[left]}_minus_{MODELS[right]}"


def _forecast_row(score_day: date, forecaster_id: str, forecast: TemporalForecast | None, actual_utilities: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {
        "score_day": score_day,
        "forecaster_id": forecaster_id,
        "prediction_ready": bool(forecast is not None and forecast.ready),
        "history_days": None if forecast is None else forecast.history_days,
        "forecast_top_name": None,
        "forecast_top_model_id": None,
        "forecast_top_gap_diagnostic": None,
    }
    for index, name in enumerate(MODELS):
        row[f"ranking_score_{name}"] = None if forecast is None else float(forecast.utilities[index])
    for left, right in PAIR_INDICES:
        label = _pair_label(left, right)
        row[f"actual_clipped_{label}"] = float(actual_utilities[left] - actual_utilities[right])
        row[f"pred_{label}"] = None if forecast is None else float(forecast.pair_mean[left, right])
        scale = None if forecast is None else _finite_or_none(forecast.pair_scale[left, right])
        row[f"intrinsic_scale_{label}"] = scale
    if forecast is not None:
        ranking = ordered_indices(forecast.utilities)
        row["forecast_top_name"] = MODELS[ranking[0]]
        row["forecast_top_model_id"] = NAME_TO_ID[MODELS[ranking[0]]]
        row["forecast_top_gap_diagnostic"] = float(forecast.utilities[ranking[0]] - forecast.utilities[ranking[1]])
        for key, value in forecast.diagnostics.items():
            row[f"diagnostic_{key}"] = value
    return row


def _empty_arbitration_row(
    *,
    score_day: date,
    forecaster_id: str,
    base_name: str,
    current_name: str,
    eligible: bool,
    selected_name: str,
    reason: str,
    action: str,
    selected_return: float,
    base_return: float,
    current_return: float,
    current_row: Any,
) -> dict[str, Any]:
    return {
        "score_day": score_day,
        "forecaster_id": forecaster_id,
        "route_eligible": eligible,
        "base_name": base_name,
        "base_model_id": NAME_TO_ID[base_name],
        "current_name": current_name,
        "current_model_id": NAME_TO_ID[current_name],
        "selected_name": selected_name,
        "selected_model_id": NAME_TO_ID[selected_name],
        "action": action,
        "decision_reason": reason,
        "candidate_name": None,
        "candidate_model_id": None,
        "candidate_equals_base": None,
        "base_score_candidate_minus_base": None,
        "base_pair_scale": None,
        "base_pair_sample_n": 0,
        "temporal_mean_candidate_minus_base": None,
        "temporal_scale": None,
        "temporal_calibration_scale": None,
        "temporal_calibration_n": 0,
        "combined_mean": None,
        "combined_scale": None,
        "lcb": None,
        "base_rank_of_candidate": None,
        "forecast_rank_of_base": None,
        "forecast_rank_of_candidate": None,
        "third_name": None,
        "base_score_third_minus_base": None,
        "temporal_mean_third_minus_base": None,
        "temporal_mean_candidate_minus_third": None,
        "current_robust_score_gap_diagnostic": _finite_or_none(current_row.robust_score_gap),
        "current_robust_reason": str(current_row.robust_reason),
        "selected_return": selected_return,
        "base_return": base_return,
        "current_return": current_return,
        "candidate_return": None,
        "candidate_return_delta_vs_base": None,
        "current_return_delta_vs_base": current_return - base_return,
        "return_delta_vs_base": selected_return - base_return,
        "return_delta_vs_current": selected_return - current_return,
        "current_overrode_base": bool(eligible and current_name != base_name),
        "override_base": bool(eligible and selected_name != base_name),
        "changed_vs_current": bool(selected_name != current_name),
    }


def _arbitration_record(
    *,
    score_day: date,
    forecaster_id: str,
    context: BaseContext,
    base_index: int,
    current_name: str,
    current_index: int,
    eligible: bool,
    forecast: TemporalForecast | None,
    calibration_errors: Mapping[tuple[int, int], list[float]],
    returns: np.ndarray,
    current_row: Any,
) -> dict[str, Any]:
    base_name = context.base_name
    base_return = float(returns[base_index])
    current_return = float(returns[current_index])
    if not eligible:
        return _empty_arbitration_row(
            score_day=score_day,
            forecaster_id=forecaster_id,
            base_name=base_name,
            current_name=current_name,
            eligible=False,
            selected_name=current_name,
            reason="outside_frozen_intervention_envelope",
            action="copy_current_routing",
            selected_return=current_return,
            base_return=base_return,
            current_return=current_return,
            current_row=current_row,
        )
    if forecast is None or not forecast.ready:
        return _empty_arbitration_row(
            score_day=score_day,
            forecaster_id=forecaster_id,
            base_name=base_name,
            current_name=current_name,
            eligible=True,
            selected_name=base_name,
            reason="temporal_forecast_unavailable",
            action="keep_base",
            selected_return=base_return,
            base_return=base_return,
            current_return=current_return,
            current_row=current_row,
        )

    candidate_index = int(ordered_indices(forecast.utilities, preferred_index=base_index)[0])
    if candidate_index == base_index:
        calibration, calibration_n = None, 0
        pair_scale, pair_n = None, 0
    else:
        calibration, calibration_n = _calibration_scale(
            calibration_errors[(min(candidate_index, base_index), max(candidate_index, base_index))],
            lookback=int(FROZEN_PLAN["cross_arbitration"]["calibration_lookback"]),
            min_periods=int(FROZEN_PLAN["cross_arbitration"]["calibration_min_periods"]),
        )
        pair_scale, pair_n = _base_pair_scale(context, candidate_index=candidate_index, base_index=base_index)
    decision = cross_arbitrate(
        base_index=base_index,
        base_scores=context.scores,
        base_pair_scale=pair_scale,
        forecast=forecast,
        temporal_calibration_scale=calibration,
        z_value=float(FROZEN_PLAN["cross_arbitration"]["z_value"]),
        economic_floor=float(FROZEN_PLAN["cross_arbitration"]["economic_floor"]),
        scale_floor=float(FROZEN_PLAN["cross_arbitration"]["scale_floor"]),
    )
    selected_name = MODELS[decision.selected_index]
    selected_return = float(returns[decision.selected_index])
    candidate_return = None if decision.candidate_index is None else float(returns[decision.candidate_index])
    return {
        "score_day": score_day,
        "forecaster_id": forecaster_id,
        "route_eligible": True,
        "base_name": base_name,
        "base_model_id": NAME_TO_ID[base_name],
        "current_name": current_name,
        "current_model_id": NAME_TO_ID[current_name],
        "selected_name": selected_name,
        "selected_model_id": NAME_TO_ID[selected_name],
        "action": decision.action,
        "decision_reason": decision.reason,
        "candidate_name": None if decision.candidate_index is None else MODELS[decision.candidate_index],
        "candidate_model_id": None if decision.candidate_index is None else NAME_TO_ID[MODELS[decision.candidate_index]],
        "candidate_equals_base": None if decision.candidate_index is None else bool(decision.candidate_index == base_index),
        "base_score_candidate_minus_base": decision.base_score_candidate_minus_base,
        "base_pair_scale": decision.base_pair_scale,
        "base_pair_sample_n": pair_n,
        "temporal_mean_candidate_minus_base": decision.temporal_mean_candidate_minus_base,
        "temporal_scale": decision.temporal_scale,
        "temporal_calibration_scale": decision.temporal_calibration_scale,
        "temporal_calibration_n": calibration_n,
        "combined_mean": decision.combined_mean,
        "combined_scale": decision.combined_scale,
        "lcb": decision.lcb,
        "base_rank_of_candidate": decision.base_rank_of_candidate,
        "forecast_rank_of_base": decision.forecast_rank_of_base,
        "forecast_rank_of_candidate": decision.forecast_rank_of_candidate,
        "third_name": None if decision.third_index is None else MODELS[decision.third_index],
        "base_score_third_minus_base": decision.base_score_third_minus_base,
        "temporal_mean_third_minus_base": decision.temporal_mean_third_minus_base,
        "temporal_mean_candidate_minus_third": decision.temporal_mean_candidate_minus_third,
        "current_robust_score_gap_diagnostic": _finite_or_none(current_row.robust_score_gap),
        "current_robust_reason": str(current_row.robust_reason),
        "selected_return": selected_return,
        "base_return": base_return,
        "current_return": current_return,
        "candidate_return": candidate_return,
        "candidate_return_delta_vs_base": None if candidate_return is None else candidate_return - base_return,
        "current_return_delta_vs_base": current_return - base_return,
        "return_delta_vs_base": selected_return - base_return,
        "return_delta_vs_current": selected_return - current_return,
        "current_overrode_base": bool(current_name != base_name),
        "override_base": bool(selected_name != base_name),
        "changed_vs_current": bool(selected_name != current_name),
    }


def _run_replay(panel: pd.DataFrame, current: pd.DataFrame, contexts: Mapping[date, BaseContext]) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = current.merge(panel, on="score_day", how="inner", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    if len(aligned) != len(current):
        missing = sorted(set(current["score_day"]) - set(aligned["score_day"]))
        raise RuntimeError(f"current routing and return panel do not align: missing {missing[:10]}")
    expected_current = np.asarray(
        [float(getattr(row, row.current_name)) for row in aligned.itertuples(index=False)],
        dtype=float,
    )
    current_returns = pd.to_numeric(aligned["selected_return"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(current_returns).all() or not np.allclose(expected_current, current_returns, atol=1e-12, rtol=0.0):
        raise RuntimeError("frozen standalone returns do not reproduce frozen current routing")

    forecaster_ids = list(FROZEN_PLAN["forecasters"].keys())
    calibration_errors: dict[str, dict[tuple[int, int], list[float]]] = {
        forecaster: {pair: [] for pair in PAIR_INDICES} for forecaster in forecaster_ids
    }
    history_utilities: list[np.ndarray] = []
    forecast_records: list[dict[str, Any]] = []
    arbitration_records: list[dict[str, Any]] = []

    for row in aligned.itertuples(index=False):
        score_day = row.score_day
        values = np.asarray([float(getattr(row, name)) for name in MODELS], dtype=float)
        actual_utilities, actual_scale = joint_clip_utilities(
            values,
            target_clip=float(FROZEN_PLAN["forecasters"]["current_pairwise_ridge_direct"]["target_clip"]),
        )
        forecasts = _ridge_forecasts(panel, score_day)
        forecasts.update(_time_series_forecasts(np.asarray(history_utilities, dtype=float)) if history_utilities else {})
        context = contexts.get(score_day)
        if context is None:
            raise RuntimeError(f"missing Base context for {score_day}")
        base_name = context.base_name
        current_name = str(row.current_name)
        base_index = MODELS.index(base_name)
        current_index = MODELS.index(current_name)
        eligible = bool(str(row.base_reason) == "margin_default" and str(row.robust_reason) in VALID_ROBUST_REASONS)

        for forecaster_id in forecaster_ids:
            forecast = forecasts.get(forecaster_id)
            forecast_records.append(_forecast_row(score_day, forecaster_id, forecast, actual_utilities))
            arbitration_records.append(
                _arbitration_record(
                    score_day=score_day,
                    forecaster_id=forecaster_id,
                    context=context,
                    base_index=base_index,
                    current_name=current_name,
                    current_index=current_index,
                    eligible=eligible,
                    forecast=forecast,
                    calibration_errors=calibration_errors[forecaster_id],
                    returns=values,
                    current_row=row,
                )
            )

        # The outcome for score_day becomes observable only after every t-day
        # forecast and decision above has been frozen.  It may affect t+1 only.
        for forecaster_id, forecast in forecasts.items():
            if not forecast.ready:
                continue
            for left, right in PAIR_INDICES:
                prediction = float(forecast.pair_mean[left, right])
                if math.isfinite(prediction):
                    actual = float(actual_utilities[left] - actual_utilities[right])
                    calibration_errors[forecaster_id][(left, right)].append(prediction - actual)
        history_utilities.append(actual_utilities)

    return pd.DataFrame(forecast_records), pd.DataFrame(arbitration_records)


def _selector_metrics(returns: pd.Series, dates: pd.Series, selected: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("selector metrics require finite returns")
    nav = np.cumprod(1.0 + values)
    volatility = float(np.std(values, ddof=1) * math.sqrt(252.0)) if len(values) > 1 else float("nan")
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * math.sqrt(252.0)) if volatility > 0.0 else float("nan")
    names = selected.astype(str).to_numpy()
    switches = int((names[1:] != names[:-1]).sum()) if len(names) > 1 else 0
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


def _conditional_metrics(delta: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return {"n": 0, "wins": 0, "losses": 0}
    wins = int((values > 0.0).sum())
    result: dict[str, Any] = {
        "n": int(len(values)),
        "wins": wins,
        "losses": int((values < 0.0).sum()),
        "mean_delta_bp": float(values.mean() * 1e4),
        "median_delta_bp": float(np.median(values) * 1e4),
        "sum_delta_bp": float(values.sum() * 1e4),
        "hit_rate": float((values > 0.0).mean()),
    }
    if len(values) > 1:
        result["paired_t_p_one_sided"] = float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue)
        result["sign_p_one_sided"] = float(stats.binomtest(wins, len(values), 0.5, alternative="greater").pvalue)
    return result


def _summary_tables(arbitration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = arbitration.loc[arbitration["forecaster_id"] == arbitration["forecaster_id"].iloc[0]].copy()
    current = current.sort_values("score_day").reset_index(drop=True)
    base_route = np.where(current["route_eligible"].astype(bool), current["base_name"], current["current_name"])
    base_returns = np.asarray(
        [float(row.base_return) if bool(row.route_eligible) else float(row.current_return) for row in current.itertuples(index=False)],
        dtype=float,
    )
    base_metrics = _selector_metrics(pd.Series(base_returns), current["score_day"], pd.Series(base_route))
    current_metrics = _selector_metrics(current["current_return"], current["score_day"], current["current_name"])
    rows: list[dict[str, Any]] = [
        {"strategy": "frozen_current_routing", **current_metrics, "route_eligible_days": int(current["route_eligible"].sum()), "override_base_days": int(current["current_overrode_base"].sum()), "changed_vs_current_days": 0, "total_return_delta_vs_current": 0.0, "sharpe_delta_vs_current": 0.0},
        {"strategy": "base_fallback_within_envelope", **base_metrics, "route_eligible_days": int(current["route_eligible"].sum()), "override_base_days": 0, "changed_vs_current_days": int((pd.Series(base_route) != current["current_name"]).sum()), "total_return_delta_vs_current": float(base_metrics["total_return"] - current_metrics["total_return"]), "sharpe_delta_vs_current": float(base_metrics["sharpe"] - current_metrics["sharpe"])},
    ]
    condition_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    forecasters = list(dict.fromkeys(arbitration["forecaster_id"].astype(str)))
    for forecaster in forecasters:
        subset = arbitration.loc[arbitration["forecaster_id"] == forecaster].sort_values("score_day").reset_index(drop=True)
        metrics = _selector_metrics(subset["selected_return"], subset["score_day"], subset["selected_name"])
        rows.append(
            {
                "strategy": forecaster,
                **metrics,
                "route_eligible_days": int(subset["route_eligible"].sum()),
                "override_base_days": int(subset["override_base"].sum()),
                "changed_vs_current_days": int(subset["changed_vs_current"].sum()),
                "total_return_delta_vs_current": float(metrics["total_return"] - current_metrics["total_return"]),
                "sharpe_delta_vs_current": float(metrics["sharpe"] - current_metrics["sharpe"]),
            }
        )
        for condition, mask, column in (
            ("override_base", subset["route_eligible"].astype(bool) & subset["override_base"].astype(bool), "return_delta_vs_base"),
            ("changed_vs_current", subset["changed_vs_current"].astype(bool), "return_delta_vs_current"),
            ("all_eligible", subset["route_eligible"].astype(bool), "return_delta_vs_current"),
        ):
            condition_rows.append({"forecaster_id": forecaster, "condition": condition, **_conditional_metrics(subset.loc[mask, column])})

        for part, indices in enumerate(np.array_split(np.arange(len(subset)), 3), start=1):
            if not len(indices):
                continue
            window = subset.iloc[indices]
            window_metrics = _selector_metrics(window["selected_return"], window["score_day"], window["selected_name"])
            current_window = _selector_metrics(window["current_return"], window["score_day"], window["current_name"])
            slice_rows.append(
                {
                    "forecaster_id": forecaster,
                    "slice": f"chronological_{part}",
                    **window_metrics,
                    "route_eligible_days": int(window["route_eligible"].sum()),
                    "override_base_days": int(window["override_base"].sum()),
                    "total_return_delta_vs_current": float(window_metrics["total_return"] - current_window["total_return"]),
                    "sharpe_delta_vs_current": float(window_metrics["sharpe"] - current_window["sharpe"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(condition_rows), pd.DataFrame(slice_rows)


def _prediction_diagnostics(forecasts: pd.DataFrame, arbitration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    for forecaster, group in forecasts.groupby("forecaster_id", sort=True):
        for left, right in PAIR_INDICES:
            label = _pair_label(left, right)
            prediction = pd.to_numeric(group[f"pred_{label}"], errors="coerce")
            actual = pd.to_numeric(group[f"actual_clipped_{label}"], errors="coerce")
            valid = group["prediction_ready"].astype(bool) & prediction.notna() & actual.notna()
            y_pred = prediction.loc[valid].to_numpy(dtype=float)
            y_true = actual.loc[valid].to_numpy(dtype=float)
            row: dict[str, Any] = {"forecaster_id": forecaster, "pair": label, "n": int(len(y_true))}
            if len(y_true) >= 2:
                row.update(
                    {
                        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan"),
                        "oos_r2": float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)) if np.sum((y_true - y_true.mean()) ** 2) > 0.0 else float("nan"),
                        "direction_hit_rate": float(np.mean(np.sign(y_pred) == np.sign(y_true))) if np.any(np.abs(y_true) > 1e-15) else float("nan"),
                        "mae_bp": float(np.mean(np.abs(y_true - y_pred)) * 1e4),
                        "rmse_bp": float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))) * 1e4),
                    }
                )
            prediction_rows.append(row)

    calibration_rows: list[dict[str, Any]] = []
    for forecaster, group in arbitration.groupby("forecaster_id", sort=True):
        direct = group.loc[
            group["route_eligible"].astype(bool)
            & group["candidate_name"].notna()
            & ~group["candidate_equals_base"].fillna(False)
            & group["temporal_mean_candidate_minus_base"].notna()
        ].copy()
        for condition, subset in (("direct_candidate_cases", direct), ("cross_lcb_override", direct.loc[direct["action"] == "override_base"])):
            actual = pd.to_numeric(subset["candidate_return_delta_vs_base"], errors="coerce").dropna()
            lcb = pd.to_numeric(subset["lcb"], errors="coerce")
            calibration_rows.append(
                {
                    "forecaster_id": forecaster,
                    "condition": condition,
                    "n": int(len(subset)),
                    "actual_positive_rate": float((actual > 0.0).mean()) if len(actual) else float("nan"),
                    "mean_actual_delta_bp": float(actual.mean() * 1e4) if len(actual) else float("nan"),
                    "mean_lcb_bp": float(lcb.mean() * 1e4) if lcb.notna().any() else float("nan"),
                    "lcb_available_n": int(lcb.notna().sum()),
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(calibration_rows)


def _input_manifest(
    *,
    input_root: Path,
    return_paths: Mapping[str, Path],
    state_path: Path,
    current_path: Path,
    return_panel: pd.DataFrame,
    state: pd.DataFrame,
    current: pd.DataFrame,
    base_reproduction: Mapping[str, Any],
    live_cfg: Mapping[str, Any],
    strategy_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    files = [current_path, state_path, *(return_paths[name] for name in MODELS), input_root / "live_config.json5", input_root / "strategy01_config.json5"]
    source_hashes_path = input_root / "snapshot_hashes.json"
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8-sig")) if source_hashes_path.exists() else None
    source_files = [
        REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py",
        REPO_ROOT / "harness" / "model_switch_temporal_arbitration.py",
        Path(__file__).resolve(),
    ]
    return {
        "run_class": "research_only_offline_temporal_arbitration",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "input_root": str(input_root),
        "input_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": int(path.stat().st_size)} for path in files
        ],
        "snapshot_source_hashes": source_hashes,
        "source_code": [{"path": str(path), "sha256": sha256(path)} for path in source_files],
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
        "models": {name: NAME_TO_ID[name] for name in MODELS},
        "date_coverage": {
            "returns": {"rows": int(len(return_panel)), "start": str(return_panel["score_day"].min()), "end": str(return_panel["score_day"].max())},
            "state": {"rows": int(len(state)), "start": str(state["score_day"].min()), "end": str(state["score_day"].max()), "complete_feature_rows": int(np.isfinite(state[list(FEATURES)].to_numpy(dtype=float)).all(axis=1).sum())},
            "current_routing": {"rows": int(len(current)), "start": str(current["score_day"].min()), "end": str(current["score_day"].max())},
        },
        "base_reproduction": dict(base_reproduction),
        "label_contract": {
            "turnover_ratio": _finite_or_none(strategy_cfg.get("turnover_ratio")),
            "selection_equivalence": "verified frozen strategy config has turnover_ratio=1.0; no path-dependent partial-turnover cost added",
        },
        "live_config_mode": live_cfg.get("model_switch", {}).get("mode"),
        "pit_provenance": {
            "status": "not_strict_1429_certified",
            "caveat": "frozen state has legacy <=14:30 / seg1400_1430 provenance while live cutoff is 14:29; this research preserves and discloses it, not repairs it",
        },
        "frozen_plan": FROZEN_PLAN,
    }


def _csv_block(frame: pd.DataFrame) -> list[str]:
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def _write_summary(
    run_root: Path,
    *,
    summary: pd.DataFrame,
    conditional: pd.DataFrame,
    prediction: pd.DataFrame,
    calibration: pd.DataFrame,
    base_reproduction: Mapping[str, Any],
) -> None:
    text = [
        "# Temporal Model-Switch Arbitration Replay",
        "",
        "## Status",
        "",
        "- Research-only offline replay; no live config, DB, scheduler, model state, `results/live`, or `results/analysis` was written.",
        "- Input is a byte-copied snapshot. The Base 60/40/trim20_lcb10 decision was reproduced through the read-only production function.",
        f"- Base reproduction: `{base_reproduction['checked_days'] - base_reproduction['mismatch_days']}/{base_reproduction['checked_days']}` exact id/reason/gap matches.",
        "- The override action is direct `r-b` dynamic LCB > 0. The old Robust top1-top2 5bp gap is only an audit field, never this tool's gate.",
        "",
        "## Contract",
        "",
        "- Primary intervention envelope is frozen to historical `base_reason=margin_default` plus numeric existing Robust diagnostic dates. All other routes copy frozen current routing exactly.",
        "- Base support records `B_r-B_b`, candidate Base rank, Base paired-neighbour scale, forecast rank of Base, and third-model evidence separately; no unlike-scale score is silently added without uncertainty accounting.",
        "- The full-liquidation turnover_ratio=1.0 contract makes the three frozen shadow `day_return` streams valid daily selector counterfactuals for this comparison only.",
        "",
        "## Selector summary",
        "",
        *_csv_block(summary),
        "",
        "## Conditional decision metrics",
        "",
        *_csv_block(conditional),
        "",
        "## Forecast diagnostics",
        "",
        *_csv_block(prediction),
        "",
        "## Direct-pair calibration",
        "",
        *_csv_block(calibration),
        "",
        "## Limitations",
        "",
        "- The state artifact has a confirmed 14:29/14:30 provenance limitation. This replay does not establish strict 14:29 PIT compliance.",
        "- This window has informed earlier research. Any candidate that appears promising here still needs a prospectively frozen shadow period before a separate live-change decision.",
        "- No switch-aware partial-turnover contract or continuous holding cost was introduced; the current production contract is full daily liquidation.",
        "",
        "## Artifacts",
        "",
        "- `resolved_plan.json`, `input_manifest.json`, `daily_base_snapshots.csv`, `daily_forecasts.csv`, `daily_cross_arbitration.csv`",
        "- `prediction_oos_metrics.csv`, `direct_pair_calibration.csv`, `selector_summary.csv`, `conditional_metrics.csv`, `chronological_slices.csv`, `nav_compare.png`",
    ]
    (run_root / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_nav_plot(run_root: Path, arbitration: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5.5))
    baseline = arbitration.loc[arbitration["forecaster_id"] == arbitration["forecaster_id"].iloc[0]].sort_values("score_day")
    ax.plot(baseline["score_day"], np.cumprod(1.0 + baseline["current_return"].to_numpy(dtype=float)), label="frozen_current", linewidth=2.0, color="black")
    for forecaster, group in arbitration.groupby("forecaster_id", sort=False):
        group = group.sort_values("score_day")
        ax.plot(group["score_day"], np.cumprod(1.0 + group["selected_return"].to_numpy(dtype=float)), label=str(forecaster), linewidth=1.15)
    ax.set_title("Research-only temporal arbitration NAV comparison")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(run_root / "nav_compare.png", dpi=160)
    plt.close(fig)


def _write_outputs(
    run_root: Path,
    *,
    base_snapshots: pd.DataFrame,
    forecasts: pd.DataFrame,
    arbitration: pd.DataFrame,
    summary: pd.DataFrame,
    conditional: pd.DataFrame,
    slices: pd.DataFrame,
    prediction: pd.DataFrame,
    calibration: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    artifacts = {
        "daily_base_snapshots.csv": base_snapshots,
        "daily_forecasts.csv": forecasts,
        "daily_cross_arbitration.csv": arbitration,
        "selector_summary.csv": summary,
        "conditional_metrics.csv": conditional,
        "chronological_slices.csv": slices,
        "prediction_oos_metrics.csv": prediction,
        "direct_pair_calibration.csv": calibration,
    }
    for name, frame in artifacts.items():
        frame.to_csv(run_root / name, index=False)
    (run_root / "resolved_plan.json").write_text(json.dumps(FROZEN_PLAN, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (run_root / "input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_nav_plot(run_root, arbitration)
    _write_summary(
        run_root,
        summary=summary,
        conditional=conditional,
        prediction=prediction,
        calibration=calibration,
        base_reproduction=manifest["base_reproduction"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="byte-copied research input snapshot")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT, help="must stay beneath the approved experiments root")
    parser.add_argument("--run-name", default=None, help="optional new leaf directory; existing paths are refused")
    parser.add_argument("--no-strict-baseline-reproduction", action="store_true", help="diagnostic only; never use for a promotable result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = _assert_experiment_output_root(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(f"input snapshot not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_safe_git(['rev-parse', '--short', 'HEAD'])}"
    if Path(run_name).name != run_name:
        raise ValueError("run-name must be a single directory name")
    run_root = output_root / run_name
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment run: {run_root}")

    return_panel, return_paths = _read_return_panel(input_root)
    state, state_path = _read_state(input_root)
    current, current_path = _read_current(input_root)
    panel = return_panel.merge(state, on="score_day", how="left", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    live_cfg, strategy_cfg, base_cfg = _load_frozen_configs(input_root, return_paths, state_path)
    contexts, base_snapshots, reproduction = _base_contexts(
        current,
        switch_cfg=base_cfg,
        strict=not bool(args.no_strict_baseline_reproduction),
    )
    forecasts, arbitration = _run_replay(panel, current, contexts)
    summary, conditional, slices = _summary_tables(arbitration)
    prediction, calibration = _prediction_diagnostics(forecasts, arbitration)
    manifest = _input_manifest(
        input_root=input_root,
        return_paths=return_paths,
        state_path=state_path,
        current_path=current_path,
        return_panel=return_panel,
        state=state,
        current=current,
        base_reproduction=reproduction,
        live_cfg=live_cfg,
        strategy_cfg=strategy_cfg,
    )

    run_root.mkdir(parents=False, exist_ok=False)
    _write_outputs(
        run_root,
        base_snapshots=base_snapshots,
        forecasts=forecasts,
        arbitration=arbitration,
        summary=summary,
        conditional=conditional,
        slices=slices,
        prediction=prediction,
        calibration=calibration,
        manifest=manifest,
    )
    print(f"OUTPUT_ROOT {run_root}")
    print("BASE_REPRODUCTION", json.dumps(reproduction, ensure_ascii=False, default=_json_default))
    print("SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
