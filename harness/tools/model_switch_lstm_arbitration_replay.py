"""Research-only causal LSTM replay for model-switch arbitration.

This is intentionally a separate experiment from the frozen 2026-07-29
Ridge/EWMA/Hedge/DMA comparison.  It reads that experiment's byte-copied input
snapshot, reconstructs the production Base in read-only mode, and writes only
beneath ``D:/cbond_on/results/experiments/model_switch_lstm_20260730``.

The LSTM consumes a 44-feature T1430 state sequence ending on the current
score day and predicts two coherent Helmert return contrasts.  It never sees a
same-day return in its input.  Eight small architectures are selected by
chronological validation *contrast RMSE*, never by the final holdout.  The
final selected model is refitted on train plus validation only and evaluated
once on an untouched chronological holdout.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.model_switch_lstm_arbitration import (
    CPU_DEVICE,
    CausalSequenceSamples,
    FeatureScaler,
    LSTMFitResult,
    LSTMTrial,
    assert_targets_before,
    build_causal_sequence_samples,
    checkpoint_payload,
    fit_feature_scaler,
    fit_lstm_fixed_epochs,
    fit_lstm_with_early_stopping,
    make_lstm_forecast,
    predict_lstm_contrasts,
    small_lstm_grid,
    split_chronological_samples,
    transform_samples,
)
from harness.model_switch_temporal_arbitration import (
    HELMERT,
    TemporalForecast,
    cross_arbitrate,
    joint_clip_utilities,
    ordered_indices,
)
from harness.tools.model_switch_temporal_arbitration_replay import (
    FEATURES,
    ID_TO_NAME,
    MODELS,
    NAME_TO_ID,
    PAIR_INDICES,
    VALID_ROBUST_REASONS,
    BaseContext,
    _base_contexts,
    _base_pair_scale,
    _calibration_scale,
    _finite_or_none,
    _json_default,
    _load_frozen_configs,
    _read_current,
    _read_return_panel,
    _read_state,
    _safe_git,
    _selector_metrics,
    sha256,
)


EXPERIMENT_ROOT = Path(r"D:\cbond_on\results\experiments\model_switch_lstm_20260730")
DEFAULT_INPUT_ROOT = Path(
    r"D:\cbond_on\results\experiments\model_switch_temporal_arbitration_20260729\input_snapshot_20260729"
)
TARGET_CLIP = 0.0075
CALIBRATION_LOOKBACK = 120
CALIBRATION_MIN_PERIODS = 60
SCALE_FLOOR = 0.0001
Z_VALUE = 1.0
ECONOMIC_FLOOR = 0.0
INTERNAL_EARLY_STOP_FRACTION = 0.20

SPLITS: dict[str, tuple[date, date]] = {
    "train": (date(2024, 5, 8), date(2025, 10, 15)),
    "validation": (date(2025, 10, 16), date(2026, 2, 9)),
    "final_holdout": (date(2026, 2, 10), date(2026, 7, 27)),
}
EXPECTED_SPLIT_COUNTS = {
    "train": {"rows": 350, "complete_rows": 350},
    "validation": {"rows": 81, "complete_rows": 81},
    "final_holdout": {"rows": 107, "complete_rows": 101},
}

LSTM_PLAN: dict[str, Any] = {
    "schema_version": 1,
    "run_class": "research_only_offline_lstm_arbitration",
    "models": list(MODELS),
    "input_contract": {
        "state_features": "frozen path_full_t1430 44 features",
        "sequence": "state[t-L+1:t] inclusive; state[t] may be used at the T1430 decision point",
        "forbidden_input": "same-day or future shadow return is never an LSTM input",
        "missing_state_policy": "missing/nonfinite state or return breaks the sequence; no interpolation or imputation",
        "target": "two Helmert contrasts of jointly clipped three-model day returns",
        "target_clip": TARGET_CLIP,
    },
    "splits": {
        name: {"start": start.isoformat(), "end": end.isoformat(), **EXPECTED_SPLIT_COUNTS[name]}
        for name, (start, end) in SPLITS.items()
    },
    "hyperparameter_grid": {
        "sequence_length": [20, 40],
        "hidden_size": [8, 16],
        "head_dropout": [0.0, 0.1],
        "trial_count": 8,
        "architecture": "one-layer unidirectional CPU LSTM -> explicit dropout head -> linear(2)",
    },
    "training": {
        "device": CPU_DEVICE,
        "seed": 20_260_730,
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "max_epochs": 80,
        "early_stop_patience": 12,
        "internal_early_stop_fraction": INTERNAL_EARLY_STOP_FRACTION,
        "batch_size": 32,
        "fixed_target_scale": 10_000.0,
        "scaler": "per-feature mean/std fit on train samples only for selection; train+validation only for final refit",
    },
    "selection": {
        "selection_period": "validation only",
        "primary": "lowest two-contrast OOS RMSE in bp",
        "tie_break_1": "higher two-contrast direction hit rate",
        "tie_break_2": "stable lexical trial id",
        "not_used": ["final holdout returns", "final holdout Sharpe", "final holdout epochs", "seed search"],
    },
    "final_evaluation": {
        "refit": "selected architecture is fitted once on train+validation for the selected early-stop epoch count",
        "holdout_training": "none; no holdout label or state is used in model fitting or scaler fitting",
        "temporal_calibration": "validation OOS residuals initialise pairwise RMSE, then only prior holdout residuals are appended",
    },
    "arbitration": {
        "intervention_envelope": "base_reason=margin_default and existing robust_reason in score_best/margin_default",
        "action_rule": "direct candidate-vs-Base LCB > 0; no top1-top2 confidence gate",
        "z_value": Z_VALUE,
        "economic_floor": ECONOMIC_FLOOR,
        "scale_floor": SCALE_FLOOR,
        "calibration_lookback": CALIBRATION_LOOKBACK,
        "calibration_min_periods": CALIBRATION_MIN_PERIODS,
    },
    "known_limitations": [
        "state artifact is legacy <=14:30 / seg1400_1430 and is not strict-14:29 PIT certified",
        "the final holdout is historical and must not be represented as prospective live evidence",
        "full liquidation turnover_ratio=1.0 is retained; no path-dependent switch-cost contract is introduced",
    ],
}


@dataclass(frozen=True)
class PreparedInputs:
    aligned: pd.DataFrame
    return_paths: Mapping[str, Path]
    state_path: Path
    current_path: Path
    state: pd.DataFrame
    live_cfg: Mapping[str, Any]
    strategy_cfg: Mapping[str, Any]
    switch_cfg: dict[str, Any]
    actual_utilities: np.ndarray
    actual_contrasts: np.ndarray
    state_available_rows: np.ndarray
    complete_rows: np.ndarray


def _assert_experiment_output_root(output_root: Path) -> Path:
    allowed = EXPERIMENT_ROOT.resolve()
    output = output_root.resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError(f"output root must stay under {allowed}; received {output}")
    return output


def _require_run_name(value: str | None) -> str:
    if value is None:
        return f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{_safe_git(['rev-parse', '--short', 'HEAD'])}"
    candidate = Path(value)
    if not value.strip() or candidate.name != value or value in {".", ".."}:
        raise ValueError("run name must be a new simple directory name")
    return value


def _date_mask(values: pd.Series, bounds: tuple[date, date]) -> np.ndarray:
    start, end = bounds
    return ((values >= start) & (values <= end)).to_numpy(dtype=bool)


def _prepare_inputs(input_root: Path) -> PreparedInputs:
    return_panel, return_paths = _read_return_panel(input_root)
    state, state_path = _read_state(input_root)
    current, current_path = _read_current(input_root)
    live_cfg, strategy_cfg, switch_cfg = _load_frozen_configs(input_root, return_paths, state_path)
    state_columns = ["score_day", *FEATURES, "_state_present"]
    panel = return_panel.merge(state[state_columns], on="score_day", how="left", validate="one_to_one")
    aligned = current.merge(panel, on="score_day", how="inner", validate="one_to_one").sort_values("score_day").reset_index(drop=True)
    if len(aligned) != len(current) or len(aligned) != len(return_panel):
        raise RuntimeError("frozen current routing, returns, and state panel failed exact date alignment")
    expected_current = np.asarray([float(getattr(row, row.current_name)) for row in aligned.itertuples(index=False)], dtype=float)
    observed_current = pd.to_numeric(aligned["selected_return"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(observed_current).all() or not np.allclose(expected_current, observed_current, atol=1e-12, rtol=0.0):
        raise RuntimeError("frozen standalone returns do not reproduce frozen current routing")

    value_matrix = aligned[list(MODELS)].to_numpy(dtype=float)
    utilities: list[np.ndarray] = []
    for values in value_matrix:
        utility, _ = joint_clip_utilities(values, target_clip=TARGET_CLIP)
        utilities.append(utility)
    actual_utilities = np.asarray(utilities, dtype=float)
    actual_contrasts = actual_utilities @ HELMERT.T
    state_present = aligned["_state_present"].fillna(False).astype(bool).to_numpy(dtype=bool)
    state_available_rows = (
        state_present
        & np.isfinite(aligned[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    )
    complete_rows = state_available_rows & np.isfinite(value_matrix).all(axis=1)
    return PreparedInputs(
        aligned=aligned,
        return_paths=return_paths,
        state_path=state_path,
        current_path=current_path,
        state=state,
        live_cfg=live_cfg,
        strategy_cfg=strategy_cfg,
        switch_cfg=switch_cfg,
        actual_utilities=actual_utilities,
        actual_contrasts=actual_contrasts,
        state_available_rows=state_available_rows,
        complete_rows=complete_rows,
    )


def _validate_split_contract(prepared: PreparedInputs) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    all_dates = prepared.aligned["score_day"]
    for name, bounds in SPLITS.items():
        mask = _date_mask(all_dates, bounds)
        masks[name] = mask
        expected = EXPECTED_SPLIT_COUNTS[name]
        actual_rows = int(mask.sum())
        actual_complete = int(prepared.complete_rows[mask].sum())
        if actual_rows != expected["rows"] or actual_complete != expected["complete_rows"]:
            raise RuntimeError(
                f"frozen {name} split mismatch: expected rows/complete={expected['rows']}/{expected['complete_rows']}; "
                f"observed={actual_rows}/{actual_complete}"
            )
    memberships = np.asarray([sum(mask[index] for mask in masks.values()) for index in range(len(prepared.aligned))], dtype=int)
    if not np.all(memberships == 1):
        raise RuntimeError("the predeclared chronological splits do not partition the frozen return panel")
    return masks


def _sample_sets(prepared: PreparedInputs, masks: Mapping[str, np.ndarray], trial: LSTMTrial) -> dict[str, CausalSequenceSamples]:
    features = prepared.aligned[list(FEATURES)].to_numpy(dtype=float)
    result: dict[str, CausalSequenceSamples] = {}
    for name, mask in masks.items():
        targets = np.flatnonzero(mask)
        result[name] = build_causal_sequence_samples(
            features,
            prepared.actual_contrasts,
            prepared.state_available_rows,
            sequence_length=trial.sequence_length,
            target_indices=targets,
            require_target=True,
        )
    if not len(result["train"]) or not len(result["validation"]) or not len(result["final_holdout"]):
        raise RuntimeError(f"trial {trial.trial_id} has an empty required sample set")
    assert_targets_before(result["train"], int(result["validation"].target_indices.min()))
    assert_targets_before(result["train"], int(result["final_holdout"].target_indices.min()))
    return result


def _forecast_row(
    *,
    score_day: date,
    period: str,
    panel_index: int,
    trial_id: str,
    forecast: TemporalForecast | None,
    actual_utilities: np.ndarray,
    actual_contrasts: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "score_day": score_day,
        "period": period,
        "panel_index": int(panel_index),
        "forecaster_id": trial_id,
        "prediction_ready": bool(forecast is not None and forecast.ready),
        "history_days": None if forecast is None else int(forecast.history_days),
        "forecast_top_name": None,
        "forecast_top_model_id": None,
        "forecast_top_gap_diagnostic": None,
        "actual_contrast_1": float(actual_contrasts[0]),
        "actual_contrast_2": float(actual_contrasts[1]),
        "pred_contrast_1": None,
        "pred_contrast_2": None,
    }
    for index, name in enumerate(MODELS):
        row[f"ranking_score_{name}"] = None if forecast is None else float(forecast.utilities[index])
    for left, right in PAIR_INDICES:
        label = f"{MODELS[left]}_minus_{MODELS[right]}"
        row[f"actual_clipped_{label}"] = float(actual_utilities[left] - actual_utilities[right])
        row[f"pred_{label}"] = None if forecast is None else float(forecast.pair_mean[left, right])
        row[f"intrinsic_scale_{label}"] = None if forecast is None else _finite_or_none(forecast.pair_scale[left, right])
    if forecast is not None:
        ranking = ordered_indices(forecast.utilities)
        row["forecast_top_name"] = MODELS[ranking[0]]
        row["forecast_top_model_id"] = NAME_TO_ID[MODELS[ranking[0]]]
        row["forecast_top_gap_diagnostic"] = float(forecast.utilities[ranking[0]] - forecast.utilities[ranking[1]])
        predicted_contrasts = HELMERT @ forecast.utilities
        row["pred_contrast_1"] = float(predicted_contrasts[0])
        row["pred_contrast_2"] = float(predicted_contrasts[1])
        for key, value in forecast.diagnostics.items():
            row[f"diagnostic_{key}"] = value
    return row


def _outside_envelope_record(
    *,
    score_day: date,
    period: str,
    panel_index: int,
    trial_id: str,
    context: BaseContext,
    current_name: str,
    returns: np.ndarray,
    current_row: Any,
    reason: str,
    action: str,
) -> dict[str, Any]:
    base_index = MODELS.index(context.base_name)
    current_index = MODELS.index(current_name)
    base_return = float(returns[base_index])
    current_return = float(returns[current_index])
    return {
        "score_day": score_day,
        "period": period,
        "panel_index": int(panel_index),
        "forecaster_id": trial_id,
        "route_eligible": False,
        "base_name": context.base_name,
        "base_model_id": NAME_TO_ID[context.base_name],
        "current_name": current_name,
        "current_model_id": NAME_TO_ID[current_name],
        "selected_name": current_name,
        "selected_model_id": NAME_TO_ID[current_name],
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
        "selected_return": current_return,
        "base_return": base_return,
        "current_return": current_return,
        "candidate_return": None,
        "candidate_return_delta_vs_base": None,
        "current_return_delta_vs_base": current_return - base_return,
        "return_delta_vs_base": current_return - base_return,
        "return_delta_vs_current": 0.0,
        "current_overrode_base": bool(current_name != context.base_name),
        "override_base": False,
        "changed_vs_current": False,
    }


def _arbitration_record(
    *,
    score_day: date,
    period: str,
    panel_index: int,
    trial_id: str,
    context: BaseContext,
    current_name: str,
    forecast: TemporalForecast | None,
    calibration_errors: Mapping[tuple[int, int], list[float]],
    returns: np.ndarray,
    current_row: Any,
) -> dict[str, Any]:
    base_index = MODELS.index(context.base_name)
    current_index = MODELS.index(current_name)
    eligible = bool(str(current_row.base_reason) == "margin_default" and str(current_row.robust_reason) in VALID_ROBUST_REASONS)
    if not eligible:
        return _outside_envelope_record(
            score_day=score_day,
            period=period,
            panel_index=panel_index,
            trial_id=trial_id,
            context=context,
            current_name=current_name,
            returns=returns,
            current_row=current_row,
            reason="outside_frozen_intervention_envelope",
            action="copy_current_routing",
        )
    base_return = float(returns[base_index])
    current_return = float(returns[current_index])
    if forecast is None or not forecast.ready:
        return {
            **_outside_envelope_record(
                score_day=score_day,
                period=period,
                panel_index=panel_index,
                trial_id=trial_id,
                context=context,
                current_name=current_name,
                returns=returns,
                current_row=current_row,
                reason="lstm_forecast_unavailable",
                action="keep_base",
            ),
            "route_eligible": True,
            "selected_name": context.base_name,
            "selected_model_id": NAME_TO_ID[context.base_name],
            "selected_return": base_return,
            "return_delta_vs_base": 0.0,
            "return_delta_vs_current": base_return - current_return,
            "changed_vs_current": bool(context.base_name != current_name),
        }

    candidate_index = int(ordered_indices(forecast.utilities, preferred_index=base_index)[0])
    if candidate_index == base_index:
        calibration, calibration_n = None, 0
        pair_scale, pair_n = None, 0
    else:
        pair_key = (min(candidate_index, base_index), max(candidate_index, base_index))
        calibration, calibration_n = _calibration_scale(
            calibration_errors[pair_key],
            lookback=CALIBRATION_LOOKBACK,
            min_periods=CALIBRATION_MIN_PERIODS,
        )
        pair_scale, pair_n = _base_pair_scale(context, candidate_index=candidate_index, base_index=base_index)
    decision = cross_arbitrate(
        base_index=base_index,
        base_scores=context.scores,
        base_pair_scale=pair_scale,
        forecast=forecast,
        temporal_calibration_scale=calibration,
        z_value=Z_VALUE,
        economic_floor=ECONOMIC_FLOOR,
        scale_floor=SCALE_FLOOR,
    )
    selected_name = MODELS[decision.selected_index]
    selected_return = float(returns[decision.selected_index])
    candidate_return = None if decision.candidate_index is None else float(returns[decision.candidate_index])
    return {
        "score_day": score_day,
        "period": period,
        "panel_index": int(panel_index),
        "forecaster_id": trial_id,
        "route_eligible": True,
        "base_name": context.base_name,
        "base_model_id": NAME_TO_ID[context.base_name],
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
        "current_overrode_base": bool(current_name != context.base_name),
        "override_base": bool(decision.selected_index != base_index),
        "changed_vs_current": bool(selected_name != current_name),
    }


def _empty_calibration_errors() -> dict[tuple[int, int], list[float]]:
    return {pair: [] for pair in PAIR_INDICES}


def _copy_calibration_errors(source: Mapping[tuple[int, int], Sequence[float]]) -> dict[tuple[int, int], list[float]]:
    return {pair: [float(value) for value in source[pair]] for pair in PAIR_INDICES}


def _append_calibration_errors(
    errors: dict[tuple[int, int], list[float]],
    *,
    forecast: TemporalForecast | None,
    actual_utilities: np.ndarray,
) -> None:
    """Append only the just-realised outcome, after its decision is frozen."""

    if forecast is None or not forecast.ready:
        return
    for left, right in PAIR_INDICES:
        prediction = float(forecast.pair_mean[left, right])
        actual = float(actual_utilities[left] - actual_utilities[right])
        if math.isfinite(prediction) and math.isfinite(actual):
            errors[(left, right)].append(prediction - actual)


def _run_arbitration_period(
    *,
    prepared: PreparedInputs,
    contexts: Mapping[date, BaseContext],
    period: str,
    mask: np.ndarray,
    trial: LSTMTrial,
    forecasts_by_index: Mapping[int, TemporalForecast],
    initial_calibration_errors: Mapping[tuple[int, int], Sequence[float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[int, int], list[float]]]:
    """Make all daily choices before adding that day's realised errors."""

    errors = _empty_calibration_errors() if initial_calibration_errors is None else _copy_calibration_errors(initial_calibration_errors)
    forecast_rows: list[dict[str, Any]] = []
    arbitration_rows: list[dict[str, Any]] = []
    for panel_index in np.flatnonzero(mask):
        row = prepared.aligned.iloc[int(panel_index)]
        score_day = row["score_day"]
        context = contexts.get(score_day)
        if context is None:
            raise RuntimeError(f"missing Base context for {score_day}")
        forecast = forecasts_by_index.get(int(panel_index))
        values = row[list(MODELS)].to_numpy(dtype=float)
        forecast_rows.append(
            _forecast_row(
                score_day=score_day,
                period=period,
                panel_index=int(panel_index),
                trial_id=trial.trial_id,
                forecast=forecast,
                actual_utilities=prepared.actual_utilities[int(panel_index)],
                actual_contrasts=prepared.actual_contrasts[int(panel_index)],
            )
        )
        decision_row = _arbitration_record(
            score_day=score_day,
            period=period,
            panel_index=int(panel_index),
            trial_id=trial.trial_id,
            context=context,
            current_name=str(row["current_name"]),
            forecast=forecast,
            calibration_errors=errors,
            returns=values,
            current_row=row,
        )
        decision_row["prediction_ready"] = bool(forecast is not None and forecast.ready)
        arbitration_rows.append(decision_row)
        _append_calibration_errors(errors, forecast=forecast, actual_utilities=prepared.actual_utilities[int(panel_index)])
    return pd.DataFrame(forecast_rows), pd.DataFrame(arbitration_rows), errors


def _contrast_forecast_metrics(forecasts: pd.DataFrame) -> dict[str, Any]:
    valid = forecasts.loc[forecasts["prediction_ready"].astype(bool)].copy()
    predicted = valid[["pred_contrast_1", "pred_contrast_2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    actual = valid[["actual_contrast_1", "actual_contrast_2"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(actual).all(axis=1)
    predicted = predicted[finite]
    actual = actual[finite]
    result: dict[str, Any] = {"forecast_ready_days": int(len(predicted))}
    if not len(predicted):
        return result | {
            "two_contrast_rmse_bp": float("nan"),
            "two_contrast_mae_bp": float("nan"),
            "two_contrast_direction_hit_rate": float("nan"),
        }
    error = predicted - actual
    result.update(
        {
            "two_contrast_rmse_bp": float(math.sqrt(float(np.mean(error**2))) * 1e4),
            "two_contrast_mae_bp": float(np.mean(np.abs(error)) * 1e4),
            "two_contrast_direction_hit_rate": float(np.mean(np.sign(predicted) == np.sign(actual))),
        }
    )
    for column, index in (("contrast_1", 0), ("contrast_2", 1)):
        y_pred = predicted[:, index]
        y_true = actual[:, index]
        denominator = float(np.sum((y_true - y_true.mean()) ** 2))
        result[f"{column}_corr"] = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) >= 2 and np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan")
        result[f"{column}_oos_r2"] = float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator) if denominator > 0.0 else float("nan")
    return result


def _pair_prediction_diagnostics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, group in forecasts.groupby("period", sort=False):
        for left, right in PAIR_INDICES:
            label = f"{MODELS[left]}_minus_{MODELS[right]}"
            prediction = pd.to_numeric(group[f"pred_{label}"], errors="coerce")
            actual = pd.to_numeric(group[f"actual_clipped_{label}"], errors="coerce")
            valid = group["prediction_ready"].astype(bool) & prediction.notna() & actual.notna()
            y_pred = prediction.loc[valid].to_numpy(dtype=float)
            y_true = actual.loc[valid].to_numpy(dtype=float)
            row: dict[str, Any] = {"period": period, "forecaster_id": str(group["forecaster_id"].iloc[0]), "pair": label, "n": int(len(y_true))}
            if len(y_true) >= 2:
                denominator = float(np.sum((y_true - y_true.mean()) ** 2))
                row.update(
                    {
                        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]) if np.std(y_pred) > 0.0 and np.std(y_true) > 0.0 else float("nan"),
                        "oos_r2": float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator) if denominator > 0.0 else float("nan"),
                        "direction_hit_rate": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
                        "mae_bp": float(np.mean(np.abs(y_true - y_pred)) * 1e4),
                        "rmse_bp": float(math.sqrt(float(np.mean((y_true - y_pred) ** 2))) * 1e4),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _selector_summary(arbitration: pd.DataFrame, *, period: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare selected LSTM routing with aligned current and Base fallback."""

    subset = arbitration.sort_values("score_day").reset_index(drop=True)
    if subset.empty:
        raise ValueError("selector summary requires at least one decision")
    current_metrics = _selector_metrics(subset["current_return"], subset["score_day"], subset["current_name"])
    base_names = np.where(subset["route_eligible"].astype(bool), subset["base_name"], subset["current_name"])
    base_returns = np.where(
        subset["route_eligible"].astype(bool),
        pd.to_numeric(subset["base_return"], errors="raise").to_numpy(dtype=float),
        pd.to_numeric(subset["current_return"], errors="raise").to_numpy(dtype=float),
    )
    base_metrics = _selector_metrics(pd.Series(base_returns), subset["score_day"], pd.Series(base_names))
    lstm_metrics = _selector_metrics(subset["selected_return"], subset["score_day"], subset["selected_name"])
    summary = pd.DataFrame(
        [
            {
                "period": period,
                "strategy": "frozen_current_routing",
                **current_metrics,
                "route_eligible_days": int(subset["route_eligible"].sum()),
                "forecast_ready_days": int(subset["prediction_ready"].sum()),
                "override_base_days": int(subset["current_overrode_base"].sum()),
                "changed_vs_current_days": 0,
                "total_return_delta_vs_current": 0.0,
                "sharpe_delta_vs_current": 0.0,
            },
            {
                "period": period,
                "strategy": "base_fallback_within_envelope",
                **base_metrics,
                "route_eligible_days": int(subset["route_eligible"].sum()),
                "forecast_ready_days": 0,
                "override_base_days": 0,
                "changed_vs_current_days": int((pd.Series(base_names) != subset["current_name"]).sum()),
                "total_return_delta_vs_current": float(base_metrics["total_return"] - current_metrics["total_return"]),
                "sharpe_delta_vs_current": float(base_metrics["sharpe"] - current_metrics["sharpe"]),
            },
            {
                "period": period,
                "strategy": str(subset["forecaster_id"].iloc[0]),
                **lstm_metrics,
                "route_eligible_days": int(subset["route_eligible"].sum()),
                "forecast_ready_days": int(subset["prediction_ready"].sum()),
                "override_base_days": int(subset["override_base"].sum()),
                "changed_vs_current_days": int(subset["changed_vs_current"].sum()),
                "total_return_delta_vs_current": float(lstm_metrics["total_return"] - current_metrics["total_return"]),
                "sharpe_delta_vs_current": float(lstm_metrics["sharpe"] - current_metrics["sharpe"]),
            },
        ]
    )

    condition_rows: list[dict[str, Any]] = []
    for condition, mask, column in (
        ("override_base", subset["route_eligible"].astype(bool) & subset["override_base"].astype(bool), "return_delta_vs_base"),
        ("changed_vs_current", subset["changed_vs_current"].astype(bool), "return_delta_vs_current"),
        ("all_eligible", subset["route_eligible"].astype(bool), "return_delta_vs_current"),
    ):
        values = pd.to_numeric(subset.loc[mask, column], errors="coerce").dropna().to_numpy(dtype=float)
        wins = int((values > 0.0).sum()) if len(values) else 0
        item: dict[str, Any] = {
            "period": period,
            "forecaster_id": str(subset["forecaster_id"].iloc[0]),
            "condition": condition,
            "n": int(len(values)),
            "wins": wins,
            "losses": int((values < 0.0).sum()) if len(values) else 0,
        }
        if len(values):
            item.update(
                {
                    "mean_delta_bp": float(values.mean() * 1e4),
                    "median_delta_bp": float(np.median(values) * 1e4),
                    "sum_delta_bp": float(values.sum() * 1e4),
                    "hit_rate": float((values > 0.0).mean()),
                }
            )
        if len(values) > 1:
            item["paired_t_p_one_sided"] = float(stats.ttest_1samp(values, 0.0, alternative="greater").pvalue)
            item["sign_p_one_sided"] = float(stats.binomtest(wins, len(values), 0.5, alternative="greater").pvalue)
        condition_rows.append(item)

    slice_rows: list[dict[str, Any]] = []
    for part, positions in enumerate(np.array_split(np.arange(len(subset)), 3), start=1):
        if not len(positions):
            continue
        window = subset.iloc[positions]
        window_lstm = _selector_metrics(window["selected_return"], window["score_day"], window["selected_name"])
        window_current = _selector_metrics(window["current_return"], window["score_day"], window["current_name"])
        slice_rows.append(
            {
                "period": period,
                "forecaster_id": str(subset["forecaster_id"].iloc[0]),
                "slice": f"chronological_{part}",
                **window_lstm,
                "route_eligible_days": int(window["route_eligible"].sum()),
                "override_base_days": int(window["override_base"].sum()),
                "total_return_delta_vs_current": float(window_lstm["total_return"] - window_current["total_return"]),
                "sharpe_delta_vs_current": float(window_lstm["sharpe"] - window_current["sharpe"]),
            }
        )
    return summary, pd.DataFrame(condition_rows), pd.DataFrame(slice_rows)


def _direct_pair_calibration(arbitration: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for period, group in arbitration.groupby("period", sort=False):
        direct = group.loc[
            group["route_eligible"].astype(bool)
            & group["candidate_name"].notna()
            & ~group["candidate_equals_base"].fillna(False)
            & group["temporal_mean_candidate_minus_base"].notna()
        ].copy()
        for condition, subset in (("direct_candidate_cases", direct), ("cross_lcb_override", direct.loc[direct["action"] == "override_base"])):
            actual = pd.to_numeric(subset["candidate_return_delta_vs_base"], errors="coerce").dropna()
            lcb = pd.to_numeric(subset["lcb"], errors="coerce")
            rows.append(
                {
                    "period": period,
                    "forecaster_id": str(group["forecaster_id"].iloc[0]),
                    "condition": condition,
                    "n": int(len(subset)),
                    "actual_positive_rate": float((actual > 0.0).mean()) if len(actual) else float("nan"),
                    "mean_actual_delta_bp": float(actual.mean() * 1e4) if len(actual) else float("nan"),
                    "mean_lcb_bp": float(lcb.mean() * 1e4) if lcb.notna().any() else float("nan"),
                    "lcb_available_n": int(lcb.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _sample_span(samples: CausalSequenceSamples, aligned: pd.DataFrame) -> dict[str, Any]:
    if not len(samples):
        return {
            "sample_count": 0,
            "target_start_index": None,
            "target_end_index": None,
            "target_start_day": None,
            "target_end_day": None,
            "input_start_index": None,
            "input_end_index": None,
        }
    return {
        "sample_count": int(len(samples)),
        "target_start_index": int(samples.target_indices.min()),
        "target_end_index": int(samples.target_indices.max()),
        "target_start_day": str(aligned.iloc[int(samples.target_indices.min())]["score_day"]),
        "target_end_day": str(aligned.iloc[int(samples.target_indices.max())]["score_day"]),
        "input_start_index": int(samples.input_start_indices.min()),
        "input_end_index": int(samples.target_indices.max()),
    }


def _fit_event(
    *,
    stage: str,
    trial: LSTMTrial,
    prepared: PreparedInputs,
    scaler: FeatureScaler,
    fit_samples: CausalSequenceSamples,
    prediction_samples: CausalSequenceSamples,
    result: LSTMFitResult,
    checkpoint_path: Path | None = None,
    early_stop_samples: CausalSequenceSamples | None = None,
) -> dict[str, Any]:
    fit_span = _sample_span(fit_samples, prepared.aligned)
    predict_span = _sample_span(prediction_samples, prepared.aligned)
    event: dict[str, Any] = {
        "stage": stage,
        **trial.as_dict(),
        "device": CPU_DEVICE,
        "feature_count": int(fit_samples.feature_count),
        "scaler_fitted_sample_count": int(scaler.fitted_sample_count),
        "scaler_target_end_index": fit_span["target_end_index"],
        "scaler_target_end_day": fit_span["target_end_day"],
        "fit": fit_span,
        "prediction": predict_span,
        "strictly_before_prediction": bool(
            not len(fit_samples)
            or not len(prediction_samples)
            or int(fit_samples.target_indices.max()) < int(prediction_samples.target_indices.min())
        ),
        "best_epoch": int(result.best_epoch),
        "best_validation_loss": result.best_validation_loss,
        "final_training_loss": float(result.final_training_loss),
        "epochs_ran": int(result.epochs_ran),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
    }
    if early_stop_samples is not None:
        event["early_stop"] = _sample_span(early_stop_samples, prepared.aligned)
        event["fit_before_early_stop"] = bool(
            not len(fit_samples)
            or not len(early_stop_samples)
            or int(fit_samples.target_indices.max()) < int(early_stop_samples.target_indices.min())
        )
    return event


def _save_checkpoint(path: Path, *, trial: LSTMTrial, scaler: FeatureScaler, fit_result: LSTMFitResult, split: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(trial=trial, scaler=scaler, fit_result=fit_result, split=split), path)
    return {"path": str(path), "sha256": sha256(path), "bytes": int(path.stat().st_size), "split": split}


def _forecasts_from_predictions(
    *,
    trial: LSTMTrial,
    samples: CausalSequenceSamples,
    predictions: np.ndarray,
    history_days: int,
    stage: str,
    fit_target_end_index: int,
) -> dict[int, TemporalForecast]:
    if predictions.shape != (len(samples), 2):
        raise ValueError("LSTM prediction shape does not match the sample set")
    result: dict[int, TemporalForecast] = {}
    for position, (target_index, contrast) in enumerate(zip(samples.target_indices, predictions, strict=True)):
        if int(target_index) in result:
            raise RuntimeError("duplicate LSTM target index")
        result[int(target_index)] = make_lstm_forecast(
            trial=trial,
            contrast_prediction=contrast,
            history_days=history_days,
            diagnostics={
                "stage": stage,
                "fit_target_end_index": int(fit_target_end_index),
                "sequence_start_index": int(samples.input_start_indices[position]),
                "sequence_end_index": int(target_index),
            },
        )
    return result


@dataclass(frozen=True)
class ValidationTrialResult:
    trial: LSTMTrial
    sample_sets: Mapping[str, CausalSequenceSamples]
    early_stop_result: LSTMFitResult
    validation_refit_result: LSTMFitResult
    validation_scaler: FeatureScaler
    validation_forecasts: pd.DataFrame
    validation_arbitration: pd.DataFrame
    calibration_errors: Mapping[tuple[int, int], list[float]]
    metrics: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    training_events: Sequence[Mapping[str, Any]]


def _run_validation_trial(
    *,
    prepared: PreparedInputs,
    masks: Mapping[str, np.ndarray],
    contexts: Mapping[date, BaseContext],
    trial: LSTMTrial,
    checkpoints_dir: Path,
) -> ValidationTrialResult:
    """Tune exactly one fixed-grid member without touching final-holdout data."""

    samples = _sample_sets(prepared, masks, trial)
    early_fit_raw, early_stop_raw = split_chronological_samples(
        samples["train"], validation_fraction=INTERNAL_EARLY_STOP_FRACTION
    )
    assert_targets_before(early_fit_raw, int(early_stop_raw.target_indices.min()))
    early_scaler = fit_feature_scaler(early_fit_raw)
    early_result = fit_lstm_with_early_stopping(
        transform_samples(early_fit_raw, early_scaler),
        transform_samples(early_stop_raw, early_scaler),
        trial=trial,
    )

    # Epoch count is fixed by only the in-train chronological early-stop split.
    # The validation period has not entered the model, scaler, or epoch choice.
    assert_targets_before(samples["train"], int(samples["validation"].target_indices.min()))
    validation_scaler = fit_feature_scaler(samples["train"])
    validation_refit_result = fit_lstm_fixed_epochs(
        transform_samples(samples["train"], validation_scaler),
        trial=trial,
        epochs=early_result.best_epoch,
        seed_offset=1,
    )
    validation_predictions = predict_lstm_contrasts(
        validation_refit_result.model,
        transform_samples(samples["validation"], validation_scaler),
    )
    forecasts_by_index = _forecasts_from_predictions(
        trial=trial,
        samples=samples["validation"],
        predictions=validation_predictions,
        history_days=len(samples["train"]),
        stage="validation_refit_train_only",
        fit_target_end_index=int(samples["train"].target_indices.max()),
    )
    validation_forecasts, validation_arbitration, calibration_errors = _run_arbitration_period(
        prepared=prepared,
        contexts=contexts,
        period="validation",
        mask=masks["validation"],
        trial=trial,
        forecasts_by_index=forecasts_by_index,
    )
    checkpoint_path = checkpoints_dir / f"{trial.trial_id}_validation_refit.pt"
    checkpoint = _save_checkpoint(
        checkpoint_path,
        trial=trial,
        scaler=validation_scaler,
        fit_result=validation_refit_result,
        split="validation_refit_train_only",
    )
    validation_selector, _, validation_slices = _selector_summary(validation_arbitration, period="validation")
    lstm_selector = validation_selector.loc[validation_selector["strategy"] == trial.trial_id].iloc[0].to_dict()
    metrics = {
        **_contrast_forecast_metrics(validation_forecasts),
        "validation_selector_total_return": float(lstm_selector["total_return"]),
        "validation_selector_sharpe": float(lstm_selector["sharpe"]),
        "validation_selector_override_base_days": int(lstm_selector["override_base_days"]),
        "validation_selector_slice_count": int(len(validation_slices)),
        "validation_prediction_coverage": float(validation_forecasts["prediction_ready"].mean()),
    }
    training_events = [
        _fit_event(
            stage="in_train_early_stopping",
            trial=trial,
            prepared=prepared,
            scaler=early_scaler,
            fit_samples=early_fit_raw,
            early_stop_samples=early_stop_raw,
            prediction_samples=samples["validation"],
            result=early_result,
        ),
        _fit_event(
            stage="validation_forecast_refit_train_only",
            trial=trial,
            prepared=prepared,
            scaler=validation_scaler,
            fit_samples=samples["train"],
            prediction_samples=samples["validation"],
            result=validation_refit_result,
            checkpoint_path=checkpoint_path,
        ),
    ]
    return ValidationTrialResult(
        trial=trial,
        sample_sets=samples,
        early_stop_result=early_result,
        validation_refit_result=validation_refit_result,
        validation_scaler=validation_scaler,
        validation_forecasts=validation_forecasts,
        validation_arbitration=validation_arbitration,
        calibration_errors=calibration_errors,
        metrics=metrics,
        checkpoint=checkpoint,
        training_events=training_events,
    )


def _grid_table(results: Sequence[ValidationTrialResult]) -> tuple[pd.DataFrame, ValidationTrialResult]:
    if not results:
        raise ValueError("LSTM grid cannot be empty")

    ordered = sorted(
        results,
        key=lambda result: (
            float(result.metrics["two_contrast_rmse_bp"]) if math.isfinite(float(result.metrics["two_contrast_rmse_bp"])) else float("inf"),
            -float(result.metrics["two_contrast_direction_hit_rate"])
            if math.isfinite(float(result.metrics["two_contrast_direction_hit_rate"]))
            else float("inf"),
            result.trial.trial_id,
        ),
    )
    selected = ordered[0]
    rank_by_id = {result.trial.trial_id: index for index, result in enumerate(ordered, start=1)}
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                **result.trial.as_dict(),
                **dict(result.metrics),
                "in_train_best_epoch": int(result.early_stop_result.best_epoch),
                "in_train_best_validation_loss": result.early_stop_result.best_validation_loss,
                "validation_refit_epochs": int(result.validation_refit_result.epochs_ran),
                "checkpoint_sha256": result.checkpoint["sha256"],
                "selection_rank": int(rank_by_id[result.trial.trial_id]),
                "selected_for_final_holdout": bool(result.trial.trial_id == selected.trial.trial_id),
            }
        )
    return pd.DataFrame(rows).sort_values("selection_rank").reset_index(drop=True), selected


@dataclass(frozen=True)
class FinalHoldoutResult:
    final_scaler: FeatureScaler
    final_fit_result: LSTMFitResult
    forecasts: pd.DataFrame
    arbitration: pd.DataFrame
    calibration_errors: Mapping[tuple[int, int], list[float]]
    checkpoint: Mapping[str, Any]
    training_event: Mapping[str, Any]
    sample_sets: Mapping[str, CausalSequenceSamples]


def _run_final_holdout(
    *,
    prepared: PreparedInputs,
    masks: Mapping[str, np.ndarray],
    contexts: Mapping[date, BaseContext],
    selected: ValidationTrialResult,
    checkpoints_dir: Path,
) -> FinalHoldoutResult:
    """Refit the selected architecture once before the untouched final period."""

    trial = selected.trial
    samples = _sample_sets(prepared, masks, trial)
    development_mask = masks["train"] | masks["validation"]
    development_samples = build_causal_sequence_samples(
        prepared.aligned[list(FEATURES)].to_numpy(dtype=float),
        prepared.actual_contrasts,
        prepared.state_available_rows,
        sequence_length=trial.sequence_length,
        target_indices=np.flatnonzero(development_mask),
        require_target=True,
    )
    if not len(development_samples):
        raise RuntimeError("selected LSTM has no train+validation samples")
    assert_targets_before(development_samples, int(samples["final_holdout"].target_indices.min()))
    final_scaler = fit_feature_scaler(development_samples)
    final_fit_result = fit_lstm_fixed_epochs(
        transform_samples(development_samples, final_scaler),
        trial=trial,
        epochs=selected.early_stop_result.best_epoch,
        seed_offset=2,
    )
    final_predictions = predict_lstm_contrasts(
        final_fit_result.model,
        transform_samples(samples["final_holdout"], final_scaler),
    )
    forecasts_by_index = _forecasts_from_predictions(
        trial=trial,
        samples=samples["final_holdout"],
        predictions=final_predictions,
        history_days=len(development_samples),
        stage="final_refit_train_validation_only",
        fit_target_end_index=int(development_samples.target_indices.max()),
    )
    forecasts, arbitration, errors = _run_arbitration_period(
        prepared=prepared,
        contexts=contexts,
        period="final_holdout",
        mask=masks["final_holdout"],
        trial=trial,
        forecasts_by_index=forecasts_by_index,
        initial_calibration_errors=selected.calibration_errors,
    )
    checkpoint_path = checkpoints_dir / f"{trial.trial_id}_final_refit_train_validation.pt"
    checkpoint = _save_checkpoint(
        checkpoint_path,
        trial=trial,
        scaler=final_scaler,
        fit_result=final_fit_result,
        split="final_refit_train_validation_only",
    )
    event = _fit_event(
        stage="final_refit_train_validation_only",
        trial=trial,
        prepared=prepared,
        scaler=final_scaler,
        fit_samples=development_samples,
        prediction_samples=samples["final_holdout"],
        result=final_fit_result,
        checkpoint_path=checkpoint_path,
    )
    return FinalHoldoutResult(
        final_scaler=final_scaler,
        final_fit_result=final_fit_result,
        forecasts=forecasts,
        arbitration=arbitration,
        calibration_errors=errors,
        checkpoint=checkpoint,
        training_event=event,
        sample_sets=samples,
    )


def _input_manifest(
    *,
    input_root: Path,
    prepared: PreparedInputs,
    base_reproduction: Mapping[str, Any],
    grid: pd.DataFrame,
    selected: ValidationTrialResult,
    final: FinalHoldoutResult,
    elapsed_seconds: float,
) -> dict[str, Any]:
    input_files = [
        prepared.current_path,
        prepared.state_path,
        *(prepared.return_paths[name] for name in MODELS),
        input_root / "live_config.json5",
        input_root / "strategy01_config.json5",
    ]
    snapshot_hashes_path = input_root / "snapshot_hashes.json"
    snapshot_hashes = json.loads(snapshot_hashes_path.read_text(encoding="utf-8-sig")) if snapshot_hashes_path.exists() else None
    source_files = [
        REPO_ROOT / "cbond_on" / "infra" / "live" / "model_switch.py",
        REPO_ROOT / "harness" / "model_switch_temporal_arbitration.py",
        REPO_ROOT / "harness" / "tools" / "model_switch_temporal_arbitration_replay.py",
        REPO_ROOT / "harness" / "model_switch_lstm_arbitration.py",
        Path(__file__).resolve(),
    ]
    split_coverage: dict[str, Any] = {}
    for name, bounds in SPLITS.items():
        mask = _date_mask(prepared.aligned["score_day"], bounds)
        split_coverage[name] = {
            "start": bounds[0].isoformat(),
            "end": bounds[1].isoformat(),
            "return_rows": int(mask.sum()),
            "state_return_complete_rows": int(prepared.complete_rows[mask].sum()),
            "selected_trial_sequence_samples": int(final.sample_sets[name].__len__()),
        }
    return {
        "run_class": "research_only_offline_lstm_arbitration",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "output_contract": "only D:/cbond_on/results/experiments/model_switch_lstm_20260730/run_* is written",
        "input_root": str(input_root),
        "input_files": [
            {"path": str(path), "sha256": sha256(path), "bytes": int(path.stat().st_size)} for path in input_files
        ],
        "snapshot_source_hashes": snapshot_hashes,
        "source_code": [{"path": str(path), "sha256": sha256(path)} for path in source_files],
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "device": CPU_DEVICE,
            "torch_deterministic_algorithms": True,
            "torch_num_threads": int(torch.get_num_threads()),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "models": {name: NAME_TO_ID[name] for name in MODELS},
        "date_coverage": {
            "returns": {
                "rows": int(len(prepared.aligned)),
                "start": str(prepared.aligned["score_day"].min()),
                "end": str(prepared.aligned["score_day"].max()),
            },
            "state": {
                "rows": int(len(prepared.state)),
                "start": str(prepared.state["score_day"].min()),
                "end": str(prepared.state["score_day"].max()),
                "complete_feature_rows": int(np.isfinite(prepared.state[list(FEATURES)].to_numpy(dtype=float)).all(axis=1).sum()),
            },
            "split_coverage": split_coverage,
        },
        "base_reproduction": dict(base_reproduction),
        "label_contract": {
            "turnover_ratio": _finite_or_none(prepared.strategy_cfg.get("turnover_ratio")),
            "selection_equivalence": "frozen strategy config confirms turnover_ratio=1.0; no path-dependent partial-turnover cost added",
        },
        "live_config_mode": prepared.live_cfg.get("model_switch", {}).get("mode"),
        "pit_provenance": {
            "status": "not_strict_1429_certified",
            "caveat": "frozen state has legacy <=14:30 / seg1400_1430 provenance while live cutoff is 14:29; this research preserves and discloses it, not repairs it",
        },
        "resolved_lstm_plan": LSTM_PLAN,
        "grid_selection": {
            "selected_trial_id": selected.trial.trial_id,
            "selected_in_train_epoch": int(selected.early_stop_result.best_epoch),
            "selection_rule": LSTM_PLAN["selection"],
            "validation_grid_rows": grid.to_dict(orient="records"),
        },
        "checkpoints": [dict(selected.checkpoint), dict(final.checkpoint)],
        "elapsed_seconds": float(elapsed_seconds),
    }


def _csv_block(frame: pd.DataFrame) -> list[str]:
    return ["```csv", frame.to_csv(index=False, float_format="%.6f").rstrip(), "```"]


def _write_summary(
    run_root: Path,
    *,
    selected: ValidationTrialResult,
    grid: pd.DataFrame,
    final_summary: pd.DataFrame,
    final_prediction: pd.DataFrame,
    final_calibration: pd.DataFrame,
    base_reproduction: Mapping[str, Any],
) -> None:
    text = [
        "# Causal LSTM Model-Switch Arbitration Replay",
        "",
        "## Status",
        "",
        "- Research-only offline experiment. No live config, DB, scheduler, model state, `results/live`, `results/analysis`, or Champion artifact was written.",
        "- Input is the byte-copied 2026-07-29 snapshot. The production Base 60/40/trim20_lcb10 decision was rebuilt through the read-only function.",
        f"- Base reproduction: `{base_reproduction['checked_days'] - base_reproduction['mismatch_days']}/{base_reproduction['checked_days']}` exact id/reason/gap matches.",
        f"- Selected architecture: `{selected.trial.trial_id}`; the selected epoch count is `{selected.early_stop_result.best_epoch}` from an internal chronological train-only early-stop split.",
        "- All eight trials were ranked only by validation two-contrast RMSE. The final holdout was excluded from trial, epoch, and seed selection.",
        "",
        "## Fixed contract",
        "",
        "- `path44_state_only`: 44 frozen T1430 features, with a state sequence ending on the score day. Same-day return is a target only, never an input.",
        "- Target is two Helmert contrasts of jointly clipped full-liquidation shadow returns. The resulting three-model utility forecast is algebraically coherent.",
        "- Missing/nonfinite state rows break a sequence; no imputation, interpolation, or cross-gap sequence is permitted.",
        "- Primary route envelope and direct candidate-versus-Base LCB are unchanged from the prior research. A model-internal top1-top2 gap is diagnostic only.",
        "",
        "## Validation grid (selection period only)",
        "",
        *_csv_block(grid),
        "",
        "## Final chronological holdout selector metrics",
        "",
        *_csv_block(final_summary),
        "",
        "## Final holdout forecast diagnostics",
        "",
        *_csv_block(final_prediction),
        "",
        "## Final holdout direct-pair calibration",
        "",
        *_csv_block(final_calibration),
        "",
        "## Limitations",
        "",
        "- The state artifact has a confirmed 14:29/14:30 provenance limitation. This replay does not establish strict 14:29 PIT compliance.",
        "- The final holdout is historical, not prospective shadow evidence. It must not be promoted to live or used to change the confidence allowance without a separate approved workflow.",
        "- This is one predeclared eight-trial development grid. A later `disp7 + lagged utility` representation, if desired, must be a separately registered experiment rather than a post-result modification.",
        "- The current full-liquidation contract remains unchanged; no partial-turnover or continuous holding/switch cost was introduced.",
        "",
        "## Artifacts",
        "",
        "- `resolved_lstm_plan.json`, `input_manifest.json`, `daily_base_snapshots.csv`, `training_events.csv`, and `checkpoints/`",
        "- `validation_grid_metrics.csv`, `daily_validation_forecasts.csv`, `daily_validation_cross_arbitration.csv`",
        "- `daily_lstm_forecasts.csv`, `daily_cross_arbitration.csv`, `prediction_oos_metrics.csv`, `direct_pair_calibration.csv`",
        "- `selector_summary.csv`, `conditional_metrics.csv`, `chronological_slices.csv`, `nav_compare.png`",
    ]
    (run_root / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _write_nav_plot(run_root: Path, arbitration: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = arbitration.sort_values("score_day").reset_index(drop=True)
    base_returns = np.where(
        ordered["route_eligible"].astype(bool),
        ordered["base_return"].to_numpy(dtype=float),
        ordered["current_return"].to_numpy(dtype=float),
    )
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(ordered["score_day"], np.cumprod(1.0 + ordered["current_return"].to_numpy(dtype=float)), label="frozen_current", linewidth=2.0, color="black")
    ax.plot(ordered["score_day"], np.cumprod(1.0 + base_returns), label="base_fallback", linewidth=1.2, color="grey")
    ax.plot(ordered["score_day"], np.cumprod(1.0 + ordered["selected_return"].to_numpy(dtype=float)), label=str(ordered["forecaster_id"].iloc[0]), linewidth=1.4)
    ax.set_title("Research-only LSTM arbitration: final chronological holdout")
    ax.set_ylabel("NAV")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_root / "nav_compare.png", dpi=160)
    plt.close(fig)


def _write_outputs(
    run_root: Path,
    *,
    base_snapshots: pd.DataFrame,
    grid: pd.DataFrame,
    selected: ValidationTrialResult,
    validation_forecasts: pd.DataFrame,
    validation_arbitration: pd.DataFrame,
    final: FinalHoldoutResult,
    final_summary: pd.DataFrame,
    conditional: pd.DataFrame,
    slices: pd.DataFrame,
    prediction: pd.DataFrame,
    calibration: pd.DataFrame,
    training_events: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> None:
    artifacts = {
        "daily_base_snapshots.csv": base_snapshots,
        "validation_grid_metrics.csv": grid,
        "daily_validation_forecasts.csv": validation_forecasts,
        "daily_validation_cross_arbitration.csv": validation_arbitration,
        "daily_lstm_forecasts.csv": final.forecasts,
        "daily_cross_arbitration.csv": final.arbitration,
        "selector_summary.csv": final_summary,
        "conditional_metrics.csv": conditional,
        "chronological_slices.csv": slices,
        "prediction_oos_metrics.csv": prediction,
        "direct_pair_calibration.csv": calibration,
        "training_events.csv": training_events,
    }
    for name, frame in artifacts.items():
        frame.to_csv(run_root / name, index=False)
    (run_root / "resolved_lstm_plan.json").write_text(
        json.dumps(LSTM_PLAN, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (run_root / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    _write_nav_plot(run_root, final.arbitration)
    _write_summary(
        run_root,
        selected=selected,
        grid=grid,
        final_summary=final_summary,
        final_prediction=prediction,
        final_calibration=calibration,
        base_reproduction=manifest["base_reproduction"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT, help="frozen byte-copied research input snapshot")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT, help="must stay beneath the approved LSTM experiments root")
    parser.add_argument("--run-name", default=None, help="optional new leaf directory; an existing path is refused")
    parser.add_argument("--no-strict-baseline-reproduction", action="store_true", help="diagnostic only; never use for a promotable result")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    input_root = args.input_root.resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"frozen input root does not exist: {input_root}")
    output_root = _assert_experiment_output_root(args.output_root)
    run_name = _require_run_name(args.run_name)
    run_root = output_root / run_name
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = run_root / "checkpoints"

    prepared = _prepare_inputs(input_root)
    masks = _validate_split_contract(prepared)
    contexts, base_snapshots, base_reproduction = _base_contexts(
        prepared.aligned,
        switch_cfg=prepared.switch_cfg,
        strict=not bool(args.no_strict_baseline_reproduction),
    )
    if int(base_reproduction["checked_days"]) != len(prepared.aligned) or int(base_reproduction["mismatch_days"]) != 0:
        raise RuntimeError(f"Base reproduction is incomplete: {base_reproduction}")

    validation_results: list[ValidationTrialResult] = []
    for trial in small_lstm_grid(seed=int(LSTM_PLAN["training"]["seed"])):
        print(f"[LSTM validation] {trial.trial_id}", flush=True)
        validation_results.append(
            _run_validation_trial(
                prepared=prepared,
                masks=masks,
                contexts=contexts,
                trial=trial,
                checkpoints_dir=checkpoints_dir,
            )
        )
    grid, selected = _grid_table(validation_results)
    print(f"[LSTM selected] {selected.trial.trial_id}", flush=True)

    final = _run_final_holdout(
        prepared=prepared,
        masks=masks,
        contexts=contexts,
        selected=selected,
        checkpoints_dir=checkpoints_dir,
    )
    final_summary, conditional, slices = _selector_summary(final.arbitration, period="final_holdout")
    final_prediction = _pair_prediction_diagnostics(final.forecasts)
    final_calibration = _direct_pair_calibration(final.arbitration)
    validation_forecasts = pd.concat([result.validation_forecasts for result in validation_results], ignore_index=True)
    validation_arbitration = pd.concat([result.validation_arbitration for result in validation_results], ignore_index=True)
    training_events = pd.DataFrame([event for result in validation_results for event in result.training_events] + [final.training_event])
    elapsed_seconds = float((datetime.now(timezone.utc) - started_at).total_seconds())
    manifest = _input_manifest(
        input_root=input_root,
        prepared=prepared,
        base_reproduction=base_reproduction,
        grid=grid,
        selected=selected,
        final=final,
        elapsed_seconds=elapsed_seconds,
    )
    _write_outputs(
        run_root,
        base_snapshots=base_snapshots,
        grid=grid,
        selected=selected,
        validation_forecasts=validation_forecasts,
        validation_arbitration=validation_arbitration,
        final=final,
        final_summary=final_summary,
        conditional=conditional,
        slices=slices,
        prediction=final_prediction,
        calibration=final_calibration,
        training_events=training_events,
        manifest=manifest,
    )
    selected_metrics = grid.loc[grid["selected_for_final_holdout"].astype(bool)].iloc[0]
    print(
        json.dumps(
            {
                "run_root": str(run_root),
                "selected_trial": selected.trial.trial_id,
                "validation_two_contrast_rmse_bp": float(selected_metrics["two_contrast_rmse_bp"]),
                "final_selector": final_summary.to_dict(orient="records"),
                "base_reproduction": base_reproduction,
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
