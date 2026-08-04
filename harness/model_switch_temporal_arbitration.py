"""Pure, research-only primitives for temporal model-switch arbitration.

This module intentionally has no filesystem, configuration, scheduler, or
database access.  It is used only by the offline replay in
``harness/tools/model_switch_temporal_arbitration_replay.py``.  In particular,
it does *not* alter the live ``scoreopt_t1430_fusion_gate`` policy.

The central action test compares a temporal candidate directly with the Base
winner.  A temporal model's internal top-one/top-two gap is recorded as a
diagnostic, never used as the override condition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


SQRT2 = math.sqrt(2.0)
SQRT6 = math.sqrt(6.0)
HELMERT = np.asarray(
    [
        [1.0 / SQRT2, -1.0 / SQRT2, 0.0],
        [1.0 / SQRT6, 1.0 / SQRT6, -2.0 / SQRT6],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TemporalForecast:
    """A causally produced, three-expert temporal forecast.

    ``utilities`` supplies the forecast ranking.  ``pair_mean[i, j]`` is the
    direct expected advantage of model ``i`` over model ``j``; it may differ
    slightly from the utility difference for the legacy independent-pair Ridge
    only, because that legacy model is not algebraically coherent.
    """

    forecaster_id: str
    utilities: np.ndarray
    pair_mean: np.ndarray
    pair_scale: np.ndarray
    history_days: int
    ready: bool
    diagnostics: Mapping[str, float | int | str]


@dataclass(frozen=True)
class CrossArbitration:
    """The audit payload for one direct Base-versus-temporal decision."""

    selected_index: int
    candidate_index: int | None
    action: str
    reason: str
    base_score_candidate_minus_base: float | None
    base_pair_scale: float | None
    temporal_mean_candidate_minus_base: float | None
    temporal_scale: float | None
    temporal_calibration_scale: float | None
    combined_mean: float | None
    combined_scale: float | None
    lcb: float | None
    base_rank_of_candidate: int | None
    forecast_rank_of_base: int | None
    forecast_rank_of_candidate: int | None
    third_index: int | None
    base_score_third_minus_base: float | None
    temporal_mean_third_minus_base: float | None
    temporal_mean_candidate_minus_third: float | None


def _as_three(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_history(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def joint_clip_utilities(values: Sequence[float] | np.ndarray, *, target_clip: float) -> tuple[np.ndarray, float]:
    """Centre and jointly scale three returns without breaking transitivity."""

    if target_clip <= 0.0:
        raise ValueError("target_clip must be positive")
    raw = _as_three(values, name="values")
    centred = raw - float(raw.mean())
    pairwise_max = float(np.max(np.abs(centred[:, None] - centred[None, :])))
    scale = min(1.0, float(target_clip) / pairwise_max) if pairwise_max > 0.0 else 1.0
    return centred * scale, float(scale)


def utilities_to_contrasts(utilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map a sum-zero three-expert vector into the two Helmert contrasts."""

    vector = _as_three(utilities, name="utilities")
    if not math.isclose(float(vector.sum()), 0.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("utilities must sum to zero")
    return HELMERT @ vector


def contrasts_to_utilities(contrasts: Sequence[float] | np.ndarray) -> np.ndarray:
    """Invert the two Helmert contrasts into coherent sum-zero utilities."""

    vector = np.asarray(contrasts, dtype=float)
    if vector.shape != (2,):
        raise ValueError(f"contrasts must have shape (2,), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("contrasts must be finite")
    return HELMERT.T @ vector


def pairwise_from_utilities(utilities: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = _as_three(utilities, name="utilities")
    return vector[:, None] - vector[None, :]


def ordered_indices(
    scores: Sequence[float] | np.ndarray,
    *,
    preferred_index: int | None = None,
) -> tuple[int, int, int]:
    """Stable descending ranks, preferring Base only on numerical ties."""

    values = _as_three(scores, name="scores")
    if preferred_index is not None and preferred_index not in (0, 1, 2):
        raise ValueError("preferred_index must be in [0, 2]")
    indices = list(range(3))
    indices.sort(
        key=lambda idx: (
            -float(values[idx]),
            0 if preferred_index is not None and idx == preferred_index else 1,
            idx,
        )
    )
    return tuple(indices)  # type: ignore[return-value]


def rank_of(index: int, scores: Sequence[float] | np.ndarray, *, preferred_index: int | None = None) -> int:
    return int(ordered_indices(scores, preferred_index=preferred_index).index(index) + 1)


def weighted_mean_covariance(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return weighted mean, unbiased weighted covariance, and effective n."""

    matrix = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(weight):
        raise ValueError("values and weights have incompatible shapes")
    if not len(weight) or not np.isfinite(matrix).all() or not np.isfinite(weight).all() or np.any(weight < 0.0):
        raise ValueError("weighted statistics require finite non-negative inputs")
    total = float(weight.sum())
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    normalized = weight / total
    mean = np.sum(matrix * normalized[:, None], axis=0)
    effective_n = float(1.0 / np.sum(normalized**2))
    centred = matrix - mean
    denominator = 1.0 - float(np.sum(normalized**2))
    if denominator <= 1e-12:
        covariance = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    else:
        covariance = (centred * normalized[:, None]).T @ centred / denominator
    return mean, 0.5 * (covariance + covariance.T), effective_n


def exponential_weights(length: int, *, half_life: float) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    if half_life <= 0.0:
        raise ValueError("half_life must be positive")
    ages = np.arange(length - 1, -1, -1, dtype=float)
    return np.exp(math.log(0.5) * ages / float(half_life))


def ewma_utility_forecast(history_utilities: np.ndarray, *, half_life: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Predict coherent utilities using only realised history before today."""

    history = _as_history(history_utilities, name="history_utilities")
    weights = exponential_weights(len(history), half_life=half_life)
    mean, covariance, effective_n = weighted_mean_covariance(history, weights)
    return mean, covariance, effective_n


def _psd_matrix(matrix: np.ndarray, *, variance_floor: float) -> np.ndarray:
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, float(variance_floor))
    fixed = vectors @ np.diag(values) @ vectors.T
    return 0.5 * (fixed + fixed.T)


def shrunk_observation_covariance(
    history_contrasts: np.ndarray,
    *,
    shrinkage: float,
    std_floor: float,
) -> np.ndarray:
    """Causal 2D observation covariance with spherical shrinkage."""

    history = np.asarray(history_contrasts, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or len(history) < 2:
        raise ValueError("history_contrasts must have shape (n>=2, 2)")
    if not np.isfinite(history).all():
        raise ValueError("history_contrasts must be finite")
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if std_floor <= 0.0:
        raise ValueError("std_floor must be positive")
    covariance = np.cov(history, rowvar=False, ddof=1)
    spherical = np.eye(2, dtype=float) * float(np.trace(covariance) / 2.0)
    result = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * spherical
    return _psd_matrix(result, variance_floor=float(std_floor) ** 2)


def _kalman_update(
    level: np.ndarray,
    covariance: np.ndarray,
    observation: np.ndarray,
    observation_covariance: np.ndarray,
    process_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One Joseph-form local-level update; return posterior and predictive F."""

    prior_covariance = _psd_matrix(covariance + process_covariance, variance_floor=1e-16)
    predictive_covariance = _psd_matrix(prior_covariance + observation_covariance, variance_floor=1e-16)
    gain = prior_covariance @ np.linalg.inv(predictive_covariance)
    posterior_level = level + gain @ (observation - level)
    identity = np.eye(2, dtype=float)
    posterior_covariance = (identity - gain) @ prior_covariance @ (identity - gain).T + gain @ observation_covariance @ gain.T
    return posterior_level, _psd_matrix(posterior_covariance, variance_floor=1e-16), predictive_covariance


def local_level_forecast(
    history_contrasts: np.ndarray,
    *,
    process_ratio: float,
    shrinkage: float = 0.25,
    std_floor: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, float | int]]:
    """Causal 2D local-level one-step forecast from strictly prior outcomes."""

    history = np.asarray(history_contrasts, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or len(history) < 2:
        raise ValueError("history_contrasts must have shape (n>=2, 2)")
    if process_ratio < 0.0:
        raise ValueError("process_ratio must be non-negative")
    observation_covariance = shrunk_observation_covariance(history, shrinkage=shrinkage, std_floor=std_floor)
    process_covariance = float(process_ratio) * observation_covariance
    level = history[0].copy()
    covariance = observation_covariance.copy()
    for observation in history:
        level, covariance, _ = _kalman_update(
            level,
            covariance,
            observation,
            observation_covariance,
            process_covariance,
        )
    predictive_covariance = _psd_matrix(covariance + process_covariance + observation_covariance, variance_floor=1e-16)
    return level, predictive_covariance, {
        "history_days": int(len(history)),
        "process_ratio": float(process_ratio),
        "observation_cov_trace": float(np.trace(observation_covariance)),
    }


def _logpdf_zero_mean(error: np.ndarray, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not math.isfinite(float(logdet)):
        raise ValueError("predictive covariance is not positive definite")
    quadratic = float(error.T @ np.linalg.solve(covariance, error))
    return float(-0.5 * (2.0 * math.log(2.0 * math.pi) + float(logdet) + quadratic))


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    return shifted / float(shifted.sum())


def dma_local_level_forecast(
    history_contrasts: np.ndarray,
    *,
    process_ratios: Sequence[float] = (0.0, 0.02, 0.10),
    forgetting: float = 0.99,
    shrinkage: float = 0.25,
    std_floor: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, float | int]]:
    """Dynamic model averaging across fixed local-level process variances."""

    history = np.asarray(history_contrasts, dtype=float)
    ratios = np.asarray(process_ratios, dtype=float)
    if history.ndim != 2 or history.shape[1] != 2 or len(history) < 2:
        raise ValueError("history_contrasts must have shape (n>=2, 2)")
    if not len(ratios) or not np.isfinite(ratios).all() or np.any(ratios < 0.0):
        raise ValueError("process_ratios must be finite non-negative values")
    if not 0.0 < forgetting <= 1.0:
        raise ValueError("forgetting must be in (0, 1]")

    observation_covariance = shrunk_observation_covariance(history, shrinkage=shrinkage, std_floor=std_floor)
    count = len(ratios)
    levels = np.repeat(history[[0]], count, axis=0)
    covariances = np.repeat(observation_covariance[None, :, :], count, axis=0)
    weights = np.full(count, 1.0 / count, dtype=float)

    for observation in history:
        prior_weights = weights**float(forgetting)
        prior_weights = prior_weights / float(prior_weights.sum())
        log_weights: list[float] = []
        updated_levels: list[np.ndarray] = []
        updated_covariances: list[np.ndarray] = []
        for index, ratio in enumerate(ratios):
            process_covariance = float(ratio) * observation_covariance
            prior_covariance = _psd_matrix(covariances[index] + process_covariance, variance_floor=1e-16)
            predictive_covariance = _psd_matrix(prior_covariance + observation_covariance, variance_floor=1e-16)
            log_weights.append(math.log(float(prior_weights[index])) + _logpdf_zero_mean(observation - levels[index], predictive_covariance))
            level, covariance, _ = _kalman_update(
                levels[index],
                covariances[index],
                observation,
                observation_covariance,
                process_covariance,
            )
            updated_levels.append(level)
            updated_covariances.append(covariance)
        weights = _normalize_log_weights(np.asarray(log_weights, dtype=float))
        levels = np.asarray(updated_levels, dtype=float)
        covariances = np.asarray(updated_covariances, dtype=float)

    prediction_weights = weights**float(forgetting)
    prediction_weights = prediction_weights / float(prediction_weights.sum())
    component_covariances = np.asarray(
        [
            _psd_matrix(covariances[index] + float(ratio) * observation_covariance + observation_covariance, variance_floor=1e-16)
            for index, ratio in enumerate(ratios)
        ],
        dtype=float,
    )
    mean = np.sum(levels * prediction_weights[:, None], axis=0)
    covariance = np.zeros((2, 2), dtype=float)
    for index in range(count):
        delta = levels[index] - mean
        covariance += float(prediction_weights[index]) * (component_covariances[index] + np.outer(delta, delta))
    return mean, _psd_matrix(covariance, variance_floor=1e-16), prediction_weights, {
        "history_days": int(len(history)),
        "forgetting": float(forgetting),
        "observation_cov_trace": float(np.trace(observation_covariance)),
    }


def fixed_share_hedge_weights(
    history_utilities: np.ndarray,
    *,
    target_clip: float,
    eta: float,
    share: float,
) -> np.ndarray:
    """Full-information fixed-share Hedge weights after strictly prior days."""

    history = _as_history(history_utilities, name="history_utilities")
    if target_clip <= 0.0 or eta < 0.0 or not 0.0 <= share <= 1.0:
        raise ValueError("invalid fixed-share Hedge parameters")
    weights = np.full(3, 1.0 / 3.0, dtype=float)
    for utilities in history:
        reward = np.clip(utilities / float(target_clip), -1.0, 1.0)
        log_weight = np.log(np.maximum(weights, 1e-300)) + float(eta) * reward
        normalized = _normalize_log_weights(log_weight)
        weights = (1.0 - float(share)) * normalized + float(share) / 3.0
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RuntimeError("fixed-share Hedge produced invalid weights")
    return weights / float(weights.sum())


def pair_scale_from_utility_covariance(covariance: np.ndarray) -> np.ndarray:
    """Convert a coherent three-utility covariance to direct pair scales."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("utility covariance must have shape (3, 3)")
    result = np.zeros((3, 3), dtype=float)
    for left in range(3):
        for right in range(3):
            direction = np.zeros(3, dtype=float)
            direction[left] = 1.0
            direction[right] -= 1.0
            variance = float(direction @ matrix @ direction)
            result[left, right] = math.sqrt(max(0.0, variance))
    return result


def utility_covariance_from_contrast_covariance(covariance: np.ndarray) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("contrast covariance must have shape (2, 2)")
    return HELMERT.T @ matrix @ HELMERT


def make_utility_forecast(
    forecaster_id: str,
    utilities: Sequence[float] | np.ndarray,
    *,
    pair_mean: np.ndarray | None = None,
    pair_scale: np.ndarray | None = None,
    history_days: int,
    ready: bool,
    diagnostics: Mapping[str, float | int | str] | None = None,
) -> TemporalForecast:
    value = _as_three(utilities, name="utilities")
    mean = pairwise_from_utilities(value) if pair_mean is None else np.asarray(pair_mean, dtype=float)
    scale = np.full((3, 3), np.nan, dtype=float) if pair_scale is None else np.asarray(pair_scale, dtype=float)
    if mean.shape != (3, 3) or scale.shape != (3, 3):
        raise ValueError("pair_mean and pair_scale must both have shape (3, 3)")
    if not np.isfinite(mean).all():
        raise ValueError("pair_mean must be finite")
    if np.isfinite(scale).any() and np.nanmin(scale) < 0.0:
        raise ValueError("pair_scale must be non-negative")
    return TemporalForecast(
        forecaster_id=forecaster_id,
        utilities=value,
        pair_mean=mean,
        pair_scale=scale,
        history_days=int(history_days),
        ready=bool(ready),
        diagnostics=dict(diagnostics or {}),
    )


def cross_arbitrate(
    *,
    base_index: int,
    base_scores: Sequence[float] | np.ndarray | None,
    base_pair_scale: float | None,
    forecast: TemporalForecast,
    temporal_calibration_scale: float | None,
    z_value: float = 1.0,
    economic_floor: float = 0.0,
    scale_floor: float = 0.0001,
) -> CrossArbitration:
    """Apply the dynamic direct ``r-b`` LCB rule without a top-two gap gate."""

    if base_index not in (0, 1, 2):
        raise ValueError("base_index must be in [0, 2]")
    if z_value < 0.0 or scale_floor <= 0.0:
        raise ValueError("z_value must be non-negative and scale_floor positive")

    forecast_rank = ordered_indices(forecast.utilities, preferred_index=base_index)
    candidate = int(forecast_rank[0])
    if candidate == base_index:
        return CrossArbitration(
            selected_index=base_index,
            candidate_index=candidate,
            action="keep_base",
            reason="candidate_agrees_base",
            base_score_candidate_minus_base=0.0,
            base_pair_scale=None,
            temporal_mean_candidate_minus_base=0.0,
            temporal_scale=None,
            temporal_calibration_scale=temporal_calibration_scale,
            combined_mean=None,
            combined_scale=None,
            lcb=None,
            base_rank_of_candidate=1,
            forecast_rank_of_base=1,
            forecast_rank_of_candidate=1,
            third_index=None,
            base_score_third_minus_base=None,
            temporal_mean_third_minus_base=None,
            temporal_mean_candidate_minus_third=None,
        )

    if base_scores is None:
        return CrossArbitration(
            selected_index=base_index,
            candidate_index=candidate,
            action="keep_base",
            reason="base_scores_unavailable",
            base_score_candidate_minus_base=None,
            base_pair_scale=base_pair_scale,
            temporal_mean_candidate_minus_base=None,
            temporal_scale=None,
            temporal_calibration_scale=temporal_calibration_scale,
            combined_mean=None,
            combined_scale=None,
            lcb=None,
            base_rank_of_candidate=None,
            forecast_rank_of_base=rank_of(base_index, forecast.utilities, preferred_index=base_index),
            forecast_rank_of_candidate=rank_of(candidate, forecast.utilities, preferred_index=base_index),
            third_index=None,
            base_score_third_minus_base=None,
            temporal_mean_third_minus_base=None,
            temporal_mean_candidate_minus_third=None,
        )

    scores = _as_three(base_scores, name="base_scores")
    third = next(index for index in range(3) if index not in {base_index, candidate})
    base_mu = float(scores[candidate] - scores[base_index])
    temporal_mu = float(forecast.pair_mean[candidate, base_index]) if forecast.ready else float("nan")
    intrinsic_scale = float(forecast.pair_scale[candidate, base_index])
    available_temporal_scales = [
        value
        for value in (intrinsic_scale, temporal_calibration_scale)
        if value is not None and math.isfinite(float(value)) and float(value) >= 0.0
    ]
    temporal_scale = max(available_temporal_scales) if available_temporal_scales else None
    base_scale = None if base_pair_scale is None or not math.isfinite(float(base_pair_scale)) else max(float(base_pair_scale), scale_floor)
    if not forecast.ready or not math.isfinite(temporal_mu):
        reason = "temporal_forecast_unavailable"
    elif base_scale is None:
        reason = "base_pair_uncertainty_unavailable"
    elif temporal_scale is None:
        reason = "temporal_uncertainty_unavailable"
    else:
        temporal_scale = max(float(temporal_scale), scale_floor)
        temporal_precision = 1.0 / temporal_scale**2
        base_precision = 1.0 / base_scale**2
        temporal_weight = temporal_precision / (temporal_precision + base_precision)
        base_weight = 1.0 - temporal_weight
        combined_mean = temporal_weight * temporal_mu + base_weight * base_mu
        combined_scale = temporal_weight * temporal_scale + base_weight * base_scale
        lcb = combined_mean - float(z_value) * combined_scale
        selected = candidate if lcb > economic_floor else base_index
        return CrossArbitration(
            selected_index=selected,
            candidate_index=candidate,
            action="override_base" if selected == candidate else "keep_base",
            reason="cross_lcb_override_base" if selected == candidate else "cross_lcb_keep_base",
            base_score_candidate_minus_base=base_mu,
            base_pair_scale=base_scale,
            temporal_mean_candidate_minus_base=temporal_mu,
            temporal_scale=temporal_scale,
            temporal_calibration_scale=temporal_calibration_scale,
            combined_mean=combined_mean,
            combined_scale=combined_scale,
            lcb=lcb,
            base_rank_of_candidate=rank_of(candidate, scores),
            forecast_rank_of_base=rank_of(base_index, forecast.utilities, preferred_index=base_index),
            forecast_rank_of_candidate=rank_of(candidate, forecast.utilities, preferred_index=base_index),
            third_index=third,
            base_score_third_minus_base=float(scores[third] - scores[base_index]),
            temporal_mean_third_minus_base=float(forecast.pair_mean[third, base_index]),
            temporal_mean_candidate_minus_third=float(forecast.pair_mean[candidate, third]),
        )

    return CrossArbitration(
        selected_index=base_index,
        candidate_index=candidate,
        action="keep_base",
        reason=reason,
        base_score_candidate_minus_base=base_mu,
        base_pair_scale=base_scale,
        temporal_mean_candidate_minus_base=None if not math.isfinite(temporal_mu) else temporal_mu,
        temporal_scale=temporal_scale,
        temporal_calibration_scale=temporal_calibration_scale,
        combined_mean=None,
        combined_scale=None,
        lcb=None,
        base_rank_of_candidate=rank_of(candidate, scores),
        forecast_rank_of_base=rank_of(base_index, forecast.utilities, preferred_index=base_index),
        forecast_rank_of_candidate=rank_of(candidate, forecast.utilities, preferred_index=base_index),
        third_index=third,
        base_score_third_minus_base=float(scores[third] - scores[base_index]),
        temporal_mean_third_minus_base=float(forecast.pair_mean[third, base_index]) if forecast.ready else None,
        temporal_mean_candidate_minus_third=float(forecast.pair_mean[candidate, third]) if forecast.ready else None,
    )
