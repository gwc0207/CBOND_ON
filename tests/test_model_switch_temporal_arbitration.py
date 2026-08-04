from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from harness.model_switch_temporal_arbitration import (
    HELMERT,
    contrasts_to_utilities,
    cross_arbitrate,
    dma_local_level_forecast,
    fixed_share_hedge_weights,
    joint_clip_utilities,
    local_level_forecast,
    make_utility_forecast,
    pairwise_from_utilities,
    utilities_to_contrasts,
)
from harness.tools.model_switch_temporal_arbitration_replay import BaseContext, FEATURES, MODELS, _run_replay


def _pair_scales(value: float) -> np.ndarray:
    output = np.full((3, 3), value, dtype=float)
    np.fill_diagonal(output, 0.0)
    return output


def test_joint_clip_helmert_round_trip_and_pairwise_transitivity() -> None:
    utilities, scale = joint_clip_utilities(np.asarray([0.03, -0.01, 0.00]), target_clip=0.0075)
    assert scale < 1.0
    assert math.isclose(float(utilities.sum()), 0.0, abs_tol=1e-12)
    assert float(np.max(np.abs(utilities[:, None] - utilities[None, :]))) <= 0.0075 + 1e-12
    reconstructed = contrasts_to_utilities(utilities_to_contrasts(utilities))
    assert np.allclose(reconstructed, utilities, atol=1e-12, rtol=0.0)
    pairs = pairwise_from_utilities(utilities)
    assert np.allclose(pairs + pairs.T, 0.0, atol=1e-12, rtol=0.0)
    assert math.isclose(float(pairs[0, 1] + pairs[1, 2]), float(pairs[0, 2]), abs_tol=1e-12)


def test_fixed_share_weights_are_valid_and_outcome_updates_only_the_next_state() -> None:
    history = np.asarray(
        [
            [0.001, -0.001, 0.0],
            [-0.002, 0.001, 0.001],
            [0.003, -0.001, -0.002],
        ],
        dtype=float,
    )
    before = fixed_share_hedge_weights(history, target_clip=0.0075, eta=0.1, share=1.0 / 60.0)
    repeated = fixed_share_hedge_weights(history.copy(), target_clip=0.0075, eta=0.1, share=1.0 / 60.0)
    after = fixed_share_hedge_weights(
        np.vstack([history, np.asarray([0.006, -0.003, -0.003])]),
        target_clip=0.0075,
        eta=0.1,
        share=1.0 / 60.0,
    )
    assert np.allclose(before, repeated, atol=0.0, rtol=0.0)
    assert np.all(before >= 0.0)
    assert math.isclose(float(before.sum()), 1.0, abs_tol=1e-12)
    assert not np.allclose(before, after)


def test_local_level_and_dma_covariances_are_psd_and_dma_weights_sum_to_one() -> None:
    history_utilities = np.asarray(
        [
            [0.001, -0.001, 0.0],
            [-0.001, 0.002, -0.001],
            [0.002, -0.001, -0.001],
            [0.0005, 0.0002, -0.0007],
            [-0.0015, 0.001, 0.0005],
            [0.0003, -0.0006, 0.0003],
        ],
        dtype=float,
    )
    contrasts = history_utilities @ HELMERT.T
    _, local_covariance, _ = local_level_forecast(contrasts, process_ratio=0.02)
    _, dma_covariance, weights, _ = dma_local_level_forecast(contrasts)
    assert np.linalg.eigvalsh(local_covariance).min() >= -1e-12
    assert np.linalg.eigvalsh(dma_covariance).min() >= -1e-12
    assert np.all(weights >= 0.0)
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)


def test_cross_arbitration_overrides_on_direct_lcb_even_when_top_two_gap_is_below_5bp() -> None:
    # Temporal top-R versus Base has only a 2bp ranking gap.  The direct
    # r-b LCB nevertheless passes because both evidence sources support R.
    utilities = np.asarray([0.0, 0.00020, -0.00020])
    forecast = make_utility_forecast(
        "fixture",
        utilities,
        pair_scale=_pair_scales(0.000001),
        history_days=120,
        ready=True,
    )
    decision = cross_arbitrate(
        base_index=0,
        base_scores=np.asarray([0.1000000, 0.1000150, 0.0900000]),
        base_pair_scale=0.000001,
        forecast=forecast,
        temporal_calibration_scale=0.000001,
        z_value=1.0,
        economic_floor=0.0,
    )
    assert utilities[1] - utilities[0] < 0.0005
    assert decision.candidate_index == 1
    assert decision.selected_index == 1
    assert decision.action == "override_base"
    assert decision.reason == "cross_lcb_override_base"


def test_cross_arbitration_rejects_large_internal_gap_when_direct_evidence_is_not_positive() -> None:
    # The old top1-top2 gap would be 10bp here, but Base strongly contradicts
    # candidate R and the combined direct LCB is negative.
    utilities = np.asarray([0.0, 0.0010, -0.0010])
    forecast = make_utility_forecast(
        "fixture",
        utilities,
        pair_scale=_pair_scales(0.0001),
        history_days=120,
        ready=True,
    )
    decision = cross_arbitrate(
        base_index=0,
        base_scores=np.asarray([0.004, 0.000, -0.003]),
        base_pair_scale=0.0001,
        forecast=forecast,
        temporal_calibration_scale=0.0001,
        z_value=1.0,
        economic_floor=0.0,
    )
    assert utilities[1] - utilities[0] > 0.0005
    assert decision.candidate_index == 1
    assert decision.selected_index == 0
    assert decision.action == "keep_base"
    assert decision.reason == "cross_lcb_keep_base"


def test_cross_arbitration_records_candidate_base_rank_instead_of_using_base_top_two_gap() -> None:
    forecast = make_utility_forecast(
        "fixture",
        np.asarray([0.0, 0.002, -0.001]),
        pair_scale=_pair_scales(0.0001),
        history_days=120,
        ready=True,
    )
    decision = cross_arbitrate(
        base_index=0,
        # Candidate index 1 is Base's third-ranked model; this must be stored
        # as B_r-B_b rather than substituted with Base's top1-top2 gap.
        base_scores=np.asarray([0.005, 0.001, 0.003]),
        base_pair_scale=0.0001,
        forecast=forecast,
        temporal_calibration_scale=0.0001,
        z_value=1.0,
        economic_floor=0.0,
    )
    assert decision.candidate_index == 1
    assert decision.base_rank_of_candidate == 3
    assert math.isclose(decision.base_score_candidate_minus_base or 0.0, -0.004, abs_tol=1e-12)
    assert decision.forecast_rank_of_base == 2


def test_cross_arbitration_abstains_when_direct_base_scores_are_unavailable() -> None:
    forecast = make_utility_forecast(
        "fixture",
        np.asarray([0.0, 0.002, -0.001]),
        pair_scale=_pair_scales(0.0001),
        history_days=120,
        ready=True,
    )
    decision = cross_arbitrate(
        base_index=0,
        base_scores=None,
        base_pair_scale=None,
        forecast=forecast,
        temporal_calibration_scale=0.0001,
    )
    assert decision.selected_index == 0
    assert decision.reason == "base_scores_unavailable"


def test_replay_prefix_is_unchanged_when_only_future_state_and_returns_are_poisoned() -> None:
    """The sequential replay must never let t+1 data alter a t decision."""

    dates = pd.bdate_range("2024-01-02", periods=190).date
    rows: list[dict[str, object]] = []
    current_rows: list[dict[str, object]] = []
    contexts: dict[object, BaseContext] = {}
    neighbours = np.asarray(
        [[0.001 + i * 1e-6, 0.0004 - i * 1e-6, -0.0002] for i in range(40)],
        dtype=float,
    )
    for index, score_day in enumerate(dates):
        regsim = 0.001 * math.sin(index / 7.0)
        ensemble = 0.001 * math.cos(index / 11.0)
        hl20 = -0.0005 * math.sin(index / 13.0)
        row: dict[str, object] = {
            "score_day": score_day,
            "Regsim": regsim,
            "Ensemble": ensemble,
            "HL20": hl20,
            "_state_present": True,
        }
        for feature_index, feature in enumerate(FEATURES):
            row[feature] = math.sin(index / (feature_index + 3.0)) + feature_index * 1e-4
        rows.append(row)
        current_rows.append(
            {
                "score_day": score_day,
                "current_name": "Regsim",
                "base_name": "Regsim",
                "selected_return": regsim,
                "base_reason": "margin_default",
                "robust_reason": "margin_default",
                "robust_score_gap": 0.0,
            }
        )
        contexts[score_day] = BaseContext(
            score_day=score_day,
            base_name="Regsim",
            base_reason="margin_default",
            base_score_gap=0.0,
            scores=np.asarray([0.001, 0.0, -0.001]),
            neighbour_returns=neighbours,
            history_days=40,
        )

    panel = pd.DataFrame(rows)
    current = pd.DataFrame(current_rows)
    cutoff = dates[150]
    original_forecasts, original_decisions = _run_replay(panel, current, contexts)

    poisoned = panel.copy()
    future_mask = poisoned["score_day"] > cutoff
    for model in MODELS:
        poisoned.loc[future_mask, model] = 99.0
    for feature in FEATURES:
        poisoned.loc[future_mask, feature] = -999.0
    poisoned_current = current.copy()
    poisoned_current.loc[future_mask, "selected_return"] = poisoned.loc[future_mask, "Regsim"].to_numpy()
    poisoned_forecasts, poisoned_decisions = _run_replay(poisoned, poisoned_current, contexts)

    forecast_columns = [column for column in original_forecasts.columns if column != "score_day"]
    decision_columns = [column for column in original_decisions.columns if column != "score_day"]
    assert_frame_equal(
        original_forecasts.loc[original_forecasts["score_day"] <= cutoff, forecast_columns].reset_index(drop=True),
        poisoned_forecasts.loc[poisoned_forecasts["score_day"] <= cutoff, forecast_columns].reset_index(drop=True),
        check_dtype=False,
    )
    assert_frame_equal(
        original_decisions.loc[original_decisions["score_day"] <= cutoff, decision_columns].reset_index(drop=True),
        poisoned_decisions.loc[poisoned_decisions["score_day"] <= cutoff, decision_columns].reset_index(drop=True),
        check_dtype=False,
    )
