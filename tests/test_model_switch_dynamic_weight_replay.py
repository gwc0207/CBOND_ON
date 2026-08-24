from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from harness.tools.model_switch_dynamic_weight_replay import (
    MODELS,
    VARIANTS,
    build_weight_replay,
    causal_cross_model_scale,
    equal_weights,
    rank_blend_scores,
    softmax_weights,
    summarise_sleeve_proxy,
)


def _synthetic_source(days: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        signal = math.sin(index / 7.0)
        returns = np.asarray(
            [
                0.0004 + 0.0010 * signal,
                0.0004 - 0.0008 * signal,
                0.0004 + 0.0002 * math.cos(index / 5.0),
            ]
        )
        ready = index >= 5
        utility = np.asarray([0.0010 * signal, -0.0007 * signal, -0.0003 * signal])
        rows.append(
            {
                "score_day": score_day,
                "prediction_ready": ready,
                "training_end": dates[index - 1] if ready else None,
                "execution_metadata_complete": index < days - 2,
                "basegap_return": returns[0],
                **{f"realized_return_{model}": returns[position] for position, model in enumerate(MODELS)},
                **{f"utility_{model.lower()}": utility[position] if ready else np.nan for position, model in enumerate(MODELS)},
            }
        )
    return pd.DataFrame(rows)


def test_softmax_weights_are_simplex_and_not_a_winner_threshold() -> None:
    weights = softmax_weights(np.asarray([0.00001, 0.0, -0.00001]), scale=0.001)
    assert np.all(weights > 0.0)
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)
    assert weights[0] > weights[1] > weights[2]
    assert not np.allclose(weights, equal_weights())


def test_weight_replay_does_not_let_same_or_future_outcomes_change_t_weights() -> None:
    source = _synthetic_source()
    cutoff = source.loc[140, "score_day"]
    original = build_weight_replay(source)
    poisoned = source.copy()
    future = poisoned["score_day"] >= cutoff
    for model in MODELS:
        poisoned.loc[future, f"realized_return_{model}"] = 99.0
    replayed = build_weight_replay(poisoned)
    weight_columns = [
        "score_day",
        "causal_cross_model_scale",
        "forecast_reason",
        "history_ewm_reason",
        *[
            f"weight_{variant}_{model}"
            for variant in VARIANTS
            for model in MODELS
        ],
    ]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, weight_columns].reset_index(drop=True),
        replayed.loc[replayed["score_day"] <= cutoff, weight_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_forecast_unavailable_rows_use_equal_weight_not_regsim() -> None:
    replay = build_weight_replay(_synthetic_source())
    first = replay.iloc[0]
    for variant in ("forecast_softmax_causal_scale", "forecast_softmax_causal_scale_shrink50"):
        weights = np.asarray([first[f"weight_{variant}_{model}"] for model in MODELS], dtype=float)
        assert np.allclose(weights, equal_weights(), atol=1e-15, rtol=0.0)
    assert first["forecast_reason"] == "forecast_or_past_scale_unavailable_equal_weight"


def test_causal_scale_uses_only_cross_model_relative_returns() -> None:
    history = np.asarray([[0.001, -0.001, 0.0], [0.004, 0.002, 0.003]], dtype=float)
    shifted = history + 0.200
    assert math.isclose(
        causal_cross_model_scale(history) or 0.0,
        causal_cross_model_scale(shifted) or 0.0,
        abs_tol=1e-15,
    )


def test_rank_blend_is_score_level_rank_normalized_and_requires_same_universe() -> None:
    scores = {
        "Regsim": pd.Series([3.0, 2.0, 1.0], index=["A", "B", "C"]),
        "Ensemble": pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"]),
        "HL20": pd.Series([3.0, 1.0, 2.0], index=["A", "B", "C"]),
    }
    one_hot = rank_blend_scores(scores, [1.0, 0.0, 0.0], score_day="2024-01-02")
    assert one_hot.sort_values("score", ascending=False)["code"].tolist() == ["A", "B", "C"]
    mixed = rank_blend_scores(scores, [0.5, 0.5, 0.0], score_day="2024-01-02")
    assert math.isclose(float(mixed["score"].sum()), 2.0, abs_tol=1e-12)
    invalid = dict(scores)
    invalid["HL20"] = invalid["HL20"].drop(index="C")
    with pytest.raises(ValueError, match="identical model score universes"):
        rank_blend_scores(invalid, [1.0 / 3.0] * 3, score_day="2024-01-02")


def test_metadata_sensitivity_keeps_incomplete_suffix_visible() -> None:
    source = _synthetic_source(days=24)
    replay = build_weight_replay(source)
    _, _, _, diagnostics, sensitivity = summarise_sleeve_proxy(replay, source)
    assert set(sensitivity["scope"]) == {"all_return_rows", "complete_execution_metadata_only"}
    complete = sensitivity.loc[
        (sensitivity["scope"] == "complete_execution_metadata_only")
        & (sensitivity["strategy"] == "forecast_softmax_causal_scale")
    ].iloc[0]
    assert int(complete["days"]) == 22
    assert int(complete["excluded_metadata_days"]) == 2
    assert {"mean_weight_entropy", "mean_effective_models", "total_weight_turnover"}.issubset(diagnostics.columns)
