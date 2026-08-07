from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from cbond_on.infra.live.model_switch import SwitchDecision, _scoreopt_trim20_lcb10
from harness.tools.basegap_similarity_weight_replay import (
    distance_weights,
    weighted_base_decision,
    weighted_trim20_lcb10,
)


def _base_decision(*, distances: list[float]) -> SwitchDecision:
    score_day = date(2026, 2, 20)
    model_ids = ["champion", "challenger_a", "challenger_b"]
    names = ["Champion", "A", "B"]
    similar_days: list[dict[str, object]] = []
    returns: list[list[float]] = []
    for index, distance in enumerate(distances):
        values = [0.0030 + index * 0.0001, 0.0010 - index * 0.00005, -0.0004 + index * 0.00002]
        returns.append(values)
        similar_days.append(
            {
                "trade_date": score_day - timedelta(days=len(distances) - index),
                "distance": distance,
                "model_returns": [
                    {"model_id": model_id, "name": name, "day_return": value}
                    for model_id, name, value in zip(model_ids, names, values, strict=True)
                ],
            }
        )
    matrix = np.asarray(returns, dtype=float)
    details = [
        {
            "role": "champion" if index == 0 else "challenger",
            "model_id": model_id,
            "name": names[index],
            "score": _scoreopt_trim20_lcb10(pd.Series(matrix[:, index])),
            "history_end": score_day - timedelta(days=1),
            "history_days": len(distances),
        }
        for index, model_id in enumerate(model_ids)
    ]
    scores = [float(item["score"]) for item in details]
    ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    gap = scores[ranked[0]] - scores[ranked[1]]
    return SwitchDecision(
        enabled=True,
        mode="scoreopt_t1430_dispersion",
        metric="trim20_lcb10",
        lookback_days=60,
        min_periods=len(distances),
        threshold=0.0005,
        score_day=score_day,
        selected_model_id=model_ids[ranked[0]] if gap > 0.0005 else "champion",
        selected_name=names[ranked[0]] if gap > 0.0005 else "Champion",
        champion_model_id="champion",
        champion_name="Champion",
        challenger_model_id=model_ids[ranked[1]],
        challenger_name=names[ranked[1]],
        champion_score=scores[0],
        challenger_score=max(scores[1:]),
        score_diff=gap,
        history_end=score_day - timedelta(days=1),
        history_days=len(distances),
        reason="score_best" if gap > 0.0005 else "margin_default",
        candidate_scores=details,
        similar_days=similar_days,
    )


def test_uniform_weights_call_the_current_production_statistic_exactly() -> None:
    values = np.asarray([-0.012, -0.001, 0.0005, 0.003, 0.007, 0.009], dtype=float)
    weights = np.full(len(values), 1.0 / len(values), dtype=float)
    actual, audit = weighted_trim20_lcb10(values, weights)
    expected = _scoreopt_trim20_lcb10(pd.Series(values))
    assert actual == expected
    assert audit["uniform_compatibility"] is True
    assert math.isclose(float(audit["effective_sample_size"]), float(len(values)), abs_tol=1e-12)


def test_equal_distances_reduce_the_full_base_score_to_equal_weight() -> None:
    decision = _base_decision(distances=[2.0] * 40)
    weighted, audit = weighted_base_decision(decision)
    assert audit["equal_distance_fallback"] is True
    assert weighted.selected_model_id == decision.selected_model_id
    assert weighted.reason == decision.reason
    before = {item["model_id"]: item["score"] for item in decision.candidate_scores or []}
    after = {item["model_id"]: item["score"] for item in weighted.candidate_scores or []}
    assert after == before
    assert math.isclose(float(audit["effective_sample_size"]), 40.0, abs_tol=1e-12)


def test_nearer_distances_get_larger_finite_normalized_weights() -> None:
    weights, audit = distance_weights(np.asarray([0.5, 1.0, 2.0, 3.0], dtype=float))
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12)
    assert weights[0] > weights[1] > weights[2] > weights[3]
    assert math.isclose(float(audit["distance_median"]), 1.5, abs_tol=1e-12)
    assert float(audit["effective_sample_size"]) < 4.0


def test_weighted_base_refuses_a_noncausal_future_neighbour() -> None:
    decision = _base_decision(distances=[float(index + 1) for index in range(40)])
    assert decision.similar_days is not None
    decision.similar_days[0]["trade_date"] = decision.score_day
    with pytest.raises(ValueError, match="non-causal Base neighbour"):
        weighted_base_decision(decision)
