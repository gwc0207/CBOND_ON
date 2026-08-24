from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from harness.model_switch_temporal_arbitration import contrasts_to_utilities, utilities_to_contrasts
from harness.tools.model_switch_relative_return_replay import (
    MODELS,
    build_score_disagreement_features,
    predict_coherent_ridge,
    predict_pairwise_logit,
    relative_utilities_from_returns,
    run_selector,
    select_highest_utility,
)


def test_relative_target_is_translation_invariant_and_coherent() -> None:
    original = np.asarray([0.003, -0.001, 0.002])
    shifted = original + 0.041
    utilities = relative_utilities_from_returns(original)
    assert np.allclose(utilities, relative_utilities_from_returns(shifted), atol=1e-15, rtol=0.0)
    assert math.isclose(float(utilities.sum()), 0.0, abs_tol=1e-15)
    assert np.allclose(contrasts_to_utilities(utilities_to_contrasts(utilities)), utilities, atol=1e-15, rtol=0.0)


def test_direct_argmax_has_no_threshold_or_basegap_dependency() -> None:
    assert select_highest_utility(np.asarray([-0.001, 0.000001, 0.0])) == 1
    assert select_highest_utility(np.asarray([1.0, 1.0, 0.0])) == 0
    assert select_highest_utility(np.asarray([1.0, np.nextafter(1.0, math.inf), 0.0])) == 1


def test_coherent_ridge_prediction_is_sum_zero_and_pairwise_transitive() -> None:
    values = np.linspace(-1.0, 1.0, 140)
    history_x = np.column_stack([values, values**2])
    common = 0.0002 * np.sin(values)
    history_returns = np.column_stack(
        [common + 0.0010 * values, common - 0.0007 * values, common + 0.00015 * values]
    )
    forecast = predict_coherent_ridge(history_x, history_returns, np.asarray([0.9, 0.81]))
    assert math.isclose(float(forecast.utilities.sum()), 0.0, abs_tol=1e-12)
    assert np.allclose(forecast.pairwise + forecast.pairwise.T, 0.0, atol=1e-12, rtol=0.0)
    assert math.isclose(
        float(forecast.pairwise[0, 1] + forecast.pairwise[1, 2]),
        float(forecast.pairwise[0, 2]),
        abs_tol=1e-12,
    )


def test_pairwise_ranker_is_direct_argmax_without_a_confidence_gate() -> None:
    values = np.linspace(-1.0, 1.0, 140)
    history_x = np.column_stack([values, values**2])
    history_returns = np.column_stack([0.001 * values, -0.001 * values, 0.0001 * np.sin(values)])
    forecast = predict_pairwise_logit(history_x, history_returns, np.asarray([0.8, 0.64]))
    assert math.isclose(float(forecast.utilities.sum()), 0.0, abs_tol=1e-12)
    assert select_highest_utility(forecast.utilities) == 0


def _synthetic_panel(days: int = 170) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        x = math.sin(index / 7.0)
        common = 0.0003 * math.cos(index / 11.0)
        rows.append(
            {
                "score_day": score_day,
                "Regsim": common + 0.0008 * x,
                "Ensemble": common - 0.0006 * x,
                "HL20": common + 0.0002 * math.sin(index / 5.0),
                "state_feature": x,
                "basegap_selected_name": "Regsim",
            }
        )
    return pd.DataFrame(rows)


def test_replay_keeps_t_decision_independent_of_same_and_future_outcomes() -> None:
    panel = _synthetic_panel()
    cutoff = panel.loc[145, "score_day"]
    original = run_selector(panel, variant="fixture", feature_cols=["state_feature"])
    poisoned = panel.copy()
    mask = poisoned["score_day"] >= cutoff
    poisoned.loc[mask, list(MODELS)] = 99.0
    poisoned.loc[poisoned["score_day"] > cutoff, "state_feature"] = -999.0
    replayed = run_selector(poisoned, variant="fixture", feature_cols=["state_feature"])
    columns = [
        "score_day",
        "selected_name",
        "selected_model_id",
        "training_rows",
        "training_end",
        "prediction_ready",
        "utility_regsim",
        "utility_ensemble",
        "utility_hl20",
    ]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, columns].reset_index(drop=True),
        replayed.loc[replayed["score_day"] <= cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )
    ready = original.loc[original["prediction_ready"]]
    assert (ready["training_end"] < ready["score_day"]).all()


def _write_scores(root, model: str, score_day: str, values: list[float]) -> None:
    path = root / model / score_day[:7] / f"{score_day}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"trade_date": score_day, "code": [f"110{i:03d}.SH" for i in range(len(values))], "score": values}
    ).to_csv(path, index=False)


def test_score_geometry_uses_only_aligned_current_and_previous_scores(tmp_path) -> None:
    root = tmp_path / "scores"
    dates = ["2024-01-02", "2024-01-04", "2024-01-05"]
    base = list(np.arange(25, dtype=float))
    for offset, score_day in enumerate(dates):
        _write_scores(root, "Regsim", score_day, base)
        _write_scores(root, "Ensemble", score_day, list(reversed(base)) if offset == 1 else base)
        _write_scores(root, "HL20", score_day, list(np.roll(base, offset)))
    before = build_score_disagreement_features(root, dates[:2])
    _write_scores(root, "Ensemble", dates[2], [999.0] * 25)
    after = build_score_disagreement_features(root, dates[:2])
    assert pd.isna(before.loc[0, "score_prev_top20_jaccard_Regsim"])
    assert np.isfinite(before.loc[1].drop(labels=["score_day"]).to_numpy(dtype=float)).all()
    assert_frame_equal(before, after, check_dtype=False)
