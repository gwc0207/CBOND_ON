from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from harness.tools.model_switch_basegap_scoring_geometry_replay import (
    BASELINE_VARIANT,
    DEFAULT_OUTPUT_ROOT,
    MODELS,
    _assert_output_root,
    _median_half_gaussian_weights,
    _target_sample,
    preregistered_variants,
    run_scoring_geometry_variant,
)
from harness.tools.model_switch_basegap_tuning_replay import run_basegap_variant


FEATURES = ("path_a", "path_b")


def _panel(days: int = 42) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        state = math.sin(index / 3.0) + index / 50.0
        rows.append(
            {
                "score_day": score_day,
                "Regsim": 0.0015 * state + 0.0002 * math.cos(index / 2.0),
                "Ensemble": -0.0010 * state + 0.00015 * math.sin(index / 4.0),
                "HL20": 0.0007 * math.cos(index / 5.0) - 0.00005 * state,
                "path_a": state,
                "path_b": state * state + 0.1 * math.cos(index),
                "state_available": True,
                "execution_metadata_complete": True,
            }
        )
    return pd.DataFrame(rows)


def _geometry(
    panel: pd.DataFrame,
    *,
    standardization: str = "zscore",
    weighting: str = "equal",
    target: str = "absolute",
) -> pd.DataFrame:
    return run_scoring_geometry_variant(
        panel,
        feature_cols=FEATURES,
        standardization=standardization,
        neighbor_weighting=weighting,
        target=target,
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="trim20_lcb10",
        variant="fixture",
    )


def test_preregistered_family_has_exactly_twelve_fixed_runs() -> None:
    variants = preregistered_variants()
    assert len(variants) == 12
    assert [row["variant"] for row in variants].count(BASELINE_VARIANT) == 1
    assert {
        (row["state_standardization"], row["neighbor_weighting"], row["target"])
        for row in variants
    } == {
        (standardization, weighting, target)
        for standardization in ("zscore", "median_mad")
        for weighting in ("equal", "median_half_gaussian")
        for target in ("absolute", "relative_equal_weight", "pairwise_mean")
    }
    assert {
        (row["lookback_days"], row["nearest_k"], row["min_periods"], row["metric"])
        for row in variants
    } == {(240, 60, 60, "trim20_lcb10")}


def test_zscore_equal_absolute_matches_stage_a_kernel_on_fixture() -> None:
    panel = _panel()
    stage_a_kernel = run_basegap_variant(
        panel,
        feature_cols=FEATURES,
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="trim20_lcb10",
        variant="stage_a_fixture",
    )
    geometry = _geometry(panel)
    fields = [
        "score_day",
        "basegap_selected_name",
        "basegap_reason",
        "basegap_history_days",
        "basegap_history_end",
        "basegap_score_gap",
        *(f"basegap_score_{model}" for model in MODELS),
        "neighbor_min_day",
        "neighbor_max_day",
    ]
    assert_frame_equal(
        geometry[fields], stage_a_kernel[fields], check_dtype=False, atol=1e-14, rtol=0.0
    )


def test_all_twelve_variants_run_with_strict_predecessor_neighbours() -> None:
    panel = _panel()
    outputs: list[pd.DataFrame] = []
    for spec in preregistered_variants():
        output = run_scoring_geometry_variant(
            panel,
            feature_cols=FEATURES,
            standardization=str(spec["state_standardization"]),
            neighbor_weighting=str(spec["neighbor_weighting"]),
            target=str(spec["target"]),
            lookback_days=10,
            nearest_k=4,
            min_periods=4,
            metric="trim20_lcb10",
            variant=str(spec["variant"]),
        )
        active = output.loc[output["selection_active"]]
        assert not active.empty
        assert (
            pd.to_datetime(active["neighbor_max_day"]).dt.date
            < pd.to_datetime(active["score_day"]).dt.date
        ).all()
        outputs.append(output)
    assert pd.concat(outputs, ignore_index=True)["variant"].nunique() == 12


def test_selection_does_not_read_current_or_future_outcomes_or_future_state() -> None:
    panel = _panel()
    cutoff = panel.loc[24, "score_day"]
    original = _geometry(panel, standardization="median_mad", weighting="median_half_gaussian", target="pairwise_mean")
    poisoned = panel.copy()
    poisoned.loc[poisoned["score_day"] >= cutoff, list(MODELS)] = 99.0
    poisoned.loc[poisoned["score_day"] > cutoff, list(FEATURES)] = -999.0
    replayed = _geometry(poisoned, standardization="median_mad", weighting="median_half_gaussian", target="pairwise_mean")
    fields = [
        "score_day",
        "basegap_selected_name",
        "basegap_reason",
        "basegap_history_days",
        "basegap_score_gap",
        *(f"basegap_score_{model}" for model in MODELS),
        "neighbor_min_day",
        "neighbor_max_day",
    ]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, fields].reset_index(drop=True),
        replayed.loc[replayed["score_day"] <= cutoff, fields].reset_index(drop=True),
        check_dtype=False,
        atol=1e-14,
        rtol=0.0,
    )


def test_relative_and_pairwise_targets_are_symmetric_zero_sum_contrasts() -> None:
    sample = pd.DataFrame(
        {
            "Regsim": [0.02, -0.01, 0.03],
            "Ensemble": [0.00, 0.04, -0.02],
            "HL20": [-0.01, 0.01, 0.00],
        }
    )
    relative = _target_sample(sample, target="relative_equal_weight")
    pairwise = _target_sample(sample, target="pairwise_mean")
    assert np.allclose(relative.sum(axis=1), 0.0, atol=1e-15)
    assert np.allclose(pairwise.sum(axis=1), 0.0, atol=1e-15)
    # With exactly three candidates, averaging two pairwise differences is
    # 1.5 times the equal-weight-relative contrast; it changes score scale,
    # not the candidate symmetry.
    assert_frame_equal(pairwise, relative * 1.5, check_dtype=False, atol=1e-15, rtol=0.0)


def test_median_half_gaussian_is_positive_and_halves_at_the_median_distance() -> None:
    weights = _median_half_gaussian_weights(pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30]))
    assert (weights > 0.0).all()
    assert weights.loc[20] == pytest.approx(0.5)
    assert weights.loc[10] > weights.loc[20] > weights.loc[30]


def test_output_root_rejects_non_research_paths(tmp_path: Path) -> None:
    allowed = _assert_output_root(DEFAULT_OUTPUT_ROOT / "unit_test_leaf")
    assert allowed.name == "unit_test_leaf"
    with pytest.raises(ValueError, match="research output"):
        _assert_output_root(tmp_path)
