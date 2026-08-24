from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from harness.tools.model_switch_basegap_state_family_replay import (
    causal_robust_z,
    daily_factor_statistics,
    run_block_basegap_variant,
)
from harness.tools.model_switch_basegap_tuning_replay import MODELS, run_basegap_variant


PATH = ("path_a", "path_b")


def _panel(days: int = 32) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        x = math.sin(index / 3.0)
        rows.append(
            {
                "score_day": score_day,
                "Regsim": 0.0012 * x,
                "Ensemble": -0.0008 * x + 0.00005,
                "HL20": 0.0003 * math.cos(index / 2.0),
                "path_a": x,
                "path_b": x * x,
                "execution_metadata_complete": True,
                "state_available": True,
            }
        )
    return pd.DataFrame(rows)


def test_path_only_block_distance_matches_single_block_basegap() -> None:
    panel = _panel()
    direct = run_basegap_variant(
        panel,
        feature_cols=PATH,
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="trim20_lcb10",
        variant="direct",
    )
    block = run_block_basegap_variant(
        panel,
        blocks={"path": PATH},
        variant="block",
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="trim20_lcb10",
    )
    columns = [
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
    assert_frame_equal(direct[columns], block[columns], check_dtype=False, atol=1e-14, rtol=0.0)


def test_block_replay_has_strict_predecessor_and_no_future_outcome_dependency() -> None:
    panel = _panel()
    cutoff = panel.loc[20, "score_day"]
    original = run_block_basegap_variant(
        panel,
        blocks={"path": PATH},
        variant="fixture",
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="mean",
    )
    poisoned = panel.copy()
    poisoned.loc[poisoned["score_day"] >= cutoff, list(MODELS)] = 99.0
    poisoned.loc[poisoned["score_day"] > cutoff, list(PATH)] = -999.0
    replayed = run_block_basegap_variant(
        poisoned,
        blocks={"path": PATH},
        variant="fixture",
        lookback_days=10,
        nearest_k=4,
        min_periods=4,
        metric="mean",
    )
    columns = ["score_day", "basegap_selected_name", "basegap_score_gap", "neighbor_min_day", "neighbor_max_day"]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, columns].reset_index(drop=True),
        replayed.loc[replayed["score_day"] <= cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )
    active = original.loc[original["selection_active"]]
    assert (pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all()


def test_causal_robust_z_does_not_read_future_values() -> None:
    raw = pd.DataFrame({"x": np.arange(20, dtype=float), "y": np.arange(20, dtype=float) ** 2})
    cutoff = 13
    original = causal_robust_z(raw, lookback=8, min_periods=4)
    poisoned = raw.copy()
    poisoned.loc[cutoff + 1 :, :] = 999999.0
    replayed = causal_robust_z(poisoned, lookback=8, min_periods=4)
    assert_frame_equal(original.iloc[: cutoff + 1], replayed.iloc[: cutoff + 1], check_dtype=False)


def test_factor_statistics_produce_compact_return_free_structure_fields() -> None:
    rng = np.random.default_rng(7)
    factors = [f"f{index:02d}" for index in range(50)]
    frame = pd.DataFrame(rng.normal(size=(120, 50)), columns=factors)
    frame.iloc[:3, :2] = np.nan
    stats, medians, log_iqr = daily_factor_statistics(frame, factors)
    assert len(stats) == 8
    assert stats["factor_universe_n"] == 120.0
    assert 0.0 < stats["factor_cell_missing_rate"] < 1.0
    assert len(medians) == 50 and len(log_iqr) == 50
    assert math.isfinite(stats["factor_absrho_mean_all"])
    assert math.isfinite(stats["factor_absrho_mean_cross_legacy_mined"])


def test_duplicate_block_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="more than one block"):
        run_block_basegap_variant(
            _panel(),
            blocks={"one": ("path_a",), "two": ("path_a",)},
            variant="bad",
            lookback_days=10,
            nearest_k=4,
            min_periods=4,
            metric="mean",
        )
