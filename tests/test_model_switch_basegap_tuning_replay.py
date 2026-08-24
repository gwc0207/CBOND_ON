from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from harness.tools.model_switch_basegap_tuning_replay import (
    BASELINE_POINT,
    DEFAULT_OUTPUT_ROOT,
    MODELS,
    _assert_output_root,
    _scope_frame,
    parity_audit,
    preregistered_grid,
    run_basegap_variant,
)


FEATURES = ("state_one", "state_two")


def _fixture_panel(days: int = 24) -> pd.DataFrame:
    score_days = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(score_days):
        state = float(np.sin(index / 4.0))
        rows.append(
            {
                "score_day": score_day,
                "Regsim": 0.0012 * state + 0.0001 * np.cos(index),
                "Ensemble": -0.0009 * state + 0.0001 * np.sin(index),
                "HL20": 0.0005 * np.cos(index / 3.0),
                "state_one": state,
                "state_two": state * state,
                "execution_metadata_complete": True,
            }
        )
    return pd.DataFrame(rows)


def _replay(panel: pd.DataFrame) -> pd.DataFrame:
    return run_basegap_variant(
        panel,
        feature_cols=FEATURES,
        lookback_days=8,
        nearest_k=3,
        min_periods=3,
        metric="mean",
        variant="fixture",
    )


def test_preregistered_grid_is_bounded_and_contains_only_one_production_point() -> None:
    grid = preregistered_grid()
    assert len(grid) == 51
    baselines = [row for row in grid if row["is_production_baseline"]]
    assert len(baselines) == 1
    assert {key: baselines[0][key] for key in BASELINE_POINT} == BASELINE_POINT
    assert all(int(row["nearest_k"]) <= int(row["lookback_days"]) for row in grid)


def test_output_root_rejects_non_research_path(tmp_path: Path) -> None:
    allowed = _assert_output_root(DEFAULT_OUTPUT_ROOT / "unit_test_leaf")
    assert allowed.name == "unit_test_leaf"
    with pytest.raises(ValueError, match="research output"):
        _assert_output_root(tmp_path)


def test_replay_does_not_use_same_day_or_future_returns_or_state() -> None:
    panel = _fixture_panel()
    cutoff = panel.loc[14, "score_day"]
    original = _replay(panel)
    poisoned = panel.copy()
    poisoned.loc[poisoned["score_day"] >= cutoff, list(MODELS)] = 99.0
    poisoned.loc[poisoned["score_day"] > cutoff, list(FEATURES)] = -999.0
    replayed = _replay(poisoned)
    compare_columns = [
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
        original.loc[original["score_day"] <= cutoff, compare_columns].reset_index(drop=True),
        replayed.loc[replayed["score_day"] <= cutoff, compare_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_every_active_neighbour_is_strictly_predecessor() -> None:
    replay = _replay(_fixture_panel())
    active = replay.loc[replay["selection_active"]]
    assert not active.empty
    assert (pd.to_datetime(active["neighbor_max_day"]).dt.date < pd.to_datetime(active["score_day"]).dt.date).all()
    assert (active["neighbor_count"] == 3).all()


def test_missing_state_row_is_excluded_before_lookback_tail_like_production() -> None:
    panel = _fixture_panel(12)
    panel["state_available"] = True
    # For day 11 and lookback=3, production first inner-joins state then takes
    # the last three state-available rows: 7, 9, 10.  Tailing returns first
    # would wrongly retain 8, 9, 10 and leave only two valid neighbours.
    panel.loc[8, "state_available"] = False
    panel.loc[8, list(FEATURES)] = np.nan
    replay = run_basegap_variant(
        panel,
        feature_cols=FEATURES,
        lookback_days=3,
        nearest_k=3,
        min_periods=3,
        metric="mean",
        variant="fixture_missing_state",
    )
    row = replay.iloc[11]
    assert row["selection_active"]
    assert row["neighbor_count"] == 3
    assert row["neighbor_min_day"] == panel.loc[7, "score_day"]


def test_unready_rows_are_excluded_from_active_evaluation() -> None:
    replay = _replay(_fixture_panel())
    assert (~replay["selection_active"]).any()
    full = _scope_frame(replay, start=None, end=None, active_only=False)
    active = _scope_frame(replay, start=None, end=None, active_only=True)
    assert len(full) == len(replay)
    assert len(active) == int(replay["selection_active"].sum())
    assert active["selection_active"].all()


def test_parity_audit_treats_two_missing_history_end_values_as_equal() -> None:
    rows = pd.DataFrame(
        {
            "score_day": [date(2024, 1, 2)],
            "basegap_selected_name": ["Regsim"],
            "basegap_reason": ["insufficient_history"],
            "basegap_history_days": [0],
            "basegap_history_end": [None],
            "basegap_score_gap": [np.nan],
            "basegap_score_Regsim": [np.nan],
            "basegap_score_Ensemble": [np.nan],
            "basegap_score_HL20": [np.nan],
        }
    )
    audit = parity_audit(rows, rows.copy(), label="fixture")
    assert (audit["mismatch_rows"] == 0).all()
