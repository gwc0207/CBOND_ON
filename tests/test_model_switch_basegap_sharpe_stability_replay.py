from __future__ import annotations

import numpy as np
import pandas as pd

from harness.tools.model_switch_basegap_sharpe_stability_replay import (
    BLOCK_COUNT,
    common_active_design_days,
    build_sharpe_stability_scorecard,
    relative_sharpe_bootstrap,
)


def _rows(variant: str, *, active_start: int, selected: list[float], regsim: list[float]) -> list[dict[str, object]]:
    days = pd.bdate_range("2024-05-08", periods=len(selected))
    return [
        {
            "variant": variant,
            "score_day": day,
            "selected_return": selected[index],
            "realized_return_Regsim": regsim[index],
            "selection_active": index >= active_start,
            "execution_metadata_complete": True,
            "lookback_days": 60 if variant == "short" else 240,
            "nearest_k": 20,
            "min_periods": 20,
            "metric": "trim20_lcb10",
        }
        for index, day in enumerate(days)
    ]


def _fixture() -> pd.DataFrame:
    days = 320
    regsim = [0.001 + 0.00025 * np.sin(index / 7.0) for index in range(days)]
    stable = [0.0015 + 0.00025 * np.sin(index / 6.0) for index in range(days)]
    # Higher simple mean than stable, but concentrated in one block and weak in the others.
    unstable = [
        (0.006 if index < 44 else -0.0002) + 0.00025 * np.cos(index / 5.0)
        for index in range(days)
    ]
    rows = []
    rows += _rows("stable", active_start=40, selected=stable, regsim=regsim)
    rows += _rows("unstable", active_start=60, selected=unstable, regsim=regsim)
    # Fill the fixed-family contract with distinct variants; stable copies are sufficient for
    # common-date and deterministic ranking tests.
    for index in range(49):
        rows += _rows(f"copy_{index:02d}", active_start=60, selected=stable, regsim=regsim)
    return pd.DataFrame(rows)


def test_common_active_design_days_removes_warmup_advantage() -> None:
    frame = _fixture()
    days = common_active_design_days(frame)
    assert len(days) == 239
    assert days.min() == pd.Timestamp("2024-07-31")


def test_stability_scorecard_prefers_stable_return_path_over_concentrated_path() -> None:
    frame = _fixture()
    scorecard, blocks, common_days = build_sharpe_stability_scorecard(frame, bootstrap_reps=101)
    assert len(common_days) >= BLOCK_COUNT * 20
    assert len(blocks.loc[blocks["variant"] == "stable"]) == BLOCK_COUNT
    stable = scorecard.loc[scorecard["variant"] == "stable"].iloc[0]
    unstable = scorecard.loc[scorecard["variant"] == "unstable"].iloc[0]
    assert stable["selected_sharpe_stability_q"] > unstable["selected_sharpe_stability_q"]


def test_scorecard_requires_fixed_family_shape() -> None:
    frame = _fixture().loc[lambda x: x["variant"] != "copy_48"].copy()
    with np.testing.assert_raises_regex(ValueError, "fixed grid"):
        common_active_design_days(frame)


def test_relative_sharpe_bootstrap_is_bounded_on_the_fixed_common_cohort() -> None:
    frame = _fixture()
    days = common_active_design_days(frame)
    output = relative_sharpe_bootstrap(frame, days, reps=101, seed=7)
    assert len(output) == 51
    assert set(output["relative_sharpe_bootstrap_n"]) == {101}
    assert output["relative_sharpe_bootstrap_probability_positive"].between(0.0, 1.0).all()
