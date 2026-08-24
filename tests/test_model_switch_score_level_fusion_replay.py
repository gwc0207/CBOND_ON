from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from harness.tools import model_switch_score_level_fusion_replay as replay


def _synthetic_main(days: int = 24) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        signal = math.sin(index / 4.0)
        returns = np.asarray(
            [
                0.0008 + 0.0011 * signal,
                0.0008 - 0.0007 * signal,
                0.0008 + 0.0002 * math.cos(index / 3.0),
            ]
        )
        ready = index >= 4
        utility = np.asarray([0.0010 * signal, -0.0006 * signal, -0.0004 * signal])
        rows.append(
            {
                "score_day": score_day,
                "prediction_ready": ready,
                "training_end": dates[index - 1] if ready else None,
                **{replay.MODEL_RETURN_COLUMNS[model]: returns[position] for position, model in enumerate(replay.MODELS)},
                **{
                    replay.MODEL_UTILITY_COLUMNS[model]: utility[position] if ready else np.nan
                    for position, model in enumerate(replay.MODELS)
                },
            }
        )
    return pd.DataFrame(rows)


def test_future_and_same_day_realized_returns_cannot_change_existing_weights() -> None:
    source = _synthetic_main()
    cutoff = source.loc[14, "score_day"]
    original = replay.build_daily_weights(source)
    poisoned = source.copy()
    mask = poisoned["score_day"] >= cutoff
    for column in replay.MODEL_RETURN_COLUMNS.values():
        poisoned.loc[mask, column] = 99.0
    rerun = replay.build_daily_weights(poisoned)
    columns = [
        "score_day",
        "history_complete_rows",
        "history_end",
        "causal_temperature",
        "ridge_weight_reason",
        *[f"weight_{variant}_{model}" for variant in (replay.EQUAL_VARIANT, replay.RIDGE_VARIANT) for model in replay.MODELS],
    ]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, columns].reset_index(drop=True),
        rerun.loc[rerun["score_day"] <= cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_warmup_is_equal_and_all_weight_vectors_are_simplexes() -> None:
    weights = replay.build_daily_weights(_synthetic_main())
    first = weights.iloc[0]
    for model in replay.MODELS:
        assert math.isclose(first[f"weight_{replay.RIDGE_VARIANT}_{model}"], 1.0 / 3.0, abs_tol=1e-15)
    assert first["ridge_weight_reason"] == "warmup_or_input_unavailable_equal_weight"
    for variant in (replay.EQUAL_VARIANT, replay.RIDGE_VARIANT):
        matrix = weights[[f"weight_{variant}_{model}" for model in replay.MODELS]].to_numpy(dtype=float)
        assert np.isfinite(matrix).all()
        assert (matrix >= 0.0).all()
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_rank_fusion_is_positive_affine_invariant() -> None:
    codes = [f"C{index:02d}" for index in range(25)]
    scores = {
        "Regsim": pd.Series(np.arange(25, dtype=float), index=codes),
        "Ensemble": pd.Series(np.roll(np.arange(25, dtype=float), 7), index=codes),
        "HL20": pd.Series(np.arange(24, -1, -1, dtype=float), index=codes),
    }
    weights = [0.2, 0.5, 0.3]
    original = replay.rank_blend_scores(scores, weights, score_day="2024-01-02").sort_values("code").reset_index(drop=True)
    transformed = {
        "Regsim": scores["Regsim"] * 17.0 - 4.0,
        "Ensemble": scores["Ensemble"] * 0.25 + 19.0,
        "HL20": scores["HL20"] * 3.0 + 1.0,
    }
    changed = replay.rank_blend_scores(transformed, weights, score_day="2024-01-02").sort_values("code").reset_index(drop=True)
    assert_frame_equal(original, changed, check_exact=True)


def test_rank_fusion_fails_closed_on_score_universe_mismatch() -> None:
    codes = [f"C{index:02d}" for index in range(25)]
    scores = {
        "Regsim": pd.Series(np.arange(25, dtype=float), index=codes),
        "Ensemble": pd.Series(np.arange(25, dtype=float), index=codes),
        "HL20": pd.Series(np.arange(24, dtype=float), index=codes[:-1]),
    }
    with pytest.raises(ValueError, match="identical model score universes"):
        replay.rank_blend_scores(scores, [1.0 / 3.0] * 3, score_day="2024-01-02")


def test_output_root_cannot_escape_dedicated_research_root(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "score_level_v1"
    monkeypatch.setattr(replay, "DEFAULT_OUTPUT_ROOT", allowed)
    assert replay._assert_output_root(allowed / "child") == (allowed / "child").resolve()
    with pytest.raises(ValueError, match="must stay under"):
        replay._assert_output_root(tmp_path / "elsewhere")
