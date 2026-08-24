from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from harness.tools import model_switch_downside_risk_fusion_replay as replay


def _synthetic_main(days: int = 48) -> pd.DataFrame:
    """Minimal causal input: no Ridge utility or current/future fields."""

    dates = pd.bdate_range("2024-01-02", periods=days).date
    rows: list[dict[str, object]] = []
    for index, score_day in enumerate(dates):
        returns = np.asarray(
            [
                -0.0060 if index % 3 == 0 else 0.0020,
                -0.0030 if index % 4 == 0 else 0.0010,
                -0.0015 if index % 5 == 0 else 0.0005,
            ]
        )
        rows.append(
            {
                "score_day": score_day,
                **{
                    replay.MODEL_RETURN_COLUMNS[model]: returns[position]
                    for position, model in enumerate(replay.MODELS)
                },
            }
        )
    return pd.DataFrame(rows)


def _weight_columns() -> list[str]:
    return [
        f"weight_{variant}_{model}"
        for variant in replay.SCORE_VARIANTS
        for model in replay.MODELS
    ]


def test_future_and_same_day_returns_cannot_change_existing_risk_weights() -> None:
    source = _synthetic_main()
    cutoff = source.loc[31, "score_day"]
    original = replay.build_daily_weights(source)
    poisoned = source.copy()
    mask = poisoned["score_day"] >= cutoff
    for column in replay.MODEL_RETURN_COLUMNS.values():
        poisoned.loc[mask, column] = 99.0
    rerun = replay.build_daily_weights(poisoned)
    columns = [
        "score_day",
        "history_return_rows",
        "history_start",
        "history_end",
        "downside_risk_reason",
        *[f"downside_deviation_{model}" for model in replay.MODELS],
        *_weight_columns(),
    ]
    assert_frame_equal(
        original.loc[original["score_day"] <= cutoff, columns].reset_index(drop=True),
        rerun.loc[rerun["score_day"] <= cutoff, columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_warmup_is_exactly_equal_and_all_weight_vectors_are_simplexes() -> None:
    weights = replay.build_daily_weights(_synthetic_main())
    warmup = weights.iloc[: replay.MIN_HISTORY_DAYS]
    for model in replay.MODELS:
        assert np.allclose(
            warmup[f"weight_{replay.DOWNSIDE_RISK_PARITY_VARIANT}_{model}"].to_numpy(dtype=float),
            1.0 / len(replay.MODELS),
            atol=1e-15,
            rtol=0.0,
        )
    assert warmup["downside_risk_reason"].eq("warmup_equal_weight").all()
    assert weights.iloc[replay.MIN_HISTORY_DAYS]["downside_risk_reason"] == "causal_ewma_inverse_downside_deviation"
    for variant in replay.SCORE_VARIANTS:
        matrix = weights[[f"weight_{variant}_{model}" for model in replay.MODELS]].to_numpy(dtype=float)
        assert np.isfinite(matrix).all()
        assert (matrix >= 0.0).all()
        assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_inverse_downside_risk_parity_matches_the_declared_formula() -> None:
    deviation = np.asarray([0.020, 0.010, 0.005])
    weights = replay.inverse_downside_risk_parity_weights(deviation)
    assert np.allclose(weights, np.asarray([1.0, 2.0, 4.0]) / 7.0, atol=1e-15, rtol=0.0)
    assert weights[2] > weights[1] > weights[0]


def test_downside_deviation_ignores_positive_return_magnitude() -> None:
    history = np.asarray(
        [
            [-0.010, 0.004, -0.001],
            [0.001, -0.003, 0.007],
            [-0.004, 0.002, -0.006],
            [0.003, -0.002, 0.001],
        ]
    )
    changed_positive = history.copy()
    changed_positive[changed_positive > 0.0] *= 10_000.0
    original = replay.causal_ewma_downside_deviation(history)
    changed = replay.causal_ewma_downside_deviation(changed_positive)
    assert np.allclose(original, changed, atol=1e-15, rtol=0.0)


def test_weight_builder_requires_only_standalone_returns_not_ridge_utility() -> None:
    frame = _synthetic_main()
    assert not any("utility" in column.lower() for column in frame.columns)
    result = replay.build_daily_weights(frame)
    assert len(result) == len(frame)
    assert math.isclose(
        float(result.iloc[-1][f"weight_{replay.DOWNSIDE_RISK_PARITY_VARIANT}_Regsim"]
        + result.iloc[-1][f"weight_{replay.DOWNSIDE_RISK_PARITY_VARIANT}_Ensemble"]
        + result.iloc[-1][f"weight_{replay.DOWNSIDE_RISK_PARITY_VARIANT}_HL20"]),
        1.0,
        abs_tol=1e-12,
    )


def test_relative_path_diagnostics_report_compounded_relative_loss_and_mdd() -> None:
    days = pd.bdate_range("2024-01-02", periods=5).date
    baseline = pd.DataFrame({"score_day": days, "day_return": [0.0] * 5})
    candidate = pd.DataFrame({"score_day": days, "day_return": [0.10, -0.20, 0.10, 0.0, 0.0]})
    daily, summary = replay._relative_path_diagnostics(baseline, candidate, strategy="candidate")
    assert math.isclose(float(summary["relative_mdd_candidate_over_regsim"]), -0.20, abs_tol=1e-15)
    assert summary["relative_mdd_peak_score_day"] == str(days[0])
    assert summary["relative_mdd_trough_score_day"] == str(days[1])
    assert math.isclose(float(daily.loc[4, "relative_5d_return"]), -0.032, abs_tol=1e-15)
    assert summary["worst_relative_5d_end_score_day"] == str(days[-1])


def test_output_root_cannot_escape_dedicated_research_root(tmp_path, monkeypatch) -> None:
    allowed = tmp_path / "model_switch_downside_fusion_20260811"
    monkeypatch.setattr(replay, "DEFAULT_OUTPUT_ROOT", allowed)
    assert replay._assert_output_root(allowed / "child") == (allowed / "child").resolve()
    with pytest.raises(ValueError, match="must stay under"):
        replay._assert_output_root(tmp_path / "elsewhere")
