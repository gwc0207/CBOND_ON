from __future__ import annotations

import numpy as np
import pandas as pd

from harness.tools.validate_r88_strategy_health import (
    BASELINE_WINDOW,
    _add_degradation_diagnostics,
    _add_rolling_metrics,
    _calibrate_cusum_limit,
    _daily_score_quality,
    _pettitt,
)


def _daily_fixture(periods: int = 140) -> pd.DataFrame:
    dates = pd.bdate_range("2025-06-02", periods=periods)
    t = np.arange(periods, dtype=float)
    rank_ic = np.where(t < 90, 0.03 + 0.01 * np.sin(t), -0.02 + 0.01 * np.sin(t))
    return pd.DataFrame(
        {
            "candidate": "fixed",
            "trade_date": dates,
            "rank_ic": rank_ic,
            "ic": rank_ic / 2.0,
            "score_std": 0.002 + 0.0001 * np.sin(t),
            "score_range": 0.02 + 0.001 * np.cos(t),
            "raw_score_top20_bottom20_label_gap": 0.001 + 0.0001 * np.cos(t),
            "partial_label_coverage": False,
            "strategy_universe_ic_count": 300,
            "strategy_return": 0.001 + 0.0002 * np.sin(t),
            "strategy_benchmark_return": 0.0005 + 0.0001 * np.cos(t),
        }
    )


def test_partial_label_quality_is_audit_only() -> None:
    scores = pd.DataFrame({"trade_date": ["2026-08-12"] * 3, "code": ["a", "b", "c"], "score": [3.0, 2.0, 1.0]})
    labels = pd.DataFrame({"code": ["a", "b"], "y": [0.01, -0.01]})
    row = _daily_score_quality(scores, labels, day=pd.Timestamp("2026-08-12"), previous_scores=None)
    assert row["partial_label_coverage"] is True
    assert row["raw_score_label_join_fraction"] == 2.0 / 3.0
    assert np.isnan(row["raw_score_label_rank_ic"])
    assert np.isfinite(row["score_std"])


def test_rolling_metrics_keep_primary_ic_but_mark_small_cross_section() -> None:
    frame = _daily_fixture()
    frame.loc[100, "strategy_universe_ic_count"] = 80
    result = _add_rolling_metrics(
        frame,
        strategy_returns=frame["strategy_return"].to_numpy(dtype=float),
        benchmark_returns=frame["strategy_benchmark_return"].to_numpy(dtype=float),
    )
    assert result.loc[100, "strategy_ic_coverage_insufficient"]
    assert np.isfinite(result.loc[100, "rank_ic"])
    assert result.loc[119, "rolling60_rank_ic_data_inconclusive"]


def test_pettitt_localizes_a_simple_shift() -> None:
    values = np.r_[np.full(40, 0.03), np.full(40, -0.02)]
    dates = pd.Series(pd.bdate_range("2025-01-02", periods=len(values)))
    result = _pettitt(values, dates)
    assert result["change_index"] is not None
    assert 30 <= result["change_index"] <= 50
    assert result["mean_shift"] < 0.0


def test_cusum_limit_is_deterministic_and_positive() -> None:
    values = 0.02 + 0.01 * np.sin(np.arange(BASELINE_WINDOW + 20, dtype=float))
    first = _calibrate_cusum_limit(values, seed=17)
    second = _calibrate_cusum_limit(values, seed=17)
    assert first == second
    assert first["control_limit"] > 0.0


def test_degradation_panel_exposes_prediction_and_alpha_states() -> None:
    frame = _daily_fixture()
    rolled = _add_rolling_metrics(
        frame,
        strategy_returns=frame["strategy_return"].to_numpy(dtype=float),
        benchmark_returns=frame["strategy_benchmark_return"].to_numpy(dtype=float),
    )
    result, changes = _add_degradation_diagnostics(rolled)
    assert {"prediction_health_watch", "alpha_health_watch", "health_degradation_diagnostic"}.issubset(result.columns)
    assert len(changes) == 6
    assert set(changes["metric"]) == {"rank_ic", "rolling60_alpha_hac_t", "strategy_return"}


def test_nonpositive_alpha_is_not_mislabeled_as_negative_evidence() -> None:
    frame = _daily_fixture()
    rolled = _add_rolling_metrics(
        frame,
        strategy_returns=frame["strategy_return"].to_numpy(dtype=float),
        benchmark_returns=frame["strategy_benchmark_return"].to_numpy(dtype=float),
    )
    rolled["rolling60_alpha_hac_t"] = -0.5
    result, _ = _add_degradation_diagnostics(rolled)
    assert not result["alpha_health_degradation"].any()
    assert "NEGATIVE_ALPHA_EVIDENCE" not in set(result["health_state"])


def test_warmup_is_not_labeled_normal() -> None:
    frame = _daily_fixture()
    rolled = _add_rolling_metrics(
        frame,
        strategy_returns=frame["strategy_return"].to_numpy(dtype=float),
        benchmark_returns=frame["strategy_benchmark_return"].to_numpy(dtype=float),
    )
    result, _ = _add_degradation_diagnostics(rolled)
    assert result.iloc[0]["alpha60_status"] == "DATA_INSUFFICIENT"
    assert result.iloc[0]["health_state"] == "WARMUP_DATA_INSUFFICIENT"


def test_alpha_cusum_watch_is_not_negative_alpha_evidence() -> None:
    frame = _daily_fixture()
    rolled = _add_rolling_metrics(
        frame,
        strategy_returns=frame["strategy_return"].to_numpy(dtype=float),
        benchmark_returns=frame["strategy_benchmark_return"].to_numpy(dtype=float),
    )
    rolled["rolling60_alpha_hac_t"] = 0.4
    result, _ = _add_degradation_diagnostics(rolled)
    # A positive-but-weaker alpha is never reclassified as strict negative
    # evidence merely because the panel also has a separate decay-watch path.
    assert result["alpha_health_degradation"].eq(False).all()
    assert "NEGATIVE_ALPHA_EVIDENCE" not in set(result["health_state"])
