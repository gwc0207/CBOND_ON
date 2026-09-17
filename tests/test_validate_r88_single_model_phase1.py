from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.tools.validate_r88_single_model_phase1 import (
    RESULTS_ENCODING,
    ValidationError,
    _normalise_daily,
    _paired_delta_bootstrap,
    _raw_delta_hac,
    _render_results,
    _rolling_metrics,
    calculate_metrics,
    moving_block_bootstrap,
)


def _daily(periods: int = 80, *, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    benchmark = np.asarray([0.0004 * np.sin(index / 5.0) for index in range(periods)])
    returns = 0.0008 + 1.3 * benchmark + offset
    return pd.DataFrame(
        {
            "trade_date": dates,
            "day_return": returns,
            "benchmark_return": benchmark,
            "benchmark_method": "strict_official_prev_close_split",
        }
    )


def test_calculate_metrics_reports_return_risk_and_hac_alpha() -> None:
    metrics = calculate_metrics(_daily())
    assert metrics["n_obs"] == 80.0
    assert metrics["sharpe"] > 0.0
    assert metrics["annualized_return_simple"] > 0.0
    assert metrics["beta"] == pytest.approx(1.3, abs=1e-8)
    assert np.isfinite(metrics["alpha_hac_t"])
    assert metrics["daily_cvar_95"] <= metrics["daily_var_95"]


def test_moving_block_bootstrap_is_deterministic_and_bounded() -> None:
    daily = _daily()
    first = moving_block_bootstrap(daily, reps=101, block_length=5, seed=7)
    second = moving_block_bootstrap(daily, reps=101, block_length=5, seed=7)
    assert first == second
    assert first["bootstrap_reps"] == 101.0
    assert first["bootstrap_sharpe_q05"] <= first["bootstrap_sharpe_q50"] <= first["bootstrap_sharpe_q95"]
    assert 0.0 <= first["bootstrap_sharpe_probability_positive"] <= 1.0


def test_raw_delta_hac_is_zero_for_identical_return_paths() -> None:
    daily = _daily()
    result = _raw_delta_hac(daily, daily)
    assert result["paired_n_obs"] == float(len(daily))
    assert result["paired_delta_mean_bp"] == pytest.approx(0.0)
    assert np.isnan(result["paired_delta_hac_t"])


def test_paired_delta_bootstrap_is_deterministic_and_preserves_zero_delta() -> None:
    daily = _daily()
    first = _paired_delta_bootstrap(daily, daily, reps=101, block_length=5, seed=7)
    second = _paired_delta_bootstrap(daily, daily, reps=101, block_length=5, seed=7)
    assert first == second
    assert first["paired_delta_bootstrap_q05_bp"] == pytest.approx(0.0)
    assert first["paired_delta_bootstrap_q95_bp"] == pytest.approx(0.0)
    assert first["paired_sharpe_difference_bootstrap_q50"] == pytest.approx(0.0)


def test_rolling_metrics_has_only_complete_windows() -> None:
    daily = _daily(periods=80)
    rows = _rolling_metrics("candidate", daily, 20)
    assert len(rows) == 61
    assert rows[0]["start_trade_date"] == daily["trade_date"].iloc[0]
    assert rows[-1]["end_trade_date"] == daily["trade_date"].iloc[-1]
    assert {row["scope"] for row in rows} == {"development_2025"}


def test_normalise_daily_rejects_duplicate_dates() -> None:
    duplicated = pd.concat([_daily(periods=5), _daily(periods=1)], ignore_index=True)
    with pytest.raises(ValidationError, match="duplicate"):
        _normalise_daily(duplicated, label="fixture")


def test_results_markdown_is_chinese_and_windows_utf8_detectable(tmp_path) -> None:
    scorecard = pd.DataFrame(
        [{
            "candidate": "candidate",
            "scope": "reporting_2026",
            "sharpe": 1.234,
            "alpha_hac_t": 2.345,
            "max_drawdown": -0.012,
            "annualized_return_simple": 0.234,
            "rolling60_sharpe_median": 1.111,
        }]
    )
    text = _render_results(
        scorecard,
        pd.DataFrame(),
        evidence={
            "study_root": "D:/study",
            "candidate_count": 30,
            "regsim_common_days": 399,
            "factor_generation": {"path": "D:/factors"},
        },
    )
    assert "单模型验证" in text
    assert "解释边界" in text
    path = tmp_path / "RESULTS.md"
    path.write_text(text, encoding=RESULTS_ENCODING)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert path.read_text(encoding="utf-8-sig") == text
