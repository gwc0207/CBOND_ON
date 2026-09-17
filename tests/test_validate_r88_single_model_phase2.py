from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.tools.validate_r88_single_model_phase2 import (
    MCS_ALPHA,
    PBO_BLOCK_COUNT,
    PBO_HALF_BLOCKS,
    TRIAL_COUNT,
    _circular_block_index_matrix,
    _dsr,
    _family_relative_tests,
    _mcs_range,
    _pbo_cscv,
    _psr,
)


def _fixture(periods: int = 243, candidates: int = TRIAL_COUNT) -> tuple[pd.DataFrame, pd.Series]:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    t = np.arange(periods, dtype=float)
    benchmark = 0.0003 * np.sin(t / 7.0)
    columns: dict[str, np.ndarray] = {}
    for index in range(candidates):
        columns[f"candidate_{index:02d}"] = (
            0.0005
            + (1.0 + index / 100.0) * benchmark
            + 0.0012 * np.cos(t / (4.0 + index / 10.0) + index)
        )
    return pd.DataFrame(columns, index=dates), pd.Series(benchmark, index=dates)


def test_circular_indices_are_bounded_and_deterministic() -> None:
    first = _circular_block_index_matrix(23, block_length=5, rows=7, rng=np.random.default_rng(11))
    second = _circular_block_index_matrix(23, block_length=5, rows=7, rng=np.random.default_rng(11))
    assert np.array_equal(first, second)
    assert first.shape == (7, 23)
    assert int(first.min()) >= 0 and int(first.max()) < 23


def test_pbo_uses_all_directed_combinations_and_frozen_scales() -> None:
    returns, benchmark = _fixture()
    family, splits, candidates, ranks = _pbo_cscv(returns, benchmark)
    expected_splits = int(__import__("math").comb(PBO_BLOCK_COUNT, PBO_HALF_BLOCKS))
    assert family["directed_split_count"] == expected_splits
    assert len(splits) == expected_splits
    assert len(candidates) == TRIAL_COUNT
    assert len(ranks) == expected_splits * TRIAL_COUNT
    assert {"in_sharpe_scale_low", "in_alpha_t_scale_high"}.issubset(splits.columns)
    assert 0.0 <= family["pbo"] <= 1.0


def test_psr_and_dsr_are_finite_and_dsr_uses_trial_dispersion() -> None:
    returns, _ = _fixture(periods=100)
    values = returns.iloc[:, 0].to_numpy(dtype=float)
    psr = _psr(values)
    trial_sharpes = np.asarray([float(_psr(returns.iloc[:, i])['psr_sharpe_annualized']) / np.sqrt(252.0) for i in range(TRIAL_COUNT)])
    dsr = _dsr(values, trial_daily_sharpes=trial_sharpes)
    assert np.isfinite(psr["psr_probability"])
    assert np.isfinite(dsr["dsr_probability"])
    assert 0.0 <= psr["psr_probability"] <= 1.0
    assert 0.0 <= dsr["dsr_probability"] <= 1.0
    assert dsr["dsr_trial_count"] == TRIAL_COUNT


def test_family_relative_null_is_not_significant() -> None:
    returns, _ = _fixture(periods=80)
    deltas = returns - returns.iloc[:, [0]].to_numpy()
    deltas.iloc[:, :] = 0.0
    family, rows = _family_relative_tests(deltas, scope="development_2025", reps=31, block_length=5, seed=7)
    assert len(rows) == TRIAL_COUNT
    assert family["white_reality_check"]["p_value"] > 0.5
    assert family["hansen_spa"]["p_value_consistent"] > 0.5


def test_mcs_identical_models_are_retained() -> None:
    returns, _ = _fixture(periods=80)
    identical = pd.DataFrame(
        np.repeat(returns.iloc[:, [0]].to_numpy(), 4, axis=1),
        index=returns.index,
        columns=["a", "b", "c", "d"],
    )
    summary, membership, trace = _mcs_range(
        identical,
        scope="overall",
        reps=31,
        block_length=5,
        alpha=MCS_ALPHA,
        seed=8,
    )
    assert summary["included_count"] == 4
    assert membership["mcs_included"].all()
    assert set(membership["mcs_inclusion_p_value"]) == {1.0}
    assert (membership["mcs_final_step_p_value"] >= MCS_ALPHA).all()
    assert len(trace) == 1
    assert trace.iloc[0]["bootstrap_p_value"] >= MCS_ALPHA
