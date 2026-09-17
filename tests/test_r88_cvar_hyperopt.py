from __future__ import annotations

import numpy as np
import pandas as pd

from harness.tools import r88_cvar_hyperopt as study


def _base_returns(days: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    benchmark = np.sin(np.arange(days) / 9.0) * 0.004
    return pd.DataFrame(
        {
            "trade_date": dates,
            "benchmark_return": benchmark,
            "P6": 0.0008 + 1.55 * benchmark + np.cos(np.arange(days) / 7.0) * 0.001,
            "P3": 0.0006 + 1.35 * benchmark + np.sin(np.arange(days) / 6.0) * 0.0012,
            "Regsim": 0.0005 + 1.20 * benchmark + np.cos(np.arange(days) / 5.0) * 0.0009,
        }
    )


def test_pre_registered_grid_counts_are_locked() -> None:
    assert len(study.raw_specs()) == 54
    assert len(study.residual_specs()) == 16
    assert len(study.all_specs()) == 70
    assert len({spec.method_id for spec in study.all_specs()}) == 70


def test_tail_mean_uses_ceiling_and_lower_tail() -> None:
    value, count = study._tail_mean(np.array([0.02, -0.01, -0.03, 0.01, -0.02]), 0.20)
    assert count == 1
    assert value == -0.03
    value, count = study._tail_mean(np.array([0.02, -0.01, -0.03, 0.01, -0.02]), 0.40)
    assert count == 2
    assert value == -0.025


def test_lambda_zero_raw_variants_are_weight_identical_across_tail_fraction() -> None:
    base = _base_returns()
    left = study.CvarSpec("raw", 120, 0.05, 0.0, 0.50)
    right = study.CvarSpec("raw", 120, 0.10, 0.0, 0.50)
    left_weights, _ = study.variant_weights(base, left)
    right_weights, _ = study.variant_weights(base, right)
    assert study._weight_digest(left_weights) == study._weight_digest(right_weights)


def test_pit_beta_path_is_prior_only_and_preserves_alpha_proxy(tmp_path) -> None:
    base = _base_returns()
    path = study.build_pit_beta_adjusted_path(tmp_path, base)
    assert path.loc[: study.BETA_LOOKBACK_DAYS - 1, "fit_status"].eq("insufficient_prior_cycles").all()
    first = path.iloc[study.BETA_LOOKBACK_DAYS]
    assert first["fit_status"] == "valid"
    assert np.isfinite(first["beta_fit_P6"])
    assert np.isfinite(first["alpha_proxy_P6"])
    assert np.isfinite(first["innovation_P6"])


def test_residual_weights_do_not_use_target_or_future_return(tmp_path) -> None:
    base = _base_returns()
    pit = study.build_pit_beta_adjusted_path(tmp_path / "left", base)
    spec = study.CvarSpec("residual", 120, 0.05, 0.75, 0.50)
    weights, _ = study.variant_weights(base, spec, pit_beta_path=pit)
    altered = base.copy()
    altered.loc[220:, ["P6", "P3", "Regsim", "benchmark_return"]] += 0.50
    pit_altered = study.build_pit_beta_adjusted_path(tmp_path / "right", altered)
    altered_weights, _ = study.variant_weights(altered, spec, pit_beta_path=pit_altered)
    columns = ["weight_P6", "weight_P3", "weight_Regsim"]
    assert np.allclose(weights.loc[:219, columns], altered_weights.loc[:219, columns], rtol=0.0, atol=1e-12)


def test_all_variant_weights_are_simplex(tmp_path) -> None:
    base = _base_returns()
    pit = study.build_pit_beta_adjusted_path(tmp_path, base)
    spec = study.CvarSpec("raw", 60, 0.10, 1.50, 0.25)
    weights, _ = study.variant_weights(base, spec, pit_beta_path=pit)
    columns = ["weight_P6", "weight_P3", "weight_Regsim"]
    values = weights[columns].to_numpy(dtype=float)
    assert np.all(values >= 0.0)
    assert np.allclose(values.sum(axis=1), 1.0)
