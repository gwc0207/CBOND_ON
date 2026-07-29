from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.infra.risk.estimation import (
    estimate_factor_covariance,
    estimate_factor_returns,
    estimate_specific_risk,
)


def test_cb_risk_recovers_known_cross_sectional_factor_returns():
    rng = np.random.default_rng(11)
    n = 400
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    returns = 0.0015 + 0.012 * f1 - 0.007 * f2 + rng.normal(0.0, 1e-7, n)
    panel = pd.DataFrame(
        {
            "code": [f"{110000 + i}.SH" for i in range(n)],
            "gross_return": returns,
            "regression_weight": rng.uniform(0.5, 1.5, n),
            "F1": f1,
            "F2": f2,
        }
    )
    result = estimate_factor_returns(panel, ["F1", "F2"], min_samples=100)

    assert result.diagnostics["sample_count"] == n
    assert result.factor_returns["CB_MKT"] == pytest_approx(0.0015, 1e-5)
    assert result.factor_returns["F1"] == pytest_approx(0.012, 1e-5)
    assert result.factor_returns["F2"] == pytest_approx(-0.007, 1e-5)
    assert result.residuals.abs().max() < 1e-5


def pytest_approx(value: float, tolerance: float):
    # Kept local to avoid a dependency in the numerical hot path.
    import pytest

    return pytest.approx(value, abs=tolerance)


def test_cb_risk_covariance_is_psd_and_specific_risk_shrinks_new_names():
    rng = np.random.default_rng(12)
    history = pd.DataFrame(
        {
            "CB_MKT": rng.normal(0.0, 0.01, 140),
            "F1": rng.normal(0.0, 0.02, 140),
            "F2": rng.normal(0.0, 0.015, 140),
        }
    )
    covariance = estimate_factor_covariance(history, min_observations=120, shrinkage=0.2)
    assert np.linalg.eigvalsh(covariance.covariance_daily.to_numpy()).min() >= -1e-14
    assert covariance.covariance_daily.shape == (3, 3)

    residuals = pd.DataFrame(
        {
            "code": ["110001.SH"] * 30 + ["110002.SH"] * 2,
            "specific_return": np.r_[rng.normal(0, 0.01, 30), rng.normal(0, 0.02, 2)],
        }
    )
    specific = estimate_specific_risk(residuals, min_observations=20)
    row = specific.specific_risk.set_index("code").loc["110002.SH"]
    assert row["risk_source"] == "insufficient_history_shrunk"
    assert row["specific_variance_daily"] > 0
