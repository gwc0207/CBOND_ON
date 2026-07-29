from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.infra.risk.portfolio import attribute_active_return, calculate_portfolio_risk


def _inputs():
    exposures = pd.DataFrame(
        {
            "code": ["110001.SH", "110002.SH", "110003.SH"],
            "F1": [-1.0, 0.0, 1.0],
            "F2": [0.5, -0.5, 0.0],
        }
    )
    covariance = pd.DataFrame(
        [[0.0004, 0.0, 0.0], [0.0, 0.0009, 0.0001], [0.0, 0.0001, 0.0004]],
        index=["CB_MKT", "F1", "F2"],
        columns=["CB_MKT", "F1", "F2"],
    )
    specific = pd.DataFrame(
        {"code": exposures["code"], "specific_variance_daily": [0.0001, 0.0002, 0.0003]}
    )
    strategy = pd.DataFrame({"code": ["110001.SH", "110003.SH"], "weight": [0.4, 0.6]})
    benchmark = pd.DataFrame({"code": exposures["code"], "weight": [1 / 3, 1 / 3, 1 / 3]})
    return exposures, covariance, specific, strategy, benchmark


def test_cb_risk_portfolio_variance_contributions_reconcile():
    exposures, covariance, specific, strategy, benchmark = _inputs()
    result = calculate_portfolio_risk(
        exposures,
        covariance,
        specific,
        strategy,
        benchmark,
        factor_columns=["CB_MKT", "F1", "F2"],
    )
    assert result.summary["total_variance_daily"] > 0
    assert abs(result.summary["reconciliation_error"]) < 1e-15
    assert {"strategy_exposure", "benchmark_exposure", "active_exposure"} <= set(result.factor_exposure.columns)


def test_cb_risk_attribution_reconciles_factor_and_specific_returns():
    exposures, _, _, strategy, benchmark = _inputs()
    factor_returns = pd.Series({"CB_MKT": 0.002, "F1": 0.01, "F2": -0.004})
    residuals = pd.DataFrame(
        {"code": ["110001.SH", "110002.SH", "110003.SH"], "specific_return": [0.001, -0.002, 0.0005]}
    )
    result = attribute_active_return(
        exposures,
        strategy,
        benchmark,
        factor_returns,
        residuals,
        factor_columns=["CB_MKT", "F1", "F2"],
        cost_active_return=0.0001,
    )
    assert abs(result.summary["gross_reconciliation_error"]) < 1e-15
    assert result.summary["attribution_kind"] == "active"
    assert result.summary["net_active_return"] == result.summary["gross_active_return"] - 0.0001
