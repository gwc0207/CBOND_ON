from __future__ import annotations

import numpy as np
import pandas as pd

from cbond_on.domain.risk import factor_definitions_from_config
from cbond_on.infra.risk.exposure import build_risk_exposures, normalize_cbond_codes


def _raw_panel(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    duration = rng.uniform(0.3, 5.0, n)
    return pd.DataFrame(
        {
            "instrument_code": [110000 + i for i in range(n)],
            "exchange_code": ["SH"] * n,
            "remain_size": rng.lognormal(3.0, 0.5, n),
            "cb_amount": rng.lognormal(15.0, 0.5, n),
            "bond_prem_ratio": rng.normal(25.0, 12.0, n),
            "modify_duration": duration,
            "convexity": np.exp(1.2 * duration + rng.normal(0.0, 0.1, n)),
            "rating": np.where(np.arange(n) % 3 == 0, "AAA", "AA+"),
            "stock_volatility": rng.uniform(10.0, 90.0, n),
        }
    )


def test_cb_risk_exposure_standardizes_and_orthogonalizes():
    raw = _raw_panel()
    definitions = factor_definitions_from_config(
        [
            {"name": "SIZE", "source_columns": ["remain_size"], "transform": "log"},
            {"name": "DURATION", "source_columns": ["modify_duration"], "transform": "positive"},
            {
                "name": "CONVEXITY_ORTHO",
                "source_columns": ["convexity"],
                "transform": "log1p",
                "orthogonalize_against": ["DURATION"],
                "orthogonalize_square": True,
            },
            {"name": "CREDIT", "source_columns": ["rating"], "transform": "rating_ordinal"},
        ]
    )
    result = build_risk_exposures(raw, definitions)

    assert result.factor_columns == ("SIZE", "DURATION", "CONVEXITY_ORTHO", "CREDIT")
    assert result.exposures["code"].iloc[0] == "110000.SH"
    weights = result.exposures["regression_weight"].to_numpy()
    for column in result.factor_columns:
        assert abs(np.average(result.exposures[column], weights=weights)) < 1e-10
    assert abs(
        np.average(
            result.exposures["CONVEXITY_ORTHO"] * result.exposures["DURATION"],
            weights=weights,
        )
    ) < 1e-10
    assert (result.diagnostics["status"] == "enabled").all()


def test_cb_risk_exposure_disables_missing_factor_instead_of_zero_filling():
    raw = _raw_panel()
    definitions = factor_definitions_from_config(
        [
            {"name": "SIZE", "source_columns": ["remain_size"], "transform": "log"},
            {"name": "NOT_PRESENT", "source_columns": ["unknown"], "transform": "identity"},
        ]
    )
    result = build_risk_exposures(raw, definitions)
    assert result.factor_columns == ("SIZE",)
    row = result.diagnostics.set_index("factor").loc["NOT_PRESENT"]
    assert row["status"] == "disabled_missing_source"
    assert "NOT_PRESENT" not in result.exposures.columns


def test_normalize_cbond_codes_keeps_project_suffixes():
    frame = pd.DataFrame({"instrument_code": [110001.0, "123456"], "exchange_code": ["XSHG", "XSHE"]})
    assert normalize_cbond_codes(frame).tolist() == ["110001.SH", "123456.SZ"]
