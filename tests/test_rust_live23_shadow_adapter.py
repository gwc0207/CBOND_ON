from __future__ import annotations

import pandas as pd

from cbond_on.domain.factors.spec import FactorSpec
from cbond_on.infra.factors.rust_legacy_reference import (
    build_factor_frame_rust_legacy_reference,
)


class _FakeShadowRust:
    def __init__(self) -> None:
        self.daily_data: dict[str, pd.DataFrame] | None = None

    def compute_typed_factor_frame(
        self,
        panel_df: pd.DataFrame,
        specs_payload: list[dict],
        stock_df: pd.DataFrame | None,
        map_df: pd.DataFrame | None,
        daily_data: dict[str, pd.DataFrame] | None,
        compute_backend_params: dict,
    ) -> pd.DataFrame:
        del stock_df, map_df, compute_backend_params
        self.daily_data = daily_data
        output = panel_df.loc[:, ["dt", "code"]].drop_duplicates().copy()
        for spec in specs_payload:
            output[str(spec["output_col"])] = 1.0
        return output


def _panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-08-05"), "110001.SH", 0),
            (pd.Timestamp("2026-08-05"), "110001.SH", 1),
        ],
        names=["dt", "code", "seq"],
    )
    return pd.DataFrame(
        {
            "trade_time": [
                pd.Timestamp("2026-08-05 14:28:00"),
                pd.Timestamp("2026-08-05 14:29:00"),
            ],
            "last": [101.0, 101.1],
        },
        index=index,
    )


def test_shadow_adapter_preserves_daily_categorical_contract() -> None:
    spec = FactorSpec(
        name="base_debt_premium_floor_gap",
        factor="factor_mining_daily_catalog_v1",
        params={"signal": "base_debt_premium_floor_gap"},
    )
    module = _FakeShadowRust()
    daily = {
        "market_cbond.daily_base": pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2026-08-04")],
                "code": ["110001"],
                "exchange_code": ["SH"],
                "debt_puredebt_ratio": [1.02],
            }
        )
    }

    actual = build_factor_frame_rust_legacy_reference(
        _panel(),
        [spec],
        rust_module=module,
        daily_data=daily,
    )

    assert list(actual.columns) == ["base_debt_premium_floor_gap"]
    assert actual.index.names == ["dt", "code"]
    assert module.daily_data is not None
    observed = module.daily_data["market_cbond.daily_base"]
    assert observed.loc[0, "exchange_code"] == "SH"
    assert observed.loc[0, "debt_puredebt_ratio"] == 1.02


def test_shadow_adapter_preserves_missing_daily_source_code() -> None:
    spec = FactorSpec(
        name="base_debt_premium_floor_gap",
        factor="factor_mining_daily_catalog_v1",
        params={"signal": "base_debt_premium_floor_gap"},
    )
    module = _FakeShadowRust()
    daily = {
        "market_cbond.daily_base": pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2026-08-04")],
                "code": [pd.NA],
                "exchange_code": ["SH"],
                "debt_puredebt_ratio": [1.02],
            }
        )
    }

    build_factor_frame_rust_legacy_reference(
        _panel(),
        [spec],
        rust_module=module,
        daily_data=daily,
    )

    assert module.daily_data is not None
    observed = module.daily_data["market_cbond.daily_base"]
    assert pd.isna(observed.loc[0, "code"])
