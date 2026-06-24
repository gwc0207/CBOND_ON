from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from cbond_on.infra.model.neutralization import build_neutralizer


def test_neutralizer_removes_daily_price_exposure(tmp_path):
    raw_root = tmp_path / "raw"
    table_dir = raw_root / "market_cbond__daily_base" / "2026-01"
    table_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 5)] * 5,
            "instrument_code": [110001.0, 110002.0, 110003.0, 110004.0, 110005.0],
            "exchange_code": ["XSHG", "XSHG", "XSHG", "XSHG", "XSHG"],
            "cb_close_price": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    ).to_parquet(table_dir / "20260105.parquet", index=False)

    noise = np.array([0.10, -0.20, 0.05, 0.30, -0.25])
    price = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    frame = pd.DataFrame(
        {
            "dt": [pd.Timestamp("2026-01-05 14:30")] * 5,
            "code": ["110001.SH", "110002.SH", "110003.SH", "110004.SH", "110005.SH"],
            "factor_x": 2.0 * price + noise,
        }
    )

    neutralizer = build_neutralizer(
        {
            "enabled": True,
            "method": "ols",
            "min_count": 3,
            "exposures": [
                {
                    "name": "price",
                    "source": "market_cbond.daily_base",
                    "column": "cb_close_price",
                }
            ],
        },
        raw_data_root=raw_root,
    )
    out = neutralizer.apply(frame, ["factor_x"])

    assert abs(out["factor_x"].corr(pd.Series(price))) < 1e-10
    assert abs(out["factor_x"].mean()) < 1e-10
    assert {"dt", "code", "factor_x"} <= set(out.columns)


def test_raw_exposure_can_use_previous_trading_day(tmp_path):
    raw_root = tmp_path / "raw"
    cal_dir = raw_root / "metadata__trading_calendar"
    cal_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "calendar_date": [date(2026, 1, 5), date(2026, 1, 6)],
            "is_open": [True, True],
        }
    ).to_parquet(cal_dir / "all.parquet", index=False)

    table_dir = raw_root / "market_cbond__daily_base" / "2026-01"
    table_dir.mkdir(parents=True)
    codes = ["110001.SH", "110002.SH", "110003.SH", "110004.SH", "110005.SH"]
    pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 5)] * 5,
            "instrument_code": [110001.0, 110002.0, 110003.0, 110004.0, 110005.0],
            "exchange_code": ["XSHG"] * 5,
            "cb_close_price": [100.0, 110.0, 120.0, 130.0, 140.0],
        }
    ).to_parquet(table_dir / "20260105.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": [date(2026, 1, 6)] * 5,
            "instrument_code": [110001.0, 110002.0, 110003.0, 110004.0, 110005.0],
            "exchange_code": ["XSHG"] * 5,
            "cb_close_price": [300.0, 280.0, 260.0, 240.0, 220.0],
        }
    ).to_parquet(table_dir / "20260106.parquet", index=False)

    frame = pd.DataFrame(
        {
            "dt": [pd.Timestamp("2026-01-06 14:30")] * 5,
            "code": codes,
            "factor_x": [200.1, 219.8, 240.05, 260.3, 279.75],
        }
    )
    neutralizer = build_neutralizer(
        {
            "enabled": True,
            "method": "ols",
            "min_count": 3,
            "exposures": [
                {
                    "name": "price_tminus1",
                    "source": "market_cbond.daily_base",
                    "column": "cb_close_price",
                    "lag_trading_days": 1,
                }
            ],
        },
        raw_data_root=raw_root,
    )

    spec = neutralizer.cfg.exposures[0]
    exposure = neutralizer._raw_exposure(spec, date(2026, 1, 6), pd.Series(codes))
    assert exposure.tolist() == [100.0, 110.0, 120.0, 130.0, 140.0]

    out = neutralizer.apply(frame, ["factor_x"])
    assert abs(out["factor_x"].corr(pd.Series(exposure.to_numpy(dtype=float)))) < 1e-10


def test_neutralizer_rejects_partial_exclude_config():
    with pytest.raises(ValueError, match="partial neutralization is not supported"):
        build_neutralizer(
            {
                "enabled": True,
                "exposures": [{"name": "price", "source": "factor", "column": "price"}],
                "exclude_factors": ["amount_30m"],
            }
        )


def test_neutralizer_rejects_partial_factor_subset_config():
    with pytest.raises(ValueError, match="partial neutralization is not supported"):
        build_neutralizer(
            {
                "enabled": True,
                "factors": ["factor_x"],
                "exposures": [{"name": "price", "source": "factor", "column": "price"}],
            }
        )


def test_panel_exposure_uses_last_trade_before_panel_dt(tmp_path):
    panel_root = tmp_path / "panel"
    panel_dir = panel_root / "panels" / "cbond" / "T1430" / "2026-05"
    panel_dir.mkdir(parents=True)

    dt = pd.Timestamp("2026-05-20 14:30:00")
    panel = pd.DataFrame(
        {
            "dt": [dt] * 7,
            "code": [
                "110001.SH",
                "110001.SH",
                "110001.SH",
                "110002.SH",
                "110002.SH",
                "110002.SH",
                "110003.SH",
            ],
            "seq": [0, 1, 2, 0, 1, 2, 0],
            "trade_time": [
                pd.Timestamp("2026-05-20 14:25:00"),
                pd.Timestamp("2026-05-20 14:29:00"),
                pd.Timestamp("2026-05-20 14:31:00"),
                pd.Timestamp("2026-05-20 14:20:00"),
                pd.Timestamp("2026-05-20 14:30:00"),
                pd.Timestamp("2026-05-20 14:31:00"),
                pd.Timestamp("2026-05-19 15:00:00"),
            ],
            "last": [100.0, 101.0, 999.0, 200.0, 201.0, 999.0, 300.0],
        }
    ).set_index(["dt", "code", "seq"])
    panel.to_parquet(panel_dir / "20260520.parquet")

    neutralizer = build_neutralizer(
        {
            "enabled": True,
            "min_count": 3,
            "exposures": [
                {
                    "name": "last_1430",
                    "source": "panel",
                    "panel_name": "T1430",
                    "column": "last",
                    "select": "last_before_dt",
                }
            ],
        },
        panel_data_root=panel_root,
    )

    spec = neutralizer.cfg.exposures[0]
    out = neutralizer._panel_exposure(
        spec,
        date(2026, 5, 20),
        pd.Series(["110001.SH", "110002.SH", "110003.SH", "999999.SH"]),
    )

    assert out.iloc[:3].tolist() == [101.0, 201.0, 300.0]
    assert pd.isna(out.iloc[3])


def test_neutralizer_preserves_dt_when_exposure_source_is_missing(tmp_path):
    frame = pd.DataFrame(
        {
            "dt": [pd.Timestamp("2026-01-05 14:30")] * 3,
            "code": ["110001.SH", "110002.SH", "110003.SH"],
            "factor_x": [1.0, 2.0, 3.0],
        }
    )
    neutralizer = build_neutralizer(
        {
            "enabled": True,
            "min_count": 3,
            "exposures": [
                {
                    "name": "price",
                    "source": "market_cbond.daily_base",
                    "column": "cb_close_price",
                }
            ],
        },
        raw_data_root=tmp_path / "raw",
    )

    out = neutralizer.apply(frame, ["factor_x"])

    assert {"dt", "code", "factor_x"} <= set(out.columns)
    assert out["dt"].tolist() == frame["dt"].tolist()
    assert out["factor_x"].tolist() == frame["factor_x"].tolist()
