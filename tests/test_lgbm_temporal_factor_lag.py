from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.infra.model.impl.lgbm.trainer import (
    TemporalFactorLagSpec,
    build_dataset,
    temporal_factor_feature_columns,
)
from cbond_on.infra.model.runners import train_lgbm


class _Store:
    def __init__(self, frames: dict[date, pd.DataFrame]) -> None:
        self._frames = frames
        self.calls: list[date] = []

    def read_day(self, day: date) -> pd.DataFrame:
        self.calls.append(day)
        return self._frames.get(day, pd.DataFrame()).copy()


def _factor_frame(day: date, values: dict[str, float]) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    codes = list(values)
    return pd.DataFrame(
        {"factor": [values[code] for code in codes]},
        index=pd.MultiIndex.from_arrays(
            [[dt] * len(codes), codes],
            names=["dt", "code"],
        ),
    )


def _write_label(root: Path, day: date) -> None:
    path = root / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "code": ["110001.SH", "110002.SH"],
            "trade_time": [
                pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=42),
                pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=42),
            ],
            "y": [0.01, -0.02],
        }
    ).to_parquet(path, index=False)


def test_temporal_factor_schema_is_stable_and_opt_in() -> None:
    spec = TemporalFactorLagSpec()

    assert temporal_factor_feature_columns(["a", "b"], None) == ["a", "b"]
    assert temporal_factor_feature_columns(["a", "b"], spec) == [
        "a__t0",
        "b__t0",
        "a__lag1",
        "b__lag1",
        "a__diff1",
        "b__diff1",
    ]
    assert train_lgbm._resolve_temporal_factor_lag_spec({}) is None


def test_temporal_factor_lag_uses_exact_prior_day_by_code_before_label_merge(tmp_path: Path) -> None:
    friday = date(2026, 1, 2)
    monday = date(2026, 1, 5)
    tuesday = date(2026, 1, 6)
    label_root = tmp_path / "labels"
    _write_label(label_root, tuesday)
    spec = TemporalFactorLagSpec()
    model_cols = temporal_factor_feature_columns(["factor"], spec)
    store = _Store(
        {
            friday: _factor_frame(friday, {"110001.SH": 1.0, "110002.SH": 5.0}),
            monday: _factor_frame(monday, {"110001.SH": 3.0, "110002.SH": 2.0}),
        }
    )

    split = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=[monday],
        factor_cols=model_cols,
        raw_factor_cols=["factor"],
        preprocess_factor_cols=model_cols,
        min_count=1,
        winsor_lower=None,
        winsor_upper=None,
        zscore=False,
        factor_time="14:30",
        label_time="14:42",
        label_day_by_factor_day={monday: tuesday},
        temporal_factor_lag=spec,
        previous_factor_day_by_factor_day={monday: friday},
    )

    assert split.code.tolist() == ["110001.SH", "110002.SH"]
    assert split.x.to_dict("list") == {
        "factor__t0": [3.0, 2.0],
        "factor__lag1": [1.0, 5.0],
        "factor__diff1": [2.0, -3.0],
    }
    assert split.y.tolist() == [0.01, -0.02]
    assert set(pd.to_datetime(split.dt).dt.date) == {monday}
    assert store.calls == [monday, friday]


def test_temporal_factor_lag_does_not_bridge_a_missing_prior_day(tmp_path: Path) -> None:
    thursday = date(2026, 1, 1)
    friday = date(2026, 1, 2)
    monday = date(2026, 1, 5)
    tuesday = date(2026, 1, 6)
    label_root = tmp_path / "labels"
    _write_label(label_root, tuesday)
    spec = TemporalFactorLagSpec()
    model_cols = temporal_factor_feature_columns(["factor"], spec)
    store = _Store(
        {
            thursday: _factor_frame(thursday, {"110001.SH": 9.0, "110002.SH": 9.0}),
            monday: _factor_frame(monday, {"110001.SH": 3.0, "110002.SH": 2.0}),
        }
    )

    split = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=[monday],
        factor_cols=model_cols,
        raw_factor_cols=["factor"],
        preprocess_factor_cols=model_cols,
        min_count=1,
        winsor_lower=None,
        winsor_upper=None,
        zscore=False,
        factor_time="14:30",
        label_time="14:42",
        label_day_by_factor_day={monday: tuesday},
        temporal_factor_lag=spec,
        previous_factor_day_by_factor_day={monday: friday},
    )

    assert split.x.empty
    assert store.calls == [monday, friday]
    assert thursday not in store.calls


def test_previous_factor_day_mapping_uses_raw_calendar(monkeypatch) -> None:
    friday = date(2026, 1, 2)
    monday = date(2026, 1, 5)
    monkeypatch.setattr(
        train_lgbm,
        "list_available_trading_days_from_raw",
        lambda *args, **kwargs: [friday, monday],
    )

    mapping = train_lgbm._build_previous_factor_day_by_factor_day(
        raw_data_root="ignored",
        factor_days=[monday],
        lag_trading_days=1,
    )

    assert mapping == {monday: friday}
