from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from cbond_on.infra.model.impl.lgbm.trainer import SplitData, build_dataset
from cbond_on.infra.model.runners import train_lgbm


class _Store:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def read_day(self, day: date) -> pd.DataFrame:
        return self._frame.copy()


def _factor_frame(day: date) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    return pd.DataFrame(
        {"factor": [1.0, 2.0]},
        index=pd.MultiIndex.from_arrays(
            [[dt, dt], ["110001.SH", "110002.SH"]],
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


def test_lagged_label_source_is_reanchored_to_factor_day(tmp_path: Path) -> None:
    signal_day = date(2026, 1, 5)
    source_label_day = date(2026, 1, 6)
    label_root = tmp_path / "labels"
    _write_label(label_root, source_label_day)

    split = build_dataset(
        factor_store=_Store(_factor_frame(signal_day)),
        label_root=label_root,
        days=[signal_day],
        factor_cols=["factor"],
        min_count=1,
        winsor_lower=None,
        winsor_upper=None,
        zscore=False,
        factor_time="14:30",
        label_time="14:42",
        label_day_by_factor_day={signal_day: source_label_day},
    )

    assert set(pd.to_datetime(split.dt).dt.date) == {signal_day}
    assert split.y.tolist() == [0.01, -0.02]


def test_lagged_score_data_does_not_open_its_future_label(tmp_path: Path, monkeypatch) -> None:
    signal_day = date(2026, 1, 5)
    future_label_day = date(2026, 1, 6)

    from cbond_on.infra.model.impl.lgbm import trainer

    monkeypatch.setattr(
        trainer,
        "_read_label_day",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("future label was read")),
    )
    split = build_dataset(
        factor_store=_Store(_factor_frame(signal_day)),
        label_root=tmp_path / "labels",
        days=[signal_day],
        factor_cols=["factor"],
        min_count=1,
        winsor_lower=None,
        winsor_upper=None,
        zscore=False,
        factor_time="14:30",
        label_time="14:42",
        require_label=False,
        label_day_by_factor_day={signal_day: future_label_day},
        read_label_when_not_required=False,
    )

    assert len(split.x) == 2
    assert split.y.isna().all()


def test_label_anchor_map_uses_trading_calendar_not_calendar_day(monkeypatch) -> None:
    friday = date(2026, 1, 2)
    monday = date(2026, 1, 5)
    tuesday = date(2026, 1, 6)
    monkeypatch.setattr(
        train_lgbm,
        "list_available_trading_days_from_raw",
        lambda *args, **kwargs: [friday, monday, tuesday],
    )

    mapping = train_lgbm._build_label_anchor_by_factor_day(
        raw_data_root="ignored",
        factor_days=[friday, monday, tuesday],
        anchor_lag_trading_days=1,
    )

    assert mapping == {friday: monday, monday: tuesday, tuesday: None}


def test_rolling_payload_embargoes_one_feature_day_for_label_lag() -> None:
    days = [date(2026, 1, value) for value in range(5, 11)]

    def split(day: date) -> SplitData:
        return SplitData(
            x=pd.DataFrame({"factor": [1.0]}),
            y=pd.Series([0.01]),
            dt=pd.Series([pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)]),
            code=pd.Series(["110001.SH"]),
        )

    payload = train_lgbm._prepare_rolling_payload(
        idx=5,
        days=days,
        window_days=4,
        train_ratio=0.5,
        factor_cols=["factor"],
        train_day_cache={day: split(day) for day in days[:4]},
        test_day_cache={days[5]: split(days[5])},
        label_anchor_lag_trading_days=1,
        label_day_by_factor_day={days[2]: days[3], days[3]: days[4]},
    )

    assert payload is not None
    assert payload["embargo_feature_days"] == [days[4]]
    assert payload["max_train_feature_day"] == days[3]
    assert payload["max_train_label_day"] == days[4]
    assert payload["max_train_label_day"] < payload["test_day"]
