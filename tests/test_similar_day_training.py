from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cbond_on.infra.live.model_switch import T1430_DISPERSION_FEATURE_SETS
from cbond_on.infra.model.impl.lgbm.trainer import SplitData
from cbond_on.infra.model.runners.train_lgbm import (
    _apply_similar_day_kernel_weight,
    _prepare_rolling_payload,
)
from cbond_on.infra.model.similar_day_training import (
    SimilarDayTrainingContext,
    _gaussian_kernel_weights_for_ess,
    resolve_similar_day_training_config,
)


def _write_state_file(tmp_path, *, days: list[date]) -> str:
    cols = T1430_DISPERSION_FEATURE_SETS["path_full_t1430"]
    rows = []
    for idx, day in enumerate(days):
        row = {"trade_date": day}
        for col_idx, col in enumerate(cols):
            row[col] = float(idx * (col_idx + 1))
        rows.append(row)
    path = tmp_path / "states.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _context(tmp_path, *, days: list[date], **overrides):
    cfg = {
        "feature_engineering": {
            "similar_day_training": {
                "enabled": True,
                "state_feature_path": _write_state_file(tmp_path, days=days),
                "feature_set": "path_full_t1430",
                "candidate_lookback_days": 5,
                "train_top_k": 2,
                "validation_top_k": 1,
                "min_candidate_days": 3,
                "selection_mode": "nearest",
                "fallback": "error",
                **overrides,
            }
        }
    }
    resolved = resolve_similar_day_training_config(cfg, results_root=tmp_path)
    assert resolved is not None
    return SimilarDayTrainingContext.from_config(resolved)


def test_nearest_selection_uses_only_prior_available_days_and_local_scaling(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(tmp_path, days=days)

    selection = context.select(target_day=days[6], available_days=[*days[:6], days[7]])

    assert selection.ready
    assert selection.candidate_days == 5
    assert len(selection.train_days) == 2
    assert len(selection.validation_days) == 1
    assert all(day < days[6] for day in [*selection.train_days, *selection.validation_days])
    assert days[7] not in selection.train_days
    assert days[7] not in selection.validation_days
    assert selection.selections["rank"].tolist() == [1, 2, 3]


def test_latest_selection_is_a_causal_date_control(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(tmp_path, days=days, selection_mode="latest")

    selection = context.select(target_day=days[6], available_days=days[:6])

    assert selection.ready
    assert set(selection.train_days) == {days[4], days[5]}
    assert selection.validation_days == (days[3],)


def test_missing_current_state_returns_auditable_failure(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(5)]
    context = _context(tmp_path, days=days)

    selection = context.select(target_day=start + timedelta(days=99), available_days=days)

    assert not selection.ready
    assert selection.reason == "current_state_missing"
    assert selection.audit_rows()[0]["role"] == "fallback"


def test_rolling_payload_replaces_contiguous_split_with_similar_days(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(tmp_path, days=days)

    def split(day: date) -> SplitData:
        return SplitData(
            x=pd.DataFrame({"factor": [1.0]}),
            y=pd.Series([0.1]),
            dt=pd.Series([pd.Timestamp(day)]),
            code=pd.Series(["110001.SH"]),
        )

    payload = _prepare_rolling_payload(
        idx=6,
        days=days,
        window_days=2,
        train_ratio=0.7,
        factor_cols=["factor"],
        train_day_cache={day: split(day) for day in days[:6]},
        test_day_cache={days[6]: split(days[6])},
        similarity_context=context,
    )

    assert payload is not None
    assert len(payload["train_days"]) == 2
    assert len(payload["val_days"]) == 1
    assert len(payload["train_data"].y) == 2
    assert len(payload["val_data"].y) == 1
    assert payload["similarity_selection"].ready


def test_rolling_payload_records_state_gap_and_uses_explicit_rolling_fallback(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(7)]
    context = _context(tmp_path, days=days[:6], fallback="rolling")

    def split(day: date) -> SplitData:
        return SplitData(
            x=pd.DataFrame({"factor": [1.0]}),
            y=pd.Series([0.1]),
            dt=pd.Series([pd.Timestamp(day)]),
            code=pd.Series(["110001.SH"]),
        )

    payload = _prepare_rolling_payload(
        idx=6,
        days=days,
        window_days=5,
        train_ratio=0.7,
        factor_cols=["factor"],
        train_day_cache={day: split(day) for day in days[:6]},
        test_day_cache={days[6]: split(days[6])},
        similarity_context=context,
    )

    assert payload is not None
    assert not payload["similarity_selection"].ready
    assert payload["similarity_selection"].reason == "current_state_missing"
    assert len(payload["train_days"]) == 2
    assert len(payload["val_days"]) == 2


def test_kernel_selection_reserves_shared_validation_band_and_targets_day_ess(tmp_path) -> None:
    start = date(2024, 1, 1)
    days = [start + timedelta(days=i) for i in range(382)]
    context = _context(
        tmp_path,
        days=days,
        candidate_lookback_days=360,
        train_top_k=60,
        validation_top_k=20,
        min_candidate_days=360,
        selection_mode="kernel",
        kernel_target_effective_days=60,
    )

    selection = context.select(target_day=days[380], available_days=[*days[:380], days[381]])

    assert selection.ready
    assert selection.uses_kernel_weights
    assert selection.candidate_days == 360
    assert len(selection.train_days) == 340
    assert len(selection.validation_days) == 20
    assert not set(selection.train_days).intersection(selection.validation_days)
    assert all(day < days[380] for day in [*selection.train_days, *selection.validation_days])
    assert days[381] not in selection.train_days
    assert days[381] not in selection.validation_days
    assert selection.selections.loc[selection.selections["role"] == "validation", "rank"].tolist() == list(
        range(61, 81)
    )
    weights = np.array(list(selection.train_weights_by_day().values()), dtype=float)
    realized_ess = float(np.square(weights.sum()) / np.square(weights).sum())
    assert len(weights) == 340
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert realized_ess == pytest.approx(60.0, abs=0.1)


def test_kernel_selection_has_explicit_candidate_shortfall(tmp_path) -> None:
    start = date(2024, 1, 1)
    days = [start + timedelta(days=i) for i in range(381)]
    context = _context(
        tmp_path,
        days=days,
        candidate_lookback_days=360,
        train_top_k=60,
        validation_top_k=20,
        min_candidate_days=360,
        selection_mode="kernel",
        kernel_target_effective_days=60,
    )

    selection = context.select(target_day=days[380], available_days=days[:359])

    assert not selection.ready
    assert selection.reason == "insufficient_candidates_359_lt_360"


def test_kernel_weight_application_preserves_day_weight_ratios(tmp_path) -> None:
    start = date(2024, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(
        tmp_path,
        days=days,
        candidate_lookback_days=5,
        train_top_k=2,
        validation_top_k=1,
        min_candidate_days=5,
        selection_mode="kernel",
        kernel_target_effective_days=2,
    )
    selection = context.select(target_day=days[6], available_days=days[:6])
    assert selection.ready

    train_days = list(selection.train_days)
    frame_days = [day for day in train_days for _ in range(2)]
    split = SplitData(
        x=pd.DataFrame({"factor": np.arange(len(frame_days), dtype=float)}),
        y=pd.Series(np.zeros(len(frame_days), dtype=float)),
        dt=pd.Series(pd.to_datetime(frame_days)),
        code=pd.Series([f"1100{i:02d}.SH" for i in range(len(frame_days))]),
    )
    weighted, stats = _apply_similar_day_kernel_weight(split, selection)

    assert weighted.sample_weight is not None
    assert len(weighted.sample_weight) == len(split.y)
    assert np.isfinite(weighted.sample_weight.to_numpy(dtype=float)).all()
    observed = pd.DataFrame(
        {"day": pd.to_datetime(weighted.dt).dt.date, "weight": weighted.sample_weight}
    ).groupby("day")["weight"].sum()
    expected = pd.Series(selection.train_weights_by_day(), dtype=float)
    observed_ratio = observed / observed.sum()
    expected_ratio = expected / expected.sum()
    for day, expected_value in expected_ratio.items():
        assert observed_ratio.loc[day] == pytest.approx(expected_value)
    assert stats["similarity_kernel_final_day_effective_days"] == pytest.approx(
        selection.kernel_realized_effective_days,
        abs=0.1,
    )


def test_kernel_equal_distances_is_auditable_uniform_fallback() -> None:
    weights, bandwidth, realized_ess, status = _gaussian_kernel_weights_for_ess(
        pd.Series([2.0, 2.0, 2.0, 2.0]),
        target_effective_days=2,
        weight_floor=1e-12,
    )

    assert status == "uniform_distances"
    assert np.isinf(bandwidth)
    assert realized_ess == pytest.approx(4.0)
    assert np.allclose(weights, np.ones(4))
