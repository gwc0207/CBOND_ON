from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

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


def _write_strict_state_manifest(
    tmp_path,
    *,
    state_path: str,
    feature_set: str = "path_full_t1429",
    calendar_days: list[date] | None = None,
    forward_pit_certified: bool = True,
) -> str:
    state_file = Path(state_path)
    state_days = pd.to_datetime(pd.read_csv(state_file)["trade_date"]).dt.date.tolist()
    calendar_days = list(calendar_days if calendar_days is not None else state_days)
    calendar_path = tmp_path / "frozen_calendar.csv"
    audit_path = tmp_path / "states_audit.csv"
    manifest_path = tmp_path / "states_manifest.json"
    pd.DataFrame({"trade_date": calendar_days}).to_csv(calendar_path, index=False)
    state_day_set = set(state_days)
    audit_rows = []
    for day in calendar_days:
        built = day in state_day_set
        audit_rows.append(
            {
                "trade_date": day,
                "outcome": "built" if built else "blocked",
                "provenance_classification": (
                    "forward_pit_certified" if forward_pit_certified else "historical_reconstruction"
                ),
                "forward_pit_certified": bool(forward_pit_certified and built),
            }
        )
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    state_digest = hashlib.sha256(state_file.read_bytes()).hexdigest()
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    calendar_digest = hashlib.sha256(calendar_path.read_bytes()).hexdigest()
    certified_days = sum(row["forward_pit_certified"] for row in audit_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete" if forward_pit_certified else "historical_reconstruction_not_forward_certified",
                "strict_cutoff_time": "14:29",
                "state_feature_set": feature_set,
                "state_feature_columns": T1430_DISPERSION_FEATURE_SETS["path_full_t1430"],
                "expected_days": {
                    "count": len(calendar_days),
                    "source_kind": "test_calendar",
                    "source_path": str(calendar_path),
                    "source_sha256": calendar_digest,
                    "frozen_calendar_path": str(calendar_path),
                    "frozen_calendar_sha256": calendar_digest,
                },
                "outputs": {
                    "state_path": str(state_file),
                    "state_sha256": state_digest,
                    "audit_path": str(audit_path),
                    "audit_sha256": audit_digest,
                    "calendar_path": str(calendar_path),
                    "calendar_sha256": calendar_digest,
                },
                "counts": {
                    "requested_days": len(calendar_days),
                    "built_days": len(state_days),
                    "historical_reconstruction_days": 0 if forward_pit_certified else len(state_days),
                    "blocked_days": len(calendar_days) - len(state_days),
                    "forward_pit_certified_days": certified_days,
                },
                "certification": {
                    "status": "forward_pit_certified" if forward_pit_certified else "historical_reconstruction_not_forward_certified",
                    "forward_pit_certified": forward_pit_certified,
                    "forward_pit_certified_days": certified_days,
                    "reason": "test-only",
                    "consumer_policy": "reject unless forward_pit_certified",
                },
            }
        ),
        encoding="utf-8",
    )
    return str(manifest_path)


def _context(tmp_path, *, days: list[date], **overrides):
    state_path = _write_state_file(tmp_path, days=days)
    strict_recent_window = bool(overrides.get("strict_recent_window", False))
    feature_set = str(overrides.get("feature_set", "path_full_t1429" if strict_recent_window else "path_full_t1430"))
    state_manifest_path = (
        _write_strict_state_manifest(tmp_path, state_path=state_path, feature_set=feature_set)
        if strict_recent_window
        else None
    )
    cfg = {
        "feature_engineering": {
            "similar_day_training": {
                "enabled": True,
                "state_feature_path": state_path,
                "feature_set": feature_set,
                "candidate_lookback_days": 5,
                "train_top_k": 2,
                "validation_top_k": 1,
                "min_candidate_days": 3,
                "selection_mode": "nearest",
                "fallback": "error",
                **({"state_manifest_path": state_manifest_path} if state_manifest_path else {}),
                **overrides,
            }
        }
    }
    resolved = resolve_similar_day_training_config(cfg, results_root=tmp_path)
    assert resolved is not None
    return SimilarDayTrainingContext.from_config(resolved)


def _strict_context_config(tmp_path, *, state_path: str, state_manifest_path: str | None) -> dict:
    state_cfg = {
        "enabled": True,
        "state_feature_path": state_path,
        "feature_set": "path_full_t1429",
        "candidate_lookback_days": 5,
        "train_top_k": 2,
        "validation_top_k": 1,
        "min_candidate_days": 5,
        "selection_mode": "nearest",
        "fallback": "error",
        "strict_recent_window": True,
    }
    if state_manifest_path is not None:
        state_cfg["state_manifest_path"] = state_manifest_path
    return {"feature_engineering": {"similar_day_training": state_cfg}}


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


def test_default_candidate_window_can_still_backfill_older_complete_days(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    # The recent five-day window before day 7 is days 2..6.  Day 5 has no
    # state, so the legacy/default selector legitimately fills it with day 1.
    context = _context(tmp_path, days=[day for day in days if day != days[5]])

    selection = context.select(
        target_day=days[7],
        available_days=days[:7],
        expected_prior_days=list(reversed(days[:7])),
    )

    assert selection.ready
    assert selection.config.strict_recent_window is False
    # There are only four complete days in the latest five-day raw window;
    # a candidate count of five proves legacy mode filled from an older day.
    assert selection.candidate_days == 5


def test_strict_recent_window_derives_exact_prior_days_from_frozen_calendar(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(
        tmp_path,
        days=days,
        strict_recent_window=True,
        min_candidate_days=5,
    )

    selection = context.select(
        target_day=days[7],
        available_days=days[:7],
    )

    assert selection.ready
    assert selection.candidate_days == 5
    assert set(selection.selections["trade_date"]).issubset(set(days[2:7]))
    assert selection.audit_rows()[0]["strict_recent_window"] is True

    # The strict selector must not accept a runner/raw-calendar substitute.
    rejected = context.select(
        target_day=days[7],
        available_days=days[:7],
        expected_prior_days=days[:2],
    )
    assert not rejected.ready
    assert rejected.reason == "strict_recent_window_runtime_calendar_not_permitted"

    unknown_target = context.select(
        target_day=days[-1] + timedelta(days=1),
        available_days=days,
    )
    assert not unknown_target.ready
    assert unknown_target.reason == "strict_recent_window_target_not_in_frozen_calendar"


def test_strict_recent_window_refuses_missing_recent_trainable_day(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(
        tmp_path,
        days=days,
        strict_recent_window=True,
        min_candidate_days=5,
    )

    selection = context.select(
        target_day=days[7],
        available_days=[day for day in days[:7] if day != days[6]],
    )

    assert not selection.ready
    assert selection.candidate_days == 4
    assert selection.reason == "strict_recent_window_missing_state_0_missing_trainable_1"


def test_strict_recent_window_requires_manifest_path(tmp_path) -> None:
    start = date(2026, 1, 1)
    state_path = _write_state_file(tmp_path, days=[start + timedelta(days=i) for i in range(8)])

    with pytest.raises(ValueError, match="requires state_manifest_path"):
        resolve_similar_day_training_config(
            _strict_context_config(tmp_path, state_path=state_path, state_manifest_path=None),
            results_root=tmp_path,
        )


def test_strict_recent_window_rejects_missing_or_tampered_state_manifest(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    state_path = _write_state_file(tmp_path, days=days)
    missing_cfg = resolve_similar_day_training_config(
        _strict_context_config(
            tmp_path,
            state_path=state_path,
            state_manifest_path=str(tmp_path / "missing_manifest.json"),
        ),
        results_root=tmp_path,
    )
    assert missing_cfg is not None
    with pytest.raises(FileNotFoundError, match="state manifest missing"):
        SimilarDayTrainingContext.from_config(missing_cfg)

    manifest_path = _write_strict_state_manifest(tmp_path, state_path=state_path)
    tampered_cfg = resolve_similar_day_training_config(
        _strict_context_config(
            tmp_path,
            state_path=state_path,
            state_manifest_path=manifest_path,
        ),
        results_root=tmp_path,
    )
    assert tampered_cfg is not None
    Path(state_path).write_text(Path(state_path).read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        SimilarDayTrainingContext.from_config(tampered_cfg)


def test_strict_recent_window_rejects_nonforward_manifest_and_tampered_audit_or_calendar(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    state_path = _write_state_file(tmp_path, days=days)
    historical_manifest_path = _write_strict_state_manifest(
        tmp_path,
        state_path=state_path,
        forward_pit_certified=False,
    )
    historical_cfg = resolve_similar_day_training_config(
        _strict_context_config(
            tmp_path,
            state_path=state_path,
            state_manifest_path=historical_manifest_path,
        ),
        results_root=tmp_path,
    )
    assert historical_cfg is not None
    with pytest.raises(ValueError, match="not forward-PIT eligible"):
        SimilarDayTrainingContext.from_config(historical_cfg)

    manifest_path = _write_strict_state_manifest(tmp_path, state_path=state_path)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    audit_path = Path(payload["outputs"]["audit_path"])
    audit_original = audit_path.read_bytes()
    audit_path.write_bytes(audit_original + b"\n")
    audit_cfg = resolve_similar_day_training_config(
        _strict_context_config(tmp_path, state_path=state_path, state_manifest_path=manifest_path),
        results_root=tmp_path,
    )
    assert audit_cfg is not None
    with pytest.raises(ValueError, match="audit CSV SHA-256"):
        SimilarDayTrainingContext.from_config(audit_cfg)

    audit_path.write_bytes(audit_original)
    calendar_path = Path(payload["outputs"]["calendar_path"])
    calendar_original = calendar_path.read_bytes()
    calendar_path.write_bytes(calendar_original + b"\n")
    calendar_cfg = resolve_similar_day_training_config(
        _strict_context_config(tmp_path, state_path=state_path, state_manifest_path=manifest_path),
        results_root=tmp_path,
    )
    assert calendar_cfg is not None
    with pytest.raises(ValueError, match="frozen calendar CSV SHA-256"):
        SimilarDayTrainingContext.from_config(calendar_cfg)


def test_rolling_payload_uses_manifest_frozen_calendar_not_rolling_window(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(9)]
    context = _context(
        tmp_path,
        days=days,
        strict_recent_window=True,
        min_candidate_days=5,
    )

    def split(day: date) -> SplitData:
        return SplitData(
            x=pd.DataFrame({"factor": [1.0]}),
            y=pd.Series([0.1]),
            dt=pd.Series([pd.Timestamp(day)]),
            code=pd.Series(["110001.SH"]),
        )

    payload = _prepare_rolling_payload(
        idx=8,
        days=days,
        window_days=2,
        train_ratio=0.7,
        factor_cols=["factor"],
        train_day_cache={day: split(day) for day in days[:8]},
        test_day_cache={days[8]: split(days[8])},
        similarity_context=context,
    )

    assert payload is not None
    selection = payload["similarity_selection"]
    assert selection is not None and selection.ready
    # The ordinary rolling window is only two days, but the strict candidate
    # must use the frozen calendar's exact five predecessors of target day 8.
    assert selection.candidate_days == 5
    assert set(selection.selections["trade_date"]).issubset(set(days[3:8]))


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


def test_strict_recent_window_rejects_rolling_fallback_at_config_parse(tmp_path) -> None:
    start = date(2026, 1, 1)
    state_path = _write_state_file(tmp_path, days=[start + timedelta(days=i) for i in range(8)])
    manifest_path = _write_strict_state_manifest(tmp_path, state_path=state_path)
    cfg = _strict_context_config(
        tmp_path,
        state_path=state_path,
        state_manifest_path=manifest_path,
    )
    cfg["feature_engineering"]["similar_day_training"]["fallback"] = "rolling"

    with pytest.raises(ValueError, match="requires fallback=error"):
        resolve_similar_day_training_config(cfg, results_root=tmp_path)


def test_hard_similar60_payload_raises_instead_of_emitting_rolling_fallback(tmp_path) -> None:
    start = date(2026, 1, 1)
    days = [start + timedelta(days=i) for i in range(8)]
    context = _context(
        tmp_path,
        days=days,
        strict_recent_window=True,
        min_candidate_days=5,
    )

    def split(day: date) -> SplitData:
        return SplitData(
            x=pd.DataFrame({"factor": [1.0]}),
            y=pd.Series([0.1]),
            dt=pd.Series([pd.Timestamp(day)]),
            code=pd.Series(["110001.SH"]),
        )

    # Day 6 belongs to the frozen exact five-day pool for target day 7, but
    # is not trainable. A Hard Similar60 config must fail before constructing
    # any ordinary rolling payload/score.
    with pytest.raises(RuntimeError, match="strict_recent_window_missing_state_0_missing_trainable_1"):
        _prepare_rolling_payload(
            idx=7,
            days=days,
            window_days=2,
            train_ratio=0.7,
            factor_cols=["factor"],
            train_day_cache={day: split(day) for day in days[:6]},
            test_day_cache={days[7]: split(days[7])},
            similarity_context=context,
        )


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
