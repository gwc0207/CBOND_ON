from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.build_strict_t1429_state_history as state_tool
from cbond_on.core.config import load_config_file
from cbond_on.infra.live.model_switch import build_t1430_market_state_feature_row


def _write_snapshot(clean_root: Path, score_day: date) -> tuple[Path, int]:
    rows: list[dict[str, object]] = []
    for code, multiplier in [("110001.SH", 1.0), ("110002.SH", 1.1)]:
        for stamp, price in [
            ("09:30:00", 100.0), ("09:31:00", 101.0),
            ("09:35:00", 102.0), ("09:36:00", 103.0),
            ("10:00:00", 104.0), ("10:01:00", 105.0),
            ("10:30:00", 106.0), ("10:31:00", 107.0),
            ("11:00:00", 108.0), ("11:01:00", 109.0),
            ("13:00:00", 110.0), ("13:01:00", 111.0),
            ("13:30:00", 112.0), ("13:31:00", 113.0),
            ("14:00:00", 114.0), ("14:28:00", 115.0),
            ("14:29:00", 116.0),
            # Strict 14:29 must never read these observations.
            ("14:29:30", 1_000.0), ("14:30:00", 1_001.0),
        ]:
            rows.append(
                {
                    "code": code,
                    "trade_time": pd.Timestamp(f"{score_day} {stamp}"),
                    "last": price * multiplier,
                }
            )
    source = clean_root / "snapshot" / "cbond" / f"{score_day:%Y-%m}" / f"{score_day:%Y%m%d}.parquet"
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(source, index=False)
    return source, len(frame)


def _write_availability(
    manifest_root: Path,
    *,
    score_day: date,
    source_path: Path,
    source_rows: int,
    produced_day: date | None = None,
    produced_time: str = "14:59:00",
    declared_rows: int | None = None,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    mutate_done: Callable[[dict[str, object]], None] | None = None,
) -> None:
    production_day = produced_day or score_day
    run_id = f"run_{score_day:%Y%m%d}"
    clean = {
        "status": "success",
        "run_id": run_id,
        "trade_day": score_day.isoformat(),
        "produced_at": f"{production_day}T{produced_time}",
        "schema_version": state_tool.DATAHUB_V1_SCHEMA_VERSION,
        "required_profile": state_tool.DATAHUB_V1_REQUIRED_PROFILE,
        "assets_status": {"cbond": "success"},
        "validation": {"passed": True},
        "assets": {
            "cbond": {
                "file_path": str(source_path),
                "row_count": source_rows if declared_rows is None else declared_rows,
            }
        },
    }
    done = {
        "ready": True,
        "run_id": run_id,
        "trade_day": score_day.isoformat(),
        "produced_at": f"{production_day}T{produced_time}",
        "require_datasets": ["clean", "raw"],
        "allow_partial_manifest": False,
    }
    if mutate_manifest is not None:
        mutate_manifest(clean)
    if mutate_done is not None:
        mutate_done(done)
    clean_path = manifest_root / "clean" / f"{score_day:%Y-%m-%d}.json"
    done_path = manifest_root / "publish" / f"{score_day:%Y-%m-%d}.done"
    clean_path.parent.mkdir(parents=True)
    done_path.parent.mkdir(parents=True)
    clean_path.write_text(json.dumps(clean), encoding="utf-8")
    done_path.write_text(json.dumps(done), encoding="utf-8")


def test_same_day_v1_output_is_historical_reconstruction_not_forward_pit(tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
    )

    assert result.built_days == 1
    assert result.historical_reconstruction_days == 1
    assert result.blocked_days == 0
    assert result.paths.state_path.is_file()
    assert result.paths.audit_path.is_file()
    assert result.paths.calendar_path.is_file()
    state = pd.read_csv(result.paths.state_path)
    assert state["trade_date"].tolist() == [score_day.isoformat()]
    assert state.loc[0, "full0935_1430_mean"] == pytest.approx(
        build_t1430_market_state_feature_row(
            clean_root=clean_root,
            score_day=score_day,
            cutoff_time="14:29",
        )["full0935_1430_mean"]
    )
    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert audit["outcome"] == state_tool.HISTORICAL_RECONSTRUCTION_STATUS
    assert audit["provenance_classification"] == state_tool.HISTORICAL_RECONSTRUCTION_STATUS
    assert not bool(audit["forward_pit_certified"])
    assert bool(audit["availability_evidence_ok"])
    assert bool(audit["source_evidence_ok"])
    assert bool(audit["row_evidence_ok"])
    assert bool(audit["hash_evidence_ok"])
    assert len(audit["source_sha256"]) == 64
    assert len(audit["clean_manifest_sha256"]) == 64
    assert len(audit["publish_done_sha256"]) == 64
    assert audit["manifest_schema_version"] == state_tool.DATAHUB_V1_SCHEMA_VERSION
    assert audit["manifest_required_profile"] == state_tool.DATAHUB_V1_REQUIRED_PROFILE
    assert not bool(audit["publish_done_allow_partial_manifest"])
    assert json.loads(audit["publish_done_require_datasets"]) == ["clean", "raw"]
    assert audit["clean_manifest_produced_at"].startswith(score_day.isoformat())
    manifest = json.loads(result.paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == state_tool.HISTORICAL_RECONSTRUCTION_STATUS
    assert manifest["strict_cutoff_time"] == "14:29"
    assert manifest["expected_days"]["frozen_calendar_sha256"]
    assert manifest["outputs"]["calendar_path"] == str(result.paths.calendar_path)
    assert manifest["outputs"]["calendar_sha256"] == manifest["expected_days"]["frozen_calendar_sha256"]
    assert manifest["outputs"]["state_sha256"] == state_tool._sha256_file(result.paths.state_path)
    assert manifest["outputs"]["audit_sha256"] == state_tool._sha256_file(result.paths.audit_path)
    assert manifest["counts"] == {
        "requested_days": 1,
        "built_days": 1,
        "historical_reconstruction_days": 1,
        "blocked_days": 0,
        "forward_pit_certified_days": 0,
    }
    assert manifest["certification"]["forward_pit_certified"] is False
    assert "must reject" in manifest["certification"]["consumer_policy"]
    assert pd.read_csv(result.paths.calendar_path)["trade_date"].tolist() == [score_day.isoformat()]


@pytest.mark.parametrize(
    ("mutate_manifest", "mutate_done", "expected_reason"),
    [
        (
            lambda payload: payload.pop("schema_version"),
            None,
            "clean_manifest_schema_version_not_cbond_on_t1430_v1",
        ),
        (
            lambda payload: payload.__setitem__("schema_version", "legacy_v0"),
            None,
            "clean_manifest_schema_version_not_cbond_on_t1430_v1",
        ),
        (
            lambda payload: payload.pop("required_profile"),
            None,
            "clean_manifest_required_profile_not_cbond_on_live_t1430",
        ),
        (
            lambda payload: payload.__setitem__("validation", {}),
            None,
            "clean_manifest_validation_not_passed",
        ),
        (
            lambda payload: payload["assets"]["cbond"].pop("file_path"),  # type: ignore[index]
            None,
            "manifest_cbond_file_path_missing",
        ),
        (
            lambda payload: payload["assets"]["cbond"].__setitem__(  # type: ignore[index]
                "file_path", "C:/not-the-canonical-snapshot.parquet"
            ),
            None,
            "manifest_cbond_file_path_mismatch",
        ),
        (
            lambda payload: payload["assets"]["cbond"].pop("row_count"),  # type: ignore[index]
            None,
            "manifest_cbond_row_count_missing",
        ),
        (
            None,
            lambda payload: payload.__setitem__("allow_partial_manifest", True),
            "publish_done_allow_partial_manifest_not_false",
        ),
        (
            None,
            lambda payload: payload.__setitem__("require_datasets", ["raw"]),
            "publish_done_require_datasets_missing_clean",
        ),
    ],
    ids=[
        "schema-missing",
        "schema-legacy",
        "profile-missing",
        "validation-incomplete",
        "asset-path-missing",
        "asset-path-mismatch",
        "asset-row-count-missing",
        "done-allows-partial",
        "done-no-clean-dependency",
    ],
)
def test_legacy_or_incomplete_v1_evidence_is_blocked_before_state_build(
    tmp_path: Path,
    mutate_manifest: Callable[[dict[str, object]], None] | None,
    mutate_done: Callable[[dict[str, object]], None] | None,
    expected_reason: str,
) -> None:
    score_day = date(2026, 7, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
        mutate_manifest=mutate_manifest,
        mutate_done=mutate_done,
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
        state_builder=lambda **_: (_ for _ in ()).throw(AssertionError("invalid v1 evidence must not build state")),
    )

    assert result.built_days == 0
    assert result.historical_reconstruction_days == 0
    assert result.blocked_days == 1
    assert pd.read_csv(result.paths.state_path).empty
    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert audit["outcome"] == state_tool.BLOCKED_STATUS
    assert expected_reason in audit["reason"]
    manifest = json.loads(result.paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == state_tool.BLOCKED_STATUS
    assert manifest["certification"]["forward_pit_certified"] is False


def test_strict_generator_blocks_late_backfill_without_state_repair(tmp_path: Path) -> None:
    score_day = date(2025, 10, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
        produced_day=score_day + timedelta(days=161),
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
        state_builder=lambda **_: (_ for _ in ()).throw(AssertionError("late backfill must not build state")),
    )

    assert result.built_days == 0
    assert result.historical_reconstruction_days == 0
    assert result.blocked_days == 1
    assert pd.read_csv(result.paths.state_path).empty
    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert audit["outcome"] == "blocked"
    assert audit["reason"] == "availability_late_backfill"
    assert audit["clean_manifest_produced_at"].startswith("2026-04-09")
    manifest = json.loads(result.paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"


def test_strict_generator_blocks_same_day_availability_before_cutoff(tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
        produced_time="14:28:59",
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
        state_builder=lambda **_: (_ for _ in ()).throw(AssertionError("early availability must not build state")),
    )

    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert result.built_days == 0
    assert result.historical_reconstruction_days == 0
    assert result.blocked_days == 1
    assert audit["reason"] == "availability_before_cutoff"
    assert audit["clean_manifest_produced_at"].endswith("14:28:59")
    assert audit["publish_done_produced_at"].endswith("14:28:59")


def test_strict_generator_blocks_missing_hash_evidence_before_state_build(monkeypatch, tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
    )
    real_hash = state_tool._sha256_file

    def _no_source_hash(path: Path) -> str:
        if Path(path).resolve() == source.resolve():
            raise OSError("source hash unavailable")
        return real_hash(path)

    monkeypatch.setattr(state_tool, "_sha256_file", _no_source_hash)
    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
        state_builder=lambda **_: (_ for _ in ()).throw(AssertionError("missing hash must not build state")),
    )

    assert result.built_days == 0
    assert result.historical_reconstruction_days == 0
    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert audit["reason"] == "source_hash_failed:OSError"
    assert not bool(audit["hash_evidence_ok"])
    assert pd.read_csv(result.paths.state_path).empty


def test_strict_generator_blocks_row_count_mismatch_before_state_build(tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
        declared_rows=source_rows + 1,
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=[score_day],
        state_builder=lambda **_: (_ for _ in ()).throw(AssertionError("row mismatch must not build state")),
    )

    audit = pd.read_csv(result.paths.audit_path).iloc[0]
    assert result.built_days == 0
    assert result.historical_reconstruction_days == 0
    assert result.blocked_days == 1
    assert audit["reason"] == "manifest_cbond_row_count_mismatch"
    assert not bool(audit["row_evidence_ok"])
    assert pd.read_csv(result.paths.state_path).empty


def test_score_calendar_discovery_uses_only_canonical_daily_filenames(tmp_path: Path) -> None:
    score_root = tmp_path / "regsim_scores"
    for score_day in ["2025-10-30", "2025-10-31"]:
        path = score_root / score_day[:7] / f"{score_day}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trade_date,code,score\n", encoding="utf-8")

    assert state_tool.discover_score_calendar(score_root) == [date(2025, 10, 30), date(2025, 10, 31)]
    parser = state_tool.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--clean-root", "C:/clean",
                "--manifest-root", "C:/manifests",
                "--scratch-root", "C:/scratch",
                "--expected-days-csv", "calendar.csv",
                "--score-root", str(score_root),
            ]
        )


def test_score_root_calendar_is_frozen_under_scratch_without_label_access(tmp_path: Path) -> None:
    score_day = date(2026, 7, 30)
    score_root = tmp_path / "regsim_scores"
    score_path = score_root / f"{score_day:%Y-%m}" / f"{score_day}.csv"
    score_path.parent.mkdir(parents=True)
    score_path.write_text("trade_date,code,score\n", encoding="utf-8")
    clean_root = tmp_path / "clean"
    manifest_root = tmp_path / "manifests"
    source, source_rows = _write_snapshot(clean_root, score_day)
    _write_availability(
        manifest_root,
        score_day=score_day,
        source_path=source,
        source_rows=source_rows,
    )

    result = state_tool.build_strict_t1429_state_history(
        clean_root=clean_root,
        manifest_root=manifest_root,
        scratch_root=tmp_path / "scratch",
        expected_days=state_tool.discover_score_calendar(score_root),
        calendar_source_kind="score_root",
        calendar_source_path=score_root,
    )

    manifest = json.loads(result.paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["expected_days"]["source_kind"] == "score_root"
    assert manifest["expected_days"]["source_path"] == str(score_root)
    assert manifest["expected_days"]["frozen_calendar_sha256"]
    assert manifest["status"] == state_tool.HISTORICAL_RECONSTRUCTION_STATUS
    assert pd.read_csv(result.paths.calendar_path)["trade_date"].tolist() == [score_day.isoformat()]


def test_hard_similar60_config_is_forward_only_and_fails_closed() -> None:
    baseline = load_config_file(
        "models/lgbm/lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708"
    )
    candidate = load_config_file(
        "models/lgbm/lgbm_regsim_hard_similar60_pathfull360_1429_warm_r1_20260801"
    )
    state_cfg = candidate["feature_engineering"]["similar_day_training"]

    assert candidate["feature_engineering"]["regime"] == baseline["feature_engineering"]["regime"]
    assert candidate["sample_weight"] == baseline["sample_weight"]
    assert candidate["start"] == candidate["end"] == "2026-08-03"
    assert state_cfg["feature_set"] == "path_full_t1429"
    assert state_cfg["candidate_lookback_days"] == 360
    assert state_cfg["min_candidate_days"] == 360
    assert state_cfg["strict_recent_window"] is True
    assert state_cfg["fallback"] == "error"
    assert state_cfg["state_manifest_path"]["windows"].endswith(
        "t1430_market_state_features_pathfull_t1429_manifest.json"
    )
    assert candidate["score_only_no_target_label_read"] is True
    assert candidate["score_only_apply_tradable_filter"] is True
    assert candidate["incremental"]["warm_start"] is True
    assert candidate["experiment"]["promotion_requires"] == {
        "strict_same_day_state_days": 360,
        "forward_completed_score_days": 120,
    }
    assert candidate["artifact_output_root"]["windows"].startswith(
        "D:/cbond_on/research_scratch/"
    )
    assert candidate["neutralization_cache_root"]["windows"].startswith(
        "D:/cbond_on/research_scratch/"
    )
    assert candidate["score_output"]["windows"].startswith("D:/cbond_on/research_scratch/")
