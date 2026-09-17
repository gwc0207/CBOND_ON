from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.migrate_factor_tables as migration
from cbond_on.infra.factors.canonical_store import CanonicalFactorStore, FactorTableKind


D_LIVE_EXPERIMENT = date(2024, 1, 3)
D_HISTORY = date(2025, 1, 2)
D_UNION = date(2026, 8, 25)
D_LIVE_TAIL = date(2026, 8, 27)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frame(day: date, values: dict[str, tuple[float, ...]], *, codes: tuple[str, ...] = ("110001.SH", "110002.SZ")) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    return pd.DataFrame(
        {column: list(items) for column, items in values.items()},
        index=pd.MultiIndex.from_arrays([[dt] * len(codes), list(codes)], names=["dt", "code"]),
    )


def _write_store(root: Path, frames: dict[date, pd.DataFrame]) -> dict[date, Path]:
    paths: dict[date, Path] = {}
    for day, frame in frames.items():
        path = root / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=True)
        paths[day] = path
    return paths


def _index_sha(frame: pd.DataFrame) -> str:
    text = frame.index.to_frame(index=False).astype(str).to_csv(index=False, lineterminator="\n")
    return _sha(text)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    catalog = tmp_path / "catalog.json"
    factors = [
        ("live_only", "live_family"),
        ("shared", "shared_family"),
        ("history_only", "shared_family"),
        ("supplement_only", "shared_family"),
    ]
    catalog.write_text(
        json.dumps(
            {
                "factor_count": len(factors),
                "factors": [
                    {
                        "factor_id": factor_id,
                        "output_column": factor_id,
                        "primary_family": family,
                        "contract_hash": _sha("catalog-" + factor_id),
                        "factor_version": "catalog-v1",
                    }
                    for factor_id, family in factors
                ],
            }
        ),
        encoding="utf-8",
    )
    live_release = tmp_path / "live_release.json"
    live_release.write_text(
        json.dumps(
            {
                "factor_count": 2,
                "instances": [
                    {
                        "factor_id": factor_id,
                        "output_col": None,
                        "contract_hash": _sha("catalog-" + factor_id),
                    }
                    for factor_id in ("live_only", "shared")
                ],
            }
        ),
        encoding="utf-8",
    )
    live_profile = tmp_path / "live_profile.json5"
    live_profile.write_text("{ factors: ['live_only', 'shared'] }", encoding="utf-8")
    experiment_profile = tmp_path / "experiment_profile.json5"
    experiment_profile.write_text(
        "{ factors: ['live_only', 'shared', 'history_only'] }", encoding="utf-8"
    )

    live_root = tmp_path / "live"
    live_paths = _write_store(
        live_root,
        {
            D_LIVE_EXPERIMENT: _frame(D_LIVE_EXPERIMENT, {"live_only": (1.0, 2.0), "shared": (3.0, 4.0)}),
            D_HISTORY: _frame(D_HISTORY, {"live_only": (5.0, 6.0), "shared": (7.0, 8.0)}),
            D_UNION: _frame(D_UNION, {"live_only": (9.0, 10.0), "shared": (11.0, 12.0)}),
            D_LIVE_TAIL: _frame(D_LIVE_TAIL, {"live_only": (13.0, 14.0), "shared": (15.0, 16.0)}),
        },
    )
    experiment_root = tmp_path / "experiment"
    experiment_paths = _write_store(
        experiment_root,
        {
            D_LIVE_EXPERIMENT: _frame(
                D_LIVE_EXPERIMENT,
                {"live_only": (1.0, 2.0), "shared": (3.0, 4.0), "history_only": (5.0, 6.0)},
            ),
            D_HISTORY: _frame(
                D_HISTORY,
                {"live_only": (5.0, 6.0), "shared": (7.0, 8.0), "history_only": (9.0, 10.0)},
            ),
        },
    )
    history_root = tmp_path / "history"
    # One extra history row creates the source-index difference that requires
    # date ownership; it never shares a target family with live_only.
    history_paths = _write_store(
        history_root,
        {
            D_HISTORY: _frame(
                D_HISTORY,
                {"shared": (7.0, 8.0, 9.0), "history_only": (20.0, 21.0, 22.0)},
                codes=("110001.SH", "110002.SZ", "110003.SH"),
            )
        },
    )
    supplement_root = tmp_path / "supplement" / "factor_data"
    supplement_paths = _write_store(
        supplement_root,
        {D_UNION: _frame(D_UNION, {"history_only": (30.0, 31.0), "supplement_only": (40.0, 41.0)})},
    )

    experiment_manifest = tmp_path / "experiment_manifest.json"
    experiment_manifest.write_text(
        json.dumps(
            {
                "execution_status": "completed_rust88_backfill",
                "model_training_ready": True,
                "factor_root": str(experiment_root),
                "factor_count": 3,
                "factors": ["live_only", "shared", "history_only"],
                "time_contract": {"panel_name": "T1430"},
                "range_contract": {"start": D_LIVE_EXPERIMENT.isoformat(), "end": D_HISTORY.isoformat()},
                "profile": {"profile_sha256": _sha("frozen-profile"), "specs_sha256": _sha("frozen-specs")},
                "completion": {
                    "full_completion": True,
                    "r88_factor_store_days": [D_LIVE_EXPERIMENT.isoformat(), D_HISTORY.isoformat()],
                },
            }
        ),
        encoding="utf-8",
    )
    history_manifest = tmp_path / "history_manifest.json"
    history_manifest.write_text(
        json.dumps(
            {
                "output_root": str(history_root),
                "panel_name": "T1430",
                "output_columns": ["shared", "history_only"],
                "output_files": [
                    {
                        "trade_day": D_HISTORY.isoformat(),
                        "relative_path": history_paths[D_HISTORY].relative_to(history_root).as_posix(),
                        "sha256": _sha256_file(history_paths[D_HISTORY]),
                        "row_count": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    supplement_manifest = tmp_path / "supplement_manifest.json"
    source = pd.read_parquet(supplement_paths[D_UNION])
    supplement_manifest.write_text(
        json.dumps(
            {
                "attempt_id": "20260826T020000_synthetic",
                "status": "completed_with_coverage_gaps",
                "complete": True,
                "integrity_passed": True,
                "factor_store": {
                    "path": str(supplement_paths[D_UNION]),
                    "sha256": _sha256_file(supplement_paths[D_UNION]),
                    "index_sha256": _index_sha(source),
                    "column_order_sha256": _sha(json.dumps(["history_only", "supplement_only"], separators=(",", ":"))),
                    "columns": ["history_only", "supplement_only"],
                },
                "factor_contract_hashes": {
                    "history_only": _sha("historical-history_only"),
                    "supplement_only": _sha("historical-supplement_only"),
                },
                "factor_contract_hashes_sha256": _sha("synthetic-contract-map"),
                "coverage_quality": {"all_nan_factor_count": 1, "all_nan_factor_ids": ["supplement_only"]},
            }
        ),
        encoding="utf-8",
    )
    return {
        "catalog_path": catalog,
        "live_release_path": live_release,
        "live_profile_path": live_profile,
        "experiment_profile_path": experiment_profile,
        "live_root": live_root,
        "experiment_root": experiment_root,
        "catalog_history_root": history_root,
        "supplement_root": supplement_root,
        "experiment_manifest_path": experiment_manifest,
        "catalog_history_manifest_path": history_manifest,
        "supplement_manifest_path": supplement_manifest,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_smoke_covers_all_routes_unions_same_family_and_stays_not_consumer_ready(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    target = tmp_path / "factor_store"
    plan = migration.build_migration_plan(**inputs, target_root=target, smoke_route_days=True)

    result = migration.execute_migration(plan, accept_date_owned_catalog_routing=True)

    assert result.written_by_table["live"] == 4
    assert result.written_by_table["experiment"] == 2
    store = CanonicalFactorStore(target)
    assert store.read_table_manifest("live")["migration"]["status"] == "partial_verified"
    with pytest.raises(Exception, match="not consumer-ready"):
        store.require_consumer_ready("live")
    union = store.read_day("factor_library", D_UNION, family="shared_family")
    assert union.columns.tolist() == ["shared", "history_only", "supplement_only"]
    assert union.shape == (2, 3)
    union_contract = store.read_table_manifest("factor_library")["registered_contracts"]
    assert any(item["family"] == "shared_family" for item in union_contract)
    logical = store.require_factor_library_day_published(D_UNION)
    assert logical["total_factor_count"] == 4
    day_manifest = json.loads(
        store.partition_paths("factor_library", D_UNION, family="shared_family").manifest_path.read_text(encoding="utf-8")
    )
    contract_path = target / "factor_library" / day_manifest["contract"]["contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    materialization = {row["factor_id"]: row["materialization_contract_origin"] for row in contract["columns"]}
    assert materialization == {
        "shared": "current_catalog",
        "history_only": "source_attested",
        "supplement_only": "source_attested",
    }
    historical = {row["factor_id"]: row["materialization_contract_hash"] for row in contract["columns"]}
    assert historical["history_only"] == _sha("historical-history_only")
    assert historical["supplement_only"] == _sha("historical-supplement_only")
    evidence = day_manifest["source_evidence"]
    assert evidence["same_name_values_coalesced"] is False
    assert any(
        fragment["historical_contract_evidence"].get("source_completion", {}).get("status")
        == "completed_with_coverage_gaps"
        for fragment in evidence["fragments"]
    )


def test_smoke_then_full_is_idempotent_and_only_full_opens_gate(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    target = tmp_path / "factor_store"
    smoke = migration.build_migration_plan(**inputs, target_root=target, smoke_route_days=True)
    migration.execute_migration(smoke, accept_date_owned_catalog_routing=True)

    full = migration.build_migration_plan(**inputs, target_root=target)
    result = migration.execute_migration(full, accept_date_owned_catalog_routing=True)
    assert result.already_present_by_table["live"] >= 1
    store = CanonicalFactorStore(target)
    assert store.require_consumer_ready("live")["migration"]["status"] == "full_verified"
    assert store.require_consumer_ready("experiment")["migration"]["status"] == "full_verified"
    assert store.require_consumer_ready("factor_library")["migration"]["status"] == "full_verified"
    experiment_manifest = json.loads(
        store.partition_paths("experiment", D_LIVE_EXPERIMENT).manifest_path.read_text(encoding="utf-8")
    )
    experiment_contract = json.loads(
        (target / "experiment" / experiment_manifest["contract"]["contract_path"]).read_text(encoding="utf-8")
    )
    assert {row["materialization_contract_origin"] for row in experiment_contract["columns"]} == {
        "source_manifest_unversioned"
    }
    assert all(row["materialization_contract_hash"] is None for row in experiment_contract["columns"])


def test_target_overlap_and_source_drift_fail_before_write(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(migration.FactorTableMigrationError, match="equal, contain, or be contained"):
        migration.build_migration_plan(**inputs, target_root=inputs["live_root"] / "child")

    plan = migration.build_migration_plan(**inputs, target_root=tmp_path / "target", days=[D_LIVE_EXPERIMENT])
    live_path = inputs["live_root"] / "factors" / "T1430" / "2024-01" / "20240103.parquet"
    changed = pd.read_parquet(live_path)
    changed.iloc[0, 0] = 999.0
    changed.to_parquet(live_path, index=True)
    with pytest.raises(migration.FactorTableMigrationError, match="changed"):
        migration.execute_migration(plan, accept_date_owned_catalog_routing=True)
    assert not (tmp_path / "target").exists()


def test_source_frame_cache_reads_one_source_day_once_and_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _write_inputs(tmp_path)
    plan = migration.build_migration_plan(**inputs, target_root=tmp_path / "target", days=[D_UNION])
    cache = migration.SourceFrameCache(plan)
    source_path = plan.sources["live_input"].scan.paths_by_day[D_UNION]
    original = migration.pd.read_parquet
    calls: list[Path] = []

    def tracked(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if Path(path) == source_path:
            calls.append(Path(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(migration.pd, "read_parquet", tracked)
    first = cache.frame("live_input", D_UNION)
    second = cache.frame("live_input", D_UNION)
    assert first is second
    assert calls == [source_path]
    cache.release_day(D_UNION)
    cache.frame("live_input", D_UNION)
    assert calls == [source_path, source_path]
