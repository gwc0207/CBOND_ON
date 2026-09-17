from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    FactorColumnContract,
    FactorTableContract,
)
from cbond_on.workflows.research import factor_supplement as supplement


DAY = date(2026, 8, 25)
LIVE_ID = "live_alpha"
NONLIVE_IDS = ("research_alpha", "research_micro")


def _index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(pd.Timestamp(DAY) + pd.Timedelta(hours=14, minutes=30), "110001.SH")],
        names=["dt", "code"],
    )


def _mark_ready(store: CanonicalFactorStore, table_id: str) -> None:
    coverage = {"scope": "test_full", "operation_count": 0, "table_id": table_id}
    store.record_migration_attestation(
        table_id,
        {
            "migration_id": f"test-{table_id}-full",
            "scope": "full",
            "expected_coverage": coverage,
        },
    )
    store.mark_migration_verified(
        table_id,
        migration_id=f"test-{table_id}-full",
        coverage=coverage,
    )


def _config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    scratch_parent = tmp_path / "research_scratch"
    scratch_root = scratch_parent / "factor_supplement_v1"
    canonical_root = tmp_path / "factor_store"
    raw_root = tmp_path / "raw"
    clean_root = tmp_path / "clean"
    raw_root.mkdir()
    clean_root.mkdir()
    store = CanonicalFactorStore(canonical_root)
    store.initialize()
    _mark_ready(store, "factor_library")
    live_contract = FactorTableContract(
        (FactorColumnContract(LIVE_ID, "v1", "a" * 64, LIVE_ID),)
    )
    store.write_day(
        "live",
        DAY,
        pd.DataFrame({LIVE_ID: [1.0]}, index=_index()),
        contract=live_contract,
        source_evidence={"source": "test_live_input"},
    )
    _mark_ready(store, "live")
    monkeypatch.setattr(supplement, "RESEARCH_SCRATCH_PARENT", scratch_parent)
    monkeypatch.setattr(supplement, "CANONICAL_FACTOR_STORE_ROOT", canonical_root)
    cfg = {
        "research_only": True,
        "compute": {"engine": "research_python", "execution_policy": "research_catalog_python"},
        "inputs": {
            "raw_data_root": str(raw_root),
            "cleaned_data_root": str(clean_root),
            "live_factor_table": {"table_id": "live", "root": str(canonical_root)},
        },
        "output": {
            "scratch_root": str(scratch_root),
            "canonical_factor_store_root": str(canonical_root),
        },
    }
    return cfg, scratch_root, canonical_root, canonical_root


def _plan() -> dict:
    return {
        "disposition": "PLAN_READY_NO_EXECUTION",
        "catalog": {"path": "catalog.json", "sha256": "catalog-hash", "factor_count": 3},
        "release_exclude": {
            "path": "release.json",
            "sha256": "release-hash",
            "factor_count": 1,
            "factor_ids": [LIVE_ID],
            "excluded_catalog_factor_ids": [LIVE_ID],
            "release_factor_ids_not_in_catalog": [],
        },
        "supplement_scope": {
            "catalog_factor_count": 3,
            "non_live_catalog_factor_count": 2,
            "non_live_catalog_factor_ids": list(NONLIVE_IDS),
        },
    }


def _catalog() -> dict[str, supplement.CatalogFactor]:
    return {
        LIVE_ID: supplement.CatalogFactor(LIVE_ID, "alpha", "v1", "a" * 64, LIVE_ID),
        NONLIVE_IDS[0]: supplement.CatalogFactor(NONLIVE_IDS[0], "alpha", "v1", "b" * 64, NONLIVE_IDS[0]),
        NONLIVE_IDS[1]: supplement.CatalogFactor(NONLIVE_IDS[1], "micro", "v1", "c" * 64, NONLIVE_IDS[1]),
    }


def _install_execution_mocks(monkeypatch: pytest.MonkeyPatch, *, all_nan: bool = False) -> None:
    monkeypatch.setattr(supplement, "_assert_plan_artifacts_current", lambda _plan: None)
    monkeypatch.setattr(supplement, "_catalog_factor_map", lambda _cfg, *, plan: _catalog())
    monkeypatch.setattr(
        supplement,
        "issue_catalog_execution_permit",
        lambda *_args, **_kwargs: SimpleNamespace(catalog_sha256="catalog-hash", output_root=str(_kwargs["staging_factor_root"])),
    )
    monkeypatch.setattr(
        supplement,
        "build_permitted_factor_specs",
        lambda _permit: [SimpleNamespace(name=factor_id) for factor_id in NONLIVE_IDS],
    )
    monkeypatch.setattr(supplement, "load_config_file", lambda _key: {"panel_name": "T1430", "lead_minutes": 1})

    def fake_pipeline(_panel_root, factor_root, _start, _end, **kwargs):
        assert [spec.name for spec in kwargs["specs"]] == list(NONLIVE_IDS)
        FactorStore(Path(factor_root), panel_name="T1430").write_day(
            DAY,
            pd.DataFrame(
                {
                    NONLIVE_IDS[0]: [float("nan") if all_nan else 2.0],
                    NONLIVE_IDS[1]: [3.0],
                },
                index=_index(),
            ),
        )
        return object()

    monkeypatch.setattr(supplement, "run_factor_pipeline", fake_pipeline)


def test_execute_publishes_canonical_family_day_preserves_nan_and_removes_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, scratch_root, canonical_root, _live_root = _config(monkeypatch, tmp_path)
    _install_execution_mocks(monkeypatch, all_nan=True)
    legacy_sentinel = scratch_root / "factor_data" / "historic.txt"
    legacy_sentinel.parent.mkdir(parents=True)
    legacy_sentinel.write_text("do not write here", encoding="utf-8")

    manifest, manifest_path = supplement.execute_one_score_day(
        cfg,
        plan=_plan(),
        score_day=DAY,
        attempt_id="test-publish",
        scratch_root=scratch_root,
    )

    assert manifest_path.is_file()
    assert manifest["complete"] is True
    assert manifest["status"] == "completed_with_coverage_gaps"
    assert manifest["canonical_factor_library"]["logical_publish_status"] == "published"
    assert manifest["ephemeral_staging"]["removed"] is True
    assert legacy_sentinel.read_text(encoding="utf-8") == "do not write here"
    assert not list((scratch_root / "staging").glob("**/factor_data"))
    assert manifest["legacy_scratch_factor_store"]["writer"] == "forbidden"

    store = CanonicalFactorStore(canonical_root)
    logical = store.require_factor_library_day_published(DAY)
    assert logical["total_factor_count"] == 3
    alpha = store.read_day("factor_library", DAY, family="alpha")
    assert alpha.columns.tolist() == [LIVE_ID, NONLIVE_IDS[0]]
    assert alpha[NONLIVE_IDS[0]].isna().all()
    evidence_path = store.partition_paths("factor_library", DAY, family="alpha").manifest_path
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))["source_evidence"]
    assert NONLIVE_IDS[0] in evidence["coverage_quality"]["all_nan_factor_ids"]
    assert evidence["legacy_scratch_factor_store"] == "not_used"


def test_missing_live_companion_blocks_before_any_canonical_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, scratch_root, canonical_root, _live_root = _config(monkeypatch, tmp_path)
    canonical = CanonicalFactorStore(canonical_root)
    paths = canonical.partition_paths("live", DAY)
    paths.parquet_path.unlink()
    paths.manifest_path.unlink()
    paths.done_path.unlink()
    _install_execution_mocks(monkeypatch)

    with pytest.raises(FileNotFoundError, match="canonical factor day is absent"):
        supplement._read_live_factor_frame(cfg, score_day=DAY, live_ids=[LIVE_ID])
    assert not (scratch_root / "factor_data").exists()
    assert not list((canonical_root / "factor_library").glob("**/20260825.parquet"))


def test_compute_failure_is_audit_only_and_never_writes_legacy_factor_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg, scratch_root, canonical_root, _live_root = _config(monkeypatch, tmp_path)
    _install_execution_mocks(monkeypatch)
    monkeypatch.setattr(
        supplement,
        "run_factor_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic compute failure")),
    )

    manifest, path = supplement.execute_one_score_day(
        cfg,
        plan=_plan(),
        score_day=DAY,
        attempt_id="test-failure",
        scratch_root=scratch_root,
    )

    assert path.is_file()
    assert manifest["complete"] is False
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["ephemeral_staging"]["removed"] is True
    assert not (scratch_root / "factor_data").exists()
    assert not list((canonical_root / "factor_library").glob("**/20260825.parquet"))
    assert manifest["side_effect_boundary"]["live_runtime"] == "not_called"
    assert manifest["side_effect_boundary"]["database_write"] == "not_called"
