from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.app.usecases import factor_build_runtime, live_runtime
from cbond_on.app.usecases.run_factor_batch import execute as run_factor_batch_execute
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    CanonicalFactorStoreIntegrityError,
    FactorColumnContract,
    FactorTableContract,
)
from cbond_on.infra.factors.factor_table_resolution import (
    CanonicalFactorTableReader,
    assert_factor_table_read_only,
    build_factor_reader,
    resolve_factor_table_reference,
)


DAY = date(2026, 8, 26)


def _contract(name: str) -> FactorTableContract:
    return FactorTableContract(
        (
            FactorColumnContract(
                factor_id=name,
                factor_version="v1",
                contract_hash=hashlib.sha256(name.encode("utf-8")).hexdigest(),
                output_column=name,
            ),
        )
    )


def _frame(name: str, *, codes: tuple[str, ...] = ("110001.SH", "110002.SZ")) -> pd.DataFrame:
    dt = pd.Timestamp(DAY) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays([[dt] * len(codes), list(codes)], names=["dt", "code"])
    return pd.DataFrame({name: [float(offset + 1) for offset in range(len(codes))]}, index=index)


def _candidate_paths(root: Path, *, table_id: str, families: list[str] | None = None) -> dict[str, object]:
    table: dict[str, object] = {"table_id": table_id, "root": str(root)}
    if families is not None:
        table["families"] = families
    return {"factor_data_root": "must_not_be_used", "factor_table": table}


def _mark_table_consumer_ready(canonical: CanonicalFactorStore, table_id: str) -> None:
    migration_id = f"test-full-{table_id}"
    coverage = {"test_scope": "full_verified_fixture", "table_id": table_id}
    canonical.record_migration_attestation(
        table_id,
        {
            "migration_id": migration_id,
            "scope": "full",
            "expected_coverage": coverage,
        },
    )
    canonical.mark_migration_verified(
        table_id,
        migration_id=migration_id,
        coverage=coverage,
    )


def _mark_all_tables_consumer_ready(canonical: CanonicalFactorStore) -> None:
    canonical.initialize()
    for table_id in ("factor_library", "experiment", "live"):
        _mark_table_consumer_ready(canonical, table_id)


def _finalize_experiment_generation(canonical: CanonicalFactorStore, generation_id: str) -> None:
    coverage = {"test_scope": "full_verified_generation", "generation_id": generation_id}
    migration_id = f"test-generation-{generation_id}"
    canonical.record_experiment_generation_attestation(
        generation_id,
        {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage},
    )
    canonical.mark_experiment_generation_verified(
        generation_id,
        migration_id=migration_id,
        coverage=coverage,
    )
    canonical.finalize_stage_generation(
        generation_id,
        expected_days=[DAY],
        verification={"test": "reader-pin"},
    )


def _activate_experiment_generation(
    canonical: CanonicalFactorStore,
    generation_id: str,
    *,
    value: float,
) -> None:
    canonical.create_stage_generation(generation_id, source_evidence={"source": "test", "generation": generation_id})
    canonical.write_day(
        "experiment",
        DAY,
        _frame("alpha").assign(alpha=value),
        contract=_contract("alpha"),
        generation_id=generation_id,
    )
    _finalize_experiment_generation(canonical, generation_id)
    canonical.activate_generation(generation_id)


def test_wide_canonical_reader_validates_day_bundle_before_exposing_frame(tmp_path: Path) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    canonical.write_day("experiment", DAY, _frame("alpha"), contract=_contract("alpha"))
    _mark_all_tables_consumer_ready(canonical)
    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference({"table_id": "experiment", "root": str(canonical.root)}),
        panel_name="T1430",
    )

    observed = reader.read_day(DAY)
    assert observed.equals(canonical.read_day("experiment", DAY))
    assert reader.has_day(DAY) is True
    assert reader.day_path(DAY) == canonical.partition_paths("experiment", DAY).parquet_path

    paths = canonical.partition_paths("experiment", DAY)
    paths.done_path.unlink()
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial"):
        reader.read_day(DAY)
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial"):
        reader.has_day(DAY)


def test_experiment_reader_pins_active_generation_and_fences_a_concurrent_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    _activate_experiment_generation(canonical, "r88-v1", value=1.0)
    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference({"table_id": "experiment", "root": str(canonical.root)}),
        panel_name="T1430",
    )
    assert reader.read_day(DAY).iloc[0, 0] == 1.0
    assert reader.root == canonical.root / "experiment" / "generations" / "r88-v1"

    canonical.create_stage_generation("r88-v2", source_evidence={"source": "test", "generation": "r88-v2"})
    canonical.write_day(
        "experiment",
        DAY,
        _frame("alpha").assign(alpha=2.0),
        contract=_contract("alpha"),
        generation_id="r88-v2",
    )
    _finalize_experiment_generation(canonical, "r88-v2")
    original_read_day = reader._store.read_day

    def _activate_after_disk_read(*args: object, **kwargs: object) -> pd.DataFrame:
        output = original_read_day(*args, **kwargs)
        assert canonical.activate_generation("r88-v2") == "activated"
        return output

    monkeypatch.setattr(reader._store, "read_day", _activate_after_disk_read)
    with pytest.raises(RuntimeError, match="active generation changed"):
        reader.read_day(DAY)
    fresh = CanonicalFactorTableReader(
        resolve_factor_table_reference({"table_id": "experiment", "root": str(canonical.root)}),
        panel_name="T1430",
    )
    assert fresh.read_day(DAY).iloc[0, 0] == 2.0


def test_factor_library_reader_uses_only_explicit_families_and_combines_columns(tmp_path: Path) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    canonical.write_day("factor_library", DAY, _frame("alpha"), contract=_contract("alpha"), family="alpha")
    canonical.write_day("factor_library", DAY, _frame("micro"), contract=_contract("micro"), family="micro")
    canonical.write_day("factor_library", DAY, _frame("unlisted"), contract=_contract("unlisted"), family="unlisted")
    canonical.publish_factor_library_day(
        DAY,
        family_contracts={"alpha": _contract("alpha"), "micro": _contract("micro"), "unlisted": _contract("unlisted")},
    )
    _mark_all_tables_consumer_ready(canonical)

    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference(
            {"table_id": "factor_library", "root": str(canonical.root), "families": ["alpha", "micro"]}
        ),
        panel_name="T1430",
    )

    observed = reader.read_day(DAY)
    assert observed.columns.tolist() == ["alpha", "micro"]
    assert "unlisted" not in observed.columns
    assert len(reader.day_paths(DAY)) == 2
    with pytest.raises(ValueError, match="multiple explicit families"):
        reader.day_path(DAY)


def test_factor_library_reader_rejects_wildcards_and_misaligned_family_coverage(tmp_path: Path) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    canonical.write_day("factor_library", DAY, _frame("alpha"), contract=_contract("alpha"), family="alpha")
    canonical.write_day(
        "factor_library",
        DAY,
        _frame("micro", codes=("110001.SH",)),
        contract=_contract("micro"),
        family="micro",
    )
    canonical.publish_factor_library_day(
        DAY,
        family_contracts={"alpha": _contract("alpha"), "micro": _contract("micro")},
    )
    _mark_all_tables_consumer_ready(canonical)

    with pytest.raises(ValueError, match="wildcards"):
        resolve_factor_table_reference(
            {"table_id": "factor_library", "root": str(canonical.root), "families": ["*"]}
        )

    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference(
            {"table_id": "factor_library", "root": str(canonical.root), "families": ["alpha", "micro"]}
        ),
        panel_name="T1430",
    )
    with pytest.raises(RuntimeError, match="incompatible"):
        reader.read_day(DAY)


def test_reader_factory_requires_a_canonical_table_for_normal_consumers(tmp_path: Path) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    canonical.write_day("live", DAY, _frame("alpha"), contract=_contract("alpha"))
    _mark_all_tables_consumer_ready(canonical)

    with pytest.raises(RuntimeError, match="normal factor consumer cannot use"):
        build_factor_reader({"factor_data_root": str(tmp_path / "legacy")}, panel_name="T1430")

    candidate = build_factor_reader(
        _candidate_paths(canonical.root, table_id="live"),
        panel_name="T1430",
    )
    assert isinstance(candidate, CanonicalFactorTableReader)
    assert candidate.read_day(DAY).columns.tolist() == ["alpha"]

    with pytest.raises(RuntimeError, match="cannot write"):
        assert_factor_table_read_only(_candidate_paths(canonical.root, table_id="live"), operation="factor batch")
    with pytest.raises(RuntimeError, match="direct factor_data_root"):
        assert_factor_table_read_only({"factor_data_root": str(tmp_path / "legacy")}, operation="factor batch")


def test_factor_library_reader_rejects_a_pre_publish_logical_day_even_when_family_file_exists(tmp_path: Path) -> None:
    canonical = CanonicalFactorStore(tmp_path / "factor_store")
    canonical.write_day("factor_library", DAY, _frame("alpha"), contract=_contract("alpha"), family="alpha")
    _mark_all_tables_consumer_ready(canonical)
    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference(
            {"table_id": "factor_library", "root": str(canonical.root), "families": ["alpha"]}
        ),
        panel_name="T1430",
    )

    with pytest.raises(FileNotFoundError, match="logical day is not published"):
        reader.read_day(DAY)


def test_candidate_profile_blocks_factor_producers_before_any_pipeline_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _candidate_paths(tmp_path / "factor_store", table_id="live")
    monkeypatch.setattr(factor_build_runtime, "load_config_file", lambda _name: paths)
    with pytest.raises(RuntimeError, match="factor build cannot write"):
        factor_build_runtime.run()

    with pytest.raises(RuntimeError, match="factor batch cannot write"):
        run_factor_batch_execute(
            cfg={
                "panel_name": "T1430",
                "compute": {"engine": "rust", "execution_policy": "rust_first"},
            },
            paths_cfg=paths,
            start=DAY,
            end=DAY,
            refresh=False,
            overwrite=False,
        )


def test_standard_live_runtime_blocks_candidate_profile_before_any_live_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _candidate_paths(tmp_path / "factor_store", table_id="live")
    calls: list[str] = []

    def fake_load(name: str) -> dict[str, object]:
        calls.append(str(name))
        if str(name) == "live":
            return {"runtime": {"paths_config": "data/candidate"}}
        if str(name) == "paths":
            return paths
        raise AssertionError(f"unexpected config request: {name}")

    monkeypatch.setattr(live_runtime, "load_config_file", fake_load)
    monkeypatch.setattr(live_runtime, "configure_live_paths_profile", lambda _cfg: None)

    with pytest.raises(RuntimeError, match="standard live runtime cannot write"):
        live_runtime.run_once(start=DAY, target=DAY)
    assert calls == ["live", "paths"]
