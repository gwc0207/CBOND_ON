from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    CanonicalFactorStoreConflictError,
    CanonicalFactorStoreIntegrityError,
    CanonicalFactorStoreValidationError,
    FactorColumnContract,
    FactorTableContract,
    FactorTableKind,
    PANEL_NAME,
    TABLE_MANIFEST_SCHEMA,
)


DAY = date(2026, 8, 26)


def _hash(seed: str) -> str:
    return (seed * 64)[:64]


def _contract(*, suffix: str = "", names: tuple[str, ...] = ("alpha", "beta")) -> FactorTableContract:
    return FactorTableContract(
        tuple(
            FactorColumnContract(
                factor_id=f"{name}{suffix}",
                factor_version=f"catalog-v1{suffix}",
                contract_hash=_hash(chr(ord("a") + position)),
                output_column=f"{name}{suffix}",
            )
            for position, name in enumerate(names)
        )
    )


def _frame(
    day: date = DAY,
    *,
    columns: tuple[str, ...] = ("alpha", "beta"),
    values: tuple[tuple[float, ...], ...] | None = None,
    codes: tuple[str, ...] = ("110002.SZ", "110001.SH"),
) -> pd.DataFrame:
    if values is None:
        values = tuple(
            tuple(float(index + column + 1) for column in range(len(columns)))
            for index in range(len(codes))
        )
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays([[dt] * len(codes), list(codes)], names=["dt", "code"])
    return pd.DataFrame(list(values), columns=list(columns), index=index)


def _mark_consumer_ready(store: CanonicalFactorStore, table_id: str) -> None:
    coverage = {"test_scope": "full_verified_fixture", "table_id": table_id}
    migration_id = f"test-full-{table_id}"
    store.record_migration_attestation(
        table_id,
        {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage},
    )
    store.mark_migration_verified(table_id, migration_id=migration_id, coverage=coverage)


def _finalize_generation(store: CanonicalFactorStore, generation_id: str) -> None:
    coverage = {"test_scope": "full_verified_generation", "generation_id": generation_id}
    migration_id = f"test-generation-{generation_id}"
    store.record_experiment_generation_attestation(
        generation_id,
        {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage},
    )
    store.mark_experiment_generation_verified(
        generation_id,
        migration_id=migration_id,
        coverage=coverage,
    )
    store.finalize_stage_generation(
        generation_id,
        expected_days=[DAY],
        verification={"test": "full-generation-fixture"},
    )


def test_initialize_creates_exactly_three_table_roots_and_identity_manifests(tmp_path: Path) -> None:
    root = tmp_path / "factor_store"
    store = CanonicalFactorStore(root)

    assert not root.exists()
    report_before = store.report()
    assert report_before.is_canonical_layout is True
    assert [row.exists for row in report_before.tables] == [False, False, False]

    store.initialize()

    assert sorted(path.name for path in root.iterdir()) == ["experiment", "factor_library", "live"]
    for kind in FactorTableKind:
        table_root = store.resolve_table_root(kind, require_manifest=True)
        manifest = store.read_table_manifest(kind)
        assert manifest["schema_version"] == TABLE_MANIFEST_SCHEMA
        assert manifest["table_id"] == kind.value
        assert manifest["display_name"] == kind.display_name
        assert manifest["panel"]["name"] == PANEL_NAME
        assert table_root == root / kind.value


def test_wide_experiment_day_is_atomic_readable_and_idempotent(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    contract = _contract()
    frame = _frame()

    result = store.write_day(
        FactorTableKind.EXPERIMENT,
        DAY,
        frame,
        contract=contract,
        source_evidence={"source": "existing_validated_factor_result", "sha256": _hash("f")},
    )

    assert result.status == "written"
    assert result.paths.parquet_path.is_file()
    assert result.paths.manifest_path.is_file()
    assert result.paths.done_path.is_file()
    assert not result.paths.lock_path.exists()

    manifest = json.loads(result.paths.manifest_path.read_text(encoding="utf-8"))
    done = json.loads(result.paths.done_path.read_text(encoding="utf-8"))
    assert manifest["table_id"] == "experiment"
    assert manifest["display_name"] == "实验用因子表"
    assert manifest["panel"]["name"] == "T1430"
    assert manifest["contract"]["factor_ids"] == ["alpha", "beta"]
    assert manifest["artifact"]["hash_semantics"] == "cbond_on_factor_frame_semantic_datetime_ns/v1"
    assert done["ready"] is True
    assert done["manifest_sha256"]

    # Input rows are canonicalised by the primary key, while values and schema
    # remain unchanged.  A retry is a read-only idempotent verification.
    output = store.read_day("experiment", DAY, expected_contract=contract)
    assert output.index.tolist() == [
        (pd.Timestamp("2026-08-26 14:30:00"), "110001.SH"),
        (pd.Timestamp("2026-08-26 14:30:00"), "110002.SZ"),
    ]
    assert output.columns.tolist() == ["alpha", "beta"]
    retry = store.write_day("实验用因子表", DAY, frame, contract=contract)
    assert retry.status == "already_present"
    assert retry.paths == result.paths

    report = store.report()
    experiment = next(item for item in report.tables if item.table_id == "experiment")
    assert experiment.manifest_valid is True
    assert experiment.completed_day_count == 1
    assert experiment.incomplete_day_count == 0
    assert experiment.registered_contract_count == 1


def test_existing_day_rejects_different_values_or_contract_without_overwrite(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    contract = _contract()
    frame = _frame()
    first = store.write_day("live", DAY, frame, contract=contract)
    before = first.paths.parquet_path.read_bytes()

    changed = _frame(values=((999.0, 2.0), (3.0, 4.0)))
    with pytest.raises(CanonicalFactorStoreConflictError, match="different values"):
        store.write_day("live", DAY, changed, contract=contract)
    assert first.paths.parquet_path.read_bytes() == before

    changed_contract = _contract(suffix="_v2")
    changed_columns = _frame(columns=("alpha_v2", "beta_v2"))
    with pytest.raises(CanonicalFactorStoreConflictError, match="different factor contract"):
        store.write_day("live", DAY, changed_columns, contract=changed_contract)
    assert first.paths.parquet_path.read_bytes() == before


def test_new_factor_contract_is_incremental_on_a_new_day_without_rewriting_history(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    first_contract = _contract()
    second_contract = _contract(suffix="_v2")
    first = store.write_day("experiment", DAY, _frame(), contract=first_contract)
    first_bytes = first.paths.parquet_path.read_bytes()
    second_day = date(2026, 8, 27)
    second = store.write_day(
        "experiment",
        second_day,
        _frame(day=second_day, columns=("alpha_v2", "beta_v2")),
        contract=second_contract,
    )

    assert first.paths.parquet_path.read_bytes() == first_bytes
    assert store.read_day("experiment", DAY, expected_contract=first_contract).columns.tolist() == ["alpha", "beta"]
    assert store.read_day("experiment", second_day, expected_contract=second_contract).columns.tolist() == [
        "alpha_v2",
        "beta_v2",
    ]
    manifest = store.read_table_manifest("experiment")
    assert [entry["contract_list_sha256"] for entry in manifest["registered_contracts"]] == [
        first_contract.sha256,
        second_contract.sha256,
    ]
    assert second.paths.parquet_path.is_file()


def test_experiment_generation_stages_then_atomically_activates_without_moving_legacy_flat(
    tmp_path: Path,
) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    legacy = store.write_day("experiment", DAY, _frame(), contract=_contract())
    _mark_consumer_ready(store, "experiment")

    generation_id = "r88-clean-direct-20260828"
    stage = store.create_stage_generation(
        generation_id,
        source_evidence={"source": "test_clean_direct", "calendar": "fixture"},
    )
    assert stage.generation_root == store.root / "experiment" / "generations" / generation_id
    assert store.active_experiment_generation_id() == "legacy_flat"
    staged = store.write_day(
        "experiment",
        DAY,
        _frame(values=((11.0, 12.0), (13.0, 14.0))),
        contract=_contract(),
        generation_id=generation_id,
    )
    assert staged.paths.parquet_path.is_file()
    assert legacy.paths.parquet_path.is_file()
    assert legacy.paths.parquet_path != staged.paths.parquet_path

    with pytest.raises(Exception, match="full_verified|not finalized|not ready"):
        store.activate_generation(generation_id)
    _finalize_generation(store, generation_id)
    assert stage.done_path is not None and stage.done_path.is_file()
    assert store.activate_generation(generation_id) == "activated"
    assert store.active_experiment_generation_id() == generation_id
    assert store.read_table_manifest("experiment")["active_generation"] == {
        "schema_version": "cbond_on_canonical_factor_experiment_generation_pointer/v1",
        "generation_id": generation_id,
    }
    assert store.read_day("experiment", DAY).iloc[0, 0] == 13.0
    experiment_report = next(item for item in store.report().tables if item.table_id == "experiment")
    assert experiment_report.completed_day_count == 1
    assert experiment_report.incomplete_day_count == 0
    assert experiment_report.registered_contract_count == 1
    assert legacy.paths.parquet_path.is_file()  # no legacy move/rewrite occurred.


def test_experiment_generation_rebuild_lock_and_layout_are_fail_closed(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()
    with store.experiment_rebuild_lock():
        with pytest.raises(Exception, match="lock is already held"):
            with store.experiment_rebuild_lock():
                pass

    # ``generations`` is an internal namespace available only under experiment.
    (store.root / "live" / "generations").mkdir()
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="unapproved top-level directory"):
        store.read_table_manifest("live")


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (_frame(codes=("110001.SH", "110001.SH")), "duplicate"),
        (_frame(day=date(2026, 8, 25)), "different dt day"),
        (
            pd.DataFrame(
                {"alpha": [1.0], "beta": [2.0]},
                index=pd.MultiIndex.from_tuples(
                    [(pd.Timestamp(DAY) + pd.Timedelta(hours=14, minutes=29), "110001.SH")],
                    names=["dt", "code"],
                ),
            ),
            "exact logical timestamp",
        ),
        (
            _frame(values=((float("inf"), 2.0), (3.0, 4.0))),
            "Inf",
        ),
    ],
)
def test_day_validation_fails_closed_for_primary_key_day_and_inf(
    tmp_path: Path,
    frame: pd.DataFrame,
    expected: str,
) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    with pytest.raises(CanonicalFactorStoreValidationError, match=expected):
        store.write_day("experiment", DAY, frame, contract=_contract())
    assert not (tmp_path / "factor_store").exists()


def test_day_validation_requires_exact_ordered_contract_columns(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    frame = _frame(columns=("beta", "alpha"))
    with pytest.raises(CanonicalFactorStoreValidationError, match="ordered factor contract"):
        store.write_day("experiment", DAY, frame, contract=_contract())


def test_factor_library_is_family_partitioned_but_remains_one_table(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    contract = _contract(names=("micro_alpha",))
    frame = _frame(columns=("micro_alpha",), values=((1.0,), (2.0,)))

    result = store.write_day(
        "factor_library",
        DAY,
        frame,
        contract=contract,
        family="microstructure.v1",
    )

    assert result.paths.table_root == tmp_path / "factor_store" / "factor_library"
    assert result.paths.parquet_path == (
        tmp_path
        / "factor_store"
        / "factor_library"
        / "microstructure.v1"
        / "factors"
        / "T1430"
        / "2026-08"
        / "20260826.parquet"
    )
    store.publish_factor_library_day(DAY, family_contracts={"microstructure.v1": contract})
    assert store.read_day("因子库内因子表", DAY, family="microstructure.v1").shape == (2, 1)
    library = store.read_table_manifest("factor_library")
    assert library["registered_contracts"][0]["family"] == "microstructure.v1"
    report = store.report()
    library_report = next(item for item in report.tables if item.table_id == "factor_library")
    assert library_report.families == ("microstructure.v1",)
    assert library_report.completed_day_count == 1

    with pytest.raises(CanonicalFactorStoreValidationError, match="requires a non-empty family"):
        store.write_day("factor_library", DAY, frame, contract=contract)
    with pytest.raises(CanonicalFactorStoreValidationError, match="does not accept a family"):
        store.write_day("experiment", DAY, _frame(), contract=_contract(), family="microstructure.v1")


def test_factor_library_logical_day_requires_exact_family_set(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    alpha = _contract(names=("alpha",))
    beta = _contract(names=("beta",))
    store.write_day("factor_library", DAY, _frame(columns=("alpha",), values=((1.0,), (2.0,))), contract=alpha, family="alpha")
    store.write_day("factor_library", DAY, _frame(columns=("beta",), values=((3.0,), (4.0,))), contract=beta, family="beta")

    with pytest.raises(CanonicalFactorStoreIntegrityError, match="exact complete family set"):
        store.publish_factor_library_day(DAY, family_contracts={"alpha": alpha})
    assert not store.library_day_paths(DAY).done_path.exists()


def test_partial_or_tampered_bundle_is_not_admitted_or_overwritten(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    contract = _contract()
    result = store.write_day("experiment", DAY, _frame(), contract=contract)

    result.paths.done_path.unlink()
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial artifact bundle"):
        store.read_day("experiment", DAY)
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial artifact bundle"):
        store.write_day("experiment", DAY, _frame(), contract=contract)

    # Restore a fresh valid day, then mutate the parquet bytes through its
    # regular writer.  The committed sha256 protects consumers from this.
    clean = CanonicalFactorStore(tmp_path / "clean_store")
    clean_result = clean.write_day("experiment", DAY, _frame(), contract=contract)
    mutated = pd.read_parquet(clean_result.paths.parquet_path)
    mutated.iloc[0, 0] = 1234.0
    mutated.to_parquet(clean_result.paths.parquet_path, index=True)
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="parquet hash"):
        clean.read_day("experiment", DAY)


def test_root_with_any_noncanonical_entry_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "factor_store"
    root.mkdir()
    (root / "unapproved_parallel_table").mkdir()
    store = CanonicalFactorStore(root)

    with pytest.raises(CanonicalFactorStoreIntegrityError, match="only three table IDs"):
        store.initialize()
    assert not (root / "live").exists()


def test_table_with_parallel_legacy_subroot_fails_closed(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()
    (tmp_path / "factor_store" / "experiment" / "old_factor_data").mkdir()

    with pytest.raises(CanonicalFactorStoreIntegrityError, match="unapproved top-level directory"):
        store.read_table_manifest("experiment")


def test_absent_day_fails_closed_after_table_initialization(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()
    with pytest.raises(FileNotFoundError, match="canonical factor day is absent"):
        store.read_day("live", DAY)


def test_full_consumer_gate_rejects_partial_attestation(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()
    coverage = {"scope": "partial", "day_count": 1}
    store.record_migration_attestation(
        "live",
        {"migration_id": "smoke", "scope": "partial", "expected_coverage": coverage},
    )
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="non-full"):
        store.mark_migration_verified("live", migration_id="smoke", coverage=coverage)
    store.mark_migration_partial("live", migration_id="smoke", coverage=coverage)
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="not consumer-ready"):
        store.require_consumer_ready("live")


def test_nan_is_allowed_but_numeric_contract_is_enforced(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    result = store.write_day(
        "experiment",
        DAY,
        _frame(values=((np.nan, 2.0), (3.0, np.nan))),
        contract=_contract(),
    )
    assert store.read_day("experiment", DAY).isna().sum().sum() == 2
    assert result.status == "written"

    object_frame = _frame(day=date(2026, 8, 27)).astype({"alpha": "object"})
    with pytest.raises(CanonicalFactorStoreValidationError, match="must be numeric"):
        store.write_day("experiment", date(2026, 8, 27), object_frame, contract=_contract())
