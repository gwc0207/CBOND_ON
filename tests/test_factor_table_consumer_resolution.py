from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import json5
import pandas as pd
import pytest

import cbond_on.core.config as config
from cbond_on.core.config import load_config_file, resolve_config_file_path
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    FactorColumnContract,
    FactorTableContract,
    FactorTableKind,
)
from cbond_on.infra.factors.factor_table_resolution import resolve_factor_table_reference


def _candidate_profile(tmp_path: Path, *, table_id: str, store_root: Path) -> dict[str, object]:
    return {
        "raw_data_root": str(tmp_path / "raw"),
        "clean_data_root": str(tmp_path / "clean"),
        "results_root": str(tmp_path / "candidate" / "results"),
        "read_only_input_roots": {
            "panel_data_root": str(tmp_path / "published_panel"),
            "label_data_root": str(tmp_path / "published_labels"),
        },
        "factor_table": {
            "table_id": table_id,
            "root": str(store_root),
            **({"families": ["alpha"]} if table_id == "factor_library" else {}),
        },
    }


def _register_library_alpha(store: CanonicalFactorStore) -> None:
    day = date(2026, 8, 26)
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    frame = pd.DataFrame(
        {"alpha": [1.0]},
        index=pd.MultiIndex.from_tuples([(dt, "110001.SH")], names=["dt", "code"]),
    )
    contract = FactorTableContract(
        (FactorColumnContract("alpha", "v1", "a" * 64, "alpha"),)
    )
    store.write_day("factor_library", day, frame, contract=contract, family="alpha")
    store.publish_factor_library_day(day, family_contracts={"alpha": contract})


def _mark_table_consumer_ready(store: CanonicalFactorStore, kind: FactorTableKind | str) -> None:
    table_id = FactorTableKind(kind).value
    migration_id = f"test-full-{table_id}"
    coverage = {"test_scope": "full_verified_fixture", "table_id": table_id}
    store.record_migration_attestation(
        table_id,
        {
            "migration_id": migration_id,
            "scope": "full",
            "expected_coverage": coverage,
        },
    )
    store.mark_migration_verified(
        table_id,
        migration_id=migration_id,
        coverage=coverage,
    )


def _mark_all_tables_consumer_ready(store: CanonicalFactorStore) -> None:
    for kind in FactorTableKind:
        _mark_table_consumer_ready(store, kind)


@pytest.mark.parametrize("kind", list(FactorTableKind))
def test_candidate_profile_resolves_only_the_declared_canonical_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: FactorTableKind,
) -> None:
    for name in (
        "CBOND_ON_RUNTIME_ROOT",
        "CBOND_ON_PATHS_PROFILE",
        "CBOND_ON_RAW_ROOT",
        "CBOND_ON_CLEAN_ROOT",
        "CBOND_ON_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CBOND_ON_PATHS_CONFIG", "candidate_factor_table.json5")

    store_root = tmp_path / "factor_store"
    store = CanonicalFactorStore(store_root)
    store.initialize()
    if kind is FactorTableKind.FACTOR_LIBRARY:
        _register_library_alpha(store)
    _mark_all_tables_consumer_ready(store)
    legacy_root = tmp_path / "legacy_factor_data"
    legacy_root.mkdir()

    resolved = config._apply_runtime_paths_profile(
        _candidate_profile(tmp_path, table_id=kind.value, store_root=store_root)
    )

    expected_root = store_root / kind.value / "alpha" if kind is FactorTableKind.FACTOR_LIBRARY else store_root / kind.value
    assert Path(resolved["factor_data_root"]) == expected_root
    assert Path(resolved["factor_data_root"]) != legacy_root
    assert resolved["factor_table"] == {
        "table_id": kind.value,
        "root": store_root.as_posix(),
        "resolved_factor_data_root": expected_root.as_posix(),
        **({"families": ["alpha"]} if kind is FactorTableKind.FACTOR_LIBRARY else {}),
    }


def test_candidate_table_reference_fails_closed_when_manifest_is_absent(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="manifest|three table|canonical"):
        resolve_factor_table_reference(
            {"table_id": "live", "root": str(tmp_path / "missing_factor_store")}
        )


def test_candidate_table_reference_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    store_root = tmp_path / "factor_store"
    store = CanonicalFactorStore(store_root)
    store.initialize()
    _mark_all_tables_consumer_ready(store)
    manifest_path = store_root / "live" / "table_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["table_id"] = "experiment"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(Exception, match="incompatible|mismatch|identity"):
        resolve_factor_table_reference({"table_id": "live", "root": str(store_root)})


def test_candidate_profile_rejects_a_direct_factor_root_alongside_table_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CBOND_ON_PATHS_CONFIG", "candidate_factor_table.json5")
    store_root = tmp_path / "factor_store"
    CanonicalFactorStore(store_root).initialize()
    profile = _candidate_profile(tmp_path, table_id="live", store_root=store_root)
    inputs = dict(profile["read_only_input_roots"])
    inputs["factor_data_root"] = str(tmp_path / "old_direct_root")
    profile["read_only_input_roots"] = inputs

    with pytest.raises(ValueError, match="must not declare"):
        config._apply_runtime_paths_profile(profile)


def test_factor_library_candidate_requires_an_explicit_family(tmp_path: Path) -> None:
    store_root = tmp_path / "factor_store"
    store = CanonicalFactorStore(store_root)
    store.initialize()
    _mark_all_tables_consumer_ready(store)

    with pytest.raises(ValueError, match="families.*required"):
        resolve_factor_table_reference({"table_id": "factor_library", "root": str(store_root)})


def test_initialized_or_smoke_table_cannot_resolve_as_a_candidate_input(tmp_path: Path) -> None:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.initialize()

    with pytest.raises(Exception, match="not consumer-ready"):
        resolve_factor_table_reference({"table_id": "live", "root": str(store.root)})

    store.record_migration_attestation(
        "live",
        {
            "migration_id": "smoke-live",
            "scope": "partial",
            "expected_coverage": {"test_scope": "smoke", "day_count": 1},
        },
    )
    store.mark_migration_partial(
        "live",
        migration_id="smoke-live",
        coverage={"test_scope": "smoke", "day_count": 1},
    )
    with pytest.raises(Exception, match="not consumer-ready"):
        resolve_factor_table_reference({"table_id": "live", "root": str(store.root)})


def test_active_live_profile_binds_the_admitted_canonical_live_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The active live profile reaches only the declared canonical live table."""

    for name in (
        "CBOND_ON_PATHS_CONFIG",
        "CBOND_ON_RUNTIME_ROOT",
        "CBOND_ON_PATHS_PROFILE",
        "CBOND_ON_RAW_ROOT",
        "CBOND_ON_CLEAN_ROOT",
        "CBOND_ON_DATA_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    live_cfg = load_config_file("live")
    active_paths = load_config_file(live_cfg["runtime"]["paths_config"])

    assert active_paths["factor_table"] == {
        "table_id": "live",
        "root": "D:/cbond_on/factor_store",
        "writer": "admitted_live",
        "resolved_factor_data_root": "D:/cbond_on/factor_store/live",
    }
    assert Path(active_paths["factor_data_root"]) == Path("D:/cbond_on/factor_store/live")


def test_nonactive_live_candidate_binds_the_live_table_id_profile() -> None:
    """The candidate is explicit and cannot alter the scheduler's active profile."""

    candidate_path = resolve_config_file_path("live/live_panel_store_candidate_20260827")
    candidate = json5.load(candidate_path.open(encoding="utf-8"))
    assert candidate["candidate"]["scheduler_selectable"] is False
    assert candidate["runtime"]["paths_config"] == "data/paths_factor_table_live_candidate_20260827"

    paths_path = resolve_config_file_path(candidate["runtime"]["paths_config"])
    raw_paths = json5.load(paths_path.open(encoding="utf-8"))
    assert raw_paths["factor_table"] == {
        "table_id": "live",
        "root": "D:/cbond_on/factor_store",
    }
    assert "factor_data_root" not in raw_paths["read_only_input_roots"]
