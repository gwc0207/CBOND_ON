from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cbond_on.app.usecases.factor_select_runtime import _write_factor_correlation_report
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    CanonicalFactorStoreIntegrityError,
    FactorColumnContract,
    FactorTableContract,
)
from cbond_on.infra.factors.factor_table_resolution import CanonicalFactorTableReader, resolve_factor_table_reference


def _contract() -> FactorTableContract:
    return FactorTableContract(
        (
            FactorColumnContract("f1", "v1", hashlib.sha256(b"f1").hexdigest(), "f1"),
            FactorColumnContract("f2", "v1", hashlib.sha256(b"f2").hexdigest(), "f2"),
        )
    )


def _frame(day: date) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays(
        [[dt] * 4, ["110001.SH", "110002.SH", "110003.SZ", "110004.SZ"]],
        names=["dt", "code"],
    )
    return pd.DataFrame({"f1": [-2.0, -0.5, 0.5, 2.0], "f2": [1.0, -1.0, 0.25, -0.5]}, index=index)


def _mark_ready(store: CanonicalFactorStore) -> None:
    for table_id in ("live", "experiment", "factor_library"):
        migration_id = f"test-full-{table_id}"
        coverage = {"test": "factor_select", "table_id": table_id}
        store.record_migration_attestation(
            table_id,
            {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage},
        )
        store.mark_migration_verified(table_id, migration_id=migration_id, coverage=coverage)


def _reader(tmp_path: Path) -> tuple[CanonicalFactorStore, CanonicalFactorTableReader, tuple[date, date]]:
    store = CanonicalFactorStore(tmp_path / "factor_store")
    days = (date(2026, 8, 24), date(2026, 8, 25))
    for day in days:
        store.write_day("live", day, _frame(day), contract=_contract())
    _mark_ready(store)
    reader = CanonicalFactorTableReader(
        resolve_factor_table_reference({"table_id": "live", "root": str(store.root)}),
        panel_name="T1430",
    )
    return store, reader, days


def test_factor_selection_correlation_reads_only_manifest_validated_days(tmp_path: Path) -> None:
    _store, reader, days = _reader(tmp_path)

    summary = _write_factor_correlation_report(
        factors=["f1", "f2"],
        out_dir=tmp_path / "report",
        factor_reader=reader,
        panel_name="T1430",
        start_day=days[0],
        end_day=days[-1],
        cfg={"min_pair_obs": 2},
    )

    assert summary["used_days"] == 2
    assert summary["missing_files"] == 0
    assert (tmp_path / "report" / "daily_coverage.csv").is_file()


def test_factor_selection_correlation_fails_closed_for_a_partial_canonical_day(tmp_path: Path) -> None:
    store, reader, days = _reader(tmp_path)
    store.partition_paths("live", days[1]).done_path.unlink()

    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial"):
        _write_factor_correlation_report(
            factors=["f1", "f2"],
            out_dir=tmp_path / "report",
            factor_reader=reader,
            panel_name="T1430",
            start_day=days[0],
            end_day=days[-1],
            cfg={"min_pair_obs": 2},
        )
