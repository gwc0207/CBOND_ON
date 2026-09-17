from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import cbond_on.infra.factors.quality as quality
from cbond_on.infra.factors.canonical_store import (
    CanonicalFactorStore,
    CanonicalFactorStoreIntegrityError,
    FactorColumnContract,
    FactorTableContract,
)


def _contract() -> FactorTableContract:
    return FactorTableContract(
        (FactorColumnContract("alpha", "v1", hashlib.sha256(b"alpha").hexdigest(), "alpha"),)
    )


def _frame(day: date, values: list[float]) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays(
        [[dt] * len(values), [f"11000{i}.SH" for i in range(1, len(values) + 1)]],
        names=["dt", "code"],
    )
    return pd.DataFrame({"alpha": values}, index=index)


def _mark_ready(store: CanonicalFactorStore) -> None:
    for table_id in ("live", "experiment", "factor_library"):
        migration_id = f"test-full-{table_id}"
        coverage = {"test": "quality", "table_id": table_id}
        store.record_migration_attestation(
            table_id,
            {"migration_id": migration_id, "scope": "full", "expected_coverage": coverage},
        )
        store.mark_migration_verified(table_id, migration_id=migration_id, coverage=coverage)


def _paths(store: CanonicalFactorStore) -> dict[str, object]:
    return {
        "raw_data_root": "unused",
        "factor_table": {"table_id": "live", "root": str(store.root)},
    }


def _cfg() -> dict[str, object]:
    return {
        "panel_name": "T1430",
        "factors": [{"name": "alpha", "factor": "test_operator"}],
    }


def test_quality_scan_uses_canonical_reader_and_reports_frame_statistics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    days = [date(2026, 8, 24), date(2026, 8, 25)]
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.write_day("live", days[0], _frame(days[0], [1.0, 2.0, 3.0]), contract=_contract())
    store.write_day("live", days[1], _frame(days[1], [1.0, float("nan"), 2.0]), contract=_contract())
    _mark_ready(store)
    monkeypatch.setattr(quality, "load_config_file", lambda _name: {"panel_name": "T1430"})
    monkeypatch.setattr(quality, "list_trading_days_from_raw", lambda *_args, **_kwargs: days)

    result = quality.run_factor_quality_scan(
        factor_cfg=_cfg(),
        paths_cfg=_paths(store),
        start=days[0],
        end=days[-1],
    )

    alpha = result["factor_health"][0]
    assert alpha["present_days"] == 2
    assert alpha["avg_non_null_ratio"] == pytest.approx(5.0 / 6.0)
    assert result["factor_table"]["table_id"] == "live"


def test_quality_scan_and_cleanup_fail_closed_for_canonical_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    day = date(2026, 8, 24)
    store = CanonicalFactorStore(tmp_path / "factor_store")
    store.write_day("live", day, _frame(day, [1.0, 2.0]), contract=_contract())
    _mark_ready(store)
    monkeypatch.setattr(quality, "load_config_file", lambda _name: {"panel_name": "T1430"})
    monkeypatch.setattr(quality, "list_trading_days_from_raw", lambda *_args, **_kwargs: [day])

    store.partition_paths("live", day).done_path.unlink()
    with pytest.raises(CanonicalFactorStoreIntegrityError, match="partial"):
        quality.run_factor_quality_scan(
            factor_cfg=_cfg(),
            paths_cfg=_paths(store),
            start=day,
            end=day,
        )
    with pytest.raises(RuntimeError, match="cannot mutate a canonical"):
        quality.cleanup_factor_store_columns(
            factor_dir=store.table_root("live") / "factors" / "T1430",
            start=day,
            end=day,
            columns_to_remove=["alpha"],
        )
