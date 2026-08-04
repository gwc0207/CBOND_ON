from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.merge_factor_mining_stores as merger


DAYS = (date(2025, 1, 2), date(2025, 1, 3))


def _frame(day: date, *, column: str, values: tuple[float, float], codes: tuple[str, str] = ("110001.SH", "110002.SZ")) -> pd.DataFrame:
    dt = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays([[dt, dt], list(codes)], names=["dt", "code"])
    return pd.DataFrame({column: list(values)}, index=index)


def _write_store(root: Path, *, column: str, frames: dict[date, pd.DataFrame]) -> None:
    for day, frame in frames.items():
        path = root / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=True)


def _two_sources(scratch_parent: Path) -> tuple[Path, Path]:
    first = scratch_parent / "source_a"
    second = scratch_parent / "source_b"
    _write_store(
        first,
        column="alpha",
        frames={
            DAYS[0]: _frame(DAYS[0], column="alpha", values=(1.0, 2.0)),
            DAYS[1]: _frame(DAYS[1], column="alpha", values=(3.0, 4.0)),
        },
    )
    _write_store(
        second,
        column="beta",
        frames={
            DAYS[0]: _frame(DAYS[0], column="beta", values=(10.0, 20.0)),
            DAYS[1]: _frame(DAYS[1], column="beta", values=(30.0, 40.0)),
        },
    )
    return first, second


def test_default_cli_preflight_reads_synthetic_scratch_stores_without_creating_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    first, second = _two_sources(scratch_parent)
    output = scratch_parent / "merged"

    assert merger.main(
        [
            "--input-root",
            str(first),
            "--input-root",
            str(second),
            "--output-root",
            str(output),
        ]
    ) == 0

    stdout = capsys.readouterr().out
    assert '"execute_requested": false' in stdout
    assert '"day_count": 2' in stdout
    assert '"output_column_count": 2' in stdout
    assert not output.exists()


def test_execute_merges_synthetic_stores_and_writes_manifest_catalog_and_audit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    first, second = _two_sources(scratch_parent)
    output = scratch_parent / "merged"
    catalog = tmp_path / "combined_catalog.json"
    catalog.write_text(
        json.dumps({"families": {"family_alpha": ["alpha"], "family_beta": ["beta"]}}),
        encoding="utf-8",
    )

    plan = merger.preflight_merge(
        input_roots=[str(first), str(second)],
        output_root=str(output),
        family_catalog=str(catalog),
    )
    assert merger.execute_merge(plan) == output.resolve()

    merged = pd.read_parquet(output / "factors" / "T1430" / "2025-01" / "20250102.parquet")
    assert merged.columns.tolist() == ["alpha", "beta"]
    assert merged.loc[(pd.Timestamp("2025-01-02 14:30:00"), "110001.SH"), "alpha"] == 1.0
    assert merged.loc[(pd.Timestamp("2025-01-02 14:30:00"), "110002.SZ"), "beta"] == 20.0
    assert (output / "factor_mining_family_catalog.json").read_text(encoding="utf-8") == catalog.read_text(encoding="utf-8")

    audit = pd.read_csv(output / "factor_source_audit.csv")
    assert audit["factor"].tolist() == ["alpha", "beta"]
    manifest = json.loads((output / "factor_mining_store_merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["research_only"] is True
    assert manifest["day_count"] == 2
    assert manifest["output_column_count"] == 2
    assert [source["file_count"] if "file_count" in source else len(source["files"]) for source in manifest["sources"]] == [2, 2]
    assert all(len(file["sha256"]) == 64 for source in manifest["sources"] for file in source["files"])


def test_no_catalog_emits_factor_to_source_audit_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    first, second = _two_sources(scratch_parent)
    output = scratch_parent / "merged_no_catalog"

    merger.execute_merge(merger.preflight_merge(input_roots=[str(first), str(second)], output_root=str(output)))

    assert not (output / "factor_mining_family_catalog.json").exists()
    audit = pd.read_csv(output / "factor_source_audit.csv")
    assert audit.set_index("factor")["source_position"].to_dict() == {"alpha": 1, "beta": 2}
    manifest = json.loads((output / "factor_mining_store_merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["family_catalog"]["supplied"] is False
    assert manifest["family_catalog"]["audit_mapping"] == "factor_source_audit.csv"


def test_family_catalog_explicitly_selects_a_vetted_subset_before_copying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    first, second = _two_sources(scratch_parent)
    for day in DAYS:
        path = first / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        frame = pd.read_parquet(path)
        frame["excluded_unvetted"] = 99.0
        frame.to_parquet(path, index=True)
    catalog = tmp_path / "vetted_combined_catalog.json"
    catalog.write_text(
        json.dumps({"families": {"vetted": ["alpha"], "expansion": ["beta"]}}),
        encoding="utf-8",
    )
    output = scratch_parent / "merged_vetted"

    plan = merger.preflight_merge(
        input_roots=[str(first), str(second)],
        output_root=str(output),
        family_catalog=str(catalog),
    )
    assert plan.selected_columns_by_source == (("alpha",), ("beta",))
    merger.execute_merge(plan)

    merged = pd.read_parquet(output / "factors" / "T1430" / "2025-01" / "20250102.parquet")
    assert merged.columns.tolist() == ["alpha", "beta"]
    assert "excluded_unvetted" not in merged.columns


def test_rejects_calendar_columns_catalog_and_output_reuse_and_outer_unions_indexes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)

    first, second = _two_sources(scratch_parent)
    with pytest.raises(ValueError, match="calendar mismatch"):
        (second / "factors" / "T1430" / "2025-01" / "20250103.parquet").unlink()
        merger.preflight_merge(input_roots=[str(first), str(second)], output_root=str(scratch_parent / "calendar_fail"))

    first, second = _two_sources(scratch_parent / "index_case")
    _write_store(
        second,
        column="beta",
        frames={
            DAYS[0]: _frame(DAYS[0], column="beta", values=(10.0, 20.0), codes=("110001.SH", "110003.SZ")),
            DAYS[1]: _frame(DAYS[1], column="beta", values=(30.0, 40.0), codes=("110001.SH", "110003.SZ")),
        },
    )
    index_output = scratch_parent / "index_case" / "index_outer_union"
    index_plan = merger.preflight_merge(
        input_roots=[str(first), str(second)],
        output_root=str(index_output),
    )
    assert len(index_plan.output_indexes[DAYS[0]]) == 3
    merger.execute_merge(index_plan)
    index_merged = pd.read_parquet(index_output / "factors" / "T1430" / "2025-01" / "20250102.parquet")
    dt = pd.Timestamp("2025-01-02 14:30:00")
    assert index_merged.index.tolist() == [(dt, "110001.SH"), (dt, "110002.SZ"), (dt, "110003.SZ")]
    assert pd.isna(index_merged.loc[(dt, "110002.SZ"), "beta"])
    assert pd.isna(index_merged.loc[(dt, "110003.SZ"), "alpha"])
    index_manifest = json.loads((index_output / "factor_mining_store_merge_manifest.json").read_text(encoding="utf-8"))
    assert index_manifest["index_alignment"]["mode"] == "per_day_outer_union_missing_source_values_are_nan"
    assert index_manifest["output_files"][0]["source_row_counts"] == [2, 2]
    assert index_manifest["output_files"][0]["source_missing_rows"] == [1, 1]

    first, second = _two_sources(scratch_parent / "column_case")
    _write_store(
        second,
        column="alpha",
        frames={
            DAYS[0]: _frame(DAYS[0], column="alpha", values=(10.0, 20.0)),
            DAYS[1]: _frame(DAYS[1], column="alpha", values=(30.0, 40.0)),
        },
    )
    with pytest.raises(ValueError, match="duplicate factor columns"):
        merger.preflight_merge(input_roots=[str(first), str(second)], output_root=str(scratch_parent / "column_fail"))

    first, second = _two_sources(scratch_parent / "catalog_case")
    invalid_catalog = tmp_path / "invalid_catalog.json"
    invalid_catalog.write_text(json.dumps({"families": {"only": ["alpha"]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="zero factors"):
        merger.preflight_merge(
            input_roots=[str(first), str(second)],
            output_root=str(scratch_parent / "catalog_fail"),
            family_catalog=str(invalid_catalog),
        )

    existing_output = scratch_parent / "existing"
    existing_output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="output reuse"):
        merger.preflight_merge(input_roots=[str(first), str(second)], output_root=str(existing_output))


def test_preflight_rejects_infinite_source_factor_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scratch_parent = tmp_path / "research_scratch"
    monkeypatch.setattr(merger, "_RESEARCH_SCRATCH_PARENT", scratch_parent)
    first, second = _two_sources(scratch_parent)
    source_day = first / "factors" / "T1430" / "2025-01" / "20250102.parquet"
    frame = pd.read_parquet(source_day)
    frame.loc[(pd.Timestamp("2025-01-02 14:30:00"), "110001.SH"), "alpha"] = float("inf")
    frame.to_parquet(source_day, index=True)

    with pytest.raises(ValueError, match="infinite factor values"):
        merger.preflight_merge(
            input_roots=[str(first), str(second)],
            output_root=str(scratch_parent / "infinite_fail"),
        )
