from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.plan_factor_table_migration as planner


DAY_1 = date(2026, 8, 20)
DAY_2 = date(2026, 8, 21)


def _frame(day: date, values: dict[str, tuple[float, float]]) -> pd.DataFrame:
    timestamp = pd.Timestamp(day) + pd.Timedelta(hours=14, minutes=30)
    index = pd.MultiIndex.from_arrays(
        [[timestamp, timestamp], ["110001.SH", "110002.SZ"]], names=["dt", "code"]
    )
    return pd.DataFrame({name: list(items) for name, items in values.items()}, index=index)


def _write_store(root: Path, frames: dict[date, pd.DataFrame]) -> None:
    for day, frame in frames.items():
        path = root / "factors" / "T1430" / f"{day:%Y-%m}" / f"{day:%Y%m%d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=True)


def _write_contracts(root: Path) -> tuple[Path, Path, Path, Path]:
    catalog = root / "factor_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "factor_count": 4,
                "snapshot_id": "synthetic",
                "factors": [
                    {
                        "factor_id": "base_a",
                        "primary_family": "base_family",
                        "contract_hash": "hash_base_a",
                        "factor_version": "v1",
                    },
                    {
                        "factor_id": "research_x",
                        "primary_family": "research_family",
                        "contract_hash": "hash_research_x",
                        "factor_version": "v1",
                    },
                    {
                        "factor_id": "research_y",
                        "primary_family": "research_family",
                        "contract_hash": "hash_research_y",
                        "factor_version": "v1",
                    },
                    {
                        "factor_id": "research_z",
                        "primary_family": "research_family",
                        "contract_hash": "hash_research_z",
                        "factor_version": "v1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    release = root / "release.json"
    release.write_text(
        json.dumps(
            {
                "factor_count": 2,
                "instances": [
                    {"factor_id": "base_a", "contract_hash": "hash_base_a"},
                    {"factor_id": "research_x", "contract_hash": "hash_research_x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    live_profile = root / "live_profile.json5"
    live_profile.write_text(
        "{ admission_profile: 'synthetic_live', specs_sha256: 'synthetic', "
        "factors: ['base_a', 'research_x'] }",
        encoding="utf-8",
    )
    profile = root / "experiment_profile.json5"
    profile.write_text(
        "{ admission_profile: 'synthetic_experiment', specs_sha256: 'synthetic', "
        "factors: ['base_a', 'research_x', 'research_y'] }",
        encoding="utf-8",
    )
    return catalog, release, live_profile, profile


def _inputs(tmp_path: Path) -> dict[str, Path]:
    catalog, release, live_profile, profile = _write_contracts(tmp_path)
    live = tmp_path / "live"
    experiment = tmp_path / "experiment"
    catalog_history = tmp_path / "catalog_history"
    supplement = tmp_path / "supplement" / "factor_data"
    _write_store(
        live,
        {
            DAY_1: _frame(DAY_1, {"base_a": (1.0, 2.0), "research_x": (3.0, 4.0)}),
            DAY_2: _frame(DAY_2, {"base_a": (5.0, 6.0), "research_x": (7.0, 8.0)}),
        },
    )
    _write_store(
        experiment,
        {
            DAY_1: _frame(
                DAY_1,
                {"base_a": (1.0, 2.0), "research_x": (3.0, 4.0), "research_y": (9.0, 10.0)},
            )
        },
    )
    _write_store(
        catalog_history,
        {DAY_1: _frame(DAY_1, {"research_x": (3.0, 4.0), "research_y": (30.0, 40.0), "research_z": (5.0, 6.0)})},
    )
    _write_store(
        supplement,
        {DAY_2: _frame(DAY_2, {"research_y": (50.0, 60.0), "research_z": (70.0, 80.0)})},
    )
    return {
        "catalog_path": catalog,
        "live_release_path": release,
        "live_profile_path": live_profile,
        "experiment_profile_path": profile,
        "live_root": live,
        "experiment_root": experiment,
        "catalog_history_root": catalog_history,
        "supplement_root": supplement,
    }


def test_build_plan_is_read_only_and_maps_the_three_tables(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    paths_before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))

    plan = planner.build_plan(**inputs, value_conflict_check=True, cross_table_value_audit=True)

    assert plan["read_only"] is True
    assert plan["execution_boundary"]["writes_any_files"] is False
    assert plan["catalog"]["factor_count"] == 4
    assert set(plan["contracts"]) == {"实盘所需要的因子表", "实验用因子表", "因子库内因子表"}
    assert plan["contracts"]["实盘所需要的因子表"]["release_profile_order_match"] is True
    assert plan["contracts"]["实验用因子表"]["source_membership"]["ordered_membership_match"] is True
    assert plan["composition_checks"]["live_and_supplement_union_matches_catalog"] is True
    assert plan["composition_checks"]["live_and_supplement_factor_ids_disjoint"] is True
    assert any(
        row["target_table"] == "因子库内因子表"
        and row["source"] == "catalog_history_input"
        and row["family_routes"] == [
            {"family": "research_family", "factor_ids": ["research_x", "research_y", "research_z"]}
        ]
        for row in plan["candidate_routes"]
    )
    comparisons = {(item["left_source"], item["right_source"]): item for item in plan["source_comparisons"]}
    assert comparisons[("live_input", "experiment_input")]["value"]["finite_value_conflict_cell_count"] == 0
    assert comparisons[("experiment_input", "catalog_history_input")]["value"]["finite_value_conflict_cell_count"] == 2
    assert paths_before == sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))


def test_cli_prints_json_and_never_accepts_an_output_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inputs = _inputs(tmp_path)
    args: list[str] = []
    for name, value in inputs.items():
        args.extend([f"--{name.replace('_', '-')}", str(value)])
    assert planner.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["execution_boundary"]["stdout_only"] is True
    with pytest.raises(SystemExit):
        planner.main(args + ["--output", str(tmp_path / "illegal")])
    assert not (tmp_path / "illegal").exists()


def test_plan_fails_closed_when_live_store_does_not_match_its_release(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = inputs["live_root"] / "factors" / "T1430" / "2026-08" / "20260820.parquet"
    frame = pd.read_parquet(first).drop(columns=["research_x"])
    frame.to_parquet(first, index=True)

    plan = planner.build_plan(**inputs, value_conflict_check=False)
    membership = plan["contracts"]["实盘所需要的因子表"]["source_membership"]
    assert membership["missing_factor_ids"] == ["research_x"]
    assert membership["ordered_membership_match"] is False
