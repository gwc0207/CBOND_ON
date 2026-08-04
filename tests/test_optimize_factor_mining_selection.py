from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import harness.tools.optimize_factor_mining_selection as optimizer


def _manifest() -> dict[str, object]:
    return {
        "factor_contract": {"start": "2025-01-01"},
        "metric_contract": {"ic_threshold_strictly_greater_than": 0.02},
        "redundancy_contract": {
            "within_family_threshold_strictly_less_than": 0.80,
            "cross_family_threshold_strictly_less_than": 0.70,
            "minimum_common_valid_days": 200,
            "max_selected": 100,
        },
    }


def _write_screen(root: Path, *, with_missing_pair: bool = False) -> None:
    root.mkdir(parents=True)
    (root / "screen_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
    pd.DataFrame(
        {
            "factor": ["factor_a", "factor_b", "factor_c", "factor_d"],
            "family": ["one", "one", "two", "three"],
            "abs_overall_mean_pearson_ic": [0.50, 0.40, 0.30, 0.20],
            "eligible_for_redundancy": [True, True, True, True],
            "selection_status": ["selected", "rejected", "rejected", "selected"],
        }
    ).to_csv(root / "factor_screen.csv", index=False)
    rows = [
        ("factor_a", "one", "factor_b", "one", "within_family", 0.80, 300, 0.90),
        ("factor_a", "one", "factor_c", "two", "cross_family", 0.70, 300, 0.90),
        ("factor_a", "one", "factor_d", "three", "cross_family", 0.70, 300, 0.10),
        ("factor_b", "one", "factor_c", "two", "cross_family", 0.70, 300, 0.10),
        ("factor_b", "one", "factor_d", "three", "cross_family", 0.70, 300, 0.10),
        ("factor_c", "two", "factor_d", "three", "cross_family", 0.70, 300, 0.10),
    ]
    if with_missing_pair:
        rows = [row for row in rows if {row[0], row[2]} != {"factor_c", "factor_d"}]
    pd.DataFrame(
        rows,
        columns=[
            "factor_a",
            "family_a",
            "factor_b",
            "family_b",
            "relation",
            "threshold",
            "common_valid_days",
            "redundancy",
        ],
    ).to_csv(root / "factor_pair_redundancy.csv", index=False)


def test_optimizer_selects_maximum_cardinality_then_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "research_scratch"
    monkeypatch.setattr(optimizer, "_RESEARCH_SCRATCH_PARENT", scratch)
    source = tmp_path / "completed_screen"
    output = scratch / "optimized"
    _write_screen(source)

    result = optimizer.run_optimizer(screen_dir=source, output_dir=output)

    assert result == output.resolve()
    selected = pd.read_csv(output / "optimized_accepted_factors.csv")
    assert selected["factor"].tolist() == ["factor_b", "factor_c", "factor_d"]
    summary = json.loads((output / "selection_summary.json").read_text(encoding="utf-8"))
    assert summary["maximum_selected_count"] == 3
    assert summary["source_greedy_selected_count"] == 2
    assert summary["conflict_edge_count"] == 2


def test_optimizer_treats_missing_pair_evidence_as_conflict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "research_scratch"
    monkeypatch.setattr(optimizer, "_RESEARCH_SCRATCH_PARENT", scratch)
    source = tmp_path / "completed_screen"
    output = scratch / "optimized"
    _write_screen(source, with_missing_pair=True)

    optimizer.run_optimizer(screen_dir=source, output_dir=output)

    conflicts = pd.read_csv(output / "selection_conflict_edges.csv")
    assert "redundancy_evidence_missing_fail_closed" in set(conflicts["reason"])
    selected = pd.read_csv(output / "optimized_accepted_factors.csv")
    assert len(selected) == 2
    assert not {"factor_c", "factor_d"}.issubset(set(selected["factor"]))


def test_optimizer_refuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "research_scratch"
    monkeypatch.setattr(optimizer, "_RESEARCH_SCRATCH_PARENT", scratch)
    source = tmp_path / "completed_screen"
    output = scratch / "optimized"
    _write_screen(source)
    output.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        optimizer.run_optimizer(screen_dir=source, output_dir=output)
