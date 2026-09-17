from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import pytest

import harness.tools.compose_research_factor_catalog as composer


EXPANSION_MODULE = "cbond_on.domain.factors.operators.research_factor_mining_daily_expansion_v1"
INCREMENTAL_MODULE = "cbond_on.domain.factors.operators.research_factor_mining_daily_incremental_v1"


def _write_vetted(path: Path, mapping: dict[str, list[str]] | None = None) -> Path:
    payload = mapping or OrderedDict(
        [
            ("v3_daily_price_subset", ["dpx_body_range", "dret_tstat"]),
            ("v3_structural_subset", ["base_premium_z"]),
        ]
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_plan_preserves_vetted_subset_and_collects_repeatable_research_modules(tmp_path: Path) -> None:
    vetted_path = _write_vetted(tmp_path / "vetted_v3.json")

    plan = composer.build_composition_plan(
        vetted_v3_catalog=vetted_path,
        module_names=[EXPANSION_MODULE, INCREMENTAL_MODULE],
    )

    assert list(plan.catalog.items())[:2] == [
        ("v3_daily_price_subset", ["dpx_body_range", "dret_tstat"]),
        ("v3_structural_subset", ["base_premium_z"]),
    ]
    assert plan.catalog["daily_return_regime_transition"][0] == "dret_last_return"
    assert plan.catalog["conditional_premium_repricing_residual"][0] == "cprr_premium_level_residual"
    # The vetted fixture contains three signals across two families; do not
    # conflate its signal count with the family count.
    assert plan.family_count == 2 + 8 + 5
    assert plan.signal_count == 3 + 64 + 28
    assert [summary.identifier for summary in plan.modules] == [EXPANSION_MODULE, INCREMENTAL_MODULE]
    assert all(len(summary.sha256) == 64 for summary in plan.modules)

    manifest = composer.build_manifest(plan)
    expected_hash = hashlib.sha256(composer._canonical_json_bytes(plan.catalog)).hexdigest()
    assert manifest["combined"]["family_catalog_sha256"] == expected_hash
    assert manifest["vetted_v3"]["signal_count"] == 3
    assert [item["signal_count"] for item in manifest["research_modules"]] == [64, 28]


def test_wrapped_strict_pit_v3_family_schema_is_normalized_fail_closed(tmp_path: Path) -> None:
    wrapped = tmp_path / "strict_pit_v3.json"
    wrapped.write_text(
        json.dumps(
            {
                "families": {
                    "v3_daily_subset": ["v3_factor_a", "v3_factor_b"],
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    catalog, summary = composer.load_vetted_v3_family_catalog(wrapped)

    assert catalog == {"v3_daily_subset": ["v3_factor_a", "v3_factor_b"]}
    assert summary.family_count == 1
    assert summary.signal_count == 2

    malformed = tmp_path / "strict_pit_v3_with_metadata.json"
    malformed.write_text(
        json.dumps(
            {"families": {"v3_daily_subset": ["v3_factor_a"]}, "version": "v3"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wrapper must contain only"):
        composer.load_vetted_v3_family_catalog(malformed)


def test_dry_run_writes_nothing_and_execute_writes_only_a_fresh_child(tmp_path: Path) -> None:
    vetted_path = _write_vetted(tmp_path / "vetted_v3.json")
    scratch_root = tmp_path / "research_scratch"
    scratch_root.mkdir()
    plan = composer.build_composition_plan(
        vetted_v3_catalog=vetted_path,
        module_names=[INCREMENTAL_MODULE],
    )

    dry_run = composer.write_composition(
        plan,
        output_name="combined_20260803",
        execute=False,
        scratch_root=scratch_root,
    )
    assert dry_run.executed is False
    assert dry_run.output_dir == scratch_root / "combined_20260803"
    assert not dry_run.output_dir.exists()
    assert list(scratch_root.iterdir()) == []

    executed = composer.write_composition(
        plan,
        output_name="combined_20260803",
        execute=True,
        scratch_root=scratch_root,
    )
    assert executed.executed is True
    assert executed.output_dir.parent == scratch_root
    assert sorted(path.name for path in executed.output_dir.iterdir()) == [
        "catalog_manifest.json",
        "family_catalog.json",
    ]
    assert json.loads(executed.catalog_path.read_text(encoding="utf-8")) == plan.catalog
    manifest = json.loads(executed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["combined"]["family_count"] == plan.family_count
    assert manifest["combined"]["signal_count"] == plan.signal_count

    with pytest.raises(FileExistsError, match="already exists"):
        composer.write_composition(
            plan,
            output_name="combined_20260803",
            execute=True,
            scratch_root=scratch_root,
        )


def test_duplicate_signal_and_family_ambiguities_are_rejected(tmp_path: Path) -> None:
    signal_collision = _write_vetted(
        tmp_path / "signal_collision.json",
        {"v3_subset": ["dret_last_return"]},
    )
    with pytest.raises(ValueError, match="signal ambiguity"):
        composer.build_composition_plan(
            vetted_v3_catalog=signal_collision,
            module_names=[EXPANSION_MODULE],
        )

    family_collision = _write_vetted(
        tmp_path / "family_collision.json",
        {"daily_return_regime_transition": ["v3_placeholder"]},
    )
    with pytest.raises(ValueError, match="family ambiguity"):
        composer.build_composition_plan(
            vetted_v3_catalog=family_collision,
            module_names=[EXPANSION_MODULE],
        )


def test_invalid_module_prefix_and_invalid_vetted_signal_mapping_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must start"):
        composer.load_research_module_catalog("json")

    duplicate_signal = _write_vetted(
        tmp_path / "invalid_vetted.json",
        {"one": ["same_signal"], "two": ["same_signal"]},
    )
    with pytest.raises(ValueError, match="signal ambiguity"):
        composer.load_vetted_v3_family_catalog(duplicate_signal)


def test_output_name_cannot_escape_scratch_root(tmp_path: Path) -> None:
    vetted_path = _write_vetted(tmp_path / "vetted_v3.json")
    scratch_root = tmp_path / "research_scratch"
    scratch_root.mkdir()
    plan = composer.build_composition_plan(
        vetted_v3_catalog=vetted_path,
        module_names=[INCREMENTAL_MODULE],
    )

    with pytest.raises(ValueError, match="invalid fresh scratch child name"):
        composer.write_composition(
            plan,
            output_name="../outside",
            execute=False,
            scratch_root=scratch_root,
        )
