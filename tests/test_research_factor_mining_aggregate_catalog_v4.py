from __future__ import annotations

from collections import Counter

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v4 as aggregate,
)


def test_v4_locks_v3_prefix_and_excludes_insufficiently_evidenced_sources() -> None:
    assert aggregate_v3.CATALOG_VERSION == aggregate._REQUIRED_V3_VERSION
    assert aggregate.SOURCE_MODULES[: len(aggregate_v3.SOURCE_MODULES)] == (
        aggregate_v3.SOURCE_MODULES
    )
    modules = tuple(
        module.__name__.rsplit(".", 1)[-1] for _, module in aggregate.SOURCE_MODULES
    )
    assert len(modules) == 23
    assert (
        "research_factor_mining_cross_sectional_rank_lattice_healthy_catalog_v1"
        not in modules
    )
    assert "research_factor_mining_cross_asset_event_clock_healthy_v1" not in modules
    assert aggregate.SOURCE_MODULE_NAMES == tuple(
        f"cbond_on.domain.factors.operators.{module}" for module in modules
    )
    assert aggregate._EXCLUDED_UNTIL_FURTHER_EVIDENCE == (
        "cbond_on.domain.factors.operators."
        "research_factor_mining_cross_sectional_rank_lattice_healthy_catalog_v1",
    )


def test_v4_preserves_source_entries_and_rejects_duplicate_families_or_signals() -> (
    None
):
    expected = tuple(
        entry
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    entries = aggregate.factor_mining_catalog()

    assert entries == expected
    assert all(left is right for left, right in zip(entries, expected, strict=True))
    assert len(entries) == 193
    assert len({entry.signal for entry in entries}) == 193
    assert len({entry.family for entry in entries}) == 67
    expected_kernels = Counter(
        entry.kernel
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    assert Counter(entry.kernel for entry in entries) == expected_kernels
    for kernel in expected_kernels:
        assert FactorRegistry.get(kernel) is not None
