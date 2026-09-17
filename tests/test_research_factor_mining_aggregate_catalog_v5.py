from __future__ import annotations

from collections import Counter

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v4 as aggregate_v4,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v5 as aggregate,
)


def test_v5_locks_the_v4_prefix_and_has_only_new_audited_sources() -> None:
    assert aggregate_v4.CATALOG_VERSION == aggregate._REQUIRED_V4_VERSION
    assert aggregate.SOURCE_MODULES[: len(aggregate_v4.SOURCE_MODULES)] == (
        aggregate_v4.SOURCE_MODULES
    )
    modules = tuple(
        module.__name__.rsplit(".", 1)[-1] for _, module in aggregate.SOURCE_MODULES
    )
    assert len(modules) == 25
    assert (
        "research_factor_mining_daily_credit_duration_term_supply_dynamics_v2"
        not in modules
    )
    assert (
        "research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1"
        in modules
    )
    assert "research_factor_mining_intraday_information_clock_geometry_v1" in modules
    assert aggregate.SOURCE_MODULE_NAMES == tuple(
        f"cbond_on.domain.factors.operators.{module}" for module in modules
    )


def test_v5_preserves_entries_and_has_no_signal_or_family_collision() -> None:
    expected = tuple(
        entry
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    entries = aggregate.factor_mining_catalog()

    assert entries == expected
    assert all(left is right for left, right in zip(entries, expected, strict=True))
    assert len(entries) == 214
    assert len({entry.signal for entry in entries}) == 214
    assert len({entry.family for entry in entries}) == 75
    expected_kernels = Counter(
        entry.kernel
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    assert Counter(entry.kernel for entry in entries) == expected_kernels
    for kernel in expected_kernels:
        assert FactorRegistry.get(kernel) is not None
