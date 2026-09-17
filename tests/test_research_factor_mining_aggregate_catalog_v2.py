from __future__ import annotations

from collections import Counter

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v2 as aggregate,
)


def test_v2_has_only_full_window_safe_sources_and_registered_kernels() -> None:
    modules = tuple(
        module.__name__.rsplit(".", 1)[-1] for _, module in aggregate.SOURCE_MODULES
    )

    assert len(modules) == 15
    assert "research_factor_mining_daily_twap_microsegments_v1" not in modules
    assert aggregate.SOURCE_MODULE_NAMES == tuple(
        f"cbond_on.domain.factors.operators.{module}" for module in modules
    )
    for _, module in aggregate.SOURCE_MODULES:
        registered = FactorRegistry.get(module.KERNEL_NAME)
        assert registered.__module__ == module.__name__


def test_v2_preserves_original_entries_with_no_signal_or_family_collision() -> None:
    expected = tuple(
        entry
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    entries = aggregate.factor_mining_catalog()

    assert entries == expected
    assert all(left is right for left, right in zip(entries, expected, strict=True))
    assert len(entries) == 139
    assert len({entry.signal for entry in entries}) == 139
    assert len({entry.family for entry in entries}) == 49
    assert Counter(entry.kernel for entry in entries) == {
        module.KERNEL_NAME: len(tuple(module.factor_mining_catalog()))
        for _, module in aggregate.SOURCE_MODULES
    }
