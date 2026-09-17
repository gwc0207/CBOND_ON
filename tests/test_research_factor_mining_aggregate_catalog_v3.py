from __future__ import annotations

from collections import Counter

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate,
)


def test_v3_has_only_smoke_healthy_sources_and_registered_kernels() -> None:
    modules = tuple(
        module.__name__.rsplit(".", 1)[-1] for _, module in aggregate.SOURCE_MODULES
    )

    assert len(modules) == 11
    assert "research_factor_mining_daily_twap_microsegments_v1" not in modules
    assert "research_factor_mining_daily_debt_floor_wedge_dynamics_v1" not in modules
    assert "research_factor_mining_limit_pressure_temporal_topology_v1" not in modules
    assert (
        "research_factor_mining_intraday_conversion_parity_dynamics_v1" not in modules
    )
    assert "research_factor_mining_intraday_trigger_state_response_v1" not in modules
    assert (
        "research_factor_mining_intraday_state_gated_microstructure_v1" not in modules
    )
    assert (
        "research_factor_mining_intraday_state_gated_microstructure_healthy_catalog_v1"
        in modules
    )
    assert aggregate.SOURCE_MODULE_NAMES == tuple(
        f"cbond_on.domain.factors.operators.{module}" for module in modules
    )
    for _, module in aggregate.SOURCE_MODULES:
        registered = FactorRegistry.get(module.KERNEL_NAME)
        if module.__name__.endswith("_healthy_catalog_v1"):
            assert (
                registered.__module__ == "cbond_on.domain.factors.operators."
                "research_factor_mining_intraday_state_gated_microstructure_v1"
            )
        else:
            assert registered.__module__ == module.__name__


def test_v3_preserves_original_entries_with_no_signal_or_family_collision() -> None:
    expected = tuple(
        entry
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    entries = aggregate.factor_mining_catalog()

    assert entries == expected
    assert all(left is right for left, right in zip(entries, expected, strict=True))
    assert len(entries) == 97
    assert len({entry.signal for entry in entries}) == 97
    assert len({entry.family for entry in entries}) == 35
    assert Counter(entry.kernel for entry in entries) == {
        module.KERNEL_NAME: len(tuple(module.factor_mining_catalog()))
        for _, module in aggregate.SOURCE_MODULES
    }
