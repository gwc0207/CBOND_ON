from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import research_factor_mining_aggregate_catalog_v1 as aggregate


EXPECTED_SOURCE_MODULES = (
    "research_factor_mining_conditional_response_residual_v1",
    "research_factor_mining_intraday_temporal_shape_v1",
    "research_factor_mining_intraday_transmission_response_v1",
    "research_factor_mining_quote_geometry_microprice_v1",
    "research_factor_mining_intraday_lunch_reopen_dynamics_v1",
    "research_factor_mining_cross_asset_book_geometry_v1",
    "research_factor_mining_daily_twap_microsegments_v1",
    "research_factor_mining_daily_debt_floor_wedge_dynamics_v1",
    "research_factor_mining_daily_mark_barrier_dynamics_v1",
    "research_factor_mining_daily_interday_topology_v1",
    "research_factor_mining_intraday_execution_discreteness_v1",
    "research_factor_mining_intraday_state_gated_microstructure_v1",
    "research_factor_mining_limit_pressure_temporal_topology_v1",
    "research_factor_mining_intraday_conversion_parity_dynamics_v1",
    "research_factor_mining_underlying_cohort_intraday_rank_state_v1",
    "research_factor_mining_intraday_trigger_state_response_v1",
)


def _entry(*, family: str, signal: str) -> SimpleNamespace:
    return SimpleNamespace(
        family=family,
        signal=signal,
        kernel="synthetic_kernel",
        hypothesis="synthetic hypothesis",
    )


def test_aggregate_has_exact_sixteen_explicit_sources_and_registers_every_kernel() -> None:
    assert tuple(module.__name__.rsplit(".", 1)[-1] for _, module in aggregate.SOURCE_MODULES) == EXPECTED_SOURCE_MODULES
    assert aggregate.SOURCE_MODULE_NAMES == tuple(
        f"cbond_on.domain.factors.operators.{name}" for name in EXPECTED_SOURCE_MODULES
    )
    assert len(aggregate.SOURCE_MODULES) == 16

    for _, module in aggregate.SOURCE_MODULES:
        registered = FactorRegistry.get(module.KERNEL_NAME)
        assert registered.__module__ == module.__name__
        assert registered.__name__.startswith("FactorMining")


def test_aggregate_preserves_source_entries_in_order_with_unique_signals_and_families() -> None:
    expected = tuple(
        entry
        for _, module in aggregate.SOURCE_MODULES
        for entry in module.factor_mining_catalog()
    )
    entries = aggregate.factor_mining_catalog()

    assert entries == expected
    assert all(left is right for left, right in zip(entries, expected, strict=True))
    assert len(entries) == 151
    assert len({entry.signal for entry in entries}) == 151
    assert len({entry.family for entry in entries}) == 53
    assert Counter(entry.kernel for entry in entries) == {
        module.KERNEL_NAME: len(tuple(module.factor_mining_catalog()))
        for _, module in aggregate.SOURCE_MODULES
    }

    owners: dict[str, str] = {}
    for source, module in aggregate.SOURCE_MODULES:
        for entry in module.factor_mining_catalog():
            assert owners.setdefault(entry.family, source) == source


def test_aggregate_rejects_duplicate_signal_and_cross_source_family_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate signal"):
        aggregate._validate_source_catalogs(
            (
                ("one", (_entry(family="family_one", signal="same_signal"),)),
                ("two", (_entry(family="family_two", signal="same_signal"),)),
            )
        )
    with pytest.raises(ValueError, match="duplicate family across sources"):
        aggregate._validate_source_catalogs(
            (
                ("one", (_entry(family="same_family", signal="signal_one"),)),
                ("two", (_entry(family="same_family", signal="signal_two"),)),
            )
        )
    with pytest.raises(ValueError, match="ambiguous family spelling"):
        aggregate._validate_source_catalogs(
            (
                ("one", (_entry(family="Family", signal="signal_one"),)),
                ("two", (_entry(family="family", signal="signal_two"),)),
            )
        )
