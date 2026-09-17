from __future__ import annotations

from collections import Counter

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_rank_lattice_healthy_catalog_v1 as healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_rank_lattice_v1 as source,
)


EXPECTED_SIGNALS = (
    "csl_option_moneyness_stockvol_lattice",
    "csl_option_premium_duration_lattice",
    "csl_option_moneyness_premium_vol_wedge",
    "csl_liquidity_turnover_size_lattice",
    "csl_liquidity_reallocation_stockvol_lattice",
    "csl_liquidity_trade_size_flow_wedge",
    "csl_barrier_call_put_curvature_lattice",
    "csl_barrier_trigger_premium_wedge",
    "csl_barrier_call_put_moneyness_skew",
)
EXPECTED_FAMILY_COUNTS = {
    "cross_sectional_optionality_rank_lattice": 3,
    "cross_sectional_liquidity_capacity_rank_lattice": 3,
    "cross_sectional_barrier_geometry_rank_lattice": 3,
}
EXPECTED_EXCLUDED_FLOOR_SIGNALS = {
    "csl_floor_debtgap_yield_lattice",
    "csl_floor_redemption_distance_lattice",
    "csl_floor_credit_redemption_wedge",
}


def test_healthy_wrapper_exposes_exact_nine_audited_non_floor_entries() -> None:
    entries = healthy.cross_sectional_rank_lattice_healthy_catalog()
    source_by_signal = {
        entry.signal: entry for entry in source.cross_sectional_rank_lattice_catalog()
    }

    assert healthy.factor_mining_catalog() == entries
    assert tuple(entry.signal for entry in entries) == EXPECTED_SIGNALS
    assert len(entries) == 9
    assert Counter(entry.family for entry in entries) == EXPECTED_FAMILY_COUNTS
    assert set(healthy.HEALTHY_FAMILIES) == set(EXPECTED_FAMILY_COUNTS)
    assert set(healthy.HEALTHY_SIGNALS) == set(EXPECTED_SIGNALS)
    assert set(healthy.EXCLUDED_FLOOR_SIGNALS) == EXPECTED_EXCLUDED_FLOOR_SIGNALS
    assert set(EXPECTED_SIGNALS).isdisjoint(EXPECTED_EXCLUDED_FLOOR_SIGNALS)
    assert {entry.signal for entry in source_by_signal.values()} - set(EXPECTED_SIGNALS) == (
        EXPECTED_EXCLUDED_FLOOR_SIGNALS
    )
    assert all(entry is source_by_signal[entry.signal] for entry in entries)


def test_healthy_wrapper_import_registers_only_the_shared_source_kernel() -> None:
    assert healthy.SOURCE_MODULE_NAME == source.__name__
    assert healthy.KERNEL_NAME == source.KERNEL_NAME
    assert FactorRegistry.get(healthy.KERNEL_NAME) is source.FactorMiningCrossSectionalRankLatticeV1
    assert {entry.kernel for entry in healthy.factor_mining_catalog()} == {healthy.KERNEL_NAME}


def test_healthy_wrapper_rejects_duplicate_source_signal_fail_closed() -> None:
    source_entries = source.cross_sectional_rank_lattice_catalog()

    with pytest.raises(ValueError, match="duplicate signal: csl_option_moneyness_stockvol_lattice"):
        healthy._select_audited_entries((*source_entries, source_entries[0]))
