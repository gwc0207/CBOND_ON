from __future__ import annotations

from collections import Counter

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_microstructure_neighborhood_healthy_catalog_v1 as healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_microstructure_neighborhood_v1 as source,
)


EXPECTED_HEALTHY_SIGNALS = (
    "csn_pql_retention_neighbor_gap",
    "csn_pql_refill_neighbor_gap",
    "csn_pql_churn_neighbor_gap",
    "csn_taq_buy_initiation_neighbor_gap",
    "csn_taq_sell_initiation_neighbor_gap",
    "csn_taq_imbalance_neighbor_gap",
    "csn_dlc_bid_coherence_neighbor_gap",
    "csn_dlc_ask_coherence_neighbor_gap",
    "csn_dlc_crosslag_neighbor_gap",
)
EXPECTED_EXCLUDED_SIGNALS = {
    "csn_lql_terminal_lock_neighbor_gap",
    "csn_lql_occupancy_neighbor_gap",
    "csn_lql_transition_neighbor_gap",
}
EXPECTED_FAMILY_COUNTS = {
    "csn_passive_queue_local_dislocation": 3,
    "csn_quote_initiation_local_dislocation": 3,
    "csn_depth_cascade_local_dislocation": 3,
}


def test_healthy_catalogue_reuses_exact_nonempty_source_entries_and_kernel() -> None:
    entries = healthy.factor_mining_catalog()
    source_by_signal = {entry.signal: entry for entry in source.factor_mining_catalog()}

    assert healthy.factor_mining_catalog() == entries
    assert tuple(entry.signal for entry in entries) == EXPECTED_HEALTHY_SIGNALS
    assert healthy.HEALTHY_SIGNALS == EXPECTED_HEALTHY_SIGNALS
    assert len(entries) == 9
    assert Counter(entry.family for entry in entries) == EXPECTED_FAMILY_COUNTS
    assert set(healthy.HEALTHY_FAMILIES) == set(EXPECTED_FAMILY_COUNTS)
    assert set(healthy.EXCLUDED_SIGNALS) == EXPECTED_EXCLUDED_SIGNALS
    assert healthy.EXCLUDED_FAMILY not in {entry.family for entry in entries}
    assert not {entry.signal for entry in entries} & EXPECTED_EXCLUDED_SIGNALS
    assert {entry.kernel for entry in entries} == {source.KERNEL_NAME}
    assert all(entry is source_by_signal[entry.signal] for entry in entries)
    assert healthy.SOURCE_MODULE_NAME == source.__name__
    assert healthy.KERNEL_NAME == source.KERNEL_NAME
    assert (
        FactorRegistry.get(healthy.KERNEL_NAME)
        is source.FactorMiningCrossSectionalMicrostructureNeighborhoodV1
    )


def test_healthy_catalogue_rejects_duplicate_or_drifted_source_entries() -> None:
    source_entries = tuple(source.factor_mining_catalog())

    with pytest.raises(ValueError, match="duplicate signal names"):
        healthy._validated_source_entries(
            (source_entries[0], source_entries[0], *source_entries[2:])
        )
    with pytest.raises(ValueError, match="signal contract changed"):
        healthy._validated_source_entries((*source_entries[1:], source_entries[0]))
