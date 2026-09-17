from __future__ import annotations

from collections import Counter

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_state_gated_microstructure_healthy_catalog_v1 as healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_intraday_state_gated_microstructure_v1 as source,
)


EXPECTED_HEALTHY_SIGNALS = (
    "isgm_duration_quote_revision_concentration",
    "isgm_duration_quote_revision_late_tilt",
    "isgm_duration_bid_ask_revision_clock_gap",
    "isgm_credit_wide_spread_longest_spell_share",
    "isgm_credit_wide_spread_entry_clock",
    "isgm_credit_wide_spread_recovery_share",
    "isgm_stockvol_trade_quote_clock_center_gap",
    "isgm_stockvol_trade_quote_profile_cosine",
    "isgm_stockvol_silent_trade_share",
)
EXPECTED_EXCLUDED_SIGNALS = {
    "isgm_floorpremium_passive_depth_refresh_share",
    "isgm_floorpremium_passive_depth_side_balance",
    "isgm_floorpremium_passive_depth_late_tilt",
}
EXPECTED_FAMILY_COUNTS = {
    "duration_gated_quote_revision_calendar": 3,
    "credit_gated_spread_spell_topology": 3,
    "stockvol_gated_trade_quote_clock_decoupling": 3,
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
        is source.FactorMiningIntradayStateGatedMicrostructureV1
    )


def test_healthy_catalogue_rejects_duplicate_or_drifted_source_entries() -> None:
    source_entries = tuple(source.factor_mining_catalog())

    with pytest.raises(ValueError, match="duplicate signal names"):
        healthy._validated_source_entries(
            (source_entries[0], source_entries[0], *source_entries[2:])
        )
    with pytest.raises(ValueError, match="signal contract changed"):
        healthy._validated_source_entries((*source_entries[1:], source_entries[0]))
