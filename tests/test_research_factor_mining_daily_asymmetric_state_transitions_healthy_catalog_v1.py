from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_asymmetric_state_transitions_healthy_catalog_v1 as healthy,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_asymmetric_state_transitions_v1 as source,
)


EXPECTED_HEALTHY_SIGNALS = (
    "ydpt_yield_rise_return_beta60",
    "ydpt_yield_fall_return_beta60",
    "ydpt_yield_return_beta_asymmetry60",
    "dcrt_density_body_asymmetry60",
    "dcrt_density_range_asymmetry60",
    "dcrt_density_sign_agreement60",
    "svcr_stress_gap_reversal_rate60",
)
EXPECTED_EXCLUDED_SIGNALS = {
    "svcr_relief_gap_fade_rate60",
    "svcr_stress_relief_reversal_spread60",
}
EXPECTED_FAMILY_COUNTS = {
    "prior_yield_directional_price_pass_through": 3,
    "prior_duration_convexity_range_transition": 3,
    "prior_stockvol_candle_reversal_state": 1,
}


def test_healthy_catalogue_reuses_exact_source_entries_and_kernel() -> None:
    entries = healthy.factor_mining_catalog()
    source_by_signal = {entry.signal: entry for entry in source.factor_mining_catalog()}

    assert healthy.factor_mining_catalog() == entries
    assert tuple(entry.signal for entry in entries) == EXPECTED_HEALTHY_SIGNALS
    assert healthy.HEALTHY_SIGNALS == EXPECTED_HEALTHY_SIGNALS
    assert len(entries) == 7
    assert Counter(entry.family for entry in entries) == EXPECTED_FAMILY_COUNTS
    assert set(healthy.HEALTHY_FAMILIES) == set(EXPECTED_FAMILY_COUNTS)
    assert set(healthy.EXCLUDED_SIGNALS) == EXPECTED_EXCLUDED_SIGNALS
    assert not {entry.signal for entry in entries} & EXPECTED_EXCLUDED_SIGNALS
    assert {entry.kernel for entry in entries} == {source.KERNEL_NAME}
    assert all(entry is source_by_signal[entry.signal] for entry in entries)
    assert healthy.SOURCE_MODULE_NAME == source.__name__
    assert healthy.KERNEL_NAME == source.KERNEL_NAME
    assert (
        FactorRegistry.get(healthy.KERNEL_NAME)
        is source.FactorMiningDailyAsymmetricStateTransitionsV1
    )


def test_healthy_catalogue_rejects_duplicate_or_drifted_source_entries() -> None:
    source_entries = tuple(source.factor_mining_catalog())

    with pytest.raises(ValueError, match="duplicate signal names"):
        healthy._validated_source_entries(
            (source_entries[0], source_entries[0], *source_entries[2:])
        )
    with pytest.raises(ValueError, match="signal contract changed"):
        healthy._validated_source_entries((*source_entries[1:], source_entries[0]))
    with pytest.raises(ValueError, match="family contract changed"):
        healthy._validated_source_entries(
            (
                *source_entries[:6],
                replace(source_entries[6], family="unexpected"),
                *source_entries[7:],
            )
        )
    with pytest.raises(ValueError, match="unexpected kernel"):
        healthy._validated_source_entries(
            (
                *source_entries[:6],
                replace(source_entries[6], kernel="unexpected"),
                *source_entries[7:],
            )
        )
