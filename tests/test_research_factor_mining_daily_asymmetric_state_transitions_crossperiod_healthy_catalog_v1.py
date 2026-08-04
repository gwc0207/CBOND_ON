from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_asymmetric_state_transitions_crossperiod_healthy_catalog_v1 as crossperiod,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_asymmetric_state_transitions_healthy_catalog_v1 as upstream,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_asymmetric_state_transitions_v1 as source,
)


EXPECTED_SIGNALS = (
    "ydpt_yield_rise_return_beta60",
    "ydpt_yield_fall_return_beta60",
    "ydpt_yield_return_beta_asymmetry60",
    "dcrt_density_body_asymmetry60",
    "dcrt_density_range_asymmetry60",
    "dcrt_density_sign_agreement60",
)
EXPECTED_FAMILY_COUNTS = {
    "prior_yield_directional_price_pass_through": 3,
    "prior_duration_convexity_range_transition": 3,
}
EXPECTED_EXCLUDED_UPSTREAM_SIGNALS = {"svcr_stress_gap_reversal_rate60"}


def test_crossperiod_healthy_catalogue_reuses_exact_upstream_entries_and_kernel() -> (
    None
):
    entries = crossperiod.factor_mining_catalog()
    upstream_by_signal = {
        entry.signal: entry for entry in upstream.factor_mining_catalog()
    }

    assert crossperiod.factor_mining_catalog() == entries
    assert tuple(entry.signal for entry in entries) == EXPECTED_SIGNALS
    assert crossperiod.CROSSPERIOD_HEALTHY_SIGNALS == EXPECTED_SIGNALS
    assert len(entries) == 6
    assert Counter(entry.family for entry in entries) == EXPECTED_FAMILY_COUNTS
    assert set(crossperiod.CROSSPERIOD_HEALTHY_FAMILIES) == set(EXPECTED_FAMILY_COUNTS)
    assert (
        set(crossperiod.EXCLUDED_UPSTREAM_SIGNALS) == EXPECTED_EXCLUDED_UPSTREAM_SIGNALS
    )
    assert not {entry.signal for entry in entries} & EXPECTED_EXCLUDED_UPSTREAM_SIGNALS
    assert all(entry is upstream_by_signal[entry.signal] for entry in entries)
    assert crossperiod.UPSTREAM_MODULE_NAME == upstream.__name__
    assert crossperiod.SOURCE_MODULE_NAME == source.__name__
    assert crossperiod.KERNEL_NAME == source.KERNEL_NAME
    assert {entry.kernel for entry in entries} == {source.KERNEL_NAME}
    assert (
        FactorRegistry.get(crossperiod.KERNEL_NAME)
        is source.FactorMiningDailyAsymmetricStateTransitionsV1
    )


def test_crossperiod_healthy_catalogue_rejects_duplicate_or_drifted_upstream_entries() -> (
    None
):
    entries = tuple(upstream.factor_mining_catalog())

    with pytest.raises(ValueError, match="duplicate signal names"):
        crossperiod._validated_upstream_entries((entries[0], entries[0], *entries[2:]))
    with pytest.raises(ValueError, match="entry order or signal contract changed"):
        crossperiod._validated_upstream_entries((*entries[1:], entries[0]))
    with pytest.raises(ValueError, match="family contract changed"):
        crossperiod._validated_upstream_entries(
            (
                *entries[:5],
                replace(entries[5], family="unexpected"),
                entries[6],
            )
        )
    with pytest.raises(ValueError, match="unexpected kernel"):
        crossperiod._validated_upstream_entries(
            (
                *entries[:5],
                replace(entries[5], kernel="unexpected"),
                entries[6],
            )
        )
