from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_healthy_catalog_v1 as healthy,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_credit_duration_term_supply_dynamics_v2 as source,
)


EXPECTED_HEALTHY_SIGNALS = (
    "cdmr_yield_duration_delta_corr20",
    "cdmr_yield_convexity_delta_corr20",
    "cdmr_duration_convexity_delta_corr20",
    "tsrr2_supply_term_level_corr20",
    "tsrr2_duration_term_ratio_delta5",
    "cfar_yield_amount_capacity_delta_corr20",
    "cfar_yield_volume_capacity_delta_corr20",
    "catr_convexity_duration_delta_corr20",
    "catr_convexity_anchor_delta_corr20",
)
EXPECTED_EXCLUDED_SIGNALS = {
    "tsrr2_supply_term_standardized_oos20",
    "cfar_yield_price_supply_standardized_oos20",
    "catr_convexity_state_standardized_oos20",
}
EXPECTED_FAMILY_COUNTS = {
    "prior_credit_duration_convexity_motion_robust": 3,
    "prior_term_supply_roll_rebalancing_robust": 2,
    "prior_credit_flow_absorption_dynamics_robust": 2,
    "prior_convexity_anchor_transition_robust": 2,
}


def test_healthy_catalogue_reuses_exact_source_entries_order_and_kernel() -> None:
    entries = healthy.factor_mining_catalog()
    source_by_signal = {entry.signal: entry for entry in source.factor_mining_catalog()}

    assert healthy.factor_mining_catalog() == entries
    assert tuple(entry.signal for entry in entries) == EXPECTED_HEALTHY_SIGNALS
    assert healthy.HEALTHY_SIGNALS == EXPECTED_HEALTHY_SIGNALS
    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert Counter(entry.family for entry in entries) == EXPECTED_FAMILY_COUNTS
    assert tuple(healthy.HEALTHY_FAMILIES) == tuple(EXPECTED_FAMILY_COUNTS)
    assert set(healthy.EXCLUDED_SIGNALS) == EXPECTED_EXCLUDED_SIGNALS
    assert not {entry.signal for entry in entries} & EXPECTED_EXCLUDED_SIGNALS
    assert {entry.kernel for entry in entries} == {source.KERNEL_NAME}
    assert all(entry is source_by_signal[entry.signal] for entry in entries)
    assert healthy.SOURCE_MODULE_NAME == source.__name__
    assert healthy.SOURCE_CATALOG_VERSION == source.CATALOG_VERSION
    assert healthy.KERNEL_NAME == source.KERNEL_NAME
    assert (
        FactorRegistry.get(healthy.KERNEL_NAME)
        is source.FactorMiningDailyCreditDurationTermSupplyDynamicsV2
    )


def test_healthy_catalogue_rejects_source_order_entry_family_kernel_and_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                *source_entries[:3],
                replace(source_entries[3], family="unexpected"),
                *source_entries[4:],
            )
        )
    with pytest.raises(ValueError, match="unexpected kernel"):
        healthy._validated_source_entries(
            (
                *source_entries[:3],
                replace(source_entries[3], kernel="unexpected"),
                *source_entries[4:],
            )
        )

    monkeypatch.setattr(source, "CATALOG_VERSION", "unexpected")
    with pytest.raises(ValueError, match="catalogue version changed"):
        healthy._validated_source_entries(source_entries)
