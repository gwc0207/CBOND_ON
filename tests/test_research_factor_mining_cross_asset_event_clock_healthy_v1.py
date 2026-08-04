from __future__ import annotations

from collections import Counter

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.defs import research_factor_mining_cross_asset_event_clock_healthy_v1 as healthy
from cbond_on.domain.factors.defs import research_factor_mining_cross_asset_event_clock_v1 as source


EXPECTED_HEALTHY_SIGNALS = (
    "xca_event_burst_coactivity",
    "xca_stock_leads_bond_event_clock",
    "xca_signed_execution_agreement",
    "xca_stock_leads_bond_signed_execution",
    "xca_signed_execution_intensity_gap",
)


def test_healthy_wrapper_keeps_exact_audited_source_entries_and_excludes_lull() -> None:
    source_entries = source.factor_mining_catalog()
    entries = healthy.factor_mining_catalog()
    expected = tuple(entry for entry in source_entries if entry.signal in EXPECTED_HEALTHY_SIGNALS)

    assert healthy.HEALTHY_SIGNALS == EXPECTED_HEALTHY_SIGNALS
    assert healthy.EXCLUDED_SIGNALS == frozenset({"xca_lull_overlap_excess"})
    assert tuple(entry.signal for entry in entries) == EXPECTED_HEALTHY_SIGNALS
    assert entries == expected
    assert all(actual is original for actual, original in zip(entries, expected, strict=True))
    assert "xca_lull_overlap_excess" not in {entry.signal for entry in entries}
    assert Counter(entry.family for entry in entries) == {
        "cross_asset_event_clock_transmission": 2,
        "cross_asset_signed_execution_transmission": 3,
    }


def test_healthy_wrapper_uses_only_the_existing_registered_source_kernel() -> None:
    entries = healthy.factor_mining_catalog()

    assert healthy.SOURCE_MODULE == source.__name__
    assert healthy.SOURCE_KERNEL_NAME == source.KERNEL_NAME
    assert {entry.kernel for entry in entries} == {source.KERNEL_NAME}
    assert FactorRegistry.get(source.KERNEL_NAME) is source.FactorMiningCrossAssetEventClockV1


def test_healthy_wrapper_rejects_source_catalogue_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    source_entries = source.factor_mining_catalog()
    monkeypatch.setattr(source, "factor_mining_catalog", lambda: source_entries[:-1])

    with pytest.raises(ValueError, match="source catalogue drift"):
        healthy.factor_mining_catalog()
