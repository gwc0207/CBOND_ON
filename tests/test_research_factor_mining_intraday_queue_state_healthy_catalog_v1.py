from __future__ import annotations

from collections import Counter

import pytest

from cbond_on.core.registry import FactorRegistry
from cbond_on.domain.factors.operators import research_factor_mining_intraday_queue_state_healthy_catalog_v1 as healthy
from cbond_on.domain.factors.operators import research_factor_mining_intraday_queue_state_v1 as queue


def test_healthy_queue_state_catalogue_has_the_exact_nine_non_lock_entries_and_registered_kernel() -> None:
    entries = healthy.factor_mining_catalog()

    assert tuple(entry.signal for entry in entries) == healthy.HEALTHY_SIGNALS
    assert Counter(entry.family for entry in entries) == {
        "passive_queue_lifecycle_after_trade": 3,
        "trade_at_quote_initiation_state": 3,
        "depth_layer_update_cascade": 3,
    }
    assert len(entries) == 9
    assert len({entry.signal for entry in entries}) == 9
    assert not {entry.signal for entry in entries} & set(healthy.EXCLUDED_SIGNALS)
    assert healthy.EXCLUDED_FAMILY not in {entry.family for entry in entries}
    assert {entry.kernel for entry in entries} == {queue.KERNEL_NAME}
    assert FactorRegistry.get(queue.KERNEL_NAME) is queue.FactorMiningIntradayQueueStateV1

    source_by_signal = {entry.signal: entry for entry in queue.factor_mining_catalog()}
    assert all(entry is source_by_signal[entry.signal] for entry in entries)


def test_healthy_queue_state_catalogue_rejects_duplicate_or_drifted_source_entries() -> None:
    source_entries = tuple(queue.factor_mining_catalog())

    with pytest.raises(ValueError, match="duplicate signal names"):
        healthy._validated_source_entries((source_entries[0], source_entries[0], *source_entries[2:]))
    with pytest.raises(ValueError, match="signal contract changed"):
        healthy._validated_source_entries((*source_entries[1:], source_entries[0]))
