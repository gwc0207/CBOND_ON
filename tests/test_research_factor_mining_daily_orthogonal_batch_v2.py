from __future__ import annotations

from collections import Counter

from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v2 as batch,
)


def test_catalogue_preserves_v1_and_adds_one_unique_capacity_family() -> None:
    entries = batch.factor_mining_catalog()

    assert len(entries) == 11
    assert len({entry.signal for entry in entries}) == 11
    assert Counter(entry.family for entry in entries) == {
        "prior_asymmetric_equity_beta": 2,
        "prior_capacity_normalized_return_rank_coupling": 1,
        "prior_daily_ohlc_wick_path_asymmetry": 2,
        "prior_joint_return_liquidity_state_topology": 1,
        "prior_relative_return_flow_rank_coupling": 2,
        "prior_relative_return_flow_tail_contradiction": 1,
        "prior_return_liquidity_information_dependence": 2,
    }


def test_catalogue_source_order_keeps_v1_prefix_immutable() -> None:
    assert batch.SOURCE_MODULE_NAMES[-1] == (
        "cbond_on.domain.factors.defs."
        "research_factor_mining_daily_capacity_rank_coupling_v1"
    )
    assert len(batch.SOURCE_MODULE_NAMES) == 6
