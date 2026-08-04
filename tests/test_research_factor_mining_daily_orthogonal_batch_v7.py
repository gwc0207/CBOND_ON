from __future__ import annotations

from collections import Counter

from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v6 as batch_v6,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v7 as batch_v7,
)


def test_catalogue_preserves_v6_and_adds_observable_seasoning_family() -> None:
    entries = batch_v7.factor_mining_catalog()

    assert len(entries) == 23
    assert len({entry.signal for entry in entries}) == 23
    assert Counter(entry.family for entry in entries) == {
        "prior_asymmetric_equity_beta": 2,
        "prior_bond_stock_copula_tail_dependence": 1,
        "prior_bond_stock_cross_sectional_rank_concordance": 3,
        "prior_bond_stock_return_flow_information_dependence": 1,
        "prior_capacity_normalized_return_rank_coupling": 1,
        "prior_daily_ohlc_wick_path_asymmetry": 2,
        "prior_joint_return_liquidity_state_topology": 1,
        "prior_liquidity_channel_composition": 5,
        "prior_observable_market_seasoning": 2,
        "prior_relative_return_flow_rank_coupling": 2,
        "prior_relative_return_flow_tail_contradiction": 1,
        "prior_return_liquidity_information_dependence": 2,
    }


def test_catalogue_source_order_keeps_v6_prefix_immutable() -> None:
    assert batch_v7.SOURCE_MODULE_NAMES[:-1] == batch_v6.SOURCE_MODULE_NAMES
    assert batch_v7.SOURCE_MODULE_NAMES[-1] == (
        "cbond_on.domain.factors.defs."
        "research_factor_mining_daily_observable_seasoning_v1"
    )
    assert len(batch_v7.SOURCE_MODULE_NAMES) == 11
