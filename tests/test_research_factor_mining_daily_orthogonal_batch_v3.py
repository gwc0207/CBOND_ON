from __future__ import annotations

from collections import Counter

from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v2 as batch_v2,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v3 as batch_v3,
)


def test_catalogue_preserves_v2_and_adds_only_copula_tail_family() -> None:
    entries = batch_v3.factor_mining_catalog()

    assert len(entries) == 12
    assert len({entry.signal for entry in entries}) == 12
    assert Counter(entry.family for entry in entries) == {
        "prior_asymmetric_equity_beta": 2,
        "prior_bond_stock_copula_tail_dependence": 1,
        "prior_capacity_normalized_return_rank_coupling": 1,
        "prior_daily_ohlc_wick_path_asymmetry": 2,
        "prior_joint_return_liquidity_state_topology": 1,
        "prior_relative_return_flow_rank_coupling": 2,
        "prior_relative_return_flow_tail_contradiction": 1,
        "prior_return_liquidity_information_dependence": 2,
    }


def test_catalogue_source_order_keeps_v2_prefix_immutable() -> None:
    assert batch_v3.SOURCE_MODULE_NAMES[:-1] == batch_v2.SOURCE_MODULE_NAMES
    assert batch_v3.SOURCE_MODULE_NAMES[-1] == (
        "cbond_on.domain.factors.operators."
        "research_factor_mining_daily_bond_stock_copula_tail_dependence_healthy_catalog_v1"
    )
    assert len(batch_v3.SOURCE_MODULE_NAMES) == 7
