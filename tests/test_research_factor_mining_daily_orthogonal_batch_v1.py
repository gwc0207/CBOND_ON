from __future__ import annotations

from collections import Counter

from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v1 as batch,
)


def test_catalogue_has_expected_unique_research_family_structure() -> None:
    entries = batch.factor_mining_catalog()

    assert len(entries) == 10
    assert len({entry.signal for entry in entries}) == 10
    assert Counter(entry.family for entry in entries) == {
        "prior_asymmetric_equity_beta": 2,
        "prior_daily_ohlc_wick_path_asymmetry": 2,
        "prior_joint_return_liquidity_state_topology": 1,
        "prior_relative_return_flow_rank_coupling": 2,
        "prior_relative_return_flow_tail_contradiction": 1,
        "prior_return_liquidity_information_dependence": 2,
    }


def test_catalogue_declares_exact_member_modules() -> None:
    assert batch.SOURCE_MODULE_NAMES == (
        "cbond_on.domain.factors.operators.research_factor_mining_daily_asymmetric_equity_beta_v1",
        "cbond_on.domain.factors.operators.research_factor_mining_daily_return_liquidity_topology_v1",
        "cbond_on.domain.factors.operators.research_factor_mining_daily_relative_rank_flow_coupling_v2",
        "cbond_on.domain.factors.operators.research_factor_mining_daily_relative_rank_tail_contradiction_v1",
        "cbond_on.domain.factors.operators.research_factor_mining_daily_ohlc_wick_path_asymmetry_v1",
    )
