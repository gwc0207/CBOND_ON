"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_daily_relative_rank_flow_coupling_v2'
OPERATOR_VERSION = '20260803_daily_relative_rank_flow_coupling_v2'
OPERATOR_CLASS = 'FactorMiningDailyRelativeRankFlowCouplingV2'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_daily_relative_rank_flow_coupling_v2'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_daily_relative_rank_flow_coupling_v2.py'
IMPLEMENTATION_SHA256 = '863e1523876b6af00f81f527e251ce3ab872da1863efe95f721a29397cafadfe'
CONTRACT_PATH = 'operators/factor_mining_daily_relative_rank_flow_coupling_v2/contract.json'


def definition_payload() -> dict[str, object]:
    """Return immutable identity metadata without importing the runtime implementation."""

    return {
        'operator_id': OPERATOR_ID,
        'operator_version': OPERATOR_VERSION,
        'operator_class': OPERATOR_CLASS,
        'implementation_module': IMPLEMENTATION_MODULE,
        'implementation_path': IMPLEMENTATION_PATH,
        'implementation_sha256': IMPLEMENTATION_SHA256,
        'contract_path': CONTRACT_PATH,
    }
