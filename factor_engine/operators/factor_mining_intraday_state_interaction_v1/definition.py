"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_state_interaction_v1'
OPERATOR_VERSION = '20260803_intraday_state_interaction_v1'
OPERATOR_CLASS = 'FactorMiningIntradayStateInteractionV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_state_interaction_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_state_interaction_v1.py'
IMPLEMENTATION_SHA256 = '8be8a9bdc056bc401059a4333874f329b0ce7ad101ccb1a0132dccab9bbd9480'
CONTRACT_PATH = 'operators/factor_mining_intraday_state_interaction_v1/contract.json'


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
