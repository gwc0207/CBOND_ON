"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_trigger_state_response_v1'
OPERATOR_VERSION = '20260803_intraday_trigger_state_response_v1'
OPERATOR_CLASS = 'FactorMiningIntradayTriggerStateResponseV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_trigger_state_response_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_trigger_state_response_v1.py'
IMPLEMENTATION_SHA256 = '29164ea2f2032ffd420777bc8758a6be7e662dc890db4dadd7b949c611fe3088'
CONTRACT_PATH = 'operators/factor_mining_intraday_trigger_state_response_v1/contract.json'


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
