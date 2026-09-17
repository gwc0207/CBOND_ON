"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_daily_twap_microstructure_v1'
OPERATOR_VERSION = '20260803_daily_twap_microstructure_v1'
OPERATOR_CLASS = 'FactorMiningDailyTwapMicrostructureV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_daily_twap_microstructure_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_daily_twap_microstructure_v1.py'
IMPLEMENTATION_SHA256 = 'b13092be8fa54cc323484744e7da05859b1a97eeb35a522db6ca7a0c4a30e0c2'
CONTRACT_PATH = 'operators/factor_mining_daily_twap_microstructure_v1/contract.json'


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
