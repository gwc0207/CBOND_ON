"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'factor_mining_intraday_conversion_parity_dynamics_v1'
OPERATOR_VERSION = '20260803_intraday_conversion_parity_dynamics_v1'
OPERATOR_CLASS = 'FactorMiningIntradayConversionParityDynamicsV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.research_factor_mining_intraday_conversion_parity_dynamics_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/research_factor_mining_intraday_conversion_parity_dynamics_v1.py'
IMPLEMENTATION_SHA256 = 'f16673c23bb06b6efd8c2a7f23afa821bd16865e7b9784a7d5dd59fe7e5ac685'
CONTRACT_PATH = 'operators/factor_mining_intraday_conversion_parity_dynamics_v1/contract.json'


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
