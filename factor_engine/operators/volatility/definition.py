"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'volatility'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VolatilityFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.volatility'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/volatility.py'
IMPLEMENTATION_SHA256 = 'a9b534652b072e4e745420cde8f2c16c4038c145455f280d93a8df656ec089e6'
CONTRACT_PATH = 'operators/volatility/contract.json'


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
