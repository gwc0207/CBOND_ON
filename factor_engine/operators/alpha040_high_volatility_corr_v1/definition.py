"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha040_high_volatility_corr_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha040HighVolatilityCorrV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha040_high_volatility_corr_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha040_high_volatility_corr_v1.py'
IMPLEMENTATION_SHA256 = 'fc6c5df6be3aeb343164069c3e87df5a4a0e1cfcc2d9daf6e6cd61d5071fd6c6'
CONTRACT_PATH = 'operators/alpha040_high_volatility_corr_v1/contract.json'


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
