"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha008_open_return_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha008OpenReturnMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha008_open_return_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha008_open_return_momentum_v1.py'
IMPLEMENTATION_SHA256 = '93c1946a26ff23c7436e0adf629ccee96271b147467b1ba843ddb8045d0ac01b'
CONTRACT_PATH = 'operators/alpha008_open_return_momentum_v1/contract.json'


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
