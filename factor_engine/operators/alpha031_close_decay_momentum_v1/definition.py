"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha031_close_decay_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha031CloseDecayMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha031_close_decay_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha031_close_decay_momentum_v1.py'
IMPLEMENTATION_SHA256 = '5b58790bd89fa152fd03b58577eb668dd704f508cb492de0fa9d853eb947f1b1'
CONTRACT_PATH = 'operators/alpha031_close_decay_momentum_v1/contract.json'


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
