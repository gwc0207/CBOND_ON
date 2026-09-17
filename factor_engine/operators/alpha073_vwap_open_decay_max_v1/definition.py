"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha073_vwap_open_decay_max_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha073VwapOpenDecayMaxV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha073_vwap_open_decay_max_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha073_vwap_open_decay_max_v1.py'
IMPLEMENTATION_SHA256 = '220c4d061f138594da260a12086a3c9b7b5935c4905b2f3b286af427f0ad4605'
CONTRACT_PATH = 'operators/alpha073_vwap_open_decay_max_v1/contract.json'


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
