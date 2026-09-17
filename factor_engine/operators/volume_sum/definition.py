"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'volume_sum'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VolumeSumFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.volume_sum'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/volume_sum.py'
IMPLEMENTATION_SHA256 = 'da35c90620bd14003682808ce04a24db10da8b456bec186fbcebb2ffe4cca40a'
CONTRACT_PATH = 'operators/volume_sum/contract.json'


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
