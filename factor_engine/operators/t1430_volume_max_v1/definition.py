"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volume_max_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolumeMaxV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volume_max_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volume_max_v1.py'
IMPLEMENTATION_SHA256 = '61e61c3662f54356d2d0d221e694a8ac8de3bb1bcae44a8887cc1fea636d3af0'
CONTRACT_PATH = 'operators/t1430_volume_max_v1/contract.json'


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
