"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_range_norm_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430RangeNormVolumeV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_range_norm_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_range_norm_volume_v1.py'
IMPLEMENTATION_SHA256 = '035054b57c36d424ca4601410c3c098b5eecb750bb8e1f398da5c0006b5427eb'
CONTRACT_PATH = 'operators/t1430_range_norm_volume_v1/contract.json'


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
