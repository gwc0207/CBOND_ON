"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha060_price_range_volume_scale_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha060PriceRangeVolumeScaleV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha060_price_range_volume_scale_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha060_price_range_volume_scale_v1.py'
IMPLEMENTATION_SHA256 = '04a89a12d1b9487589895ec00e5453f6ed22086c8cf0865b8a760eaa02362e95'
CONTRACT_PATH = 'operators/alpha060_price_range_volume_scale_v1/contract.json'


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
