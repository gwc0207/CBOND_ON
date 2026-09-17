"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'volume_price_trend_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VolumePriceTrendV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.volume_price_trend_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/volume_price_trend_v1.py'
IMPLEMENTATION_SHA256 = '1220f5ecaad2774309327bde783606b83c9f5f4ecd45317b464059ac01de95b4'
CONTRACT_PATH = 'operators/volume_price_trend_v1/contract.json'


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
