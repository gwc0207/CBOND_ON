"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'microprice_bias'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'MicropriceBiasFactor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.microprice_bias'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/microprice_bias.py'
IMPLEMENTATION_SHA256 = 'c474e7733ff09beaee05f13ffd588d437433f6e6493f5451bc065dded60f3d15'
CONTRACT_PATH = 'operators/microprice_bias/contract.json'


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
