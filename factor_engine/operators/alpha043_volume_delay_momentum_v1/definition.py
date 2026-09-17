"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha043_volume_delay_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha043VolumeDelayMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha043_volume_delay_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha043_volume_delay_momentum_v1.py'
IMPLEMENTATION_SHA256 = 'aa43103669b345243a1a6d6f5bbf15c05c55720a0960f050514142af79bc1202'
CONTRACT_PATH = 'operators/alpha043_volume_delay_momentum_v1/contract.json'


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
