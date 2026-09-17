"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha052_low_momentum_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha052LowMomentumVolumeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha052_low_momentum_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha052_low_momentum_volume_v1.py'
IMPLEMENTATION_SHA256 = '36b33b3abd2907911ac5b375886b4eaa765243a648ad082afe24b76ad3f75c5e'
CONTRACT_PATH = 'operators/alpha052_low_momentum_volume_v1/contract.json'


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
