"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha039_volume_decay_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha039VolumeDecayMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha039_volume_decay_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha039_volume_decay_momentum_v1.py'
IMPLEMENTATION_SHA256 = 'ecd01f4302e325c8d6aab9c565cd00f2b85305940322d3425938bad37a6661a5'
CONTRACT_PATH = 'operators/alpha039_volume_decay_momentum_v1/contract.json'


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
