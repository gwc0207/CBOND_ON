"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha007_volume_breakout_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha007VolumeBreakoutV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha007_volume_breakout_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha007_volume_breakout_v1.py'
IMPLEMENTATION_SHA256 = 'cf3845c3c66a3bd02d30d2a4e14078edd5f9f1c34fc8c7e360d51bbf72694089'
CONTRACT_PATH = 'operators/alpha007_volume_breakout_v1/contract.json'


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
