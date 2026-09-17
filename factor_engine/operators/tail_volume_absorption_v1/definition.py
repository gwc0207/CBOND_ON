"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'tail_volume_absorption_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'TailVolumeAbsorptionV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.tail_volume_absorption_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/tail_volume_absorption_v1.py'
IMPLEMENTATION_SHA256 = 'fd033bc7e16999a5541ee801bdb7535a92e7f88d08131619c110775e6c1ce893'
CONTRACT_PATH = 'operators/tail_volume_absorption_v1/contract.json'


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
