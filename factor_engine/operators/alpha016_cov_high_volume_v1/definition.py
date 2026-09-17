"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha016_cov_high_volume_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha016CovHighVolumeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha016_cov_high_volume_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha016_cov_high_volume_v1.py'
IMPLEMENTATION_SHA256 = 'e2f4624f29c5f6d120413a1e75c16eb88c47fb082ed14d1aef9f14f677bd59b0'
CONTRACT_PATH = 'operators/alpha016_cov_high_volume_v1/contract.json'


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
