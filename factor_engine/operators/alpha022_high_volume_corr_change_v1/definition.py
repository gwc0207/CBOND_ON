"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha022_high_volume_corr_change_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha022HighVolumeCorrChangeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha022_high_volume_corr_change_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha022_high_volume_corr_change_v1.py'
IMPLEMENTATION_SHA256 = 'e24335261e5d894acce3f920e6c08ff7593590091d575c7407c5aab32bdc0ac0'
CONTRACT_PATH = 'operators/alpha022_high_volume_corr_change_v1/contract.json'


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
