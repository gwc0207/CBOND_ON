"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha006_corr_open_volume_neg_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha006CorrOpenVolumeNegV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha006_corr_open_volume_neg_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha006_corr_open_volume_neg_v1.py'
IMPLEMENTATION_SHA256 = 'f428c5774b7d5365f1ce6a4732a8cdc6bc7c34920bf43786b28f9cd4bc99080c'
CONTRACT_PATH = 'operators/alpha006_corr_open_volume_neg_v1/contract.json'


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
