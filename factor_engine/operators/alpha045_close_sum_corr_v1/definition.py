"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha045_close_sum_corr_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha045CloseSumCorrV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha045_close_sum_corr_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha045_close_sum_corr_v1.py'
IMPLEMENTATION_SHA256 = '1e34955acb9f9fd6e7eda3cd3fd74cac0183515fa99989935bb28e634a65c8a0'
CONTRACT_PATH = 'operators/alpha045_close_sum_corr_v1/contract.json'


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
