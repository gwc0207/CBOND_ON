"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha062_vwap_open_rank_compare_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha062VwapOpenRankCompareV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha062_vwap_open_rank_compare_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha062_vwap_open_rank_compare_v1.py'
IMPLEMENTATION_SHA256 = 'a7726662af9d562ebcf3987ecbffe6c919f09c4bb0b81100ad41d9dc0e4e6984'
CONTRACT_PATH = 'operators/alpha062_vwap_open_rank_compare_v1/contract.json'


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
