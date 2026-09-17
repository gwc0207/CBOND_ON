"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha005_vwap_gap_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha005VwapGapV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha005_vwap_gap_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha005_vwap_gap_v1.py'
IMPLEMENTATION_SHA256 = '416e5d6a6aea5e9e3665a4514f77d08cdddcf8499750aabd822ebb0774e73908'
CONTRACT_PATH = 'operators/alpha005_vwap_gap_v1/contract.json'


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
