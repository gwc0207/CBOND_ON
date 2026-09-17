"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha024_close_trend_filter_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha024CloseTrendFilterV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha024_close_trend_filter_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha024_close_trend_filter_v1.py'
IMPLEMENTATION_SHA256 = 'fcd144646b20270cb3da1d470bf7a3081838db7b59d4c076b9c0ea647cdc99b9'
CONTRACT_PATH = 'operators/alpha024_close_trend_filter_v1/contract.json'


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
