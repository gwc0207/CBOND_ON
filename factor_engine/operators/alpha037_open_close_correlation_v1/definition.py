"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha037_open_close_correlation_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha037OpenCloseCorrelationV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha037_open_close_correlation_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha037_open_close_correlation_v1.py'
IMPLEMENTATION_SHA256 = '1e5a6895f7043eec084b2f72269b7a48c08868f82d5d34b3a20480d0631b34f8'
CONTRACT_PATH = 'operators/alpha037_open_close_correlation_v1/contract.json'


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
