"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_mid_return_30m_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430MidReturn30mV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_mid_return_30m_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_mid_return_30m_v1.py'
IMPLEMENTATION_SHA256 = 'a8b209078b0351d0d6919e9f76519f61b515c94f5a2982ad516b8be4888cfb05'
CONTRACT_PATH = 'operators/t1430_mid_return_30m_v1/contract.json'


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
