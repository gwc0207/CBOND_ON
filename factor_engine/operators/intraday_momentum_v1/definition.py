"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'intraday_momentum_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'IntradayMomentumV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.intraday_momentum_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/intraday_momentum_v1.py'
IMPLEMENTATION_SHA256 = '440d436d7d0309631be09fe2e8b6f42907c677446d0e38cd84e32626aef5d88b'
CONTRACT_PATH = 'operators/intraday_momentum_v1/contract.json'


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
