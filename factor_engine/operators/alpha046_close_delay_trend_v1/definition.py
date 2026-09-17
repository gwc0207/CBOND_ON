"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha046_close_delay_trend_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha046CloseDelayTrendV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha046_close_delay_trend_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha046_close_delay_trend_v1.py'
IMPLEMENTATION_SHA256 = 'acb86127d3be93fc129c0f66ee0fdee7bb46b6d9b2f443a95595e8748930a66e'
CONTRACT_PATH = 'operators/alpha046_close_delay_trend_v1/contract.json'


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
