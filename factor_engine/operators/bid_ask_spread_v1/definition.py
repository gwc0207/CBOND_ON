"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'bid_ask_spread_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'BidAskSpreadV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.bid_ask_spread_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/bid_ask_spread_v1.py'
IMPLEMENTATION_SHA256 = 'f45c02f8679643b5941adc326b432dd5b7595b3c527a319272c12d085c215094'
CONTRACT_PATH = 'operators/bid_ask_spread_v1/contract.json'


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
