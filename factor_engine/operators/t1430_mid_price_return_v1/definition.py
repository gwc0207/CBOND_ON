"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_mid_price_return_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430MidPriceReturnV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_mid_price_return_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_mid_price_return_v1.py'
IMPLEMENTATION_SHA256 = '58fe4b11da438ed4500aa9f715b61c2bb368beacf1da9587b137cd76ea8242fc'
CONTRACT_PATH = 'operators/t1430_mid_price_return_v1/contract.json'


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
