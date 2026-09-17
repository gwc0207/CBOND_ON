"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_microprice_liquidity_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430MicropriceLiquidityV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_microprice_liquidity_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_microprice_liquidity_v1.py'
IMPLEMENTATION_SHA256 = 'a08a770a1a9e6fadfb98579035f108b36a4ba9c345daaf864a4b96adc871f8ab'
CONTRACT_PATH = 'operators/t1430_microprice_liquidity_v1/contract.json'


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
