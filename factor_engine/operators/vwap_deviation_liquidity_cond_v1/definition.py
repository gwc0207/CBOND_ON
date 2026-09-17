"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'vwap_deviation_liquidity_cond_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'VwapDeviationLiquidityCondV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.vwap_deviation_liquidity_cond_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/vwap_deviation_liquidity_cond_v1.py'
IMPLEMENTATION_SHA256 = 'e3a410297c7301f9517a3239d87b395066a981e4eaa18bcad3f2b4a50984f769'
CONTRACT_PATH = 'operators/vwap_deviation_liquidity_cond_v1/contract.json'


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
