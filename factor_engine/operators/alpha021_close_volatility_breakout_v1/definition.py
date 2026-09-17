"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha021_close_volatility_breakout_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha021CloseVolatilityBreakoutV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha021_close_volatility_breakout_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha021_close_volatility_breakout_v1.py'
IMPLEMENTATION_SHA256 = '5033bbe93ddb2e902ddf588ab7ae6a986ad55e8d7cb1e8eba81742322a4d2590'
CONTRACT_PATH = 'operators/alpha021_close_volatility_breakout_v1/contract.json'


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
