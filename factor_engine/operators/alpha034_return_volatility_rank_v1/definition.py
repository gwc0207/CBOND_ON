"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha034_return_volatility_rank_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha034ReturnVolatilityRankV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha034_return_volatility_rank_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha034_return_volatility_rank_v1.py'
IMPLEMENTATION_SHA256 = '798dfe1b511dd9c3a8cc419279cefe56f7f68c37b53fdf32fcaecca84c4bc717'
CONTRACT_PATH = 'operators/alpha034_return_volatility_rank_v1/contract.json'


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
