"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 't1430_volatility_turnover_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'T1430VolatilityTurnoverV1'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.t1430_volatility_turnover_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/t1430_volatility_turnover_v1.py'
IMPLEMENTATION_SHA256 = '062ef30b163db964c85ff5c8b5f843a800fea4020324812119e07fac05908482'
CONTRACT_PATH = 'operators/t1430_volatility_turnover_v1/contract.json'


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
