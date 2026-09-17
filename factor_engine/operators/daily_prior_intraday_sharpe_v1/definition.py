"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'daily_prior_intraday_sharpe_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'DailyPriorIntradaySharpeV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.daily_prior_intraday_sharpe_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/daily_prior_intraday_sharpe_v1.py'
IMPLEMENTATION_SHA256 = 'bd663d308b069ac7870347d8dc71bb0ab4f06409eaca07d24d245cb16cd912b1'
CONTRACT_PATH = 'operators/daily_prior_intraday_sharpe_v1/contract.json'


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
