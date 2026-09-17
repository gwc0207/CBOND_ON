"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha004_ts_rank_low_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha004TsRankLowV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha004_ts_rank_low_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha004_ts_rank_low_v1.py'
IMPLEMENTATION_SHA256 = 'f5266dbd18f74d06a6a4fa995c26fffa17e09d4810c9d85b0bdb06aabf51b075'
CONTRACT_PATH = 'operators/alpha004_ts_rank_low_v1/contract.json'


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
