"""Generated operator identity entry; executable source stays in domain/factors/operators."""

from __future__ import annotations

GENERATED_OPERATOR_DEFINITION = True
OPERATOR_ID = 'alpha038_close_rank_ratio_v1'
OPERATOR_VERSION = 'legacy-unversioned'
OPERATOR_CLASS = 'Alpha038CloseRankRatioV1Factor'
IMPLEMENTATION_MODULE = 'cbond_on.domain.factors.operators.alpha038_close_rank_ratio_v1'
IMPLEMENTATION_PATH = 'cbond_on/domain/factors/operators/alpha038_close_rank_ratio_v1.py'
IMPLEMENTATION_SHA256 = '4ca7e6efae145d48b7dbc8c1315f07f040eda6c8facd64f58991b5cf6c359290'
CONTRACT_PATH = 'operators/alpha038_close_rank_ratio_v1/contract.json'


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
