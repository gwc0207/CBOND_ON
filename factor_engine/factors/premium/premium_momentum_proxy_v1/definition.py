"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'premium_momentum_proxy_v1'
FACTOR_VERSION = 'catalog-20260826-86d6d0216005-c00e33920ccc'
PRIMARY_FAMILY = 'premium'
OPERATOR_ID = 'premium_momentum_proxy_v1'
FIXED_PARAMS = {'windowsize': 10}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/premium_momentum_proxy_v1'
CONTRACT_PATH = 'factors/premium/premium_momentum_proxy_v1/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/premium_momentum_proxy_v1/contract.json'


def definition_payload() -> dict[str, object]:
    """Return this immutable instance's execution metadata as a fresh mapping."""

    return {
        'factor_id': FACTOR_ID,
        'factor_version': FACTOR_VERSION,
        'primary_family': PRIMARY_FAMILY,
        'operator_id': OPERATOR_ID,
        'fixed_params': dict(FIXED_PARAMS),
        'output_col': OUTPUT_COL,
        'rust_contract_id': RUST_CONTRACT_ID,
        'operator_contract_path': OPERATOR_CONTRACT_PATH,
    }


def build_factor_spec():
    """Build an exact ``FactorSpec`` without importing or registering an operator."""

    from cbond_on.domain.factors.spec import FactorSpec

    return FactorSpec(
        name=FACTOR_ID,
        factor=OPERATOR_ID,
        params=dict(FIXED_PARAMS),
        output_col=OUTPUT_COL,
        rust_contract_id=RUST_CONTRACT_ID,
    )
