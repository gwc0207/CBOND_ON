"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'cprr_premium_level_residual'
FACTOR_VERSION = 'catalog-20260826-8675e38d85e0-6e9ab768da45'
PRIMARY_FAMILY = 'conditional_premium_repricing_residual'
OPERATOR_ID = 'factor_mining_daily_incremental_v1'
FIXED_PARAMS = {'family': 'conditional_premium_repricing_residual', 'signal': 'cprr_premium_level_residual'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/conditional_premium_repricing_residual/cprr_premium_level_residual/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_daily_incremental_v1/contract.json'


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
