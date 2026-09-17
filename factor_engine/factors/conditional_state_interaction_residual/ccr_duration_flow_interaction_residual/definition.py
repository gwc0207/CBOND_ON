"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'ccr_duration_flow_interaction_residual'
FACTOR_VERSION = 'catalog-20260826-e34f22ad6e59-38c1af8c35ab'
PRIMARY_FAMILY = 'conditional_state_interaction_residual'
OPERATOR_ID = 'factor_mining_conditional_response_residual_v1'
FIXED_PARAMS = {'family': 'conditional_state_interaction_residual',
 'signal': 'ccr_duration_flow_interaction_residual'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/conditional_state_interaction_residual/ccr_duration_flow_interaction_residual/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_conditional_response_residual_v1/contract.json'


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
