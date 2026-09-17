"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'qed_prior_quote_direction_switch_rate'
FACTOR_VERSION = 'catalog-20260826-75adf422d18f-d4d541b3752d'
PRIMARY_FAMILY = 'quote_side_sequence_dynamics'
OPERATOR_ID = 'factor_mining_quote_execution_dynamics_v1'
FIXED_PARAMS = {'family': 'quote_side_sequence_dynamics', 'signal': 'qed_prior_quote_direction_switch_rate'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/quote_side_sequence_dynamics/qed_prior_quote_direction_switch_rate/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_quote_execution_dynamics_v1/contract.json'


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
