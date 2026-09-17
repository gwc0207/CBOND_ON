"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'qed_prior_quote_location_dispersion'
FACTOR_VERSION = 'catalog-20260826-75adf422d18f-c7a4429d6538'
PRIMARY_FAMILY = 'quote_execution_location_shape'
OPERATOR_ID = 'factor_mining_quote_execution_dynamics_v1'
FIXED_PARAMS = {'family': 'quote_execution_location_shape', 'signal': 'qed_prior_quote_location_dispersion'}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/qed_prior_quote_location_dispersion'
CONTRACT_PATH = 'factors/quote_execution_location_shape/qed_prior_quote_location_dispersion/contract.json'
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
