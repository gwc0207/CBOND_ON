"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'heot_joint_state_js_divergence20_60'
FACTOR_VERSION = 'catalog-20260826-9754df8eec77-ccf698efe02f'
PRIMARY_FAMILY = 'historical_execution_open_transition_regime'
OPERATOR_ID = 'factor_mining_historical_execution_to_open_transition_state_v1'
FIXED_PARAMS = {'family': 'historical_execution_open_transition_regime',
 'signal': 'heot_joint_state_js_divergence20_60'}
OUTPUT_COL = None
RUST_CONTRACT_ID = None
CONTRACT_PATH = 'factors/historical_execution_open_transition_regime/heot_joint_state_js_divergence20_60/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_historical_execution_to_open_transition_state_v1/contract.json'


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
