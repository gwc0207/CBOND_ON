"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'rlmi_return_deal_sign_mutual_information60'
FACTOR_VERSION = 'catalog-20260826-089d8a9cd07f-849297227bff'
PRIMARY_FAMILY = 'prior_return_liquidity_information_dependence'
OPERATOR_ID = 'factor_mining_daily_return_liquidity_topology_v1'
FIXED_PARAMS = {'family': 'prior_return_liquidity_information_dependence',
 'signal': 'rlmi_return_deal_sign_mutual_information60'}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/rlmi_return_deal_sign_mutual_information60'
CONTRACT_PATH = 'factors/prior_return_liquidity_information_dependence/rlmi_return_deal_sign_mutual_information60/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/factor_mining_daily_return_liquidity_topology_v1/contract.json'


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
