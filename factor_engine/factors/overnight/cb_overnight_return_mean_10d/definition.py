"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'cb_overnight_return_mean_10d'
FACTOR_VERSION = 'catalog-20260826-f0673926fc59-db6393016288'
PRIMARY_FAMILY = 'overnight'
OPERATOR_ID = 'daily_overnight_return_mean_v1'
FIXED_PARAMS = {'buy_col': 'twap_1442_1457',
 'sell_col': 'twap_0930_1000',
 'source': 'market_cbond.daily_twap',
 'time_tag': '0930_1000',
 'window': 10}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/cb_overnight_return_mean_10d'
CONTRACT_PATH = 'factors/overnight/cb_overnight_return_mean_10d/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/daily_overnight_return_mean_v1/contract.json'


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
