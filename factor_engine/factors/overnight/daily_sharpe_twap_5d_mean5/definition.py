"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'daily_sharpe_twap_5d_mean5'
FACTOR_VERSION = 'catalog-20260826-6471d7eb4298-e3ad86c65be6'
PRIMARY_FAMILY = 'overnight'
OPERATOR_ID = 'daily_sharpe_mean_v1'
FIXED_PARAMS = {'annualize': False,
 'buy_col': 'twap_1442_1457',
 'min_periods': 3,
 'sell_col': 'twap_0930_0935',
 'source': 'market_cbond.daily_twap',
 'time_tag': '0930_0935',
 'window': 5}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/daily_sharpe_twap_5d_mean5'
CONTRACT_PATH = 'factors/overnight/daily_sharpe_twap_5d_mean5/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/daily_sharpe_mean_v1/contract.json'


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
