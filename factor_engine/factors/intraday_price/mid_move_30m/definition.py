"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'mid_move_30m'
FACTOR_VERSION = 'catalog-20260826-1709caac6b91-86f3b779c1b4'
PRIMARY_FAMILY = 'intraday_price'
OPERATOR_ID = 'midprice_move'
FIXED_PARAMS = {'ask_col': 'ask_price1', 'bid_col': 'bid_price1', 'window_minutes': 30}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/mid_move_30m'
CONTRACT_PATH = 'factors/intraday_price/mid_move_30m/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/midprice_move/contract.json'


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
