"""Generated parameterized factor definition; never registers a runtime operator."""

from __future__ import annotations

GENERATED_FACTOR_DEFINITION = True
FACTOR_ID = 'alpha024_close_trend_filter_v1'
FACTOR_VERSION = 'catalog-20260826-fcd144646b20-5bb098680b7a'
PRIMARY_FAMILY = 'alpha'
OPERATOR_ID = 'alpha024_close_trend_filter_v1'
FIXED_PARAMS = {'delta_window': 20,
 'short_delta_window': 3,
 'sum_window': 20,
 'trend_threshold': 0.05,
 'ts_min_window': 20}
OUTPUT_COL = None
RUST_CONTRACT_ID = 'live50_r5/alpha024_close_trend_filter_v1'
CONTRACT_PATH = 'factors/alpha/alpha024_close_trend_filter_v1/contract.json'
OPERATOR_CONTRACT_PATH = 'operators/alpha024_close_trend_filter_v1/contract.json'


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
