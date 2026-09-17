"""Research-only batch of independently pre-screened daily information families.

This catalogue is intentionally not a successor to, or an append onto, an
accepted list.  It packages only individually smoke-tested daily families for
one future scratch-only full build after the active aggregate v5 build has
naturally completed and passed its immutable-root audit.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_asymmetric_equity_beta_v1 as asymmetric_equity_beta,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_ohlc_wick_path_asymmetry_v1 as ohlc_wick_path,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_relative_rank_flow_coupling_v2 as rank_flow_coupling,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_relative_rank_tail_contradiction_v1 as rank_tail,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_return_liquidity_topology_v1 as return_liquidity_topology,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v1"

# Every source is research-only and has a distinct factor/family namespace.
# v2 is used instead of rank-coupling v1 so the amount signal is never built
# twice under different catalogue roots.
SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    ("asymmetric_equity_beta", asymmetric_equity_beta),
    ("return_liquidity_topology", return_liquidity_topology),
    ("relative_rank_flow_coupling_v2", rank_flow_coupling),
    ("relative_rank_tail_contradiction", rank_tail),
    ("daily_ohlc_wick_path_asymmetry", ohlc_wick_path),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the validated union of the future daily research batch."""

    return aggregate_v3._validate_source_catalogs(
        (source_name, module.factor_mining_catalog())
        for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic scratch expansion runner entrypoint."""

    return factor_mining_daily_orthogonal_batch_catalog()


__all__ = [
    "CATALOG_VERSION",
    "SOURCE_MODULE_NAMES",
    "SOURCE_MODULES",
    "factor_mining_catalog",
    "factor_mining_daily_orthogonal_batch_catalog",
]
