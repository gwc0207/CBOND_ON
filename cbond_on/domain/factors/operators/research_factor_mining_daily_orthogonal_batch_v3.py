"""Research-only successor to daily orthogonal batch v2.

v3 preserves the immutable v2 catalogue and appends only the independently
pre-screened bond/stock empirical-copula tail family.  It remains a future
scratch catalogue: nothing here is imported by ``defs.__init__`` or wired to
live, model, contract, database, or scheduler paths.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_bond_stock_copula_tail_dependence_healthy_catalog_v1 as copula_tail_healthy,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v2 as batch_v2,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v3"
_REQUIRED_V2_VERSION = "20260803_daily_orthogonal_batch_v2"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v2.SOURCE_MODULES,
    ("bond_stock_copula_tail_dependence", copula_tail_healthy),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v2_contract() -> None:
    if batch_v2.CATALOG_VERSION != _REQUIRED_V2_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v3 requires immutable v2 version "
            f"{_REQUIRED_V2_VERSION!r}, got {batch_v2.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v2-preserving union plus the audited tail family."""

    _assert_v2_contract()
    return aggregate_v3._validate_source_catalogs(
        (source_name, module.factor_mining_catalog())
        for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic scratch expansion-runner entrypoint."""

    return factor_mining_daily_orthogonal_batch_catalog()


__all__ = [
    "CATALOG_VERSION",
    "SOURCE_MODULE_NAMES",
    "SOURCE_MODULES",
    "factor_mining_catalog",
    "factor_mining_daily_orthogonal_batch_catalog",
]
