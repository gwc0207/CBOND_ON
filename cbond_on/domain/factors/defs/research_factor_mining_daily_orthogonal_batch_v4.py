"""Research-only successor to daily orthogonal batch v3.

v4 preserves the immutable v3 source prefix and adds the fully pre-screened
bond/stock cross-sectional-rank concordance family.  It is only a future
scratch catalogue, never a live/model/profile registration.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1 as rank_concordance,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v3 as batch_v3,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v4"
_REQUIRED_V3_VERSION = "20260803_daily_orthogonal_batch_v3"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v3.SOURCE_MODULES,
    ("bond_stock_cross_sectional_rank_concordance", rank_concordance),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v3_contract() -> None:
    if batch_v3.CATALOG_VERSION != _REQUIRED_V3_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v4 requires immutable v3 version "
            f"{_REQUIRED_V3_VERSION!r}, got {batch_v3.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v3-preserving union plus rank-concordance candidates."""

    _assert_v3_contract()
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
