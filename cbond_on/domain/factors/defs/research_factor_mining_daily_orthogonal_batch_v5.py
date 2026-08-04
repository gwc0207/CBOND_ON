"""Research-only successor to daily orthogonal batch v4.

v5 preserves the immutable v4 source prefix and appends only the pre-screened
bond-stock return/flow information family.  It is a future scratch catalogue,
not a live/model/profile registration.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_bond_stock_return_flow_information_v1 as return_flow_information,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v4 as batch_v4,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v5"
_REQUIRED_V4_VERSION = "20260803_daily_orthogonal_batch_v4"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v4.SOURCE_MODULES,
    ("bond_stock_return_flow_information", return_flow_information),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v4_contract() -> None:
    if batch_v4.CATALOG_VERSION != _REQUIRED_V4_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v5 requires immutable v4 version "
            f"{_REQUIRED_V4_VERSION!r}, got {batch_v4.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v4-preserving union plus return/flow information candidate."""

    _assert_v4_contract()
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
