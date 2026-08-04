"""Research-only successor to daily orthogonal batch v1.

v2 preserves the immutable v1 source list and adds the independently audited
capacity-normalized rank-coupling family. It is a future scratch catalogue,
not an incremental change to v1's evidence or to any accepted list.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_capacity_rank_coupling_v1 as capacity_rank_coupling,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v1 as batch_v1,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v2"
_REQUIRED_V1_VERSION = "20260803_daily_orthogonal_batch_v1"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v1.SOURCE_MODULES,
    ("capacity_normalized_rank_coupling", capacity_rank_coupling),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v1_contract() -> None:
    if batch_v1.CATALOG_VERSION != _REQUIRED_V1_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v2 requires immutable v1 version "
            f"{_REQUIRED_V1_VERSION!r}, got {batch_v1.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the validated v1-preserving plus capacity-coupling union."""

    _assert_v1_contract()
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
