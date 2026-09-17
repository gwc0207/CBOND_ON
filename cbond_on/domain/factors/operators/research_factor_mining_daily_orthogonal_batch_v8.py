"""Research-only successor to daily orthogonal batch v7.

v8 preserves the immutable v7 source prefix and appends one separately
audited market-breadth-regime relationship member.  It remains a future
scratch catalogue only and does not modify live or model factor profiles.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_breadth_regime_relation_v1 as breadth_relation,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v7 as batch_v7,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v8"
_REQUIRED_V7_VERSION = "20260803_daily_orthogonal_batch_v7"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v7.SOURCE_MODULES,
    ("breadth_regime_relation", breadth_relation),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v7_contract() -> None:
    if batch_v7.CATALOG_VERSION != _REQUIRED_V7_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v8 requires immutable v7 version "
            f"{_REQUIRED_V7_VERSION!r}, got {batch_v7.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v7-preserving union plus breadth-regime candidate."""

    _assert_v7_contract()
    return aggregate_v3._validate_source_catalogs(
        (source_name, module.factor_mining_catalog())
        for source_name, module in SOURCE_MODULES
    )


def factor_mining_catalog() -> tuple[object, ...]:
    """Generic scratch-expansion runner entrypoint."""

    return factor_mining_daily_orthogonal_batch_catalog()


__all__ = [
    "CATALOG_VERSION",
    "SOURCE_MODULE_NAMES",
    "SOURCE_MODULES",
    "factor_mining_catalog",
    "factor_mining_daily_orthogonal_batch_catalog",
]
