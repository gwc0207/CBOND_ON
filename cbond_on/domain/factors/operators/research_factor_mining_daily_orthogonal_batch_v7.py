"""Research-only successor to daily orthogonal batch v6.

v7 preserves the immutable v6 source prefix and adds the two-member
observable-market-seasoning family after its independent full-window IC and
redundancy audit.  This catalogue is for a future scratch build only; it does
not register a live/model/profile factor pack.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.operators import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_observable_seasoning_v1 as observable_seasoning,
)
from cbond_on.domain.factors.operators import (
    research_factor_mining_daily_orthogonal_batch_v6 as batch_v6,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v7"
_REQUIRED_V6_VERSION = "20260803_daily_orthogonal_batch_v6"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v6.SOURCE_MODULES,
    ("observable_seasoning", observable_seasoning),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v6_contract() -> None:
    if batch_v6.CATALOG_VERSION != _REQUIRED_V6_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v7 requires immutable v6 version "
            f"{_REQUIRED_V6_VERSION!r}, got {batch_v6.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v6-preserving union plus observable-seasoning candidates."""

    _assert_v6_contract()
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
