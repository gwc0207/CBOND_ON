"""Research-only successor to daily orthogonal batch v5.

v6 keeps the immutable v5 source prefix and appends the five-member
liquidity-channel-composition family.  Its known within-family alternatives
remain explicit nodes for the eventual global exact MIS; no family label is
used to bypass pairwise redundancy constraints.  This is a future scratch
catalogue only, never a live/model/profile registration.
"""

from __future__ import annotations

from types import ModuleType

from cbond_on.domain.factors.defs import (
    research_factor_mining_aggregate_catalog_v3 as aggregate_v3,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_liquidity_channel_composition_v1 as liquidity_composition,
)
from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_orthogonal_batch_v5 as batch_v5,
)


CATALOG_VERSION = "20260803_daily_orthogonal_batch_v6"
_REQUIRED_V5_VERSION = "20260803_daily_orthogonal_batch_v5"

SOURCE_MODULES: tuple[tuple[str, ModuleType], ...] = (
    *batch_v5.SOURCE_MODULES,
    ("liquidity_channel_composition", liquidity_composition),
)
SOURCE_MODULE_NAMES = tuple(module.__name__ for _, module in SOURCE_MODULES)


def _assert_v5_contract() -> None:
    if batch_v5.CATALOG_VERSION != _REQUIRED_V5_VERSION:
        raise RuntimeError(
            "daily orthogonal batch v6 requires immutable v5 version "
            f"{_REQUIRED_V5_VERSION!r}, got {batch_v5.CATALOG_VERSION!r}"
        )


def factor_mining_daily_orthogonal_batch_catalog() -> tuple[object, ...]:
    """Return the v5-preserving union plus liquidity-composition candidates."""

    _assert_v5_contract()
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
