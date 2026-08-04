"""Qualified one-signal view of the bond/stock copula-tail family.

The source module keeps all three pre-registered mechanisms for unit-level
audit.  This wrapper exposes only the upper-tail member that passed the fixed
381-day IC and provisional v7 redundancy pre-screen, avoiding a full build of
two expressions already rejected at the immutable IC threshold.
"""

from __future__ import annotations

from cbond_on.domain.factors.defs import (
    research_factor_mining_daily_bond_stock_copula_tail_dependence_v1 as source,
)


CATALOG_VERSION = "20260803_daily_bond_stock_copula_tail_dependence_healthy_catalog_v1"
_QUALIFIED_SIGNAL = "bsct_upper_tail_dependence60"


def factor_mining_catalog() -> tuple[object, ...]:
    """Return exactly the independently qualified upper-tail candidate."""

    entries = tuple(
        entry for entry in source.factor_mining_catalog() if entry.signal == _QUALIFIED_SIGNAL
    )
    if len(entries) != 1:
        raise RuntimeError(
            "copula-tail healthy catalogue requires exactly one qualified signal: "
            f"{_QUALIFIED_SIGNAL}"
        )
    return entries


__all__ = ["CATALOG_VERSION", "factor_mining_catalog"]
