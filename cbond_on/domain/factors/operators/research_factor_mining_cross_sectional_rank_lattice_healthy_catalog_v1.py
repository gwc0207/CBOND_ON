"""Research-only healthy wrapper for the cross-sectional rank-lattice catalogue.

The source module owns the sole factor implementation and registration.  This
wrapper imports that module explicitly, then exposes only the nine audited
optionality, liquidity, and barrier signals.  The three floor-credit signals
remain deliberately excluded.  Any source-catalogue drift fails closed rather
than silently admitting a new signal into a future research build.
"""

from __future__ import annotations

from collections.abc import Iterable

from cbond_on.domain.factors.operators import (
    research_factor_mining_cross_sectional_rank_lattice_v1 as _rank_lattice,
)


CatalogEntry = _rank_lattice.CatalogEntry
SOURCE_MODULE_NAME = _rank_lattice.__name__
KERNEL_NAME = _rank_lattice.KERNEL_NAME
CATALOG_VERSION = "20260803_cross_sectional_rank_lattice_healthy_catalog_v1"

# Preserve the source catalogue's family-first order after removing the
# unaudited floor-credit family.  This deterministic order is also the exact
# order consumed by the generic research expansion runner.
HEALTHY_FAMILIES = (
    "cross_sectional_optionality_rank_lattice",
    "cross_sectional_liquidity_capacity_rank_lattice",
    "cross_sectional_barrier_geometry_rank_lattice",
)
HEALTHY_SIGNALS = (
    "csl_option_moneyness_stockvol_lattice",
    "csl_option_premium_duration_lattice",
    "csl_option_moneyness_premium_vol_wedge",
    "csl_liquidity_turnover_size_lattice",
    "csl_liquidity_reallocation_stockvol_lattice",
    "csl_liquidity_trade_size_flow_wedge",
    "csl_barrier_call_put_curvature_lattice",
    "csl_barrier_trigger_premium_wedge",
    "csl_barrier_call_put_moneyness_skew",
)
EXCLUDED_FLOOR_SIGNALS = (
    "csl_floor_debtgap_yield_lattice",
    "csl_floor_redemption_distance_lattice",
    "csl_floor_credit_redemption_wedge",
)
_EXPECTED_FAMILY_BY_SIGNAL = {
    **{
        signal: "cross_sectional_optionality_rank_lattice"
        for signal in HEALTHY_SIGNALS[:3]
    },
    **{
        signal: "cross_sectional_liquidity_capacity_rank_lattice"
        for signal in HEALTHY_SIGNALS[3:6]
    },
    **{
        signal: "cross_sectional_barrier_geometry_rank_lattice"
        for signal in HEALTHY_SIGNALS[6:]
    },
    **{
        signal: "cross_sectional_floor_credit_rank_lattice"
        for signal in EXCLUDED_FLOOR_SIGNALS
    },
}
_EXPECTED_SOURCE_SIGNALS = frozenset(_EXPECTED_FAMILY_BY_SIGNAL)


def _select_audited_entries(entries: Iterable[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    """Select the immutable healthy subset while rejecting source drift."""

    by_signal: dict[str, CatalogEntry] = {}
    for entry in entries:
        signal = str(getattr(entry, "signal", "")).strip()
        if not signal:
            raise ValueError("rank-lattice source catalogue has an empty signal")
        if signal in by_signal:
            raise ValueError(f"rank-lattice source catalogue has duplicate signal: {signal}")
        by_signal[signal] = entry

    observed_signals = frozenset(by_signal)
    missing = sorted(_EXPECTED_SOURCE_SIGNALS - observed_signals)
    unexpected = sorted(observed_signals - _EXPECTED_SOURCE_SIGNALS)
    if missing or unexpected:
        raise ValueError(
            "rank-lattice source catalogue drift: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for signal, expected_family in _EXPECTED_FAMILY_BY_SIGNAL.items():
        entry = by_signal[signal]
        if entry.family != expected_family:
            raise ValueError(
                "rank-lattice source catalogue family drift: "
                f"{signal} expected {expected_family}, got {entry.family}"
            )
        if entry.kernel != KERNEL_NAME:
            raise ValueError(
                "rank-lattice source catalogue kernel drift: "
                f"{signal} expected {KERNEL_NAME}, got {entry.kernel}"
            )

    return tuple(by_signal[signal] for signal in HEALTHY_SIGNALS)


_CATALOG = _select_audited_entries(_rank_lattice.cross_sectional_rank_lattice_catalog())


def cross_sectional_rank_lattice_healthy_catalog() -> tuple[CatalogEntry, ...]:
    """Return exactly the nine audited non-floor research candidates."""

    return _CATALOG


def factor_mining_catalog() -> tuple[CatalogEntry, ...]:
    """Generic factor-mining runner entrypoint for the healthy subset."""

    return cross_sectional_rank_lattice_healthy_catalog()


__all__ = [
    "CATALOG_VERSION",
    "CatalogEntry",
    "EXCLUDED_FLOOR_SIGNALS",
    "HEALTHY_FAMILIES",
    "HEALTHY_SIGNALS",
    "KERNEL_NAME",
    "SOURCE_MODULE_NAME",
    "cross_sectional_rank_lattice_healthy_catalog",
    "factor_mining_catalog",
]
