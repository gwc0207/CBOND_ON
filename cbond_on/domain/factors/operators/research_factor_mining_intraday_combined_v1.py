"""Explicit combined strict-PIT intraday research catalogue.

This module is intentionally not imported by :mod:`defs.__init__`.  It only
composes two independently audited research catalogues so a scratch factor
batch can load the T1430 panel once and produce the whole intraday candidate
set.  It has no factor implementation, I/O, configuration, or live effect.
"""

from __future__ import annotations

from collections.abc import Iterable

from cbond_on.domain.factors.operators import research_factor_mining_expansion_intraday_v1 as _micro
from cbond_on.domain.factors.operators import research_factor_mining_path_transform_v1 as _path


EXPANSION_VERSION = "20260803_intraday_combined_v1"


def _unique_entries(entries: Iterable[object]) -> tuple[object, ...]:
    """Reject any accidental cross-module output-name collision."""

    out = tuple(entries)
    signals = [str(getattr(entry, "signal", "")) for entry in out]
    if not signals or any(not signal for signal in signals):
        raise ValueError("combined intraday catalogue has an invalid signal")
    if len(set(signals)) != len(signals):
        raise ValueError("combined intraday catalogue has duplicate signal names")
    return out


def factor_mining_catalog() -> tuple[object, ...]:
    """Return 13 family-first, strict-PIT intraday candidate families."""

    return _unique_entries(
        (
            *_micro.factor_mining_expansion_intraday_catalog(),
            *_path.factor_mining_path_transform_catalog(),
        )
    )
