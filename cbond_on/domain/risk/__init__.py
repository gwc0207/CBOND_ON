"""Convertible-bond multi-factor risk model contracts.

The package deliberately contains no live-selection or database behavior.  It
defines the stable vocabulary used by the independent CB-Risk pipeline.
"""

from .contracts import FactorDefinition, RiskModelError, factor_definitions_from_config

__all__ = [
    "FactorDefinition",
    "RiskModelError",
    "factor_definitions_from_config",
]
