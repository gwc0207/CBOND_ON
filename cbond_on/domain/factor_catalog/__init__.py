"""Read-only access and validation for the generated factor catalogue.

This package intentionally has no dependency on the live factor runtime.  It
models the identity and provenance layer only; existing ``FactorRegistry`` and
all live admission paths remain unchanged until a separately approved phase.
"""

from .resolver import (
    catalog_path,
    live_release_path,
    load_factor_catalog,
    load_live_release,
    operator_catalog_path,
    resolve_factor_instance,
    resolve_operator_modules,
)
from .validation import FactorCatalogValidationError, load_json, validate_factor_catalog

__all__ = [
    "catalog_path",
    "FactorCatalogValidationError",
    "live_release_path",
    "load_factor_catalog",
    "load_live_release",
    "load_json",
    "operator_catalog_path",
    "resolve_factor_instance",
    "resolve_operator_modules",
    "validate_factor_catalog",
]
