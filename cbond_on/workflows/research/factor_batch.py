from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Final

from cbond_on.app.pipelines.factor_batch_pipeline import execute as run_factor_batch_pipeline
from cbond_on.common.factor_execution_policy import validate_research_execution_policy
from cbond_on.core.registry import FactorRegistry, RegistryError
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg


# These modules deliberately stay outside ``defs.__init__``.  The only generic
# loader that may import them is this research workflow, and only after a
# research config has named an allowlisted module explicitly.
_RESEARCH_FACTOR_MODULES: Final[dict[str, frozenset[str]]] = {
    "cbond_on.domain.factors.defs.ai_factory_wave80_intraday_v1": frozenset(
        {"ai_factory_wave80_intraday_v1"}
    ),
    "cbond_on.domain.factors.defs.daily_prior_intraday_return_surprise_v1": frozenset(
        {"daily_prior_intraday_return_surprise_v1"}
    ),
    "cbond_on.domain.factors.defs.daily_prior_intraday_sharpe_v1": frozenset(
        {"daily_prior_intraday_sharpe_v1"}
    ),
    "cbond_on.domain.factors.defs.parity_adjusted_stock_lag_v1": frozenset(
        {"parity_adjusted_stock_lag_v1"}
    ),
    "cbond_on.domain.factors.defs.parity_adjusted_stock_lag_v2": frozenset(
        {"parity_adjusted_stock_lag_v2"}
    ),
    "cbond_on.domain.factors.defs.research_factor_mining_catalog_v1": frozenset(
        {
            "factor_mining_intraday_catalog_v1",
            "factor_mining_daily_catalog_v1",
            "factor_mining_cross_asset_catalog_v1",
            "factor_mining_hybrid_catalog_v1",
        }
    ),
    "cbond_on.domain.factors.defs.t1430_amount_accel_depth_delta_v2": frozenset(
        {"t1430_amount_accel_depth_delta_v2"}
    ),
    "cbond_on.domain.factors.defs.tail_path_efficiency_5m_v1": frozenset(
        {"tail_path_efficiency_5m_v1"}
    ),
}
_RESEARCH_FACTOR_TO_MODULE: Final[dict[str, str]] = {
    factor_key: module_path
    for module_path, factor_keys in _RESEARCH_FACTOR_MODULES.items()
    for factor_key in factor_keys
}


def load_research_factor_modules(factor_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Explicitly register allowlisted research kernels for one research batch.

    Only the top-level batch config is authoritative.  A factor pack is not
    allowed to turn a default batch into a research-kernel loader implicitly.
    """

    raw_modules = factor_cfg.get("research_factor_modules", [])
    if raw_modules is None:
        raw_modules = []
    if not isinstance(raw_modules, (list, tuple)):
        raise TypeError("factor_config.research_factor_modules must be a list")
    declared_modules = tuple(str(raw_module).strip() for raw_module in raw_modules)
    if any(not module_path for module_path in declared_modules):
        raise ValueError("factor_config.research_factor_modules must not contain an empty module")
    if len(set(declared_modules)) != len(declared_modules):
        raise ValueError("factor_config.research_factor_modules must not contain duplicates")
    if declared_modules and factor_cfg.get("research_only") is not True:
        raise ValueError("research_factor_modules requires top-level research_only: true")

    unknown_modules = sorted(set(declared_modules).difference(_RESEARCH_FACTOR_MODULES))
    if unknown_modules:
        raise ValueError(
            "research_factor_modules contains module(s) outside the static allowlist: "
            + ", ".join(unknown_modules)
        )

    configured_research_keys = {
        str(spec.factor).strip()
        for spec in load_factor_specs_from_cfg(factor_cfg)
        if str(spec.factor).strip() in _RESEARCH_FACTOR_TO_MODULE
    }
    missing_modules = sorted(
        {
            _RESEARCH_FACTOR_TO_MODULE[factor_key]
            for factor_key in configured_research_keys
            if _RESEARCH_FACTOR_TO_MODULE[factor_key] not in declared_modules
        }
    )
    if missing_modules:
        raise ValueError(
            "research-only factor config must set research_only: true and declare "
            "the required research_factor_modules: "
            + ", ".join(missing_modules)
        )

    for module_path in declared_modules:
        module = import_module(module_path)
        for factor_key in _RESEARCH_FACTOR_MODULES[module_path]:
            try:
                registered = FactorRegistry.get(factor_key)
            except RegistryError as exc:
                raise RuntimeError(
                    f"research module {module_path} did not register expected factor {factor_key}"
                ) from exc
            if registered.__module__ != module.__name__:
                raise RuntimeError(
                    f"research module {module_path} registered {factor_key} from "
                    f"unexpected module {registered.__module__}"
                )
    return declared_modules


def prepare_factor_modules(factor_cfg: dict[str, Any]) -> None:
    """Register baseline factors plus explicitly declared research-only modules.

    This is intentionally a research-workflow concern.  Live runtime imports
    only the baseline ``defs`` package, so an undeclared research kernel fails
    closed instead of becoming available to a live factor config by accident.
    """

    # Preserve ordinary factor-batch behavior without re-exporting research
    # modules from the baseline package.
    import_module("cbond_on.domain.factors.defs")
    load_research_factor_modules(factor_cfg)


def run(factor_cfg: dict[str, Any], *, paths_cfg: dict[str, Any]) -> Path:
    validate_research_execution_policy(factor_cfg)
    prepare_factor_modules(factor_cfg)
    return run_factor_batch_pipeline(factor_cfg, paths_cfg=paths_cfg)
