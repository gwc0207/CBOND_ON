from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from cbond_on.app.pipelines.factor_batch_pipeline import execute as run_factor_batch_pipeline
from cbond_on.common.factor_execution_policy import validate_research_execution_policy
from cbond_on.core.registry import FactorRegistry, RegistryError
from cbond_on.domain.factors.operator_default_loader import load_default_operators
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg


# These modules deliberately stay outside ``defs.__init__``.  The only generic
# loader that may import them is this research workflow, and only after a
# research config has named an allowlisted module explicitly.
_LEGACY_RESEARCH_FACTOR_MODULES: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "cbond_on.domain.factors.operators.ai_factory_wave80_intraday_v1": frozenset(
        {"ai_factory_wave80_intraday_v1"}
    ),
    "cbond_on.domain.factors.operators.daily_prior_intraday_return_surprise_v1": frozenset(
        {"daily_prior_intraday_return_surprise_v1"}
    ),
    "cbond_on.domain.factors.operators.daily_prior_intraday_sharpe_v1": frozenset(
        {"daily_prior_intraday_sharpe_v1"}
    ),
    "cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v1": frozenset(
        {"parity_adjusted_stock_lag_v1"}
    ),
    "cbond_on.domain.factors.operators.parity_adjusted_stock_lag_v2": frozenset(
        {"parity_adjusted_stock_lag_v2"}
    ),
    "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1": frozenset(
        {
            "factor_mining_intraday_catalog_v1",
            "factor_mining_daily_catalog_v1",
            "factor_mining_cross_asset_catalog_v1",
            "factor_mining_hybrid_catalog_v1",
        }
    ),
    "cbond_on.domain.factors.operators.t1430_amount_accel_depth_delta_v2": frozenset(
        {"t1430_amount_accel_depth_delta_v2"}
    ),
    "cbond_on.domain.factors.operators.tail_path_efficiency_5m_v1": frozenset(
        {"tail_path_efficiency_5m_v1"}
    ),
})


# R88 is an isolated research admission profile, not a live-factor extension.
# Its module metadata is intentionally duplicated as a static Python contract:
# a profile cannot authorize an arbitrary import merely by changing JSON.  The
# profile must instead reproduce this exact mapping and ordered module list.
R88_RESEARCH_ADMISSION_PROFILE: Final[str] = "research_r88_rust88_20260825"
R88_RESEARCH_FACTOR_MODULE_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "factor_mining_cross_sectional_microstructure_neighborhood_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_cross_sectional_microstructure_neighborhood_v1"
        ),
        "factor_mining_daily_asymmetric_equity_beta_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_asymmetric_equity_beta_v1"
        ),
        "factor_mining_daily_asymmetric_state_transitions_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_asymmetric_state_transitions_v1"
        ),
        "factor_mining_daily_bond_stock_copula_tail_dependence_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_bond_stock_copula_tail_dependence_v1"
        ),
        "factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
        ),
        "factor_mining_daily_bond_stock_return_flow_information_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_bond_stock_return_flow_information_v1"
        ),
        "factor_mining_daily_capacity_rank_coupling_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_capacity_rank_coupling_v1"
        ),
        "factor_mining_daily_catalog_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1"
        ),
        "factor_mining_daily_contract_stock_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_contract_stock_v1"
        ),
        "factor_mining_daily_expansion_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_expansion_v1"
        ),
        "factor_mining_daily_incremental_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_incremental_v1"
        ),
        "factor_mining_daily_liquidity_channel_composition_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_liquidity_channel_composition_v1"
        ),
        "factor_mining_daily_observable_seasoning_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_observable_seasoning_v1"
        ),
        "factor_mining_daily_ohlc_wick_path_asymmetry_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_ohlc_wick_path_asymmetry_v1"
        ),
        "factor_mining_daily_relative_rank_flow_coupling_v2": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_relative_rank_flow_coupling_v2"
        ),
        "factor_mining_daily_relative_rank_tail_contradiction_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_relative_rank_tail_contradiction_v1"
        ),
        "factor_mining_daily_return_liquidity_topology_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_daily_return_liquidity_topology_v1"
        ),
        "factor_mining_daily_twap_microstructure_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_daily_twap_microstructure_v1"
        ),
        "factor_mining_hybrid_catalog_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1"
        ),
        "factor_mining_intraday_catalog_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1"
        ),
        "factor_mining_intraday_execution_discreteness_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_intraday_execution_discreteness_v1"
        ),
        "factor_mining_intraday_expansion_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_expansion_intraday_v1"
        ),
        "factor_mining_intraday_joint_state_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_intraday_joint_state_v1"
        ),
        "factor_mining_intraday_state_gated_microstructure_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_intraday_state_gated_microstructure_v1"
        ),
        "factor_mining_intraday_transmission_response_v1": (
            "cbond_on.domain.factors.operators."
            "research_factor_mining_intraday_transmission_response_v1"
        ),
        "factor_mining_orderbook_repricing_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_orderbook_repricing_v1"
        ),
        "factor_mining_quote_execution_dynamics_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_quote_execution_dynamics_v1"
        ),
        "factor_mining_quote_geometry_microprice_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_quote_geometry_microprice_v1"
        ),
        "factor_mining_structural_neighborhood_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_structural_neighborhood_v1"
        ),
        "factor_mining_underlying_cohort_distribution_v1": (
            "cbond_on.domain.factors.operators.research_factor_mining_underlying_cohort_distribution_v1"
        ),
    }
)
R88_RESEARCH_FACTOR_MODULES: Final[tuple[str, ...]] = tuple(
    sorted(set(R88_RESEARCH_FACTOR_MODULE_MAP.values()))
)


def _invert_factor_module_map(mapping: Mapping[str, str]) -> Mapping[str, frozenset[str]]:
    by_module: dict[str, set[str]] = {}
    for factor_key, module_path in mapping.items():
        by_module.setdefault(module_path, set()).add(factor_key)
    return MappingProxyType(
        {module_path: frozenset(factor_keys) for module_path, factor_keys in by_module.items()}
    )


_R88_RESEARCH_FACTOR_MODULES_BY_MODULE: Final[Mapping[str, frozenset[str]]] = (
    _invert_factor_module_map(R88_RESEARCH_FACTOR_MODULE_MAP)
)


def _merge_research_module_allowlists(
    base: Mapping[str, frozenset[str]],
    additions: Mapping[str, frozenset[str]],
) -> Mapping[str, frozenset[str]]:
    merged: dict[str, set[str]] = {
        module_path: set(factor_keys) for module_path, factor_keys in base.items()
    }
    for module_path, factor_keys in additions.items():
        merged.setdefault(module_path, set()).update(factor_keys)
    return MappingProxyType(
        {module_path: frozenset(factor_keys) for module_path, factor_keys in merged.items()}
    )


_RESEARCH_FACTOR_MODULES: Final[Mapping[str, frozenset[str]]] = _merge_research_module_allowlists(
    _LEGACY_RESEARCH_FACTOR_MODULES,
    _R88_RESEARCH_FACTOR_MODULES_BY_MODULE,
)
_RESEARCH_FACTOR_TO_MODULE: Final[dict[str, str]] = {
    factor_key: module_path
    for module_path, factor_keys in _RESEARCH_FACTOR_MODULES.items()
    for factor_key in factor_keys
}


def _require_module_list(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list")
    modules = tuple(str(raw_module).strip() for raw_module in value)
    if any(not module_path for module_path in modules):
        raise ValueError(f"{field} must not contain an empty module")
    if len(set(modules)) != len(modules):
        raise ValueError(f"{field} must not contain duplicates")
    return modules


def _require_factor_module_map(value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    normalized: dict[str, str] = {}
    for raw_factor_key, raw_module_path in value.items():
        factor_key = str(raw_factor_key).strip()
        module_path = str(raw_module_path).strip()
        if not factor_key or not module_path:
            raise ValueError(f"{field} must not contain an empty factor key or module")
        if factor_key in normalized:
            raise ValueError(f"{field} must not contain duplicate factor keys")
        normalized[factor_key] = module_path
    return normalized


def _exact_r88_modules_error(declared_modules: tuple[str, ...]) -> ValueError:
    expected = set(R88_RESEARCH_FACTOR_MODULES)
    actual = set(declared_modules)
    missing = sorted(expected.difference(actual))
    unknown = sorted(actual.difference(expected))
    order = "" if missing or unknown else "; module order differs from the frozen R88 allowlist"
    return ValueError(
        "R88 research_factor_modules must exactly equal the static R88 allowlist: "
        f"missing={missing}, unknown={unknown}{order}"
    )


def validate_r88_research_module_contract(
    *,
    declared_modules: object,
    module_map: object,
    factor_keys: Iterable[object],
) -> tuple[str, ...]:
    """Validate the immutable module contract for the R88 research batch.

    The profile lists modules and factor-key mapping for auditability, but the
    executable allowlist remains the constants in this workflow.  Validation
    is deliberately import-free, so missing or altered metadata fails before
    a research kernel can enter the registry.
    """

    modules = _require_module_list(declared_modules, field="R88 research_factor_modules")
    if modules != R88_RESEARCH_FACTOR_MODULES:
        raise _exact_r88_modules_error(modules)

    normalized_map = _require_factor_module_map(
        module_map,
        field="R88 research_factor_module_map",
    )
    expected_map = dict(R88_RESEARCH_FACTOR_MODULE_MAP)
    if normalized_map != expected_map:
        missing = sorted(set(expected_map).difference(normalized_map))
        unknown = sorted(set(normalized_map).difference(expected_map))
        mismatched = sorted(
            factor_key
            for factor_key in set(expected_map).intersection(normalized_map)
            if expected_map[factor_key] != normalized_map[factor_key]
        )
        raise ValueError(
            "R88 research_factor_module_map must exactly equal the static R88 mapping: "
            f"missing={missing}, unknown={unknown}, mismatched={mismatched}"
        )

    configured_keys = {str(raw_factor_key).strip() for raw_factor_key in factor_keys}
    missing_factor_keys = sorted(set(expected_map).difference(configured_keys))
    if missing_factor_keys:
        raise ValueError(
            "R88 factor config is missing static research factor key(s): "
            + ", ".join(missing_factor_keys)
        )
    return modules


def load_research_factor_modules(factor_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Explicitly register allowlisted research kernels for one research batch.

    Only the top-level batch config is authoritative.  A factor pack is not
    allowed to turn a default batch into a research-kernel loader implicitly.
    """

    declared_modules = _require_module_list(
        factor_cfg.get("research_factor_modules", []),
        field="factor_config.research_factor_modules",
    )
    r88_requested = (
        "research_factor_admission_profile" in factor_cfg
        or "research_factor_module_map" in factor_cfg
    )
    if (declared_modules or r88_requested) and factor_cfg.get("research_only") is not True:
        raise ValueError("research_factor_modules requires top-level research_only: true")

    specs = load_factor_specs_from_cfg(factor_cfg)
    configured_factor_keys = {str(spec.factor).strip() for spec in specs}
    if r88_requested:
        admission_profile = str(factor_cfg.get("research_factor_admission_profile", "")).strip()
        if admission_profile != R88_RESEARCH_ADMISSION_PROFILE:
            raise ValueError(
                "research_factor_admission_profile must equal the frozen R88 profile: "
                f"{R88_RESEARCH_ADMISSION_PROFILE}"
            )
        if "research_factor_module_map" not in factor_cfg:
            raise ValueError("R88 research factor config requires research_factor_module_map")
        declared_modules = validate_r88_research_module_contract(
            declared_modules=declared_modules,
            module_map=factor_cfg["research_factor_module_map"],
            factor_keys=configured_factor_keys,
        )
        expected_keys_by_module = _R88_RESEARCH_FACTOR_MODULES_BY_MODULE
    else:
        if "research_factor_module_map" in factor_cfg:
            raise ValueError(
                "research_factor_module_map requires research_factor_admission_profile"
            )
        expected_keys_by_module = _RESEARCH_FACTOR_MODULES

    unknown_modules = sorted(set(declared_modules).difference(_RESEARCH_FACTOR_MODULES))
    if unknown_modules:
        raise ValueError(
            "research_factor_modules contains module(s) outside the static allowlist: "
            + ", ".join(unknown_modules)
        )

    configured_research_keys = configured_factor_keys.intersection(_RESEARCH_FACTOR_TO_MODULE)
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
        for factor_key in expected_keys_by_module[module_path]:
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
    no broad default registry.  An undeclared research kernel therefore fails
    closed instead of becoming available to a live factor config by accident.
    """

    # Preserve ordinary factor-batch behavior without re-exporting research
    # modules from the baseline package.
    load_default_operators()
    load_research_factor_modules(factor_cfg)


def run(factor_cfg: dict[str, Any], *, paths_cfg: dict[str, Any]) -> Path:
    validate_research_execution_policy(factor_cfg)
    prepare_factor_modules(factor_cfg)
    return run_factor_batch_pipeline(factor_cfg, paths_cfg=paths_cfg)
