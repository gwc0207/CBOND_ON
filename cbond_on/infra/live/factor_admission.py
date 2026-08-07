"""Explicit production admission for the frozen 2026-08-05 Rust-50 contract.

Some factor metadata lives outside ``domain.factors.defs.__init__`` so normal
research imports cannot expand the production surface accidentally.  This
module imports that metadata only after a configuration opts into the one
ordered Rust-50 profile.  It never creates a second computation route.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
from typing import Any, Final, Sequence

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.registry import FactorRegistry, RegistryError
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg


LIVE50_RUST50_PROFILE: Final[str] = "live50_rust50_20260806"
LIVE50_RUST_CONTRACT_PREFIX: Final[str] = "live50_r5/"
_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CONTRACT_REGISTRY_PATH: Final[Path] = _PACKAGE_ROOT / "factor_contracts" / "registry.json5"
_CONTRACT_PROFILE_PATH: Final[Path] = (
    _PACKAGE_ROOT / "factor_contracts" / "profiles" / "live50_rust50_20260806.json5"
)

# This is one immutable ordered feature contract.  It is intentionally not
# assembled from historical factor batches: the live runtime receives and
# computes all fifty specs through the same Rust API.
LIVE50_COLUMNS: Final[tuple[str, ...]] = (
    "cb_overnight_return_mean_20d",
    "cb_overnight_return_mean_5d",
    "cb_overnight_return_mean_60d",
    "range_30m",
    "cb_overnight_return_mean_10d",
    "amount_30m",
    "vol_30m",
    "alpha001_signed_power_v1",
    "alpha030_close_sign_volume_v1",
    "volume_30m",
    "depth_weighted_imbalance_v1",
    "alpha024_close_trend_filter_v1",
    "cb_overnight_sharpe_20_0930_0935",
    "daily_sharpe_twap_5d_mean5",
    "cb_overnight_sharpe_5_0930_0935",
    "mid_move_30m",
    "premium_momentum_proxy_v1",
    "volen_f60_s10_l3",
    "daily_sharpe_twap_20d_mean5",
    "cb_overnight_return_mean_40d",
    "alpha041_geometric_mean_vwap_v1",
    "alpha078_low_vwap_adv_corr_v1",
    "mom_slope_30m",
    "alpha019_close_momentum_sign_v1",
    "ret_10m",
    "alpha025_return_volume_vwap_range_v1",
    "alpha050_volume_vwap_corr_max_v1",
    "base_debt_premium_floor_gap",
    "bsfst_stock_return_bond_flow_mutual_information60",
    "bssrc_bond_stock_rank_correlation60",
    "bstk_tail_cocrash_residual20",
    "dliq_volume_return_corr20",
    "dohw_intraday_sign_range_asymmetry60",
    "dohw_mean_wick_asymmetry60",
    "dredemption_bondpremium_interaction",
    "dredemption_premium_z20",
    "dret_drawup_drawdown_asym",
    "dret_volatility_20",
    "drt_rebound_from_low20",
    "dtwap_morning_slope20",
    "lcc_amount_trade_size_information60",
    "lcc_volume_deal_information60",
    "lrd_cross_side_reprice_symmetry",
    "prcn_return_capacity_rank_corr60",
    "qed_prior_quote_lag2_agreement",
    "qed_prior_quote_location_dispersion",
    "qed_prior_quote_tail_penetration",
    "rjst_amount_joint_transition_entropy60",
    "rlmi_return_deal_sign_mutual_information60",
    "ydpt_yield_fall_return_beta60",
)

_DEFAULT_FACTOR_METADATA_MODULE: Final[str] = "cbond_on.domain.factors.defs"
_CATALOG_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_catalog_v1"
_BOND_STOCK_RETURN_FLOW_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_bond_stock_return_flow_information_v1"
)
_BOND_STOCK_RANK_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
)
_CONTRACT_STOCK_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_daily_contract_stock_v1"
_DAILY_EXPANSION_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_daily_expansion_v1"
_OHLC_WICK_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_ohlc_wick_path_asymmetry_v1"
)
_DAILY_INCREMENTAL_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_daily_incremental_v1"
_LIQUIDITY_CHANNEL_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_liquidity_channel_composition_v1"
)
_ORDERBOOK_REPRICE_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_orderbook_repricing_v1"
_CAPACITY_RANK_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_capacity_rank_coupling_v1"
)
_QUOTE_EXECUTION_MODULE: Final[str] = "cbond_on.domain.factors.defs.research_factor_mining_quote_execution_dynamics_v1"
_RETURN_LIQUIDITY_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_return_liquidity_topology_v1"
)
_ASYMMETRIC_STATE_MODULE: Final[str] = (
    "cbond_on.domain.factors.defs.research_factor_mining_daily_asymmetric_state_transitions_v1"
)

# This is the complete explicitly declared metadata-registration surface for
# the one ordered live-50 contract.  The standard ``defs`` package is loaded
# internally for every admission, so it is intentionally not repeated in the
# live config: that keeps already-running scheduler processes compatible while
# still validating all fifty feature registrations through one Rust route.
LIVE50_REGISTRATION_MODULES: Final[tuple[str, ...]] = (
    _CATALOG_MODULE,
    _BOND_STOCK_RETURN_FLOW_MODULE,
    _BOND_STOCK_RANK_MODULE,
    _CONTRACT_STOCK_MODULE,
    _DAILY_EXPANSION_MODULE,
    _OHLC_WICK_MODULE,
    _DAILY_INCREMENTAL_MODULE,
    _LIQUIDITY_CHANNEL_MODULE,
    _ORDERBOOK_REPRICE_MODULE,
    _CAPACITY_RANK_MODULE,
    _QUOTE_EXECUTION_MODULE,
    _RETURN_LIQUIDITY_MODULE,
    _ASYMMETRIC_STATE_MODULE,
)

LIVE50_RUNTIME_METADATA_MODULES: Final[tuple[str, ...]] = (
    _DEFAULT_FACTOR_METADATA_MODULE,
    *LIVE50_REGISTRATION_MODULES,
)


@dataclass(frozen=True)
class Live50FactorAdmission:
    """Evidence that the frozen unified live-50 factor contract was loaded."""

    profile: str
    modules: tuple[str, ...]
    factor_columns: tuple[str, ...]
    feature_contract: str


def _require_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list")
    values = tuple(str(item).strip() for item in value)
    if any(not item for item in values):
        raise ValueError(f"{field} must not contain an empty value")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _validate_unified_rust_config(factor_cfg: dict[str, Any], *, specs: Sequence[FactorSpec]) -> None:
    compute_cfg = factor_cfg.get("compute")
    if not isinstance(compute_cfg, dict):
        raise TypeError("live50 factor admission requires compute to be an object")
    engine = str(compute_cfg.get("engine", "")).strip().lower()
    if engine != "rust":
        raise ValueError(
            "live50 factor admission requires compute.engine='rust'"
        )
    if str(compute_cfg.get("execution_policy", "")).strip().lower() != "rust_first":
        raise ValueError(
            "live50 factor admission requires compute.execution_policy='rust_first'"
        )
    retired = sorted(
        key
        for key in ("rust_columns", "python_columns", "preserve_existing_rust_columns")
        if key in compute_cfg
    )
    if retired:
        raise ValueError(
            "live50 unified Rust path must not configure retired hybrid fields: "
            + ", ".join(retired)
        )
def _canonical_spec_payload(specs: Sequence[FactorSpec]) -> list[dict[str, object]]:
    """Return the immutable, JSON-safe live factor instance contract."""

    return [
        {
            "name": str(spec.name),
            "factor": str(spec.factor),
            "params": dict(spec.params or {}),
            "output_col": spec.output_col,
            "rust_contract_id": spec.rust_contract_id,
        }
        for spec in specs
    ]


def _canonical_spec_sha256(specs: Sequence[FactorSpec]) -> str:
    try:
        encoded = json.dumps(
            _canonical_spec_payload(specs),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("live50 factor specs cannot be canonically fingerprinted") from exc
    return hashlib.sha256(encoded).hexdigest()


def _expected_rust_contract_id(column: str) -> str:
    return f"{LIVE50_RUST_CONTRACT_PREFIX}{column}"


def _validate_unified_live50_specs(specs: Sequence[FactorSpec]) -> None:
    """Validate all fifty ordered instances before any panel or store access."""

    columns = tuple(build_factor_col(spec) for spec in specs)
    if columns != LIVE50_COLUMNS:
        raise ValueError(
            "live50 factor specs must be exactly the frozen ordered 50-column contract"
        )
    contract_ids = [str(spec.rust_contract_id or "").strip() for spec in specs]
    if len(set(contract_ids)) != len(contract_ids):
        raise ValueError("live50 factor specs must not reuse a rust_contract_id")
    for column, spec, contract_id in zip(LIVE50_COLUMNS, specs, contract_ids):
        if spec.name != column or spec.output_col not in {None, column}:
            raise ValueError(
                f"live50 factor {column} must use name/output_col={column!r}"
            )
        expected_id = _expected_rust_contract_id(column)
        if contract_id != expected_id:
            raise ValueError(
                f"live50 factor {column} must use rust_contract_id={expected_id!r}"
            )
    profile_payload = load_json_like(_CONTRACT_PROFILE_PATH)
    if not isinstance(profile_payload, dict):  # guarded again for direct callers.
        raise RuntimeError("live50 factor contract profile must be an object")
    expected_hash = str(profile_payload.get("specs_sha256", "")).strip().lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise RuntimeError("live50 factor contract profile must provide a SHA-256 specs_sha256")
    actual_hash = _canonical_spec_sha256(specs)
    if actual_hash != expected_hash:
        raise ValueError(
            "live50 factor spec payload differs from the frozen profile: "
            f"expected_sha256={expected_hash}, actual_sha256={actual_hash}"
        )


def _validate_registered_factor_metadata(
    modules: Sequence[str],
    specs: Sequence[FactorSpec],
) -> None:
    """Load and validate metadata for every feature in the one live-50 pack.

    Registry metadata is needed for factor-context planning, but it is never a
    compute fallback.  Loading the complete fixed module list before checking
    all fifty specs keeps registration symmetric with the one Rust execution
    contract and prevents a feature from entering via an implicit import.
    """

    runtime_modules = (_DEFAULT_FACTOR_METADATA_MODULE, *modules)
    if runtime_modules != LIVE50_RUNTIME_METADATA_MODULES:
        raise RuntimeError("live50 runtime metadata module contract drift")
    for module_path in runtime_modules:
        import_module(module_path)
    for spec in specs:
        factor_key = str(spec.factor).strip()
        try:
            registered = FactorRegistry.get(factor_key)
        except RegistryError as exc:
            raise RuntimeError(
                f"live50 registration did not register factor {factor_key} for {build_factor_col(spec)}"
            ) from exc
        if not registered.__module__.startswith(f"{_DEFAULT_FACTOR_METADATA_MODULE}."):
            raise RuntimeError(
                f"live50 factor {factor_key} for {build_factor_col(spec)} registered from "
                f"unexpected module {registered.__module__}"
            )


def validate_live50_factor_contract_admission(
    *,
    specs: Sequence[FactorSpec] | None = None,
) -> None:
    """Fail closed unless the dedicated contract profile admits all live-50 specs.

    The registry and profile are independent gates for the one standard Rust
    50-column route, rather than trusting a factor pack alone.  When concrete
    specs are supplied, every registered implementation is checked against
    its matching frozen instance as well.
    """

    if not _CONTRACT_REGISTRY_PATH.exists() or not _CONTRACT_PROFILE_PATH.exists():
        raise RuntimeError(
            "live50 factor contract files are missing: "
            f"registry={_CONTRACT_REGISTRY_PATH}, profile={_CONTRACT_PROFILE_PATH}"
        )
    registry_payload = load_json_like(_CONTRACT_REGISTRY_PATH)
    profile_payload = load_json_like(_CONTRACT_PROFILE_PATH)
    if not isinstance(registry_payload, dict) or not isinstance(profile_payload, dict):
        raise RuntimeError("live50 factor contract registry/profile must be objects")
    if profile_payload.get("research_only") is True:
        raise RuntimeError("live50 factor contract profile must not be research_only")
    if str(profile_payload.get("admission_profile", "")).strip() != LIVE50_RUST50_PROFILE:
        raise RuntimeError(
            "live50 factor contract profile admission_profile does not match "
            f"{LIVE50_RUST50_PROFILE!r}"
        )
    if str(profile_payload.get("registry", "")).strip() != "../registry.json5":
        raise RuntimeError("live50 factor contract profile must reference ../registry.json5")
    if str(profile_payload.get("execution_policy", "")).strip().lower() != "rust_first":
        raise RuntimeError("live50 factor contract profile must set execution_policy='rust_first'")
    declared_hash = str(profile_payload.get("specs_sha256", "")).strip().lower()
    if len(declared_hash) != 64 or any(char not in "0123456789abcdef" for char in declared_hash):
        raise RuntimeError("live50 factor contract profile must provide a SHA-256 specs_sha256")
    profile_columns = _require_string_list(
        profile_payload.get("factors"), field="live50 contract profile.factors"
    )
    if profile_columns != LIVE50_COLUMNS:
        raise RuntimeError(
            "live50 factor contract profile must contain exactly the frozen ordered 50 columns"
        )

    raw_contracts = registry_payload.get("factors")
    if not isinstance(raw_contracts, list):
        raise RuntimeError("live50 factor contract registry.factors must be a list")
    contracts_by_name: dict[str, dict[str, object]] = {}
    duplicate_names: set[str] = set()
    for raw_contract in raw_contracts:
        if not isinstance(raw_contract, dict):
            continue
        name = str(raw_contract.get("name", "")).strip()
        if not name:
            continue
        if name in contracts_by_name:
            duplicate_names.add(name)
        contracts_by_name[name] = raw_contract
    relevant_duplicates = sorted(duplicate_names.intersection(LIVE50_COLUMNS))
    if relevant_duplicates:
        raise RuntimeError(
            "live50 factor contract registry has duplicate admitted names: "
            + ", ".join(relevant_duplicates)
        )

    specs_by_column = (
        {build_factor_col(spec): spec for spec in specs}
        if specs is not None
        else {}
    )
    if specs is not None and tuple(specs_by_column) != LIVE50_COLUMNS:
        raise RuntimeError(
            "live50 contract implementation validation requires exactly the frozen ordered 50 specs"
        )

    for column in LIVE50_COLUMNS:
        contract = contracts_by_name.get(column)
        if contract is None:
            raise RuntimeError(f"live50 factor contract registry is missing {column}")
        if contract.get("live_enabled") is not True:
            raise RuntimeError(f"live50 factor contract {column} must set live_enabled=true")
        if contract.get("model_enabled") is not True:
            raise RuntimeError(f"live50 factor contract {column} must set model_enabled=true")
        implementation = str(contract.get("implementation", "")).strip()
        if not implementation:
            raise RuntimeError(f"live50 factor contract {column} must declare implementation")
        if specs is None:
            continue
        spec = specs_by_column[column]
        if implementation != str(spec.factor).strip():
            raise RuntimeError(
                f"live50 factor contract {column} implementation mismatch: "
                f"expected={spec.factor}, actual={implementation}"
            )
        expected_family = str(dict(spec.params or {}).get("family", "")).strip()
        family = str(contract.get("family", "")).strip()
        if expected_family and family != expected_family:
            raise RuntimeError(
                f"live50 factor contract {column} family mismatch: "
                f"expected={expected_family}, actual={family}"
            )


def _validate_model_feature_contract(raw_admission: dict[str, Any]) -> str:
    ref = str(raw_admission.get("model_feature_contract", "")).strip()
    if not ref:
        raise ValueError("live_factor_admission.model_feature_contract is required")
    payload = load_json_like(resolve_config_path(ref))
    if not isinstance(payload, dict):
        raise RuntimeError("live50 model feature contract must be an object")
    factors = _require_string_list(
        payload.get("factors"), field="live50 model feature contract.factors"
    )
    if factors != LIVE50_COLUMNS:
        raise RuntimeError(
            "live50 model feature contract must contain exactly the frozen ordered 50 columns"
        )
    return ref


def prepare_live50_factor_admission(
    factor_cfg: dict[str, Any],
    *,
    specs: Sequence[FactorSpec] | None = None,
) -> Live50FactorAdmission | None:
    """Prepare the sole ordered Rust-50 production contract before execution.

    Normal factor configurations have no ``live_factor_admission`` block.
    Declaring that block opts into the only production contract and therefore
    requires all fifty exact instances; partial or alternate packs fail before
    any panel or FactorStore access.
    """

    spec_list = list(specs) if specs is not None else load_factor_specs_from_cfg(factor_cfg)
    raw_admission = factor_cfg.get("live_factor_admission")
    if raw_admission is None:
        return None
    if not isinstance(raw_admission, dict):
        raise TypeError("live_factor_admission must be an object")
    if factor_cfg.get("research_only") is True or "research_factor_modules" in factor_cfg:
        raise ValueError(
            "live_factor_admission is a production boundary and cannot use research_only "
            "or research_factor_modules"
        )
    if raw_admission.get("enabled") is not True:
        raise ValueError("live_factor_admission.enabled must be true")
    profile = str(raw_admission.get("profile", "")).strip()
    if profile != LIVE50_RUST50_PROFILE:
        raise ValueError(
            "live_factor_admission.profile must be "
            f"{LIVE50_RUST50_PROFILE!r}"
        )
    declared_modules = _require_string_list(
        raw_admission.get("modules"), field="live_factor_admission.modules"
    )
    if set(declared_modules) != set(LIVE50_REGISTRATION_MODULES) or len(declared_modules) != len(
        LIVE50_REGISTRATION_MODULES
    ):
        unknown = sorted(set(declared_modules).difference(LIVE50_REGISTRATION_MODULES))
        missing = sorted(set(LIVE50_REGISTRATION_MODULES).difference(declared_modules))
        raise ValueError(
            "live_factor_admission.modules must be exactly the static live50 registration "
            f"allowlist: missing={missing}, unknown={unknown}"
        )

    _validate_unified_rust_config(factor_cfg, specs=spec_list)
    _validate_unified_live50_specs(spec_list)
    validate_live50_factor_contract_admission(specs=spec_list)
    feature_contract = _validate_model_feature_contract(raw_admission)
    _validate_registered_factor_metadata(declared_modules, spec_list)

    return Live50FactorAdmission(
        profile=profile,
        modules=LIVE50_RUNTIME_METADATA_MODULES,
        factor_columns=LIVE50_COLUMNS,
        feature_contract=feature_contract,
    )


__all__ = [
    "LIVE50_COLUMNS",
    "LIVE50_REGISTRATION_MODULES",
    "LIVE50_RUNTIME_METADATA_MODULES",
    "LIVE50_RUST_CONTRACT_PREFIX",
    "LIVE50_RUST50_PROFILE",
    "Live50FactorAdmission",
    "prepare_live50_factor_admission",
    "validate_live50_factor_contract_admission",
]
