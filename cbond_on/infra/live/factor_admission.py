"""Explicit production admission for the frozen 2026-08-05 Rust-50 contract.

Some factor metadata lives outside ``domain.factors.operators.__init__`` so normal
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
from cbond_on.core.registry import OperatorRegistry, RegistryError
from cbond_on.domain.factor_catalog import (
    FactorCatalogValidationError,
    load_live_release,
    resolve_factor_instance,
    resolve_operator_modules,
)
from cbond_on.domain.factors.spec import FactorSpec, build_factor_col
from cbond_on.infra.factors.quality import load_factor_specs_from_cfg


LIVE50_RUST50_PROFILE: Final[str] = "live50_rust50_20260806"
LIVE50_RUST_CONTRACT_PREFIX: Final[str] = "live50_r5/"
LIVE50_RELEASE_ID: Final[str] = "live50_rust50_operator_source_20260826"
_PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT: Final[Path] = _PACKAGE_ROOT.parent
_CONTRACT_REGISTRY_PATH: Final[Path] = _PACKAGE_ROOT / "factor_contracts" / "registry.json5"
_CONTRACT_PROFILE_PATH: Final[Path] = (
    _PACKAGE_ROOT / "factor_contracts" / "profiles" / "live50_rust50_20260806.json5"
)

# This is one immutable ordered feature contract.  It is intentionally not
# assembled from historical factor batches: the live runtime receives and
# computes all fifty specs through the same Rust API.
_LEGACY_LIVE50_COLUMNS: Final[tuple[str, ...]] = (
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

_LEGACY_DEFAULT_FACTOR_METADATA_MODULE: Final[str] = "cbond_on.domain.factors.operators"
_CATALOG_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_catalog_v1"
_BOND_STOCK_RETURN_FLOW_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_bond_stock_return_flow_information_v1"
)
_BOND_STOCK_RANK_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_bond_stock_cross_sectional_rank_concordance_v1"
)
_CONTRACT_STOCK_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_daily_contract_stock_v1"
_DAILY_EXPANSION_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_daily_expansion_v1"
_OHLC_WICK_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_ohlc_wick_path_asymmetry_v1"
)
_DAILY_INCREMENTAL_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_daily_incremental_v1"
_LIQUIDITY_CHANNEL_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_liquidity_channel_composition_v1"
)
_ORDERBOOK_REPRICE_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_orderbook_repricing_v1"
_CAPACITY_RANK_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_capacity_rank_coupling_v1"
)
_QUOTE_EXECUTION_MODULE: Final[str] = "cbond_on.domain.factors.operators.research_factor_mining_quote_execution_dynamics_v1"
_RETURN_LIQUIDITY_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_return_liquidity_topology_v1"
)
_ASYMMETRIC_STATE_MODULE: Final[str] = (
    "cbond_on.domain.factors.operators.research_factor_mining_daily_asymmetric_state_transitions_v1"
)

# This is the complete explicitly declared metadata-registration surface for
# the one ordered live-50 contract.  The standard operator package is loaded
# internally for every admission, so it is intentionally not repeated in the
# live config: that keeps already-running scheduler processes compatible while
# still validating all fifty feature registrations through one Rust route.
_LEGACY_LIVE50_REGISTRATION_MODULES: Final[tuple[str, ...]] = (
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

_LEGACY_LIVE50_RUNTIME_METADATA_MODULES: Final[tuple[str, ...]] = (
    _LEGACY_DEFAULT_FACTOR_METADATA_MODULE,
    *_LEGACY_LIVE50_REGISTRATION_MODULES,
)


def _catalog_live50_release() -> dict[str, Any]:
    """Read the one immutable live release without importing implementations."""

    try:
        release = dict(load_live_release(LIVE50_RELEASE_ID))
    except FactorCatalogValidationError as exc:
        raise RuntimeError(
            "live50 factor catalog release cannot be resolved before admission"
        ) from exc
    instances = release.get("instances")
    if not isinstance(instances, list) or len(instances) != 50:
        raise RuntimeError("live50 factor catalog release must contain exactly 50 instances")
    return release


_LIVE50_CATALOG_RELEASE: Final[dict[str, Any]] = _catalog_live50_release()
_LIVE50_CATALOG_INSTANCES: Final[tuple[dict[str, Any], ...]] = tuple(
    dict(item) for item in _LIVE50_CATALOG_RELEASE["instances"]
)
LIVE50_COLUMNS: Final[tuple[str, ...]] = tuple(
    str(item.get("output_col") or item.get("factor_id") or "").strip()
    for item in _LIVE50_CATALOG_INSTANCES
)
if LIVE50_COLUMNS != _LEGACY_LIVE50_COLUMNS:
    raise RuntimeError("live50 catalog release column order differs from frozen legacy contract")

try:
    _LIVE50_CATALOG_OPERATOR_METADATA: Final[tuple[dict[str, Any], ...]] = tuple(
        dict(item)
        for item in resolve_operator_modules(
            (str(item.get("factor_id", "")).strip() for item in _LIVE50_CATALOG_INSTANCES)
        )
    )
except FactorCatalogValidationError as exc:
    raise RuntimeError("live50 catalog release references unresolved operator metadata") from exc

LIVE50_REGISTRATION_MODULES: Final[tuple[str, ...]] = tuple(
    str(item.get("implementation_module", "")).strip()
    for item in _LIVE50_CATALOG_OPERATOR_METADATA
)
if len(LIVE50_REGISTRATION_MODULES) != len(set(LIVE50_REGISTRATION_MODULES)) or any(
    not item for item in LIVE50_REGISTRATION_MODULES
):
    raise RuntimeError("live50 catalog release has invalid operator module metadata")
LIVE50_RUNTIME_METADATA_MODULES: Final[tuple[str, ...]] = LIVE50_REGISTRATION_MODULES


@dataclass(frozen=True)
class Live50FactorAdmission:
    """Evidence that the frozen unified live-50 factor contract was loaded."""

    profile: str
    release_id: str
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_catalog_release_binding(specs: Sequence[FactorSpec]) -> None:
    """Prove that the frozen live pack is exactly the Catalog live release.

    The catalog is the identity source; the pre-existing pack/profile remain
    independent execution gates.  Both must agree before implementation
    modules are imported or any market data is read.
    """

    source_artifacts = _LIVE50_CATALOG_RELEASE.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise RuntimeError("live50 catalog release is missing source artifact evidence")
    expected_sources = {
        "live50_pack": _PACKAGE_ROOT / "config" / "factor" / "packs" / "live_screened_no_winsor_50_20260805.json5",
        "live50_profile": _CONTRACT_PROFILE_PATH,
    }
    for key, path in expected_sources.items():
        artifact = source_artifacts.get(key)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"live50 catalog release is missing {key} evidence")
        if str(artifact.get("path", "")).replace("\\", "/") != str(
            path.relative_to(_REPOSITORY_ROOT).as_posix()
        ):
            raise RuntimeError(f"live50 catalog release {key} path drift")
        expected_sha = str(artifact.get("sha256", "")).strip().lower()
        if len(expected_sha) != 64 or _sha256_file(path) != expected_sha:
            raise RuntimeError(f"live50 catalog release {key} hash drift")

    if len(specs) != len(_LIVE50_CATALOG_INSTANCES):
        raise RuntimeError("live50 catalog release/spec count mismatch")
    for release_item, spec in zip(_LIVE50_CATALOG_INSTANCES, specs):
        factor_id = str(release_item.get("factor_id", "")).strip()
        column = build_factor_col(spec)
        if factor_id != column:
            raise RuntimeError(
                f"live50 catalog release order mismatch: expected={factor_id}, actual={column}"
            )
        try:
            identity = resolve_factor_instance(
                factor_id,
                factor_version=str(release_item.get("factor_version", "")),
                contract_hash=str(release_item.get("contract_hash", "")),
            )
        except FactorCatalogValidationError as exc:
            raise RuntimeError(f"live50 catalog identity mismatch for {factor_id}") from exc
        if str(identity.get("operator_id", "")).strip() != str(spec.factor).strip():
            raise RuntimeError(f"live50 catalog operator mismatch for {factor_id}")
        if str(release_item.get("operator_id", "")).strip() != str(spec.factor).strip():
            raise RuntimeError(f"live50 release operator mismatch for {factor_id}")
        if dict(release_item.get("runtime_params") or {}) != dict(spec.params or {}):
            raise RuntimeError(f"live50 release parameter mismatch for {factor_id}")
        if str(release_item.get("rust_contract_id", "")).strip() != str(
            spec.rust_contract_id or ""
        ).strip():
            raise RuntimeError(f"live50 release Rust contract mismatch for {factor_id}")


def _validate_registered_operator_metadata(specs: Sequence[FactorSpec]) -> None:
    """Load only modules bound by the admitted release and verify operators.

    Importing the full operator package is deliberately absent here: the historical package
    initializer registered 194 unrelated operators.  The Catalog release
    supplies the exact 33 implementation modules needed by the 50 instances.
    """

    expected_module_by_operator = {
        str(item.get("operator_id", "")).strip(): str(
            item.get("implementation_module", "")
        ).strip()
        for item in _LIVE50_CATALOG_OPERATOR_METADATA
    }
    for module_path in LIVE50_REGISTRATION_MODULES:
        if not module_path.startswith("cbond_on.domain.factors.operators."):
            raise RuntimeError(f"unsafe live50 operator module: {module_path}")
        import_module(module_path)
    for spec in specs:
        operator_id = str(spec.factor).strip()
        expected_module = expected_module_by_operator.get(operator_id)
        if not expected_module:
            raise RuntimeError(
                f"live50 catalog has no admitted module for operator {operator_id}"
            )
        try:
            registered = OperatorRegistry.get(operator_id)
        except RegistryError as exc:
            raise RuntimeError(
                f"live50 operator registration missing {operator_id} for {build_factor_col(spec)}"
            ) from exc
        if registered.__module__ != expected_module:
            raise RuntimeError(
                f"live50 operator {operator_id} for {build_factor_col(spec)} registered from "
                f"unexpected module {registered.__module__}, expected={expected_module}"
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
    release_id = str(raw_admission.get("release_id", "")).strip()
    if release_id != LIVE50_RELEASE_ID:
        raise ValueError(
            "live_factor_admission.release_id must be "
            f"{LIVE50_RELEASE_ID!r}"
        )
    declared_modules = _require_string_list(
        raw_admission.get("modules"), field="live_factor_admission.modules"
    )
    # The config retains its compact historical declaration of the 13 research
    # modules.  The full 33-module runtime surface is resolved exclusively
    # from the immutable Catalog release below.
    if set(declared_modules) != set(_LEGACY_LIVE50_REGISTRATION_MODULES) or len(
        declared_modules
    ) != len(_LEGACY_LIVE50_REGISTRATION_MODULES):
        unknown = sorted(
            set(declared_modules).difference(_LEGACY_LIVE50_REGISTRATION_MODULES)
        )
        missing = sorted(
            set(_LEGACY_LIVE50_REGISTRATION_MODULES).difference(declared_modules)
        )
        raise ValueError(
            "live_factor_admission.modules must be exactly the static live50 registration "
            "research-module allowlist: "
            f"allowlist: missing={missing}, unknown={unknown}"
        )

    _validate_unified_rust_config(factor_cfg, specs=spec_list)
    _validate_unified_live50_specs(spec_list)
    _validate_catalog_release_binding(spec_list)
    validate_live50_factor_contract_admission(specs=spec_list)
    feature_contract = _validate_model_feature_contract(raw_admission)
    _validate_registered_operator_metadata(spec_list)

    return Live50FactorAdmission(
        profile=profile,
        release_id=release_id,
        modules=LIVE50_RUNTIME_METADATA_MODULES,
        factor_columns=LIVE50_COLUMNS,
        feature_contract=feature_contract,
    )


__all__ = [
    "LIVE50_COLUMNS",
    "LIVE50_RELEASE_ID",
    "LIVE50_REGISTRATION_MODULES",
    "LIVE50_RUNTIME_METADATA_MODULES",
    "LIVE50_RUST_CONTRACT_PREFIX",
    "LIVE50_RUST50_PROFILE",
    "Live50FactorAdmission",
    "prepare_live50_factor_admission",
    "validate_live50_factor_contract_admission",
]
