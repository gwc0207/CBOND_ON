from __future__ import annotations

from typing import Any, Mapping

from cbond_on.domain.risk import factor_definitions_from_config
from cbond_on.schemas.config._base import ConfigValidationError, require_keys, require_mapping


_OFFLINE_MODES = {"offline_replay", "offline_shadow"}


def validate_risk_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the isolated CB-Risk configuration and reject live side effects."""

    out = require_mapping(cfg, name="risk config")
    require_keys(out, name="risk config", keys=("risk_model_id", "mode", "asof", "inputs", "factors", "estimation", "output"))
    mode = str(out.get("mode") or "").strip()
    if mode not in _OFFLINE_MODES:
        raise ConfigValidationError(f"risk mode must be one of {sorted(_OFFLINE_MODES)}, got {mode!r}")

    asof = require_mapping(out["asof"], name="risk.asof")
    require_keys(asof, name="risk.asof", keys=("intraday_cutoff", "static_exposure_lag_trading_days"))
    if int(asof["static_exposure_lag_trading_days"]) < 1:
        raise ConfigValidationError("risk.asof.static_exposure_lag_trading_days must be >= 1")
    if bool(asof.get("require_pit_certified", False)) is False and mode not in _OFFLINE_MODES:
        raise ConfigValidationError("PIT-unverified risk input may only be used in offline mode")

    inputs = require_mapping(out["inputs"], name="risk.inputs")
    require_keys(
        inputs,
        name="risk.inputs",
        keys=("source", "base_table", "twap_table", "pool_table", "buy_twap_col", "sell_twap_col"),
    )
    if str(inputs.get("source")) != "datahub_local_only":
        raise ConfigValidationError("CB-Risk currently supports datahub_local_only inputs only")

    factor_definitions_from_config(out["factors"])
    estimation = require_mapping(out["estimation"], name="risk.estimation")
    require_keys(estimation, name="risk.estimation", keys=("min_samples", "covariance_half_life_days"))
    if int(estimation["min_samples"]) < 30:
        raise ConfigValidationError("risk.estimation.min_samples must be >= 30")
    if float(estimation["covariance_half_life_days"]) <= 0:
        raise ConfigValidationError("risk covariance half life must be positive")

    output = require_mapping(out["output"], name="risk.output")
    if bool(output.get("write_db", False)):
        raise ConfigValidationError("CB-Risk v1 must not write a database")
    if bool(output.get("live_hook_enabled", False)):
        raise ConfigValidationError("CB-Risk v1 must not attach to the live chain")
    if any(key in out for key in ("scheduler", "live", "db_backend", "db_table")):
        raise ConfigValidationError("live/scheduler/database fields do not belong in CB-Risk config")
    return out
