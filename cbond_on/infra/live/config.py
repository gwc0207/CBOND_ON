from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file
from cbond_on.infra.live.factor_admission import LIVE50_COLUMNS, prepare_live50_factor_admission


def configure_live_paths_profile(live_cfg: dict) -> Path | None:
    """Apply an explicitly versioned paths profile for this live process.

    A live factor-set migration may need an isolated FactorStore while keeping
    the ordinary ``results/live`` publication root. The profile is declared in
    live configuration rather than relying on a Dashboard-parent environment
    variable. A conflicting inherited profile is an error because it would
    silently score a model against a different factor universe.
    """

    runtime_raw = live_cfg.get("runtime", {})
    if runtime_raw is None:
        return None
    if not isinstance(runtime_raw, dict):
        raise TypeError("live_config.runtime must be an object when provided")
    ref = str(runtime_raw.get("paths_config", "")).strip()
    if not ref:
        return None

    requested = resolve_config_path(ref).resolve()
    inherited = str(os.environ.get("CBOND_ON_PATHS_CONFIG", "")).strip()
    if inherited:
        current = resolve_config_path(inherited).resolve()
        if current != requested:
            raise RuntimeError(
                "live paths profile conflict: "
                f"environment={current}, live_config.runtime.paths_config={requested}"
            )
    os.environ["CBOND_ON_PATHS_CONFIG"] = str(requested)
    return requested


def assert_no_date_fields_in_live_config(schedule_cfg: dict, model_cfg: dict) -> None:
    schedule_forbidden = ["start", "target"]
    model_forbidden = ["start", "end"]

    bad_schedule = [
        key for key in schedule_forbidden
        if key in schedule_cfg and str(schedule_cfg.get(key)).strip() not in {"", "None", "none", "null"}
    ]
    bad_model = [
        key for key in model_forbidden
        if key in model_cfg and str(model_cfg.get(key)).strip() not in {"", "None", "none", "null"}
    ]

    if bad_schedule or bad_model:
        parts: list[str] = []
        if bad_schedule:
            parts.append(f"schedule.{','.join(bad_schedule)}")
        if bad_model:
            parts.append(f"model_score.{','.join(bad_model)}")
        raise ValueError(
            "live_config date fields are not allowed; "
            "live always resolves runtime day from current date. "
            f"remove: {', '.join(parts)}"
        )


def load_strategy_config(path_text: str | None) -> dict:
    if not path_text:
        return {}
    path = resolve_config_path(path_text)
    return load_json_like(path)


def load_live_factor_runtime(live_cfg: dict) -> tuple[str, dict]:
    factor_group = dict(live_cfg.get("factor", {}))
    factor_cfg_key = str(factor_group.get("config", "live/live_factors")).strip()
    if not factor_cfg_key:
        raise ValueError("live_config.factor.config must not be empty")
    factor_cfg = dict(load_config_file(factor_cfg_key))

    inline_factors = factor_cfg.get("factors")
    factor_files = factor_cfg.get("factor_files")
    has_inline = isinstance(inline_factors, list) and len(inline_factors) > 0
    has_files = isinstance(factor_files, list) and len(factor_files) > 0
    if has_files and len(factor_files) != 1:
        raise ValueError("live_factors must contain exactly one factor_files entry")
    if not has_inline and not has_files:
        raise ValueError("live_factors must define non-empty factors or one factor_files entry")
    # There is no longer a generic/legacy live factor mode.  Any configuration
    # selected by live runtime must prove the one ordered Rust-50 admission
    # here, before DataHub readiness, factor reads, or downstream scoring.
    if prepare_live50_factor_admission(factor_cfg) is None:
        raise RuntimeError("live runtime requires the admitted unified Rust-50 factor contract")
    return factor_cfg_key, factor_cfg


def validate_live50_model_score_config(
    model_score_cfg: Mapping[str, Any],
    *,
    expected_model_id: str | None = None,
    source: str = "live model config",
) -> str:
    """Fail closed unless one score source consumes the full ordered Rust-50 set.

    Factor admission alone is insufficient: a stale model config could silently
    select a subset from the live-50 FactorStore.  Every primary model and
    every model-switch source therefore has to resolve to the same exact
    ordered fifty feature names before score I/O begins.
    """

    models_raw = model_score_cfg.get("models")
    if not isinstance(models_raw, Mapping) or not models_raw:
        raise ValueError(f"{source} models must be a non-empty object")
    models = {str(k).strip(): v for k, v in models_raw.items() if str(k).strip()}
    if len(models) != 1:
        raise ValueError(f"{source} must contain exactly one model entry")
    model_id, model_entry_raw = next(iter(models.items()))
    if expected_model_id and model_id != expected_model_id:
        raise ValueError(
            f"{source} model mismatch: expected={expected_model_id}, actual={model_id}"
        )
    if not isinstance(model_entry_raw, Mapping):
        raise TypeError(f"{source} model entry {model_id} must be an object")
    model_cfg_key = str(model_entry_raw.get("model_config", "")).strip()
    if not model_cfg_key:
        raise ValueError(f"{source} model entry {model_id} requires model_config")
    model_cfg = dict(load_config_file(model_cfg_key))
    raw_factors = model_cfg.get("factors")
    if not isinstance(raw_factors, (list, tuple)):
        raise TypeError(f"{source} model_config {model_cfg_key} factors must be a list")
    factors = tuple(str(item).strip() for item in raw_factors)
    if factors != LIVE50_COLUMNS:
        raise ValueError(
            f"{source} model_config {model_cfg_key} must use exactly the ordered live50 "
            "feature contract"
        )
    return model_id


def load_live_model_runtime(live_cfg: dict) -> tuple[str, dict, str]:
    model_group = dict(live_cfg.get("model_score", {}))
    model_cfg_key = str(model_group.get("config", "live/live_models")).strip()
    if not model_cfg_key:
        raise ValueError("live_config.model_score.config must not be empty")
    model_score_cfg = dict(load_config_file(model_cfg_key))

    models_raw = model_score_cfg.get("models")
    if not isinstance(models_raw, dict) or not models_raw:
        raise ValueError("live_models.models must be a non-empty object")
    models = {str(k).strip(): v for k, v in models_raw.items() if str(k).strip()}
    if len(models) != 1:
        raise ValueError("live_models must contain exactly one model entry")
    only_model_id = next(iter(models.keys()))

    requested_model_id = str(
        model_group.get("model_id")
        or model_score_cfg.get("model_id")
        or model_score_cfg.get("default_model_id")
        or ""
    ).strip()
    if requested_model_id and requested_model_id != only_model_id:
        raise ValueError(
            "live model mismatch: "
            f"live_config.model_score.model_id={requested_model_id}, "
            f"live_models only model={only_model_id}"
        )

    validate_live50_model_score_config(
        model_score_cfg,
        expected_model_id=only_model_id,
        source=f"live model config {model_cfg_key}",
    )

    model_score_cfg["model_id"] = only_model_id
    model_score_cfg["default_model_id"] = only_model_id
    return model_cfg_key, model_score_cfg, only_model_id
