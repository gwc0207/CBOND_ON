from __future__ import annotations

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.core.config import load_config_file


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
    return factor_cfg_key, factor_cfg


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

    model_score_cfg["model_id"] = only_model_id
    model_score_cfg["default_model_id"] = only_model_id
    return model_cfg_key, model_score_cfg, only_model_id
