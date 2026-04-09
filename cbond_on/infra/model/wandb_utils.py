from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _to_python_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if np.isscalar(value):
            value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return value
    except Exception:
        pass
    return str(value)


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in payload.items():
        key = str(k).strip()
        if not key:
            continue
        val = _to_python_scalar(v)
        if val is None:
            continue
        out[key] = val
    return out


class WandbLogger:
    def __init__(self, run: Any, cfg: dict[str, Any]) -> None:
        self._run = run
        self._cfg = cfg
        self.enabled = run is not None
        self.log_iter_metrics = _to_bool(cfg.get("log_iter_metrics"), default=True)
        self.log_every_n_iter = max(1, int(cfg.get("log_every_n_iter", 10)))
        max_points = cfg.get("max_iter_points")
        self.max_iter_points = int(max_points) if max_points is not None else 0

    def log(self, payload: dict[str, Any], *, step: int | None = None, prefix: str | None = None) -> None:
        if not self.enabled:
            return
        clean = _clean_payload(payload)
        if not clean:
            return
        if prefix:
            clean = {f"{prefix}/{k}": v for k, v in clean.items()}
        try:
            self._run.log(clean, step=step)
        except Exception:
            pass

    def log_history(
        self,
        rows: list[dict[str, Any]],
        *,
        step_key: str = "iteration",
        prefix: str | None = None,
    ) -> None:
        if not self.enabled or not self.log_iter_metrics or not rows:
            return
        count = 0
        for row in rows:
            step = row.get(step_key)
            try:
                step_num = int(step) if step is not None else None
            except Exception:
                step_num = None
            if step_num is not None and (step_num % self.log_every_n_iter) != 0:
                continue
            self.log(row, step=step_num, prefix=prefix)
            count += 1
            if self.max_iter_points > 0 and count >= self.max_iter_points:
                break

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        try:
            if summary:
                clean = _clean_payload(summary)
                if clean:
                    for k, v in clean.items():
                        self._run.summary[k] = v
            self._run.finish()
        except Exception:
            pass


def _merge_wandb_cfg(
    *,
    execution_cfg: dict[str, Any] | None,
    model_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(execution_cfg, dict):
        w = execution_cfg.get("wandb")
        if isinstance(w, dict):
            out.update(w)
    if isinstance(model_cfg, dict):
        w = model_cfg.get("wandb")
        if isinstance(w, dict):
            out.update(w)
    return out


def init_wandb_logger(
    *,
    execution_cfg: dict[str, Any] | None,
    model_cfg: dict[str, Any] | None,
    model_name: str,
    model_type: str,
    start: date,
    end: date,
    extra_config: dict[str, Any] | None = None,
) -> WandbLogger:
    cfg = _merge_wandb_cfg(execution_cfg=execution_cfg, model_cfg=model_cfg)
    enabled = _to_bool(cfg.get("enabled"), default=False)
    if not enabled:
        return WandbLogger(None, cfg)

    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[wandb] disabled: import failed: {type(exc).__name__}: {exc}")
        return WandbLogger(None, cfg)

    project = str(cfg.get("project", "cbond_on")).strip() or "cbond_on"
    entity = str(cfg.get("entity", "")).strip() or None
    mode = str(cfg.get("mode", "online")).strip().lower() or "online"
    group = str(cfg.get("group", "")).strip() or None
    job_type = str(cfg.get("job_type", "train")).strip() or "train"
    save_code = _to_bool(cfg.get("save_code"), default=False)
    tags_raw = cfg.get("tags", [])
    tags = [str(t).strip() for t in tags_raw] if isinstance(tags_raw, (list, tuple)) else []
    tags = [t for t in tags if t]
    if model_name not in tags:
        tags.append(model_name)
    if model_type not in tags:
        tags.append(model_type)
    run_name = str(cfg.get("run_name", "")).strip()
    if not run_name:
        run_name = f"{model_name}_{start:%Y%m%d}_{end:%Y%m%d}_{datetime.now():%H%M%S}"

    run_config = {
        "model_name": model_name,
        "model_type": model_type,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    if extra_config:
        run_config.update(extra_config)

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            mode=mode,
            group=group,
            job_type=job_type,
            tags=tags,
            config=run_config,
            save_code=save_code,
            reinit=True,
        )
        print(
            "[wandb] enabled:",
            f"project={project}",
            f"mode={mode}",
            f"run={run_name}",
        )
        return WandbLogger(run, cfg)
    except Exception as exc:
        print(f"[wandb] disabled: init failed: {type(exc).__name__}: {exc}")
        return WandbLogger(None, cfg)


