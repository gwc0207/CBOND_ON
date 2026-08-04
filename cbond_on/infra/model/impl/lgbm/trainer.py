
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    lgb = None
    _LIGHTGBM_IMPORT_ERROR = exc

from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)
from cbond_on.infra.model.neutralization import FactorNeutralizer


_GPU_HINTS = (
    "gpu",
    "cuda",
    "opencl",
    "boost_compute",
    "gpu tree learner",
    "clgetplatformids",
    "not enabled in this build",
)


def _lgbm_gpu_requested(params: dict) -> bool:
    for key in ("device", "device_type"):
        val = str(params.get(key, "")).strip().lower()
        if val in {"gpu", "cuda", "opencl"}:
            return True
    return False


def _lgbm_cpu_params(params: dict) -> dict:
    out = dict(params)
    for key in ("device", "device_type", "gpu_platform_id", "gpu_device_id"):
        out.pop(key, None)
    out["device"] = "cpu"
    return out


def _looks_like_lgbm_gpu_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _GPU_HINTS)


@dataclass
class SplitData:
    x: pd.DataFrame
    y: pd.Series
    dt: pd.Series
    code: pd.Series
    sample_weight: pd.Series | None = None


@dataclass(frozen=True)
class LabelTargetTransformSpec:
    """Explicit, opt-in target treatment for cross-sectional LGBM training.

    The transform only applies to labels that have already been admitted to a
    training or validation split.  It is deliberately separate from factor
    standardisation and is never applied to a score-day feature frame.
    """

    mode: str = "none"
    ddof: int = 0
    min_std: float = 1e-12
    loss_day_mass: str = "preserve"

    def __post_init__(self) -> None:
        mode = str(self.mode or "none").strip().lower()
        if mode not in {"none", "zscore_day"}:
            raise ValueError("label target transform mode must be none or zscore_day")
        if int(self.ddof) not in {0, 1}:
            raise ValueError("label target transform ddof must be 0 or 1")
        min_std = float(self.min_std)
        if not np.isfinite(min_std) or min_std <= 0.0:
            raise ValueError("label target transform min_std must be finite and > 0")
        loss_day_mass = str(self.loss_day_mass or "preserve").strip().lower()
        if loss_day_mass not in {"preserve", "equal"}:
            raise ValueError("label target transform loss_day_mass must be preserve or equal")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "ddof", int(self.ddof))
        object.__setattr__(self, "min_std", min_std)
        object.__setattr__(self, "loss_day_mass", loss_day_mass)


def transform_label_targets_by_day(
    y: pd.Series,
    dt: pd.Series,
    spec: LabelTargetTransformSpec | None,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    """Return a training target whose completed days are independently scaled.

    ``zscore_day`` uses only the labels present in each individual completed
    day.  It therefore cannot expose a score day to its own label.  Invalid or
    near-constant target days become explicit missing targets so the caller can
    drop and audit the whole completed day rather than silently falling back to
    raw labels.
    """

    resolved = spec or LabelTargetTransformSpec()
    if len(y) != len(dt):
        raise ValueError(
            "label target transform length mismatch: "
            f"labels={len(y)} dt={len(dt)}"
        )
    if resolved.mode == "none":
        return y.copy(), {
            "mode": resolved.mode,
            "ddof": int(resolved.ddof),
            "min_std": float(resolved.min_std),
            "loss_day_mass": resolved.loss_day_mass,
            "day_count": 0,
        }

    target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float, copy=True)
    day = pd.to_datetime(dt, errors="coerce").dt.normalize()
    if len(target) == 0:
        return pd.Series(target, index=y.index, dtype=float), {
            "mode": resolved.mode,
            "ddof": int(resolved.ddof),
            "min_std": float(resolved.min_std),
            "loss_day_mass": resolved.loss_day_mass,
            "day_count": 0,
        }
    if pd.isna(day).any():
        raise ValueError("label target transform received an invalid label day")
    if not np.isfinite(target).all():
        raise ValueError("label target transform received non-finite labels")

    frame = pd.DataFrame({"day": day.to_numpy()})
    transformed = target.copy()
    stds: list[float] = []
    dropped_rows = 0
    dropped_days = 0
    for _, positions in frame.groupby("day", sort=False).groups.items():
        pos = np.asarray(positions, dtype=np.int64)
        values = target[pos]
        std = float(np.std(values, ddof=resolved.ddof))
        if not np.isfinite(std) or std <= resolved.min_std:
            transformed[pos] = np.nan
            dropped_rows += int(len(pos))
            dropped_days += 1
            continue
        transformed[pos] = (values - float(np.mean(values))) / std
        stds.append(std)
    return pd.Series(transformed, index=y.index, dtype=float), {
        "mode": resolved.mode,
        "ddof": int(resolved.ddof),
        "min_std": float(resolved.min_std),
        "loss_day_mass": resolved.loss_day_mass,
        "day_count": int(len(stds)),
        "raw_day_std_min": float(min(stds)) if stds else float("nan"),
        "raw_day_std_max": float(max(stds)) if stds else float("nan"),
        "dropped_day_count": int(dropped_days),
        "dropped_row_count": int(dropped_rows),
    }


def _equalize_day_loss_mass(
    sample_weight: pd.Series | None,
    dt: pd.Series,
) -> tuple[pd.Series, dict[str, float | int | str]]:
    """Give each completed day equal aggregate MSE mass at mean weight one."""

    n_rows = len(dt)
    if sample_weight is None:
        base = np.ones(n_rows, dtype=float)
    else:
        base = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=float, copy=True)
    if base.shape[0] != n_rows:
        raise ValueError(
            "equal-day loss mass length mismatch: "
            f"weights={base.shape[0]} dt={n_rows}"
        )
    if n_rows == 0:
        return pd.Series(dtype=float), {
            "loss_day_mass": "equal",
            "loss_day_count": 0,
            "loss_day_mass_min": float("nan"),
            "loss_day_mass_max": float("nan"),
        }
    if not np.isfinite(base).all() or np.any(base <= 0.0):
        raise ValueError("equal-day loss mass requires finite positive base weights")
    day = pd.to_datetime(dt, errors="coerce").dt.normalize()
    if day.isna().any():
        raise ValueError("equal-day loss mass received an invalid label day")
    frame = pd.DataFrame({"day": day.to_numpy()})
    groups = list(frame.groupby("day", sort=False).groups.values())
    target_day_mass = float(n_rows) / float(len(groups))
    out = base.copy()
    masses: list[float] = []
    for positions in groups:
        pos = np.asarray(positions, dtype=np.int64)
        current_mass = float(np.sum(base[pos]))
        if not np.isfinite(current_mass) or current_mass <= 0.0:
            raise ValueError("equal-day loss mass encountered a non-positive day weight")
        out[pos] = base[pos] * (target_day_mass / current_mass)
        masses.append(float(np.sum(out[pos])))
    if not np.isfinite(out).all() or np.any(out <= 0.0):
        raise RuntimeError("equal-day loss mass produced invalid weights")
    return pd.Series(out, index=dt.index, dtype=float), {
        "loss_day_mass": "equal",
        "loss_day_count": int(len(groups)),
        "loss_day_mass_min": float(min(masses)),
        "loss_day_mass_max": float(max(masses)),
        "loss_weight_mean": float(np.mean(out)),
    }


def prepare_lgbm_target_split(
    split: SplitData,
    spec: LabelTargetTransformSpec | None,
    *,
    apply_loss_day_mass: bool,
) -> tuple[SplitData, dict[str, float | int | str]]:
    """Transform completed labels and optionally equalise only training loss mass."""

    target, audit = transform_label_targets_by_day(split.y, split.dt, spec)
    if target.empty:
        return SplitData(
            x=split.x.copy(),
            y=target.copy(),
            dt=split.dt.copy(),
            code=split.code.copy(),
            sample_weight=split.sample_weight.copy() if split.sample_weight is not None else None,
        ), {
            **audit,
            "input_row_count": 0,
            "output_row_count": 0,
            "loss_day_mass_applied": False,
        }
    keep = np.isfinite(target.to_numpy(dtype=float, copy=False))
    if not keep.any():
        raise ValueError("label target transform removed every completed training row")
    positions = np.flatnonzero(keep)
    prepared = SplitData(
        x=split.x.iloc[positions].reset_index(drop=True),
        y=target.iloc[positions].reset_index(drop=True),
        dt=split.dt.iloc[positions].reset_index(drop=True),
        code=split.code.iloc[positions].reset_index(drop=True),
        sample_weight=(
            split.sample_weight.iloc[positions].reset_index(drop=True)
            if split.sample_weight is not None
            else None
        ),
    )
    audit = {
        **audit,
        "input_row_count": int(len(split.y)),
        "output_row_count": int(len(prepared.y)),
        "loss_day_mass_applied": bool(apply_loss_day_mass and (spec or LabelTargetTransformSpec()).loss_day_mass == "equal"),
    }
    resolved = spec or LabelTargetTransformSpec()
    if apply_loss_day_mass and resolved.loss_day_mass == "equal":
        equal_weights, weight_audit = _equalize_day_loss_mass(
            prepared.sample_weight,
            prepared.dt,
        )
        prepared = SplitData(
            x=prepared.x,
            y=prepared.y,
            dt=prepared.dt,
            code=prepared.code,
            sample_weight=equal_weights,
        )
        audit.update(weight_audit)
    return prepared, audit


@dataclass(frozen=True)
class TemporalFactorLagSpec:
    """An opt-in raw factor-state expansion for a prior trading day.

    This is deliberately independent of label alignment.  For a factor day
    ``T`` it emits current state, immediate-prior-state, and raw difference
    columns.  The caller supplies the previous *calendar* trading day mapping
    so missing factor files can never be silently bridged with an older day.
    """

    lag_trading_days: int = 1
    outputs: tuple[str, ...] = ("t0", "lag1", "diff1")
    missing_policy: str = "inner"

    def __post_init__(self) -> None:
        if int(self.lag_trading_days) != 1:
            raise ValueError("temporal_factor_lag currently supports lag_trading_days=1 only")
        allowed_outputs = {"t0", "lag1", "diff1"}
        if not self.outputs or any(item not in allowed_outputs for item in self.outputs):
            raise ValueError("temporal_factor_lag.outputs must be a non-empty subset of t0, lag1, diff1")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("temporal_factor_lag.outputs must not contain duplicates")
        if self.missing_policy != "inner":
            raise ValueError("temporal_factor_lag.missing_policy must be inner")


def temporal_factor_feature_columns(
    base_factor_cols: Sequence[str],
    spec: TemporalFactorLagSpec | None,
) -> list[str]:
    """Return a stable model schema for the opt-in temporal factor inputs."""
    base = [str(column) for column in base_factor_cols]
    if spec is None:
        return list(base)
    suffix_by_output = {"t0": "__t0", "lag1": "__lag1", "diff1": "__diff1"}
    output = [f"{column}{suffix_by_output[kind]}" for kind in spec.outputs for column in base]
    if len(output) != len(set(output)) or set(output).intersection(base):
        raise ValueError("temporal_factor_lag output columns collide with base factor columns")
    return output


def _normalise_missing_values_config(missing_values: dict[str, Any] | None) -> dict[str, Any]:
    if missing_values in (None, "", []):
        return {}
    if not isinstance(missing_values, dict):
        raise TypeError("missing_values config must be an object")
    return dict(missing_values)


def _normalise_sample_weight_config(sample_weight: dict[str, Any] | None) -> dict[str, Any]:
    if sample_weight in (None, "", []):
        return {}
    if not isinstance(sample_weight, dict):
        raise TypeError("sample_weight config must be an object")
    cfg = dict(sample_weight)
    raw_schemes = cfg.get("schemes", cfg.get("rules"))
    if raw_schemes is None and cfg.get("type") is not None:
        raw_schemes = [cfg]
    if raw_schemes in (None, "", []):
        raw_schemes = []
    if not isinstance(raw_schemes, list):
        raise TypeError("sample_weight.schemes must be a list")
    schemes: list[dict[str, Any]] = []
    for item in raw_schemes:
        if not isinstance(item, dict):
            raise TypeError("sample_weight.schemes entries must be objects")
        schemes.append(dict(item))
    cfg["schemes"] = schemes
    return cfg


def _sample_weight_enabled(sample_weight: dict[str, Any]) -> bool:
    return bool(sample_weight.get("enabled", False) and sample_weight.get("schemes"))


def _rank_pct_by_day(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("dt", group_keys=False)[column].rank(pct=True, method="average")


def _normalise_weight_series(
    w: pd.Series,
    dt: pd.Series,
    *,
    mode: str,
) -> pd.Series:
    mode = str(mode or "day_mean").strip().lower()
    if mode in {"none", "off", "false"}:
        return w
    if mode in {"day_mean", "dt_mean", "daily_mean"}:
        denom = w.groupby(dt).transform("mean").replace(0.0, np.nan)
        return (w / denom).fillna(1.0)
    if mode in {"global_mean", "mean"}:
        denom = float(w.mean())
        if denom > 0 and np.isfinite(denom):
            return w / denom
        return pd.Series(1.0, index=w.index)
    raise ValueError(f"unsupported sample_weight.normalize: {mode}")


def _apply_sample_weight_config(df: pd.DataFrame, sample_weight: dict[str, Any] | None) -> pd.Series | None:
    cfg = _normalise_sample_weight_config(sample_weight)
    if not _sample_weight_enabled(cfg):
        return None
    w = pd.Series(1.0, index=df.index, dtype=float)
    for scheme in cfg.get("schemes", []):
        kind = str(scheme.get("type", scheme.get("kind", ""))).strip().lower()
        if kind in {"time_decay", "recency", "recent"}:
            continue
        multiplier = float(scheme.get("multiplier", scheme.get("weight", 1.0)))
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"sample_weight multiplier must be finite and > 0: {multiplier}")
        if kind in {"label_top_quantile", "top_label_quantile", "label_top_pct", "topk_label"}:
            if "y" not in df.columns:
                raise KeyError("sample_weight label_top_quantile requires y column")
            quantile = float(scheme.get("quantile", scheme.get("pct", 0.8)))
            pct = _rank_pct_by_day(df, "y")
            w *= np.where(pct >= quantile, multiplier, 1.0)
            continue
        if kind in {"positive_label", "positive_return", "label_positive"}:
            if "y" not in df.columns:
                raise KeyError("sample_weight positive_label requires y column")
            threshold = float(scheme.get("threshold", 0.0))
            w *= np.where(pd.to_numeric(df["y"], errors="coerce") > threshold, multiplier, 1.0)
            continue
        if kind in {"liquidity_quantile", "top_liquidity", "liquidity_top_quantile"}:
            column = str(scheme.get("column", "amount_30m")).strip()
            if column not in df.columns:
                raise KeyError(f"sample_weight liquidity column not found: {column}")
            quantile = float(scheme.get("quantile", scheme.get("pct", 0.8)))
            pct = _rank_pct_by_day(df, column)
            w *= np.where(pct >= quantile, multiplier, 1.0)
            continue
        raise ValueError(f"unsupported sample_weight scheme type: {kind}")

    clip_cfg = cfg.get("clip", {})
    if clip_cfg is None:
        clip_cfg = {}
    if not isinstance(clip_cfg, dict):
        raise TypeError("sample_weight.clip must be an object")
    clip_min = clip_cfg.get("min", cfg.get("min"))
    clip_max = clip_cfg.get("max", cfg.get("max"))
    if clip_min is not None or clip_max is not None:
        w = w.clip(
            lower=float(clip_min) if clip_min is not None else None,
            upper=float(clip_max) if clip_max is not None else None,
        )
    w = _normalise_weight_series(w, df["dt"], mode=str(cfg.get("normalize", "day_mean")))
    w = w.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return w.astype(float)


def _missing_values_enabled(missing_values: dict[str, Any]) -> bool:
    return bool(
        missing_values.get("enabled", False)
        or missing_values.get("keep_nan", False)
        or missing_values.get("min_available_factors") is not None
        or missing_values.get("add_valid_count_features", False)
        or missing_values.get("valid_count_features")
    )


def _coerce_factor_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Iterable):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out
    raise TypeError(f"{field} must be a string or list of strings")


def _default_missing_count_kind(name: str) -> str:
    lowered = name.strip().lower()
    if lowered.endswith("_missing_count") or "missing_count" in lowered:
        return "missing_count"
    return "valid_count"


def _iter_missing_feature_specs(
    missing_values: dict[str, Any],
    raw_factor_cols: list[str],
) -> list[tuple[str, str, list[str]]]:
    if not _missing_values_enabled(missing_values):
        return []
    if not (
        bool(missing_values.get("add_valid_count_features", False))
        or missing_values.get("valid_count_features")
    ):
        return []

    raw_specs = missing_values.get("valid_count_features")
    if raw_specs in (None, "", []):
        raw_specs = {"factor_valid_count": {"columns": "__all__", "kind": "valid_count"}}

    items: list[tuple[str, Any]]
    if isinstance(raw_specs, dict):
        items = [(str(k).strip(), v) for k, v in raw_specs.items()]
    elif isinstance(raw_specs, list):
        items = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                raise TypeError("missing_values.valid_count_features list entries must be objects")
            name = str(spec.get("name", "")).strip()
            items.append((name, spec))
    else:
        raise TypeError("missing_values.valid_count_features must be an object or list")

    out: list[tuple[str, str, list[str]]] = []
    for name, spec in items:
        if not name:
            raise ValueError("missing count feature name must be non-empty")
        if spec is False or spec in (None, "", []):
            continue
        kind = _default_missing_count_kind(name)
        columns_value: Any = "__all__"
        if isinstance(spec, dict):
            if not bool(spec.get("enabled", True)):
                continue
            kind = str(spec.get("kind", spec.get("mode", kind))).strip().lower()
            columns_value = spec.get(
                "columns",
                spec.get("factors", spec.get("source_factors", "__all__")),
            )
        elif spec is True:
            columns_value = "__all__"
        else:
            columns_value = spec
        if kind not in {"valid_count", "missing_count"}:
            raise ValueError(f"unsupported missing count feature kind: {kind}")
        if isinstance(columns_value, str) and columns_value.strip().lower() in {"__all__", "all"}:
            source_cols = list(raw_factor_cols)
        else:
            source_cols = _coerce_factor_list(columns_value, field=f"missing feature {name}.columns")
        if not source_cols:
            raise ValueError(f"missing count feature has no source columns: {name}")
        out.append((name, kind, source_cols))
    return out


def missing_value_feature_columns(missing_values: dict[str, Any] | None) -> list[str]:
    cfg = _normalise_missing_values_config(missing_values)
    if not _missing_values_enabled(cfg):
        return []
    raw_specs = cfg.get("valid_count_features")
    if not (bool(cfg.get("add_valid_count_features", False)) or raw_specs):
        return []
    if raw_specs in (None, "", []):
        return ["factor_valid_count"]
    if isinstance(raw_specs, dict):
        out: list[str] = []
        for name, spec in raw_specs.items():
            feature_name = str(name).strip()
            if not feature_name or spec is False or spec in (None, "", []):
                continue
            if isinstance(spec, dict) and not bool(spec.get("enabled", True)):
                continue
            out.append(feature_name)
        return out
    if isinstance(raw_specs, list):
        names: list[str] = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                raise TypeError("missing_values.valid_count_features list entries must be objects")
            if not bool(spec.get("enabled", True)):
                continue
            name = str(spec.get("name", "")).strip()
            if name:
                names.append(name)
        return names
    raise TypeError("missing_values.valid_count_features must be an object or list")


def _add_missing_value_features(
    df: pd.DataFrame,
    specs: list[tuple[str, str, list[str]]],
) -> pd.DataFrame:
    if not specs:
        return df
    work = df.copy()
    for name, kind, source_cols in specs:
        missing_source_cols = [c for c in source_cols if c not in work.columns]
        if missing_source_cols:
            raise KeyError(f"missing count feature {name} source columns not found: {missing_source_cols}")
        valid_count = work[source_cols].notna().sum(axis=1).astype(float)
        if kind == "valid_count":
            work[name] = valid_count
        else:
            work[name] = float(len(source_cols)) - valid_count
    return work


def _resolve_standardization_config(
    standardization: dict[str, Any] | None,
    *,
    legacy_zscore: bool,
) -> dict[str, Any]:
    if standardization is None:
        return {
            "enabled": bool(legacy_zscore),
            "method": "zscore" if legacy_zscore else "none",
            "mad_scale": 1.4826,
            "robust_fallback": "std",
            "tanh_scale": 3.0,
        }
    if not isinstance(standardization, dict):
        raise TypeError("standardization config must be an object")

    enabled = bool(standardization.get("enabled", True))
    method = str(standardization.get("method", "zscore")).strip().lower()
    aliases = {
        "off": "none",
        "false": "none",
        "disabled": "none",
        "rank": "rank_pct_centered",
        "rank_pct": "rank_pct_centered",
        "rank_pct_center": "rank_pct_centered",
        "robust": "robust_zscore",
        "robust_z": "robust_zscore",
        "tanh": "tanh_zscore",
        "tanh_z": "tanh_zscore",
    }
    method = aliases.get(method, method)
    if not enabled:
        method = "none"
    if method not in {"none", "zscore", "rank_pct_centered", "robust_zscore", "tanh_zscore"}:
        raise ValueError(f"unsupported standardization.method: {method}")

    groupby = str(standardization.get("groupby", "dt")).strip().lower()
    if groupby != "dt":
        raise ValueError("standardization.groupby currently only supports 'dt'")
    stage = str(standardization.get("stage", "after_neutralization")).strip().lower()
    if stage not in {"after_neutralization", "post_neutralization"}:
        raise ValueError("standardization.stage currently only supports 'after_neutralization'")

    robust_cfg = standardization.get("robust", {})
    if robust_cfg is None:
        robust_cfg = {}
    if not isinstance(robust_cfg, dict):
        raise TypeError("standardization.robust must be an object")
    tanh_cfg = standardization.get("tanh", {})
    if tanh_cfg is None:
        tanh_cfg = {}
    if not isinstance(tanh_cfg, dict):
        raise TypeError("standardization.tanh must be an object")

    return {
        "enabled": method != "none",
        "method": method,
        "mad_scale": float(robust_cfg.get("mad_scale", 1.4826)),
        "robust_fallback": str(robust_cfg.get("fallback", "std")).strip().lower(),
        "tanh_scale": float(tanh_cfg.get("scale", 3.0)),
    }


def _iter_existing_label_days(label_root: Path, start: date, end: date) -> Iterable[date]:
    if not label_root.exists():
        return
    days: set[date] = set()
    for path in label_root.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        if start <= day <= end:
            days.add(day)
    for day in sorted(days):
        yield day


def _read_label_day(
    label_root: Path,
    day: date,
    *,
    factor_time: str,
    label_time: str,
    anchor_day: date | None = None,
) -> pd.DataFrame:
    """Read one standard label file and align it to a factor day.

    ``day`` identifies the source label file.  Normally it is also the
    factor day, so the historical behaviour is unchanged.  Research variants
    can supply ``anchor_day`` when a factor day is intentionally trained
    against a later standard execution label.  The label time is always
    filtered on the source file's date before the merge timestamp is re-anchored.
    """
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    path = label_root / month / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    if "trade_time" in df.columns:
        dt_series = pd.to_datetime(df["trade_time"], errors="coerce")
    elif "dt" in df.columns:
        dt_series = pd.to_datetime(df["dt"], errors="coerce")
    else:
        return pd.DataFrame()
    df["dt"] = dt_series

    try:
        label_h, label_m = map(int, label_time.split(":"))
        factor_h, factor_m = map(int, factor_time.split(":"))
    except Exception:
        return pd.DataFrame()
    label_t = dt_time(label_h, label_m)
    df = df[df["dt"].dt.time == label_t]
    if df.empty:
        return df
    if anchor_day is None:
        base_date = df["dt"].dt.normalize()
    else:
        base_date = pd.Series(pd.Timestamp(anchor_day), index=df.index)
    df["dt"] = base_date + pd.Timedelta(hours=factor_h, minutes=factor_m)
    return df


def build_tradable_code_map(
    *,
    raw_data_root: str | Path,
    days: Sequence[date],
    buy_twap_col: str,
    sell_twap_col: str,
    min_amount: float = 0.0,
    min_volume: float = 0.0,
    twap_table: str = "market_cbond.daily_twap",
    asset: str = "cbond",
    pool_cfg: UpstreamPoolConfig | None = None,
) -> dict[date, set[str]]:
    # Compatibility parameters are intentionally ignored here.
    # Eligibility must come only from the T-1 o_0005 allowlist.
    _ = (buy_twap_col, sell_twap_col, min_amount, min_volume, twap_table, asset)
    unique_days = sorted(set(days))
    if not unique_days:
        return {}
    upstream_pool_cfg = pool_cfg or load_upstream_pool_config()

    out: dict[date, set[str]] = {}
    for day in unique_days:
        pool_codes, pool_info = resolve_pool_codes_for_trade_day(
            raw_data_root=raw_data_root,
            trade_day=day,
            pool_cfg=upstream_pool_cfg,
        )
        if bool(pool_info.get("fallback_no_filter", False)):
            raise RuntimeError(
                "[pool_filter] required pool is unavailable; no-filter fallback is disabled: "
                f"trade_day={day:%Y-%m-%d} "
                f"expected_pool_day={pool_info.get('pool_day_expected')} "
                f"reason={pool_info.get('fallback_reason')} "
                f"nearest_pool_day={pool_info.get('nearest_pool_day')}"
            )
        if not pool_codes:
            continue
        out[day] = set(str(code) for code in pool_codes)
    return out


def _split_days(days: list[date], train_ratio: float, val_ratio: float) -> tuple[list[date], list[date], list[date]]:
    n_days = len(days)
    n_train = max(1, int(n_days * train_ratio))
    n_val = max(1, int(n_days * val_ratio))
    if n_train + n_val >= n_days:
        n_val = max(1, n_days - n_train - 1)
    n_test = n_days - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train = max(1, n_train - 1)
    train_days = days[:n_train]
    val_days = days[n_train:n_train + n_val]
    test_days = days[n_train + n_val:]
    return train_days, val_days, test_days


def _apply_factor_groupwise(
    df: pd.DataFrame,
    factor_cols: list[str],
    transform,
) -> pd.DataFrame:
    def _process(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["dt"] = group.name
        for col in factor_cols:
            s = g[col]
            if s.isna().all():
                continue
            g[col] = transform(s)
        return g

    return df.groupby("dt", group_keys=False).apply(_process, include_groups=False)


def _zscore_series(s: pd.Series) -> pd.Series:
    mean = s.mean()
    std = s.std(ddof=0)
    if std > 0:
        return (s - mean) / std
    return s - mean


def _robust_zscore_series(s: pd.Series, *, mad_scale: float, fallback: str) -> pd.Series:
    median = s.median()
    mad = (s - median).abs().median()
    scale = mad_scale * mad
    if scale > 0:
        return (s - median) / scale
    if fallback in {"std", "zscore"}:
        return _zscore_series(s)
    if fallback in {"center", "demean"}:
        return s - median
    if fallback == "zero":
        return s * 0.0
    raise ValueError(f"unsupported robust_zscore fallback: {fallback}")


def _apply_winsor_only(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float | None,
    upper_q: float | None,
) -> pd.DataFrame:
    if lower_q is None and upper_q is None:
        return df

    def _clip(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower_q) if lower_q is not None else None
        hi = s.quantile(upper_q) if upper_q is not None else None
        return s.clip(lower=lo, upper=hi)

    return _apply_factor_groupwise(df, factor_cols, _clip)


def _apply_standardization(
    df: pd.DataFrame,
    factor_cols: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    method = str(config.get("method", "none"))
    if method == "none" or not bool(config.get("enabled", False)):
        return df
    if method == "zscore":
        return _apply_factor_groupwise(df, factor_cols, _zscore_series)
    if method == "rank_pct_centered":
        return _apply_factor_groupwise(
            df,
            factor_cols,
            lambda s: 2.0 * s.rank(pct=True, method="average") - 1.0,
        )
    if method == "robust_zscore":
        mad_scale = float(config.get("mad_scale", 1.4826))
        fallback = str(config.get("robust_fallback", "std")).strip().lower()
        return _apply_factor_groupwise(
            df,
            factor_cols,
            lambda s: _robust_zscore_series(s, mad_scale=mad_scale, fallback=fallback),
        )
    if method == "tanh_zscore":
        scale = float(config.get("tanh_scale", 3.0))
        if scale <= 0:
            raise ValueError("standardization.tanh.scale must be > 0")

        def _tanh(s: pd.Series) -> pd.Series:
            z = _zscore_series(s)
            return pd.Series(np.tanh(z.to_numpy(dtype=float) / scale), index=s.index)

        return _apply_factor_groupwise(df, factor_cols, _tanh)
    raise ValueError(f"unsupported standardization.method: {method}")


def _apply_winsor_zscore(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float | None,
    upper_q: float | None,
    zscore: bool,
) -> pd.DataFrame:
    work = _apply_winsor_only(
        df,
        factor_cols,
        lower_q=lower_q,
        upper_q=upper_q,
    )
    if zscore:
        work = _apply_standardization(
            work,
            factor_cols,
            {"enabled": True, "method": "zscore"},
        )
    return work


def _apply_factor_preprocess(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float | None,
    upper_q: float | None,
    zscore: bool,
    neutralizer: FactorNeutralizer | None = None,
    standardization: dict[str, Any] | None = None,
) -> pd.DataFrame:
    standardization_cfg = _resolve_standardization_config(
        standardization,
        legacy_zscore=zscore,
    )
    work = _apply_winsor_only(
        df,
        factor_cols,
        lower_q=lower_q,
        upper_q=upper_q,
    )
    if neutralizer is not None and neutralizer.enabled:
        work = neutralizer.apply(work, factor_cols)
    return _apply_standardization(work, factor_cols, standardization_cfg)


def describe_standardization(
    standardization: dict[str, Any] | None,
    *,
    legacy_zscore: bool,
) -> dict[str, Any]:
    return _resolve_standardization_config(standardization, legacy_zscore=legacy_zscore)


def _read_factor_frame_for_dataset(
    *,
    factor_store: FactorStore,
    day: date,
    raw_cols: Sequence[str],
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    """Read one unprocessed factor-day frame with aliases resolved."""
    fdf = factor_store.read_day(day)
    if fdf.empty:
        return pd.DataFrame()
    if not isinstance(fdf.index, pd.MultiIndex):
        fdf = fdf.reset_index().set_index(["dt", "code"])
    fdf = fdf.reset_index()
    missing_alias_sources = [source for source in aliases.values() if source not in fdf.columns]
    if missing_alias_sources:
        return pd.DataFrame()
    for alias, source in aliases.items():
        fdf[alias] = fdf[source]
    missing_cols = [column for column in raw_cols if column not in fdf.columns]
    if missing_cols:
        return pd.DataFrame()
    return fdf


def _build_temporal_factor_frame(
    *,
    current: pd.DataFrame,
    previous: pd.DataFrame,
    raw_cols: Sequence[str],
    spec: TemporalFactorLagSpec,
) -> pd.DataFrame:
    """Join raw T and P(T) factors by code and emit a stable temporal schema."""
    current_cols = ["dt", "code", *raw_cols]
    previous_cols = ["code", *raw_cols]
    current_part = current[current_cols].copy()
    previous_part = previous[previous_cols].copy()
    if current_part["code"].duplicated().any() or previous_part["code"].duplicated().any():
        raise ValueError("temporal_factor_lag requires one factor row per code per day")

    merged = current_part.merge(previous_part, on="code", how="inner", suffixes=("__current", "__previous"))
    if merged.empty:
        return merged

    out = merged[["dt", "code"]].copy()
    for kind in spec.outputs:
        for column in raw_cols:
            current_col = f"{column}__current"
            previous_col = f"{column}__previous"
            if kind == "t0":
                out[f"{column}__t0"] = merged[current_col]
            elif kind == "lag1":
                out[f"{column}__lag1"] = merged[previous_col]
            elif kind == "diff1":
                out[f"{column}__diff1"] = merged[current_col] - merged[previous_col]
            else:  # Defensive: TemporalFactorLagSpec validates the option.
                raise ValueError(f"unsupported temporal factor output: {kind}")
    return out


def build_dataset(
    *,
    factor_store: FactorStore,
    label_root: Path,
    days: Sequence[date],
    factor_cols: list[str],
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    factor_time: str,
    label_time: str,
    require_label: bool = True,
    tradable_code_map: dict[date, set[str]] | None = None,
    tradable_strict: bool = False,
    neutralizer: FactorNeutralizer | None = None,
    factor_aliases: dict[str, str] | None = None,
    standardization: dict[str, Any] | None = None,
    raw_factor_cols: list[str] | None = None,
    preprocess_factor_cols: list[str] | None = None,
    missing_values: dict[str, Any] | None = None,
    sample_weight: dict[str, Any] | None = None,
    label_day_by_factor_day: Mapping[date, date | None] | None = None,
    read_label_when_not_required: bool = True,
    apply_tradable_filter_when_label_not_required: bool = False,
    temporal_factor_lag: TemporalFactorLagSpec | None = None,
    previous_factor_day_by_factor_day: Mapping[date, date | None] | None = None,
) -> SplitData:
    aliases = {str(k): str(v) for k, v in (factor_aliases or {}).items()}
    raw_cols = list(raw_factor_cols or factor_cols)
    preprocess_cols = list(preprocess_factor_cols or factor_cols)
    missing_cfg = _normalise_missing_values_config(missing_values)
    missing_enabled = _missing_values_enabled(missing_cfg)
    if temporal_factor_lag is not None and missing_enabled:
        raise ValueError("temporal_factor_lag does not support missing_values yet")
    if temporal_factor_lag is not None and previous_factor_day_by_factor_day is None:
        raise ValueError("temporal_factor_lag requires previous_factor_day_by_factor_day")
    keep_nan = bool(missing_cfg.get("keep_nan", missing_enabled))
    min_available_raw = missing_cfg.get("min_available_factors")
    min_available_factors = int(min_available_raw) if min_available_raw is not None else len(raw_cols)
    if missing_enabled and (min_available_factors < 0 or min_available_factors > len(raw_cols)):
        raise ValueError(
            "missing_values.min_available_factors must be between 0 and the number "
            f"of raw factors ({len(raw_cols)})"
        )
    missing_feature_specs = _iter_missing_feature_specs(missing_cfg, raw_cols)
    frames: list[pd.DataFrame] = []
    for day in days:
        fdf = _read_factor_frame_for_dataset(
            factor_store=factor_store,
            day=day,
            raw_cols=raw_cols,
            aliases=aliases,
        )
        if fdf.empty:
            continue
        if temporal_factor_lag is not None:
            previous_day = previous_factor_day_by_factor_day.get(day)
            if previous_day is None:
                # Strict immediate-prior policy: do not bridge a missing
                # calendar day with any earlier factor snapshot.
                continue
            previous_fdf = _read_factor_frame_for_dataset(
                factor_store=factor_store,
                day=previous_day,
                raw_cols=raw_cols,
                aliases=aliases,
            )
            if previous_fdf.empty:
                continue
            fdf = _build_temporal_factor_frame(
                current=fdf,
                previous=previous_fdf,
                raw_cols=raw_cols,
                spec=temporal_factor_lag,
            )
            if fdf.empty:
                continue
        label_day = label_day_by_factor_day.get(day, day) if label_day_by_factor_day is not None else day
        if not require_label and not read_label_when_not_required:
            label_df = pd.DataFrame(columns=["dt", "code", "y"])
        elif label_day is None:
            label_df = pd.DataFrame(columns=["dt", "code", "y"])
        elif label_day == day:
            # Preserve the legacy call shape for the default same-day contract.
            label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
        else:
            label_df = _read_label_day(
                label_root,
                label_day,
                factor_time=factor_time,
                label_time=label_time,
                anchor_day=day,
            )
        if label_df.empty or "dt" not in label_df.columns:
            if require_label:
                continue
            label_df = pd.DataFrame(columns=["dt", "code", "y"])
        label_df = label_df[["dt", "code", "y"]].dropna()
        if label_df.empty and not require_label:
            merged = fdf.copy()
            merged["y"] = np.nan
        else:
            merged = fdf.merge(label_df, on=["dt", "code"], how="inner")
        if merged.empty:
            continue
        # The historical test path obtains its score universe from the
        # same-day label inner join, then applies the T-1 allowlist below.
        # A causal score-only path deliberately does not open that label, so
        # it must opt in explicitly to applying the same existing allowlist.
        # Keep the default false: ordinary callers retain their prior contract.
        if tradable_code_map is not None and (
            require_label or apply_tradable_filter_when_label_not_required
        ):
            allowed_codes = tradable_code_map.get(day)
            if not allowed_codes:
                if tradable_strict:
                    continue
            else:
                merged = merged[merged["code"].astype(str).isin(allowed_codes)]
                if merged.empty:
                    continue
        if missing_enabled:
            available_count = merged[raw_cols].notna().sum(axis=1)
            merged = merged[available_count >= min_available_factors]
            if merged.empty:
                continue
            merged = _add_missing_value_features(merged, missing_feature_specs)
            missing_model_cols = [c for c in factor_cols if c not in merged.columns]
            if missing_model_cols:
                raise KeyError(f"model feature columns not found after missing feature processing: {missing_model_cols}")
            if require_label:
                merged = merged.dropna(subset=["y"])
            if not keep_nan:
                drop_subset = list(factor_cols) + (["y"] if require_label else [])
                merged = merged.dropna(subset=drop_subset)
        elif require_label:
            merged = merged.dropna(subset=factor_cols + ["y"])
        else:
            merged = merged.dropna(subset=factor_cols)
        if merged.empty:
            continue
        counts = merged.groupby("dt")["code"].transform("size")
        merged = merged[counts >= min_count]
        if merged.empty:
            continue
        row_weight = _apply_sample_weight_config(merged, sample_weight)
        if row_weight is not None:
            merged = merged.copy()
            merged["_sample_weight"] = row_weight
            frame_cols = ["dt", "code"] + factor_cols + ["y", "_sample_weight"]
        else:
            frame_cols = ["dt", "code"] + factor_cols + ["y"]
        frames.append(merged[frame_cols])

    if not frames:
        empty = pd.DataFrame(columns=["dt", "code"] + factor_cols + ["y"])
        return SplitData(empty[factor_cols], empty["y"], empty["dt"], empty["code"])

    data = pd.concat(frames, ignore_index=True)
    data = _apply_factor_preprocess(
        data,
        preprocess_cols,
        lower_q=winsor_lower,
        upper_q=winsor_upper,
        zscore=zscore,
        neutralizer=neutralizer,
        standardization=standardization,
    )
    return SplitData(
        x=data[factor_cols].copy(),
        y=data["y"].copy(),
        dt=data["dt"].copy(),
        code=data["code"].copy(),
        sample_weight=data["_sample_weight"].copy() if "_sample_weight" in data.columns else None,
    )


def _ic_by_day(df: pd.DataFrame, factor_col: str) -> pd.Series:
    def _calc(group: pd.DataFrame) -> float:
        g = group[[factor_col, "y"]].dropna()
        if len(g) < 2:
            return np.nan
        return g[factor_col].corr(g["y"], method="pearson")
    return df.groupby("dt").apply(_calc, include_groups=False)


def _rank_ic_by_day(df: pd.DataFrame, factor_col: str) -> pd.Series:
    def _calc(group: pd.DataFrame) -> float:
        g = group[[factor_col, "y"]].dropna()
        if len(g) < 2:
            return np.nan
        return g[factor_col].corr(g["y"], method="spearman")
    return df.groupby("dt").apply(_calc, include_groups=False)


def _build_day_group_indices(dt: pd.Series) -> list[np.ndarray]:
    if dt is None or len(dt) == 0:
        return []
    dt_arr = pd.to_datetime(dt, errors="coerce").to_numpy()
    valid_mask = ~pd.isna(dt_arr)
    if not valid_mask.any():
        return []
    valid_pos = np.where(valid_mask)[0]
    dt_valid = dt_arr[valid_mask]
    _, inv = np.unique(dt_valid, return_inverse=True)
    groups: list[np.ndarray] = []
    for g in range(int(inv.max()) + 1):
        pos = valid_pos[inv == g]
        if pos.size >= 2:
            groups.append(pos.astype(np.int64, copy=False))
    return groups


def _mean_abs_ic_by_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[np.ndarray],
    *,
    eps: float = 1e-12,
) -> float:
    if not groups:
        return float("nan")
    vals: list[float] = []
    for idx in groups:
        y = y_true[idx]
        p = y_pred[idx]
        if y.size < 2:
            continue
        yc = y - np.mean(y)
        pc = p - np.mean(p)
        b = float(np.dot(yc, yc))
        c = float(np.dot(pc, pc))
        if b <= eps or c <= eps:
            continue
        a = float(np.dot(yc, pc))
        r = a / np.sqrt(b * c + eps)
        if np.isfinite(r):
            vals.append(abs(float(r)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _mean_pearson_ic_by_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[np.ndarray],
    *,
    eps: float = 1e-12,
) -> float:
    """Equal-weighted, within-day Pearson IC for LightGBM early stopping."""

    if not groups:
        return float("nan")
    values: list[float] = []
    for idx in groups:
        y = y_true[idx]
        p = y_pred[idx]
        if y.size < 2:
            continue
        yc = y - np.mean(y)
        pc = p - np.mean(p)
        y_ss = float(np.dot(yc, yc))
        p_ss = float(np.dot(pc, pc))
        if y_ss <= eps or p_ss <= eps:
            continue
        corr = float(np.dot(yc, pc)) / np.sqrt(y_ss * p_ss + eps)
        if np.isfinite(corr):
            values.append(corr)
    if not values:
        return float("nan")
    return float(np.mean(values))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks, equivalent to pandas rank(method='average')."""
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(arr, kind="mergesort")
    sorted_arr = arr[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _mean_rank_ic_by_groups(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[np.ndarray],
    *,
    eps: float = 1e-12,
) -> float:
    if not groups:
        return float("nan")
    vals: list[float] = []
    for idx in groups:
        y = y_true[idx]
        p = y_pred[idx]
        if y.size < 2:
            continue
        yr = _average_ranks(y)
        pr = _average_ranks(p)
        yc = yr - np.mean(yr)
        pc = pr - np.mean(pr)
        b = float(np.dot(yc, yc))
        c = float(np.dot(pc, pc))
        if b <= eps or c <= eps:
            continue
        a = float(np.dot(yc, pc))
        r = a / np.sqrt(b * c + eps)
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _make_abs_ic_objective(
    dt: pd.Series,
    *,
    eps: float = 1e-12,
) -> tuple[callable, list[np.ndarray]]:
    groups = _build_day_group_indices(dt)

    def _objective(y_true, y_pred):
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
        grad = np.zeros_like(p, dtype=float)
        hess = np.ones_like(p, dtype=float)
        if not groups:
            return grad, hess

        n_groups = float(len(groups))
        for idx in groups:
            yg = y[idx]
            pg = p[idx]
            if yg.size < 2:
                continue
            yc = yg - np.mean(yg)
            pc = pg - np.mean(pg)
            b = float(np.dot(yc, yc))
            c = float(np.dot(pc, pc))
            if b <= eps:
                continue
            if c <= eps:
                # Correlation is undefined when pred variance is near zero.
                # Use a fallback gradient to break the zero-gradient dead start.
                r = 0.0
                dr_dp = yc / np.sqrt(b + eps)
            else:
                a = float(np.dot(yc, pc))
                denom = np.sqrt(b * c + eps)
                r = a / denom
                dr_dp = (yc / denom) - (r * pc / (c + eps))
            # Subgradient of |r|; choose +1 at r == 0.
            dabs_dr = 1.0 if r >= 0 else -1.0
            grad[idx] += -(dabs_dr * dr_dp) / n_groups
        return grad, hess

    return _objective, groups


def _dir_acc(y: np.ndarray, pred: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float((np.sign(y) == np.sign(pred)).mean())


def _bin_dir_by_day(df: pd.DataFrame, bins: int) -> list[tuple[int, float, int]]:
    def _calc(group: pd.DataFrame) -> list[tuple[int, float, int]]:
        g = group[["pred", "y"]].dropna()
        n = len(g)
        if n < bins:
            return []
        try:
            labels = pd.qcut(g["pred"].rank(pct=True, method="average"), bins, labels=False, duplicates="drop")
        except Exception:
            return []
        g = g.assign(_bin=labels)
        out = []
        for b, gg in g.groupby("_bin"):
            if gg.empty:
                continue
            acc = float((np.sign(gg["pred"]) == np.sign(gg["y"])).mean())
            out.append((int(b), acc, int(len(gg))))
        return out

    all_rows: dict[int, list[tuple[float, int]]] = {}
    for _, group in df.groupby("dt"):
        for b, acc, n in _calc(group):
            all_rows.setdefault(b, []).append((acc, n))
    results = []
    for b in sorted(all_rows.keys()):
        vals = all_rows[b]
        if not vals:
            continue
        # average accuracy across days (equal weight by day)
        accs = [v[0] for v in vals]
        total_n = sum(v[1] for v in vals)
        results.append((b, float(np.mean(accs)), total_n))
    return results


def evaluate_metrics(
    x: pd.DataFrame,
    y: pd.Series,
    dt: pd.Series,
    pred: np.ndarray,
    bins: int,
) -> dict:
    if x.empty:
        return {
            "mse": float("nan"),
            "r2": float("nan"),
            "dir": float("nan"),
            "ic_mean": float("nan"),
            "ic_ir": float("nan"),
            "rank_ic_mean": float("nan"),
            "rank_ic_ir": float("nan"),
            "bin_dir": [],
        }
    from sklearn.metrics import mean_squared_error, r2_score

    mse = float(mean_squared_error(y, pred))
    r2 = float(r2_score(y, pred))
    dir_acc = _dir_acc(y.to_numpy(), pred)

    df = pd.DataFrame({"dt": dt, "y": y, "pred": pred})
    ic = _ic_by_day(df, "pred").dropna()
    rank_ic = _rank_ic_by_day(df, "pred").dropna()

    ic_mean = float(ic.mean()) if not ic.empty else float("nan")
    ic_ir = float(ic.mean() / ic.std(ddof=0)) if ic.std(ddof=0) else float("nan")
    rank_ic_mean = float(rank_ic.mean()) if not rank_ic.empty else float("nan")
    rank_ic_ir = float(rank_ic.mean() / rank_ic.std(ddof=0)) if rank_ic.std(ddof=0) else float("nan")

    bin_dir = _bin_dir_by_day(df, bins=bins)

    return {
        "mse": mse,
        "r2": r2,
        "dir": dir_acc,
        "ic_mean": ic_mean,
        "ic_ir": ic_ir,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
        "bin_dir": bin_dir,
    }


def train_lgbm(
    *,
    train: SplitData,
    val: SplitData,
    lgbm_params: dict,
    early_stopping_rounds: int | None = None,
    loss_mode: str = "mse",
    init_model: object | str | Path | None = None,
    label_target_transform: LabelTargetTransformSpec | None = None,
    early_stopping_metric: str = "rank_ic",
) -> tuple[object, dict]:
    if lgb is None:
        detail = ""
        if _LIGHTGBM_IMPORT_ERROR is not None:
            detail = f" ({type(_LIGHTGBM_IMPORT_ERROR).__name__}: {_LIGHTGBM_IMPORT_ERROR})"
        raise RuntimeError(f"lightgbm is not installed{detail}")
    params = dict(lgbm_params)
    mode = str(loss_mode or "mse").lower()
    target_spec = label_target_transform or LabelTargetTransformSpec()
    train_fit, train_target_audit = prepare_lgbm_target_split(
        train,
        target_spec,
        apply_loss_day_mass=True,
    )
    val_fit, val_target_audit = prepare_lgbm_target_split(
        val,
        target_spec,
        apply_loss_day_mass=False,
    )
    train_y = train_fit.y
    val_y = val_fit.y
    requested_metric = str(early_stopping_metric or "rank_ic").strip().lower()
    metric_aliases = {
        "rank": "rank_ic",
        "rank_ic": "rank_ic",
        "pearson": "pearson_ic",
        "pearson_ic": "pearson_ic",
        "ic": "pearson_ic",
        "abs": "abs_ic",
        "abs_ic": "abs_ic",
    }
    if requested_metric not in metric_aliases:
        raise ValueError(
            "early_stopping_metric must be rank_ic, pearson_ic, or abs_ic"
        )
    eval_metric_name = metric_aliases[requested_metric]
    if eval_metric_name == "pearson_ic":
        # LightGBM otherwise injects regression l2 beside the custom metric;
        # disabling that default keeps this explicit opt-in stopping rule tied
        # to the requested equal-day Pearson IC alone.
        params.setdefault("metric", "None")
    log_eval_period = max(1, int(params.pop("log_eval_period", 10)))
    train_groups: list[np.ndarray] = []
    val_groups: list[np.ndarray] = []
    train_rank_groups = _build_day_group_indices(train_fit.dt)
    val_rank_groups = _build_day_group_indices(val_fit.dt)
    if mode in {"ic_abs", "abs_ic", "icabs"}:
        objective_fn, train_groups = _make_abs_ic_objective(train_fit.dt)
        params["objective"] = objective_fn
        eval_metric_name = "abs_ic"
        val_groups = _build_day_group_indices(val_fit.dt)
    gpu_requested = _lgbm_gpu_requested(params)
    model = lgb.LGBMRegressor(**params)
    history: list[dict] = []
    sample_weight = None
    if train_fit.sample_weight is not None:
        sample_weight_arr = pd.to_numeric(train_fit.sample_weight, errors="coerce").to_numpy(dtype=float)
        if sample_weight_arr.shape[0] != train_y.shape[0]:
            raise ValueError(
                "train sample_weight length mismatch: "
                f"weights={sample_weight_arr.shape[0]} labels={train_y.shape[0]}"
            )
        sample_weight_arr = np.where(np.isfinite(sample_weight_arr), sample_weight_arr, 1.0)
        if np.any(sample_weight_arr <= 0):
            raise ValueError("train sample_weight values must be > 0")
        sample_weight = sample_weight_arr

    def _eval_rank_ic(y_true, y_pred):
        if y_true is None or y_pred is None:
            return ("rank_ic", 0.0, True)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape[0] == train_y.shape[0]:
            groups = train_rank_groups
        else:
            groups = val_rank_groups
        val_ic = _mean_rank_ic_by_groups(y_true, y_pred, groups)
        if not np.isfinite(val_ic):
            val_ic = 0.0
        return ("rank_ic", val_ic, True)

    def _eval_pearson_ic(y_true, y_pred):
        if y_true is None or y_pred is None:
            return ("pearson_ic", 0.0, True)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        # Pearson IC stopping evaluates validation only (see _fit_once), so
        # there is no ambiguous train/validation inference from row counts.
        # This matters when the two splits happen to have the same number of
        # rows but different per-day group boundaries.
        val_ic = _mean_pearson_ic_by_groups(y_true, y_pred, val_rank_groups)
        if not np.isfinite(val_ic):
            val_ic = 0.0
        return ("pearson_ic", val_ic, True)

    def _eval_abs_ic(y_true, y_pred):
        if y_true is None or y_pred is None:
            return ("abs_ic", 0.0, True)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape[0] == train_y.shape[0]:
            groups = train_groups if train_groups else _build_day_group_indices(train_fit.dt)
        else:
            groups = val_groups if val_groups else _build_day_group_indices(val_fit.dt)
        val_abs_ic = _mean_abs_ic_by_groups(y_true, y_pred, groups)
        if not np.isfinite(val_abs_ic):
            val_abs_ic = 0.0
        return ("abs_ic", float(val_abs_ic), True)

    def _record_callback(env):
        if env.model is None:
            return
        iteration = env.iteration + 1
        train_scores = []
        val_scores = []
        for data_name, metric_name, value, _ in env.evaluation_result_list:
            if metric_name != eval_metric_name:
                continue
            if data_name == "training":
                train_scores.append(value)
            elif data_name in ("valid_0", "valid_1", "valid"):
                val_scores.append(value)
        train_metric = float(train_scores[-1]) if train_scores else float("nan")
        val_metric = float(val_scores[-1]) if val_scores else float("nan")
        history.append(
            {
                "iteration": iteration,
                "train_rank_ic": train_metric if eval_metric_name == "rank_ic" else float("nan"),
                "val_rank_ic": val_metric if eval_metric_name == "rank_ic" else float("nan"),
                "train_pearson_ic": train_metric if eval_metric_name == "pearson_ic" else float("nan"),
                "val_pearson_ic": val_metric if eval_metric_name == "pearson_ic" else float("nan"),
                "train_abs_ic": train_metric if eval_metric_name == "abs_ic" else float("nan"),
                "val_abs_ic": val_metric if eval_metric_name == "abs_ic" else float("nan"),
                "train_r2": float("nan"),
                "val_r2": float("nan"),
            }
        )
        if iteration % log_eval_period != 0:
            return
        metric_label = eval_metric_name
        print(
            f"iter {iteration:03d} "
            f"train_{metric_label}={train_metric:.4f} val_{metric_label}={val_metric:.4f}"
        )
    def _fit_once(estimator) -> None:
        base_fit_kwargs = {}
        if init_model is not None:
            base_fit_kwargs["init_model"] = init_model
        if sample_weight is not None:
            base_fit_kwargs["sample_weight"] = sample_weight
        fit_kwargs = dict(base_fit_kwargs)
        if early_stopping_rounds is not None and val_fit.x is not None and not val_fit.x.empty:
            if eval_metric_name == "rank_ic":
                eval_metric = _eval_rank_ic
            elif eval_metric_name == "pearson_ic":
                eval_metric = _eval_pearson_ic
            else:
                eval_metric = _eval_abs_ic
            # The explicit Pearson-IC candidate must stop on validation IC
            # alone.  Supplying the train set as a second eval set gives the
            # sklearn custom-metric callback no dataset identity; row-count
            # inference is unsafe when train and validation sizes coincide.
            eval_set = [(val_fit.x, val_y)] if eval_metric_name == "pearson_ic" else [
                (train_fit.x, train_y),
                (val_fit.x, val_y),
            ]
            fit_kwargs = {
                **base_fit_kwargs,
                "eval_set": eval_set,
                "eval_metric": eval_metric,
            }
            # lightgbm sklearn API changed early stopping signature
            try:
                estimator.fit(
                    train_fit.x,
                    train_y,
                    **fit_kwargs,
                    early_stopping_rounds=int(early_stopping_rounds),
                    verbose=False,
                    callbacks=[_record_callback],
                )
                return
            except TypeError:
                callbacks = []
                if hasattr(lgb, "early_stopping"):
                    callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
                if hasattr(lgb, "record_evaluation"):
                    eval_result: dict = {}
                    callbacks.append(lgb.record_evaluation(eval_result))
                callbacks.append(lambda env: _record_callback(env))
                try:
                    estimator.fit(train_fit.x, train_y, **fit_kwargs, callbacks=callbacks)
                except TypeError:
                    # Older sklearn wrappers may not accept init_model.
                    fit_kwargs.pop("init_model", None)
                    estimator.fit(train_fit.x, train_y, **fit_kwargs, callbacks=callbacks)
                return
        # no early stopping
        try:
            estimator.fit(train_fit.x, train_y, **base_fit_kwargs)
        except TypeError:
            base_fit_kwargs.pop("init_model", None)
            estimator.fit(train_fit.x, train_y)

    try:
        _fit_once(model)
    except Exception as exc:
        if not (gpu_requested and _looks_like_lgbm_gpu_error(exc)):
            raise
        cpu_params = _lgbm_cpu_params(params)
        print(
            "[LightGBM] GPU unavailable, fallback to CPU:",
            f"{type(exc).__name__}: {exc}",
        )
        history.clear()
        model = lgb.LGBMRegressor(**cpu_params)
        _fit_once(model)
        params = cpu_params

    return model, params | {
        "history": history,
        "early_stopping_metric": eval_metric_name,
        "label_target_transform": {
            "mode": target_spec.mode,
            "ddof": int(target_spec.ddof),
            "min_std": float(target_spec.min_std),
            "loss_day_mass": target_spec.loss_day_mass,
            "train": train_target_audit,
            "val": val_target_audit,
        },
    }

