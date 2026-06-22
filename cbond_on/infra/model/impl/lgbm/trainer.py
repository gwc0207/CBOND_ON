
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def _normalise_missing_values_config(missing_values: dict[str, Any] | None) -> dict[str, Any]:
    if missing_values in (None, "", []):
        return {}
    if not isinstance(missing_values, dict):
        raise TypeError("missing_values config must be an object")
    return dict(missing_values)


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


def _read_label_day(label_root: Path, day: date, *, factor_time: str, label_time: str) -> pd.DataFrame:
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
    base_date = df["dt"].dt.normalize()
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
) -> SplitData:
    aliases = {str(k): str(v) for k, v in (factor_aliases or {}).items()}
    raw_cols = list(raw_factor_cols or factor_cols)
    preprocess_cols = list(preprocess_factor_cols or factor_cols)
    missing_cfg = _normalise_missing_values_config(missing_values)
    missing_enabled = _missing_values_enabled(missing_cfg)
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
        fdf = factor_store.read_day(day)
        if fdf.empty:
            continue
        if not isinstance(fdf.index, pd.MultiIndex):
            fdf = fdf.reset_index().set_index(["dt", "code"])
        fdf = fdf.reset_index()
        missing_alias_sources = [source for source in aliases.values() if source not in fdf.columns]
        if missing_alias_sources:
            continue
        for alias, source in aliases.items():
            fdf[alias] = fdf[source]
        # Backward compatibility: old factor files may miss newly added columns.
        # Skip these days instead of raising KeyError in dropna(subset=...).
        missing_cols = [c for c in raw_cols if c not in fdf.columns]
        if missing_cols:
            continue
        label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
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
        if require_label and tradable_code_map is not None:
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
        frames.append(merged[["dt", "code"] + factor_cols + ["y"]])

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
) -> tuple[object, dict]:
    if lgb is None:
        detail = ""
        if _LIGHTGBM_IMPORT_ERROR is not None:
            detail = f" ({type(_LIGHTGBM_IMPORT_ERROR).__name__}: {_LIGHTGBM_IMPORT_ERROR})"
        raise RuntimeError(f"lightgbm is not installed{detail}")
    params = dict(lgbm_params)
    mode = str(loss_mode or "mse").lower()
    eval_metric_name = "rank_ic"
    log_eval_period = max(1, int(params.pop("log_eval_period", 10)))
    train_groups: list[np.ndarray] = []
    val_groups: list[np.ndarray] = []
    train_rank_groups = _build_day_group_indices(train.dt)
    val_rank_groups = _build_day_group_indices(val.dt)
    if mode in {"ic_abs", "abs_ic", "icabs"}:
        objective_fn, train_groups = _make_abs_ic_objective(train.dt)
        params["objective"] = objective_fn
        eval_metric_name = "abs_ic"
        val_groups = _build_day_group_indices(val.dt)
    gpu_requested = _lgbm_gpu_requested(params)
    model = lgb.LGBMRegressor(**params)
    history: list[dict] = []

    def _eval_rank_ic(y_true, y_pred):
        if y_true is None or y_pred is None:
            return ("rank_ic", 0.0, True)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape[0] == train.y.shape[0]:
            groups = train_rank_groups
        else:
            groups = val_rank_groups
        val_ic = _mean_rank_ic_by_groups(y_true, y_pred, groups)
        if not np.isfinite(val_ic):
            val_ic = 0.0
        return ("rank_ic", val_ic, True)

    def _eval_abs_ic(y_true, y_pred):
        if y_true is None or y_pred is None:
            return ("abs_ic", 0.0, True)
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if y_true.shape[0] == train.y.shape[0]:
            groups = train_groups if train_groups else _build_day_group_indices(train.dt)
        else:
            groups = val_groups if val_groups else _build_day_group_indices(val.dt)
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
                "train_abs_ic": train_metric if eval_metric_name == "abs_ic" else float("nan"),
                "val_abs_ic": val_metric if eval_metric_name == "abs_ic" else float("nan"),
                "train_r2": float("nan"),
                "val_r2": float("nan"),
            }
        )
        if iteration % log_eval_period != 0:
            return
        metric_label = "rank_ic" if eval_metric_name == "rank_ic" else "abs_ic"
        print(
            f"iter {iteration:03d} "
            f"train_{metric_label}={train_metric:.4f} val_{metric_label}={val_metric:.4f}"
        )
    def _fit_once(estimator) -> None:
        base_fit_kwargs = {}
        if init_model is not None:
            base_fit_kwargs["init_model"] = init_model
        fit_kwargs = dict(base_fit_kwargs)
        if early_stopping_rounds is not None and val.x is not None and not val.x.empty:
            fit_kwargs = {
                **base_fit_kwargs,
                "eval_set": [(train.x, train.y), (val.x, val.y)],
                "eval_metric": _eval_abs_ic if eval_metric_name == "abs_ic" else _eval_rank_ic,
            }
            # lightgbm sklearn API changed early stopping signature
            try:
                estimator.fit(
                    train.x,
                    train.y,
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
                    estimator.fit(train.x, train.y, **fit_kwargs, callbacks=callbacks)
                except TypeError:
                    # Older sklearn wrappers may not accept init_model.
                    fit_kwargs.pop("init_model", None)
                    estimator.fit(train.x, train.y, **fit_kwargs, callbacks=callbacks)
                return
        # no early stopping
        try:
            estimator.fit(train.x, train.y, **base_fit_kwargs)
        except TypeError:
            base_fit_kwargs.pop("init_model", None)
            estimator.fit(train.x, train.y)

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

    return model, params | {"history": history}

