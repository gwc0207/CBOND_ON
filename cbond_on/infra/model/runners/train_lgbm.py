
from __future__ import annotations

import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.naming import make_window_label
from cbond_on.core.trading_days import list_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.model.wandb_utils import init_wandb_logger
from cbond_on.infra.model.score_io import load_scores_by_date, write_scores_by_date
from cbond_on.infra.model.preprocess_config import parse_winsor_bounds
from cbond_on.infra.model.neutralization import build_neutralizer
from cbond_on.infra.model.score_guard import (
    score_guard_bin_stats,
    score_guard_flags,
    score_guard_stats,
)
from cbond_on.infra.model.impl.lgbm.trainer import (
    SplitData,
    build_dataset,
    build_tradable_code_map,
    describe_standardization,
    evaluate_metrics,
    missing_value_feature_columns,
    train_lgbm,
    _iter_existing_label_days,
    _split_days,
)
from cbond_on.infra.factors.tail_features import tail_feature_output_columns, tail_features_enabled


def _load_model_config(path: Path | None) -> dict:
    if path is None:
        return load_config_file("models/lgbm/lgbm_factor_MSE")
    return load_config_file(path)


def _parse_factor_aliases(cfg: dict) -> dict[str, str]:
    raw = cfg.get("factor_aliases", {})
    if raw in (None, "", []):
        return {}
    if not isinstance(raw, dict):
        raise TypeError("lgbm config factor_aliases must be an object: {alias: source}")
    aliases: dict[str, str] = {}
    for alias_raw, source_raw in raw.items():
        alias = str(alias_raw or "").strip()
        source = str(source_raw or "").strip()
        if not alias or not source:
            raise ValueError("lgbm config factor_aliases entries must have non-empty alias and source")
        if alias == source:
            raise ValueError(f"lgbm config factor_aliases alias must differ from source: {alias}")
        aliases[alias] = source
    return aliases


def _select_factor_cols(sample: pd.DataFrame, cfg: dict, factor_aliases: dict[str, str] | None = None) -> list[str]:
    cols = cfg.get("factors")
    if cols:
        base = [str(c) for c in cols]
    else:
        exclude = {"dt", "code"}
        base = [c for c in sample.columns if c not in exclude]
    extra = [str(c) for c in (cfg.get("extra_factors") or []) if str(c).strip()]
    if tail_features_enabled(cfg.get("tail_features")):
        extra.extend(
            tail_feature_output_columns(
                cfg.get("tail_features"),
                source_columns=base,
            )
        )
    seen: set[str] = set()
    out: list[str] = []
    for col in [*base, *extra, *list((factor_aliases or {}).keys())]:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _resolve_missing_values_config(cfg: dict) -> dict:
    fe_cfg = cfg.get("feature_engineering", {})
    if fe_cfg is None:
        fe_cfg = {}
    if not isinstance(fe_cfg, dict):
        raise TypeError("feature_engineering must be an object")
    raw = fe_cfg.get("missing_values", cfg.get("missing_values", {}))
    if raw in (None, "", []):
        return {}
    if not isinstance(raw, dict):
        raise TypeError("missing_values must be an object")
    return dict(raw)


def _resolve_feature_contribution_config(cfg: dict) -> dict:
    fe_cfg = cfg.get("feature_engineering", {})
    if fe_cfg is None:
        fe_cfg = {}
    if not isinstance(fe_cfg, dict):
        raise TypeError("feature_engineering must be an object")
    raw = fe_cfg.get("feature_contribution", cfg.get("feature_contribution", {}))
    if raw in (None, "", []):
        return {}
    if not isinstance(raw, dict):
        raise TypeError("feature_contribution must be an object")
    return dict(raw)


def _iter_feature_contribution_entries(contribution_cfg: dict) -> list[tuple[str, float]]:
    entries: list[tuple[str, float]] = []
    for name, value in (contribution_cfg.get("values") or {}).items():
        entries.append((str(name), float(value)))
    for group in contribution_cfg.get("groups") or []:
        if not isinstance(group, dict):
            raise TypeError("feature_contribution.groups entries must be objects")
        value = float(group.get("value", contribution_cfg.get("default", 1.0)))
        for name in group.get("features") or []:
            entries.append((str(name), value))
    return entries


def _feature_contribution_values(feature_cols: list[str], contribution_cfg: dict) -> list[float] | None:
    if not contribution_cfg or not bool(contribution_cfg.get("enabled", True)):
        return None
    default = float(contribution_cfg.get("default", 1.0))
    by_name: dict[str, float] = {}
    for name, value in _iter_feature_contribution_entries(contribution_cfg):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"feature_contribution value must be finite and >= 0: {name}={value}")
        by_name[name] = value
    if not np.isfinite(default) or default < 0:
        raise ValueError(f"feature_contribution default must be finite and >= 0: {default}")
    return [float(by_name.get(col, default)) for col in feature_cols]


def _feature_contribution_summary(feature_cols: list[str], contribution_cfg: dict) -> dict:
    values = _feature_contribution_values(feature_cols, contribution_cfg)
    if values is None:
        return {"enabled": False, "default": 1.0, "weighted_count": 0, "missing_features": []}
    default = float(contribution_cfg.get("default", 1.0))
    feature_set = set(feature_cols)
    configured = [name for name, _ in _iter_feature_contribution_entries(contribution_cfg)]
    missing = sorted({name for name in configured if name not in feature_set})
    weighted = [col for col, value in zip(feature_cols, values, strict=True) if abs(float(value) - default) > 1e-12]
    return {
        "enabled": True,
        "default": default,
        "weighted_count": len(weighted),
        "missing_features": missing,
        "min": float(min(values)) if values else default,
        "max": float(max(values)) if values else default,
    }


def _with_feature_contribution_params(
    lgbm_params: dict,
    feature_cols: list[str],
    contribution_cfg: dict,
) -> dict:
    params = dict(lgbm_params)
    values = _feature_contribution_values(feature_cols, contribution_cfg)
    if values is None:
        return params
    if "feature_contri" in params or "feature_penalty" in params:
        raise ValueError(
            "Set feature_contribution at model config level, not lgbm_params.feature_contri/feature_penalty"
        )
    params["feature_contri"] = values
    return params


def _format_bins(bin_dir: list[tuple[int, float, int]]) -> str:
    if not bin_dir:
        return "n/a"
    return ",".join([f"{b}:{acc:.3f}({n})" for b, acc, n in bin_dir])


def _load_existing_score_days(score_output: Path) -> set[date]:
    try:
        cache = load_scores_by_date(score_output)
    except FileNotFoundError:
        return set()
    except Exception as exc:
        print(f"[rolling] failed to read existing scores for incremental mode: {exc}")
        return set()
    return set(cache.keys())


def _parse_checkpoint_day(stem: str) -> date | None:
    try:
        return datetime.strptime(stem, "%Y-%m-%d").date()
    except Exception:
        return None


def _checkpoint_path(state_dir: Path, day: date) -> Path:
    return state_dir / f"{day:%Y-%m-%d}.txt"


def _find_previous_checkpoint(state_dir: Path, day: date) -> Path | None:
    if not state_dir.exists():
        return None
    best_day: date | None = None
    best_path: Path | None = None
    for path in state_dir.glob("*.txt"):
        ckpt_day = _parse_checkpoint_day(path.stem)
        if ckpt_day is None or ckpt_day >= day:
            continue
        if best_day is None or ckpt_day > best_day:
            best_day = ckpt_day
            best_path = path
    return best_path


def _concat_split_data(parts: list[SplitData], factor_cols: list[str]) -> SplitData:
    valid_parts = [p for p in parts if p is not None and not p.x.empty]
    if not valid_parts:
        empty = pd.DataFrame(columns=factor_cols)
        return SplitData(
            x=empty.copy(),
            y=pd.Series(dtype=float),
            dt=pd.Series(dtype="datetime64[ns]"),
            code=pd.Series(dtype=str),
        )
    x = pd.concat([p.x for p in valid_parts], ignore_index=True)
    y = pd.concat([p.y for p in valid_parts], ignore_index=True)
    dt = pd.concat([p.dt for p in valid_parts], ignore_index=True)
    code = pd.concat([p.code for p in valid_parts], ignore_index=True)
    return SplitData(x=x, y=y, dt=dt, code=code)


@dataclass
class _PcaGroupModel:
    name: str
    input_cols: list[str]
    output_cols: list[str]
    fill_values: pd.Series
    center: np.ndarray
    components: np.ndarray
    explained_variance_ratio: list[float]


@dataclass
class _PcaFeatureTransformer:
    mode: str
    original_cols: list[str]
    output_cols: list[str]
    remove_cols: set[str]
    groups: list[_PcaGroupModel]

    @property
    def feature_cols(self) -> list[str]:
        if self.mode == "replace":
            return [c for c in self.original_cols if c not in self.remove_cols] + self.output_cols
        return self.original_cols + self.output_cols


def _resolve_pca_feature_config(cfg: dict, factor_cols: list[str]) -> dict:
    fe_cfg = cfg.get("feature_engineering", {})
    if fe_cfg is None:
        fe_cfg = {}
    if not isinstance(fe_cfg, dict):
        raise TypeError("feature_engineering must be an object")
    raw = fe_cfg.get("pca", cfg.get("pca_features", {}))
    if raw in (None, "", []):
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("feature_engineering.pca must be an object")
    enabled = bool(raw.get("enabled", False))
    mode = str(raw.get("mode", "append")).strip().lower()
    if mode not in {"append", "replace"}:
        raise ValueError("feature_engineering.pca.mode must be append or replace")
    groups_raw = raw.get("groups", [])
    if groups_raw is None:
        groups_raw = []
    if not isinstance(groups_raw, list):
        raise TypeError("feature_engineering.pca.groups must be a list")

    known_cols = set(factor_cols)
    groups: list[dict] = []
    for idx, item in enumerate(groups_raw, start=1):
        if not isinstance(item, dict):
            raise TypeError("feature_engineering.pca.groups entries must be objects")
        name = str(item.get("name", f"group{idx}")).strip()
        if not name:
            raise ValueError("feature_engineering.pca group name must be non-empty")
        cols = [str(c).strip() for c in item.get("factors", []) if str(c).strip()]
        if not cols:
            raise ValueError(f"feature_engineering.pca group {name} has no factors")
        missing = [c for c in cols if c not in known_cols]
        if missing:
            raise ValueError(f"feature_engineering.pca group {name} missing factors: {missing}")
        n_components = int(item.get("n_components", 1))
        if n_components <= 0:
            raise ValueError(f"feature_engineering.pca group {name} n_components must be > 0")
        if n_components > len(cols):
            raise ValueError(
                f"feature_engineering.pca group {name} n_components={n_components} "
                f"> factor_count={len(cols)}"
            )
        min_features = int(item.get("min_features", n_components))
        if len(cols) < min_features:
            raise ValueError(
                f"feature_engineering.pca group {name} factor_count={len(cols)} "
                f"< min_features={min_features}"
            )
        groups.append(
            {
                "name": name,
                "factors": cols,
                "n_components": n_components,
                "fillna": str(item.get("fillna", raw.get("fillna", "median"))).strip().lower(),
                "center": bool(item.get("center", raw.get("center", True))),
            }
        )
    return {
        "enabled": enabled,
        "mode": mode,
        "groups": groups,
    }


def _preview_pca_feature_cols(factor_cols: list[str], pca_cfg: dict) -> list[str]:
    if not bool(pca_cfg.get("enabled", False)):
        return list(factor_cols)
    output_cols: list[str] = []
    remove_cols: set[str] = set()
    for group in pca_cfg.get("groups", []):
        name = str(group["name"])
        n_components = int(group.get("n_components", 1))
        output_cols.extend([f"pca_{name}_pc{i}" for i in range(1, n_components + 1)])
        remove_cols.update(str(c) for c in group.get("factors", []))
    if str(pca_cfg.get("mode", "append")) == "replace":
        return [c for c in factor_cols if c not in remove_cols] + output_cols
    return list(factor_cols) + output_cols


def _pca_fill_values(x: pd.DataFrame, cols: list[str], fillna: str) -> pd.Series:
    if fillna == "median":
        return x[cols].median(axis=0, skipna=True).fillna(0.0)
    if fillna == "mean":
        return x[cols].mean(axis=0, skipna=True).fillna(0.0)
    if fillna == "zero":
        return pd.Series(0.0, index=cols)
    raise ValueError(f"unsupported feature_engineering.pca fillna: {fillna}")


def _fit_pca_feature_transformer(train_x: pd.DataFrame, pca_cfg: dict) -> _PcaFeatureTransformer | None:
    if not bool(pca_cfg.get("enabled", False)):
        return None
    if train_x.empty:
        raise ValueError("cannot fit PCA features on empty training data")

    original_cols = list(train_x.columns)
    mode = str(pca_cfg.get("mode", "append"))
    group_models: list[_PcaGroupModel] = []
    all_output_cols: list[str] = []
    remove_cols: set[str] = set()
    for group in pca_cfg.get("groups", []):
        name = str(group["name"])
        input_cols = [str(c) for c in group["factors"]]
        n_components = int(group.get("n_components", 1))
        fill_values = _pca_fill_values(train_x, input_cols, str(group.get("fillna", "median")))
        mat = train_x[input_cols].astype(float).fillna(fill_values).to_numpy(dtype=float)
        center = mat.mean(axis=0) if bool(group.get("center", True)) else np.zeros(len(input_cols), dtype=float)
        work = mat - center
        if work.shape[0] < 2:
            raise ValueError(f"cannot fit PCA group {name}: less than 2 training rows")
        _, singular_values, vt = np.linalg.svd(work, full_matrices=False)
        if vt.shape[0] < n_components:
            raise ValueError(
                f"cannot fit PCA group {name}: available_components={vt.shape[0]} "
                f"< requested={n_components}"
            )
        components = vt[:n_components].copy()
        for i in range(components.shape[0]):
            anchor = int(np.argmax(np.abs(components[i])))
            if components[i, anchor] < 0:
                components[i] *= -1.0
        variances = singular_values ** 2
        total_variance = float(np.sum(variances))
        ratios = (
            (variances[:n_components] / total_variance).astype(float).tolist()
            if total_variance > 0
            else [0.0 for _ in range(n_components)]
        )
        output_cols = [f"pca_{name}_pc{i}" for i in range(1, n_components + 1)]
        all_output_cols.extend(output_cols)
        remove_cols.update(input_cols)
        group_models.append(
            _PcaGroupModel(
                name=name,
                input_cols=input_cols,
                output_cols=output_cols,
                fill_values=fill_values,
                center=center,
                components=components,
                explained_variance_ratio=[float(v) for v in ratios],
            )
        )
    return _PcaFeatureTransformer(
        mode=mode,
        original_cols=original_cols,
        output_cols=all_output_cols,
        remove_cols=remove_cols,
        groups=group_models,
    )


def _apply_pca_feature_transformer(
    split: SplitData,
    transformer: _PcaFeatureTransformer | None,
) -> SplitData:
    if transformer is None:
        return split
    x = split.x.copy()
    for group in transformer.groups:
        mat = (
            x[group.input_cols]
            .astype(float)
            .fillna(group.fill_values)
            .to_numpy(dtype=float)
        )
        scores = (mat - group.center) @ group.components.T
        for idx, col in enumerate(group.output_cols):
            x[col] = scores[:, idx]
    x = x[transformer.feature_cols].copy()
    return SplitData(x=x, y=split.y, dt=split.dt, code=split.code)


def _pca_transformer_summary(transformer: _PcaFeatureTransformer | None) -> list[dict]:
    if transformer is None:
        return []
    rows: list[dict] = []
    for group in transformer.groups:
        for idx, ratio in enumerate(group.explained_variance_ratio, start=1):
            rows.append(
                {
                    "group": group.name,
                    "component": idx,
                    "explained_variance_ratio": float(ratio),
                    "input_count": int(len(group.input_cols)),
                    "output_col": group.output_cols[idx - 1],
                }
            )
    return rows


def _build_daily_split_cache(
    *,
    days: list[date],
    factor_store: FactorStore,
    label_root: Path,
    factor_cols: list[str],
    raw_factor_cols: list[str] | None,
    preprocess_factor_cols: list[str] | None,
    min_count: int,
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    factor_time: str,
    label_time: str,
    require_label: bool,
    tradable_code_map: dict[date, set[str]] | None,
    tradable_strict: bool,
    neutralizer,
    factor_aliases: dict[str, str] | None = None,
    standardization: dict | None = None,
    missing_values: dict | None = None,
) -> dict[date, SplitData]:
    cache: dict[date, SplitData] = {}
    total = len(days)
    for i, day in enumerate(days, start=1):
        split = build_dataset(
            factor_store=factor_store,
            label_root=label_root,
            days=[day],
            factor_cols=factor_cols,
            raw_factor_cols=raw_factor_cols,
            preprocess_factor_cols=preprocess_factor_cols,
            min_count=min_count,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            require_label=require_label,
            tradable_code_map=tradable_code_map,
            tradable_strict=tradable_strict,
            neutralizer=neutralizer,
            factor_aliases=factor_aliases,
            standardization=standardization,
            missing_values=missing_values,
        )
        if not split.x.empty:
            cache[day] = split
        if i == total or i % 50 == 0:
            tag = "label" if require_label else "test"
            print(f"[rolling] cache_build {tag}: {i}/{total}")
    return cache


def _prepare_rolling_payload(
    *,
    idx: int,
    days: list[date],
    window_days: int,
    train_ratio: float,
    factor_cols: list[str],
    train_day_cache: dict[date, SplitData],
    test_day_cache: dict[date, SplitData],
) -> dict | None:
    window = days[idx - window_days + 1: idx + 1]
    train_pool = window[:-1]
    test_day = window[-1]
    if len(train_pool) < 2:
        return None
    n_pool = len(train_pool)
    n_train = max(1, int(n_pool * train_ratio))
    n_val = n_pool - n_train
    if n_val <= 0:
        n_val = 1
        n_train = max(1, n_pool - n_val)
    train_days = list(train_pool[:n_train])
    val_days = list(train_pool[n_train:n_train + n_val])
    test_data = _concat_split_data(
        [test_day_cache[d] for d in [test_day] if d in test_day_cache],
        factor_cols,
    )
    if test_data.x.empty:
        return None
    train_data = _concat_split_data(
        [train_day_cache[d] for d in train_days if d in train_day_cache],
        factor_cols,
    )
    val_data = _concat_split_data(
        [train_day_cache[d] for d in val_days if d in train_day_cache],
        factor_cols,
    )
    return {
        "test_day": test_day,
        "train_days": train_days,
        "val_days": val_days,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
    }


def main(
    *,
    config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    execution: dict | None = None,
) -> None:
    execution_cfg = dict(execution or {})
    env_refit = os.getenv("CBOND_REFIT_EVERY_N_DAYS")
    env_train_processes = os.getenv("CBOND_TRAIN_PROCESSES")
    env_prep_workers = os.getenv("CBOND_PREP_WORKERS")
    env_prefetch_windows = os.getenv("CBOND_PREFETCH_WINDOWS")
    env_shards = os.getenv("CBOND_SCORE_PARALLEL_SHARDS")
    env_shard_index = os.getenv("CBOND_SCORE_PARALLEL_SHARD_INDEX")
    refit_every_n_days = max(
        1,
        int(
            execution_cfg.get(
                "refit_every_n_days",
                env_refit if env_refit is not None else 1,
            )
        ),
    )
    train_processes = max(
        1,
        int(
            execution_cfg.get(
                "train_processes",
                env_train_processes if env_train_processes is not None else 1,
            )
        ),
    )
    prep_workers = max(
        1,
        int(
            execution_cfg.get(
                "prep_workers",
                env_prep_workers if env_prep_workers is not None else 1,
            )
        ),
    )
    prefetch_windows = max(
        1,
        int(
            execution_cfg.get(
                "prefetch_windows",
                env_prefetch_windows if env_prefetch_windows is not None else 2,
            )
        ),
    )
    parallel_shards = max(
        1,
        int(
            execution_cfg.get(
                "parallel_shards",
                env_shards if env_shards is not None else train_processes,
            )
        ),
    )
    parallel_shard_index = int(
        execution_cfg.get(
            "parallel_shard_index",
            env_shard_index if env_shard_index is not None else 0,
        )
    )
    if parallel_shard_index < 0 or parallel_shard_index >= parallel_shards:
        raise ValueError(
            f"parallel_shard_index must be in [0, {parallel_shards - 1}], "
            f"got {parallel_shard_index}"
        )
    paths_cfg = load_config_file("paths")
    cfg_file = Path(config_path) if config_path else None
    if cfg_file is None and len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            cfg_file = candidate
    cfg = _load_model_config(cfg_file)
    score_guard_cfg = dict(cfg.get("score_guard", {}))
    execution_guard_cfg = execution_cfg.get("score_guard")
    if isinstance(execution_guard_cfg, dict):
        score_guard_cfg.update(execution_guard_cfg)
    score_guard_enabled = bool(score_guard_cfg.get("enabled", True))
    score_guard_equal_tol = float(score_guard_cfg.get("equal_tol", 1e-12))
    score_guard_warn_same_sign = bool(score_guard_cfg.get("warn_same_sign", False))
    score_guard_fail_on_all_equal = bool(score_guard_cfg.get("fail_on_all_equal", False))
    score_guard_fail_all_equal_days = max(0, int(score_guard_cfg.get("fail_all_equal_days", 0)))
    score_guard_bin_guard_enabled = bool(score_guard_cfg.get("bin_guard_enabled", False))
    score_guard_required_bins = max(2, int(score_guard_cfg.get("required_bins", 20)))
    score_guard_bin_min_samples = max(
        1,
        int(score_guard_cfg.get("bin_guard_min_samples", score_guard_required_bins)),
    )
    score_guard_fail_on_insufficient_bins = bool(
        score_guard_cfg.get("fail_on_insufficient_bins", False)
    )
    score_guard_fail_insufficient_bins_days = max(
        0,
        int(score_guard_cfg.get("fail_insufficient_bins_days", 0)),
    )
    print(
        "[score_guard]",
        f"enabled={score_guard_enabled}",
        f"equal_tol={score_guard_equal_tol}",
        f"warn_same_sign={score_guard_warn_same_sign}",
        f"fail_on_all_equal={score_guard_fail_on_all_equal}",
        f"fail_all_equal_days={score_guard_fail_all_equal_days}",
        f"bin_guard_enabled={score_guard_bin_guard_enabled}",
        f"required_bins={score_guard_required_bins}",
        f"bin_guard_min_samples={score_guard_bin_min_samples}",
        f"fail_on_insufficient_bins={score_guard_fail_on_insufficient_bins}",
        f"fail_insufficient_bins_days={score_guard_fail_insufficient_bins_days}",
    )

    cfg_start = parse_date(cfg.get("start"))
    cfg_end = parse_date(cfg.get("end"))
    desired_start = parse_date(start) if start else cfg_start
    desired_end = parse_date(end) if end else cfg_end
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    if desired_start > desired_end:
        raise ValueError("start date must be <= end date")

    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])

    panel_name = cfg.get("panel_name")
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:42"))
    raw_root = paths_cfg["raw_data_root"]
    panel_root = paths_cfg["panel_data_root"]

    scan_start = desired_start
    rolling_cfg = cfg.get("rolling", {})
    rolling_enabled = bool(rolling_cfg.get("enabled", False))
    window_days = int(rolling_cfg.get("window_days", 301))
    if rolling_enabled:
        lookback_days = prev_trading_days_from_raw(
            raw_root,
            desired_start,
            window_days,
            kind="snapshot",
            asset="cbond",
        )
        if lookback_days:
            scan_start = lookback_days[0]
    days = list(_iter_existing_label_days(label_root, scan_start, desired_end))
    if not days:
        raise RuntimeError("no label days found for range")
    days = sorted(set(days))
    if cutoff_day is not None:
        days = [d for d in days if d <= cutoff_day]
        if not days:
            raise RuntimeError("no label days left after label_cutoff filter")

    def _factor_exists(day: date) -> bool:
        label = panel_name or make_window_label(window_minutes)
        month = f"{day.year:04d}-{day.month:02d}"
        filename = f"{day.strftime('%Y%m%d')}.parquet"
        path = factor_root / "factors" / label / month / filename
        return path.exists()

    # allow scoring for target days without labels (e.g., latest day in live)
    last_label_day = max(days) if days else None
    if last_label_day and desired_end > last_label_day:
        trade_days = list_trading_days_from_raw(
            raw_root,
            last_label_day,
            desired_end,
            kind="snapshot",
            asset="cbond",
        )
        extra_days = [d for d in trade_days if d > last_label_day and _factor_exists(d)]
        if extra_days:
            days = sorted(set(days + extra_days))

    tradable_cfg: dict = {
        "enabled": True,
        "strict": True,
        "twap_table": "market_cbond.daily_twap",
        "asset": "cbond",
        "buy_twap_col": "twap_1442_1457",
        "sell_twap_col": "twap_0930_0939",
        "min_amount": 0.0,
        "min_volume": 0.0,
    }
    model_tradable_cfg = cfg.get("tradable_filter")
    if isinstance(model_tradable_cfg, dict):
        tradable_cfg.update(model_tradable_cfg)
    exec_tradable_cfg = execution_cfg.get("tradable_filter")
    if isinstance(exec_tradable_cfg, dict):
        tradable_cfg.update(exec_tradable_cfg)
    tradable_enabled = bool(tradable_cfg.get("enabled", True))
    tradable_strict = bool(tradable_cfg.get("strict", True))
    tradable_code_map: dict[date, set[str]] | None = None
    if tradable_enabled:
        tradable_code_map = build_tradable_code_map(
            raw_data_root=raw_root,
            days=days,
            buy_twap_col=str(tradable_cfg.get("buy_twap_col", "twap_1442_1457")),
            sell_twap_col=str(tradable_cfg.get("sell_twap_col", "twap_0930_0939")),
            min_amount=float(tradable_cfg.get("min_amount", 0.0)),
            min_volume=float(tradable_cfg.get("min_volume", 0.0)),
            twap_table=str(tradable_cfg.get("twap_table", "market_cbond.daily_twap")),
            asset=str(tradable_cfg.get("asset", "cbond")),
        )
        print(
            "[tradable] enabled",
            f"buy={tradable_cfg.get('buy_twap_col')}",
            f"sell={tradable_cfg.get('sell_twap_col')}",
            f"min_amount={float(tradable_cfg.get('min_amount', 0.0))}",
            f"min_volume={float(tradable_cfg.get('min_volume', 0.0))}",
            f"strict={tradable_strict}",
            f"mapped_days={len(tradable_code_map)}",
        )
    else:
        print("[tradable] disabled")

    train_cfg = cfg.get("train", {})
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))

    if window_days < 2:
        raise ValueError("rolling.window_days must be >= 2")

    if not rolling_enabled and abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_days: list[date] = []
    val_days: list[date] = []
    test_days: list[date] = []
    if not rolling_enabled:
        days = [d for d in days if desired_start <= d <= desired_end]
        if not days:
            raise RuntimeError("no label days found for desired range")
        train_days, val_days, test_days = _split_days(days, train_ratio, val_ratio)
        print(f"train days: {len(train_days)}, val days: {len(val_days)}, test days: {len(test_days)}")

    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)
    # pick factor columns from first available day
    sample = pd.DataFrame()
    sample_days = days if rolling_enabled else train_days
    for day in sample_days:
        sample = store.read_day(day)
        if not sample.empty:
            break
    if sample.empty:
        raise RuntimeError("no factor data found")
    if isinstance(sample.index, pd.MultiIndex):
        sample = sample.reset_index()
    factor_aliases = _parse_factor_aliases(cfg)
    raw_factor_cols = _select_factor_cols(sample, cfg, factor_aliases)
    missing_values = _resolve_missing_values_config(cfg)
    missing_feature_cols = missing_value_feature_columns(missing_values)
    factor_cols = list(raw_factor_cols)
    seen_factor_cols = set(factor_cols)
    for col in missing_feature_cols:
        if col in seen_factor_cols:
            continue
        factor_cols.append(col)
        seen_factor_cols.add(col)
    preprocess_factor_cols = list(raw_factor_cols)

    winsor_lower, winsor_upper = parse_winsor_bounds(cfg.get("winsor", {}))
    zscore = bool(cfg.get("zscore", True))
    standardization = cfg.get("standardization")
    standardization_summary = describe_standardization(
        standardization,
        legacy_zscore=zscore,
    )
    min_count = int(cfg.get("min_count", 30))
    bins = int(cfg.get("bins", 5))
    neutralizer = build_neutralizer(
        cfg.get("neutralization"),
        raw_data_root=raw_root,
        panel_data_root=panel_root,
    )
    pca_feature_cfg = _resolve_pca_feature_config(cfg, factor_cols)
    pca_enabled = bool(pca_feature_cfg.get("enabled", False))
    model_feature_cols = _preview_pca_feature_cols(factor_cols, pca_feature_cfg)
    feature_contribution_cfg = _resolve_feature_contribution_config(cfg)
    feature_contribution_summary = _feature_contribution_summary(model_feature_cols, feature_contribution_cfg)
    print(
        "[standardization]",
        f"enabled={bool(standardization_summary.get('enabled', False))}",
        f"method={standardization_summary.get('method')}",
        "stage=after_neutralization",
    )
    print(
        "[pca_features]",
        f"enabled={pca_enabled}",
        f"mode={pca_feature_cfg.get('mode')}",
        f"groups={len(pca_feature_cfg.get('groups', []))}",
        f"feature_count={len(model_feature_cols)}",
    )
    print(
        "[missing_values]",
        f"enabled={bool(missing_values)}",
        f"min_available_factors={missing_values.get('min_available_factors', 'all') if missing_values else 'all'}",
        f"keep_nan={bool(missing_values.get('keep_nan', False)) if missing_values else False}",
        f"missing_feature_count={len(missing_feature_cols)}",
    )
    print(
        "[feature_contribution]",
        f"enabled={bool(feature_contribution_summary.get('enabled', False))}",
        f"default={feature_contribution_summary.get('default')}",
        f"weighted_count={feature_contribution_summary.get('weighted_count')}",
        f"min={feature_contribution_summary.get('min', feature_contribution_summary.get('default'))}",
        f"max={feature_contribution_summary.get('max', feature_contribution_summary.get('default'))}",
    )
    if feature_contribution_summary.get("missing_features"):
        print(
            "[feature_contribution] missing configured features:",
            ",".join(str(x) for x in feature_contribution_summary["missing_features"]),
        )

    lgbm_params = cfg.get("lgbm_params", {})
    grid_cfg = cfg.get("grid_search", {})
    early_rounds = cfg.get("early_stopping_rounds")
    loss_mode = str(cfg.get("loss_mode", "mse")).lower()

    # results output dir
    results_root = Path(paths_cfg["results_root"])
    model_name = cfg.get("model_name", "lgbm_factor")
    date_label = f"{desired_start.strftime('%Y-%m-%d')}_{desired_end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = init_wandb_logger(
        execution_cfg=execution_cfg,
        model_cfg=cfg,
        model_name=str(model_name),
        model_type="lgbm",
        start=desired_start,
        end=desired_end,
        extra_config={
            "rolling_enabled": bool(rolling_enabled),
            "window_days": int(window_days),
            "loss_mode": str(loss_mode),
        },
    )
    wandb_logger.log(
        {
            "refit_every_n_days": int(refit_every_n_days),
            "train_processes": int(train_processes),
            "prep_workers": int(prep_workers),
            "prefetch_windows": int(prefetch_windows),
            "parallel_shards": int(parallel_shards),
            "parallel_shard_index": int(parallel_shard_index),
            "factor_count": int(len(factor_cols)),
            "model_feature_count": int(len(model_feature_cols)),
            "neutralization_enabled": bool(neutralizer is not None and neutralizer.enabled),
            "standardization_enabled": bool(standardization_summary.get("enabled", False)),
            "standardization_method": str(standardization_summary.get("method", "none")),
            "pca_features_enabled": bool(pca_enabled),
            "pca_features_mode": str(pca_feature_cfg.get("mode", "append")),
            "pca_features_groups": int(len(pca_feature_cfg.get("groups", []))),
            "feature_contribution_enabled": bool(feature_contribution_summary.get("enabled", False)),
            "feature_contribution_weighted_count": int(feature_contribution_summary.get("weighted_count", 0)),
            "feature_contribution_min": float(
                feature_contribution_summary.get("min", feature_contribution_summary.get("default", 1.0))
            ),
            "feature_contribution_max": float(
                feature_contribution_summary.get("max", feature_contribution_summary.get("default", 1.0))
            ),
        },
        prefix="run",
    )
    if neutralizer is not None and neutralizer.enabled:
        wandb_logger.log(neutralizer.summary(), prefix="neutralization")
    score_output = resolve_output_path(
        cfg.get("score_output"),
        default_path=results_root / "scores" / model_name,
        results_root=results_root,
    )
    score_overwrite = bool(cfg.get("score_overwrite", False))
    score_dedupe = bool(cfg.get("score_dedupe", True))
    incremental_cfg = dict(cfg.get("incremental", {}))
    incremental_enabled = bool(incremental_cfg.get("enabled", True))
    incremental_skip_existing = bool(incremental_cfg.get("skip_existing_scores", True))
    incremental_warm_start = bool(incremental_cfg.get("warm_start", True))
    incremental_save_state = bool(incremental_cfg.get("save_state", True))
    if pca_enabled and incremental_warm_start:
        print("[pca_features] disable warm_start because PCA basis is refit per train window")
        incremental_warm_start = False
    if parallel_shards > 1 and incremental_warm_start:
        print("[rolling] parallel_shards>1: disable warm_start to avoid cross-shard dependency")
        incremental_warm_start = False
    state_dir_raw = str(incremental_cfg.get("state_dir", "")).strip()
    state_dir = resolve_output_path(
        state_dir_raw if state_dir_raw else None,
        default_path=results_root / "model_state" / model_name,
        results_root=results_root,
    )
    if incremental_enabled and (incremental_warm_start or incremental_save_state):
        state_dir.mkdir(parents=True, exist_ok=True)

    def _metric_name_for_mode(mode: str) -> str:
        text = str(mode or "mse").lower()
        if text in {"ic_abs", "abs_ic", "icabs"}:
            return "abs_ic"
        return "rank_ic"

    def _best_val_metric(hist: list[dict], metric_name: str) -> float:
        if not hist:
            return float("nan")
        key = f"val_{metric_name}"
        vals = [h.get(key) for h in hist if h.get(key) is not None]
        if not vals:
            return float("nan")
        return float(np.nanmax(vals))

    def _best_iter_from_hist(hist: list[dict], metric_name: str) -> int | None:
        best_val = float("-inf")
        best_it = None
        key = f"val_{metric_name}"
        for h in hist:
            v = h.get(key)
            it = h.get("iteration")
            if v is None or it is None or np.isnan(v):
                continue
            if float(v) > best_val:
                best_val = float(v)
                best_it = int(it)
        return best_it

    metric_name = _metric_name_for_mode(loss_mode)

    if rolling_enabled:
        if len(days) < window_days:
            raise ValueError(
                f"rolling window_days={window_days} exceeds available days={len(days)}"
            )
        rolling_rows: list[dict] = []
        score_rows: list[pd.DataFrame] = []
        desired_days = [d for d in days if desired_start <= d <= desired_end]
        if not desired_days:
            raise RuntimeError("no label days found for desired range")
        target_days = list(desired_days)
        if incremental_enabled and incremental_skip_existing and not score_overwrite:
            existing_days = _load_existing_score_days(score_output)
            target_days = [d for d in desired_days if d not in existing_days]
            skipped = len(desired_days) - len(target_days)
            if skipped > 0:
                print(
                    f"[rolling] incremental skip_existing_scores=True, "
                    f"skip={skipped}, pending={len(target_days)}"
                )
        if not target_days:
            print("[rolling] incremental: no pending target days, skip training")
            (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "features.json").write_text(json.dumps(model_feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved rolling: {out_dir}")
            print(f"saved scores: {score_output}")
            wandb_logger.finish({"status": "no_pending_target_days"})
            return
        day_to_idx = {d: i for i, d in enumerate(days)}
        target_indices = [day_to_idx[d] for d in target_days if d in day_to_idx]
        min_idx = window_days - 1
        valid_indices = [i for i in target_indices if i >= min_idx]
        dropped = len(target_indices) - len(valid_indices)
        if dropped > 0:
            print(
                f"[rolling] skip {dropped} target day(s): insufficient history window "
                f"(need window_days={window_days})"
            )
        if not valid_indices:
            raise RuntimeError("rolling has no valid target day after incremental filtering")
        if rolling_enabled:
            print(
                f"[rolling] execution refit_every_n_days={refit_every_n_days} "
                f"prep_workers={prep_workers} prefetch_windows={prefetch_windows} "
                f"train_processes={train_processes} "
                f"parallel_shards={parallel_shards} parallel_shard_index={parallel_shard_index}"
            )
        if parallel_shards > 1:
            all_count = len(valid_indices)
            valid_indices = [
                idx
                for seq, idx in enumerate(valid_indices)
                if (seq % parallel_shards) == parallel_shard_index
            ]
            print(
                f"[rolling] shard assignment {parallel_shard_index + 1}/{parallel_shards}: "
                f"{len(valid_indices)}/{all_count} target day(s)"
            )
        if not valid_indices:
            print("[rolling] no target day assigned for this shard, skip training")
            (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "features.json").write_text(json.dumps(model_feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved rolling: {out_dir}")
            print(f"saved scores: {score_output}")
            wandb_logger.finish({"status": "no_target_for_shard"})
            return
        train_cache_days = sorted(
            {
                d
                for idx in valid_indices
                for d in days[idx - window_days + 1: idx]
            }
        )
        test_cache_days = sorted({days[idx] for idx in valid_indices})
        print(
            f"[rolling] prebuild caches: train_days={len(train_cache_days)} "
            f"test_days={len(test_cache_days)}"
        )
        train_day_cache = _build_daily_split_cache(
            days=train_cache_days,
            factor_store=store,
            label_root=label_root,
            factor_cols=factor_cols,
            raw_factor_cols=raw_factor_cols,
            preprocess_factor_cols=preprocess_factor_cols,
            min_count=min_count,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            require_label=True,
            tradable_code_map=tradable_code_map,
            tradable_strict=tradable_strict,
            neutralizer=neutralizer,
            factor_aliases=factor_aliases,
            standardization=standardization,
            missing_values=missing_values,
        )
        test_day_cache = _build_daily_split_cache(
            days=test_cache_days,
            factor_store=store,
            label_root=label_root,
            factor_cols=factor_cols,
            raw_factor_cols=raw_factor_cols,
            preprocess_factor_cols=preprocess_factor_cols,
            min_count=min_count,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            require_label=False,
            tradable_code_map=tradable_code_map,
            tradable_strict=tradable_strict,
            neutralizer=neutralizer,
            factor_aliases=factor_aliases,
            standardization=standardization,
            missing_values=missing_values,
        )
        print(
            f"[rolling] cache_ready train_cached={len(train_day_cache)} "
            f"test_cached={len(test_day_cache)}"
        )
        total_rolls = len(valid_indices)
        active_model = None
        active_pca_transformer: _PcaFeatureTransformer | None = None
        last_refit_pos: int | None = None
        last_refit_day: date | None = None
        all_equal_days: list[date] = []
        insufficient_bin_days: list[date] = []
        pca_summary_rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=prep_workers, thread_name_prefix="roll_prep") as prep_pool:
            inflight: deque[tuple[int, int, object]] = deque()
            next_pos = 0

            def _enqueue() -> bool:
                nonlocal next_pos
                if next_pos >= total_rolls:
                    return False
                roll_pos = next_pos + 1
                idx = valid_indices[next_pos]
                fut = prep_pool.submit(
                    _prepare_rolling_payload,
                    idx=idx,
                    days=days,
                    window_days=window_days,
                    train_ratio=train_ratio,
                    factor_cols=factor_cols,
                    train_day_cache=train_day_cache,
                    test_day_cache=test_day_cache,
                )
                inflight.append((roll_pos, idx, fut))
                next_pos += 1
                return True

            for _ in range(min(prefetch_windows, total_rolls)):
                if not _enqueue():
                    break

            while inflight:
                roll_idx, idx, fut = inflight.popleft()
                _enqueue()
                payload = fut.result()
                if payload is None:
                    continue
                test_day = payload["test_day"]
                train_days = payload["train_days"]
                val_days = payload["val_days"]
                train_data = payload["train_data"]
                val_data = payload["val_data"]
                test_data = payload["test_data"]
                should_refit = (
                    active_model is None
                    or refit_every_n_days <= 1
                    or last_refit_pos is None
                    or (roll_idx - last_refit_pos) >= refit_every_n_days
                )
                print(
                    f"[rolling] {roll_idx}/{total_rolls} test_day={test_day} "
                    f"refit={'Y' if should_refit else 'N'} cadence={refit_every_n_days}"
                )
                refit_status = "reuse"
                if should_refit:
                    if train_data.x.empty:
                        if active_model is None:
                            print(f"[rolling] skip {test_day}: empty train split and no reusable model")
                            continue
                        refit_status = "reuse_empty_train"
                    else:
                        init_model = None
                        fit_pca_transformer = _fit_pca_feature_transformer(
                            train_data.x,
                            pca_feature_cfg,
                        )
                        if fit_pca_transformer is not None:
                            train_data = _apply_pca_feature_transformer(train_data, fit_pca_transformer)
                            val_data = _apply_pca_feature_transformer(val_data, fit_pca_transformer)
                            for row in _pca_transformer_summary(fit_pca_transformer):
                                pca_summary_rows.append({"trade_date": test_day, **row})
                        if incremental_enabled and incremental_warm_start:
                            prev_ckpt = _find_previous_checkpoint(state_dir, test_day)
                            if prev_ckpt is not None:
                                init_model = str(prev_ckpt)
                                print(f"[rolling] warm-start from checkpoint: {prev_ckpt.name}")
                        try:
                            model, params = train_lgbm(
                                train=train_data,
                                val=val_data,
                                lgbm_params=_with_feature_contribution_params(
                                    lgbm_params,
                                    list(train_data.x.columns),
                                    feature_contribution_cfg,
                                ),
                                early_stopping_rounds=int(early_rounds) if early_rounds else None,
                                loss_mode=loss_mode,
                                init_model=init_model,
                            )
                        except Exception as exc:
                            if init_model is None:
                                raise
                            print(f"[rolling] warm-start failed, fallback cold-start: {type(exc).__name__}: {exc}")
                            model, params = train_lgbm(
                                train=train_data,
                                val=val_data,
                                lgbm_params=_with_feature_contribution_params(
                                    lgbm_params,
                                    list(train_data.x.columns),
                                    feature_contribution_cfg,
                                ),
                                early_stopping_rounds=int(early_rounds) if early_rounds else None,
                                loss_mode=loss_mode,
                                init_model=None,
                            )
                        hist = params.get("history", []) if isinstance(params, dict) else []
                        if hist:
                            wandb_logger.log_history(
                                hist,
                                step_key="iteration",
                                prefix="iter",
                            )
                        active_model = model
                        active_pca_transformer = fit_pca_transformer
                        last_refit_pos = roll_idx
                        last_refit_day = test_day
                        refit_status = "refit"
                        if incremental_enabled and incremental_save_state:
                            ckpt_path = _checkpoint_path(state_dir, test_day)
                            try:
                                model.booster_.save_model(str(ckpt_path))
                            except Exception as exc:
                                print(f"[rolling] failed to save checkpoint {ckpt_path}: {exc}")
                if active_model is None:
                    print(f"[rolling] skip {test_day}: no trained model available")
                    continue
                if pca_enabled:
                    if active_pca_transformer is None:
                        raise RuntimeError("PCA features are enabled but no active PCA transformer is available")
                    test_data = _apply_pca_feature_transformer(test_data, active_pca_transformer)
                test_pred = active_model.predict(test_data.x)
                guard_stats = (
                    score_guard_stats(test_pred, equal_tol=score_guard_equal_tol)
                    if score_guard_enabled
                    else {}
                )
                guard_flags = (
                    score_guard_flags(guard_stats, warn_same_sign=score_guard_warn_same_sign)
                    if score_guard_enabled
                    else []
                )
                bin_guard_stats = (
                    score_guard_bin_stats(
                        test_pred,
                        target_bins=score_guard_required_bins,
                        min_count=score_guard_bin_min_samples,
                    )
                    if score_guard_enabled and score_guard_bin_guard_enabled
                    else {}
                )
                if (
                    score_guard_enabled
                    and score_guard_bin_guard_enabled
                    and bool(bin_guard_stats.get("score_bin_guard_checked", False))
                    and bool(bin_guard_stats.get("score_bin_insufficient", False))
                ):
                    guard_flags = [*guard_flags, "bins_lt_target"]
                    insufficient_bin_days.append(test_day)
                    if (
                        score_guard_fail_on_insufficient_bins
                        and len(insufficient_bin_days) > score_guard_fail_insufficient_bins_days
                    ):
                        raise RuntimeError(
                            "score_guard failed: insufficient-bin prediction day "
                            f"{test_day} actual={bin_guard_stats.get('score_bin_count')} "
                            f"target={bin_guard_stats.get('score_bin_target')}"
                        )
                if score_guard_enabled and guard_flags:
                    print(
                        f"[score_guard:{test_day}] flags={','.join(guard_flags)} "
                        f"range={guard_stats.get('score_range')} std={guard_stats.get('score_std')} "
                        f"unique={guard_stats.get('score_unique_count')}/{guard_stats.get('score_count')} "
                        f"pos_ratio={guard_stats.get('score_pos_ratio')}"
                    )
                if (
                    score_guard_enabled
                    and score_guard_bin_guard_enabled
                    and bool(bin_guard_stats.get("score_bin_guard_checked", False))
                    and bool(bin_guard_stats.get("score_bin_insufficient", False))
                ):
                    print(
                        f"[score_guard:{test_day}] bin_guard "
                        f"actual={bin_guard_stats.get('score_bin_count')} "
                        f"target={bin_guard_stats.get('score_bin_target')} "
                        f"count={guard_stats.get('score_count')}"
                    )
                if score_guard_enabled and bool(guard_stats.get("score_all_equal", False)):
                    all_equal_days.append(test_day)
                model_source_day = test_day if refit_status == "refit" else (last_refit_day or test_day)
                if not (desired_start <= test_day <= desired_end):
                    continue
                score_df = pd.DataFrame(
                    {"trade_date": test_day, "code": test_data.code, "score": test_pred}
                )
                score_rows.append(score_df)
                if np.isfinite(test_data.y).any():
                    test_metrics = evaluate_metrics(
                        test_data.x, test_data.y, test_data.dt, test_pred, bins=bins
                    )
                else:
                    test_metrics = {
                        "mse": float("nan"),
                        "r2": float("nan"),
                        "dir": float("nan"),
                        "ic_mean": float("nan"),
                        "ic_ir": float("nan"),
                        "rank_ic_mean": float("nan"),
                        "rank_ic_ir": float("nan"),
                        "bins": [],
                    }
                rolling_rows.append(
                    {
                        "trade_date": test_day,
                        "train_days": len(train_days),
                        "val_days": len(val_days),
                        "count": int(len(test_data.y)),
                        "refit": bool(refit_status == "refit"),
                        "refit_status": refit_status,
                        "model_source_day": model_source_day,
                        "refit_every_n_days": refit_every_n_days,
                        "rank_ic": test_metrics["rank_ic_mean"],
                        "ic": test_metrics["ic_mean"],
                        "dir": test_metrics["dir"],
                        "mse": test_metrics["mse"],
                        "r2": test_metrics["r2"],
                        "score_min": guard_stats.get("score_min", float("nan")),
                        "score_max": guard_stats.get("score_max", float("nan")),
                        "score_range": guard_stats.get("score_range", float("nan")),
                        "score_std": guard_stats.get("score_std", float("nan")),
                        "score_unique_count": guard_stats.get("score_unique_count", 0),
                        "score_count": guard_stats.get("score_count", int(len(test_pred))),
                        "score_pos_ratio": guard_stats.get("score_pos_ratio", float("nan")),
                        "score_neg_ratio": guard_stats.get("score_neg_ratio", float("nan")),
                        "score_zero_ratio": guard_stats.get("score_zero_ratio", float("nan")),
                        "score_same_sign": bool(guard_stats.get("score_same_sign", False)),
                        "score_all_equal": bool(guard_stats.get("score_all_equal", False)),
                        "score_bin_guard_checked": bool(
                            bin_guard_stats.get("score_bin_guard_checked", False)
                        ),
                        "score_bin_target": int(bin_guard_stats.get("score_bin_target", 0)),
                        "score_bin_count": int(bin_guard_stats.get("score_bin_count", 0)),
                        "score_bin_insufficient": bool(
                            bin_guard_stats.get("score_bin_insufficient", False)
                        ),
                    }
                )
                print(
                    f"rolling {test_day} refit={refit_status} train_days={len(train_days)} "
                    f"val_days={len(val_days)} count={len(test_data.y)} "
                    f"rank_ic={test_metrics['rank_ic_mean']:.4f}"
                )
                wandb_logger.log(
                    {
                        "trade_date": test_day,
                        "train_days": len(train_days),
                        "val_days": len(val_days),
                        "count": int(len(test_data.y)),
                        "refit": bool(refit_status == "refit"),
                        "rank_ic": test_metrics.get("rank_ic_mean"),
                        "ic": test_metrics.get("ic_mean"),
                        "dir": test_metrics.get("dir"),
                        "mse": test_metrics.get("mse"),
                        "r2": test_metrics.get("r2"),
                        "score_std": guard_stats.get("score_std", float("nan")),
                        "score_range": guard_stats.get("score_range", float("nan")),
                        "score_same_sign": bool(guard_stats.get("score_same_sign", False)),
                        "score_all_equal": bool(guard_stats.get("score_all_equal", False)),
                        "score_bin_count": int(bin_guard_stats.get("score_bin_count", 0)),
                        "score_bin_target": int(bin_guard_stats.get("score_bin_target", 0)),
                        "score_bin_insufficient": bool(
                            bin_guard_stats.get("score_bin_insufficient", False)
                        ),
                    },
                    step=int(roll_idx),
                    prefix="rolling",
                )

        if score_rows:
            all_scores = pd.concat(score_rows, ignore_index=True)
            write_scores_by_date(
                score_output,
                all_scores,
                overwrite=score_overwrite,
                dedupe=score_dedupe,
            )
        else:
            raise RuntimeError("rolling produced no scores; check window_days and data range")
        if rolling_rows:
            rr = pd.DataFrame(rolling_rows)
            rr.to_csv(out_dir / "rolling_metrics.csv", index=False)
            guard_cols = [
                "trade_date",
                "score_count",
                "score_min",
                "score_max",
                "score_range",
                "score_std",
                "score_unique_count",
                "score_pos_ratio",
                "score_neg_ratio",
                "score_zero_ratio",
                "score_same_sign",
                "score_all_equal",
                "score_bin_guard_checked",
                "score_bin_target",
                "score_bin_count",
                "score_bin_insufficient",
            ]
            present_guard_cols = [c for c in guard_cols if c in rr.columns]
            if present_guard_cols:
                rr[present_guard_cols].to_csv(out_dir / "rolling_score_guard.csv", index=False)
        if pca_summary_rows:
            pd.DataFrame(pca_summary_rows).to_csv(out_dir / "rolling_pca_features.csv", index=False)
        (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "features.json").write_text(json.dumps(model_feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved rolling: {out_dir}")
        print(f"saved scores: {score_output}")
        if score_guard_enabled:
            print(
                "[score_guard] rolling summary",
                f"all_equal_days={len(all_equal_days)}",
                f"insufficient_bin_days={len(insufficient_bin_days)}",
                f"total_days={len(rolling_rows)}",
            )
            if score_guard_fail_on_all_equal and len(all_equal_days) > score_guard_fail_all_equal_days:
                raise RuntimeError(
                    "score_guard failed: all-equal prediction days "
                    f"{len(all_equal_days)} > allowed {score_guard_fail_all_equal_days}"
                )
            if (
                score_guard_bin_guard_enabled
                and score_guard_fail_on_insufficient_bins
                and len(insufficient_bin_days) > score_guard_fail_insufficient_bins_days
            ):
                raise RuntimeError(
                    "score_guard failed: insufficient-bin prediction days "
                    f"{len(insufficient_bin_days)} > allowed {score_guard_fail_insufficient_bins_days}"
                )
        if rolling_rows:
            rr = pd.DataFrame(rolling_rows)
            wandb_logger.finish(
                {
                    "status": "ok",
                    "rolling_days": int(len(rr)),
                    "rolling_rank_ic_mean": float(pd.to_numeric(rr["rank_ic"], errors="coerce").mean()),
                    "rolling_ic_mean": float(pd.to_numeric(rr["ic"], errors="coerce").mean()),
                    "rolling_dir_mean": float(pd.to_numeric(rr["dir"], errors="coerce").mean()),
                    "score_guard_all_equal_days": int(len(all_equal_days)),
                    "score_guard_insufficient_bin_days": int(len(insufficient_bin_days)),
                }
            )
        else:
            wandb_logger.finish({"status": "ok_empty_rolling"})
        return

    train_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=train_days,
        factor_cols=factor_cols,
        raw_factor_cols=raw_factor_cols,
        preprocess_factor_cols=preprocess_factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
        tradable_code_map=tradable_code_map,
        tradable_strict=tradable_strict,
        neutralizer=neutralizer,
        factor_aliases=factor_aliases,
        standardization=standardization,
        missing_values=missing_values,
    )
    val_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=val_days,
        factor_cols=factor_cols,
        raw_factor_cols=raw_factor_cols,
        preprocess_factor_cols=preprocess_factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
        tradable_code_map=tradable_code_map,
        tradable_strict=tradable_strict,
        neutralizer=neutralizer,
        factor_aliases=factor_aliases,
        standardization=standardization,
        missing_values=missing_values,
    )
    test_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=test_days,
        factor_cols=factor_cols,
        raw_factor_cols=raw_factor_cols,
        preprocess_factor_cols=preprocess_factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
        tradable_code_map=tradable_code_map,
        tradable_strict=tradable_strict,
        neutralizer=neutralizer,
        factor_aliases=factor_aliases,
        standardization=standardization,
        missing_values=missing_values,
    )

    pca_transformer = _fit_pca_feature_transformer(train_data.x, pca_feature_cfg)
    if pca_transformer is not None:
        train_data = _apply_pca_feature_transformer(train_data, pca_transformer)
        val_data = _apply_pca_feature_transformer(val_data, pca_transformer)
        test_data = _apply_pca_feature_transformer(test_data, pca_transformer)
        pca_summary = _pca_transformer_summary(pca_transformer)
        if pca_summary:
            pd.DataFrame(pca_summary).to_csv(out_dir / "pca_features.csv", index=False)

    model = None
    params = {}
    history = []

    grid_enabled = bool(grid_cfg.get("enable", False))
    if grid_enabled:
        alpha_list = grid_cfg.get("reg_alpha", [])
        lambda_list = grid_cfg.get("reg_lambda", [])
        leaves_list = grid_cfg.get("num_leaves", [])
        depth_list = grid_cfg.get("max_depth", [])
        if not alpha_list:
            alpha_list = [lgbm_params.get("reg_alpha", 0.0)]
        if not lambda_list:
            lambda_list = [lgbm_params.get("reg_lambda", 0.0)]
        if not leaves_list:
            leaves_list = [lgbm_params.get("num_leaves", 31)]
        if not depth_list:
            depth_list = [lgbm_params.get("max_depth", -1)]
        grid_rows = []
        best_score = float("-inf")
        best_params = None
        best_model = None
        best_history = None
        best_iter = None
        for alpha in alpha_list:
            for lam in lambda_list:
                for leaves in leaves_list:
                    for depth in depth_list:
                        trial_params = dict(lgbm_params)
                        trial_params["reg_alpha"] = float(alpha)
                        trial_params["reg_lambda"] = float(lam)
                        trial_params["num_leaves"] = int(leaves)
                        trial_params["max_depth"] = int(depth)
                        trial_model, trial_meta = train_lgbm(
                            train=train_data,
                            val=val_data,
                            lgbm_params=_with_feature_contribution_params(
                                trial_params,
                                list(train_data.x.columns),
                                feature_contribution_cfg,
                            ),
                            early_stopping_rounds=int(early_rounds) if early_rounds else None,
                            loss_mode=loss_mode,
                        )
                        trial_hist = trial_meta.get("history", [])
                        trial_best = _best_val_metric(trial_hist, metric_name)
                        trial_best_iter = _best_iter_from_hist(trial_hist, metric_name)
                        grid_rows.append(
                            {
                                "reg_alpha": float(alpha),
                                "reg_lambda": float(lam),
                                "num_leaves": int(leaves),
                                "max_depth": int(depth),
                                f"best_val_{metric_name}": trial_best,
                                "best_iteration": trial_best_iter,
                            }
                        )
                        wandb_logger.log(
                            grid_rows[-1],
                            step=int(len(grid_rows)),
                            prefix="grid",
                        )
                        if not np.isnan(trial_best) and trial_best > best_score:
                            best_score = trial_best
                            best_params = trial_params
                            best_model = trial_model
                            best_history = trial_hist
                            best_iter = trial_best_iter
        pd.DataFrame(grid_rows).to_csv(out_dir / "grid_search.csv", index=False)
        if best_params is None:
            raise RuntimeError("grid search failed to produce valid params")
        (out_dir / "best_params.json").write_text(
            json.dumps(
                {
                    f"best_val_{metric_name}": best_score,
                    "best_iteration": best_iter,
                    "params": best_params,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = best_model
        params = best_params | {"history": best_history or []}
        history = best_history or []
    else:
        model, params = train_lgbm(
            train=train_data,
            val=val_data,
            lgbm_params=_with_feature_contribution_params(
                lgbm_params,
                list(train_data.x.columns),
                feature_contribution_cfg,
            ),
            early_stopping_rounds=int(early_rounds) if early_rounds else None,
            loss_mode=loss_mode,
        )
        history = params.get("history", [])

    train_pred = model.predict(train_data.x)
    val_pred = model.predict(val_data.x)
    test_pred = model.predict(test_data.x)
    split_guard_rows = []
    if score_guard_enabled:
        for split_name, pred in [("train", train_pred), ("val", val_pred), ("test", test_pred)]:
            stats = score_guard_stats(pred, equal_tol=score_guard_equal_tol)
            flags = score_guard_flags(stats, warn_same_sign=score_guard_warn_same_sign)
            split_guard_rows.append({"split": split_name, **stats})
            if flags:
                print(
                    f"[score_guard:{split_name}] flags={','.join(flags)} "
                    f"range={stats.get('score_range')} std={stats.get('score_std')} "
                    f"unique={stats.get('score_unique_count')}/{stats.get('score_count')} "
                    f"pos_ratio={stats.get('score_pos_ratio')}"
                )

    train_metrics = evaluate_metrics(train_data.x, train_data.y, train_data.dt, train_pred, bins=bins)
    val_metrics = evaluate_metrics(val_data.x, val_data.y, val_data.dt, val_pred, bins=bins)
    test_metrics = evaluate_metrics(test_data.x, test_data.y, test_data.dt, test_pred, bins=bins)

    print(
        f"train mse={train_metrics['mse']:.6f} r2={train_metrics['r2']:.4f} dir={train_metrics['dir']:.4f} "
        f"ic={train_metrics['ic_mean']:.4f} ir={train_metrics['ic_ir']:.4f} "
        f"rank_ic={train_metrics['rank_ic_mean']:.4f} rank_ir={train_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"val mse={val_metrics['mse']:.6f} r2={val_metrics['r2']:.4f} dir={val_metrics['dir']:.4f} "
        f"ic={val_metrics['ic_mean']:.4f} ir={val_metrics['ic_ir']:.4f} "
        f"rank_ic={val_metrics['rank_ic_mean']:.4f} rank_ir={val_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"test mse={test_metrics['mse']:.6f} r2={test_metrics['r2']:.4f} dir={test_metrics['dir']:.4f} "
        f"ic={test_metrics['ic_mean']:.4f} ir={test_metrics['ic_ir']:.4f} "
        f"rank_ic={test_metrics['rank_ic_mean']:.4f} rank_ir={test_metrics['rank_ic_ir']:.4f}"
    )

    print(f"train bins: {_format_bins(train_metrics['bin_dir'])}")
    print(f"val bins: {_format_bins(val_metrics['bin_dir'])}")
    print(f"test bins: {_format_bins(test_metrics['bin_dir'])}")

    # save model (best iteration if available)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter and isinstance(best_iter, int):
        model.booster_.save_model(str(out_dir / "model.txt"), num_iteration=best_iter)
    else:
        model.booster_.save_model(str(out_dir / "model.txt"))
    # save config and metrics
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "features.json").write_text(json.dumps(model_feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([
        {"split": "train", **{k: v for k, v in train_metrics.items() if k != "bin_dir"}},
        {"split": "val", **{k: v for k, v in val_metrics.items() if k != "bin_dir"}},
        {"split": "test", **{k: v for k, v in test_metrics.items() if k != "bin_dir"}},
    ])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    if score_guard_enabled and split_guard_rows:
        pd.DataFrame(split_guard_rows).to_csv(out_dir / "score_guard.csv", index=False)
        all_equal_splits = sum(1 for row in split_guard_rows if bool(row.get("score_all_equal", False)))
        print(
            "[score_guard] split summary",
            f"all_equal_splits={all_equal_splits}",
            f"total_splits={len(split_guard_rows)}",
        )
        if score_guard_fail_on_all_equal and all_equal_splits > 0:
            raise RuntimeError(
                "score_guard failed: non-rolling split has all-equal predictions"
            )

    if history:
        pd.DataFrame(history).to_csv(out_dir / "metrics_iter.csv", index=False)
        wandb_logger.log_history(history, step_key="iteration", prefix="iter")

    # save bin dir
    def _bin_df(split, metrics):
        return pd.DataFrame([
            {"split": split, "bin": b, "dir_acc": acc, "count": n} for b, acc, n in metrics["bin_dir"]
        ])
    bin_df = pd.concat([
        _bin_df("train", train_metrics),
        _bin_df("val", val_metrics),
        _bin_df("test", test_metrics),
    ], ignore_index=True)
    bin_df.to_csv(out_dir / "bin_dir.csv", index=False)
    wandb_logger.log(
        {
            "train_mse": train_metrics.get("mse"),
            "train_r2": train_metrics.get("r2"),
            "train_dir": train_metrics.get("dir"),
            "train_rank_ic": train_metrics.get("rank_ic_mean"),
            "val_mse": val_metrics.get("mse"),
            "val_r2": val_metrics.get("r2"),
            "val_dir": val_metrics.get("dir"),
            "val_rank_ic": val_metrics.get("rank_ic_mean"),
            "test_mse": test_metrics.get("mse"),
            "test_r2": test_metrics.get("r2"),
            "test_dir": test_metrics.get("dir"),
            "test_rank_ic": test_metrics.get("rank_ic_mean"),
            "score_guard_train_all_equal": bool(split_guard_rows[0]["score_all_equal"]) if score_guard_enabled and split_guard_rows else False,
            "score_guard_val_all_equal": bool(split_guard_rows[1]["score_all_equal"]) if score_guard_enabled and len(split_guard_rows) > 1 else False,
            "score_guard_test_all_equal": bool(split_guard_rows[2]["score_all_equal"]) if score_guard_enabled and len(split_guard_rows) > 2 else False,
        },
        prefix="final",
    )
    wandb_logger.finish(
        {
            "status": "ok",
            "best_iteration": getattr(model, "best_iteration_", None),
            "out_dir": str(out_dir),
        }
    )

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()


