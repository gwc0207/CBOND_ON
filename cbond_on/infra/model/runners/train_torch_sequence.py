from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import json
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.naming import make_window_label
from cbond_on.core.trading_days import list_trading_days_from_raw, prev_trading_days_from_raw
from cbond_on.common.config_utils import resolve_config_path
from cbond_on.domain.factors.storage import FactorStore
from cbond_on.infra.model.impl.lgbm.trainer import (
    _apply_winsor_zscore,
    _iter_existing_label_days,
    _read_label_day,
    _split_days,
    build_tradable_code_map,
)
from cbond_on.infra.model.impl.torch_sequence import FactorCNN1DModel, FactorLSTMModel
from cbond_on.infra.model.neutralization import FactorNeutralizer, build_neutralizer
from cbond_on.infra.model.preprocess_config import parse_winsor_bounds
from cbond_on.infra.model.score_io import load_scores_by_date, write_scores_by_date


class MissingSequenceDataError(RuntimeError):
    pass


@dataclass
class SequenceSplitData:
    x: np.ndarray
    y: np.ndarray
    dt: np.ndarray
    code: np.ndarray


class _SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]).float(),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def _load_model_config(path: Path | None) -> dict:
    if path is None:
        raise ValueError("torch sequence model config path is required")
    suffix = path.suffix.lower()
    if suffix == ".json5":
        import json5

        with path.open("r", encoding="utf-8") as handle:
            return json5.load(handle) or {}
    if suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle) or {}


def _device_from_config(train_cfg: dict) -> torch.device:
    requested = str(train_cfg.get("device", "cpu")).strip().lower()
    if requested in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_existing_score_days(score_output: Path) -> set[date]:
    try:
        cache = load_scores_by_date(score_output)
    except FileNotFoundError:
        return set()
    except Exception as exc:
        print(f"[rolling] failed to read existing scores for incremental mode: {exc}")
        return set()
    return set(cache.keys())


def _empty_split(sequence_days: int, factor_count: int) -> SequenceSplitData:
    return SequenceSplitData(
        x=np.empty((0, int(sequence_days), int(factor_count)), dtype=np.float32),
        y=np.array([], dtype=np.float32),
        dt=np.array([], dtype=object),
        code=np.array([], dtype=object),
    )


def _feature_engineering_cfg(cfg: dict | None) -> dict:
    raw = cfg if isinstance(cfg, dict) else {}
    fill_raw = raw.get("fill_missing", False)
    fill_enabled = bool(fill_raw)
    fill_value = 0.0
    add_missing_mask = False
    min_available_ratio = 0.0
    if isinstance(fill_raw, dict):
        fill_enabled = bool(fill_raw.get("enabled", True))
        fill_value = float(fill_raw.get("value", fill_raw.get("fill_value", 0.0)))
        add_missing_mask = bool(fill_raw.get("add_mask", fill_raw.get("add_missing_mask", False)))
        min_available_ratio = float(fill_raw.get("min_available_ratio", 0.0) or 0.0)

    temporal_raw = raw.get("temporal", raw.get("temporal_features", {}))
    temporal_cfg = temporal_raw if isinstance(temporal_raw, dict) else {}
    temporal_enabled = bool(temporal_cfg.get("enabled", bool(temporal_cfg)))

    def _int_list(value: object) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (str, int, float)):
            values = [value]
        else:
            values = list(value)
        out: list[int] = []
        for item in values:
            parsed = int(item)
            if parsed > 0 and parsed not in out:
                out.append(parsed)
        return out

    return {
        "clip_z": raw.get("clip_z"),
        "fill_missing_enabled": fill_enabled,
        "fill_missing_value": fill_value,
        "add_missing_mask": add_missing_mask,
        "min_available_ratio": max(0.0, min(1.0, min_available_ratio)),
        "temporal_enabled": temporal_enabled,
        "include_level": bool(temporal_cfg.get("include_level", True)),
        "diff_lags": _int_list(temporal_cfg.get("diff_lags", [])),
        "slope_windows": _int_list(temporal_cfg.get("slope_windows", [])),
    }


def _feature_names(factor_cols: list[str], feature_cfg: dict) -> list[str]:
    names: list[str] = []
    include_level = bool(feature_cfg.get("include_level", True)) or not bool(feature_cfg.get("temporal_enabled", False))
    if include_level:
        names.extend(factor_cols)
    if bool(feature_cfg.get("temporal_enabled", False)):
        for lag in feature_cfg.get("diff_lags", []):
            names.extend([f"{col}__diff{lag}" for col in factor_cols])
        for window in feature_cfg.get("slope_windows", []):
            names.extend([f"{col}__slope{window}" for col in factor_cols])
    if bool(feature_cfg.get("add_missing_mask", False)):
        names.extend([f"{col}__missing" for col in factor_cols])
    return names


def _rolling_slope_features(x: np.ndarray, window: int) -> np.ndarray:
    n_samples, seq_len, n_features = x.shape
    out = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)
    window = max(2, int(window))
    for t in range(seq_len):
        start = max(0, t - window + 1)
        y = x[:, start : t + 1, :]
        steps = y.shape[1]
        if steps < 2:
            continue
        grid = np.arange(steps, dtype=np.float32)
        grid = grid - float(grid.mean())
        denom = float(np.sum(grid * grid))
        if denom <= 0:
            continue
        y_centered = y - y.mean(axis=1, keepdims=True)
        out[:, t, :] = np.sum(y_centered * grid.reshape(1, steps, 1), axis=1) / denom
    return out


def _transform_sequence_features(x: np.ndarray, feature_cfg: dict) -> np.ndarray:
    if x.size == 0:
        return x
    work = x.astype(np.float32, copy=True)
    finite_mask = np.isfinite(work)
    min_available_ratio = float(feature_cfg.get("min_available_ratio", 0.0) or 0.0)
    if min_available_ratio > 0.0:
        coverage = finite_mask.mean(axis=2, keepdims=True)
        work = np.where(coverage >= min_available_ratio, work, np.nan)
        finite_mask = np.isfinite(work)
    clip_z = feature_cfg.get("clip_z")
    if clip_z is not None:
        bound = abs(float(clip_z))
        if bound > 0:
            work = np.where(finite_mask, np.clip(work, -bound, bound), work)
    missing_channel = (~finite_mask).astype(np.float32)
    if bool(feature_cfg.get("fill_missing_enabled", False)):
        fill_value = float(feature_cfg.get("fill_missing_value", 0.0))
        work = np.where(finite_mask, work, fill_value).astype(np.float32, copy=False)
    elif not finite_mask.all():
        work = np.where(finite_mask, work, 0.0).astype(np.float32, copy=False)

    parts: list[np.ndarray] = []
    include_level = bool(feature_cfg.get("include_level", True)) or not bool(feature_cfg.get("temporal_enabled", False))
    if include_level:
        parts.append(work)
    if bool(feature_cfg.get("temporal_enabled", False)):
        for lag in feature_cfg.get("diff_lags", []):
            lag = int(lag)
            diff = np.zeros_like(work, dtype=np.float32)
            if lag < work.shape[1]:
                diff[:, lag:, :] = work[:, lag:, :] - work[:, :-lag, :]
            parts.append(diff)
        for window in feature_cfg.get("slope_windows", []):
            parts.append(_rolling_slope_features(work, int(window)))
    if bool(feature_cfg.get("add_missing_mask", False)):
        parts.append(missing_channel)
    if not parts:
        return work
    return np.concatenate(parts, axis=2).astype(np.float32, copy=False)


def _read_factor_day(
    *,
    store: FactorStore,
    day: date,
    factor_cols: list[str],
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    missing_policy: str,
    allow_missing_values: bool,
    neutralizer: FactorNeutralizer | None = None,
) -> pd.DataFrame:
    df = store.read_day(day)
    if df.empty:
        if missing_policy in {"skip_day", "fill"}:
            return pd.DataFrame(columns=["dt", "code"] + factor_cols)
        raise MissingSequenceDataError(f"missing factor file or empty factor day: {day:%Y-%m-%d}")
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.copy()
        if "code" not in df.columns and getattr(df.index, "name", None) == "code":
            df = df.reset_index()
    if "code" not in df.columns:
        raise MissingSequenceDataError(f"factor day missing code column: {day:%Y-%m-%d}")
    if "dt" not in df.columns:
        df["dt"] = pd.Timestamp(day)
    missing_cols = [c for c in factor_cols if c not in df.columns]
    if missing_cols:
        if missing_policy == "skip_day":
            return pd.DataFrame(columns=["dt", "code"] + factor_cols)
        if missing_policy == "fill":
            df = df.copy()
            for col in missing_cols:
                df[col] = np.nan
        else:
            preview = ", ".join(missing_cols[:10])
            suffix = "" if len(missing_cols) <= 10 else f", ... +{len(missing_cols) - 10}"
            raise MissingSequenceDataError(
                f"factor day {day:%Y-%m-%d} missing {len(missing_cols)} factor column(s): {preview}{suffix}"
            )
    work = df[["dt", "code"] + factor_cols].copy()
    work["dt"] = pd.to_datetime(work["dt"], errors="coerce")
    work["code"] = work["code"].astype(str)
    for col in factor_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[factor_cols] = work[factor_cols].replace([np.inf, -np.inf], np.nan)
    drop_subset = ["dt", "code"] if allow_missing_values else ["dt", "code"] + factor_cols
    work = work.dropna(subset=drop_subset)
    if allow_missing_values:
        work = work[work[factor_cols].notna().any(axis=1)]
    if work.empty:
        if missing_policy in {"skip_day", "fill"}:
            return pd.DataFrame(columns=["dt", "code"] + factor_cols)
        raise MissingSequenceDataError(f"factor day empty after dropna: {day:%Y-%m-%d}")
    if neutralizer is not None and neutralizer.enabled:
        if winsor_lower is not None or winsor_upper is not None:
            work = _apply_winsor_zscore(
                work,
                factor_cols,
                lower_q=winsor_lower,
                upper_q=winsor_upper,
                zscore=False,
            )
        work = neutralizer.apply(work, factor_cols)
        if zscore:
            work = _apply_winsor_zscore(
                work,
                factor_cols,
                lower_q=None,
                upper_q=None,
                zscore=True,
            )
    else:
        work = _apply_winsor_zscore(
            work,
            factor_cols,
            lower_q=winsor_lower,
            upper_q=winsor_upper,
            zscore=zscore,
        )
    return work[["dt", "code"] + factor_cols].copy()


def _label_map_for_day(
    *,
    label_root: Path,
    day: date,
    factor_time: str,
    label_time: str,
    cache: dict[date, dict[str, float]],
) -> dict[str, float]:
    cached = cache.get(day)
    if cached is not None:
        return cached
    label_df = _read_label_day(
        label_root,
        day,
        factor_time=factor_time,
        label_time=label_time,
    )
    if label_df.empty or "code" not in label_df.columns or "y" not in label_df.columns:
        cache[day] = {}
        return cache[day]
    work = label_df[["code", "y"]].copy()
    work["code"] = work["code"].astype(str)
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["code", "y"]).drop_duplicates(subset=["code"], keep="last")
    cache[day] = {str(row["code"]): float(row["y"]) for _, row in work.iterrows()}
    return cache[day]


def _zscore(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if std > 0:
        return ((values - mean) / std).astype(np.float32, copy=False)
    return (values - mean).astype(np.float32, copy=False)


def _build_day_sequence(
    *,
    target_day: date,
    all_days: list[date],
    day_to_pos: dict[date, int],
    sequence_days: int,
    store: FactorStore,
    label_root: Path,
    factor_cols: list[str],
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    factor_time: str,
    label_time: str,
    missing_policy: str,
    allow_missing_values: bool,
    factor_cache: dict[date, pd.DataFrame],
    label_cache: dict[date, dict[str, float]],
    neutralizer: FactorNeutralizer | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = day_to_pos.get(target_day)
    if pos is None:
        return (
            np.empty((0, sequence_days, len(factor_cols)), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=np.float32),
        )
    start_pos = pos - int(sequence_days) + 1
    if start_pos < 0:
        return (
            np.empty((0, sequence_days, len(factor_cols)), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=np.float32),
        )
    hist_days = all_days[start_pos : pos + 1]
    frames: list[pd.DataFrame] = []
    common_codes: set[str] | None = None
    for day in hist_days:
        fdf = factor_cache.get(day)
        if fdf is None:
            fdf = _read_factor_day(
                store=store,
                day=day,
                factor_cols=factor_cols,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                missing_policy=missing_policy,
                allow_missing_values=allow_missing_values,
                neutralizer=neutralizer,
            )
            factor_cache[day] = fdf
        if fdf.empty:
            return (
                np.empty((0, sequence_days, len(factor_cols)), dtype=np.float32),
                np.array([], dtype=object),
                np.array([], dtype=np.float32),
            )
        day_frame = fdf.drop_duplicates(subset=["code"], keep="last").set_index("code")[factor_cols]
        codes = set(day_frame.index.astype(str).tolist())
        common_codes = codes if common_codes is None else common_codes.intersection(codes)
        frames.append(day_frame)
    if not common_codes:
        return (
            np.empty((0, sequence_days, len(factor_cols)), dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=np.float32),
        )
    y_map = _label_map_for_day(
        label_root=label_root,
        day=target_day,
        factor_time=factor_time,
        label_time=label_time,
        cache=label_cache,
    )
    codes = np.asarray(sorted(common_codes), dtype=object)
    x = np.stack(
        [frame.loc[codes.tolist()].to_numpy(dtype=np.float32) for frame in frames],
        axis=1,
    )
    y = np.asarray([float(y_map[c]) if c in y_map else np.nan for c in codes.tolist()], dtype=np.float32)
    return x, codes, y


def _build_split_data(
    *,
    days: Sequence[date],
    all_days: list[date],
    day_to_pos: dict[date, int],
    sequence_days: int,
    store: FactorStore,
    label_root: Path,
    factor_cols: list[str],
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    factor_time: str,
    label_time: str,
    missing_policy: str,
    require_label: bool,
    tradable_code_map: dict[date, set[str]] | None,
    tradable_strict: bool,
    label_transform: str,
    feature_cfg: dict,
    feature_names: list[str],
    factor_cache: dict[date, pd.DataFrame],
    label_cache: dict[date, dict[str, float]],
    progress_label: str,
    neutralizer: FactorNeutralizer | None = None,
) -> SequenceSplitData:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    codes_out: list[np.ndarray] = []
    total = len(days)
    for idx, day in enumerate(days, start=1):
        x_day, code_day, y_day = _build_day_sequence(
            target_day=day,
            all_days=all_days,
            day_to_pos=day_to_pos,
            sequence_days=sequence_days,
            store=store,
            label_root=label_root,
            factor_cols=factor_cols,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            missing_policy=missing_policy,
            allow_missing_values=bool(feature_cfg.get("fill_missing_enabled", False)),
            factor_cache=factor_cache,
            label_cache=label_cache,
            neutralizer=neutralizer,
        )
        if x_day.size == 0 or code_day.size == 0:
            continue
        keep = np.ones(len(code_day), dtype=bool)
        if require_label:
            keep &= np.isfinite(y_day)
            if tradable_code_map is not None:
                allowed_codes = tradable_code_map.get(day)
                if not allowed_codes:
                    if tradable_strict:
                        keep &= False
                else:
                    keep &= np.asarray([str(code) in allowed_codes for code in code_day.tolist()], dtype=bool)
        if not keep.any():
            continue
        x_keep = _transform_sequence_features(x_day[keep], feature_cfg)
        code_keep = code_day[keep]
        y_keep = y_day[keep]
        if require_label and label_transform == "zscore_day":
            y_keep = _zscore(y_keep)
        xs.append(x_keep)
        ys.append(y_keep.astype(np.float32, copy=False))
        dts.append(np.asarray([day] * len(code_keep), dtype=object))
        codes_out.append(code_keep.astype(object))
        if idx == total or idx % 20 == 0:
            print(
                f"[prep:{progress_label}] {idx}/{total} day={day} "
                f"samples={len(code_keep)} cumulative={sum(len(c) for c in codes_out)}"
            )
    if not xs:
        return _empty_split(sequence_days, len(feature_names))
    return SequenceSplitData(
        x=np.concatenate(xs, axis=0).astype(np.float32, copy=False),
        y=np.concatenate(ys, axis=0).astype(np.float32, copy=False),
        dt=np.concatenate(dts, axis=0),
        code=np.concatenate(codes_out, axis=0),
    )


def _daily_corr(dt: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, *, method: str) -> float:
    if y_true.size < 2:
        return float("nan")
    df = pd.DataFrame({"trade_date": dt, "y": y_true, "pred": y_pred})
    vals: list[float] = []
    for _, group in df.groupby("trade_date", sort=True):
        if len(group) < 2:
            continue
        corr = float(group["pred"].corr(group["y"], method=method))
        if np.isfinite(corr):
            vals.append(corr)
    return float(np.mean(vals)) if vals else float("nan")


def _eval_metrics(dt: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask].astype(float)
    pred = y_pred[mask].astype(float)
    if y.size == 0:
        return {
            "count": 0,
            "mse": float("nan"),
            "dir": float("nan"),
            "ic": float("nan"),
            "rank_ic": float("nan"),
        }
    mse = float(np.mean((y - pred) ** 2))
    direction = float((np.sign(y) == np.sign(pred)).mean())
    return {
        "count": int(y.size),
        "mse": mse,
        "dir": direction,
        "ic": _daily_corr(dt[mask], y, pred, method="pearson"),
        "rank_ic": _daily_corr(dt[mask], y, pred, method="spearman"),
    }


def _predict_numpy(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], max(1, int(batch_size))):
            batch = torch.from_numpy(x[i : i + batch_size]).float().to(device)
            out = model(batch).detach().cpu().numpy().astype(np.float32)
            preds.append(out)
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)


def _build_model(*, architecture: str, n_features: int, params: dict) -> torch.nn.Module:
    kind = str(architecture).strip().lower()
    if kind in {"lstm", "factor_lstm"}:
        return FactorLSTMModel(
            n_features=n_features,
            hidden_size=int(params.get("hidden_size", 64)),
            num_layers=int(params.get("num_layers", 1)),
            dropout=float(params.get("dropout", 0.1)),
            bidirectional=bool(params.get("bidirectional", False)),
        )
    if kind in {"cnn", "cnn1d", "factor_cnn", "factor_cnn1d"}:
        return FactorCNN1DModel(
            n_features=n_features,
            channels=int(params.get("channels", 64)),
            num_layers=int(params.get("num_layers", 3)),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )
    raise ValueError(f"unsupported torch sequence architecture: {architecture}")


def _checkpoint_score(metrics: dict, *, rank_weight: float, dir_weight: float) -> float:
    rank_ic = metrics.get("rank_ic")
    direction = metrics.get("dir")
    if np.isfinite(rank_ic) and np.isfinite(direction):
        return float(rank_weight * float(rank_ic) + dir_weight * float(direction))
    if np.isfinite(rank_ic):
        return float(rank_ic)
    mse = metrics.get("mse")
    if np.isfinite(mse):
        return float(-mse)
    return float("-inf")


def _train_one_model(
    *,
    architecture: str,
    train_data: SequenceSplitData,
    val_data: SequenceSplitData,
    model_params: dict,
    train_cfg: dict,
) -> tuple[torch.nn.Module, list[dict]]:
    if train_data.x.size == 0:
        raise RuntimeError("torch sequence train split is empty")
    device = _device_from_config(train_cfg)
    batch_size = max(1, int(train_cfg.get("batch_size", 512)))
    num_epochs = max(1, int(train_cfg.get("num_epochs", 10)))
    num_workers = max(0, int(train_cfg.get("num_workers", 0)))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    eval_every_n_epoch = max(1, int(train_cfg.get("eval_every_n_epoch", 1)))
    patience = max(0, int(train_cfg.get("early_stopping_patience", 3)))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    rank_weight = float(train_cfg.get("checkpoint_rank_weight", 1.0))
    dir_weight = float(train_cfg.get("checkpoint_dir_weight", 0.0))

    model = _build_model(
        architecture=architecture,
        n_features=int(train_data.x.shape[2]),
        params=model_params,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    loader = DataLoader(
        _SequenceDataset(train_data.x, train_data.y),
        batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle_train", True)),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    best_score = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict] = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            n = int(y_batch.numel())
            loss_sum += float(loss.item()) * n
            count += n
        train_loss = float(loss_sum / count) if count > 0 else float("nan")
        should_eval = epoch == 1 or epoch == num_epochs or (epoch % eval_every_n_epoch) == 0
        if should_eval:
            train_pred = _predict_numpy(model=model, x=train_data.x, batch_size=batch_size, device=device)
            train_metrics = _eval_metrics(train_data.dt, train_data.y, train_pred)
            if val_data.x.size > 0:
                val_pred = _predict_numpy(model=model, x=val_data.x, batch_size=batch_size, device=device)
                val_metrics = _eval_metrics(val_data.dt, val_data.y, val_pred)
            else:
                val_metrics = {"count": 0, "mse": float("nan"), "dir": float("nan"), "ic": float("nan"), "rank_ic": float("nan")}
            score = _checkpoint_score(val_metrics, rank_weight=rank_weight, dir_weight=dir_weight)
        else:
            train_metrics = {"count": int(train_data.y.size), "mse": float("nan"), "dir": float("nan"), "ic": float("nan"), "rank_ic": float("nan")}
            val_metrics = {"count": int(val_data.y.size), "mse": float("nan"), "dir": float("nan"), "ic": float("nan"), "rank_ic": float("nan")}
            score = float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_count": train_metrics["count"],
                "train_mse": train_metrics["mse"],
                "train_dir": train_metrics["dir"],
                "train_ic": train_metrics["ic"],
                "train_rank_ic": train_metrics["rank_ic"],
                "val_count": val_metrics["count"],
                "val_mse": val_metrics["mse"],
                "val_dir": val_metrics["dir"],
                "val_ic": val_metrics["ic"],
                "val_rank_ic": val_metrics["rank_ic"],
                "checkpoint_score": score,
                "eval_performed": bool(should_eval),
            }
        )
        print(
            f"epoch {epoch:03d} loss={train_loss:.6f} "
            f"train_rank_ic={train_metrics['rank_ic']:.4f} "
            f"val_rank_ic={val_metrics['rank_ic']:.4f}"
        )
        if should_eval and np.isfinite(score):
            if score > best_score + min_delta:
                best_score = float(score)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    print(f"[train] early stop epoch={epoch} patience={patience} best_score={best_score:.6f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _score_eval_daily(
    *,
    scores_df: pd.DataFrame,
    label_root: Path,
    factor_time: str,
    label_time: str,
    label_cache: dict[date, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict] = []
    if scores_df.empty:
        return pd.DataFrame()
    work = scores_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.date
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.dropna(subset=["trade_date", "code", "score"])
    for day, group in work.groupby("trade_date", sort=True):
        y_map = _label_map_for_day(
            label_root=label_root,
            day=day,
            factor_time=factor_time,
            label_time=label_time,
            cache=label_cache,
        )
        if not y_map:
            continue
        g = group.copy()
        g["code"] = g["code"].astype(str)
        g["y"] = g["code"].map(y_map)
        g = g.dropna(subset=["y", "score"])
        if g.empty:
            continue
        metrics = _eval_metrics(
            np.asarray([day] * len(g), dtype=object),
            g["y"].to_numpy(dtype=float),
            g["score"].to_numpy(dtype=float),
        )
        rows.append({"trade_date": day, **metrics})
    return pd.DataFrame(rows).sort_values("trade_date") if rows else pd.DataFrame()


def main(
    *,
    config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    execution: dict | None = None,
) -> None:
    paths_cfg = load_config_file("paths")
    cfg_file = Path(config_path) if config_path else None
    if cfg_file is None and len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            cfg_file = candidate
    cfg = _load_model_config(cfg_file)

    cfg_start = parse_date(cfg.get("start"))
    cfg_end = parse_date(cfg.get("end"))
    desired_start = parse_date(start) if start else cfg_start
    desired_end = parse_date(end) if end else cfg_end
    cutoff_day = parse_date(label_cutoff) if label_cutoff else None
    if desired_start > desired_end:
        raise ValueError("start date must be <= end date")

    model_name = str(cfg.get("model_name", "torch_sequence_model"))
    architecture = str(cfg.get("architecture", cfg.get("model_type", "lstm"))).strip().lower()
    source_cfg: dict = {}
    source_key = str(cfg.get("factor_source_config", "")).strip()
    if source_key:
        source_cfg = _load_model_config(resolve_config_path(source_key))
    factor_cols = [str(c) for c in (cfg.get("factors") or source_cfg.get("factors") or [])]
    if not factor_cols:
        raise ValueError("torch sequence config requires non-empty factors")
    sequence_days = max(2, int(cfg.get("sequence_days", 20)))
    panel_name = (
        str(cfg.get("panel_name", "")).strip()
        or str(source_cfg.get("panel_name", "")).strip()
        or make_window_label(int(cfg.get("window_minutes", source_cfg.get("window_minutes", 15))))
    )
    factor_time = str(cfg.get("factor_time", source_cfg.get("factor_time", "14:30")))
    label_time = str(cfg.get("label_time", source_cfg.get("label_time", "14:42")))
    missing_policy = str(cfg.get("missing_policy", "raise")).strip().lower()
    if missing_policy not in {"raise", "skip_day", "fill"}:
        raise ValueError("missing_policy must be one of: 'raise', 'skip_day', 'fill'")
    label_transform = str(cfg.get("label_transform", "zscore_day")).strip().lower()
    if label_transform not in {"none", "zscore_day"}:
        raise ValueError("label_transform must be 'none' or 'zscore_day'")
    winsor_lower, winsor_upper = parse_winsor_bounds(cfg.get("winsor", source_cfg.get("winsor", {})))
    zscore = bool(cfg.get("zscore", source_cfg.get("zscore", True)))
    feature_cfg = _feature_engineering_cfg(cfg.get("feature_engineering"))
    feature_names = _feature_names(factor_cols, feature_cfg)
    if missing_policy == "fill" and not bool(feature_cfg.get("fill_missing_enabled", False)):
        feature_cfg["fill_missing_enabled"] = True

    execution_cfg = dict(execution or {})
    rolling_cfg = dict(cfg.get("rolling", {}))
    rolling_enabled = bool(rolling_cfg.get("enabled", True))
    window_days = max(3, int(rolling_cfg.get("window_days", 60)))
    refit_every_n_days = max(1, int(execution_cfg.get("refit_every_n_days", cfg.get("refit_every_n_days", 20))))

    raw_root = Path(paths_cfg["raw_data_root"])
    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])
    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=int(cfg.get("window_minutes", 15)))
    neutralizer = build_neutralizer(
        cfg.get("neutralization", source_cfg.get("neutralization")),
        raw_data_root=raw_root,
    )

    lookback_count = int(window_days + sequence_days + 10)
    lookback_days = prev_trading_days_from_raw(raw_root, desired_start, lookback_count, kind="snapshot", asset="cbond")
    scan_start = lookback_days[0] if lookback_days else desired_start
    all_days = list_trading_days_from_raw(raw_root, scan_start, desired_end, kind="snapshot", asset="cbond")
    if not all_days:
        raise RuntimeError("no trading days found for torch sequence range")
    day_to_pos = {d: i for i, d in enumerate(all_days)}

    label_days_all = sorted(set(_iter_existing_label_days(label_root, scan_start, desired_end)))
    if cutoff_day is not None:
        label_days_all = [d for d in label_days_all if d <= cutoff_day]
    label_day_set = set(label_days_all)
    desired_days = [d for d in all_days if desired_start <= d <= desired_end]
    if not desired_days:
        raise RuntimeError("no target trading days in requested range")

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
    if isinstance(cfg.get("tradable_filter"), dict):
        tradable_cfg.update(cfg.get("tradable_filter"))
    if isinstance(execution_cfg.get("tradable_filter"), dict):
        tradable_cfg.update(execution_cfg.get("tradable_filter"))
    tradable_enabled = bool(tradable_cfg.get("enabled", True))
    tradable_strict = bool(tradable_cfg.get("strict", True))
    tradable_code_map = None
    if tradable_enabled:
        tradable_code_map = build_tradable_code_map(
            raw_data_root=raw_root,
            days=all_days,
            buy_twap_col=str(tradable_cfg.get("buy_twap_col", "twap_1442_1457")),
            sell_twap_col=str(tradable_cfg.get("sell_twap_col", "twap_0930_0939")),
            min_amount=float(tradable_cfg.get("min_amount", 0.0)),
            min_volume=float(tradable_cfg.get("min_volume", 0.0)),
            twap_table=str(tradable_cfg.get("twap_table", "market_cbond.daily_twap")),
            asset=str(tradable_cfg.get("asset", "cbond")),
        )
        print("[tradable] enabled", f"mapped_days={len(tradable_code_map)}", f"strict={tradable_strict}")
    else:
        print("[tradable] disabled")

    results_root = resolve_output_path(
        cfg.get("results_root"),
        default_path=paths_cfg["results_root"],
        results_root=paths_cfg["results_root"],
    )
    date_label = f"{desired_start:%Y-%m-%d}_{desired_end:%Y-%m-%d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)
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

    target_days = list(desired_days)
    if incremental_enabled and incremental_skip_existing and not score_overwrite:
        existing_days = _load_existing_score_days(score_output)
        target_days = [d for d in target_days if d not in existing_days]
        skipped = len(desired_days) - len(target_days)
        if skipped:
            print(f"[rolling] incremental skip_existing_scores=True skip={skipped} pending={len(target_days)}")
    if not target_days:
        print("[rolling] incremental: no pending target days, skip training")
        print(f"saved rolling: {out_dir}")
        print(f"saved scores: {score_output}")
        return

    train_cfg = dict(cfg.get("train", {}))
    model_params = dict(cfg.get("model_params", cfg.get("params", {})))
    print(
        "torch sequence:",
        f"model={model_name}",
        f"architecture={architecture}",
        f"device={_device_from_config(train_cfg)}",
        f"factors={len(factor_cols)}",
        f"features={len(feature_names)}",
        f"sequence_days={sequence_days}",
        f"window_days={window_days}",
        f"refit_every_n_days={refit_every_n_days}",
        f"missing_policy={missing_policy}",
        f"neutralization={bool(neutralizer is not None and neutralizer.enabled)}",
    )
    if neutralizer is not None and neutralizer.enabled:
        print("neutralization:", json.dumps(neutralizer.summary(), ensure_ascii=False))
    if cfg.get("feature_engineering"):
        print("feature engineering:", json.dumps(cfg.get("feature_engineering"), ensure_ascii=False))

    factor_cache: dict[date, pd.DataFrame] = {}
    label_cache: dict[date, dict[str, float]] = {}
    score_frames: list[pd.DataFrame] = []
    rolling_rows: list[dict] = []
    history_rows: list[dict] = []
    last_model: torch.nn.Module | None = None
    last_refit_pos: int | None = None
    last_refit_day: date | None = None

    if rolling_enabled:
        valid_positions: list[int] = []
        for d in target_days:
            pos = day_to_pos.get(d)
            if pos is None:
                continue
            if pos < max(window_days - 1, sequence_days - 1):
                continue
            valid_positions.append(pos)
        if not valid_positions:
            raise RuntimeError("rolling has no valid target day after history filtering")
        train_ratio = float(train_cfg.get("train_ratio", 0.7))
        val_ratio = float(train_cfg.get("val_ratio", 0.15))
        batch_size = max(1, int(train_cfg.get("batch_size", 512)))
        for roll_pos, idx in enumerate(valid_positions):
            roll_idx = roll_pos + 1
            test_day = all_days[idx]
            should_refit = (
                last_model is None
                or refit_every_n_days <= 1
                or last_refit_pos is None
                or (roll_pos - last_refit_pos) >= refit_every_n_days
            )
            print(
                f"[rolling] {roll_idx}/{len(valid_positions)} test_day={test_day} "
                f"refit={'Y' if should_refit else 'N'} cadence={refit_every_n_days}"
            )
            window = all_days[idx - window_days + 1 : idx + 1]
            train_days_pool = [d for d in window[:-1] if d in label_day_set]
            train_days: list[date] = []
            val_days: list[date] = []
            refit_status = "reuse"
            if should_refit:
                if len(train_days_pool) < 3:
                    if last_model is None:
                        print(f"[rolling] skip {test_day}: insufficient train days and no reusable model")
                        continue
                    refit_status = "reuse_insufficient_train"
                else:
                    train_days, val_days, _ = _split_days(train_days_pool, train_ratio, val_ratio)
                    train_data = _build_split_data(
                        days=train_days,
                        all_days=all_days,
                        day_to_pos=day_to_pos,
                        sequence_days=sequence_days,
                        store=store,
                        label_root=label_root,
                        factor_cols=factor_cols,
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper,
                        zscore=zscore,
                        factor_time=factor_time,
                        label_time=label_time,
                        missing_policy=missing_policy,
                        require_label=True,
                        tradable_code_map=tradable_code_map,
                        tradable_strict=tradable_strict,
                        label_transform=label_transform,
                        feature_cfg=feature_cfg,
                        feature_names=feature_names,
                        factor_cache=factor_cache,
                        label_cache=label_cache,
                        progress_label=f"train:{test_day}",
                        neutralizer=neutralizer,
                    )
                    val_data = _build_split_data(
                        days=val_days,
                        all_days=all_days,
                        day_to_pos=day_to_pos,
                        sequence_days=sequence_days,
                        store=store,
                        label_root=label_root,
                        factor_cols=factor_cols,
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper,
                        zscore=zscore,
                        factor_time=factor_time,
                        label_time=label_time,
                        missing_policy=missing_policy,
                        require_label=True,
                        tradable_code_map=tradable_code_map,
                        tradable_strict=tradable_strict,
                        label_transform=label_transform,
                        feature_cfg=feature_cfg,
                        feature_names=feature_names,
                        factor_cache=factor_cache,
                        label_cache=label_cache,
                        progress_label=f"val:{test_day}",
                        neutralizer=neutralizer,
                    )
                    if train_data.x.size == 0:
                        if last_model is None:
                            print(f"[rolling] skip {test_day}: empty train split and no reusable model")
                            continue
                        refit_status = "reuse_empty_train"
                    else:
                        model, hist = _train_one_model(
                            architecture=architecture,
                            train_data=train_data,
                            val_data=val_data,
                            model_params=model_params,
                            train_cfg=train_cfg,
                        )
                        last_model = model
                        last_refit_pos = roll_pos
                        last_refit_day = test_day
                        refit_status = "refit"
                        for row in hist:
                            history_rows.append({"trade_date": test_day, **row})
            if last_model is None:
                print(f"[rolling] skip {test_day}: no trained model available")
                continue
            test_data = _build_split_data(
                days=[test_day],
                all_days=all_days,
                day_to_pos=day_to_pos,
                sequence_days=sequence_days,
                store=store,
                label_root=label_root,
                factor_cols=factor_cols,
                winsor_lower=winsor_lower,
                winsor_upper=winsor_upper,
                zscore=zscore,
                factor_time=factor_time,
                label_time=label_time,
                missing_policy=missing_policy,
                require_label=False,
                tradable_code_map=None,
                tradable_strict=False,
                label_transform="none",
                feature_cfg=feature_cfg,
                feature_names=feature_names,
                factor_cache=factor_cache,
                label_cache=label_cache,
                progress_label=f"test:{test_day}",
                neutralizer=neutralizer,
            )
            if test_data.x.size == 0:
                print(f"[rolling] skip {test_day}: empty test split")
                continue
            pred = _predict_numpy(
                model=last_model,
                x=test_data.x,
                batch_size=batch_size,
                device=_device_from_config(train_cfg),
            )
            score_frames.append(
                pd.DataFrame({"trade_date": test_day, "code": test_data.code, "score": pred})
            )
            labeled = np.isfinite(test_data.y)
            metrics = _eval_metrics(test_data.dt[labeled], test_data.y[labeled], pred[labeled]) if labeled.any() else {
                "count": int(len(pred)),
                "mse": float("nan"),
                "dir": float("nan"),
                "ic": float("nan"),
                "rank_ic": float("nan"),
            }
            rolling_rows.append(
                {
                    "trade_date": test_day,
                    "refit": bool(refit_status == "refit"),
                    "refit_status": refit_status,
                    "model_source_day": last_refit_day or test_day,
                    "train_days": len(train_days),
                    "val_days": len(val_days),
                    "count": int(len(pred)),
                    **metrics,
                }
            )
            print(
                f"rolling {test_day} refit={refit_status} train_days={len(train_days)} "
                f"val_days={len(val_days)} count={len(pred)} rank_ic={metrics['rank_ic']:.4f}"
            )
    else:
        train_days_all = [d for d in desired_days if d in label_day_set]
        if len(train_days_all) < 3:
            raise RuntimeError("not enough labeled days for non-rolling training")
        train_ratio = float(train_cfg.get("train_ratio", 0.7))
        val_ratio = float(train_cfg.get("val_ratio", 0.15))
        train_days, val_days, _ = _split_days(train_days_all, train_ratio, val_ratio)
        train_data = _build_split_data(
            days=train_days,
            all_days=all_days,
            day_to_pos=day_to_pos,
            sequence_days=sequence_days,
            store=store,
            label_root=label_root,
            factor_cols=factor_cols,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            missing_policy=missing_policy,
            require_label=True,
            tradable_code_map=tradable_code_map,
            tradable_strict=tradable_strict,
            label_transform=label_transform,
            feature_cfg=feature_cfg,
            feature_names=feature_names,
            factor_cache=factor_cache,
            label_cache=label_cache,
            progress_label="train_full",
            neutralizer=neutralizer,
        )
        val_data = _build_split_data(
            days=val_days,
            all_days=all_days,
            day_to_pos=day_to_pos,
            sequence_days=sequence_days,
            store=store,
            label_root=label_root,
            factor_cols=factor_cols,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            factor_time=factor_time,
            label_time=label_time,
            missing_policy=missing_policy,
            require_label=True,
            tradable_code_map=tradable_code_map,
            tradable_strict=tradable_strict,
            label_transform=label_transform,
            feature_cfg=feature_cfg,
            feature_names=feature_names,
            factor_cache=factor_cache,
            label_cache=label_cache,
            progress_label="val_full",
            neutralizer=neutralizer,
        )
        last_model, hist = _train_one_model(
            architecture=architecture,
            train_data=train_data,
            val_data=val_data,
            model_params=model_params,
            train_cfg=train_cfg,
        )
        history_rows.extend({"trade_date": "", **row} for row in hist)

    if not score_frames:
        raise RuntimeError("torch sequence produced no scores")
    all_scores = pd.concat(score_frames, ignore_index=True)
    write_scores_by_date(
        score_output,
        all_scores,
        overwrite=score_overwrite,
        dedupe=score_dedupe,
    )
    if rolling_rows:
        pd.DataFrame(rolling_rows).to_csv(out_dir / "rolling_metrics.csv", index=False)
    if history_rows:
        pd.DataFrame(history_rows).to_csv(out_dir / "metrics_iter.csv", index=False)
    eval_daily = _score_eval_daily(
        scores_df=all_scores,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        label_cache=label_cache,
    )
    if not eval_daily.empty:
        eval_daily.to_csv(out_dir / "score_eval_daily.csv", index=False)
    summary = {
        "days": int(len(eval_daily)),
        "score_days": int(all_scores["trade_date"].nunique()),
        "score_rows": int(len(all_scores)),
        "rank_ic_mean": float(pd.to_numeric(eval_daily.get("rank_ic", pd.Series(dtype=float)), errors="coerce").mean()) if not eval_daily.empty else float("nan"),
        "ic_mean": float(pd.to_numeric(eval_daily.get("ic", pd.Series(dtype=float)), errors="coerce").mean()) if not eval_daily.empty else float("nan"),
        "all_equal_days": int(
            all_scores.groupby("trade_date")["score"].nunique(dropna=True).le(1).sum()
        ),
    }
    (out_dir / "score_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if last_model is not None:
        torch.save(last_model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "features.json").write_text(
        json.dumps(
            {
                "architecture": architecture,
                "factor_count": len(factor_cols),
                "factors": factor_cols,
                "feature_count": len(feature_names),
                "features": feature_names,
                "sequence_days": sequence_days,
                "label_transform": label_transform,
                "feature_engineering": cfg.get("feature_engineering", {}),
                "neutralization": neutralizer.summary() if neutralizer is not None else {"enabled": False},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[score_guard]", f"all_equal_days={summary['all_equal_days']}", f"score_days={summary['score_days']}")
    print(f"saved rolling: {out_dir}")
    print(f"saved scores: {score_output}")


if __name__ == "__main__":
    main()
