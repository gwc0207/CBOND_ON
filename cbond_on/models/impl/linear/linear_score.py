from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cbond_on.factors.storage import FactorStore
from cbond_on.models.score_io import write_scores_by_date


@dataclass
class ScoreResult:
    scores: pd.DataFrame
    weights_history: pd.DataFrame


def _iter_existing_label_days(label_root: Path, start: date, end: date) -> list[date]:
    current = start
    days: list[date] = []
    while current <= end:
        month = f"{current.year:04d}-{current.month:02d}"
        filename = f"{current.strftime('%Y%m%d')}.parquet"
        if (label_root / month / filename).exists():
            days.append(current)
        current = current + pd.Timedelta(days=1)
    return days


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


def _apply_winsor_zscore(
    df: pd.DataFrame,
    factor_cols: list[str],
    *,
    lower_q: float,
    upper_q: float,
    zscore: bool,
) -> pd.DataFrame:
    def _process(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["dt"] = group.name
        for col in factor_cols:
            s = g[col]
            if s.isna().all():
                continue
            lo = s.quantile(lower_q)
            hi = s.quantile(upper_q)
            s = s.clip(lower=lo, upper=hi)
            if zscore:
                mean = s.mean()
                std = s.std(ddof=0)
                if std > 0:
                    s = (s - mean) / std
                else:
                    s = s - mean
            g[col] = s
        return g

    return df.groupby("dt", group_keys=False).apply(_process, include_groups=False)


def _fit_weights(
    train_df: pd.DataFrame,
    factor_cols: list[str],
    *,
    alpha: float,
    device: str = "cpu",
    gpu_fallback_to_cpu: bool = True,
    gpu_state: dict[str, Any] | None = None,
) -> pd.Series | None:
    if train_df.empty:
        return None
    X = train_df[factor_cols]
    y = train_df["y"]
    if X.empty or y.empty:
        return None

    use_gpu = str(device or "cpu").strip().lower() in {"gpu", "cuda"}
    if use_gpu:
        try:
            import cupy as cp
            from cuml.linear_model import Ridge as CuRidge

            model = CuRidge(alpha=float(alpha), fit_intercept=True)
            x_gpu = cp.asarray(X.to_numpy(dtype=np.float32))
            y_gpu = cp.asarray(y.to_numpy(dtype=np.float32))
            model.fit(x_gpu, y_gpu)
            coef = model.coef_
            if hasattr(coef, "to_numpy"):
                coef_np = coef.to_numpy()
            elif hasattr(coef, "get"):
                coef_np = coef.get()
            else:
                coef_np = np.asarray(coef)
            return pd.Series(np.asarray(coef_np).reshape(-1), index=factor_cols, dtype=float)
        except Exception as exc:
            if not gpu_fallback_to_cpu:
                return None
            if gpu_state is not None and not bool(gpu_state.get("warned", False)):
                print(
                    "[linear] GPU requested but cuML is unavailable; fallback to CPU:",
                    f"{type(exc).__name__}: {exc}",
                )
                gpu_state["warned"] = True

    try:
        from sklearn.linear_model import Ridge
    except Exception:
        return None
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(X, y)
    return pd.Series(model.coef_, index=factor_cols, dtype=float)


def _normalize_weights(weights: pd.Series, method: str, max_weight: float) -> pd.Series:
    w = weights.clip(-max_weight, max_weight)
    if method == "l1":
        denom = w.abs().sum()
        if denom > 0:
            w = w / denom
    return w


def _score_day(
    day: date,
    factor_store: FactorStore,
    label_root: Path,
    *,
    factor_cols: list[str],
    winsor_lower: float,
    winsor_upper: float,
    zscore: bool,
    min_count: int,
    factor_time: str,
    label_time: str,
) -> pd.DataFrame:
    fdf = factor_store.read_day(day)
    if fdf.empty:
        return pd.DataFrame()
    if not isinstance(fdf.index, pd.MultiIndex):
        fdf = fdf.reset_index().set_index(["dt", "code"])
    fdf = fdf.reset_index()
    label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
    if label_df.empty or "dt" not in label_df.columns:
        return pd.DataFrame()
    label_df = label_df[["dt", "code", "y"]].dropna()
    merged = fdf.merge(label_df, on=["dt", "code"], how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged = merged.dropna(subset=factor_cols + ["y"])
    if merged.empty:
        return pd.DataFrame()
    counts = merged.groupby("dt")["code"].transform("size")
    merged = merged[counts >= min_count]
    if merged.empty:
        return pd.DataFrame()
    merged = _apply_winsor_zscore(
        merged,
        factor_cols,
        lower_q=winsor_lower,
        upper_q=winsor_upper,
        zscore=zscore,
    )
    return merged[["dt", "code"] + factor_cols + ["y"]]


def run_linear_score(
    *,
    factor_root: Path,
    label_root: Path,
    start: date,
    end: date,
    factor_cols: list[str],
    panel_name: str,
    window_minutes: int,
    factor_time: str,
    label_time: str,
    min_count: int,
    winsor_lower: float,
    winsor_upper: float,
    zscore: bool,
    lookback_days: int,
    refit_freq: int,
    regression_alpha: float,
    weight_source: str,
    fallback: str,
    max_weight: float,
    normalize_weights: str,
    manual_weights: pd.Series,
    device: str = "cpu",
    gpu_fallback_to_cpu: bool = True,
) -> ScoreResult:
    days = _iter_existing_label_days(label_root, start, end)
    if not days:
        return ScoreResult(scores=pd.DataFrame(), weights_history=pd.DataFrame())
    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)
    weights = manual_weights.copy()
    last_refit_idx: int | None = None
    score_rows: list[dict] = []
    weight_rows: list[dict] = []
    gpu_state: dict[str, Any] = {"warned": False}

    for idx, day in enumerate(days):
        day_df = _score_day(
            day,
            store,
            label_root,
            factor_cols=factor_cols,
            winsor_lower=winsor_lower,
            winsor_upper=winsor_upper,
            zscore=zscore,
            min_count=min_count,
            factor_time=factor_time,
            label_time=label_time,
        )
        if day_df.empty:
            continue

        if weight_source == "regression":
            need_refit = last_refit_idx is None or (idx - last_refit_idx) >= refit_freq
            if need_refit:
                train_days = days[max(0, idx - lookback_days):idx]
                train_frames = []
                for td in train_days:
                    tdf = _score_day(
                        td,
                        store,
                        label_root,
                        factor_cols=factor_cols,
                        winsor_lower=winsor_lower,
                        winsor_upper=winsor_upper,
                        zscore=zscore,
                        min_count=min_count,
                        factor_time=factor_time,
                        label_time=label_time,
                    )
                    if not tdf.empty:
                        train_frames.append(tdf)
                train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
                fit = _fit_weights(
                    train_df,
                    factor_cols,
                    alpha=regression_alpha,
                    device=device,
                    gpu_fallback_to_cpu=gpu_fallback_to_cpu,
                    gpu_state=gpu_state,
                )
                if fit is None:
                    if fallback == "equal":
                        weights = pd.Series(1.0, index=factor_cols) / len(factor_cols)
                    else:
                        weights = manual_weights.copy()
                else:
                    weights = fit
                weights = _normalize_weights(weights, normalize_weights, max_weight)
                for factor, weight in weights.items():
                    weight_rows.append(
                        {"trade_date": day, "factor": factor, "weight": float(weight)}
                    )
                last_refit_idx = idx

        work = day_df[factor_cols].copy()
        valid_mask = work.notna()
        denom = valid_mask.mul(weights.abs(), axis=1).sum(axis=1)
        weighted = work.mul(weights, axis=1).sum(axis=1)
        composite = weighted.where(denom > 0).div(denom)
        for code, score in zip(day_df["code"], composite, strict=False):
            if pd.isna(score):
                continue
            score_rows.append(
                {"trade_date": day, "code": code, "score": float(score)}
            )

    return ScoreResult(
        scores=pd.DataFrame(score_rows),
        weights_history=pd.DataFrame(weight_rows),
    )


def write_linear_outputs(
    *,
    result: ScoreResult,
    score_path: Path,
    weights_path: Path | None,
    meta_path: Path | None,
    meta_payload: dict,
    overwrite: bool,
) -> None:
    write_scores_by_date(
        score_path,
        result.scores,
        overwrite=overwrite,
        dedupe=True,
    )

    if weights_path is not None:
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and weights_path.exists():
            weights_path.unlink()
        if not result.weights_history.empty:
            result.weights_history.to_csv(weights_path, index=False)

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, ensure_ascii=False, indent=2, default=str)
