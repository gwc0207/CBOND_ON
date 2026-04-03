
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _LIGHTGBM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    lgb = None
    _LIGHTGBM_IMPORT_ERROR = exc

from cbond_on.factors.storage import FactorStore


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


def build_dataset(
    *,
    factor_store: FactorStore,
    label_root: Path,
    days: Sequence[date],
    factor_cols: list[str],
    min_count: int,
    winsor_lower: float,
    winsor_upper: float,
    zscore: bool,
    factor_time: str,
    label_time: str,
    require_label: bool = True,
) -> SplitData:
    frames: list[pd.DataFrame] = []
    for day in days:
        fdf = factor_store.read_day(day)
        if fdf.empty:
            continue
        if not isinstance(fdf.index, pd.MultiIndex):
            fdf = fdf.reset_index().set_index(["dt", "code"])
        fdf = fdf.reset_index()
        # Backward compatibility: old factor files may miss newly added columns.
        # Skip these days instead of raising KeyError in dropna(subset=...).
        missing_cols = [c for c in factor_cols if c not in fdf.columns]
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
        if require_label:
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
    data = _apply_winsor_zscore(
        data,
        factor_cols,
        lower_q=winsor_lower,
        upper_q=winsor_upper,
        zscore=zscore,
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
