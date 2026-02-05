
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time as dt_time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as exc:  # pragma: no cover
    lgb = None

from cbond_on.factors.storage import FactorStore


@dataclass
class SplitData:
    x: pd.DataFrame
    y: pd.Series
    dt: pd.Series
    code: pd.Series


def _iter_existing_label_days(label_root: Path, start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        month = f"{current.year:04d}-{current.month:02d}"
        filename = f"{current.strftime('%Y%m%d')}.parquet"
        path = label_root / month / filename
        if path.exists():
            yield current
        current = current + pd.Timedelta(days=1)


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

    return df.groupby("dt", group_keys=False).apply(_process)


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
) -> SplitData:
    frames: list[pd.DataFrame] = []
    for day in days:
        fdf = factor_store.read_day(day)
        if fdf.empty:
            continue
        if not isinstance(fdf.index, pd.MultiIndex):
            fdf = fdf.reset_index().set_index(["dt", "code"])
        fdf = fdf.reset_index()
        label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
        if label_df.empty:
            continue
        if "dt" not in label_df.columns:
            continue
        label_df = label_df[["dt", "code", "y"]].dropna()
        merged = fdf.merge(label_df, on=["dt", "code"], how="inner")
        if merged.empty:
            continue
        merged = merged.dropna(subset=factor_cols + ["y"])
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
    return df.groupby("dt").apply(_calc)


def _rank_ic_by_day(df: pd.DataFrame, factor_col: str) -> pd.Series:
    def _calc(group: pd.DataFrame) -> float:
        g = group[[factor_col, "y"]].dropna()
        if len(g) < 2:
            return np.nan
        return g[factor_col].corr(g["y"], method="spearman")
    return df.groupby("dt").apply(_calc)


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
) -> tuple[object, dict]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    params = dict(lgbm_params)
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {}
    if early_stopping_rounds is not None and val.x is not None and not val.x.empty:
        fit_kwargs = {
            "eval_set": [(val.x, val.y)],
            "eval_metric": "l2",
            "early_stopping_rounds": int(early_stopping_rounds),
            "verbose": False,
        }
    model.fit(train.x, train.y, **fit_kwargs)
    return model, params
