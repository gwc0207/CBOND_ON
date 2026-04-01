from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.core.naming import make_window_label
from cbond_on.models.impl.lgbm.trainer import _iter_existing_label_days, _read_label_day, _split_days
from cbond_on.models.impl.lob.lob_st import LOBSpatioTemporalModel
from cbond_on.models.score_io import load_scores_by_date, write_scores_by_date


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    dt: np.ndarray
    code: np.ndarray


class _NumpyDataset(Dataset):
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
        return load_config_file("models/lob/model")
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


def _iter_existing_panel_days(panel_root: Path) -> list[date]:
    if not panel_root.exists():
        return []
    days: set[date] = set()
    for path in panel_root.glob("*/*.parquet"):
        stem = path.stem.strip()
        if len(stem) != 8 or not stem.isdigit():
            continue
        try:
            day = datetime.strptime(stem, "%Y%m%d").date()
        except Exception:
            continue
        days.add(day)
    return sorted(days)


def _panel_day_path(panel_root: Path, day: date) -> Path:
    month = f"{day.year:04d}-{day.month:02d}"
    filename = f"{day.strftime('%Y%m%d')}.parquet"
    return panel_root / month / filename


def _lob_columns(depth_levels: int) -> tuple[list[str], list[str]]:
    if depth_levels % 2 != 0:
        raise ValueError("depth_levels must be even (ask+bid)")
    half = depth_levels // 2
    ask_price = [f"ask_price{i}" for i in range(1, half + 1)]
    bid_price = [f"bid_price{i}" for i in range(1, half + 1)]
    ask_volume = [f"ask_volume{i}" for i in range(1, half + 1)]
    bid_volume = [f"bid_volume{i}" for i in range(1, half + 1)]
    price_cols = ask_price[::-1] + bid_price
    volume_cols = ask_volume[::-1] + bid_volume
    return price_cols, volume_cols


def _resolve_device(train_cfg: dict) -> torch.device:
    requested = str(train_cfg.get("device", "cuda")).strip().lower()
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


def _empty_day_samples(seq_len: int, depth_levels: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, 2, max(seq_len, 0), max(depth_levels, 0)), dtype=np.float32),
        np.array([], dtype=object),
    )


def _resolve_preprocess_backend(train_cfg: dict) -> tuple[str, str, bool]:
    requested = str(train_cfg.get("preprocess_backend", "auto")).strip().lower()
    fallback_to_cpu = bool(train_cfg.get("preprocess_fallback_to_cpu", True))
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            f"invalid train.preprocess_backend={requested}; expected one of auto/cpu/gpu"
        )
    if requested == "cpu":
        return "cpu", "requested_cpu", fallback_to_cpu

    if not torch.cuda.is_available():
        reason = "cuda_unavailable"
        if requested == "gpu" and not fallback_to_cpu:
            raise RuntimeError(f"gpu preprocess requested but unavailable: {reason}")
        return "cpu", reason, fallback_to_cpu

    try:
        import cupy as cp  # type: ignore
        import cudf  # type: ignore

        _ = cp.asarray([1.0], dtype=cp.float32)
        _ = cudf.DataFrame({"_k": [0], "_v": [1.0]})
    except Exception as exc:
        reason = f"gpu_preprocess_unavailable:{type(exc).__name__}:{exc}"
        if requested == "gpu" and not fallback_to_cpu:
            raise RuntimeError(f"gpu preprocess requested but unavailable: {reason}") from exc
        return "cpu", reason, fallback_to_cpu

    return "gpu", "cudf_cupy_ready", fallback_to_cpu


def _normalize_x_batch(x: torch.Tensor, method: str) -> torch.Tensor:
    if method == "zscore_sample":
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))
        return (x - mean) / std
    if method == "minmax_sample":
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        denom = torch.where((x_max - x_min) > 0, (x_max - x_min), torch.ones_like(x_max))
        return (x - x_min) / denom
    return x


def _normalize_y_batch(y: torch.Tensor, method: str) -> torch.Tensor:
    if method == "zscore_batch":
        mean = y.mean()
        std = y.std()
        if std > 0:
            return (y - mean) / std
        return y - mean
    return y


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    var = float(np.var(y_true))
    if var <= 1e-12:
        return float("nan")
    mse = float(np.mean((y_true - y_pred) ** 2))
    return float(1.0 - (mse / var))


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
    if y_true.size == 0:
        return {
            "count": 0,
            "mse": float("nan"),
            "r2": float("nan"),
            "dir": float("nan"),
            "ic": float("nan"),
            "rank_ic": float("nan"),
        }
    mse = float(np.mean((y_true - y_pred) ** 2))
    r2 = _safe_r2(y_true, y_pred)
    direction = float((np.sign(y_true) == np.sign(y_pred)).mean())
    ic = _daily_corr(dt, y_true, y_pred, method="pearson")
    rank_ic = _daily_corr(dt, y_true, y_pred, method="spearman")
    return {
        "count": int(y_true.size),
        "mse": mse,
        "r2": r2,
        "dir": direction,
        "ic": ic,
        "rank_ic": rank_ic,
    }


def _predict_numpy(
    *,
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
    normalize_x: bool,
    x_norm_method: str,
) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x.shape[0], max(1, int(batch_size))):
            batch = torch.from_numpy(x[i : i + batch_size]).float().to(device)
            if normalize_x:
                batch = _normalize_x_batch(batch, x_norm_method)
            out = model(batch).detach().cpu().numpy().astype(np.float32)
            preds.append(out)
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)


def _score_for_checkpoint(metrics: dict, *, dir_weight: float, rank_weight: float) -> float:
    val_dir = metrics.get("dir")
    val_rank_ic = metrics.get("rank_ic")
    if np.isfinite(val_dir) and np.isfinite(val_rank_ic):
        return float(dir_weight * float(val_dir) + rank_weight * float(val_rank_ic))
    if np.isfinite(val_dir):
        return float(val_dir)
    if np.isfinite(val_rank_ic):
        return float(val_rank_ic)
    val_mse = metrics.get("mse")
    if np.isfinite(val_mse):
        return float(-val_mse)
    return float("-inf")


def _train_one_model(
    *,
    train_data: SplitData,
    val_data: SplitData,
    params: dict,
    train_cfg: dict,
) -> tuple[LOBSpatioTemporalModel, list[dict]]:
    if train_data.x.size == 0:
        raise RuntimeError("lob train split is empty")

    device = _resolve_device(train_cfg)
    batch_size = max(1, int(train_cfg.get("batch_size", 8)))
    num_workers = max(0, int(train_cfg.get("num_workers", 0)))
    num_epochs = max(1, int(train_cfg.get("num_epochs", 10)))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    log_batch_progress = bool(train_cfg.get("log_batch_progress", False))
    batch_log_every = max(1, int(train_cfg.get("batch_log_every", 50)))
    normalize_x = bool(train_cfg.get("normalize_x", False))
    normalize_y = bool(train_cfg.get("normalize_y", False))
    x_norm_method = str(train_cfg.get("x_norm_method", "zscore_sample"))
    y_norm_method = str(train_cfg.get("y_norm_method", "zscore_batch"))
    dir_weight = float(train_cfg.get("checkpoint_dir_weight", 1.0))
    rank_weight = float(train_cfg.get("checkpoint_corr_weight", 0.3))

    model = LOBSpatioTemporalModel(
        depth_levels=int(params.get("depth_levels", 10)),
        rbf_num_bases=int(params.get("rbf_num_bases", 16)),
        rbf_sigma=float(params.get("rbf_sigma", 0.5)),
        lstm_hidden_size=int(params.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(params.get("lstm_num_layers", 1)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(
        _NumpyDataset(train_data.x, train_data.y),
        batch_size=batch_size,
        shuffle=bool(train_cfg.get("shuffle_train", True)),
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    history: list[dict] = []
    best_score = float("-inf")
    best_state: dict | None = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        total_batches = len(train_loader)

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if normalize_x:
                x_batch = _normalize_x_batch(x_batch, x_norm_method)
            if normalize_y:
                y_batch = _normalize_y_batch(y_batch, y_norm_method)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_n = int(y_batch.numel())
            train_loss_sum += float(loss.item()) * batch_n
            train_count += batch_n

            if log_batch_progress and (
                batch_idx == 1 or batch_idx == total_batches or (batch_idx % batch_log_every) == 0
            ):
                mean_loss = float(train_loss_sum / train_count) if train_count > 0 else float("nan")
                print(
                    f"epoch {epoch:03d} batch {batch_idx}/{total_batches} "
                    f"loss={float(loss.item()):.6f} mean_loss={mean_loss:.6f} "
                    f"samples={train_count}"
                )

        train_loss = float(train_loss_sum / train_count) if train_count > 0 else float("nan")
        train_pred = _predict_numpy(
            model=model,
            x=train_data.x,
            batch_size=batch_size,
            device=device,
            normalize_x=normalize_x,
            x_norm_method=x_norm_method,
        )
        train_metrics = _eval_metrics(train_data.dt, train_data.y, train_pred)

        if val_data.x.size > 0:
            val_pred = _predict_numpy(
                model=model,
                x=val_data.x,
                batch_size=batch_size,
                device=device,
                normalize_x=normalize_x,
                x_norm_method=x_norm_method,
            )
            val_metrics = _eval_metrics(val_data.dt, val_data.y, val_pred)
        else:
            val_metrics = {
                "count": 0,
                "mse": float("nan"),
                "r2": float("nan"),
                "dir": float("nan"),
                "ic": float("nan"),
                "rank_ic": float("nan"),
            }

        checkpoint_score = _score_for_checkpoint(
            val_metrics, dir_weight=dir_weight, rank_weight=rank_weight
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse": train_metrics["mse"],
                "train_r2": train_metrics["r2"],
                "train_dir": train_metrics["dir"],
                "train_rank_ic": train_metrics["rank_ic"],
                "val_mse": val_metrics["mse"],
                "val_r2": val_metrics["r2"],
                "val_dir": val_metrics["dir"],
                "val_rank_ic": val_metrics["rank_ic"],
                "checkpoint_score": checkpoint_score,
            }
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} train_rank_ic={train_metrics['rank_ic']:.4f} "
            f"val_rank_ic={val_metrics['rank_ic']:.4f} train_r2={train_metrics['r2']:.4f} "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if np.isfinite(checkpoint_score) and checkpoint_score > best_score:
            best_score = checkpoint_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _write_score_report(
    *,
    out_dir: Path,
    scores_df: pd.DataFrame,
    eval_daily: pd.DataFrame,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip lob score report image: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax0 = axes[0, 0]
    if not eval_daily.empty and "trade_date" in eval_daily.columns:
        work = eval_daily.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work[work["trade_date"].notna()].sort_values("trade_date")
        ax0.plot(work["trade_date"], pd.to_numeric(work["rank_ic"], errors="coerce"), label="rank_ic")
        ax0.plot(work["trade_date"], pd.to_numeric(work["ic"], errors="coerce"), label="ic")
        ax0.axhline(0.0, color="black", linewidth=0.8)
        ax0.legend(loc="best")
    else:
        ax0.text(0.5, 0.5, "No labeled days", ha="center", va="center", transform=ax0.transAxes)
    ax0.set_title("Daily IC / RankIC")
    ax0.grid(alpha=0.25)

    ax1 = axes[0, 1]
    if not eval_daily.empty and "trade_date" in eval_daily.columns:
        work = eval_daily.copy()
        work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce")
        work = work[work["trade_date"].notna()].sort_values("trade_date")
        ax1.plot(work["trade_date"], pd.to_numeric(work["dir"], errors="coerce"), label="dir_acc")
        ax1.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(loc="best")
    else:
        ax1.text(0.5, 0.5, "No direction stats", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Daily Direction Accuracy")
    ax1.grid(alpha=0.25)

    ax2 = axes[1, 0]
    if not scores_df.empty:
        score_vals = pd.to_numeric(scores_df["score"], errors="coerce").dropna()
        if not score_vals.empty:
            ax2.hist(score_vals, bins=60, color="steelblue", alpha=0.85)
    ax2.set_title("Score Distribution")
    ax2.grid(alpha=0.25)

    ax3 = axes[1, 1]
    if not scores_df.empty:
        count_df = (
            scores_df.assign(trade_date=pd.to_datetime(scores_df["trade_date"], errors="coerce"))
            .dropna(subset=["trade_date"])
            .groupby("trade_date", as_index=False)["code"]
            .count()
            .rename(columns={"code": "count"})
            .sort_values("trade_date")
        )
        if not count_df.empty:
            ax3.bar(count_df["trade_date"], count_df["count"], color="seagreen", alpha=0.85)
    ax3.set_title("Scored Universe Size")
    ax3.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "score_report.png", dpi=150)
    plt.close(fig)


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
    if label_df.empty:
        cache[day] = {}
        return cache[day]
    work = label_df.copy()
    if "code" not in work.columns or "y" not in work.columns:
        cache[day] = {}
        return cache[day]
    work["code"] = work["code"].astype(str)
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work = work.dropna(subset=["code", "y"]).drop_duplicates(subset=["code"], keep="last")
    mapping = {str(row["code"]): float(row["y"]) for _, row in work.iterrows()}
    cache[day] = mapping
    return mapping


def _build_day_base_samples_cpu(
    *,
    panel_root: Path,
    day: date,
    depth_levels: int,
    seq_len: int,
    min_seq_len: int,
    panel_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not panel_path.exists():
        return _empty_day_samples(seq_len, depth_levels)

    df = pd.read_parquet(panel_path)
    if df.empty:
        return _empty_day_samples(seq_len, depth_levels)

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if "code" not in df.columns:
        raise RuntimeError(f"panel missing code column: {panel_path}")
    if "seq" not in df.columns:
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df = df.sort_values(["code", "trade_time"])
        else:
            df = df.sort_values(["code"])
        df["seq"] = df.groupby("code", sort=False).cumcount()
    else:
        df = df.sort_values(["code", "seq"])

    price_cols, volume_cols = _lob_columns(depth_levels)
    required = price_cols + volume_cols
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"panel missing lob columns for {day}: {missing_cols}"
        )

    frames: list[np.ndarray] = []
    codes: list[str] = []
    target_len = max(0, int(seq_len))
    min_len = max(1, int(min_seq_len))
    for code, group in df.groupby("code", sort=False):
        g = group
        if target_len > 0:
            if len(g) < target_len:
                continue
            g = g.tail(target_len)
        if len(g) < min_len:
            continue
        price = g[price_cols].to_numpy(dtype=np.float32)
        volume = g[volume_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(price).all() or not np.isfinite(volume).all():
            continue
        x = np.stack([price, volume], axis=0)
        frames.append(x)
        codes.append(str(code))

    if frames:
        x_day = np.stack(frames, axis=0)
        code_day = np.asarray(codes, dtype=object)
    else:
        return _empty_day_samples(target_len, depth_levels)

    return x_day, code_day


def _build_day_base_samples_gpu(
    *,
    panel_root: Path,
    day: date,
    depth_levels: int,
    seq_len: int,
    min_seq_len: int,
    panel_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import cupy as cp  # type: ignore
    import cudf  # type: ignore

    if not panel_path.exists():
        return _empty_day_samples(seq_len, depth_levels)

    try:
        gdf = cudf.read_parquet(panel_path, use_pandas_metadata=True)
    except TypeError:
        gdf = cudf.read_parquet(panel_path)
    if len(gdf) == 0:
        return _empty_day_samples(seq_len, depth_levels)

    if "code" not in gdf.columns or "seq" not in gdf.columns:
        try:
            idx = gdf.index
            idx_names = list(getattr(idx, "names", []) or [])
            if not idx_names:
                idx_name = getattr(idx, "name", None)
                if idx_name is not None:
                    idx_names = [idx_name]
            if ("code" in idx_names) or ("seq" in idx_names) or ("dt" in idx_names):
                gdf = gdf.reset_index()
        except Exception:
            pass

    if "code" not in gdf.columns:
        raise RuntimeError(f"panel missing code column after gpu reset_index: {panel_path}")
    if "seq" not in gdf.columns:
        if "trade_time" in gdf.columns:
            try:
                gdf["trade_time"] = cudf.to_datetime(gdf["trade_time"], errors="coerce")
            except Exception:
                pass
            gdf = gdf.sort_values(["code", "trade_time"])
        else:
            gdf = gdf.sort_values(["code"])
        gdf["seq"] = gdf.groupby("code").cumcount()
    else:
        gdf = gdf.sort_values(["code", "seq"])

    price_cols, volume_cols = _lob_columns(depth_levels)
    required = price_cols + volume_cols
    missing_cols = [c for c in required if c not in gdf.columns]
    if missing_cols:
        raise RuntimeError(f"panel missing lob columns for {day}: {missing_cols}")

    target_len = max(0, int(seq_len))
    min_len = max(1, int(min_seq_len))
    min_required = max(min_len, target_len if target_len > 0 else min_len)

    counts = gdf.groupby("code").size().reset_index(name="_cnt")
    keep_codes = counts[counts["_cnt"] >= min_required][["code"]]
    if len(keep_codes) == 0:
        return _empty_day_samples(target_len, depth_levels)

    gdf = gdf.merge(keep_codes, on="code", how="inner")
    if target_len > 0:
        gdf = gdf.merge(counts[["code", "_cnt"]], on="code", how="left")
        gdf["_rn"] = gdf.groupby("code").cumcount()
        gdf = gdf[gdf["_rn"] >= (gdf["_cnt"] - target_len)]
        gdf = gdf.drop(columns=["_rn", "_cnt"])

    gdf = gdf.sort_values(["code", "seq"])
    codes_order = (
        gdf["code"]
        .astype("str")
        .drop_duplicates()
        .to_pandas()
        .astype(str)
        .tolist()
    )
    if not codes_order:
        return _empty_day_samples(target_len, depth_levels)

    if target_len <= 0:
        raise RuntimeError("gpu preprocess requires seq_len > 0")

    rows = len(gdf)
    expected_rows = len(codes_order) * target_len
    if rows != expected_rows:
        raise RuntimeError(
            f"inconsistent grouped rows for gpu preprocess: rows={rows}, "
            f"codes={len(codes_order)}, target_len={target_len}"
        )

    price = gdf[price_cols].astype("float32").to_cupy().reshape((len(codes_order), target_len, depth_levels))
    volume = (
        gdf[volume_cols]
        .astype("float32")
        .to_cupy()
        .reshape((len(codes_order), target_len, depth_levels))
    )
    valid_mask = cp.isfinite(price).all(axis=(1, 2)) & cp.isfinite(volume).all(axis=(1, 2))
    if not bool(valid_mask.any()):
        return _empty_day_samples(target_len, depth_levels)

    x = cp.stack([price, volume], axis=1)
    valid_mask_np = cp.asnumpy(valid_mask)
    kept_codes = np.asarray(
        [codes_order[i] for i, ok in enumerate(valid_mask_np) if bool(ok)],
        dtype=object,
    )
    x_day = cp.asnumpy(x[valid_mask]).astype(np.float32, copy=False)
    return x_day, kept_codes


def _build_day_base_samples(
    *,
    panel_root: Path,
    day: date,
    depth_levels: int,
    seq_len: int,
    min_seq_len: int,
    base_cache: dict[date, tuple[np.ndarray, np.ndarray]],
    preprocess_backend: str,
    preprocess_gpu_fallback_to_cpu: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cached = base_cache.get(day)
    if cached is not None:
        return cached

    panel_path = _panel_day_path(panel_root, day)
    if preprocess_backend == "gpu":
        try:
            out = _build_day_base_samples_gpu(
                panel_root=panel_root,
                day=day,
                depth_levels=depth_levels,
                seq_len=seq_len,
                min_seq_len=min_seq_len,
                panel_path=panel_path,
            )
            base_cache[day] = out
            return out
        except Exception as exc:
            if not preprocess_gpu_fallback_to_cpu:
                raise
            print(
                f"[preprocess] gpu fallback to cpu for day={day}: "
                f"{type(exc).__name__}: {exc}"
            )

    out = _build_day_base_samples_cpu(
        panel_root=panel_root,
        day=day,
        depth_levels=depth_levels,
        seq_len=seq_len,
        min_seq_len=min_seq_len,
        panel_path=panel_path,
    )

    base_cache[day] = out
    return out


def _build_split_data(
    *,
    days: Sequence[date],
    panel_root: Path,
    label_root: Path,
    factor_time: str,
    label_time: str,
    depth_levels: int,
    seq_len: int,
    min_seq_len: int,
    require_label: bool,
    base_cache: dict[date, tuple[np.ndarray, np.ndarray]],
    label_cache: dict[date, dict[str, float]],
    preprocess_backend: str,
    preprocess_gpu_fallback_to_cpu: bool,
    progress_label: str | None = None,
    log_progress: bool = False,
    progress_every: int = 5,
) -> SplitData:
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    code_list: list[np.ndarray] = []
    total_samples = 0
    total_days = len(days)
    every = max(1, int(progress_every))

    if log_progress:
        print(f"[prep:{progress_label or 'split'}] start days={total_days} backend={preprocess_backend}")

    for day_idx, day in enumerate(days, start=1):
        if log_progress:
            print(f"[prep:{progress_label or 'split'}] day {day_idx}/{total_days} start day={day}")
        x_day, code_day = _build_day_base_samples(
            panel_root=panel_root,
            day=day,
            depth_levels=depth_levels,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            base_cache=base_cache,
            preprocess_backend=preprocess_backend,
            preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
        )
        if x_day.size == 0 or code_day.size == 0:
            if log_progress and (day_idx == 1 or day_idx == total_days or (day_idx % every) == 0):
                print(
                    f"[prep:{progress_label or 'split'}] day {day_idx}/{total_days} "
                    f"done day={day} samples=0 cumulative={total_samples}"
                )
            continue

        y_map = _label_map_for_day(
            label_root=label_root,
            day=day,
            factor_time=factor_time,
            label_time=label_time,
            cache=label_cache,
        )
        if require_label:
            keep_idx = [i for i, code in enumerate(code_day.tolist()) if code in y_map]
            if not keep_idx:
                continue
            keep = np.asarray(keep_idx, dtype=np.int64)
            x_keep = x_day[keep]
            code_keep = code_day[keep]
            y_keep = np.asarray([float(y_map[c]) for c in code_keep.tolist()], dtype=np.float32)
        else:
            x_keep = x_day
            code_keep = code_day
            y_keep = np.asarray(
                [float(y_map[c]) if c in y_map else np.nan for c in code_keep.tolist()],
                dtype=np.float32,
            )

        x_list.append(x_keep)
        y_list.append(y_keep)
        dt_list.append(np.asarray([day] * len(code_keep), dtype=object))
        code_list.append(code_keep.astype(object))
        total_samples += int(len(code_keep))

        if log_progress and (day_idx == 1 or day_idx == total_days or (day_idx % every) == 0):
            print(
                f"[prep:{progress_label or 'split'}] day {day_idx}/{total_days} "
                f"done day={day} samples={len(code_keep)} cumulative={total_samples}"
            )

    if not x_list:
        return SplitData(
            x=np.empty((0, 2, max(seq_len, 0), depth_levels), dtype=np.float32),
            y=np.array([], dtype=np.float32),
            dt=np.array([], dtype=object),
            code=np.array([], dtype=object),
        )

    return SplitData(
        x=np.concatenate(x_list, axis=0),
        y=np.concatenate(y_list, axis=0),
        dt=np.concatenate(dt_list, axis=0),
        code=np.concatenate(code_list, axis=0),
    )


def _build_eval_daily(
    *,
    scores_df: pd.DataFrame,
    label_root: Path,
    factor_time: str,
    label_time: str,
    label_cache: dict[date, dict[str, float]],
) -> pd.DataFrame:
    if scores_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
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
        y_true = g["y"].to_numpy(dtype=float)
        y_pred = g["score"].to_numpy(dtype=float)
        metrics = _eval_metrics(
            np.asarray([day] * len(g), dtype=object),
            y_true,
            y_pred,
        )
        rows.append(
            {
                "trade_date": day,
                "count": metrics["count"],
                "mse": metrics["mse"],
                "r2": metrics["r2"],
                "dir": metrics["dir"],
                "ic": metrics["ic"],
                "rank_ic": metrics["rank_ic"],
            }
        )
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

    panel_name = str(cfg.get("panel_name", "")).strip()
    if not panel_name:
        panel_name = make_window_label(int(cfg.get("window_minutes", 15)))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:42"))
    depth_levels = int(cfg.get("params", {}).get("depth_levels", 10))
    seq_len = int(cfg.get("params", {}).get("input_length", 0))
    if seq_len <= 0:
        seq_len = int(cfg.get("sample_count", 0))
    if seq_len <= 0:
        raise ValueError("lob config requires params.input_length (or sample_count) > 0")
    min_seq_len = int(cfg.get("min_seq_len", seq_len if seq_len > 0 else 1))

    panel_root = Path(paths_cfg["panel_data_root"]) / "panels" / "cbond" / panel_name
    label_root = Path(paths_cfg["label_data_root"])
    if not panel_root.exists():
        raise FileNotFoundError(f"panel root not found: {panel_root}")

    panel_days_all = _iter_existing_panel_days(panel_root)
    if not panel_days_all:
        raise RuntimeError(f"no panel days found under: {panel_root}")
    panel_day_set = set(panel_days_all)

    label_days_all = sorted(
        d for d in _iter_existing_label_days(label_root, panel_days_all[0], desired_end) if d in panel_day_set
    )
    if cutoff_day is not None:
        label_days_all = [d for d in label_days_all if d <= cutoff_day]
    if not label_days_all:
        raise RuntimeError("no label days found for desired range")

    days = sorted([d for d in panel_days_all if d <= desired_end and (d in label_days_all or d > label_days_all[-1])])
    desired_days = [d for d in days if desired_start <= d <= desired_end]
    if not desired_days:
        raise RuntimeError("no panel days found for desired range")

    train_cfg = dict(cfg.get("train", {}))
    preprocess_backend, preprocess_reason, preprocess_gpu_fallback_to_cpu = _resolve_preprocess_backend(
        train_cfg
    )
    log_data_prep_progress = bool(train_cfg.get("log_data_prep_progress", True))
    data_prep_progress_every = max(1, int(train_cfg.get("data_prep_progress_every", 5)))
    requested_preprocess = str(train_cfg.get("preprocess_backend", "auto")).strip().lower()
    print(
        "lob preprocess backend:",
        f"requested={requested_preprocess}",
        f"active={preprocess_backend}",
        f"reason={preprocess_reason}",
        f"fallback_to_cpu={preprocess_gpu_fallback_to_cpu}",
    )
    rolling_cfg = dict(cfg.get("rolling", {}))
    rolling_enabled = bool(rolling_cfg.get("enabled", True))
    window_days = int(rolling_cfg.get("window_days", 120))
    if window_days < 2:
        raise ValueError("rolling.window_days must be >= 2")
    execution_cfg = dict(execution or {})
    refit_every_n_days = max(1, int(execution_cfg.get("refit_every_n_days", 1)))
    cfg_out = dict(cfg)
    cfg_out["execution"] = {"refit_every_n_days": refit_every_n_days}
    cfg_out["preprocess"] = {
        "requested": requested_preprocess,
        "active": preprocess_backend,
        "fallback_to_cpu": preprocess_gpu_fallback_to_cpu,
        "reason": preprocess_reason,
    }
    cfg_out["prep_progress"] = {
        "enabled": log_data_prep_progress,
        "every_days": data_prep_progress_every,
    }
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))
    if not rolling_enabled and abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    results_root = Path(str(cfg.get("results_root") or paths_cfg["results_root"]))
    model_name = str(cfg.get("model_name", "lob_st_default"))
    date_label = f"{desired_start:%Y-%m-%d}_{desired_end:%Y-%m-%d}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    score_output = Path(cfg.get("score_output", results_root / "scores" / model_name))
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
        if skipped > 0:
            print(
                f"[rolling] incremental skip_existing_scores=True, "
                f"skip={skipped}, pending={len(target_days)}"
            )
    if not target_days:
        print("[rolling] incremental: no pending target days, skip training")
        (out_dir / "config.json").write_text(
            json.dumps(cfg_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved rolling: {out_dir}")
        print(f"saved scores: {score_output}")
        return

    base_cache: dict[date, tuple[np.ndarray, np.ndarray]] = {}
    label_cache: dict[date, dict[str, float]] = {}
    score_rows: list[pd.DataFrame] = []
    rolling_rows: list[dict] = []
    history_rows: list[dict] = []
    last_model: LOBSpatioTemporalModel | None = None

    if rolling_enabled:
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

        total_rolls = len(valid_indices)
        last_refit_pos: int | None = None
        last_refit_day: date | None = None
        for roll_pos, idx in enumerate(valid_indices):
            roll_idx = roll_pos + 1
            window = days[idx - window_days + 1 : idx + 1]
            test_day = window[-1]
            should_refit = (
                last_model is None
                or refit_every_n_days <= 1
                or last_refit_pos is None
                or (roll_pos - last_refit_pos) >= refit_every_n_days
            )
            print(
                f"[rolling] {roll_idx}/{total_rolls} test_day={test_day} "
                f"refit={'Y' if should_refit else 'N'} cadence={refit_every_n_days}"
            )

            test_data = _build_split_data(
                days=[test_day],
                panel_root=panel_root,
                label_root=label_root,
                factor_time=factor_time,
                label_time=label_time,
                depth_levels=depth_levels,
                seq_len=seq_len,
                min_seq_len=min_seq_len,
                require_label=False,
                base_cache=base_cache,
                label_cache=label_cache,
                preprocess_backend=preprocess_backend,
                preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
                progress_label=f"test:{test_day}",
                log_progress=log_data_prep_progress,
                progress_every=data_prep_progress_every,
            )
            if test_data.x.size == 0:
                print(f"[rolling] skip test_day={test_day}: empty test samples")
                continue

            roll_train_days: list[date] = []
            roll_val_days: list[date] = []
            refit_status = "reuse"
            if should_refit:
                train_pool = [d for d in window[:-1] if d in label_days_all]
                if len(train_pool) < 3:
                    if last_model is None:
                        print(
                            f"[rolling] skip test_day={test_day}: "
                            "insufficient train_pool and no reusable model"
                        )
                        continue
                    refit_status = "reuse_insufficient_train_pool"
                    print(
                        f"[rolling] fallback reuse for test_day={test_day}: "
                        f"insufficient train_pool={len(train_pool)}"
                    )
                else:
                    roll_train_days, roll_val_days, _ = _split_days(train_pool, train_ratio, val_ratio)
                    train_data = _build_split_data(
                        days=roll_train_days,
                        panel_root=panel_root,
                        label_root=label_root,
                        factor_time=factor_time,
                        label_time=label_time,
                        depth_levels=depth_levels,
                        seq_len=seq_len,
                        min_seq_len=min_seq_len,
                        require_label=True,
                        base_cache=base_cache,
                        label_cache=label_cache,
                        preprocess_backend=preprocess_backend,
                        preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
                        progress_label=f"train:{test_day}",
                        log_progress=log_data_prep_progress,
                        progress_every=data_prep_progress_every,
                    )
                    val_data = _build_split_data(
                        days=roll_val_days,
                        panel_root=panel_root,
                        label_root=label_root,
                        factor_time=factor_time,
                        label_time=label_time,
                        depth_levels=depth_levels,
                        seq_len=seq_len,
                        min_seq_len=min_seq_len,
                        require_label=True,
                        base_cache=base_cache,
                        label_cache=label_cache,
                        preprocess_backend=preprocess_backend,
                        preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
                        progress_label=f"val:{test_day}",
                        log_progress=log_data_prep_progress,
                        progress_every=data_prep_progress_every,
                    )
                    if train_data.x.size == 0:
                        if last_model is None:
                            print(
                                f"[rolling] skip test_day={test_day}: "
                                "empty train samples and no reusable model"
                            )
                            continue
                        refit_status = "reuse_empty_train"
                        print(
                            f"[rolling] fallback reuse for test_day={test_day}: "
                            "empty train samples"
                        )
                    else:
                        model, hist = _train_one_model(
                            train_data=train_data,
                            val_data=val_data,
                            params=dict(cfg.get("params", {})),
                            train_cfg=train_cfg,
                        )
                        last_model = model
                        last_refit_pos = roll_pos
                        last_refit_day = test_day
                        refit_status = "refit"
                        for row in hist:
                            history_rows.append({"trade_date": test_day, **row})

            if last_model is None:
                print(f"[rolling] skip test_day={test_day}: no reusable model")
                continue

            pred = _predict_numpy(
                model=last_model,
                x=test_data.x,
                batch_size=max(1, int(train_cfg.get("batch_size", 8))),
                device=_resolve_device(train_cfg),
                normalize_x=bool(train_cfg.get("normalize_x", False)),
                x_norm_method=str(train_cfg.get("x_norm_method", "zscore_sample")),
            )
            score_rows.append(
                pd.DataFrame(
                    {
                        "trade_date": test_day,
                        "code": test_data.code,
                        "score": pred,
                    }
                )
            )

            labeled_mask = np.isfinite(test_data.y)
            if labeled_mask.any():
                metrics = _eval_metrics(
                    test_data.dt[labeled_mask],
                    test_data.y[labeled_mask].astype(float),
                    pred[labeled_mask].astype(float),
                )
            else:
                metrics = {
                    "count": int(len(pred)),
                    "mse": float("nan"),
                    "r2": float("nan"),
                    "dir": float("nan"),
                    "ic": float("nan"),
                    "rank_ic": float("nan"),
                }
            model_source_day = test_day if refit_status == "refit" else (last_refit_day or test_day)
            rolling_rows.append(
                {
                    "trade_date": test_day,
                    "refit": bool(refit_status == "refit"),
                    "refit_status": refit_status,
                    "model_source_day": model_source_day,
                    "refit_every_n_days": refit_every_n_days,
                    "train_days": len(roll_train_days),
                    "val_days": len(roll_val_days),
                    "count": int(len(pred)),
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "dir": metrics["dir"],
                    "ic": metrics["ic"],
                    "rank_ic": metrics["rank_ic"],
                }
            )
            print(
                f"rolling {test_day} refit={refit_status} train_days={len(roll_train_days)} "
                f"val_days={len(roll_val_days)} count={len(pred)} rank_ic={metrics['rank_ic']:.4f}"
            )
    else:
        train_days = [d for d in label_days_all if desired_start <= d <= desired_end]
        if len(train_days) < 3:
            raise RuntimeError("not enough labeled days for non-rolling training")
        tr_days, va_days, _ = _split_days(train_days, train_ratio, val_ratio)
        train_data = _build_split_data(
            days=tr_days,
            panel_root=panel_root,
            label_root=label_root,
            factor_time=factor_time,
            label_time=label_time,
            depth_levels=depth_levels,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            require_label=True,
            base_cache=base_cache,
            label_cache=label_cache,
            preprocess_backend=preprocess_backend,
            preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
            progress_label="train_full",
            log_progress=log_data_prep_progress,
            progress_every=data_prep_progress_every,
        )
        val_data = _build_split_data(
            days=va_days,
            panel_root=panel_root,
            label_root=label_root,
            factor_time=factor_time,
            label_time=label_time,
            depth_levels=depth_levels,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            require_label=True,
            base_cache=base_cache,
            label_cache=label_cache,
            preprocess_backend=preprocess_backend,
            preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
            progress_label="val_full",
            log_progress=log_data_prep_progress,
            progress_every=data_prep_progress_every,
        )
        model, hist = _train_one_model(
            train_data=train_data,
            val_data=val_data,
            params=dict(cfg.get("params", {})),
            train_cfg=train_cfg,
        )
        last_model = model
        for row in hist:
            history_rows.append({"trade_date": "", **row})
        for day in target_days:
            test_data = _build_split_data(
                days=[day],
                panel_root=panel_root,
                label_root=label_root,
                factor_time=factor_time,
                label_time=label_time,
                depth_levels=depth_levels,
                seq_len=seq_len,
                min_seq_len=min_seq_len,
                require_label=False,
                base_cache=base_cache,
                label_cache=label_cache,
                preprocess_backend=preprocess_backend,
                preprocess_gpu_fallback_to_cpu=preprocess_gpu_fallback_to_cpu,
                progress_label=f"score:{day}",
                log_progress=log_data_prep_progress,
                progress_every=data_prep_progress_every,
            )
            if test_data.x.size == 0:
                continue
            pred = _predict_numpy(
                model=model,
                x=test_data.x,
                batch_size=max(1, int(train_cfg.get("batch_size", 8))),
                device=_resolve_device(train_cfg),
                normalize_x=bool(train_cfg.get("normalize_x", False)),
                x_norm_method=str(train_cfg.get("x_norm_method", "zscore_sample")),
            )
            score_rows.append(
                pd.DataFrame(
                    {
                        "trade_date": day,
                        "code": test_data.code,
                        "score": pred,
                    }
                )
            )

    if not score_rows:
        raise RuntimeError("lob produced no scores; check panel/label range and sequence length")

    all_scores = pd.concat(score_rows, ignore_index=True)
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

    eval_daily = _build_eval_daily(
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
            "count_mean": float(pd.to_numeric(eval_daily["count"], errors="coerce").mean()),
            "mse_mean": float(pd.to_numeric(eval_daily["mse"], errors="coerce").mean()),
            "r2_mean": float(pd.to_numeric(eval_daily["r2"], errors="coerce").mean()),
            "dir_mean": float(pd.to_numeric(eval_daily["dir"], errors="coerce").mean()),
            "ic_mean": float(pd.to_numeric(eval_daily["ic"], errors="coerce").mean()),
            "rank_ic_mean": float(pd.to_numeric(eval_daily["rank_ic"], errors="coerce").mean()),
        }
    else:
        summary = {
            "days": 0,
            "count_mean": float("nan"),
            "mse_mean": float("nan"),
            "r2_mean": float("nan"),
            "dir_mean": float("nan"),
            "ic_mean": float("nan"),
            "rank_ic_mean": float("nan"),
        }
    (out_dir / "score_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_score_report(out_dir=out_dir, scores_df=all_scores, eval_daily=eval_daily)

    if last_model is not None:
        model_path = out_dir / "model.pt"
        torch.save(last_model.state_dict(), model_path)
        weights_path_text = str(cfg.get("weights_path", "")).strip()
        if weights_path_text:
            weights_path = Path(weights_path_text)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(last_model.state_dict(), weights_path)

    (out_dir / "config.json").write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "features.json").write_text(
        json.dumps(
            {
                "panel_name": panel_name,
                "depth_levels": depth_levels,
                "sequence_length": seq_len,
                "min_seq_len": min_seq_len,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved rolling: {out_dir}")
    print(f"saved scores: {score_output}")


if __name__ == "__main__":
    main()
