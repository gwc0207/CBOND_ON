from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.models.impl.lob_st import LOBSpatioTemporalModel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class LOBDataset(Dataset):
    def __init__(self, day_dirs: list[Path]) -> None:
        self._days: list[dict] = []
        self._cum_sizes: list[int] = []
        self._total = 0
        for day_dir in day_dirs:
            x_path = day_dir / "X.npy"
            y_path = day_dir / "y.npy"
            if not x_path.exists() or not y_path.exists():
                continue
            x_shape = np.load(x_path, mmap_mode="r").shape
            if len(x_shape) != 4:
                continue
            size = int(x_shape[0])
            if size <= 0:
                continue
            self._days.append(
                {"dir": day_dir, "x_path": x_path, "y_path": y_path, "size": size}
            )
            self._total += size
            self._cum_sizes.append(self._total)

        self._cached_idx = -1
        self._cached_x = None
        self._cached_y = None

    def __len__(self) -> int:
        return self._total

    def _load_day(self, day_idx: int) -> None:
        if day_idx == self._cached_idx:
            return
        day = self._days[day_idx]
        self._cached_x = np.load(day["x_path"], mmap_mode="r")
        self._cached_y = np.load(day["y_path"], mmap_mode="r")
        self._cached_idx = day_idx

    def __getitem__(self, idx: int):
        day_idx = bisect_left(self._cum_sizes, idx + 1)
        if day_idx >= len(self._days):
            raise IndexError("index out of range")
        offset = idx if day_idx == 0 else idx - self._cum_sizes[day_idx - 1]
        self._load_day(day_idx)
        x = torch.from_numpy(self._cached_x[offset].copy()).float()
        y = torch.tensor(self._cached_y[offset], dtype=torch.float32)
        day = self._days[day_idx]["dir"].name
        return x, y, day


def _split_days(days: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
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


class MetricTracker:
    def __init__(self) -> None:
        self.count = 0
        self.loss_sum = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sse = 0.0
        self.dir_hits = 0

    def update(self, loss: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        batch = y_true.numel()
        self.count += batch
        self.loss_sum += float(loss.item()) * batch
        y_true_f = y_true.detach()
        y_pred_f = y_pred.detach()
        self.sum_y += float(y_true_f.sum().item())
        self.sum_y2 += float((y_true_f ** 2).sum().item())
        self.sse += float(((y_true_f - y_pred_f) ** 2).sum().item())
        self.dir_hits += int((torch.sign(y_true_f) == torch.sign(y_pred_f)).sum().item())

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {"loss": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}
        loss = self.loss_sum / self.count
        mean_y = self.sum_y / self.count
        sst = self.sum_y2 - self.count * mean_y * mean_y
        r2 = 1.0 - (self.sse / sst) if sst > 0 else float("nan")
        dir_acc = self.dir_hits / self.count
        return {"loss": loss, "r2": r2, "dir_acc": dir_acc}


def _rank_ic(y_all: np.ndarray, p_all: np.ndarray) -> float:
    if y_all.size < 2:
        return float("nan")
    y_rank = pd.Series(y_all).rank(method="average")
    p_rank = pd.Series(p_all).rank(method="average")
    return float(y_rank.corr(p_rank, method="pearson"))


def _daily_rank_ic(y_all: np.ndarray, p_all: np.ndarray, d_all: np.ndarray) -> float:
    if y_all.size < 2:
        return float("nan")
    df = pd.DataFrame({"y": y_all, "p": p_all, "day": d_all})
    vals: list[float] = []
    for _, group in df.groupby("day", sort=True):
        if len(group) < 2:
            continue
        vals.append(_rank_ic(group["y"].to_numpy(), group["p"].to_numpy()))
    return float(np.nanmean(vals)) if vals else float("nan")


def _daily_rank_ic_ir(
    y_all: np.ndarray, p_all: np.ndarray, d_all: np.ndarray
) -> tuple[float, float]:
    if y_all.size < 2:
        return float("nan"), float("nan")
    df = pd.DataFrame({"y": y_all, "p": p_all, "day": d_all})
    vals: list[float] = []
    for _, group in df.groupby("day", sort=True):
        if len(group) < 2:
            continue
        vals.append(_rank_ic(group["y"].to_numpy(), group["p"].to_numpy()))
    if not vals:
        return float("nan"), float("nan")
    mean = float(np.nanmean(vals))
    std = float(np.nanstd(vals))
    ir = float(mean / std) if std > 0 else float("nan")
    return mean, ir


def _topn_dir_acc(y_all: np.ndarray, p_all: np.ndarray, top_n: int) -> float:
    if y_all.size == 0 or top_n <= 0:
        return float("nan")
    top_n = min(int(top_n), int(y_all.size))
    idx = np.argpartition(-p_all, top_n - 1)[:top_n]
    return float((np.sign(y_all[idx]) == np.sign(p_all[idx])).mean())


def _daily_topn_dir_acc(
    y_all: np.ndarray, p_all: np.ndarray, d_all: np.ndarray, top_n: int
) -> float:
    if y_all.size == 0 or top_n <= 0:
        return float("nan")
    df = pd.DataFrame({"y": y_all, "p": p_all, "day": d_all})
    vals: list[float] = []
    for _, group in df.groupby("day", sort=True):
        vals.append(_topn_dir_acc(group["y"].to_numpy(), group["p"].to_numpy(), top_n))
    return float(np.nanmean(vals)) if vals else float("nan")


def _bin_dir_acc(y_all: np.ndarray, p_all: np.ndarray, bins: int) -> list[tuple[int, float, int]]:
    if y_all.size == 0 or bins <= 1:
        return []
    try:
        ranks = pd.Series(p_all).rank(method="first")
        bins_cat = pd.qcut(ranks, bins, labels=False, duplicates="drop")
    except ValueError:
        return []
    df = pd.DataFrame({"bin": bins_cat, "y": y_all, "p": p_all}).dropna()
    if df.empty:
        return []
    grouped = df.groupby("bin", dropna=True)
    results: list[tuple[int, float, int]] = []
    for bin_id, group in grouped:
        acc = float((np.sign(group["y"]) == np.sign(group["p"])).mean())
        results.append((int(bin_id), acc, int(len(group))))
    results.sort(key=lambda x: x[0])
    return results


def _daily_bin_dir_acc(
    y_all: np.ndarray, p_all: np.ndarray, d_all: np.ndarray, bins: int
) -> list[tuple[int, float, int]]:
    if y_all.size == 0 or bins <= 1:
        return []
    df = pd.DataFrame({"y": y_all, "p": p_all, "day": d_all})
    acc_sum: dict[int, float] = {}
    count_sum: dict[int, int] = {}
    for _, group in df.groupby("day", sort=True):
        if group.empty:
            continue
        day_bins = _bin_dir_acc(group["y"].to_numpy(), group["p"].to_numpy(), bins)
        for bin_id, acc, cnt in day_bins:
            acc_sum[bin_id] = acc_sum.get(bin_id, 0.0) + acc * cnt
            count_sum[bin_id] = count_sum.get(bin_id, 0) + cnt
    results: list[tuple[int, float, int]] = []
    for bin_id in sorted(count_sum.keys()):
        total = count_sum[bin_id]
        avg = acc_sum.get(bin_id, 0.0) / total if total > 0 else float("nan")
        results.append((bin_id, float(avg), int(total)))
    return results


def _epoch_stats(y_all: np.ndarray, p_all: np.ndarray) -> dict[str, float]:
    if y_all.size == 0:
        return {
            "y_mean": float("nan"),
            "y_std": float("nan"),
            "y_min": float("nan"),
            "y_max": float("nan"),
            "baseline_mse": float("nan"),
            "model_mse": float("nan"),
            "p_pos": float("nan"),
            "pred_pos": float("nan"),
            "dir": float("nan"),
            "corr": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
        }
    y_mean = float(y_all.mean())
    y_std = float(y_all.std())
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    baseline_mse = float(((y_all - y_mean) ** 2).mean())
    model_mse = float(((y_all - p_all) ** 2).mean())
    p_pos = float((y_all > 0).mean())
    pred_pos = float((p_all > 0).mean())
    dir_acc = float((np.sign(y_all) == np.sign(p_all)).mean())
    corr = float(np.corrcoef(p_all, y_all)[0, 1]) if y_all.size > 1 else float("nan")
    pred_mean = float(p_all.mean())
    pred_std = float(p_all.std())
    pred_min = float(p_all.min())
    pred_max = float(p_all.max())
    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "y_min": y_min,
        "y_max": y_max,
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "p_pos": p_pos,
        "pred_pos": pred_pos,
        "dir": dir_acc,
        "corr": corr,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
    }


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


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_outputs: bool = False,
    input_length: int | None = None,
    normalize_x: bool = False,
    normalize_y: bool = False,
    x_norm_method: str = "zscore_sample",
    y_norm_method: str = "zscore_batch",
) -> tuple[
    dict[str, float],
    dict[str, float] | None,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
]:
    model.eval()
    loss_fn = torch.nn.MSELoss()
    tracker = MetricTracker()
    y_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    y_days: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, day in loader:
            x = x.to(device)
            if input_length:
                x = x[:, :, -input_length:, :] if x.shape[2] > input_length else x
            if normalize_x:
                x = _normalize_x_batch(x, x_norm_method)
            y = y.to(device)
            if normalize_y:
                y = _normalize_y_batch(y, y_norm_method)
            pred = model(x)
            loss = loss_fn(pred, y)
            tracker.update(loss, y, pred)
            if collect_outputs:
                y_list.append(y.detach().cpu().numpy())
                p_list.append(pred.detach().cpu().numpy())
                y_days.append(np.array(day))
    stats = None
    outputs = None
    if collect_outputs:
        y_all = np.concatenate(y_list, axis=0) if y_list else np.array([])
        p_all = np.concatenate(p_list, axis=0) if p_list else np.array([])
        d_all = np.concatenate(y_days, axis=0) if y_days else np.array([])
        stats = _epoch_stats(y_all, p_all)
        outputs = (y_all, p_all, d_all)
    return tracker.metrics(), stats, outputs


def main() -> None:
    paths_cfg = load_config_file("paths")
    ds_cfg = load_config_file("dataset")
    model_cfg = load_config_file("models/model")

    clean_root = Path(paths_cfg["clean_data_root"])
    output_dir = clean_root / str(ds_cfg.get("output_dir", "LOBDS"))
    if not output_dir.exists():
        raise FileNotFoundError(f"dataset output not found: {output_dir}")

    day_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir() and (p / "X.npy").exists()])
    if not day_dirs:
        raise RuntimeError("no dataset days found under output_dir")

    train_cfg = model_cfg.get("train", {})
    train_start = train_cfg.get("train_start")
    train_end = train_cfg.get("train_end")
    if train_start or train_end:
        start_date = parse_date(train_start) if train_start else None
        end_date = parse_date(train_end) if train_end else None
        filtered = []
        for p in day_dirs:
            try:
                day = parse_date(p.name)
            except Exception:
                continue
            if start_date and day < start_date:
                continue
            if end_date and day > end_date:
                continue
            filtered.append(p)
        day_dirs = filtered
        if not day_dirs:
            raise RuntimeError("no dataset days left after train_start/train_end filter")
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_days, val_days, test_days = _split_days(day_dirs, train_ratio, val_ratio)
    print(f"train days: {len(train_days)}, val days: {len(val_days)}, test days: {len(test_days)}")

    train_ds = LOBDataset(train_days)
    val_ds = LOBDataset(val_days)
    test_ds = LOBDataset(test_days)
    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}, test samples: {len(test_ds)}")

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    device_name = str(train_cfg.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    shuffle_train = bool(train_cfg.get("shuffle_train", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    params = model_cfg.get("params", {})
    input_length = params.get("input_length")
    input_length = int(input_length) if input_length else None
    model = LOBSpatioTemporalModel(
        depth_levels=int(params.get("depth_levels", 10)),
        rbf_num_bases=int(params.get("rbf_num_bases", 16)),
        rbf_sigma=float(params.get("rbf_sigma", 0.5)),
        lstm_hidden_size=int(params.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(params.get("lstm_num_layers", 1)),
    ).to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    num_epochs = int(train_cfg.get("num_epochs", 10))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))
    normalize_x = bool(train_cfg.get("normalize_x", False))
    normalize_y = bool(train_cfg.get("normalize_y", False))
    x_norm_method = str(train_cfg.get("x_norm_method", "zscore_sample"))
    y_norm_method = str(train_cfg.get("y_norm_method", "zscore_batch"))
    metrics_top_n = int(train_cfg.get("metrics_top_n", 50))
    metrics_bins = int(train_cfg.get("metrics_bins", 5))
    metrics_debug = bool(train_cfg.get("metrics_debug", False))
    if metrics_debug:
        print(f"[debug] config input_length: {input_length}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_score = float("-inf")
    best_epoch = None
    dir_weight = float(train_cfg.get("checkpoint_dir_weight", 1.0))
    corr_weight = float(train_cfg.get("checkpoint_corr_weight", 0.3))
    weights_path = Path(model_cfg.get("weights_path", "results/models/lob_st_default/model.pt"))
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        tracker = MetricTracker()
        loop = train_loader
        if tqdm is not None:
            loop = tqdm(train_loader, desc=f"epoch {epoch:03d}", unit="batch")
        train_y_list: list[np.ndarray] = []
        train_p_list: list[np.ndarray] = []
        train_day_list: list[np.ndarray] = []
        printed_shape = False
        for x, y, day in loop:
            x = x.to(device)
            if input_length:
                x = x[:, :, -input_length:, :] if x.shape[2] > input_length else x
            if normalize_x:
                x = _normalize_x_batch(x, x_norm_method)
            if metrics_debug and not printed_shape:
                print(f"[debug] train batch x shape: {tuple(x.shape)}")
                printed_shape = True
            y = y.to(device)
            if normalize_y:
                y = _normalize_y_batch(y, y_norm_method)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            tracker.update(loss, y, pred)
            train_y_list.append(y.detach().cpu().numpy())
            train_p_list.append(pred.detach().cpu().numpy())
            train_day_list.append(np.array(day))

        train_metrics = tracker.metrics()
        val_metrics, val_stats, val_outputs = (
            _evaluate(
                model,
                val_loader,
                device,
                collect_outputs=True,
                input_length=input_length,
                normalize_x=normalize_x,
                normalize_y=normalize_y,
                x_norm_method=x_norm_method,
                y_norm_method=y_norm_method,
            )
            if len(val_ds)
            else ({"loss": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}, None, None)
        )
        train_stats = _epoch_stats(
            np.concatenate(train_y_list, axis=0) if train_y_list else np.array([]),
            np.concatenate(train_p_list, axis=0) if train_p_list else np.array([]),
        )
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} train_r2={train_metrics['r2']:.4f} train_dir={train_metrics['dir_acc']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} val_r2={val_metrics['r2']:.4f} val_dir={val_metrics['dir_acc']:.4f}"
        )
        if epoch == 1:
            print(
                f"[train y stats] y_mean={train_stats['y_mean']:.6f} y_std={train_stats['y_std']:.6f} "
                f"y_min={train_stats['y_min']:.6f} y_max={train_stats['y_max']:.6f} "
                f"baseline_mse={train_stats['baseline_mse']:.6f}"
            )
            if val_stats is not None:
                print(
                    f"[val y stats] y_mean={val_stats['y_mean']:.6f} y_std={val_stats['y_std']:.6f} "
                    f"y_min={val_stats['y_min']:.6f} y_max={val_stats['y_max']:.6f} "
                    f"baseline_mse={val_stats['baseline_mse']:.6f}"
                )
        if metrics_debug:
            print(
                f"[train pred stats] pred_mean={train_stats['pred_mean']:.6f} pred_std={train_stats['pred_std']:.6f} "
                f"pred_min={train_stats['pred_min']:.6f} pred_max={train_stats['pred_max']:.6f}"
            )
            if val_stats is not None:
                print(
                    f"[val pred stats] pred_mean={val_stats['pred_mean']:.6f} pred_std={val_stats['pred_std']:.6f} "
                    f"pred_min={val_stats['pred_min']:.6f} pred_max={val_stats['pred_max']:.6f}"
                )

        train_y_all = np.concatenate(train_y_list, axis=0) if train_y_list else np.array([])
        train_p_all = np.concatenate(train_p_list, axis=0) if train_p_list else np.array([])
        train_d_all = np.concatenate(train_day_list, axis=0) if train_day_list else np.array([])
        train_rank_ic, train_rank_ic_ir = _daily_rank_ic_ir(train_y_all, train_p_all, train_d_all)
        train_top_dir = _daily_topn_dir_acc(train_y_all, train_p_all, train_d_all, metrics_top_n)
        train_bins = _daily_bin_dir_acc(train_y_all, train_p_all, train_d_all, metrics_bins)
        if val_outputs is not None:
            val_rank_ic, val_rank_ic_ir = _daily_rank_ic_ir(val_outputs[0], val_outputs[1], val_outputs[2])
            val_top_dir = _daily_topn_dir_acc(val_outputs[0], val_outputs[1], val_outputs[2], metrics_top_n)
            val_bins = _daily_bin_dir_acc(val_outputs[0], val_outputs[1], val_outputs[2], metrics_bins)
        else:
            val_rank_ic = float("nan")
            val_rank_ic_ir = float("nan")
            val_top_dir = float("nan")
            val_bins = []
        train_bins_str = ",".join([f"{b}:{a:.3f}({n})" for b, a, n in train_bins]) if train_bins else "n/a"
        val_bins_str = ",".join([f"{b}:{a:.3f}({n})" for b, a, n in val_bins]) if val_bins else "n/a"
        print(
            f"[train extra] rank_ic={train_rank_ic:.4f} rank_ic_ir={train_rank_ic_ir:.4f} "
            f"top_dir@{metrics_top_n}={train_top_dir:.4f} bins={train_bins_str}"
        )
        print(
            f"[val extra] rank_ic={val_rank_ic:.4f} rank_ic_ir={val_rank_ic_ir:.4f} "
            f"top_dir@{metrics_top_n}={val_top_dir:.4f} bins={val_bins_str}"
        )

        val_corr = val_stats["corr"] if val_stats is not None else float("nan")
        if not np.isfinite(val_corr):
            val_corr = 0.0
        checkpoint_score = (
            dir_weight * val_metrics["dir_acc"] + corr_weight * val_corr
            if not np.isnan(val_metrics["dir_acc"]) and not np.isnan(val_corr)
            else float("nan")
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_r2": train_metrics["r2"],
                "train_dir_acc": train_metrics["dir_acc"],
                "val_loss": val_metrics["loss"],
                "val_r2": val_metrics["r2"],
                "val_dir_acc": val_metrics["dir_acc"],
                "val_corr": val_corr,
                "train_rank_ic": train_rank_ic,
                "train_rank_ic_ir": train_rank_ic_ir,
                "val_rank_ic": val_rank_ic,
                "val_rank_ic_ir": val_rank_ic_ir,
                "checkpoint_score": checkpoint_score,
            }
        )

        if not np.isnan(checkpoint_score):
            if checkpoint_score > best_score:
                best_score = checkpoint_score
                best_epoch = epoch
                torch.save(model.state_dict(), weights_path)

    if weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    if best_epoch is not None:
        print(
            f"[checkpoint] epoch={best_epoch} score={best_score:.6f} "
            f"(dir_weight={dir_weight:.3f} corr_weight={corr_weight:.3f})"
        )
    test_metrics, test_stats, test_outputs = (
        _evaluate(
            model,
            test_loader,
            device,
            collect_outputs=True,
            input_length=input_length,
            normalize_x=normalize_x,
            normalize_y=normalize_y,
            x_norm_method=x_norm_method,
            y_norm_method=y_norm_method,
        )
        if len(test_ds)
        else ({"loss": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}, None, None)
    )
    print(f"test_loss={test_metrics['loss']:.6f} test_r2={test_metrics['r2']:.4f} test_dir={test_metrics['dir_acc']:.4f}")
    if test_stats is not None:
        print(
            f"[test stats] y_mean={test_stats['y_mean']:.6f} y_std={test_stats['y_std']:.6f} "
            f"y_min={test_stats['y_min']:.6f} y_max={test_stats['y_max']:.6f} "
            f"baseline_mse={test_stats['baseline_mse']:.6f} model_mse={test_stats['model_mse']:.6f} "
            f"p_pos={test_stats['p_pos']:.4f} pred_pos={test_stats['pred_pos']:.4f} "
            f"dir={test_stats['dir']:.4f} corr={test_stats['corr']:.4f} "
            f"pred_mean={test_stats['pred_mean']:.6f} pred_std={test_stats['pred_std']:.6f} "
            f"pred_min={test_stats['pred_min']:.6f} pred_max={test_stats['pred_max']:.6f}"
        )

    metrics_path = weights_path.parent / "train_metrics.csv"
    if history:
        import pandas as pd

        pd.DataFrame(history).to_csv(metrics_path, index=False)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train")
            ax.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], label="val")
            ax.set_title("Loss curve")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(weights_path.parent / "loss_curve.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    main()
