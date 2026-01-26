from __future__ import annotations

import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
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
        return x, y


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
) -> tuple[dict[str, float], dict[str, float] | None]:
    model.eval()
    loss_fn = torch.nn.MSELoss()
    tracker = MetricTracker()
    y_list: list[np.ndarray] = []
    p_list: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
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
    stats = None
    if collect_outputs:
        y_all = np.concatenate(y_list, axis=0) if y_list else np.array([])
        p_all = np.concatenate(p_list, axis=0) if p_list else np.array([])
        stats = _epoch_stats(y_all, p_all)
    return tracker.metrics(), stats


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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
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
    print(f"[debug] config input_length: {input_length}")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
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
        printed_shape = False
        for x, y in loop:
            x = x.to(device)
            if input_length:
                x = x[:, :, -input_length:, :] if x.shape[2] > input_length else x
            if normalize_x:
                x = _normalize_x_batch(x, x_norm_method)
            if not printed_shape:
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

        train_metrics = tracker.metrics()
        val_metrics, val_stats = (
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
            else ({"loss": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}, None)
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
        print(
            f"[train stats] y_mean={train_stats['y_mean']:.6f} y_std={train_stats['y_std']:.6f} "
            f"y_min={train_stats['y_min']:.6f} y_max={train_stats['y_max']:.6f} "
            f"baseline_mse={train_stats['baseline_mse']:.6f} model_mse={train_stats['model_mse']:.6f} "
            f"p_pos={train_stats['p_pos']:.4f} pred_pos={train_stats['pred_pos']:.4f} "
            f"dir={train_stats['dir']:.4f} corr={train_stats['corr']:.4f}"
        )
        if val_stats is not None:
            print(
                f"[val stats] y_mean={val_stats['y_mean']:.6f} y_std={val_stats['y_std']:.6f} "
                f"y_min={val_stats['y_min']:.6f} y_max={val_stats['y_max']:.6f} "
                f"baseline_mse={val_stats['baseline_mse']:.6f} model_mse={val_stats['model_mse']:.6f} "
                f"p_pos={val_stats['p_pos']:.4f} pred_pos={val_stats['pred_pos']:.4f} "
                f"dir={val_stats['dir']:.4f} corr={val_stats['corr']:.4f}"
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
            }
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(model.state_dict(), weights_path)

    if weights_path.exists():
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
    test_metrics, test_stats = (
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
        else ({"loss": float("nan"), "r2": float("nan"), "dir_acc": float("nan")}, None)
    )
    print(f"test_loss={test_metrics['loss']:.6f} test_r2={test_metrics['r2']:.4f} test_dir={test_metrics['dir_acc']:.4f}")
    if test_stats is not None:
        print(
            f"[test stats] y_mean={test_stats['y_mean']:.6f} y_std={test_stats['y_std']:.6f} "
            f"y_min={test_stats['y_min']:.6f} y_max={test_stats['y_max']:.6f} "
            f"baseline_mse={test_stats['baseline_mse']:.6f} model_mse={test_stats['model_mse']:.6f} "
            f"p_pos={test_stats['p_pos']:.4f} pred_pos={test_stats['pred_pos']:.4f} "
            f"dir={test_stats['dir']:.4f} corr={test_stats['corr']:.4f}"
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
