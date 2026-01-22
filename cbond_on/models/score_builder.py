from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cbond_on.models.impl.lob_st import LOBSpatioTemporalModel


@dataclass
class ScoreConfig:
    device: str = "cpu"
    batch_size: int = 16


def _load_day_inputs(day_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    x_path = day_dir / "X.npy"
    meta_path = day_dir / "meta.parquet"
    if not x_path.exists() or not meta_path.exists():
        return np.empty((0, 2, 0, 0), dtype=np.float32), pd.DataFrame()
    x = np.load(x_path, mmap_mode="r")
    meta = pd.read_parquet(meta_path)
    return x, meta


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_scores(
    *,
    dataset_root: Path,
    weights_path: Path,
    start: date,
    end: date,
    model_params: dict,
    score_cfg: ScoreConfig,
    output_path: Path,
) -> pd.DataFrame:
    device = _device_from_config(score_cfg.device)
    model = LOBSpatioTemporalModel(
        depth_levels=int(model_params.get("depth_levels", 10)),
        rbf_num_bases=int(model_params.get("rbf_num_bases", 16)),
        rbf_sigma=float(model_params.get("rbf_sigma", 0.5)),
        lstm_hidden_size=int(model_params.get("lstm_hidden_size", 256)),
        lstm_num_layers=int(model_params.get("lstm_num_layers", 1)),
    ).to(device)
    if not weights_path.exists():
        raise FileNotFoundError(f"model weights not found: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    records: list[dict] = []
    day_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    for day_dir in day_dirs:
        try:
            day = pd.to_datetime(day_dir.name, format="%Y%m%d").date()
        except ValueError:
            continue
        if day < start or day > end:
            continue
        x, meta = _load_day_inputs(day_dir)
        if x.size == 0 or meta.empty:
            continue
        if "code" not in meta.columns:
            continue
        codes = meta["code"].astype(str).tolist()
        if len(codes) != len(x):
            min_len = min(len(codes), len(x))
            codes = codes[:min_len]
            x = x[:min_len]
        batch_size = max(1, int(score_cfg.batch_size))
        for i in range(0, len(codes), batch_size):
            batch = x[i : i + batch_size]
            batch_t = torch.from_numpy(np.array(batch, dtype=np.float32)).to(device)
            with torch.no_grad():
                preds = model(batch_t).detach().cpu().numpy().astype(float)
            for code, score in zip(codes[i : i + batch_size], preds, strict=False):
                records.append({"trade_date": day, "code": code, "score": float(score)})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("no scores generated; check dataset and date range")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
