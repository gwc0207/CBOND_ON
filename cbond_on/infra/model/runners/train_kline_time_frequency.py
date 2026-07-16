from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import copy
import hashlib
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from cbond_on.common.config_utils import load_json_like
from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.infra.model.impl.torch_image import KlineImageCNN, KlineTimeFrequencyFusionCNN
from cbond_on.infra.model.runners.train_kline_image import (
    ImageDatasetData,
    _IndexedKlineDataset,
    _daily_rank_ic,
    _daily_zscore_targets,
    _device_from_config,
    _evaluate_topk,
    _files_for_range,
    _indices_between,
    _load_image_data,
    _platform_path,
    _portfolio_metrics,
    _write_plot,
    render_candlestick_batch,
)
from cbond_on.infra.model.score_io import write_scores_by_date
from cbond_on.infra.universe.pool_filter import load_upstream_pool_config


_STFT_WINDOW_CACHE: dict[tuple, torch.Tensor] = {}
_CWT_FILTER_CACHE: dict[tuple, torch.Tensor] = {}


def _close_return_signal(
    ohlc: torch.Tensor,
    *,
    return_scale: float,
    clip: float,
) -> torch.Tensor:
    if ohlc.ndim != 3 or ohlc.shape[-1] != 4:
        raise ValueError(f"ohlc must have shape (batch, time, 4), got {tuple(ohlc.shape)}")
    scale = abs(float(return_scale))
    if scale <= 0:
        raise ValueError("return_scale must be positive")
    close = ohlc[:, :, 3]
    previous = torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1)
    signal = (close - previous) / scale
    limit = abs(float(clip))
    return torch.clamp(signal, min=-limit, max=limit) if limit > 0 else signal


def render_stft_batch(
    ohlc: torch.Tensor,
    *,
    n_fft: int = 32,
    win_length: int = 32,
    hop_length: int = 4,
    return_scale: float = 0.002,
    clip: float = 8.0,
) -> torch.Tensor:
    signal = _close_return_signal(ohlc, return_scale=return_scale, clip=clip)
    n_fft = max(4, int(n_fft))
    win_length = min(n_fft, max(4, int(win_length)))
    hop_length = max(1, int(hop_length))
    key = (win_length, str(signal.device), str(signal.dtype))
    window = _STFT_WINDOW_CACHE.get(key)
    if window is None:
        window = torch.hann_window(win_length, device=signal.device, dtype=signal.dtype)
        _STFT_WINDOW_CACHE[key] = window
    spectrum = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=True,
        return_complex=True,
    )
    return torch.stack(
        [spectrum.real, spectrum.imag, torch.log1p(spectrum.abs())],
        dim=1,
    )


def _morlet_filters(
    *,
    length: int,
    num_scales: int,
    min_period: float,
    max_period: float,
    omega0: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (
        int(length),
        int(num_scales),
        float(min_period),
        float(max_period),
        float(omega0),
        str(device),
        str(dtype),
    )
    cached = _CWT_FILTER_CACHE.get(key)
    if cached is not None:
        return cached
    periods = torch.logspace(
        math.log10(float(min_period)),
        math.log10(float(max_period)),
        steps=max(2, int(num_scales)),
        device=device,
        dtype=dtype,
    )
    scales = periods * (float(omega0) / (2.0 * math.pi))
    angular_frequency = 2.0 * math.pi * torch.fft.fftfreq(length, device=device, dtype=dtype)
    shifted = scales[:, None] * angular_frequency[None, :] - float(omega0)
    filters = torch.exp(-0.5 * shifted.square())
    filters = filters * (angular_frequency[None, :] > 0).to(dtype)
    norm = torch.sqrt((filters.square().sum(dim=1, keepdim=True) / float(length)).clamp_min(1e-12))
    filters = filters / norm
    _CWT_FILTER_CACHE[key] = filters
    return filters


def render_cwt_batch(
    ohlc: torch.Tensor,
    *,
    num_scales: int = 16,
    min_period: float = 2.0,
    max_period: float = 48.0,
    omega0: float = 6.0,
    pad: int = 60,
    return_scale: float = 0.002,
    clip: float = 8.0,
) -> torch.Tensor:
    signal = _close_return_signal(ohlc, return_scale=return_scale, clip=clip)
    time_steps = int(signal.shape[-1])
    pad = min(max(0, int(pad)), max(0, time_steps - 1))
    if pad:
        padded = F.pad(signal.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    else:
        padded = signal
    filters = _morlet_filters(
        length=int(padded.shape[-1]),
        num_scales=num_scales,
        min_period=min_period,
        max_period=max_period,
        omega0=omega0,
        device=padded.device,
        dtype=padded.dtype,
    )
    spectrum = torch.fft.fft(padded, dim=-1)
    coefficients = torch.fft.ifft(spectrum[:, None, :] * filters[None, :, :], dim=-1)
    if pad:
        coefficients = coefficients[:, :, pad : pad + time_steps]
    return torch.stack(
        [coefficients.real, coefficients.imag, torch.log1p(coefficients.abs())],
        dim=1,
    )


def _render_time_frequency(ohlc: torch.Tensor, representation_cfg: dict) -> torch.Tensor:
    mode = str(representation_cfg.get("mode", "cwt")).strip().lower()
    common = {
        "return_scale": float(representation_cfg.get("return_scale", 0.002)),
        "clip": float(representation_cfg.get("clip", 8.0)),
    }
    if mode == "stft":
        return render_stft_batch(
            ohlc,
            n_fft=int(representation_cfg.get("n_fft", 32)),
            win_length=int(representation_cfg.get("win_length", 32)),
            hop_length=int(representation_cfg.get("hop_length", 4)),
            **common,
        )
    if mode in {"cwt", "candle_cwt_fusion"}:
        return render_cwt_batch(
            ohlc,
            num_scales=int(representation_cfg.get("num_scales", 16)),
            min_period=float(representation_cfg.get("min_period", 2.0)),
            max_period=float(representation_cfg.get("max_period", 48.0)),
            omega0=float(representation_cfg.get("omega0", 6.0)),
            pad=int(representation_cfg.get("pad", 60)),
            **common,
        )
    raise ValueError(f"unsupported time-frequency representation mode: {mode}")


def _build_model(representation_cfg: dict, model_cfg: dict) -> nn.Module:
    mode = str(representation_cfg.get("mode", "cwt")).strip().lower()
    if mode == "candle_cwt_fusion":
        return KlineTimeFrequencyFusionCNN(
            base_channels=int(model_cfg.get("base_channels", 12)),
            dropout=float(model_cfg.get("dropout", 0.15)),
        )
    return KlineImageCNN(
        in_channels=3,
        base_channels=int(model_cfg.get("base_channels", 16)),
        dropout=float(model_cfg.get("dropout", 0.15)),
    )


def _forward_model(
    model: nn.Module,
    ohlc: torch.Tensor,
    *,
    representation_cfg: dict,
) -> torch.Tensor:
    mode = str(representation_cfg.get("mode", "cwt")).strip().lower()
    with torch.no_grad():
        time_frequency = _render_time_frequency(ohlc, representation_cfg)
        if mode == "candle_cwt_fusion":
            candles = render_candlestick_batch(
                ohlc,
                image_height=int(representation_cfg.get("candle_image_height", 96)),
                price_limit=float(representation_cfg.get("candle_price_limit", 0.04)),
            )
        else:
            candles = None
    if candles is not None:
        return model(candles, time_frequency)
    return model(time_frequency)


def _predict(
    *,
    model: nn.Module,
    data: ImageDatasetData,
    indices: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
    representation_cfg: dict,
) -> np.ndarray:
    loader = DataLoader(
        _IndexedKlineDataset(data, indices, targets),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ohlc, _ in loader:
            prediction = _forward_model(
                model,
                ohlc.to(device),
                representation_cfg=representation_cfg,
            )
            outputs.append(prediction.detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False) if outputs else np.array([], dtype=np.float32)


def _train_model(
    *,
    data: ImageDatasetData,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    targets: np.ndarray,
    representation_cfg: dict,
    model_cfg: dict,
    train_cfg: dict,
) -> tuple[nn.Module, pd.DataFrame]:
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(train_cfg.get("torch_threads", 8))))
    device = _device_from_config(train_cfg)
    model = _build_model(representation_cfg, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.SmoothL1Loss(beta=float(train_cfg.get("huber_beta", 0.5)))
    batch_size = max(1, int(train_cfg.get("batch_size", 128)))
    train_loader = DataLoader(
        _IndexedKlineDataset(data, train_indices, targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    epochs = max(1, int(train_cfg.get("num_epochs", 4)))
    patience = max(1, int(train_cfg.get("early_stopping_patience", 2)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    best_score = -float("inf")
    best_state: dict | None = None
    stale_epochs = 0
    history: list[dict] = []
    with torch.no_grad():
        sample_shape = tuple(
            _render_time_frequency(
                torch.from_numpy(data.ohlc[train_indices[:1]]).float().to(device),
                representation_cfg,
            ).shape[1:]
        )
    mode = str(representation_cfg.get("mode", "cwt")).strip().lower()
    print(
        "kline time-frequency train:",
        f"mode={mode}",
        f"device={device}",
        f"train_samples={len(train_indices)}",
        f"valid_samples={len(valid_indices)}",
        f"epochs={epochs}",
        f"batch_size={batch_size}",
        f"representation={sample_shape}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for ohlc, target in train_loader:
            ohlc = ohlc.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_model(model, ohlc, representation_cfg=representation_cfg)
            loss = criterion(prediction, target)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        valid_pred = _predict(
            model=model,
            data=data,
            indices=valid_indices,
            targets=targets,
            batch_size=batch_size,
            device=device,
            representation_cfg=representation_cfg,
        )
        valid_rank_ic = _daily_rank_ic(
            data.trade_date[valid_indices],
            data.raw_return[valid_indices],
            valid_pred,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_rank_ic": valid_rank_ic,
        }
        history.append(row)
        print(
            f"[kline_tf:epoch] {epoch}/{epochs} "
            f"train_loss={row['train_loss']:.6f} valid_rank_ic={valid_rank_ic:.6f}",
            flush=True,
        )
        score = valid_rank_ic if math.isfinite(valid_rank_ic) else -float("inf")
        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[kline_tf:early_stop] epoch={epoch} best_rank_ic={best_score:.6f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def _cache_fingerprint(
    *,
    files: list[tuple[date, Path]],
    raw_data_root: str,
    data_cfg: dict,
    buy_cost_bps: float,
    sell_cost_bps: float,
) -> str:
    file_state = [
        {
            "day": str(day),
            "path": str(path),
            "size": int(path.stat().st_size),
            "mtime_ns": int(path.stat().st_mtime_ns),
        }
        for day, path in files
    ]
    payload = {
        "files": file_state,
        "raw_data_root": str(raw_data_root),
        "data": data_cfg,
        "buy_cost_bps": float(buy_cost_bps),
        "sell_cost_bps": float(sell_cost_bps),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_data_cache(path: Path, fingerprint: str) -> ImageDatasetData | None:
    meta_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("fingerprint")) != fingerprint:
            return None
        with np.load(path, allow_pickle=False) as saved:
            dates = np.asarray(saved["trade_date"]).astype(str)
            return ImageDatasetData(
                ohlc=np.asarray(saved["ohlc"]),
                trade_date=np.asarray([date.fromisoformat(item) for item in dates], dtype=object),
                code=np.asarray(saved["code"]).astype(str),
                raw_return=np.asarray(saved["raw_return"], dtype=np.float32),
                morning_return=np.asarray(saved["morning_return"], dtype=np.float32),
                buy_price=np.asarray(saved["buy_price"], dtype=np.float32),
                sell_price=np.asarray(saved["sell_price"], dtype=np.float32),
                sell_fallback=np.asarray(saved["sell_fallback"], dtype=bool),
                morning_coverage=np.asarray(saved["morning_coverage"], dtype=np.int16),
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_data_cache(path: Path, fingerprint: str, data: ImageDatasetData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez(
            handle,
            ohlc=data.ohlc,
            trade_date=np.asarray([str(item) for item in data.trade_date], dtype="U10"),
            code=np.asarray(data.code, dtype=str),
            raw_return=data.raw_return,
            morning_return=data.morning_return,
            buy_price=data.buy_price,
            sell_price=data.sell_price,
            sell_fallback=data.sell_fallback,
            morning_coverage=data.morning_coverage,
        )
    temp_path.replace(path)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "samples": int(len(data.code)),
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _monthly_returns(daily: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "cnn_return",
        "momentum_topk_return",
        "reversal_topk_return",
        "pool_equal_return",
    ]
    work = daily.copy()
    work["month"] = pd.to_datetime(work["trade_date"]).dt.to_period("M").astype(str)
    monthly = work.groupby("month")[columns].agg(lambda values: (1.0 + values).prod() - 1.0)
    monthly["model_excess_vs_pool"] = monthly["cnn_return"] - monthly["pool_equal_return"]
    return monthly.reset_index()


def _write_time_frequency_samples(
    *,
    out_dir: Path,
    data: ImageDatasetData,
    indices: np.ndarray,
    representation_cfg: dict,
    count: int,
) -> pd.DataFrame:
    sample_dir = out_dir / "sample_time_frequency"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx in indices[: max(0, int(count))]:
        ohlc = torch.from_numpy(data.ohlc[int(idx)]).unsqueeze(0).float()
        with torch.no_grad():
            representation = _render_time_frequency(ohlc, representation_cfg)[0].numpy()
        magnitude = np.asarray(representation[2], dtype=np.float32)
        low, high = np.nanpercentile(magnitude, [1.0, 99.0])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(magnitude)), float(np.nanmax(magnitude) + 1e-6)
        normalized = np.clip((magnitude - low) / (high - low), 0.0, 1.0)
        pixels = np.asarray(np.flipud(normalized) * 255.0, dtype=np.uint8)
        day = data.trade_date[int(idx)]
        code = str(data.code[int(idx)])
        path = sample_dir / f"{day}_{code.replace('.', '_')}.png"
        image = Image.fromarray(pixels)
        image.resize((pixels.shape[1] * 4, pixels.shape[0] * 8), Image.Resampling.NEAREST).save(path)
        rows.append(
            {
                "trade_date": day,
                "code": code,
                "path": str(path),
                "shape": str(tuple(representation.shape)),
                "morning_return": float(data.morning_return[int(idx)]),
                "return_net": float(data.raw_return[int(idx)])
                if np.isfinite(data.raw_return[int(idx)])
                else None,
            }
        )
    return pd.DataFrame(rows)


def main(
    *,
    config_path: Path | None,
    start: str | date | None = None,
    end: str | date | None = None,
    label_cutoff: str | date | None = None,
    execution: dict | None = None,
) -> None:
    _ = label_cutoff
    if config_path is None:
        raise ValueError("kline time-frequency model config path is required")
    cfg = dict(load_json_like(config_path))
    paths_cfg = load_config_file("paths")
    split_cfg = dict(cfg.get("split", {}))
    data_cfg = dict(cfg.get("data", {}))
    representation_cfg = dict(cfg.get("representation", {}))
    model_cfg = dict(cfg.get("model_params", {}))
    train_cfg = dict(cfg.get("train", {}))
    execution_cfg = dict(execution or {})

    train_start = parse_date(split_cfg.get("train_start"))
    train_end = parse_date(split_cfg.get("train_end"))
    valid_start = parse_date(split_cfg.get("valid_start"))
    valid_end = parse_date(split_cfg.get("valid_end"))
    test_start = parse_date(start or split_cfg.get("test_start"))
    test_end = parse_date(end or split_cfg.get("test_end"))
    if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
        raise ValueError(
            "kline time-frequency split must satisfy train <= valid < test without overlap: "
            f"train={train_start}..{train_end} valid={valid_start}..{valid_end} "
            f"test={test_start}..{test_end}"
        )

    kline_root = _platform_path(data_cfg.get("kline_root"))
    if not kline_root.exists():
        raise FileNotFoundError(f"kline root missing: {kline_root}")
    files = _files_for_range(kline_root, train_start, test_end)
    if not files:
        raise FileNotFoundError(f"no kline files under {kline_root} for {train_start}..{test_end}")
    pool_cfg = load_upstream_pool_config(dict(data_cfg.get("allowlist", {})))
    buy_cost_bps, sell_cost_bps, fee_source = load_fees_buy_sell_bps()
    fingerprint = _cache_fingerprint(
        files=files,
        raw_data_root=str(paths_cfg["raw_data_root"]),
        data_cfg=data_cfg,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
    )
    cache_cfg = dict(cfg.get("data_cache", {}))
    cache_path = _platform_path(cache_cfg.get("path")) if cache_cfg.get("path") else None
    data = _load_data_cache(cache_path, fingerprint) if cache_path is not None else None
    if data is None:
        print(
            "kline time-frequency data:",
            f"files={len(files)}",
            f"range={files[0][0]}..{files[-1][0]}",
            "cache=miss",
            flush=True,
        )
        workers = int(execution_cfg.get("prep_workers", train_cfg.get("prep_workers", 8)))
        data = _load_image_data(
            files=files,
            raw_data_root=str(paths_cfg["raw_data_root"]),
            pool_cfg=pool_cfg,
            data_cfg=data_cfg,
            buy_cost_bps=buy_cost_bps,
            sell_cost_bps=sell_cost_bps,
            workers=workers,
        )
        if cache_path is not None:
            _write_data_cache(cache_path, fingerprint, data)
            print(f"kline time-frequency data cache saved: {cache_path}", flush=True)
    else:
        print(f"kline time-frequency data cache hit: {cache_path} samples={len(data.code)}", flush=True)
    if data.ohlc.size == 0:
        raise RuntimeError("kline time-frequency preparation produced no samples")

    targets = _daily_zscore_targets(data)
    train_indices = _indices_between(data.trade_date, train_start, train_end)
    valid_indices = _indices_between(data.trade_date, valid_start, valid_end)
    test_indices = _indices_between(data.trade_date, test_start, test_end)
    train_indices = train_indices[np.isfinite(targets[train_indices])]
    valid_indices = valid_indices[np.isfinite(targets[valid_indices])]
    if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
        raise RuntimeError(
            "empty train/valid/test split after preparation: "
            f"train={len(train_indices)} valid={len(valid_indices)} test={len(test_indices)}"
        )

    model, history = _train_model(
        data=data,
        train_indices=train_indices,
        valid_indices=valid_indices,
        targets=targets,
        representation_cfg=representation_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    predictions = _predict(
        model=model,
        data=data,
        indices=test_indices,
        targets=targets,
        batch_size=int(train_cfg.get("batch_size", 128)),
        device=_device_from_config(train_cfg),
        representation_cfg=representation_cfg,
    )

    model_name = str(cfg.get("model_name", "kline_time_frequency_v1"))
    results_root = resolve_output_path(
        cfg.get("results_root"),
        default_path=Path(paths_cfg["results_root"]) / "analysis" / model_name,
        results_root=paths_cfg["results_root"],
    )
    out_dir = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    score_output = resolve_output_path(
        cfg.get("score_output"),
        default_path=Path(paths_cfg["results_root"]) / "scores" / "kline_time_frequency" / model_name,
        results_root=paths_cfg["results_root"],
    )
    scores = pd.DataFrame(
        {
            "trade_date": data.trade_date[test_indices],
            "code": data.code[test_indices],
            "score": predictions,
        }
    )
    write_scores_by_date(score_output, scores, overwrite=True, dedupe=True)
    daily, nav, positions, ic = _evaluate_topk(
        data=data,
        test_indices=test_indices,
        predictions=predictions,
        top_k=int(cfg.get("top_k", 20)),
    )
    metrics = pd.DataFrame(
        [
            {"strategy": model_name, **_portfolio_metrics(daily["cnn_return"])},
            {"strategy": "morning_momentum_topk", **_portfolio_metrics(daily["momentum_topk_return"])},
            {"strategy": "morning_reversal_topk", **_portfolio_metrics(daily["reversal_topk_return"])},
            {"strategy": "eligible_pool_equal", **_portfolio_metrics(daily["pool_equal_return"])},
        ]
    )
    history.to_csv(out_dir / "metrics_iter.csv", index=False)
    scores.to_csv(out_dir / "scores.csv", index=False)
    daily.to_csv(out_dir / "daily_returns.csv", index=False)
    nav.to_csv(out_dir / "nav_curve.csv", index=False)
    positions.to_csv(out_dir / "positions.csv", index=False)
    ic.to_csv(out_dir / "score_eval_daily.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    _monthly_returns(daily).to_csv(out_dir / "monthly_returns.csv", index=False)
    plot_path = _write_plot(
        out_dir,
        daily,
        nav,
        strategy_label=model_name,
        title="Morning Kline Time-Frequency Intraday Backtest",
    )
    samples = _write_time_frequency_samples(
        out_dir=out_dir,
        data=data,
        indices=test_indices,
        representation_cfg=representation_cfg,
        count=int(cfg.get("sample_image_count", 8)),
    )
    samples.to_csv(out_dir / "sample_time_frequency.csv", index=False)
    torch.save(model.state_dict(), out_dir / "model.pt")

    summary = {
        "model_name": model_name,
        "model_type": "kline_time_frequency_cnn",
        "representation": representation_cfg,
        "split": {
            "train": [str(train_start), str(train_end)],
            "valid": [str(valid_start), str(valid_end)],
            "test": [str(test_start), str(test_end)],
            "warm_start": False,
            "test_refit": False,
        },
        "label": {
            "input": f"{data_cfg.get('morning_start', '09:30')}..{data_cfg.get('morning_end_exclusive', '11:30')} exclusive",
            "buy": f"{data_cfg.get('buy_start', '13:00')}..{data_cfg.get('buy_end_exclusive', '13:05')} exclusive",
            "sell": f"{data_cfg.get('sell_start', '14:50')}..{data_cfg.get('sell_end_exclusive', '14:57')} exclusive",
            "buy_cost_bps": buy_cost_bps,
            "sell_cost_bps": sell_cost_bps,
            "fee_source": fee_source,
        },
        "samples": {
            "all": int(len(data.code)),
            "train": int(len(train_indices)),
            "valid": int(len(valid_indices)),
            "test": int(len(test_indices)),
            "test_days": int(pd.Series(data.trade_date[test_indices]).nunique()),
            "test_missing_return": int((~np.isfinite(data.raw_return[test_indices])).sum()),
        },
        "top_k": int(cfg.get("top_k", 20)),
        "metrics": metrics.to_dict(orient="records"),
        "rank_ic_mean": float(pd.to_numeric(ic["rank_ic"], errors="coerce").mean()),
        "data_cache": str(cache_path) if cache_path is not None else None,
        "score_output": str(score_output),
        "plot_path": str(plot_path),
        "result_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"saved kline time-frequency experiment: {out_dir}", flush=True)
