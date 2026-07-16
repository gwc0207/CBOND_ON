from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import copy
import json
import math
import random
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cbond_on.common.config_utils import load_json_like
from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.fees import load_fees_buy_sell_bps
from cbond_on.infra.model.impl.torch_image import KlineImageCNN
from cbond_on.infra.model.score_io import write_scores_by_date
from cbond_on.infra.universe.pool_filter import (
    UpstreamPoolConfig,
    load_upstream_pool_config,
    resolve_pool_codes_for_trade_day,
)


OHLC_COLUMNS = ["open_price", "high_price", "low_price", "close_price"]
KLINE_COLUMNS = [
    "trade_time",
    "instrument_code",
    "exchange_code",
    *OHLC_COLUMNS,
    "prev_close_price",
    "vwap",
    "twap",
]


@dataclass
class DaySamples:
    trade_date: date
    ohlc: np.ndarray
    code: np.ndarray
    raw_return: np.ndarray
    morning_return: np.ndarray
    buy_price: np.ndarray
    sell_price: np.ndarray
    sell_fallback: np.ndarray
    morning_coverage: np.ndarray


@dataclass
class ImageDatasetData:
    ohlc: np.ndarray
    trade_date: np.ndarray
    code: np.ndarray
    raw_return: np.ndarray
    morning_return: np.ndarray
    buy_price: np.ndarray
    sell_price: np.ndarray
    sell_fallback: np.ndarray
    morning_coverage: np.ndarray


class _IndexedKlineDataset(Dataset):
    def __init__(self, data: ImageDatasetData, indices: np.ndarray, targets: np.ndarray) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = targets

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.data.ohlc[idx]).float(),
            torch.tensor(float(self.targets[idx]), dtype=torch.float32),
        )


def _platform_path(value: object) -> Path:
    if not isinstance(value, dict):
        return Path(str(value)).expanduser()
    normalized = {str(key).strip().lower(): item for key, item in value.items()}
    keys = ("windows", "win") if sys.platform.startswith("win") else ("linux", "unix", "posix")
    for key in (*keys, "default", "common"):
        picked = normalized.get(key)
        if picked not in (None, ""):
            return Path(str(picked)).expanduser()
    raise ValueError("platform path has no usable value")


def _minute_of_day(text: object) -> int:
    parts = str(text).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"invalid intraday time: {text}")
    return int(parts[0]) * 60 + int(parts[1])


def _empty_day(day: date, sequence_length: int) -> DaySamples:
    return DaySamples(
        trade_date=day,
        ohlc=np.empty((0, sequence_length, 4), dtype=np.float16),
        code=np.array([], dtype=object),
        raw_return=np.array([], dtype=np.float32),
        morning_return=np.array([], dtype=np.float32),
        buy_price=np.array([], dtype=np.float32),
        sell_price=np.array([], dtype=np.float32),
        sell_fallback=np.array([], dtype=bool),
        morning_coverage=np.array([], dtype=np.int16),
    )


def _positive_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out[out.notna() & (out > 0)]


def _extract_day_samples_from_frame(
    frame: pd.DataFrame,
    *,
    trade_day: date,
    allowed_codes: set[str] | None,
    data_cfg: dict,
    buy_cost_bps: float,
    sell_cost_bps: float,
) -> DaySamples:
    morning_start = _minute_of_day(data_cfg.get("morning_start", "09:30"))
    morning_end = _minute_of_day(data_cfg.get("morning_end_exclusive", "11:30"))
    buy_start = _minute_of_day(data_cfg.get("buy_start", "13:00"))
    buy_end = _minute_of_day(data_cfg.get("buy_end_exclusive", "13:05"))
    sell_start = _minute_of_day(data_cfg.get("sell_start", "14:50"))
    sell_end = _minute_of_day(data_cfg.get("sell_end_exclusive", "14:57"))
    session_end = _minute_of_day(data_cfg.get("session_end_exclusive", "15:00"))
    min_morning_bars = int(data_cfg.get("min_morning_bars", morning_end - morning_start))
    min_buy_bars = int(data_cfg.get("min_buy_bars", 3))
    min_sell_bars = int(data_cfg.get("min_sell_bars", 4))
    sequence_length = morning_end - morning_start
    if sequence_length <= 0:
        raise ValueError("morning_end_exclusive must be after morning_start")

    work = frame.copy()
    if "code" not in work.columns:
        instrument = work["instrument_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        exchange = work["exchange_code"].astype(str).str.strip().str.upper()
        work["code"] = instrument + "." + exchange
    if allowed_codes is not None:
        work = work[work["code"].isin(allowed_codes)]
    if work.empty:
        return _empty_day(trade_day, sequence_length)

    if "vwap" not in work.columns:
        work["vwap"] = np.nan
    for col in [*OHLC_COLUMNS, "prev_close_price", "vwap", "twap"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["minute"] = pd.to_numeric(work["minute"], errors="coerce")
    work = work.dropna(subset=["code", "minute"])

    ohlc_rows: list[np.ndarray] = []
    codes: list[str] = []
    raw_returns: list[float] = []
    morning_returns: list[float] = []
    buy_prices: list[float] = []
    sell_prices: list[float] = []
    sell_fallbacks: list[bool] = []
    coverages: list[int] = []
    total_cost = (float(buy_cost_bps) + float(sell_cost_bps)) * 1e-4

    for code, group in work.groupby("code", sort=False):
        group = group.sort_values("minute").drop_duplicates(subset=["minute"], keep="last")
        group = group.set_index("minute")
        if morning_start not in group.index:
            continue
        morning = group.reindex(range(morning_start, morning_end))
        coverage = int(pd.to_numeric(morning["close_price"], errors="coerce").notna().sum())
        if coverage < min_morning_bars:
            continue
        anchor = pd.to_numeric(pd.Series([group.at[morning_start, "prev_close_price"]]), errors="coerce").iloc[0]
        if not np.isfinite(anchor) or float(anchor) <= 0:
            continue

        bars = morning[OHLC_COLUMNS].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if bars.isna().any(axis=None):
            continue
        values = bars.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(values).all() or (values <= 0).any():
            continue
        values[:, 1] = np.maximum.reduce([values[:, 1], values[:, 0], values[:, 3]])
        values[:, 2] = np.minimum.reduce([values[:, 2], values[:, 0], values[:, 3]])
        relative = np.log(values / float(anchor)).astype(np.float16)

        buy_window = group.loc[(group.index >= buy_start) & (group.index < buy_end)]
        buy = _positive_numeric(buy_window["twap"])
        if len(buy) < min_buy_bars:
            buy = _positive_numeric(buy_window["vwap"])
        if len(buy) < min_buy_bars:
            buy = _positive_numeric(buy_window["close_price"])
        sell_window = group.loc[(group.index >= sell_start) & (group.index < sell_end)]
        sell = _positive_numeric(sell_window["twap"])
        buy_price = float(buy.mean()) if len(buy) >= min_buy_bars else float("nan")
        fallback = False
        if len(sell) >= min_sell_bars:
            sell_price = float(sell.mean())
        else:
            sell = _positive_numeric(sell_window["vwap"])
            if len(sell) < min_sell_bars:
                sell = _positive_numeric(sell_window["close_price"])
            if len(sell) >= min_sell_bars:
                sell_price = float(sell.mean())
                fallback = True
            else:
                fallback_values = _positive_numeric(
                    group.loc[(group.index >= buy_end) & (group.index < session_end), "twap"]
                )
                if fallback_values.empty:
                    fallback_values = _positive_numeric(
                        group.loc[(group.index >= buy_end) & (group.index < session_end), "vwap"]
                    )
                if fallback_values.empty:
                    fallback_values = _positive_numeric(
                        group.loc[(group.index >= buy_end) & (group.index < session_end), "close_price"]
                    )
                sell_price = float(fallback_values.iloc[-1]) if not fallback_values.empty else float("nan")
                fallback = np.isfinite(sell_price)

        if np.isfinite(buy_price) and np.isfinite(sell_price) and buy_price > 0 and sell_price > 0:
            raw_return = float(sell_price / buy_price - 1.0 - total_cost)
        else:
            raw_return = float("nan")

        ohlc_rows.append(relative)
        codes.append(str(code))
        raw_returns.append(raw_return)
        morning_returns.append(float(values[-1, 3] / float(anchor) - 1.0))
        buy_prices.append(buy_price)
        sell_prices.append(sell_price)
        sell_fallbacks.append(bool(fallback))
        coverages.append(coverage)

    if not ohlc_rows:
        return _empty_day(trade_day, sequence_length)
    return DaySamples(
        trade_date=trade_day,
        ohlc=np.stack(ohlc_rows).astype(np.float16, copy=False),
        code=np.asarray(codes, dtype=object),
        raw_return=np.asarray(raw_returns, dtype=np.float32),
        morning_return=np.asarray(morning_returns, dtype=np.float32),
        buy_price=np.asarray(buy_prices, dtype=np.float32),
        sell_price=np.asarray(sell_prices, dtype=np.float32),
        sell_fallback=np.asarray(sell_fallbacks, dtype=bool),
        morning_coverage=np.asarray(coverages, dtype=np.int16),
    )


def _read_day_samples(
    path: Path,
    *,
    trade_day: date,
    raw_data_root: str,
    pool_cfg: UpstreamPoolConfig,
    data_cfg: dict,
    buy_cost_bps: float,
    sell_cost_bps: float,
) -> DaySamples:
    pool_codes, pool_info = resolve_pool_codes_for_trade_day(
        raw_data_root=raw_data_root,
        trade_day=trade_day,
        pool_cfg=pool_cfg,
        enabled=True,
    )
    if bool(pool_info.get("fallback_no_filter", False)):
        raise RuntimeError(
            "required o_0005 allowlist unavailable: "
            f"trade_day={trade_day} reason={pool_info.get('fallback_reason')}"
        )

    source_mode = str(data_cfg.get("source_mode", "minute_kline")).strip().lower()
    morning_start = _minute_of_day(data_cfg.get("morning_start", "09:30"))
    morning_end = _minute_of_day(data_cfg.get("morning_end_exclusive", "11:30"))
    afternoon_start = _minute_of_day(data_cfg.get("buy_start", "13:00"))
    session_end = _minute_of_day(data_cfg.get("session_end_exclusive", "15:00"))
    if source_mode == "clean_snapshot":
        table = pq.read_table(
            path,
            columns=["code", "trade_time", "pre_close", "last"],
            use_threads=False,
        )
        time_ns = np.asarray(pc.cast(table["trade_time"], pa.int64())).astype(np.int64, copy=False)
        minute = ((time_ns // 60_000_000_000) % (24 * 60)).astype(np.int16, copy=False)
    elif source_mode == "minute_kline":
        table = pq.read_table(path, columns=KLINE_COLUMNS, use_threads=False)
        time_us = np.asarray(pc.cast(table["trade_time"], pa.int64())).astype(np.int64, copy=False)
        minute = (time_us // 60_000_000).astype(np.int16, copy=False)
    else:
        raise ValueError(f"unsupported kline image data.source_mode: {source_mode}")
    keep = ((minute >= morning_start) & (minute < morning_end)) | (
        (minute >= afternoon_start) & (minute < session_end)
    )
    selected = np.flatnonzero(keep)
    table = table.take(pa.array(selected)).append_column("minute", pa.array(minute[keep]))
    frame = table.to_pandas()
    if source_mode == "clean_snapshot":
        frame["last"] = pd.to_numeric(frame["last"], errors="coerce")
        frame = (
            frame.sort_values(["code", "trade_time"])
            .groupby(["code", "minute"], sort=False)
            .agg(
                open_price=("last", "first"),
                high_price=("last", "max"),
                low_price=("last", "min"),
                close_price=("last", "last"),
                prev_close_price=("pre_close", "first"),
                twap=("last", "mean"),
                vwap=("last", "mean"),
            )
            .reset_index()
        )
    return _extract_day_samples_from_frame(
        frame,
        trade_day=trade_day,
        allowed_codes=pool_codes,
        data_cfg=data_cfg,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
    )


def _date_from_path(path: Path) -> date | None:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(path.stem, fmt).date()
        except ValueError:
            continue
    return None


def _files_for_range(root: Path, start: date, end: date) -> list[tuple[date, Path]]:
    rows: list[tuple[date, Path]] = []
    for path in root.glob("20??-??/*.parquet"):
        day = _date_from_path(path)
        if day is not None and start <= day <= end:
            rows.append((day, path))
    rows.sort(key=lambda item: item[0])
    return rows


def _concat_day_samples(rows: list[DaySamples], sequence_length: int) -> ImageDatasetData:
    valid = [row for row in rows if row.ohlc.size > 0]
    if not valid:
        return ImageDatasetData(
            ohlc=np.empty((0, sequence_length, 4), dtype=np.float16),
            trade_date=np.array([], dtype=object),
            code=np.array([], dtype=object),
            raw_return=np.array([], dtype=np.float32),
            morning_return=np.array([], dtype=np.float32),
            buy_price=np.array([], dtype=np.float32),
            sell_price=np.array([], dtype=np.float32),
            sell_fallback=np.array([], dtype=bool),
            morning_coverage=np.array([], dtype=np.int16),
        )
    return ImageDatasetData(
        ohlc=np.concatenate([row.ohlc for row in valid], axis=0).astype(np.float16, copy=False),
        trade_date=np.concatenate(
            [np.asarray([row.trade_date] * len(row.code), dtype=object) for row in valid]
        ),
        code=np.concatenate([row.code for row in valid]),
        raw_return=np.concatenate([row.raw_return for row in valid]).astype(np.float32, copy=False),
        morning_return=np.concatenate([row.morning_return for row in valid]).astype(np.float32, copy=False),
        buy_price=np.concatenate([row.buy_price for row in valid]).astype(np.float32, copy=False),
        sell_price=np.concatenate([row.sell_price for row in valid]).astype(np.float32, copy=False),
        sell_fallback=np.concatenate([row.sell_fallback for row in valid]),
        morning_coverage=np.concatenate([row.morning_coverage for row in valid]),
    )


def _load_image_data(
    *,
    files: list[tuple[date, Path]],
    raw_data_root: str,
    pool_cfg: UpstreamPoolConfig,
    data_cfg: dict,
    buy_cost_bps: float,
    sell_cost_bps: float,
    workers: int,
) -> ImageDatasetData:
    sequence_length = _minute_of_day(data_cfg.get("morning_end_exclusive", "11:30")) - _minute_of_day(
        data_cfg.get("morning_start", "09:30")
    )
    by_day: dict[date, DaySamples] = {}
    total = len(files)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(
                _read_day_samples,
                path,
                trade_day=day,
                raw_data_root=raw_data_root,
                pool_cfg=pool_cfg,
                data_cfg=data_cfg,
                buy_cost_bps=buy_cost_bps,
                sell_cost_bps=sell_cost_bps,
            ): (day, path)
            for day, path in files
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            day, path = futures[future]
            try:
                by_day[day] = future.result()
            except Exception as exc:
                raise RuntimeError(f"failed to prepare kline image day {day} from {path}: {exc}") from exc
            if completed == 1 or completed % 20 == 0 or completed == total:
                sample_count = sum(len(item.code) for item in by_day.values())
                print(
                    f"[kline_image:prepare] {completed}/{total} "
                    f"last_day={day} samples={sample_count}",
                    flush=True,
                )
    ordered = [by_day[day] for day, _ in files if day in by_day]
    return _concat_day_samples(ordered, sequence_length)


def render_candlestick_batch(
    ohlc: torch.Tensor,
    *,
    image_height: int,
    price_limit: float,
) -> torch.Tensor:
    if ohlc.ndim != 3 or ohlc.shape[-1] != 4:
        raise ValueError(f"ohlc must have shape (batch, time, 4), got {tuple(ohlc.shape)}")
    limit = abs(float(price_limit))
    if limit <= 0:
        raise ValueError("price_limit must be positive")
    height = max(8, int(image_height))
    clipped = torch.clamp(ohlc, min=-limit, max=limit)
    y = torch.round((limit - clipped) * ((height - 1) / (2.0 * limit))).long()
    y = torch.clamp(y, min=0, max=height - 1)
    y_open = y[:, :, 0]
    y_high = torch.minimum(y[:, :, 1], torch.minimum(y_open, y[:, :, 3]))
    y_low = torch.maximum(y[:, :, 2], torch.maximum(y_open, y[:, :, 3]))
    y_close = y[:, :, 3]
    grid = torch.arange(height, device=ohlc.device).view(1, height, 1)
    wick = (grid >= y_high.unsqueeze(1)) & (grid <= y_low.unsqueeze(1))
    body_top = torch.minimum(y_open, y_close)
    body_bottom = torch.maximum(y_open, y_close)
    body = (grid >= body_top.unsqueeze(1)) & (grid <= body_bottom.unsqueeze(1))
    up = (ohlc[:, :, 3] >= ohlc[:, :, 0]).unsqueeze(1)
    return torch.stack(
        [wick.float(), (body & up).float(), (body & ~up).float()],
        dim=1,
    )


def _daily_zscore_targets(data: ImageDatasetData) -> np.ndarray:
    targets = np.full(len(data.raw_return), np.nan, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "idx": np.arange(len(data.raw_return), dtype=np.int64),
            "trade_date": data.trade_date,
            "raw_return": data.raw_return,
        }
    )
    for _, group in frame.groupby("trade_date", sort=False):
        values = pd.to_numeric(group["raw_return"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        mean = float(values[finite].mean())
        std = float(values[finite].std(ddof=0))
        transformed = values - mean
        if std > 1e-12:
            transformed = transformed / std
        targets[group.loc[finite, "idx"].to_numpy(dtype=np.int64)] = transformed[finite].astype(np.float32)
    return targets


def _indices_between(days: np.ndarray, start: date, end: date) -> np.ndarray:
    return np.flatnonzero(np.asarray([(start <= item <= end) for item in days], dtype=bool))


def _device_from_config(train_cfg: dict) -> torch.device:
    requested = str(train_cfg.get("device", "cpu")).strip().lower()
    if requested in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _predict(
    *,
    model: nn.Module,
    data: ImageDatasetData,
    indices: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: torch.device,
    image_height: int,
    price_limit: float,
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
            ohlc = ohlc.to(device)
            images = render_candlestick_batch(
                ohlc,
                image_height=image_height,
                price_limit=price_limit,
            )
            outputs.append(model(images).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False) if outputs else np.array([], dtype=np.float32)


def _daily_rank_ic(days: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> float:
    frame = pd.DataFrame({"trade_date": days, "actual": actual, "predicted": predicted})
    values: list[float] = []
    for _, group in frame.groupby("trade_date"):
        group = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(group) < 3:
            continue
        corr = group["actual"].corr(group["predicted"], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def _train_model(
    *,
    data: ImageDatasetData,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    targets: np.ndarray,
    model_cfg: dict,
    train_cfg: dict,
    data_cfg: dict,
) -> tuple[nn.Module, pd.DataFrame]:
    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(train_cfg.get("torch_threads", 8))))
    device = _device_from_config(train_cfg)
    model = KlineImageCNN(
        in_channels=3,
        base_channels=int(model_cfg.get("base_channels", 16)),
        dropout=float(model_cfg.get("dropout", 0.15)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.SmoothL1Loss(beta=float(train_cfg.get("huber_beta", 0.5)))
    batch_size = max(1, int(train_cfg.get("batch_size", 256)))
    train_loader = DataLoader(
        _IndexedKlineDataset(data, train_indices, targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    image_height = int(data_cfg.get("image_height", 64))
    price_limit = float(data_cfg.get("price_limit", 0.10))
    epochs = max(1, int(train_cfg.get("num_epochs", 4)))
    patience = max(1, int(train_cfg.get("early_stopping_patience", 2)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    best_score = -float("inf")
    best_state: dict | None = None
    stale_epochs = 0
    history: list[dict] = []

    print(
        "kline image train:",
        f"device={device}",
        f"train_samples={len(train_indices)}",
        f"valid_samples={len(valid_indices)}",
        f"epochs={epochs}",
        f"batch_size={batch_size}",
        f"image={image_height}x{data.ohlc.shape[1]}",
        f"price_limit={price_limit}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for ohlc, target in train_loader:
            ohlc = ohlc.to(device)
            target = target.to(device)
            images = render_candlestick_batch(
                ohlc,
                image_height=image_height,
                price_limit=price_limit,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
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
            image_height=image_height,
            price_limit=price_limit,
        )
        valid_actual = data.raw_return[valid_indices]
        valid_rank_ic = _daily_rank_ic(data.trade_date[valid_indices], valid_actual, valid_pred)
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "valid_rank_ic": valid_rank_ic,
        }
        history.append(epoch_row)
        print(
            f"[kline_image:epoch] {epoch}/{epochs} "
            f"train_loss={epoch_row['train_loss']:.6f} valid_rank_ic={valid_rank_ic:.6f}",
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
                print(f"[kline_image:early_stop] epoch={epoch} best_rank_ic={best_score:.6f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def _portfolio_metrics(returns: pd.Series) -> dict:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return {
            "days": 0,
            "period_return": float("nan"),
            "annualized_return": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "win_rate": float("nan"),
        }
    nav = (1.0 + values).cumprod()
    period_return = float(nav.iloc[-1] - 1.0)
    annualized = float(nav.iloc[-1] ** (252.0 / len(values)) - 1.0) if nav.iloc[-1] > 0 else float("nan")
    std = float(values.std(ddof=1))
    sharpe = float(values.mean() / std * math.sqrt(252.0)) if std > 1e-12 else float("nan")
    drawdown = nav / nav.cummax() - 1.0
    return {
        "days": int(len(values)),
        "period_return": period_return,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((values > 0).mean()),
        "mean_daily_return": float(values.mean()),
    }


def _evaluate_topk(
    *,
    data: ImageDatasetData,
    test_indices: np.ndarray,
    predictions: np.ndarray,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "trade_date": data.trade_date[test_indices],
            "code": data.code[test_indices],
            "score": predictions,
            "morning_return": data.morning_return[test_indices],
            "return_net": data.raw_return[test_indices],
            "buy_price": data.buy_price[test_indices],
            "sell_price": data.sell_price[test_indices],
            "sell_fallback": data.sell_fallback[test_indices],
        }
    )
    daily_rows: list[dict] = []
    position_rows: list[dict] = []
    ic_rows: list[dict] = []
    k = max(1, int(top_k))
    for day, group in frame.groupby("trade_date", sort=True):
        group = group.copy()
        labeled = group[pd.to_numeric(group["return_net"], errors="coerce").notna()]
        rank_ic = (
            float(labeled["score"].corr(labeled["return_net"], method="spearman"))
            if len(labeled) >= 3
            else float("nan")
        )
        ic_rows.append({"trade_date": day, "rank_ic": rank_ic, "count": int(len(labeled))})

        cnn = group.sort_values(["score", "code"], ascending=[False, True]).head(k).copy()
        momentum = group.sort_values(["morning_return", "code"], ascending=[False, True]).head(k).copy()
        reversal = group.sort_values(["morning_return", "code"], ascending=[True, True]).head(k).copy()
        cnn_values = pd.to_numeric(cnn["return_net"], errors="coerce").fillna(0.0)
        momentum_values = pd.to_numeric(momentum["return_net"], errors="coerce").fillna(0.0)
        reversal_values = pd.to_numeric(reversal["return_net"], errors="coerce").fillna(0.0)
        pool_values = pd.to_numeric(group["return_net"], errors="coerce").fillna(0.0)
        daily_rows.append(
            {
                "trade_date": day,
                "count": int(len(cnn)),
                "cnn_return": float(cnn_values.mean()),
                "momentum_topk_return": float(momentum_values.mean()),
                "reversal_topk_return": float(reversal_values.mean()),
                "pool_equal_return": float(pool_values.mean()),
                "cnn_missing_return_count": int(pd.to_numeric(cnn["return_net"], errors="coerce").isna().sum()),
                "cnn_fallback_sell_count": int(cnn["sell_fallback"].astype(bool).sum()),
            }
        )
        for rank, (_, row) in enumerate(cnn.iterrows(), start=1):
            position_rows.append(
                {
                    "trade_date": day,
                    "code": row["code"],
                    "rank": rank,
                    "weight": 1.0 / len(cnn),
                    "score": float(row["score"]),
                    "morning_return": float(row["morning_return"]),
                    "buy_price": row["buy_price"],
                    "sell_price": row["sell_price"],
                    "sell_fallback": bool(row["sell_fallback"]),
                    "return_net": row["return_net"],
                }
            )
    daily = pd.DataFrame(daily_rows).sort_values("trade_date")
    nav = daily[["trade_date"]].copy()
    nav["cnn_nav"] = (1.0 + daily["cnn_return"].fillna(0.0)).cumprod()
    nav["momentum_topk_nav"] = (1.0 + daily["momentum_topk_return"].fillna(0.0)).cumprod()
    nav["reversal_topk_nav"] = (1.0 + daily["reversal_topk_return"].fillna(0.0)).cumprod()
    nav["pool_equal_nav"] = (1.0 + daily["pool_equal_return"].fillna(0.0)).cumprod()
    return daily, nav, pd.DataFrame(position_rows), pd.DataFrame(ic_rows)


def _write_plot(
    out_dir: Path,
    daily: pd.DataFrame,
    nav: pd.DataFrame,
    *,
    strategy_label: str = "Kline image CNN",
    title: str = "Morning Kline Image Intraday Backtest",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(nav))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].plot(x, nav["cnn_nav"], label=strategy_label, linewidth=1.8)
    axes[0].plot(x, nav["momentum_topk_nav"], label="Morning momentum TopK", linewidth=1.3)
    axes[0].plot(x, nav["reversal_topk_nav"], label="Morning reversal TopK", linewidth=1.3)
    axes[0].plot(x, nav["pool_equal_nav"], label="Eligible pool equal", linewidth=1.1)
    axes[0].set_title(title)
    axes[0].set_ylabel("NAV")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    rolling = daily[
        ["cnn_return", "momentum_topk_return", "reversal_topk_return", "pool_equal_return"]
    ].rolling(20).mean()
    axes[1].plot(x, rolling["cnn_return"], label=f"{strategy_label} 20D mean")
    axes[1].plot(x, rolling["momentum_topk_return"], label="Momentum 20D mean")
    axes[1].plot(x, rolling["reversal_topk_return"], label="Reversal 20D mean")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("20D mean return")
    axes[1].grid(alpha=0.25)
    if len(nav):
        ticks = np.linspace(0, len(nav) - 1, min(8, len(nav)), dtype=int)
        labels = pd.to_datetime(nav.iloc[ticks]["trade_date"]).dt.strftime("%Y-%m-%d")
        axes[1].set_xticks(ticks, labels, rotation=25, ha="right")
    axes[1].legend()
    fig.tight_layout()
    path = out_dir / "backtest_report.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_sample_images(
    *,
    out_dir: Path,
    data: ImageDatasetData,
    indices: np.ndarray,
    image_height: int,
    price_limit: float,
    count: int,
) -> pd.DataFrame:
    sample_dir = out_dir / "sample_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for idx in indices[: max(0, int(count))]:
        tensor = torch.from_numpy(data.ohlc[int(idx)]).unsqueeze(0)
        image = render_candlestick_batch(
            tensor,
            image_height=image_height,
            price_limit=price_limit,
        )[0].numpy()
        rgb = np.full((image_height, image.shape[2], 3), 255, dtype=np.uint8)
        wick = image[0] > 0
        up = image[1] > 0
        down = image[2] > 0
        rgb[wick] = np.array([90, 90, 90], dtype=np.uint8)
        rgb[up] = np.array([220, 35, 35], dtype=np.uint8)
        rgb[down] = np.array([0, 145, 80], dtype=np.uint8)
        day = data.trade_date[int(idx)]
        code = str(data.code[int(idx)])
        filename = f"{day}_{code.replace('.', '_')}.png"
        path = sample_dir / filename
        Image.fromarray(rgb).resize((image.shape[2] * 4, image_height * 4), Image.Resampling.NEAREST).save(path)
        rows.append(
            {
                "trade_date": day,
                "code": code,
                "path": str(path),
                "morning_return": float(data.morning_return[int(idx)]),
                "return_net": float(data.raw_return[int(idx)]) if np.isfinite(data.raw_return[int(idx)]) else None,
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
        raise ValueError("kline image model config path is required")
    cfg = dict(load_json_like(config_path))
    paths_cfg = load_config_file("paths")
    split_cfg = dict(cfg.get("split", {}))
    data_cfg = dict(cfg.get("data", {}))
    train_cfg = dict(cfg.get("train", {}))
    model_cfg = dict(cfg.get("model_params", {}))
    execution_cfg = dict(execution or {})

    train_start = parse_date(split_cfg.get("train_start"))
    train_end = parse_date(split_cfg.get("train_end"))
    valid_start = parse_date(split_cfg.get("valid_start"))
    valid_end = parse_date(split_cfg.get("valid_end"))
    test_start = parse_date(start or split_cfg.get("test_start"))
    test_end = parse_date(end or split_cfg.get("test_end"))
    if not (train_start <= train_end < valid_start <= valid_end < test_start <= test_end):
        raise ValueError(
            "kline image split must satisfy train <= valid < test without overlap: "
            f"train={train_start}..{train_end} valid={valid_start}..{valid_end} "
            f"test={test_start}..{test_end}"
        )

    kline_root = _platform_path(data_cfg.get("kline_root"))
    if not kline_root.exists():
        raise FileNotFoundError(f"kline root missing: {kline_root}")
    files = _files_for_range(kline_root, train_start, test_end)
    if not files:
        raise FileNotFoundError(f"no kline files under {kline_root} for {train_start}..{test_end}")

    allowlist_cfg = dict(data_cfg.get("allowlist", {}))
    pool_cfg = load_upstream_pool_config(allowlist_cfg)
    buy_cost_bps, sell_cost_bps, fee_source = load_fees_buy_sell_bps()
    workers = int(execution_cfg.get("prep_workers", train_cfg.get("prep_workers", 8)))
    print(
        "kline image experiment:",
        f"files={len(files)}",
        f"range={files[0][0]}..{files[-1][0]}",
        f"train={train_start}..{train_end}",
        f"valid={valid_start}..{valid_end}",
        f"test={test_start}..{test_end}",
        f"fees={buy_cost_bps:.2f}+{sell_cost_bps:.2f}bps source={fee_source}",
        flush=True,
    )
    data = _load_image_data(
        files=files,
        raw_data_root=str(paths_cfg["raw_data_root"]),
        pool_cfg=pool_cfg,
        data_cfg=data_cfg,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        workers=workers,
    )
    if data.ohlc.size == 0:
        raise RuntimeError("kline image preparation produced no samples")

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
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
    )
    device = _device_from_config(train_cfg)
    predictions = _predict(
        model=model,
        data=data,
        indices=test_indices,
        targets=targets,
        batch_size=int(train_cfg.get("batch_size", 256)),
        device=device,
        image_height=int(data_cfg.get("image_height", 64)),
        price_limit=float(data_cfg.get("price_limit", 0.10)),
    )

    model_name = str(cfg.get("model_name", "kline_image_cnn_morning_v1"))
    results_root = resolve_output_path(
        cfg.get("results_root"),
        default_path=Path(paths_cfg["results_root"]) / "analysis" / model_name,
        results_root=paths_cfg["results_root"],
    )
    out_dir = results_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    score_output = resolve_output_path(
        cfg.get("score_output"),
        default_path=Path(paths_cfg["results_root"]) / "scores" / "kline_image" / model_name,
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
            {"strategy": "kline_image_cnn", **_portfolio_metrics(daily["cnn_return"])},
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
    plot_path = _write_plot(out_dir, daily, nav)
    sample_manifest = _write_sample_images(
        out_dir=out_dir,
        data=data,
        indices=test_indices,
        image_height=int(data_cfg.get("image_height", 64)),
        price_limit=float(data_cfg.get("price_limit", 0.10)),
        count=int(cfg.get("sample_image_count", 8)),
    )
    sample_manifest.to_csv(out_dir / "sample_images.csv", index=False)
    torch.save(model.state_dict(), out_dir / "model.pt")

    price_limit = float(data_cfg.get("price_limit", 0.10))
    summary = {
        "model_name": model_name,
        "model_type": "kline_image_cnn",
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
        "image": {
            "source_mode": str(data_cfg.get("source_mode", "minute_kline")),
            "height": int(data_cfg.get("image_height", 64)),
            "width": int(data.ohlc.shape[1]),
            "channels": ["wick", "up_body", "down_body"],
            "anchor": "09:30 prev_close_price",
            "fixed_log_return_limit": price_limit,
            "clip_fraction": float((np.abs(data.ohlc) > price_limit).mean()),
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
    print(f"saved kline image experiment: {out_dir}", flush=True)
