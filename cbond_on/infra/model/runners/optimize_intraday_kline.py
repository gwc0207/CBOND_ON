from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import json
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cbond_on.common.config_utils import load_json_like
from cbond_on.core.config import parse_date
from cbond_on.infra.model.impl.torch_image import KlineImageCNN
from cbond_on.infra.model.impl.torch_sequence import build_intraday_sequence_model
from cbond_on.infra.model.runners.train_kline_image import (
    ImageDatasetData,
    _daily_zscore_targets,
    _evaluate_topk,
    _indices_between,
    _portfolio_metrics,
    render_candlestick_batch,
)


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date


class _IntradayDataset(Dataset):
    def __init__(
        self,
        data: ImageDatasetData,
        indices: np.ndarray,
        z_targets: np.ndarray,
        rank_targets: np.ndarray,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.z_targets = z_targets
        self.rank_targets = rank_targets

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return (
            torch.from_numpy(self.data.ohlc[idx]).float(),
            torch.tensor(float(self.z_targets[idx]), dtype=torch.float32),
            torch.tensor(float(self.rank_targets[idx]), dtype=torch.float32),
        )


class _DayBatchSampler(Sampler[list[int]]):
    def __init__(self, days: np.ndarray, *, shuffle: bool, seed: int) -> None:
        frame = pd.DataFrame({"position": np.arange(len(days), dtype=np.int64), "day": days})
        self.groups = [
            group["position"].to_numpy(dtype=np.int64).tolist()
            for _, group in frame.groupby("day", sort=True)
        ]
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.iteration = 0

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.Random(self.seed + self.iteration).shuffle(order)
        self.iteration += 1
        for idx in order:
            yield self.groups[idx]

    def __len__(self) -> int:
        return len(self.groups)


def _platform_path(value: object) -> Path:
    if isinstance(value, dict):
        keys = ("windows", "win") if sys.platform.startswith("win") else ("linux", "unix", "posix")
        normalized = {str(key).strip().lower(): item for key, item in value.items()}
        for key in (*keys, "default", "common"):
            picked = normalized.get(key)
            if picked not in (None, ""):
                return Path(str(picked)).expanduser()
        raise ValueError("platform path has no usable value")
    return Path(str(value)).expanduser()


def _load_cached_data(path: Path) -> ImageDatasetData:
    if not path.exists():
        raise FileNotFoundError(f"intraday kline cache missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        date_values = np.asarray(saved["trade_date"]).astype(str)
        unique_dates, inverse = np.unique(date_values, return_inverse=True)
        parsed_dates = np.asarray([date.fromisoformat(item) for item in unique_dates], dtype=object)
        return ImageDatasetData(
            ohlc=np.asarray(saved["ohlc"]),
            trade_date=parsed_dates[inverse],
            code=np.asarray(saved["code"]).astype(str),
            raw_return=np.asarray(saved["raw_return"], dtype=np.float32),
            morning_return=np.asarray(saved["morning_return"], dtype=np.float32),
            buy_price=np.asarray(saved["buy_price"], dtype=np.float32),
            sell_price=np.asarray(saved["sell_price"], dtype=np.float32),
            sell_fallback=np.asarray(saved["sell_fallback"], dtype=bool),
            morning_coverage=np.asarray(saved["morning_coverage"], dtype=np.int16),
        )


def _daily_rank_targets(data: ImageDatasetData) -> np.ndarray:
    targets = np.full(len(data.raw_return), np.nan, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "idx": np.arange(len(data.raw_return), dtype=np.int64),
            "trade_date": data.trade_date,
            "raw_return": data.raw_return,
        }
    )
    for _, group in frame.groupby("trade_date", sort=False):
        values = pd.to_numeric(group["raw_return"], errors="coerce")
        finite = values.notna()
        if not finite.any():
            continue
        ranks = values[finite].rank(method="average", pct=True).to_numpy(dtype=np.float64)
        ranks = ranks * 2.0 - 1.0
        targets[group.loc[finite, "idx"].to_numpy(dtype=np.int64)] = ranks.astype(np.float32)
    return targets


def build_intraday_feature_batch(ohlc: torch.Tensor, feature_cfg: dict) -> torch.Tensor:
    if ohlc.ndim != 3 or ohlc.shape[-1] != 4:
        raise ValueError(f"ohlc must have shape (batch, time, 4), got {tuple(ohlc.shape)}")
    level_scale = abs(float(feature_cfg.get("level_scale", 0.04)))
    change_scale = abs(float(feature_cfg.get("change_scale", 0.002)))
    clip = abs(float(feature_cfg.get("clip", 8.0)))
    if level_scale <= 0 or change_scale <= 0:
        raise ValueError("feature level_scale and change_scale must be positive")
    open_price, high, low, close = [ohlc[:, :, idx] for idx in range(4)]
    previous_close = torch.cat([torch.zeros_like(close[:, :1]), close[:, :-1]], dim=1)
    close_return = close - previous_close
    body = close - open_price
    first_return_mode = str(feature_cfg.get("first_return_mode", "gap")).strip().lower()
    if first_return_mode == "body":
        close_return = close_return.clone()
        close_return[:, 0] = body[:, 0]
    elif first_return_mode == "zero":
        close_return = close_return.clone()
        close_return[:, 0] = 0.0
    elif first_return_mode != "gap":
        raise ValueError(f"unsupported first_return_mode: {first_return_mode}")
    candle_range = torch.clamp(high - low, min=0.0)
    upper_wick = torch.clamp(high - torch.maximum(open_price, close), min=0.0)
    lower_wick = torch.clamp(torch.minimum(open_price, close) - low, min=0.0)
    close_location = 2.0 * (close - low) / candle_range.clamp_min(1e-6) - 1.0
    parts: list[torch.Tensor] = []
    if bool(feature_cfg.get("include_levels", True)):
        parts.extend([open_price / level_scale, high / level_scale, low / level_scale, close / level_scale])
    if bool(feature_cfg.get("include_shape", True)):
        parts.extend(
            [
                close_return / change_scale,
                body / change_scale,
                candle_range / change_scale,
                upper_wick / change_scale,
                lower_wick / change_scale,
                close_location,
            ]
        )
    if not parts:
        raise ValueError("intraday feature configuration produced no channels")
    features = torch.stack(parts, dim=2)
    return torch.clamp(features, min=-clip, max=clip) if clip > 0 else features


def _build_model(candidate: dict, sequence_length: int, n_features: int) -> nn.Module:
    architecture = str(candidate.get("architecture", "inception")).strip().lower()
    model_params = dict(candidate.get("model_params", {}))
    if architecture == "image_cnn":
        return KlineImageCNN(
            in_channels=3,
            base_channels=int(model_params.get("base_channels", 16)),
            dropout=float(model_params.get("dropout", 0.15)),
        )
    return build_intraday_sequence_model(
        architecture,
        n_features=n_features,
        sequence_length=sequence_length,
        model_params=model_params,
    )


def _forward_model(model: nn.Module, ohlc: torch.Tensor, candidate: dict) -> torch.Tensor:
    architecture = str(candidate.get("architecture", "inception")).strip().lower()
    with torch.no_grad():
        if architecture == "image_cnn":
            image_cfg = dict(candidate.get("image", {}))
            inputs = render_candlestick_batch(
                ohlc,
                image_height=int(image_cfg.get("height", 96)),
                price_limit=float(image_cfg.get("price_limit", 0.04)),
            )
        else:
            inputs = build_intraday_feature_batch(ohlc, dict(candidate.get("features", {})))
    return model(inputs)


def _loss_requires_day_batches(loss_name: str) -> bool:
    return str(loss_name).strip().lower() in {"listnet", "hybrid_listnet", "tail_pairwise"}


def _compute_loss(
    prediction: torch.Tensor,
    z_target: torch.Tensor,
    rank_target: torch.Tensor,
    *,
    loss_cfg: dict,
) -> torch.Tensor:
    name = str(loss_cfg.get("name", "huber_z")).strip().lower()
    beta = float(loss_cfg.get("huber_beta", 0.5))
    if name == "huber_z":
        return F.smooth_l1_loss(prediction, z_target, beta=beta)
    if name == "huber_rank":
        return F.smooth_l1_loss(prediction, rank_target, beta=beta)
    if name in {"listnet", "hybrid_listnet"}:
        target_temperature = max(1e-3, float(loss_cfg.get("target_temperature", 0.75)))
        prediction_temperature = max(1e-3, float(loss_cfg.get("prediction_temperature", 1.0)))
        target_distribution = torch.softmax(torch.clamp(z_target, -3.0, 3.0) / target_temperature, dim=0)
        listnet = -(target_distribution * F.log_softmax(prediction / prediction_temperature, dim=0)).sum()
        if name == "listnet":
            return listnet
        point_weight = float(loss_cfg.get("point_weight", 0.2))
        return listnet + point_weight * F.smooth_l1_loss(prediction, rank_target, beta=beta)
    if name == "tail_pairwise":
        top_n = min(max(1, int(loss_cfg.get("top_n", 20))), len(prediction) // 2)
        order = torch.argsort(rank_target)
        top_scores = prediction[order[-top_n:]]
        comparison_pool = prediction[order[: max(top_n, len(prediction) // 2)]]
        pairwise = F.softplus(-(top_scores[:, None] - comparison_pool[None, :])).mean()
        point_weight = float(loss_cfg.get("point_weight", 0.2))
        return pairwise + point_weight * F.smooth_l1_loss(prediction, rank_target, beta=beta)
    raise ValueError(f"unsupported intraday optimization loss: {name}")


def _train_model(
    *,
    data: ImageDatasetData,
    train_indices: np.ndarray,
    z_targets: np.ndarray,
    rank_targets: np.ndarray,
    candidate: dict,
    default_train_cfg: dict,
    seed: int,
) -> tuple[nn.Module, pd.DataFrame]:
    train_cfg = {**default_train_cfg, **dict(candidate.get("train", {}))}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(train_cfg.get("torch_threads", 8))))
    device = torch.device("cpu")
    dummy = torch.from_numpy(data.ohlc[train_indices[:1]]).float()
    n_features = int(build_intraday_feature_batch(dummy, dict(candidate.get("features", {}))).shape[-1])
    model = _build_model(candidate, int(data.ohlc.shape[1]), n_features).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    dataset = _IntradayDataset(data, train_indices, z_targets, rank_targets)
    loss_cfg = dict(candidate.get("loss", {"name": "huber_z"}))
    loss_name = str(loss_cfg.get("name", "huber_z"))
    if _loss_requires_day_batches(loss_name):
        loader = DataLoader(
            dataset,
            batch_sampler=_DayBatchSampler(
                data.trade_date[train_indices],
                shuffle=True,
                seed=seed,
            ),
            num_workers=0,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(train_cfg.get("batch_size", 256))),
            shuffle=True,
            num_workers=0,
        )
    epochs = max(1, int(train_cfg.get("num_epochs", 3)))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for ohlc, z_target, rank_target in loader:
            ohlc = ohlc.to(device)
            z_target = z_target.to(device)
            rank_target = rank_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward_model(model, ohlc, candidate)
            loss = _compute_loss(
                prediction,
                z_target,
                rank_target,
                loss_cfg=loss_cfg,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
        }
        history.append(row)
        print(
            f"[intraday_opt:epoch] {candidate['name']} {epoch}/{epochs} "
            f"loss={row['train_loss']:.6f}",
            flush=True,
        )
    return model, pd.DataFrame(history)


def _predict(
    *,
    model: nn.Module,
    data: ImageDatasetData,
    indices: np.ndarray,
    z_targets: np.ndarray,
    rank_targets: np.ndarray,
    candidate: dict,
    default_train_cfg: dict,
) -> np.ndarray:
    train_cfg = {**default_train_cfg, **dict(candidate.get("train", {}))}
    loader = DataLoader(
        _IntradayDataset(data, indices, z_targets, rank_targets),
        batch_size=max(1, int(train_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=0,
    )
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for ohlc, _, _ in loader:
            outputs.append(_forward_model(model, ohlc, candidate).detach().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False) if outputs else np.array([], dtype=np.float32)


def _parse_folds(rows: list[dict]) -> list[WalkForwardFold]:
    folds = [
        WalkForwardFold(
            name=str(row["name"]),
            train_start=parse_date(row["train_start"]),
            train_end=parse_date(row["train_end"]),
            test_start=parse_date(row["test_start"]),
            test_end=parse_date(row["test_end"]),
        )
        for row in rows
    ]
    for fold in folds:
        if not fold.train_start <= fold.train_end < fold.test_start <= fold.test_end:
            raise ValueError(f"invalid walk-forward fold: {fold}")
    return folds


def _excess_metrics(daily: pd.DataFrame) -> dict:
    return _portfolio_metrics(daily["cnn_return"] - daily["pool_equal_return"])


def _run_candidate(
    *,
    data: ImageDatasetData,
    z_targets: np.ndarray,
    rank_targets: np.ndarray,
    candidate: dict,
    folds: list[WalkForwardFold],
    train_cfg: dict,
    top_k: int,
    out_dir: Path,
) -> dict:
    candidate_dir = out_dir / str(candidate["name"])
    candidate_dir.mkdir(parents=True, exist_ok=True)
    daily_parts: list[pd.DataFrame] = []
    position_parts: list[pd.DataFrame] = []
    ic_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    history_parts: list[pd.DataFrame] = []
    candidate_train_cfg = {**train_cfg, **dict(candidate.get("train", {}))}
    base_seed = int(candidate_train_cfg.get("seed", 42))
    for fold_index, fold in enumerate(folds):
        train_indices = _indices_between(data.trade_date, fold.train_start, fold.train_end)
        train_indices = train_indices[np.isfinite(z_targets[train_indices])]
        test_indices = _indices_between(data.trade_date, fold.test_start, fold.test_end)
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise RuntimeError(
                f"empty intraday optimization fold {fold.name}: "
                f"train={len(train_indices)} test={len(test_indices)}"
            )
        print(
            f"[intraday_opt:fold] candidate={candidate['name']} fold={fold.name} "
            f"train={len(train_indices)} test={len(test_indices)}",
            flush=True,
        )
        model, history = _train_model(
            data=data,
            train_indices=train_indices,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=candidate,
            default_train_cfg=train_cfg,
            seed=base_seed + fold_index,
        )
        predictions = _predict(
            model=model,
            data=data,
            indices=test_indices,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=candidate,
            default_train_cfg=train_cfg,
        )
        score_parts.append(
            pd.DataFrame(
                {
                    "fold": fold.name,
                    "trade_date": data.trade_date[test_indices],
                    "code": data.code[test_indices],
                    "score": predictions,
                }
            )
        )
        daily, _, positions, ic = _evaluate_topk(
            data=data,
            test_indices=test_indices,
            predictions=predictions,
            top_k=top_k,
        )
        daily.insert(0, "fold", fold.name)
        positions.insert(0, "fold", fold.name)
        ic.insert(0, "fold", fold.name)
        history.insert(0, "fold", fold.name)
        daily_parts.append(daily)
        position_parts.append(positions)
        ic_parts.append(ic)
        history_parts.append(history)
        model_metrics = _portfolio_metrics(daily["cnn_return"])
        pool_metrics = _portfolio_metrics(daily["pool_equal_return"])
        excess_metrics = _excess_metrics(daily)
        fold_rows.append(
            {
                "fold": fold.name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "train_samples": int(len(train_indices)),
                "test_samples": int(len(test_indices)),
                "model_return": model_metrics["period_return"],
                "model_sharpe": model_metrics["sharpe"],
                "pool_return": pool_metrics["period_return"],
                "pool_sharpe": pool_metrics["sharpe"],
                "excess_return": model_metrics["period_return"] - pool_metrics["period_return"],
                "excess_sharpe": excess_metrics["sharpe"],
                "rank_ic": float(pd.to_numeric(ic["rank_ic"], errors="coerce").mean()),
            }
        )
        del model
    daily_all = pd.concat(daily_parts, ignore_index=True).sort_values("trade_date")
    positions_all = pd.concat(position_parts, ignore_index=True)
    ic_all = pd.concat(ic_parts, ignore_index=True)
    scores_all = pd.concat(score_parts, ignore_index=True)
    histories = pd.concat(history_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    model_metrics = _portfolio_metrics(daily_all["cnn_return"])
    pool_metrics = _portfolio_metrics(daily_all["pool_equal_return"])
    excess_metrics = _excess_metrics(daily_all)
    positive_fold_rate = float((fold_metrics["excess_return"] > 0).mean())
    median_fold_excess_sharpe = float(pd.to_numeric(fold_metrics["excess_sharpe"], errors="coerce").median())
    aggregate_excess_sharpe = float(excess_metrics["sharpe"])
    robust_score = (
        0.5 * aggregate_excess_sharpe
        + 0.3 * median_fold_excess_sharpe
        + 0.2 * (2.0 * positive_fold_rate - 1.0)
    )
    summary = {
        "candidate": candidate,
        "top_k": top_k,
        "development_folds": [fold.name for fold in folds],
        "model_metrics": model_metrics,
        "pool_metrics": pool_metrics,
        "excess_metrics": excess_metrics,
        "rank_ic_mean": float(pd.to_numeric(ic_all["rank_ic"], errors="coerce").mean()),
        "positive_excess_fold_rate": positive_fold_rate,
        "median_fold_excess_sharpe": median_fold_excess_sharpe,
        "worst_fold_excess_return": float(fold_metrics["excess_return"].min()),
        "robust_score": robust_score,
        "result_dir": str(candidate_dir),
    }
    daily_all.to_csv(candidate_dir / "daily_returns.csv", index=False)
    positions_all.to_csv(candidate_dir / "positions.csv", index=False)
    ic_all.to_csv(candidate_dir / "score_eval_daily.csv", index=False)
    scores_all.to_csv(candidate_dir / "scores.csv", index=False)
    histories.to_csv(candidate_dir / "train_history.csv", index=False)
    fold_metrics.to_csv(candidate_dir / "fold_metrics.csv", index=False)
    (candidate_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _load_leaderboard(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in root.glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "candidate": summary["candidate"]["name"],
                "architecture": summary["candidate"]["architecture"],
                "loss": summary["candidate"].get("loss", {}).get("name", "huber_z"),
                "period_return": summary["model_metrics"]["period_return"],
                "sharpe": summary["model_metrics"]["sharpe"],
                "max_drawdown": summary["model_metrics"]["max_drawdown"],
                "pool_return": summary["pool_metrics"]["period_return"],
                "excess_return": summary["model_metrics"]["period_return"]
                - summary["pool_metrics"]["period_return"],
                "excess_sharpe": summary["excess_metrics"]["sharpe"],
                "rank_ic_mean": summary["rank_ic_mean"],
                "positive_excess_fold_rate": summary["positive_excess_fold_rate"],
                "worst_fold_excess_return": summary["worst_fold_excess_return"],
                "robust_score": summary["robust_score"],
                "result_dir": summary["result_dir"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["robust_score", "excess_sharpe"], ascending=False)


def _summarize_ensemble_daily(
    daily: pd.DataFrame,
    ic: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    fold_rows: list[dict] = []
    for fold, group in daily.groupby("fold", sort=False):
        model_metrics = _portfolio_metrics(group["cnn_return"])
        pool_metrics = _portfolio_metrics(group["pool_equal_return"])
        excess_metrics = _excess_metrics(group)
        fold_ic = ic.loc[ic["fold"] == fold, "rank_ic"]
        fold_rows.append(
            {
                "fold": fold,
                "model_return": model_metrics["period_return"],
                "model_sharpe": model_metrics["sharpe"],
                "pool_return": pool_metrics["period_return"],
                "pool_sharpe": pool_metrics["sharpe"],
                "excess_return": model_metrics["period_return"] - pool_metrics["period_return"],
                "excess_sharpe": excess_metrics["sharpe"],
                "rank_ic": float(pd.to_numeric(fold_ic, errors="coerce").mean()),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    model_metrics = _portfolio_metrics(daily["cnn_return"])
    pool_metrics = _portfolio_metrics(daily["pool_equal_return"])
    excess_metrics = _excess_metrics(daily)
    positive_fold_rate = float((fold_metrics["excess_return"] > 0).mean())
    median_fold_excess_sharpe = float(pd.to_numeric(fold_metrics["excess_sharpe"], errors="coerce").median())
    robust_score = (
        0.5 * float(excess_metrics["sharpe"])
        + 0.3 * median_fold_excess_sharpe
        + 0.2 * (2.0 * positive_fold_rate - 1.0)
    )
    return (
        {
            "model_metrics": model_metrics,
            "pool_metrics": pool_metrics,
            "excess_metrics": excess_metrics,
            "rank_ic_mean": float(pd.to_numeric(ic["rank_ic"], errors="coerce").mean()),
            "positive_excess_fold_rate": positive_fold_rate,
            "median_fold_excess_sharpe": median_fold_excess_sharpe,
            "worst_fold_excess_return": float(fold_metrics["excess_return"].min()),
            "robust_score": robust_score,
        },
        fold_metrics,
    )


def _run_development_ensemble(
    *,
    data: ImageDatasetData,
    folds: list[WalkForwardFold],
    members: list[str],
    ensemble_name: str,
    top_k: int,
    output_root: Path,
) -> dict:
    if not members:
        raise ValueError("development ensemble requires members")
    ensemble_dir = output_root / "ensembles" / ensemble_name
    if ensemble_dir.exists():
        raise RuntimeError(f"development ensemble already exists: {ensemble_dir}")
    ensemble_dir.mkdir(parents=True, exist_ok=False)
    fold_by_day: dict[date, str] = {}
    test_indices_parts: list[np.ndarray] = []
    for fold in folds:
        indices = _indices_between(data.trade_date, fold.test_start, fold.test_end)
        test_indices_parts.append(indices)
        for day in np.unique(data.trade_date[indices]):
            fold_by_day[day] = fold.name
    test_indices = np.concatenate(test_indices_parts)
    merged = pd.DataFrame(
        {
            "global_index": test_indices,
            "trade_date": data.trade_date[test_indices],
            "code": data.code[test_indices],
        }
    )
    merged["fold"] = merged["trade_date"].map(fold_by_day)
    rank_columns: list[str] = []
    for member_index, member in enumerate(members):
        score_path = output_root / member / "scores.csv"
        if not score_path.exists():
            raise FileNotFoundError(f"ensemble member scores missing: {score_path}")
        scores = pd.read_csv(score_path)
        scores["trade_date"] = pd.to_datetime(scores["trade_date"]).dt.date
        score_col = f"score_{member_index}"
        rank_col = f"rank_{member_index}"
        scores = scores[["trade_date", "code", "score"]].rename(columns={"score": score_col})
        merged = merged.merge(scores, on=["trade_date", "code"], how="left", validate="one_to_one")
        if merged[score_col].isna().any():
            raise RuntimeError(f"ensemble member has missing scores: {member}")
        merged[rank_col] = merged.groupby("trade_date")[score_col].rank(method="average", pct=True)
        rank_columns.append(rank_col)
    merged["ensemble_score"] = merged[rank_columns].mean(axis=1)
    daily, nav, positions, ic = _evaluate_topk(
        data=data,
        test_indices=merged["global_index"].to_numpy(dtype=np.int64),
        predictions=merged["ensemble_score"].to_numpy(dtype=np.float32),
        top_k=top_k,
    )
    daily.insert(0, "fold", daily["trade_date"].map(fold_by_day))
    positions.insert(0, "fold", positions["trade_date"].map(fold_by_day))
    ic.insert(0, "fold", ic["trade_date"].map(fold_by_day))
    metrics, fold_metrics = _summarize_ensemble_daily(daily, ic)
    summary = {
        "ensemble_name": ensemble_name,
        "members": members,
        **metrics,
        "result_dir": str(ensemble_dir),
    }
    merged.to_csv(ensemble_dir / "scores.csv", index=False)
    daily.to_csv(ensemble_dir / "daily_returns.csv", index=False)
    nav.to_csv(ensemble_dir / "nav_curve.csv", index=False)
    positions.to_csv(ensemble_dir / "positions.csv", index=False)
    ic.to_csv(ensemble_dir / "score_eval_daily.csv", index=False)
    fold_metrics.to_csv(ensemble_dir / "fold_metrics.csv", index=False)
    (ensemble_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _run_final(
    *,
    data: ImageDatasetData,
    z_targets: np.ndarray,
    rank_targets: np.ndarray,
    candidate: dict,
    final_fold: WalkForwardFold,
    train_cfg: dict,
    top_k: int,
    output_root: Path,
) -> dict:
    final_root = output_root.parent / "final_holdout" if output_root.name == "development" else output_root / "final_holdout"
    final_dir = final_root / str(candidate["name"])
    if final_dir.exists():
        raise RuntimeError(f"final holdout already evaluated for candidate: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=False)
    train_indices = _indices_between(data.trade_date, final_fold.train_start, final_fold.train_end)
    train_indices = train_indices[np.isfinite(z_targets[train_indices])]
    test_indices = _indices_between(data.trade_date, final_fold.test_start, final_fold.test_end)
    model, history = _train_model(
        data=data,
        train_indices=train_indices,
        z_targets=z_targets,
        rank_targets=rank_targets,
        candidate=candidate,
        default_train_cfg=train_cfg,
        seed=int({**train_cfg, **dict(candidate.get("train", {}))}.get("seed", 42)),
    )
    predictions = _predict(
        model=model,
        data=data,
        indices=test_indices,
        z_targets=z_targets,
        rank_targets=rank_targets,
        candidate=candidate,
        default_train_cfg=train_cfg,
    )
    daily, nav, positions, ic = _evaluate_topk(
        data=data,
        test_indices=test_indices,
        predictions=predictions,
        top_k=top_k,
    )
    model_metrics = _portfolio_metrics(daily["cnn_return"])
    pool_metrics = _portfolio_metrics(daily["pool_equal_return"])
    summary = {
        "candidate": candidate,
        "fold": {
            "train_start": final_fold.train_start,
            "train_end": final_fold.train_end,
            "test_start": final_fold.test_start,
            "test_end": final_fold.test_end,
        },
        "model_metrics": model_metrics,
        "pool_metrics": pool_metrics,
        "excess_metrics": _excess_metrics(daily),
        "rank_ic_mean": float(pd.to_numeric(ic["rank_ic"], errors="coerce").mean()),
        "result_dir": str(final_dir),
    }
    history.to_csv(final_dir / "train_history.csv", index=False)
    daily.to_csv(final_dir / "daily_returns.csv", index=False)
    nav.to_csv(final_dir / "nav_curve.csv", index=False)
    positions.to_csv(final_dir / "positions.csv", index=False)
    ic.to_csv(final_dir / "score_eval_daily.csv", index=False)
    torch.save(model.state_dict(), final_dir / "model.pt")
    (final_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def _run_final_ensemble(
    *,
    data: ImageDatasetData,
    z_targets: np.ndarray,
    rank_targets: np.ndarray,
    candidates: list[dict],
    ensemble_name: str,
    final_fold: WalkForwardFold,
    train_cfg: dict,
    top_k: int,
    output_root: Path,
) -> dict:
    final_root = output_root.parent / "final_holdout" if output_root.name == "development" else output_root / "final_holdout"
    final_dir = final_root / ensemble_name
    if final_dir.exists():
        raise RuntimeError(f"final holdout ensemble already evaluated: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=False)
    state_dir = final_dir / "model_states"
    state_dir.mkdir(parents=True, exist_ok=False)
    train_indices = _indices_between(data.trade_date, final_fold.train_start, final_fold.train_end)
    train_indices = train_indices[np.isfinite(z_targets[train_indices])]
    test_indices = _indices_between(data.trade_date, final_fold.test_start, final_fold.test_end)
    scores = pd.DataFrame(
        {
            "global_index": test_indices,
            "trade_date": data.trade_date[test_indices],
            "code": data.code[test_indices],
        }
    )
    rank_columns: list[str] = []
    histories: list[pd.DataFrame] = []
    for member_index, candidate in enumerate(candidates):
        candidate_train_cfg = {**train_cfg, **dict(candidate.get("train", {}))}
        model, history = _train_model(
            data=data,
            train_indices=train_indices,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=candidate,
            default_train_cfg=train_cfg,
            seed=int(candidate_train_cfg.get("seed", 42)),
        )
        predictions = _predict(
            model=model,
            data=data,
            indices=test_indices,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=candidate,
            default_train_cfg=train_cfg,
        )
        score_col = f"score_{member_index}"
        rank_col = f"rank_{member_index}"
        scores[score_col] = predictions
        scores[rank_col] = scores.groupby("trade_date")[score_col].rank(method="average", pct=True)
        rank_columns.append(rank_col)
        history.insert(0, "member", candidate["name"])
        histories.append(history)
        torch.save(model.state_dict(), state_dir / f"{candidate['name']}.pt")
        del model
    scores["ensemble_score"] = scores[rank_columns].mean(axis=1)
    daily, nav, positions, ic = _evaluate_topk(
        data=data,
        test_indices=test_indices,
        predictions=scores["ensemble_score"].to_numpy(dtype=np.float32),
        top_k=top_k,
    )
    model_metrics = _portfolio_metrics(daily["cnn_return"])
    pool_metrics = _portfolio_metrics(daily["pool_equal_return"])
    summary = {
        "ensemble_name": ensemble_name,
        "members": [candidate["name"] for candidate in candidates],
        "top_k": top_k,
        "fold": {
            "train_start": final_fold.train_start,
            "train_end": final_fold.train_end,
            "test_start": final_fold.test_start,
            "test_end": final_fold.test_end,
        },
        "model_metrics": model_metrics,
        "pool_metrics": pool_metrics,
        "excess_metrics": _excess_metrics(daily),
        "rank_ic_mean": float(pd.to_numeric(ic["rank_ic"], errors="coerce").mean()),
        "result_dir": str(final_dir),
    }
    pd.concat(histories, ignore_index=True).to_csv(final_dir / "train_history.csv", index=False)
    scores.to_csv(final_dir / "scores.csv", index=False)
    daily.to_csv(final_dir / "daily_returns.csv", index=False)
    nav.to_csv(final_dir / "nav_curve.csv", index=False)
    positions.to_csv(final_dir / "positions.csv", index=False)
    ic.to_csv(final_dir / "score_eval_daily.csv", index=False)
    (final_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary


def main(
    *,
    config_path: Path,
    candidate_names: list[str] | None = None,
    stage: str = "development",
    final_candidate_name: str | None = None,
    ensemble_name: str | None = None,
    ensemble_members: list[str] | None = None,
) -> None:
    cfg = dict(load_json_like(config_path))
    cache_path = _platform_path(cfg.get("data_cache"))
    output_root = _platform_path(cfg.get("results_root"))
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"intraday optimization cache load: {cache_path}", flush=True)
    data = _load_cached_data(cache_path)
    z_targets = _daily_zscore_targets(data)
    rank_targets = _daily_rank_targets(data)
    train_cfg = dict(cfg.get("train", {}))
    top_k = int(cfg.get("top_k", 20))
    candidates = [dict(row) for row in cfg.get("candidates", [])]
    by_name = {str(candidate["name"]): candidate for candidate in candidates}
    if stage in {"final", "final_ensemble"}:
        final_cfg = dict(cfg.get("final_holdout", {}))
        final_fold = _parse_folds(
            [
                {
                    "name": "final_holdout",
                    "train_start": final_cfg["train_start"],
                    "train_end": final_cfg["train_end"],
                    "test_start": final_cfg["test_start"],
                    "test_end": final_cfg["test_end"],
                }
            ]
        )[0]
    if stage == "final":
        if not final_candidate_name or final_candidate_name not in by_name:
            raise ValueError("final stage requires a configured final_candidate_name")
        summary = _run_final(
            data=data,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=by_name[final_candidate_name],
            final_fold=final_fold,
            train_cfg=train_cfg,
            top_k=int(final_cfg.get("top_k", top_k)),
            output_root=output_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
        return
    if stage == "final_ensemble":
        if not ensemble_name or not ensemble_members:
            raise ValueError("final_ensemble stage requires ensemble_name and ensemble_members")
        unknown = [member for member in ensemble_members if member not in by_name]
        if unknown:
            raise ValueError(f"unknown final ensemble members: {unknown}")
        summary = _run_final_ensemble(
            data=data,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidates=[by_name[member] for member in ensemble_members],
            ensemble_name=ensemble_name,
            final_fold=final_fold,
            train_cfg=train_cfg,
            top_k=int(final_cfg.get("top_k", top_k)),
            output_root=output_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
        return
    if stage == "development_ensemble":
        if not ensemble_name or not ensemble_members:
            raise ValueError("development_ensemble requires ensemble_name and ensemble_members")
        folds = _parse_folds([dict(row) for row in cfg.get("development_folds", [])])
        summary = _run_development_ensemble(
            data=data,
            folds=folds,
            members=ensemble_members,
            ensemble_name=ensemble_name,
            top_k=top_k,
            output_root=output_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
        return
    if stage != "development":
        raise ValueError(f"unsupported intraday optimization stage: {stage}")
    folds = _parse_folds([dict(row) for row in cfg.get("development_folds", [])])
    selected = candidate_names or list(by_name)
    for name in selected:
        if name not in by_name:
            raise ValueError(f"unknown intraday optimization candidate: {name}")
        summary_path = output_root / name / "summary.json"
        if summary_path.exists():
            print(f"[intraday_opt:skip] existing candidate={name}", flush=True)
            continue
        summary = _run_candidate(
            data=data,
            z_targets=z_targets,
            rank_targets=rank_targets,
            candidate=by_name[name],
            folds=folds,
            train_cfg=train_cfg,
            top_k=top_k,
            out_dir=output_root,
        )
        print(
            f"[intraday_opt:done] candidate={name} "
            f"return={summary['model_metrics']['period_return']:.4%} "
            f"sharpe={summary['model_metrics']['sharpe']:.4f} "
            f"excess_sharpe={summary['excess_metrics']['sharpe']:.4f} "
            f"robust_score={summary['robust_score']:.4f}",
            flush=True,
        )
    leaderboard = _load_leaderboard(output_root)
    leaderboard.to_csv(output_root / "leaderboard.csv", index=False)
    print(leaderboard.to_string(index=False), flush=True)
