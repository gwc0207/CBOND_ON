from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cbond_on.core.config import resolve_output_path
from cbond_on.infra.live.model_switch import T1430_DISPERSION_FEATURE_SETS


@dataclass(frozen=True)
class SimilarDayTrainingConfig:
    state_feature_path: Path
    feature_set: str
    feature_cols: tuple[str, ...]
    candidate_lookback_days: int
    candidate_buffer_days: int
    train_top_k: int
    validation_top_k: int
    min_candidate_days: int
    selection_mode: str
    fallback: str
    kernel_target_effective_days: int | None = None
    kernel_weight_floor: float = 1e-12


@dataclass(frozen=True)
class SimilarDaySelection:
    target_day: date
    config: SimilarDayTrainingConfig
    candidate_days: int
    train_days: tuple[date, ...]
    validation_days: tuple[date, ...]
    selections: pd.DataFrame
    reason: str | None = None
    kernel_bandwidth: float | None = None
    kernel_realized_effective_days: float | None = None
    kernel_weight_status: str | None = None

    @property
    def ready(self) -> bool:
        return self.reason is None

    @property
    def uses_kernel_weights(self) -> bool:
        return self.ready and self.config.selection_mode == "kernel"

    def train_weights_by_day(self) -> dict[date, float]:
        if not self.uses_kernel_weights or "weight" not in self.selections.columns:
            return {}
        train = self.selections.loc[self.selections["role"] == "train", ["trade_date", "weight"]].copy()
        train["weight"] = pd.to_numeric(train["weight"], errors="coerce")
        train = train.dropna(subset=["trade_date", "weight"])
        return {
            pd.Timestamp(item.trade_date).date(): float(item.weight)
            for item in train.itertuples(index=False)
        }

    def summary(self) -> dict[str, object]:
        train = self.selections[self.selections["role"] == "train"]
        val = self.selections[self.selections["role"] == "validation"]
        summary: dict[str, object] = {
            "similarity_enabled": True,
            "similarity_mode": self.config.selection_mode,
            "similarity_candidate_days": int(self.candidate_days),
            "similarity_train_days": int(len(self.train_days)),
            "similarity_validation_days": int(len(self.validation_days)),
            "similarity_train_distance_min": float(train["distance"].min()) if not train.empty else float("nan"),
            "similarity_train_distance_max": float(train["distance"].max()) if not train.empty else float("nan"),
            "similarity_validation_distance_min": float(val["distance"].min()) if not val.empty else float("nan"),
            "similarity_validation_distance_max": float(val["distance"].max()) if not val.empty else float("nan"),
            "similarity_reason": self.reason or "ok",
        }
        if self.config.selection_mode == "kernel":
            weights = pd.to_numeric(train.get("weight"), errors="coerce").dropna()
            day_ess = _effective_sample_size(weights.to_numpy(dtype=float)) if not weights.empty else float("nan")
            summary.update(
                {
                    "similarity_kernel_target_effective_days": self.config.kernel_target_effective_days,
                    "similarity_kernel_realized_effective_days": day_ess,
                    "similarity_kernel_bandwidth": self.kernel_bandwidth,
                    "similarity_kernel_weight_min": float(weights.min()) if not weights.empty else float("nan"),
                    "similarity_kernel_weight_max": float(weights.max()) if not weights.empty else float("nan"),
                    "similarity_kernel_weight_status": self.kernel_weight_status or "not_ready",
                }
            )
        return summary

    def audit_rows(self) -> list[dict[str, object]]:
        if self.selections.empty:
            return [
                {
                    "target_day": self.target_day,
                    "role": "fallback",
                    "rank": None,
                    "selected_day": None,
                    "distance": None,
                    "candidate_days": int(self.candidate_days),
                    "feature_set": self.config.feature_set,
                    "feature_count": len(self.config.feature_cols),
                    "selection_mode": self.config.selection_mode,
                    "state_feature_path": str(self.config.state_feature_path),
                    "weight": None,
                    "kernel_bandwidth": self.kernel_bandwidth,
                    "kernel_target_effective_days": self.config.kernel_target_effective_days,
                    "kernel_realized_effective_days": self.kernel_realized_effective_days,
                    "kernel_weight_status": self.kernel_weight_status,
                    "reason": self.reason or "empty_selection",
                }
            ]
        rows: list[dict[str, object]] = []
        for item in self.selections.itertuples(index=False):
            rows.append(
                {
                    "target_day": self.target_day,
                    "role": str(item.role),
                    "rank": int(item.rank),
                    "selected_day": item.trade_date,
                    "distance": float(item.distance),
                    "candidate_days": int(self.candidate_days),
                    "feature_set": self.config.feature_set,
                    "feature_count": len(self.config.feature_cols),
                    "selection_mode": self.config.selection_mode,
                    "state_feature_path": str(self.config.state_feature_path),
                    "weight": float(item.weight) if hasattr(item, "weight") and pd.notna(item.weight) else None,
                    "kernel_bandwidth": self.kernel_bandwidth,
                    "kernel_target_effective_days": self.config.kernel_target_effective_days,
                    "kernel_realized_effective_days": self.kernel_realized_effective_days,
                    "kernel_weight_status": self.kernel_weight_status,
                    "reason": "ok",
                }
            )
        return rows


def resolve_similar_day_training_config(
    model_cfg: dict,
    *,
    results_root: str | Path,
) -> SimilarDayTrainingConfig | None:
    feature_cfg = model_cfg.get("feature_engineering", {})
    if feature_cfg is None:
        feature_cfg = {}
    if not isinstance(feature_cfg, dict):
        raise TypeError("feature_engineering must be an object")
    raw = feature_cfg.get("similar_day_training", model_cfg.get("similar_day_training", {}))
    if raw in (None, "", [], False):
        return None
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict):
        raise TypeError("feature_engineering.similar_day_training must be a bool or object")
    if not bool(raw.get("enabled", False)):
        return None

    feature_set = str(raw.get("feature_set", "path_full_t1430")).strip().lower()
    feature_cols = T1430_DISPERSION_FEATURE_SETS.get(feature_set)
    if not feature_cols:
        raise ValueError(f"unsupported similar_day_training.feature_set: {feature_set}")
    path_raw = raw.get("state_feature_path")
    if path_raw in (None, ""):
        raise ValueError("similar_day_training.state_feature_path is required")
    state_path = resolve_output_path(
        path_raw,
        default_path=Path(results_root) / "analysis" / "t1430_market_state_features.csv",
        results_root=results_root,
    )
    candidate_lookback_days = int(raw.get("candidate_lookback_days", 360))
    candidate_buffer_days = int(raw.get("candidate_buffer_days", max(10, candidate_lookback_days // 10)))
    train_top_k = int(raw.get("train_top_k", raw.get("nearest_k", 60)))
    validation_top_k = int(raw.get("validation_top_k", 20))
    min_candidate_days = int(raw.get("min_candidate_days", train_top_k + validation_top_k))
    selection_mode = str(raw.get("selection_mode", "nearest")).strip().lower()
    fallback = str(raw.get("fallback", "error")).strip().lower()
    kernel_target_effective_days = raw.get("kernel_target_effective_days")
    if selection_mode == "kernel" and kernel_target_effective_days is None:
        kernel_target_effective_days = train_top_k
    if kernel_target_effective_days is not None:
        kernel_target_effective_days = int(kernel_target_effective_days)
    kernel_weight_floor = float(raw.get("kernel_weight_floor", 1e-12))
    if candidate_lookback_days <= 0:
        raise ValueError("similar_day_training.candidate_lookback_days must be positive")
    if candidate_buffer_days < 0:
        raise ValueError("similar_day_training.candidate_buffer_days must be non-negative")
    if train_top_k <= 0 or validation_top_k <= 0:
        raise ValueError("similar_day_training.train_top_k and validation_top_k must be positive")
    if train_top_k + validation_top_k > candidate_lookback_days:
        raise ValueError("similar_day_training train_top_k + validation_top_k exceeds candidate lookback")
    if min_candidate_days < train_top_k + validation_top_k:
        raise ValueError("similar_day_training.min_candidate_days must cover train and validation samples")
    if selection_mode not in {"nearest", "latest", "kernel"}:
        raise ValueError("similar_day_training.selection_mode must be nearest, latest, or kernel")
    if fallback not in {"error", "rolling"}:
        raise ValueError("similar_day_training.fallback must be error or rolling")
    if selection_mode == "kernel":
        if kernel_target_effective_days is None or kernel_target_effective_days <= 0:
            raise ValueError("similar_day_training.kernel_target_effective_days must be positive for kernel mode")
        if kernel_target_effective_days > candidate_lookback_days - validation_top_k:
            raise ValueError(
                "similar_day_training.kernel_target_effective_days exceeds available kernel train days"
            )
    if not np.isfinite(kernel_weight_floor) or kernel_weight_floor <= 0 or kernel_weight_floor >= 1:
        raise ValueError("similar_day_training.kernel_weight_floor must be finite and in (0, 1)")
    return SimilarDayTrainingConfig(
        state_feature_path=state_path,
        feature_set=feature_set,
        feature_cols=tuple(feature_cols),
        candidate_lookback_days=candidate_lookback_days,
        candidate_buffer_days=candidate_buffer_days,
        train_top_k=train_top_k,
        validation_top_k=validation_top_k,
        min_candidate_days=min_candidate_days,
        selection_mode=selection_mode,
        fallback=fallback,
        kernel_target_effective_days=kernel_target_effective_days,
        kernel_weight_floor=kernel_weight_floor,
    )


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    total = float(values.sum())
    denom = float(np.square(values).sum())
    if total <= 0 or denom <= 0:
        return float("nan")
    return total * total / denom


def _gaussian_kernel_weights_for_ess(
    distances: pd.Series,
    *,
    target_effective_days: int,
    weight_floor: float,
) -> tuple[np.ndarray, float, float, str]:
    values = pd.to_numeric(distances, errors="coerce").to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("kernel similarity distances must be finite and non-negative")
    if target_effective_days <= 0 or target_effective_days > values.size:
        raise ValueError("kernel target effective days must be within the kernel training set")

    squared = np.square(values)
    relative_squared = squared - float(np.min(squared))
    if np.allclose(relative_squared, 0.0, rtol=0.0, atol=1e-15):
        weights = np.ones(values.size, dtype=float)
        return weights, float("inf"), _effective_sample_size(weights), "uniform_distances"

    scale = float(np.sqrt(np.max(relative_squared)))

    def _weights_at(bandwidth: float) -> np.ndarray:
        log_weights = -0.5 * relative_squared / float(bandwidth * bandwidth)
        log_weights -= float(np.max(log_weights))
        raw = np.exp(log_weights)
        return np.maximum(raw, weight_floor)

    lower = max(scale * 1e-12, np.finfo(float).tiny)
    upper = max(scale, lower * 2.0)
    lower_weights = _weights_at(lower)
    lower_ess = _effective_sample_size(lower_weights)
    if target_effective_days <= lower_ess + 1e-9:
        normalized = lower_weights / float(np.mean(lower_weights))
        return normalized, lower, _effective_sample_size(normalized), "tied_nearest"

    upper_weights = _weights_at(upper)
    while _effective_sample_size(upper_weights) < target_effective_days and upper < scale * 1e12:
        upper *= 2.0
        upper_weights = _weights_at(upper)
    if _effective_sample_size(upper_weights) < target_effective_days:
        normalized = upper_weights / float(np.mean(upper_weights))
        return normalized, upper, _effective_sample_size(normalized), "target_unreachable"

    for _ in range(80):
        middle = (lower + upper) / 2.0
        middle_weights = _weights_at(middle)
        if _effective_sample_size(middle_weights) < target_effective_days:
            lower = middle
        else:
            upper = middle
    weights = _weights_at((lower + upper) / 2.0)
    normalized = weights / float(np.mean(weights))
    return normalized, (lower + upper) / 2.0, _effective_sample_size(normalized), "ok"


class SimilarDayTrainingContext:
    def __init__(self, config: SimilarDayTrainingConfig, states: pd.DataFrame) -> None:
        self.config = config
        self._states = states

    @classmethod
    def from_config(cls, config: SimilarDayTrainingConfig) -> "SimilarDayTrainingContext":
        if not config.state_feature_path.exists():
            raise FileNotFoundError(
                f"similar_day_training state feature history missing: {config.state_feature_path}"
            )
        states = pd.read_csv(config.state_feature_path)
        if "trade_date" not in states.columns:
            raise KeyError("similar_day_training state feature history missing trade_date")
        missing = [col for col in config.feature_cols if col not in states.columns]
        if missing:
            raise KeyError(f"similar_day_training state feature history missing columns {missing}")
        states = states.copy()
        states["trade_date"] = pd.to_datetime(states["trade_date"], errors="coerce").dt.date
        for col in config.feature_cols:
            states[col] = pd.to_numeric(states[col], errors="coerce")
        states = states.dropna(subset=["trade_date"])
        states = states.drop_duplicates(subset=["trade_date"], keep="last")
        states = states.dropna(subset=list(config.feature_cols)).sort_values("trade_date").reset_index(drop=True)
        if states.empty:
            raise ValueError("similar_day_training state feature history has no complete rows")
        return cls(config, states)

    @property
    def state_days(self) -> int:
        return int(len(self._states))

    def select(
        self,
        *,
        target_day: date,
        available_days: Iterable[date],
    ) -> SimilarDaySelection:
        available = {pd.Timestamp(day).date() for day in available_days}
        current_rows = self._states[self._states["trade_date"] == target_day]
        if current_rows.empty:
            return self._failure(target_day, "current_state_missing")
        current = current_rows.iloc[-1][list(self.config.feature_cols)]
        candidates = self._states[
            (self._states["trade_date"] < target_day)
            & self._states["trade_date"].isin(available)
        ].sort_values("trade_date")
        candidates = candidates.tail(self.config.candidate_lookback_days).copy()
        candidate_days = int(len(candidates))
        if candidate_days < self.config.min_candidate_days:
            return self._failure(
                target_day,
                f"insufficient_candidates_{candidate_days}_lt_{self.config.min_candidate_days}",
                candidate_days=candidate_days,
            )

        feature_cols = list(self.config.feature_cols)
        means = candidates[feature_cols].mean(axis=0)
        stds = candidates[feature_cols].std(axis=0).replace(0, np.nan).fillna(1.0)
        current_z = (current - means) / stds
        candidates["distance"] = (
            ((candidates[feature_cols] - means) / stds - current_z) ** 2
        ).sum(axis=1).pow(0.5)
        if self.config.selection_mode in {"nearest", "kernel"}:
            ranked = candidates.sort_values(["distance", "trade_date"], ascending=[True, True])
        else:
            ranked = candidates.sort_values("trade_date", ascending=False)
        required = self.config.train_top_k + self.config.validation_top_k
        if len(ranked) < required:
            return self._failure(
                target_day,
                f"insufficient_ranked_candidates_{len(ranked)}_lt_{required}",
                candidate_days=candidate_days,
            )
        if self.config.selection_mode == "kernel":
            # Preserve ranks 61--80 as the same state-local validation band used by
            # Hard Similar60. All remaining candidates form the continuously weighted fit set.
            selected = ranked[["trade_date", "distance"]].copy()
            selected["rank"] = np.arange(1, len(selected) + 1, dtype=int)
            validation_mask = selected["rank"].between(
                self.config.train_top_k + 1,
                self.config.train_top_k + self.config.validation_top_k,
            )
            selected["role"] = np.where(validation_mask, "validation", "train")
            train_mask = selected["role"] == "train"
            kernel_weights, bandwidth, realized_ess, kernel_status = _gaussian_kernel_weights_for_ess(
                selected.loc[train_mask, "distance"],
                target_effective_days=int(self.config.kernel_target_effective_days or 0),
                weight_floor=self.config.kernel_weight_floor,
            )
            selected["weight"] = np.nan
            selected.loc[train_mask, "weight"] = kernel_weights
        else:
            selected = ranked.iloc[:required][["trade_date", "distance"]].copy()
            selected["rank"] = np.arange(1, len(selected) + 1, dtype=int)
            selected["role"] = np.where(
                selected["rank"] <= self.config.train_top_k,
                "train",
                "validation",
            )
            selected["weight"] = np.nan
            bandwidth = None
            realized_ess = None
            kernel_status = None
        train_days = tuple(sorted(selected.loc[selected["role"] == "train", "trade_date"].tolist()))
        validation_days = tuple(sorted(selected.loc[selected["role"] == "validation", "trade_date"].tolist()))
        return SimilarDaySelection(
            target_day=target_day,
            config=self.config,
            candidate_days=candidate_days,
            train_days=train_days,
            validation_days=validation_days,
            selections=selected.reset_index(drop=True),
            kernel_bandwidth=bandwidth,
            kernel_realized_effective_days=realized_ess,
            kernel_weight_status=kernel_status,
        )

    def _failure(
        self,
        target_day: date,
        reason: str,
        *,
        candidate_days: int = 0,
    ) -> SimilarDaySelection:
        return SimilarDaySelection(
            target_day=target_day,
            config=self.config,
            candidate_days=candidate_days,
            train_days=(),
            validation_days=(),
            selections=pd.DataFrame(columns=["trade_date", "distance", "rank", "role", "weight"]),
            reason=reason,
        )
