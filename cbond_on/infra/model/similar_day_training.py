from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

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
    strict_recent_window: bool = False
    state_manifest_path: Path | None = None
    kernel_target_effective_days: int | None = None
    kernel_weight_floor: float = 1e-12


@dataclass(frozen=True)
class StrictStateManifestBinding:
    """Verified immutable inputs for the strict Similar60 selector.

    The frozen score calendar is deliberately loaded once from the manifest
    contract.  Strict selection must never substitute a calendar inferred from
    currently discoverable raw data, labels, or factor files.
    """

    manifest_path: Path
    audit_path: Path
    calendar_path: Path
    frozen_calendar_days: tuple[date, ...]
    audit_by_day: Mapping[date, Mapping[str, object]]


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
            "similarity_strict_recent_window": bool(self.config.strict_recent_window),
            "similarity_state_manifest_path": (
                str(self.config.state_manifest_path)
                if self.config.state_manifest_path is not None
                else ""
            ),
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
                    "strict_recent_window": bool(self.config.strict_recent_window),
                    "state_feature_path": str(self.config.state_feature_path),
                    "state_manifest_path": (
                        str(self.config.state_manifest_path)
                        if self.config.state_manifest_path is not None
                        else None
                    ),
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
                    "strict_recent_window": bool(self.config.strict_recent_window),
                    "state_feature_path": str(self.config.state_feature_path),
                    "state_manifest_path": (
                        str(self.config.state_manifest_path)
                        if self.config.state_manifest_path is not None
                        else None
                    ),
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
    strict_recent_window_raw = raw.get("strict_recent_window", False)
    if not isinstance(strict_recent_window_raw, (bool, int)):
        raise TypeError("similar_day_training.strict_recent_window must be a boolean")
    strict_recent_window = bool(strict_recent_window_raw)
    manifest_raw = raw.get("state_manifest_path")
    state_manifest_path = (
        resolve_output_path(
            manifest_raw,
            default_path=state_path.with_name("t1430_market_state_features_pathfull_t1429_manifest.json"),
            results_root=results_root,
        )
        if manifest_raw not in (None, "")
        else None
    )
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
    if strict_recent_window and min_candidate_days != candidate_lookback_days:
        raise ValueError(
            "similar_day_training.strict_recent_window requires min_candidate_days "
            "to equal candidate_lookback_days"
        )
    if strict_recent_window and feature_set != "path_full_t1429":
        raise ValueError("similar_day_training.strict_recent_window requires feature_set=path_full_t1429")
    if strict_recent_window and state_manifest_path is None:
        raise ValueError("similar_day_training.strict_recent_window requires state_manifest_path")
    if selection_mode not in {"nearest", "latest", "kernel"}:
        raise ValueError("similar_day_training.selection_mode must be nearest, latest, or kernel")
    if fallback not in {"error", "rolling"}:
        raise ValueError("similar_day_training.fallback must be error or rolling")
    if strict_recent_window and fallback != "error":
        raise ValueError(
            "similar_day_training.strict_recent_window requires fallback=error; "
            "Hard Similar60 must not emit a rolling fallback score"
        )
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
        strict_recent_window=strict_recent_window,
        state_manifest_path=state_manifest_path,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_path(left: Path, right: Path) -> bool:
    """Compare manifest/config paths without making case sensitivity implicit."""

    try:
        left_text = str(left.resolve())
    except OSError:
        left_text = str(left)
    try:
        right_text = str(right.resolve())
    except OSError:
        right_text = str(right)
    return left_text.replace("/", "\\").casefold() == right_text.replace("/", "\\").casefold()


def _require_sha256(value: object, *, field: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"strict similar-day state manifest missing valid {field}")
    return digest


def _manifest_output_path(
    value: object,
    *,
    field: str,
    manifest_path: Path,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"strict similar-day state manifest missing {field}")
    path = Path(value.strip())
    # The V1 producer writes absolute paths.  Resolving a relative path against
    # the manifest is still deterministic and makes a malformed/legacy payload
    # fail later on its mandatory byte-level path binding rather than silently
    # reading from the process working directory.
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path


def _verify_hashed_artifact(
    path: Path,
    *,
    expected_sha256: object,
    field: str,
) -> str:
    expected = _require_sha256(expected_sha256, field=f"{field}_sha256")
    if not path.is_file():
        raise FileNotFoundError(f"strict similar-day {field} missing: {path}")
    try:
        actual = _sha256_file(path).lower()
    except OSError as exc:
        raise ValueError(f"strict similar-day {field} unreadable: {path}") from exc
    if actual != expected:
        raise ValueError(f"strict similar-day {field} SHA-256 does not match manifest")
    return actual


def _normalise_calendar_days(frame: pd.DataFrame, *, artifact_name: str) -> tuple[date, ...]:
    if "trade_date" not in frame.columns:
        raise KeyError(f"strict similar-day {artifact_name} missing trade_date")
    parsed = pd.to_datetime(frame["trade_date"], errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"strict similar-day {artifact_name} has invalid trade_date")
    days = tuple(pd.Timestamp(item).date() for item in parsed)
    if not days:
        raise ValueError(f"strict similar-day {artifact_name} has no trade dates")
    if len(set(days)) != len(days):
        raise ValueError(f"strict similar-day {artifact_name} has duplicate trade_date")
    if tuple(sorted(days)) != days:
        raise ValueError(f"strict similar-day {artifact_name} must be strictly ascending")
    return days


def _read_frozen_calendar(path: Path) -> tuple[date, ...]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"strict similar-day frozen calendar unreadable: {path}") from exc
    return _normalise_calendar_days(frame, artifact_name="frozen calendar")


def _audit_forward_pit_certified(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1"}
    return False


def _read_strict_audit(
    path: Path,
    *,
    frozen_calendar_days: tuple[date, ...],
) -> dict[date, Mapping[str, object]]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"strict similar-day audit unreadable: {path}") from exc
    required = {"trade_date", "outcome", "provenance_classification", "forward_pit_certified"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"strict similar-day audit missing columns {missing}")
    audit_days = _normalise_calendar_days(frame, artifact_name="audit")
    if set(audit_days) != set(frozen_calendar_days):
        raise ValueError("strict similar-day audit dates do not exactly match frozen calendar")
    records = frame.to_dict(orient="records")
    by_day: dict[date, Mapping[str, object]] = {}
    for day, record in zip(audit_days, records, strict=True):
        outcome = str(record.get("outcome", "")).strip().lower()
        provenance = str(record.get("provenance_classification", "")).strip().lower()
        if not outcome or not provenance:
            raise ValueError("strict similar-day audit has blank outcome or provenance classification")
        by_day[day] = {
            **record,
            "outcome": outcome,
            "provenance_classification": provenance,
            "forward_pit_certified": _audit_forward_pit_certified(
                record.get("forward_pit_certified")
            ),
        }
    return by_day


def _require_nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"strict similar-day state manifest invalid {field}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strict similar-day state manifest invalid {field}") from exc
    if parsed < 0:
        raise ValueError(f"strict similar-day state manifest invalid {field}")
    return parsed


def _verify_strict_state_manifest(config: SimilarDayTrainingConfig) -> StrictStateManifestBinding:
    """Load the forward-certified, byte-bound strict Similar60 inputs.

    A historical reconstruction may have valid same-day values, but it does
    not establish an immutable forward point-in-time artifact.  This consumer
    therefore accepts only the future, explicit forward-PIT certificate; it
    never treats a blocked or reconstructed manifest as an eligible fallback.
    """

    manifest_path = config.state_manifest_path
    if manifest_path is None:
        raise ValueError("strict_recent_window requires state_manifest_path")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"strict similar-day state manifest missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"strict similar-day state manifest unreadable: {manifest_path} ({type(exc).__name__})"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("strict similar-day state manifest must be an object")
    if payload.get("schema_version") != 1:
        raise ValueError("strict similar-day state manifest schema_version must be 1")
    if str(payload.get("status", "")).strip().lower() != "complete":
        raise ValueError(
            "strict similar-day state manifest must be complete; "
            "historical reconstruction and blocked manifests are not forward-PIT eligible"
        )
    certification = payload.get("certification")
    if not isinstance(certification, dict):
        raise ValueError("strict similar-day state manifest missing certification")
    if str(certification.get("status", "")).strip().lower() != "forward_pit_certified":
        raise ValueError("strict similar-day state manifest certification.status must be forward_pit_certified")
    if certification.get("forward_pit_certified") is not True:
        raise ValueError("strict similar-day state manifest is not forward-PIT certified")
    if str(payload.get("strict_cutoff_time", "")).strip() != "14:29":
        raise ValueError("strict similar-day state manifest cutoff must be 14:29")
    if str(payload.get("state_feature_set", "")).strip().lower() != config.feature_set:
        raise ValueError("strict similar-day state manifest feature_set mismatch")
    raw_columns = payload.get("state_feature_columns")
    if not isinstance(raw_columns, list) or tuple(str(item) for item in raw_columns) != config.feature_cols:
        raise ValueError("strict similar-day state manifest feature columns mismatch")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("strict similar-day state manifest outputs must be an object")
    manifest_state_path = _manifest_output_path(
        outputs.get("state_path"),
        field="outputs.state_path",
        manifest_path=manifest_path,
    )
    if not _same_path(manifest_state_path, config.state_feature_path):
        raise ValueError("strict similar-day state manifest state_path does not match configured state_feature_path")
    _verify_hashed_artifact(
        config.state_feature_path,
        expected_sha256=outputs.get("state_sha256"),
        field="state CSV",
    )
    audit_path = _manifest_output_path(
        outputs.get("audit_path"),
        field="outputs.audit_path",
        manifest_path=manifest_path,
    )
    _verify_hashed_artifact(
        audit_path,
        expected_sha256=outputs.get("audit_sha256"),
        field="audit CSV",
    )
    calendar_path = _manifest_output_path(
        outputs.get("calendar_path"),
        field="outputs.calendar_path",
        manifest_path=manifest_path,
    )
    calendar_hash = _verify_hashed_artifact(
        calendar_path,
        expected_sha256=outputs.get("calendar_sha256"),
        field="frozen calendar CSV",
    )
    expected_days = payload.get("expected_days")
    if not isinstance(expected_days, dict):
        raise ValueError("strict similar-day state manifest expected_days must be an object")
    if not _same_path(
        _manifest_output_path(
            expected_days.get("frozen_calendar_path"),
            field="expected_days.frozen_calendar_path",
            manifest_path=manifest_path,
        ),
        calendar_path,
    ):
        raise ValueError("strict similar-day expected_days frozen calendar path does not match outputs.calendar_path")
    expected_calendar_hash = _require_sha256(
        expected_days.get("frozen_calendar_sha256"),
        field="expected_days.frozen_calendar_sha256",
    )
    if expected_calendar_hash != calendar_hash:
        raise ValueError("strict similar-day expected_days frozen calendar SHA-256 does not match outputs")
    _require_sha256(expected_days.get("source_sha256"), field="expected_days.source_sha256")
    frozen_calendar_days = _read_frozen_calendar(calendar_path)
    if _require_nonnegative_int(expected_days.get("count"), field="expected_days.count") != len(
        frozen_calendar_days
    ):
        raise ValueError("strict similar-day expected_days.count does not match frozen calendar")
    audit_by_day = _read_strict_audit(
        audit_path,
        frozen_calendar_days=frozen_calendar_days,
    )
    certified_days = sum(
        bool(record["forward_pit_certified"]) for record in audit_by_day.values()
    )
    certification_count = _require_nonnegative_int(
        certification.get("forward_pit_certified_days"),
        field="certification.forward_pit_certified_days",
    )
    if certification_count != certified_days:
        raise ValueError(
            "strict similar-day certification.forward_pit_certified_days does not match audit"
        )
    counts = payload.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("strict similar-day state manifest counts must be an object")
    if _require_nonnegative_int(
        counts.get("forward_pit_certified_days"),
        field="counts.forward_pit_certified_days",
    ) != certified_days:
        raise ValueError("strict similar-day counts.forward_pit_certified_days does not match audit")
    return StrictStateManifestBinding(
        manifest_path=manifest_path,
        audit_path=audit_path,
        calendar_path=calendar_path,
        frozen_calendar_days=frozen_calendar_days,
        audit_by_day=audit_by_day,
    )


def _verify_strict_state_rows(
    states: pd.DataFrame,
    *,
    binding: StrictStateManifestBinding,
) -> None:
    """Make the hashed state CSV and hashed audit describe the same rows."""

    state_days = set(states["trade_date"])
    calendar_days = set(binding.frozen_calendar_days)
    unexpected_state_days = sorted(state_days.difference(calendar_days))
    if unexpected_state_days:
        raise ValueError("strict similar-day state CSV contains dates outside frozen calendar")
    built_audit_days = {
        day
        for day, record in binding.audit_by_day.items()
        if str(record["outcome"]).strip().lower() == "built"
    }
    if state_days != built_audit_days:
        raise ValueError("strict similar-day state CSV dates do not exactly match audit built dates")


class SimilarDayTrainingContext:
    def __init__(
        self,
        config: SimilarDayTrainingConfig,
        states: pd.DataFrame,
        *,
        strict_binding: StrictStateManifestBinding | None = None,
    ) -> None:
        self.config = config
        self._states = states
        self._strict_binding = strict_binding

    @classmethod
    def from_config(cls, config: SimilarDayTrainingConfig) -> "SimilarDayTrainingContext":
        if not config.state_feature_path.exists():
            raise FileNotFoundError(
                f"similar_day_training state feature history missing: {config.state_feature_path}"
            )
        strict_binding = (
            _verify_strict_state_manifest(config)
            if config.strict_recent_window
            else None
        )
        states = pd.read_csv(config.state_feature_path)
        if "trade_date" not in states.columns:
            raise KeyError("similar_day_training state feature history missing trade_date")
        if strict_binding is not None:
            raw_state_days = pd.to_datetime(states["trade_date"], errors="coerce")
            if raw_state_days.isna().any():
                raise ValueError("strict similar-day state CSV has invalid trade_date")
            if raw_state_days.dt.date.duplicated().any():
                raise ValueError("strict similar-day state CSV has duplicate trade_date")
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
        if strict_binding is not None:
            _verify_strict_state_rows(states, binding=strict_binding)
        return cls(config, states, strict_binding=strict_binding)

    @property
    def state_days(self) -> int:
        return int(len(self._states))

    def select(
        self,
        *,
        target_day: date,
        available_days: Iterable[date],
        expected_prior_days: Iterable[date] | None = None,
    ) -> SimilarDaySelection:
        available = {pd.Timestamp(day).date() for day in available_days}
        if self.config.strict_recent_window:
            # ``expected_prior_days`` used to be supplied by the runner's
            # raw-data calendar.  Hard Similar60 must not let currently
            # discoverable days alter its candidate window; the manifest's
            # frozen score calendar is the only calendar authority.
            if expected_prior_days is not None:
                return self._failure(
                    target_day,
                    "strict_recent_window_runtime_calendar_not_permitted",
                )
            binding = self._strict_binding
            if binding is None:
                raise RuntimeError("strict similar-day context missing verified manifest binding")
            frozen_calendar_days = binding.frozen_calendar_days
            if target_day not in binding.audit_by_day:
                return self._failure(target_day, "strict_recent_window_target_not_in_frozen_calendar")
            target_audit = binding.audit_by_day[target_day]
            if (
                not bool(target_audit["forward_pit_certified"])
                or str(target_audit["outcome"]).strip().lower() != "built"
            ):
                return self._failure(
                    target_day,
                    "strict_recent_window_target_not_forward_pit_certified",
                )
            target_index = frozen_calendar_days.index(target_day)
            if target_index < self.config.candidate_lookback_days:
                return self._failure(
                    target_day,
                    "strict_recent_window_frozen_prior_days_"
                    f"{target_index}_lt_{self.config.candidate_lookback_days}",
                    candidate_days=target_index,
                )
            expected_window = frozen_calendar_days[
                target_index - self.config.candidate_lookback_days:target_index
            ]
            uncertified = [
                day
                for day in expected_window
                if (
                    not bool(binding.audit_by_day[day]["forward_pit_certified"])
                    or str(binding.audit_by_day[day]["outcome"]).strip().lower() != "built"
                )
            ]
            if uncertified:
                return self._failure(
                    target_day,
                    "strict_recent_window_prior_not_forward_pit_certified_"
                    f"{len(uncertified)}",
                    candidate_days=self.config.candidate_lookback_days - len(uncertified),
                )
            current_rows = self._states[self._states["trade_date"] == target_day]
            if current_rows.empty:
                return self._failure(target_day, "current_state_missing")
            current = current_rows.iloc[-1][list(self.config.feature_cols)]
            state_days = set(self._states["trade_date"])
            missing_state = [day for day in expected_window if day not in state_days]
            missing_trainable = [day for day in expected_window if day not in available]
            eligible_days = [
                day
                for day in expected_window
                if day in state_days and day in available
            ]
            if missing_state or missing_trainable:
                return self._failure(
                    target_day,
                    (
                        "strict_recent_window_missing_state_"
                        f"{len(missing_state)}_missing_trainable_{len(missing_trainable)}"
                    ),
                    candidate_days=len(eligible_days),
                )
            candidates = self._states[
                self._states["trade_date"].isin(expected_window)
            ].sort_values("trade_date").copy()
        else:
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
