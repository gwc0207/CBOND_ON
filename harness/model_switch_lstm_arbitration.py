"""Pure, research-only building blocks for causal LSTM model arbitration.

The live model-switch selector does not import this module.  It intentionally
contains only offline sequence construction, deterministic CPU fitting, and
the conversion from two Helmert contrasts back to a coherent three-model
forecast.  Filesystem, database, configuration, and scheduler access belong
to the isolated replay tool.

Timing contract
---------------
For a score day ``t`` a sequence uses state observations ending at ``t`` and
predicts the two return contrasts for ``t``.  The state is observable at the
T1430 decision point; the return is not.  Any model used for a prediction at
``t`` is fitted only on sample targets strictly before ``t``.  Missing state
rows break a sequence -- this module never interpolates them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import os
import random
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from harness.model_switch_temporal_arbitration import TemporalForecast, contrasts_to_utilities, make_utility_forecast


# Work in basis points during optimization.  This is a fixed numerical
# conditioning transform, not a fitted label normalizer.
TARGET_SCALE = 10_000.0
CPU_DEVICE = "cpu"


@dataclass(frozen=True)
class LSTMTrial:
    """One member of the deliberately small, predeclared LSTM grid."""

    sequence_length: int
    hidden_size: int
    dropout: float
    seed: int = 20_260_730
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 80
    patience: int = 12
    batch_size: int = 32
    grad_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.sequence_length <= 1:
            raise ValueError("sequence_length must be greater than one")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer parameters")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0 or self.grad_clip_norm <= 0.0:
            raise ValueError("epoch, patience, batch_size, and grad_clip_norm values must be positive")

    @property
    def trial_id(self) -> str:
        dropout_label = f"{self.dropout:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"lstm_seq{self.sequence_length}_hidden{self.hidden_size}_drop{dropout_label}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"trial_id": self.trial_id}


def small_lstm_grid(*, seed: int = 20_260_730) -> tuple[LSTMTrial, ...]:
    """Return the eight fixed trials approved for the first LSTM pass."""

    return tuple(
        LSTMTrial(sequence_length=length, hidden_size=hidden, dropout=dropout, seed=seed)
        for length in (20, 40)
        for hidden in (8, 16)
        for dropout in (0.0, 0.1)
    )


@dataclass(frozen=True)
class CausalSequenceSamples:
    """Contiguous state windows and their same-day, post-decision labels."""

    inputs: np.ndarray
    targets: np.ndarray
    target_indices: np.ndarray
    input_start_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3:
            raise ValueError("inputs must have shape (samples, sequence, features)")
        if self.targets.ndim != 2 or self.targets.shape[1] != 2:
            raise ValueError("targets must have shape (samples, 2)")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError("inputs and targets must have the same sample count")
        if self.target_indices.ndim != 1 or self.input_start_indices.ndim != 1:
            raise ValueError("sample index vectors must be one-dimensional")
        if len(self.target_indices) != self.inputs.shape[0] or len(self.input_start_indices) != self.inputs.shape[0]:
            raise ValueError("sample index vectors must match the sample count")

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.inputs.shape[2])

    @property
    def sequence_length(self) -> int:
        return int(self.inputs.shape[1])


def build_causal_sequence_samples(
    features: np.ndarray,
    contrast_targets: np.ndarray,
    complete_rows: Sequence[bool] | np.ndarray,
    *,
    sequence_length: int,
    target_indices: Iterable[int],
    require_target: bool = True,
) -> CausalSequenceSamples:
    """Build samples without crossing a missing input-state row.

    ``target_indices`` may contain any chronological subset, but each returned
    window is exactly ``[target-sequence_length+1, target]``.  Thus the caller
    can prove an outer model sees only target labels before a later prediction
    day by restricting the endpoint set.  ``complete_rows`` describes input
    availability only.  With ``require_target=False``, an endpoint may have an
    unavailable target and still yield a prediction window; this makes it
    explicit that the same-day return never controls input construction.
    """

    feature_matrix = np.asarray(features, dtype=float)
    target_matrix = np.asarray(contrast_targets, dtype=float)
    complete = np.asarray(complete_rows, dtype=bool)
    if feature_matrix.ndim != 2:
        raise ValueError("features must have shape (days, features)")
    if target_matrix.shape != (feature_matrix.shape[0], 2):
        raise ValueError("contrast_targets must have shape (days, 2)")
    if complete.shape != (feature_matrix.shape[0],):
        raise ValueError("complete_rows must have one value per day")
    if sequence_length <= 1:
        raise ValueError("sequence_length must be greater than one")

    requested = np.asarray(list(target_indices), dtype=int)
    if requested.ndim != 1:
        raise ValueError("target_indices must be one-dimensional")
    if len(requested) and (requested.min() < 0 or requested.max() >= len(feature_matrix)):
        raise IndexError("target index falls outside the input panel")
    if len(requested) and np.any(np.diff(requested) < 0):
        raise ValueError("target_indices must be chronological")
    if len(requested) and len(np.unique(requested)) != len(requested):
        raise ValueError("target_indices must be unique")

    valid_input_rows = complete & np.isfinite(feature_matrix).all(axis=1)
    valid_targets = np.isfinite(target_matrix).all(axis=1)
    sample_inputs: list[np.ndarray] = []
    sample_targets: list[np.ndarray] = []
    endpoints: list[int] = []
    starts: list[int] = []
    for endpoint in requested:
        start = int(endpoint) - sequence_length + 1
        if start < 0 or not bool(valid_input_rows[start : int(endpoint) + 1].all()):
            continue
        if require_target and not bool(valid_targets[int(endpoint)]):
            continue
        sample_inputs.append(feature_matrix[start : int(endpoint) + 1].copy())
        sample_targets.append(target_matrix[int(endpoint)].copy())
        endpoints.append(int(endpoint))
        starts.append(start)

    if not sample_inputs:
        return CausalSequenceSamples(
            inputs=np.empty((0, sequence_length, feature_matrix.shape[1]), dtype=float),
            targets=np.empty((0, 2), dtype=float),
            target_indices=np.empty(0, dtype=int),
            input_start_indices=np.empty(0, dtype=int),
        )
    return CausalSequenceSamples(
        inputs=np.stack(sample_inputs, axis=0),
        targets=np.stack(sample_targets, axis=0),
        target_indices=np.asarray(endpoints, dtype=int),
        input_start_indices=np.asarray(starts, dtype=int),
    )


def assert_targets_before(samples: CausalSequenceSamples, prediction_index: int) -> None:
    """Raise unless every label used by a fit precedes the prediction day."""

    if len(samples) and int(np.max(samples.target_indices)) >= int(prediction_index):
        raise ValueError(
            f"training target index {int(np.max(samples.target_indices))} is not strictly before prediction index {prediction_index}"
        )


@dataclass(frozen=True)
class FeatureScaler:
    """Per-feature scaler fitted solely on a caller-supplied training panel."""

    mean: np.ndarray
    scale: np.ndarray
    fitted_sample_count: int

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.scale.ndim != 1 or self.mean.shape != self.scale.shape:
            raise ValueError("mean and scale must be same-length vectors")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0.0):
            raise ValueError("scaler values must be finite and positive")

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.mean):
            raise ValueError("values have a different feature count from this scaler")
        return (array - self.mean) / self.scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fitted_sample_count": int(self.fitted_sample_count),
        }


def fit_feature_scaler(samples: CausalSequenceSamples) -> FeatureScaler:
    """Fit a scaler on all and only state values in the supplied samples."""

    if not len(samples):
        raise ValueError("cannot fit a scaler on zero samples")
    flat = np.asarray(samples.inputs, dtype=float).reshape(-1, samples.feature_count)
    if not np.isfinite(flat).all():
        raise ValueError("scaler samples must be finite")
    mean = np.mean(flat, axis=0)
    scale = np.std(flat, axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return FeatureScaler(mean=mean, scale=scale, fitted_sample_count=int(len(samples)))


def transform_samples(samples: CausalSequenceSamples, scaler: FeatureScaler) -> CausalSequenceSamples:
    """Return a float32 standardized copy while retaining causal metadata."""

    return CausalSequenceSamples(
        inputs=np.asarray(scaler.transform(samples.inputs), dtype=np.float32),
        targets=np.asarray(samples.targets, dtype=np.float32),
        target_indices=samples.target_indices.copy(),
        input_start_indices=samples.input_start_indices.copy(),
    )


def split_chronological_samples(samples: CausalSequenceSamples, *, validation_fraction: float) -> tuple[CausalSequenceSamples, CausalSequenceSamples]:
    """Split a non-empty sample series into early fit and late early-stop sets."""

    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if len(samples) < 10:
        raise ValueError("at least ten samples are required for chronological early stopping")
    validation_count = max(1, int(np.ceil(len(samples) * validation_fraction)))
    fit_count = len(samples) - validation_count
    if fit_count < 1:
        raise ValueError("no samples remain for model fitting")

    def take(start: int, end: int) -> CausalSequenceSamples:
        return CausalSequenceSamples(
            inputs=samples.inputs[start:end].copy(),
            targets=samples.targets[start:end].copy(),
            target_indices=samples.target_indices[start:end].copy(),
            input_start_indices=samples.input_start_indices[start:end].copy(),
        )

    return take(0, fit_count), take(fit_count, len(samples))


def seed_cpu_deterministically(seed: int) -> None:
    """Configure deterministic CPU-only PyTorch execution for a single fit."""

    # Python's hash seed is consumed at process startup, but setting and
    # recording it still makes any subprocess/checkpoint provenance explicit.
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    # This experiment deliberately forbids a CUDA/device-dependent code path.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch allows this call only before certain thread-pool use.  The
        # fixed one-thread model path remains deterministic if it is already
        # configured by a previous fit in the same process.
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class LSTMContrastModel(nn.Module):
    """A shallow unidirectional LSTM producing two coherent contrasts."""

    def __init__(self, *, feature_count: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.lstm = nn.LSTM(input_size=feature_count, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # ``nn.LSTM`` ignores its dropout argument for a single layer.  This
        # explicit head dropout is therefore the actual 0/0.1 grid dimension.
        self.head_dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.head(self.head_dropout(encoded[:, -1, :]))


@dataclass(frozen=True)
class LSTMFitResult:
    model: LSTMContrastModel
    best_epoch: int
    best_validation_loss: float | None
    final_training_loss: float
    epochs_ran: int


def _validate_training_arrays(samples: CausalSequenceSamples) -> None:
    if not len(samples):
        raise ValueError("LSTM fit requires at least one sample")
    if not np.isfinite(samples.inputs).all() or not np.isfinite(samples.targets).all():
        raise ValueError("LSTM fit requires finite samples")


def _run_epoch(
    model: LSTMContrastModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    criterion = nn.MSELoss(reduction="mean")
    for start in range(0, len(inputs), batch_size):
        end = min(len(inputs), start + batch_size)
        x_batch = inputs[start:end]
        y_batch = targets[start:end]
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(end - start)
        total_rows += int(end - start)
    return total_loss / float(total_rows)


def _evaluate_loss(model: LSTMContrastModel, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    criterion = nn.MSELoss(reduction="mean")
    model.eval()
    with torch.no_grad():
        return float(criterion(model(inputs), targets).detach().cpu())


def _tensors(samples: CausalSequenceSamples) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_training_arrays(samples)
    inputs = torch.as_tensor(samples.inputs, dtype=torch.float32, device=CPU_DEVICE)
    targets = torch.as_tensor(samples.targets * TARGET_SCALE, dtype=torch.float32, device=CPU_DEVICE)
    return inputs, targets


def fit_lstm_with_early_stopping(
    fit_samples: CausalSequenceSamples,
    early_stop_samples: CausalSequenceSamples,
    *,
    trial: LSTMTrial,
) -> LSTMFitResult:
    """Fit one trial, using only an earlier chronological split for stopping."""

    _validate_training_arrays(fit_samples)
    _validate_training_arrays(early_stop_samples)
    if fit_samples.feature_count != early_stop_samples.feature_count:
        raise ValueError("fit and early-stop data use different feature counts")
    if int(np.max(fit_samples.target_indices)) >= int(np.min(early_stop_samples.target_indices)):
        raise ValueError("early-stop samples must follow fit samples chronologically")

    seed_cpu_deterministically(trial.seed)
    model = LSTMContrastModel(feature_count=fit_samples.feature_count, hidden_size=trial.hidden_size, dropout=trial.dropout).to(CPU_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay)
    fit_x, fit_y = _tensors(fit_samples)
    stop_x, stop_y = _tensors(early_stop_samples)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_loss = float("inf")
    final_train_loss = float("nan")
    waiting = 0
    for epoch in range(1, trial.max_epochs + 1):
        final_train_loss = _run_epoch(
            model,
            optimizer,
            fit_x,
            fit_y,
            batch_size=trial.batch_size,
            grad_clip_norm=trial.grad_clip_norm,
        )
        validation_loss = _evaluate_loss(model, stop_x, stop_y)
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= trial.patience:
                break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("early stopping did not produce a model state")
    model.load_state_dict(best_state)
    return LSTMFitResult(
        model=model,
        best_epoch=int(best_epoch),
        best_validation_loss=float(best_loss),
        final_training_loss=float(final_train_loss),
        epochs_ran=int(epoch),
    )


def fit_lstm_fixed_epochs(samples: CausalSequenceSamples, *, trial: LSTMTrial, epochs: int, seed_offset: int = 0) -> LSTMFitResult:
    """Refit a selected architecture without exposing a later test period."""

    _validate_training_arrays(samples)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_cpu_deterministically(int(trial.seed) + int(seed_offset))
    model = LSTMContrastModel(feature_count=samples.feature_count, hidden_size=trial.hidden_size, dropout=trial.dropout).to(CPU_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay)
    inputs, targets = _tensors(samples)
    final_loss = float("nan")
    for _ in range(int(epochs)):
        final_loss = _run_epoch(
            model,
            optimizer,
            inputs,
            targets,
            batch_size=trial.batch_size,
            grad_clip_norm=trial.grad_clip_norm,
        )
    return LSTMFitResult(
        model=model,
        best_epoch=int(epochs),
        best_validation_loss=None,
        final_training_loss=float(final_loss),
        epochs_ran=int(epochs),
    )


def predict_lstm_contrasts(model: LSTMContrastModel, samples: CausalSequenceSamples) -> np.ndarray:
    """Predict unscaled two-contrast values for causal sample windows."""

    if not len(samples):
        return np.empty((0, 2), dtype=float)
    if not np.isfinite(samples.inputs).all():
        raise ValueError("prediction inputs must be finite")
    model.eval()
    inputs = torch.as_tensor(samples.inputs, dtype=torch.float32, device=CPU_DEVICE)
    with torch.no_grad():
        prediction = model(inputs).detach().cpu().numpy().astype(float)
    return prediction / TARGET_SCALE


def make_lstm_forecast(
    *,
    trial: LSTMTrial,
    contrast_prediction: Sequence[float] | np.ndarray,
    history_days: int,
    diagnostics: dict[str, float | int | str] | None = None,
) -> TemporalForecast:
    """Convert two outputs into a coherent, calibration-only LSTM forecast."""

    contrasts = np.asarray(contrast_prediction, dtype=float)
    if contrasts.shape != (2,) or not np.isfinite(contrasts).all():
        raise ValueError("contrast_prediction must be finite with shape (2,)")
    utilities = contrasts_to_utilities(contrasts)
    payload: dict[str, float | int | str] = {
        "sequence_length": int(trial.sequence_length),
        "hidden_size": int(trial.hidden_size),
        "dropout": float(trial.dropout),
        "target_scale": float(TARGET_SCALE),
        **(diagnostics or {}),
    }
    return make_utility_forecast(
        trial.trial_id,
        utilities,
        history_days=int(history_days),
        ready=True,
        diagnostics=payload,
    )


def checkpoint_payload(*, trial: LSTMTrial, scaler: FeatureScaler, fit_result: LSTMFitResult, split: str) -> dict[str, Any]:
    """Produce a portable research checkpoint payload for ``torch.save``."""

    return {
        "format_version": 1,
        "research_only": True,
        "device": CPU_DEVICE,
        "target_scale": TARGET_SCALE,
        "split": str(split),
        "trial": trial.as_dict(),
        "scaler": scaler.as_dict(),
        "fit": {
            "best_epoch": int(fit_result.best_epoch),
            "best_validation_loss": fit_result.best_validation_loss,
            "final_training_loss": float(fit_result.final_training_loss),
            "epochs_ran": int(fit_result.epochs_ran),
        },
        "model_state_dict": {key: value.detach().cpu() for key, value in fit_result.model.state_dict().items()},
    }
