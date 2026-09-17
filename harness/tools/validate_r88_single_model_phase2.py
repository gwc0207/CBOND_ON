"""Research-only multiple-testing and stability diagnostics for frozen R88.

This tool consumes the already completed R88 rolling-OOS return artifacts.  It
does not train, score, rebuild factors/panels, call the live runtime, write a
database, or modify the Phase-1/R88 inputs.

The frozen trial family is the 30 candidates in the R88 study.  The primary
contracts are deliberately explicit:

* PBO/CSCV and development DSR use the 2025 development period only;
* White RC / SPA-style family tests and MCS compare raw daily returns with
  Regsim on date-aligned common days;
* beta, HAC residual alpha, and risk stability are reported for development,
  reporting, and overall scopes;
* all resampling uses the same circular moving-block scheme within a scope.

The SPA and MCS implementations are transparent NumPy/SciPy diagnostics.  The
output names and report state exactly which studentization/range approximation
is used; they must not be read as a live promotion decision or as correction
for any search outside the frozen 30-candidate family.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.tools.validate_r88_single_model_phase1 import (  # noqa: E402
    BENCHMARK_METHOD,
    DEV_END,
    DEV_START,
    DEFAULT_REGSIM_PATH,
    REPORT_START,
    CandidateSource,
    ValidationError,
    _load_candidate_sources,
    _load_regsim,
    _load_r88_returns,
    _moving_block_index_matrix,
    _ols_hac,
    _scope,
    _sha256_file,
)


STUDY_ID = "r88_joint_factor_lgbm_20260828_r1"
DEFAULT_STUDY_ROOT = Path(
    r"D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/"
    r"runs/rolling_20250102_20260827"
)
DEFAULT_PHASE1_ROOT = Path(
    r"D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1/"
    r"phase1_20260901_r3"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"D:/cbond_on/research_scratch/r88_single_model_validation_phase2_20260901_r1"
)
DEFAULT_OUTPUT_RUN = "phase2_20260901_r1"

ANNUALIZATION = 252.0
TRIAL_COUNT = 30
PBO_BLOCK_COUNT = 10
PBO_HALF_BLOCKS = 5
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_REPS = 10_000
MCS_BOOTSTRAP_REPS = 5_000
MCS_ALPHA = 0.10
RANDOM_SEED = 20_260_902
RESULTS_ENCODING = "utf-8-sig"
ROLLING_RISK_WINDOWS = (60, 120)
SCOPE_NAMES = ("development_2025", "reporting_2026", "overall")
EXPECTED_IMMUTABLE_HASH_COUNT = 129


class Phase2ValidationError(RuntimeError):
    """Raised when a frozen Phase-2 input or statistical contract fails."""


@dataclass(frozen=True)
class FrozenInputs:
    sources: tuple[CandidateSource, ...]
    plan: dict[str, Any]
    integrity: dict[str, Any]
    integrity_verification: dict[str, Any]
    expected_days: pd.DatetimeIndex
    candidate_returns: pd.DataFrame
    benchmark_returns: pd.Series
    regsim: pd.DataFrame
    phase1_evidence: dict[str, Any]
    study_root: Path
    phase1_root: Path
    regsim_path: Path


def _json_default(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase2ValidationError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = parent.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise Phase2ValidationError(f"{label} must stay below {root}: {resolved}") from exc
    return resolved


def _make_run_root(output_root: Path, run_name: str) -> Path:
    root = _resolve_inside(output_root, DEFAULT_OUTPUT_ROOT, label="output root")
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise Phase2ValidationError(f"run name must be a single path component: {run_name!r}")
    run_root = _resolve_inside(root / run_name, DEFAULT_OUTPUT_ROOT, label="run root")
    if run_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise Phase2ValidationError("a fresh child run directory is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    return run_root


def _finite_array(values: Iterable[float], *, label: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2 or not np.isfinite(array).all():
        raise Phase2ValidationError(f"{label} must contain at least two finite values")
    return array


def _sharpe(values: Iterable[float]) -> float:
    array = _finite_array(values, label="return series")
    std = float(np.std(array, ddof=1))
    return float(np.mean(array) / std * math.sqrt(ANNUALIZATION)) if std > 0 else float("nan")


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise Phase2ValidationError("objective values contain non-finite entries")
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low <= 1e-15:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - low) / (high - low)


def _scope_dates(days: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    if name == "development_2025":
        return days[(days >= DEV_START) & (days <= DEV_END)]
    if name == "reporting_2026":
        return days[days >= REPORT_START]
    if name == "overall":
        return days
    raise ValueError(name)


def _scope_frame(daily: pd.DataFrame, name: str) -> pd.DataFrame:
    """Apply the Phase-1 scope contract to a normalized daily frame."""

    if name == "development_2025":
        return daily.loc[(daily["trade_date"] >= DEV_START) & (daily["trade_date"] <= DEV_END)].copy()
    if name == "reporting_2026":
        return daily.loc[daily["trade_date"] >= REPORT_START].copy()
    if name == "overall":
        return daily.copy()
    raise ValueError(name)


def _load_phase1_evidence(phase1_root: Path, *, study_root: Path, regsim_path: Path) -> dict[str, Any]:
    phase1_root = phase1_root.resolve()
    status = _read_json(phase1_root / "run_status.json")
    if status.get("status") != "completed":
        raise Phase2ValidationError(f"Phase-1 run is not completed: {status.get('status')!r}")
    manifest = _read_json(phase1_root / "run_manifest.json")
    safety = {
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "factor_store_writes": False,
        "model_scoring_called": False,
        "model_training_called": False,
    }
    for key, expected in safety.items():
        if manifest.get(key) is not expected:
            raise Phase2ValidationError(f"Phase-1 safety flag {key} is not {expected}")
    evidence = _read_json(phase1_root / "input_evidence.json")
    if evidence.get("study_id") != STUDY_ID or evidence.get("candidate_count") != TRIAL_COUNT:
        raise Phase2ValidationError("Phase-1 evidence does not belong to the frozen R88 family")
    if evidence.get("candidate_calendar_days") != 401:
        raise Phase2ValidationError("Phase-1 evidence has an unexpected candidate calendar")
    if evidence.get("study_plan_sha256") != _sha256_file(study_root / "study_plan.json"):
        raise Phase2ValidationError("Phase-1 study plan hash no longer matches the frozen source")
    if evidence.get("study_integrity_sha256") != _sha256_file(study_root / "study_integrity.json"):
        raise Phase2ValidationError("Phase-1 study integrity hash no longer matches the frozen source")
    if evidence.get("regsim", {}).get("sha256") != _sha256_file(regsim_path):
        raise Phase2ValidationError("Phase-1 Regsim hash no longer matches the requested source")
    verification = evidence.get("study_integrity_verification", {})
    if verification.get("status") != "passed" or verification.get("hash_count") != EXPECTED_IMMUTABLE_HASH_COUNT:
        raise Phase2ValidationError("Phase-1 did not record the complete immutable-input verification")
    scorecard = pd.read_csv(phase1_root / "single_model_scorecard.csv")
    if len(scorecard) != TRIAL_COUNT * len(SCOPE_NAMES):
        raise Phase2ValidationError("Phase-1 scorecard row count is inconsistent")
    return {
        "phase1_status": status,
        "phase1_manifest": manifest,
        "phase1_evidence": evidence,
        "scorecard_rows": len(scorecard),
    }


def load_frozen_inputs(
    *,
    study_root: Path = DEFAULT_STUDY_ROOT,
    phase1_root: Path = DEFAULT_PHASE1_ROOT,
    regsim_path: Path = DEFAULT_REGSIM_PATH,
) -> FrozenInputs:
    """Load and validate only the existing frozen R88/Regsim artifacts."""

    study_root = study_root.resolve()
    phase1_root = phase1_root.resolve()
    regsim_path = regsim_path.resolve()
    sources, plan, integrity, integrity_verification = _load_candidate_sources(study_root)
    if len(sources) != TRIAL_COUNT:
        raise Phase2ValidationError("frozen R88 source count is not 30")
    expected_days = pd.DatetimeIndex(pd.to_datetime(plan["exact_calendar"]["score_days"]))
    if len(expected_days) != 401 or expected_days.has_duplicates:
        raise Phase2ValidationError("frozen R88 calendar is not the expected 401-day calendar")
    by_candidate = {
        source.candidate: _load_r88_returns(source, expected_days)
        for source in sources
    }
    candidate_returns = pd.DataFrame(
        {
            source.candidate: by_candidate[source.candidate]["day_return"].to_numpy(dtype=float)
            for source in sources
        },
        index=expected_days,
    )
    benchmark = by_candidate[sources[0].candidate]["benchmark_return"].copy()
    benchmark.index = expected_days
    for source in sources[1:]:
        current = by_candidate[source.candidate]["benchmark_return"].copy()
        current.index = expected_days
        if not np.array_equal(current.to_numpy(dtype=float), benchmark.to_numpy(dtype=float)):
            raise Phase2ValidationError(f"R88 benchmark differs for {source.candidate}")
    regsim = _load_regsim(regsim_path)
    phase1_meta = _load_phase1_evidence(phase1_root, study_root=study_root, regsim_path=regsim_path)
    return FrozenInputs(
        sources=tuple(sources),
        plan=plan,
        integrity=integrity,
        integrity_verification=integrity_verification,
        expected_days=expected_days,
        candidate_returns=candidate_returns,
        benchmark_returns=benchmark,
        regsim=regsim,
        phase1_evidence=phase1_meta["phase1_evidence"],
        study_root=study_root,
        phase1_root=phase1_root,
        regsim_path=regsim_path,
    )


def _circular_block_index_matrix(
    n_obs: int,
    *,
    block_length: int,
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular moving-block indices, preserving serial dependence."""

    if n_obs < 2 or block_length <= 0 or rows <= 0:
        raise ValueError("n_obs, block_length and rows must be positive")
    count = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs, size=(rows, count), dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, :, None] + offsets[None, None, :]) % n_obs).reshape(rows, -1)[:, :n_obs]


def _circular_bootstrap_means(
    values: np.ndarray,
    *,
    reps: int,
    block_length: int,
    seed: int,
    batch_size: int = 250,
) -> np.ndarray:
    """Return centered moving-block bootstrap means for every column."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("values must be a two-dimensional non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("bootstrap values contain non-finite entries")
    if reps <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    result = np.empty((reps, matrix.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    cursor = 0
    while cursor < reps:
        size = min(batch_size, reps - cursor)
        indices = _circular_block_index_matrix(
            matrix.shape[0], block_length=block_length, rows=size, rng=rng
        )
        result[cursor : cursor + size] = centered[indices].mean(axis=1)
        cursor += size
    return result


def _hac_mean_long_run_variance(values: np.ndarray) -> tuple[float, int]:
    values = _finite_array(values, label="HAC series")
    n_obs = len(values)
    lag = max(0, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
    centered = values - float(np.mean(values))
    omega = float(np.mean(centered * centered))
    for offset in range(1, min(lag, n_obs - 1) + 1):
        weight = 1.0 - offset / (lag + 1.0)
        gamma = float(np.mean(centered[offset:] * centered[:-offset]))
        omega += 2.0 * weight * gamma
    return max(omega, 0.0), lag


def _psr(values: np.ndarray, *, target_sharpe_annualized: float = 0.0) -> dict[str, float]:
    """Probabilistic Sharpe Ratio under the standard finite-sample approximation."""

    array = _finite_array(values, label="PSR series")
    n_obs = len(array)
    std = float(np.std(array, ddof=1))
    if std <= 0.0:
        return {
            "psr_n_obs": float(n_obs),
            "psr_sharpe_annualized": float("nan"),
            "psr_target_sharpe_annualized": float(target_sharpe_annualized),
            "psr_skewness": float("nan"),
            "psr_kurtosis_pearson": float("nan"),
            "psr_standard_error_daily": float("nan"),
            "psr_z": float("nan"),
            "psr_probability": float("nan"),
        }
    daily_sharpe = float(np.mean(array) / std)
    annualized_sharpe = daily_sharpe * math.sqrt(ANNUALIZATION)
    skewness = float(skew(array, bias=False)) if n_obs >= 3 else float("nan")
    kurtosis_pearson = float(kurtosis(array, fisher=False, bias=False)) if n_obs >= 4 else float("nan")
    if not np.isfinite([skewness, kurtosis_pearson]).all():
        return {
            "psr_n_obs": float(n_obs),
            "psr_sharpe_annualized": annualized_sharpe,
            "psr_target_sharpe_annualized": float(target_sharpe_annualized),
            "psr_skewness": skewness,
            "psr_kurtosis_pearson": kurtosis_pearson,
            "psr_standard_error_daily": float("nan"),
            "psr_z": float("nan"),
            "psr_probability": float("nan"),
        }
    variance_factor = 1.0 - skewness * daily_sharpe + ((kurtosis_pearson - 1.0) / 4.0) * daily_sharpe**2
    standard_error = math.sqrt(max(variance_factor, 1e-15) / max(n_obs - 1, 1))
    target_daily = float(target_sharpe_annualized) / math.sqrt(ANNUALIZATION)
    z_value = (daily_sharpe - target_daily) / standard_error if standard_error > 0 else float("nan")
    return {
        "psr_n_obs": float(n_obs),
        "psr_sharpe_annualized": annualized_sharpe,
        "psr_target_sharpe_annualized": float(target_sharpe_annualized),
        "psr_skewness": skewness,
        "psr_kurtosis_pearson": kurtosis_pearson,
        "psr_standard_error_daily": standard_error,
        "psr_z": z_value,
        "psr_probability": float(norm.cdf(z_value)) if np.isfinite(z_value) else float("nan"),
    }


def _daily_sharpe(values: np.ndarray) -> float:
    array = _finite_array(values, label="daily Sharpe series")
    std = float(np.std(array, ddof=1))
    return float(np.mean(array) / std) if std > 0.0 else float("nan")


def _dsr(
    values: np.ndarray,
    *,
    trial_daily_sharpes: np.ndarray,
    trial_count: int = TRIAL_COUNT,
) -> dict[str, float]:
    """Deflated Sharpe with the frozen family's cross-trial SR variance.

    The expected maximum null Sharpe is determined by the observed dispersion
    of the 30 candidate daily Sharpe estimates.  The candidate's own skew,
    kurtosis, and length then determine the PSR-style sampling denominator.
    """

    if trial_count < 2:
        raise ValueError("trial_count must be at least two")
    result = _psr(values)
    n_obs = int(result["psr_n_obs"])
    daily_sharpe = result["psr_sharpe_annualized"] / math.sqrt(ANNUALIZATION)
    se = result["psr_standard_error_daily"]
    trial_daily_sharpes = np.asarray(trial_daily_sharpes, dtype=float)
    trial_variance = float(np.var(trial_daily_sharpes, ddof=1)) if len(trial_daily_sharpes) >= 2 else float("nan")
    if not np.isfinite([daily_sharpe, se, trial_variance]).all() or se <= 0.0 or trial_variance < 0.0:
        result.update(
            {
                "dsr_trial_count": float(trial_count),
                "dsr_trial_daily_sharpe_variance": trial_variance,
                "dsr_expected_max_z": float("nan"),
                "dsr_threshold_sharpe_annualized": float("nan"),
                "dsr_z": float("nan"),
                "dsr_probability": float("nan"),
            }
        )
        return result
    euler_gamma = 0.5772156649015329
    q1 = float(norm.ppf(1.0 - 1.0 / trial_count))
    q2 = float(norm.ppf(1.0 - 1.0 / (trial_count * math.e)))
    expected_max_z = (1.0 - euler_gamma) * q1 + euler_gamma * q2
    threshold_daily = expected_max_z * math.sqrt(trial_variance)
    dsr_z = (daily_sharpe - threshold_daily) / se
    result.update(
        {
            "dsr_trial_count": float(trial_count),
            "dsr_trial_daily_sharpe_variance": trial_variance,
            "dsr_expected_max_z": expected_max_z,
            "dsr_threshold_sharpe_annualized": threshold_daily * math.sqrt(ANNUALIZATION),
            "dsr_z": dsr_z,
            "dsr_probability": float(norm.cdf(dsr_z)),
            "dsr_n_obs": float(n_obs),
        }
    )
    return result


def _segmented_ols_hac(
    strategy: np.ndarray,
    benchmark: np.ndarray,
    *,
    segment_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """OLS/HAC alpha where CSCV gaps never create artificial adjacencies."""

    strategy = _finite_array(strategy, label="strategy")
    benchmark = _finite_array(benchmark, label="benchmark")
    if len(strategy) != len(benchmark):
        raise ValueError("strategy and benchmark lengths differ")
    n_obs = len(strategy)
    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)
        if len(segment_ids) != n_obs:
            raise ValueError("segment ids length differs from returns")
    x = np.column_stack([np.ones(n_obs, dtype=float), benchmark])
    coefficients = np.linalg.pinv(x.T @ x) @ (x.T @ strategy)
    residual = strategy - x @ coefficients
    lag = max(0, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
    score = x * residual[:, None]
    omega = score.T @ score
    for offset in range(1, min(lag, n_obs - 1) + 1):
        if segment_ids is None:
            earlier = score[:-offset]
            later = score[offset:]
        else:
            valid = segment_ids[:-offset] == segment_ids[offset:]
            earlier = score[:-offset][valid]
            later = score[offset:][valid]
        if len(earlier) == 0:
            continue
        weight = 1.0 - offset / (lag + 1.0)
        cross = later.T @ earlier
        omega += weight * (cross + cross.T)
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ omega @ bread
    alpha_se = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
    alpha_t = float(coefficients[0] / alpha_se) if alpha_se > 0.0 else float("nan")
    return {
        "alpha_daily": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "alpha_hac_se": alpha_se,
        "alpha_hac_t": alpha_t,
        "alpha_hac_p_two_sided": float(2.0 * norm.sf(abs(alpha_t))) if np.isfinite(alpha_t) else float("nan"),
        "hac_lag": float(lag),
    }


def _selection_objective(
    returns: np.ndarray,
    benchmark: np.ndarray,
    candidates: Sequence[str],
    *,
    segment_ids: np.ndarray | None = None,
    scales: Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """The original frozen 0.6-Sharpe plus 0.4-HAC-alpha-t R88 objective."""

    returns = np.asarray(returns, dtype=float)
    benchmark = _finite_array(benchmark, label="objective benchmark")
    if returns.ndim != 2 or returns.shape[0] != len(benchmark):
        raise ValueError("return matrix is incompatible with objective benchmark")
    if returns.shape[1] != len(candidates) or not np.isfinite(returns).all():
        raise ValueError("return matrix is incompatible with candidate names")
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        values = returns[:, index]
        metrics = _segmented_ols_hac(values, benchmark, segment_ids=segment_ids)
        rows.append(
            {
                "candidate": candidate,
                "annualized_sharpe": _sharpe(values),
                "alpha_hac_t": metrics["alpha_hac_t"],
                "alpha_daily_bp": metrics["alpha_daily"] * 10_000.0,
                "beta": metrics["beta"],
            }
        )
    frame = pd.DataFrame(rows)
    if scales is None:
        sharpe_values = frame["annualized_sharpe"].to_numpy(dtype=float)
        alpha_values = frame["alpha_hac_t"].to_numpy(dtype=float)
        scales = {
            "sharpe": (float(np.min(sharpe_values)), float(np.max(sharpe_values))),
            "alpha": (float(np.min(alpha_values)), float(np.max(alpha_values))),
        }

    def apply_scale(values: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
        low, high = scale
        if high - low <= 1e-15:
            return np.full(values.shape, 0.5, dtype=float)
        return (values - low) / (high - low)

    frame["sharpe_minmax"] = apply_scale(frame["annualized_sharpe"].to_numpy(dtype=float), scales["sharpe"])
    frame["alpha_hac_t_minmax"] = apply_scale(frame["alpha_hac_t"].to_numpy(dtype=float), scales["alpha"])
    frame["selection_objective"] = 0.6 * frame["sharpe_minmax"] + 0.4 * frame["alpha_hac_t_minmax"]
    frame["selection_rank"] = rankdata(-frame["selection_objective"].to_numpy(dtype=float), method="average")
    return frame.sort_values(["selection_rank", "candidate"], kind="stable").reset_index(drop=True)


def _pbo_cscv(
    returns: pd.DataFrame,
    benchmark: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CSCV PBO for the fixed R88 composite selection objective.

    Ten chronological blocks form all C(10, 5)=252 directed IS/OOS splits.
    The candidate selection metric remains the original frozen R88 objective,
    rather than a post-hoc Sharpe-only substitute.
    """

    if len(returns) != len(benchmark):
        raise ValueError("CSCV return/benchmark lengths differ")
    if returns.shape[1] != TRIAL_COUNT:
        raise Phase2ValidationError("CSCV must use the complete 30-candidate family")
    n_obs = len(returns)
    if n_obs < PBO_BLOCK_COUNT * 10:
        raise Phase2ValidationError("development sample is too short for frozen CSCV blocks")
    candidates = list(returns.columns)
    blocks = [np.asarray(block, dtype=int) for block in np.array_split(np.arange(n_obs), PBO_BLOCK_COUNT)]
    if any(len(block) == 0 for block in blocks):
        raise Phase2ValidationError("CSCV has an empty chronological block")
    values = returns.to_numpy(dtype=float)
    bench = benchmark.to_numpy(dtype=float)
    split_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    selected_lambdas: list[float] = []
    split_number = 0
    for in_blocks in combinations(range(PBO_BLOCK_COUNT), PBO_HALF_BLOCKS):
        split_number += 1
        in_set = set(in_blocks)
        out_blocks = tuple(index for index in range(PBO_BLOCK_COUNT) if index not in in_set)
        in_indices = np.concatenate([blocks[index] for index in in_blocks])
        out_indices = np.concatenate([blocks[index] for index in out_blocks])
        in_segments = np.concatenate(
            [np.full(len(blocks[index]), order, dtype=int) for order, index in enumerate(in_blocks)]
        )
        out_segments = np.concatenate(
            [np.full(len(blocks[index]), order, dtype=int) for order, index in enumerate(out_blocks)]
        )
        in_objective = _selection_objective(
            values[in_indices], bench[in_indices], candidates, segment_ids=in_segments
        ).set_index("candidate")
        frozen_scales = {
            "sharpe": (
                float(in_objective["annualized_sharpe"].min()),
                float(in_objective["annualized_sharpe"].max()),
            ),
            "alpha": (
                float(in_objective["alpha_hac_t"].min()),
                float(in_objective["alpha_hac_t"].max()),
            ),
        }
        out_objective = _selection_objective(
            values[out_indices],
            bench[out_indices],
            candidates,
            segment_ids=out_segments,
            scales=frozen_scales,
        ).set_index("candidate")
        winner = str(in_objective.sort_values(["selection_rank", "candidate"], kind="stable").index[0])
        winner_oos_rank = float(out_objective.at[winner, "selection_rank"])
        # High percentile means better OOS rank.  PBO is exactly the fraction
        # for which the selected in-sample winner lands below the OOS median.
        oos_percentile = (TRIAL_COUNT + 1.0 - winner_oos_rank) / (TRIAL_COUNT + 1.0)
        logit = float(math.log(oos_percentile / (1.0 - oos_percentile)))
        selected_lambdas.append(logit)
        split_rows.append(
            {
                "split": split_number,
                "in_block_ids": ",".join(str(index + 1) for index in in_blocks),
                "out_block_ids": ",".join(str(index + 1) for index in out_blocks),
                "in_days": int(len(in_indices)),
                "out_days": int(len(out_indices)),
                "in_winner": winner,
                "in_winner_objective": float(in_objective.at[winner, "selection_objective"]),
                "out_winner_objective": float(out_objective.at[winner, "selection_objective"]),
                "in_sharpe_scale_low": frozen_scales["sharpe"][0],
                "in_sharpe_scale_high": frozen_scales["sharpe"][1],
                "in_alpha_t_scale_low": frozen_scales["alpha"][0],
                "in_alpha_t_scale_high": frozen_scales["alpha"][1],
                "out_winner_rank": winner_oos_rank,
                "out_winner_percentile_high_is_good": oos_percentile,
                "out_winner_logit": logit,
                "out_winner_below_median": bool(oos_percentile < 0.5),
            }
        )
        for candidate in candidates:
            rank_rows.append(
                {
                    "split": split_number,
                    "candidate": candidate,
                    "in_selection_objective": float(in_objective.at[candidate, "selection_objective"]),
                    "in_selection_rank": float(in_objective.at[candidate, "selection_rank"]),
                    "out_selection_objective": float(out_objective.at[candidate, "selection_objective"]),
                    "out_selection_rank": float(out_objective.at[candidate, "selection_rank"]),
                    "is_in_winner": candidate == winner,
                }
            )
    splits = pd.DataFrame(split_rows)
    ranks = pd.DataFrame(rank_rows)
    summary_rows: list[dict[str, object]] = []
    for candidate, group in ranks.groupby("candidate", sort=True):
        selected = group.loc[group["is_in_winner"]].copy()
        in_values = group["in_selection_rank"].to_numpy(dtype=float)
        out_values = group["out_selection_rank"].to_numpy(dtype=float)
        in_centered = in_values - np.mean(in_values)
        out_centered = out_values - np.mean(out_values)
        denominator = float(np.sqrt(np.sum(in_centered**2) * np.sum(out_centered**2)))
        rank_correlation = float(np.sum(in_centered * out_centered) / denominator) if denominator > 0.0 else float("nan")
        summary_rows.append(
            {
                "candidate": candidate,
                "cscv_split_count": int(len(group)),
                "cscv_mean_in_rank": float(np.mean(in_values)),
                "cscv_mean_out_rank": float(np.mean(out_values)),
                "cscv_complement_rank_correlation_descriptive": rank_correlation,
                "cscv_selected_split_count": int(len(selected)),
                "cscv_selected_frequency": float(len(selected) / len(splits)),
                "cscv_selected_oos_median_rank": float(selected["out_selection_rank"].median()) if len(selected) else float("nan"),
                "cscv_selected_oos_below_median_fraction": float((selected["out_selection_rank"] > (TRIAL_COUNT + 1.0) / 2.0).mean()) if len(selected) else float("nan"),
            }
        )
    family = {
        "method": "CSCV PBO on frozen R88 composite objective",
        "scope": "development_2025",
        "candidate_count": TRIAL_COUNT,
        "chronological_block_count": PBO_BLOCK_COUNT,
        "in_sample_block_count": PBO_HALF_BLOCKS,
        "directed_split_count": int(len(splits)),
        "selection_objective": "0.6 * minmax(annualized Sharpe) + 0.4 * minmax(HAC alpha t)",
        "pbo": float(np.mean(np.asarray(selected_lambdas) <= 0.0)),
        "selected_oos_median_rank": float(splits["out_winner_rank"].median()),
        "selected_oos_below_median_count": int(splits["out_winner_below_median"].sum()),
        "selected_oos_logit_median": float(np.median(selected_lambdas)),
        "note": "CSCV is a symmetric resampling diagnostic, not a chronological walk-forward validation or a future failure probability.",
    }
    return family, splits, pd.DataFrame(summary_rows).sort_values("candidate").reset_index(drop=True), ranks


def _relative_hac_metrics(delta: np.ndarray) -> dict[str, float]:
    """One-sample HAC statistics for raw candidate-minus-Regsim returns."""

    values = _finite_array(delta, label="relative return")
    n_obs = len(values)
    lrv, lag = _hac_mean_long_run_variance(values)
    se = math.sqrt(lrv / n_obs) if lrv > 0.0 else float("nan")
    mean = float(np.mean(values))
    t_value = mean / se if np.isfinite(se) and se > 0.0 else float("nan")
    return {
        "relative_n_obs": float(n_obs),
        "relative_mean_bp_per_day": mean * 10_000.0,
        "relative_sharpe": _sharpe(values),
        "relative_hac_lag": float(lag),
        "relative_hac_se_bp_per_day": se * 10_000.0 if np.isfinite(se) else float("nan"),
        "relative_hac_t": t_value,
        "relative_hac_p_two_sided": float(2.0 * norm.sf(abs(t_value))) if np.isfinite(t_value) else float("nan"),
    }


def _family_relative_tests(
    deltas: pd.DataFrame,
    *,
    scope: str,
    reps: int,
    block_length: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """White RC and Hansen-style SPA tests for a fixed candidate family."""

    matrix = deltas.to_numpy(dtype=float)
    candidates = list(deltas.columns)
    if matrix.shape[1] != TRIAL_COUNT:
        raise Phase2ValidationError("relative family test must contain all 30 R88 candidates")
    if matrix.shape[0] < 30 or not np.isfinite(matrix).all():
        raise Phase2ValidationError("relative family test requires at least 30 finite common days")
    n_obs = matrix.shape[0]
    means = matrix.mean(axis=0)
    lrv = np.empty(TRIAL_COUNT, dtype=float)
    hac_lags = np.empty(TRIAL_COUNT, dtype=int)
    for index in range(TRIAL_COUNT):
        lrv[index], hac_lags[index] = _hac_mean_long_run_variance(matrix[:, index])
    lrv_sd = np.sqrt(lrv)
    if not np.isfinite(lrv_sd).all():
        raise Phase2ValidationError("relative family contains non-finite long-run variance")
    spa_eligible = lrv_sd > 1e-15
    root_n = math.sqrt(n_obs)
    rc_observed = float(root_n * np.max(means))
    spa_t = np.full(TRIAL_COUNT, np.nan, dtype=float)
    spa_t[spa_eligible] = root_n * means[spa_eligible] / lrv_sd[spa_eligible]
    spa_observed = float(np.nanmax(spa_t)) if np.any(spa_eligible) else 0.0
    loglog_threshold = math.sqrt(max(0.0, 2.0 * math.log(math.log(n_obs))))
    anchors = {
        "lower": np.where(spa_eligible & (spa_t >= 0.0), means, 0.0),
        "consistent": np.where(spa_eligible & (spa_t >= -loglog_threshold), means, 0.0),
        "upper": means.copy(),
    }
    centered = matrix - means[None, :]
    rng = np.random.default_rng(seed)
    rc_exceed = 0
    spa_exceed = {name: 0 for name in anchors}
    cursor = 0
    batch_size = min(250, reps)
    while cursor < reps:
        size = min(batch_size, reps - cursor)
        indices = _circular_block_index_matrix(
            n_obs, block_length=block_length, rows=size, rng=rng
        )
        # White RC bootstraps the jointly centered loss-differential matrix.
        rc_means = centered[indices].mean(axis=1)
        rc_values = root_n * np.max(rc_means, axis=1)
        rc_exceed += int(np.sum(rc_values >= rc_observed))
        # SPA uses the original bootstrap series and applies Hansen's three
        # fixed recentering rules, all with the same cross-candidate indices.
        raw_means = matrix[indices].mean(axis=1)
        for name, anchor in anchors.items():
            standardized = np.full((size, TRIAL_COUNT), -np.inf, dtype=float)
            standardized[:, spa_eligible] = (
                root_n
                * (raw_means[:, spa_eligible] - anchor[None, spa_eligible])
                / lrv_sd[None, spa_eligible]
            )
            values = np.max(standardized, axis=1) if np.any(spa_eligible) else np.zeros(size, dtype=float)
            spa_exceed[name] += int(np.sum(values >= spa_observed))
        cursor += size
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        metrics = _relative_hac_metrics(matrix[:, index])
        rows.append(
            {
                "scope": scope,
                "candidate": candidate,
                **metrics,
                "relative_spa_t": float(spa_t[index]),
                "relative_lrv_sd": float(lrv_sd[index]),
                "relative_spa_eligible": bool(spa_eligible[index]),
                "relative_spa_recenter_lower": float(anchors["lower"][index]),
                "relative_spa_recenter_consistent": float(anchors["consistent"][index]),
                "relative_spa_recenter_upper": float(anchors["upper"][index]),
            }
        )
    best_index = int(np.argmax(means))
    family = {
        "scope": scope,
        "common_days": int(n_obs),
        "candidate_count": TRIAL_COUNT,
        "bootstrap": {
            "method": "synchronous circular moving-block bootstrap",
            "repetitions": int(reps),
            "block_length_days": int(block_length),
            "seed": int(seed),
            "p_value_rule": "(1 + exceedances) / (1 + repetitions)",
        },
        "white_reality_check": {
            "statistic": "sqrt(T) * max candidate mean(raw day_return - Regsim raw day_return)",
            "observed": rc_observed,
            "p_value": float((rc_exceed + 1) / (reps + 1)),
            "best_mean_candidate": candidates[best_index],
            "best_mean_bp_per_day": float(means[best_index] * 10_000.0),
        },
        "hansen_spa": {
            "statistic": "max candidate sqrt(T) * mean differential / Bartlett-HAC long-run sd",
            "observed": spa_observed,
            "recenter": "lower: center nonnegative; consistent: center not-strongly-negative; upper: center all",
            "loglog_threshold": loglog_threshold,
            "p_value_lower": float((spa_exceed["lower"] + 1) / (reps + 1)),
            "p_value_consistent": float((spa_exceed["consistent"] + 1) / (reps + 1)),
            "p_value_upper": float((spa_exceed["upper"] + 1) / (reps + 1)),
            "nested_model_adjustment": False,
        },
        "interpretation": "family-level raw-return comparison only; no cross-source benchmark alpha is inferred",
    }
    return family, pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def _mcs_range(
    returns: pd.DataFrame,
    *,
    scope: str,
    reps: int,
    block_length: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Block-bootstrap range-statistic MCS for frozen raw-return loss.

    Loss is fixed before observation as ``-raw day_return``.  The range
    statistic is the maximum absolute studentized pairwise loss differential.
    The candidate removed after a rejection is the one with the largest
    positive standardized loss against another active member.
    """

    data = returns.to_numpy(dtype=float)
    names = list(returns.columns)
    if data.ndim != 2 or data.shape[0] < 30 or data.shape[1] < 2:
        raise Phase2ValidationError("MCS needs at least 30 days and two models")
    if not np.isfinite(data).all():
        raise Phase2ValidationError("MCS return matrix contains non-finite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("MCS alpha must lie in (0, 1)")
    losses = -data
    active = list(range(len(names)))
    eliminated: dict[int, dict[str, object]] = {}
    trace_rows: list[dict[str, object]] = []
    step = 0
    rng = np.random.default_rng(seed)
    # One common set of indices is frozen for the entire step-down path.  This
    # is important: regenerating a different bootstrap for each elimination
    # would change the null distribution while the model set changes.
    bootstrap_indices = _circular_block_index_matrix(
        data.shape[0], block_length=block_length, rows=reps, rng=rng
    )
    previous_inclusion_p = 0.0
    final_inclusion_p = 1.0
    final_step_p_value = 1.0
    while len(active) > 1:
        step += 1
        loss = losses[:, active]
        n_obs, n_models = loss.shape
        # Pairwise loss difference: positive means the row model has higher
        # loss (lower return) than the column model.
        pair = loss[:, :, None] - loss[:, None, :]
        pair_mean = pair.mean(axis=0)
        centered_loss = loss - loss.mean(axis=0, keepdims=True)
        # Estimate the pairwise standard errors from the same circular-block
        # bootstrap used for the range null, rather than mixing a HAC SE with a
        # block-bootstrap null.
        bootstrap_means = np.empty((reps, n_models), dtype=float)
        cursor = 0
        batch_size = min(200, reps)
        while cursor < reps:
            size = min(batch_size, reps - cursor)
            indices = bootstrap_indices[cursor : cursor + size]
            bootstrap_means[cursor : cursor + size] = centered_loss[indices].mean(axis=1)
            cursor += size
        bootstrap_pair_mean = bootstrap_means[:, :, None] - bootstrap_means[:, None, :]
        pair_sd = np.std(bootstrap_pair_mean, axis=0, ddof=1)
        valid = pair_sd > 0.0
        t_obs = np.full((n_models, n_models), np.nan, dtype=float)
        t_obs[valid] = pair_mean[valid] / pair_sd[valid]
        np.fill_diagonal(t_obs, np.nan)
        if not np.isfinite(t_obs).any():
            trace_rows.append(
                {
                    "scope": scope,
                    "step": step,
                    "active_model_count": int(n_models),
                    "range_statistic": 0.0,
                    "bootstrap_p_value": 1.0,
                    "monotone_inclusion_p_value": 1.0,
                    "alpha": alpha,
                    "mcs_rejected": False,
                    "worst_candidate_if_rejected": "",
                    "worst_mean_loss": float("nan"),
                    "worst_mean_return": float("nan"),
                }
            )
            break
        observed = float(np.nanmax(np.abs(t_obs)))
        row_max = np.nanmax(t_obs, axis=1)
        if not np.isfinite(row_max).any():
            raise Phase2ValidationError("MCS has no finite pairwise comparison")
        worst_local = int(np.nanargmax(row_max))
        with np.errstate(divide="ignore", invalid="ignore"):
            sample_t = np.divide(
                bootstrap_pair_mean,
                pair_sd[None, :, :],
                out=np.full_like(bootstrap_pair_mean, np.nan),
                where=pair_sd[None, :, :] > 0.0,
            )
        diagonal = np.arange(n_models)
        sample_t[:, diagonal, diagonal] = np.nan
        sample_statistic = np.nanmax(np.abs(sample_t), axis=(1, 2))
        exceed = int(np.sum(sample_statistic >= observed))
        p_value = float((exceed + 1) / (reps + 1))
        inclusion_p_value = max(previous_inclusion_p, p_value)
        final_step_p_value = p_value
        rejected = bool(p_value < alpha)
        worst_index = active[worst_local]
        trace_rows.append(
            {
                "scope": scope,
                "step": step,
                "active_model_count": int(n_models),
                "range_statistic": observed,
                "bootstrap_p_value": p_value,
                "monotone_inclusion_p_value": inclusion_p_value,
                "alpha": alpha,
                "mcs_rejected": rejected,
                "worst_candidate_if_rejected": names[worst_index],
                "worst_mean_loss": float(loss[:, worst_local].mean()),
                "worst_mean_return": float(data[:, active[worst_local]].mean()),
            }
        )
        if not rejected:
            break
        eliminated[worst_index] = {
            "mcs_elimination_step": step,
            "mcs_elimination_p_value": p_value,
            "mcs_inclusion_p_value": inclusion_p_value,
            "mcs_elimination_range_statistic": observed,
        }
        previous_inclusion_p = inclusion_p_value
        active.pop(worst_local)
    membership_rows: list[dict[str, object]] = []
    active_set = set(active)
    for index, name in enumerate(names):
        row = {
            "scope": scope,
            "candidate": name,
            "mean_raw_day_return_bp": float(data[:, index].mean() * 10_000.0),
            "mean_loss": float(losses[:, index].mean()),
            "mcs_alpha": alpha,
            "mcs_included": index in active_set,
            "mcs_elimination_step": np.nan,
            "mcs_elimination_p_value": np.nan,
            "mcs_elimination_range_statistic": np.nan,
                "mcs_inclusion_p_value": final_inclusion_p if index in active_set else np.nan,
                "mcs_final_step_p_value": final_step_p_value if index in active_set else np.nan,
        }
        row.update(eliminated.get(index, {}))
        membership_rows.append(row)
    trace = pd.DataFrame(trace_rows)
    summary = {
        "scope": scope,
        "method": "block-bootstrap variance range-statistic MCS approximation",
        "loss": "-raw day_return",
        "raw_return_common_days": int(data.shape[0]),
        "model_count": int(data.shape[1]),
        "bootstrap": {
            "method": "synchronous circular moving-block bootstrap",
            "repetitions": int(reps),
            "block_length_days": int(block_length),
            "seed": int(seed),
            "same_bootstrap_indices_reused_across_elimination_steps": True,
        },
        "alpha": alpha,
        "included_candidates": [names[index] for index in active],
        "included_count": int(len(active)),
        "step_count": int(step),
        "final_step_p_value": final_step_p_value,
        "retained_model_mcs_p_value": 1.0,
        "note": "MCS membership means equal expected raw-return loss could not be rejected; it is not a Sharpe ranking, profitability guarantee, or live admission.",
    }
    return summary, pd.DataFrame(membership_rows).sort_values("candidate").reset_index(drop=True), trace


def _max_drawdown(values: np.ndarray) -> float:
    nav = np.cumprod(1.0 + values)
    return float(np.min(nav / np.maximum.accumulate(nav) - 1.0))


def _rolling_regression_risk(
    returns: np.ndarray,
    benchmark: np.ndarray,
    *,
    window: int,
) -> dict[str, float]:
    if len(returns) != len(benchmark):
        raise ValueError("rolling risk return and benchmark lengths differ")
    if len(returns) < window:
        return {
            f"rolling{window}_count": 0.0,
            f"rolling{window}_beta_median": float("nan"),
            f"rolling{window}_beta_min": float("nan"),
            f"rolling{window}_beta_max": float("nan"),
            f"rolling{window}_beta_std": float("nan"),
            f"rolling{window}_alpha_bp_median": float("nan"),
            f"rolling{window}_alpha_bp_min": float("nan"),
            f"rolling{window}_alpha_bp_max": float("nan"),
            f"rolling{window}_alpha_t_median": float("nan"),
            f"rolling{window}_alpha_t_positive_fraction": float("nan"),
            f"rolling{window}_residual_vol_median": float("nan"),
        }
    beta: list[float] = []
    alpha_bp: list[float] = []
    alpha_t: list[float] = []
    residual_vol: list[float] = []
    for end in range(window, len(returns) + 1):
        metrics = _ols_hac(returns[end - window : end], benchmark[end - window : end])
        beta.append(metrics["beta"])
        alpha_bp.append(metrics["alpha_daily"] * 10_000.0)
        alpha_t.append(metrics["alpha_hac_t"])
        residual_vol.append(metrics["residual_daily_volatility"] * math.sqrt(ANNUALIZATION))
    arrays = {
        "beta": np.asarray(beta, dtype=float),
        "alpha_bp": np.asarray(alpha_bp, dtype=float),
        "alpha_t": np.asarray(alpha_t, dtype=float),
        "residual_vol": np.asarray(residual_vol, dtype=float),
    }
    finite_alpha_t = arrays["alpha_t"][np.isfinite(arrays["alpha_t"])]
    return {
        f"rolling{window}_count": float(len(beta)),
        f"rolling{window}_beta_median": float(np.median(arrays["beta"])),
        f"rolling{window}_beta_min": float(np.min(arrays["beta"])),
        f"rolling{window}_beta_max": float(np.max(arrays["beta"])),
        f"rolling{window}_beta_std": float(np.std(arrays["beta"], ddof=1)) if len(beta) > 1 else 0.0,
        f"rolling{window}_alpha_bp_median": float(np.median(arrays["alpha_bp"])),
        f"rolling{window}_alpha_bp_min": float(np.min(arrays["alpha_bp"])),
        f"rolling{window}_alpha_bp_max": float(np.max(arrays["alpha_bp"])),
        f"rolling{window}_alpha_t_median": float(np.median(finite_alpha_t)) if len(finite_alpha_t) else float("nan"),
        f"rolling{window}_alpha_t_positive_fraction": float(np.mean(finite_alpha_t > 0.0)) if len(finite_alpha_t) else float("nan"),
        f"rolling{window}_residual_vol_median": float(np.median(arrays["residual_vol"])),
    }


def _risk_stability_rows(
    inputs: FrozenInputs,
    *,
    scorecard: pd.DataFrame,
    block_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidate_names = list(inputs.candidate_returns.columns)
    for candidate in candidate_names:
        for scope in SCOPE_NAMES:
            dates = _scope_dates(inputs.expected_days, scope)
            values = inputs.candidate_returns.loc[dates, candidate].to_numpy(dtype=float)
            benchmark = inputs.benchmark_returns.loc[dates].to_numpy(dtype=float)
            score = scorecard.loc[
                (scorecard["candidate"] == candidate) & (scorecard["scope"] == scope)
            ]
            if len(score) != 1:
                raise Phase2ValidationError(f"Phase-1 scorecard row missing for {candidate}/{scope}")
            score_row = score.iloc[0].to_dict()
            blocks = block_metrics.loc[
                (block_metrics["candidate"] == candidate) & (block_metrics["scope"] == scope)
            ]
            block_sharpes = pd.to_numeric(blocks["sharpe"], errors="coerce").to_numpy(dtype=float)
            block_alphas = pd.to_numeric(blocks["alpha_hac_t"], errors="coerce").to_numpy(dtype=float)
            row: dict[str, object] = {
                "candidate": candidate,
                "scope": scope,
                "start": str(dates.min().date()),
                "end": str(dates.max().date()),
                "n_obs": int(len(values)),
                "annualized_sharpe": float(score_row["sharpe"]),
                "cumulative_return": float(score_row["cumulative_return"]),
                "max_drawdown": float(score_row["max_drawdown"]),
                "daily_var_95": float(score_row["daily_var_95"]),
                "daily_cvar_95": float(score_row["daily_cvar_95"]),
                "beta": float(score_row["beta"]),
                "alpha_daily_bp": float(score_row["alpha_daily"] * 10_000.0),
                "alpha_hac_t": float(score_row["alpha_hac_t"]),
                "alpha_hac_p_two_sided": float(score_row["alpha_hac_p_two_sided"]),
                "residual_annualized_volatility": float(score_row["residual_daily_volatility"] * math.sqrt(ANNUALIZATION)),
                "r_squared": float(score_row["r_squared"]),
                "block_sharpe_median": float(np.nanmedian(block_sharpes)),
                "block_sharpe_min": float(np.nanmin(block_sharpes)),
                "block_sharpe_max": float(np.nanmax(block_sharpes)),
                "block_sharpe_std": float(np.nanstd(block_sharpes, ddof=1)),
                "block_sharpe_positive_fraction": float(np.mean(block_sharpes > 0.0)),
                "block_alpha_t_median": float(np.nanmedian(block_alphas)),
                "block_alpha_t_positive_fraction": float(np.mean(block_alphas > 0.0)),
            }
            for window in ROLLING_RISK_WINDOWS:
                row.update(_rolling_regression_risk(values, benchmark, window=window))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["scope", "candidate"]).reset_index(drop=True)


def _aligned_relative_returns(inputs: FrozenInputs, scope: str) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    candidate_dates = _scope_dates(inputs.expected_days, scope)
    regsim_dates = pd.DatetimeIndex(inputs.regsim["trade_date"])
    dates = candidate_dates.intersection(regsim_dates).sort_values()
    if len(dates) < 30:
        raise Phase2ValidationError(f"too few R88/Regsim common days for {scope}: {len(dates)}")
    candidate = inputs.candidate_returns.loc[dates]
    regsim = inputs.regsim.set_index("trade_date").loc[dates, "day_return"]
    deltas = candidate.sub(regsim.astype(float), axis=0)
    deltas.index = dates
    return dates, deltas


def _mcs_input_returns(inputs: FrozenInputs, scope: str, *, include_regsim: bool) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    candidate_dates = _scope_dates(inputs.expected_days, scope)
    if include_regsim:
        dates = candidate_dates.intersection(pd.DatetimeIndex(inputs.regsim["trade_date"])).sort_values()
        result = inputs.candidate_returns.loc[dates].copy()
        result["Regsim"] = inputs.regsim.set_index("trade_date").loc[dates, "day_return"].to_numpy(dtype=float)
        return dates, result
    return candidate_dates, inputs.candidate_returns.loc[candidate_dates].copy()


def _read_phase1_scorecard(phase1_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorecard = pd.read_csv(phase1_root / "single_model_scorecard.csv")
    blocks = pd.read_csv(phase1_root / "time_block_metrics.csv")
    required_score = {"candidate", "scope", "sharpe", "cumulative_return", "max_drawdown", "daily_var_95", "daily_cvar_95", "beta", "alpha_daily", "alpha_hac_t", "alpha_hac_p_two_sided", "residual_daily_volatility", "r_squared"}
    if not required_score.issubset(scorecard.columns):
        raise Phase2ValidationError(f"Phase-1 scorecard missing columns: {sorted(required_score - set(scorecard.columns))}")
    if len(scorecard) != TRIAL_COUNT * len(SCOPE_NAMES) or len(blocks) != TRIAL_COUNT * len(SCOPE_NAMES) * 5:
        raise Phase2ValidationError("Phase-1 risk source row counts are inconsistent")
    return scorecard, blocks


def _psr_dsr_and_correlations(
    inputs: FrozenInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    """Calculate every candidate's PSR/DSR and return-correlation diagnostics."""

    score_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    matrices: dict[str, pd.DataFrame] = {}
    family_summary: dict[str, Any] = {}
    for scope in SCOPE_NAMES:
        dates = _scope_dates(inputs.expected_days, scope)
        scoped = inputs.candidate_returns.loc[dates]
        matrix = scoped.to_numpy(dtype=float)
        daily_sharpes = np.asarray([_daily_sharpe(matrix[:, index]) for index in range(matrix.shape[1])])
        if not np.isfinite(daily_sharpes).all():
            raise Phase2ValidationError(f"non-finite trial daily Sharpe in {scope}")
        corr = scoped.corr(method="pearson")
        matrices[scope] = corr
        corr_values = corr.to_numpy(dtype=float)
        off_diagonal = corr_values[~np.eye(TRIAL_COUNT, dtype=bool)]
        eigenvalues = np.linalg.eigvalsh(corr_values)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        participation_rank = float((np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)) if np.sum(eigenvalues**2) > 0 else float("nan")
        family_summary[scope] = {
            "n_obs": int(len(scoped)),
            "candidate_count": TRIAL_COUNT,
            "daily_sharpe_cross_candidate_variance": float(np.var(daily_sharpes, ddof=1)),
            "return_correlation_mean_off_diagonal": float(np.mean(off_diagonal)),
            "return_correlation_median_off_diagonal": float(np.median(off_diagonal)),
            "return_correlation_min_off_diagonal": float(np.min(off_diagonal)),
            "return_correlation_max_off_diagonal": float(np.max(off_diagonal)),
            "return_correlation_participation_rank": participation_rank,
            "dsr_nominal_trial_count": TRIAL_COUNT,
            "note": "DSR uses the fixed nominal trial count 30. The correlation participation rank is a descriptive sensitivity only, not a substituted trial count.",
        }
        for index, candidate in enumerate(scoped.columns):
            values = matrix[:, index]
            psr = _psr(values)
            dsr = _dsr(values, trial_daily_sharpes=daily_sharpes)
            related = corr.loc[candidate].drop(candidate).to_numpy(dtype=float)
            score_rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    **psr,
                    **{key: value for key, value in dsr.items() if key not in psr},
                    "dsr_interpretation": "formal only for the predeclared 2025 development winner; all other rows are family-conditional diagnostics",
                }
            )
            corr_rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "mean_return_correlation_to_others": float(np.mean(related)),
                    "median_return_correlation_to_others": float(np.median(related)),
                    "min_return_correlation_to_others": float(np.min(related)),
                    "max_return_correlation_to_others": float(np.max(related)),
                    "family_return_correlation_participation_rank": participation_rank,
                }
            )
    return (
        pd.DataFrame(score_rows).sort_values(["scope", "candidate"]).reset_index(drop=True),
        pd.DataFrame(corr_rows).sort_values(["scope", "candidate"]).reset_index(drop=True),
        matrices,
        family_summary,
    )


def _render_results(
    diagnostics: pd.DataFrame,
    *,
    pbo: Mapping[str, Any],
    relative_family: Mapping[str, Mapping[str, Any]],
    mcs_with_regsim: Mapping[str, Mapping[str, Any]],
    family_correlation: Mapping[str, Mapping[str, Any]],
    frozen_development_winner: str,
) -> str:
    reporting = diagnostics.loc[diagnostics["scope"] == "reporting_2026"].copy()
    reporting = reporting.sort_values("annualized_sharpe", ascending=False).head(8)
    development = diagnostics.loc[diagnostics["scope"] == "development_2025"].copy()
    development_winner = development.loc[development["candidate"] == frozen_development_winner]
    if len(development_winner) != 1:
        raise Phase2ValidationError("frozen development winner is missing from diagnostics")
    development_winner = development_winner.iloc[0]
    lines = [
        "# R88 单模型统计验证 Phase 2",
        "",
        "## 范围",
        "",
        "- 只读分析冻结 R88 30 候选的既有 rolling OOS 日收益；未重新训练、打分、计算因子或调用实盘。",
        "- PBO/CSCV 与 development DSR 只使用 2025；2026 是 reporting-only，不参与重新选模。",
        "- Regsim 相对比较只使用共同日期上的原始日收益差；不将两套冻结 benchmark 混为同一 alpha 比较。",
        "",
        "## 开发期选择过拟合诊断",
        "",
        f"- CSCV：{pbo['chronological_block_count']} 个连续块，{pbo['directed_split_count']} 个定向 IS/OOS split。",
        f"- 冻结复合目标 PBO：{pbo['pbo']:.2%}；被 IS 选中者的 OOS 中位排名：{pbo['selected_oos_median_rank']:.1f}/{TRIAL_COUNT}。",
        f"- 2025 冻结复合目标 winner：`{development_winner['candidate']}`；其 family-conditional DSR(30)={development_winner['dsr_probability']:.3f}。",
        "",
        "## 2026 reporting 代表候选",
        "",
        "| candidate | Sharpe | HAC alpha t | MDD | PSR(SR>0) | DSR(30, diagnostic) | Mean raw-return delta vs Regsim bp/day |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in reporting.itertuples(index=False):
        relative = row.relative_mean_bp_per_day if np.isfinite(row.relative_mean_bp_per_day) else float("nan")
        lines.append(
            f"| {row.candidate} | {row.annualized_sharpe:.3f} | {row.alpha_hac_t:.3f} | {row.max_drawdown:.2%} | {row.psr_probability:.3f} | {row.dsr_probability:.3f} | {relative:.2f} |"
        )
    lines.extend(["", "## 家族级相对 Regsim 检验", ""])
    for scope in ("reporting_2026", "development_2025", "overall"):
        family = relative_family[scope]
        rc = family["white_reality_check"]
        spa = family["hansen_spa"]
        lines.append(
            f"- {scope}：White RC p={rc['p_value']:.4f}；Hansen SPA consistent p={spa['p_value_consistent']:.4f}；"
            f"最佳均值候选为 `{rc['best_mean_candidate']}`（{rc['best_mean_bp_per_day']:.2f} bp/day）。"
        )
    lines.extend(["", "## MCS（loss = -raw day_return）", ""])
    for scope in ("reporting_2026", "development_2025", "overall"):
        summary = mcs_with_regsim[scope]
        included = ", ".join(summary["included_candidates"])
        lines.append(f"- {scope}：31 模型共同样本的 MCS 留存 {summary['included_count']} 个：{included}。")
    lines.extend(["", "## 候选相关性与解释边界", ""])
    reporting_corr = family_correlation["reporting_2026"]
    lines.append(
        f"- 2026 候选日收益平均两两相关={reporting_corr['return_correlation_mean_off_diagonal']:.3f}，"
        f"participation rank={reporting_corr['return_correlation_participation_rank']:.2f}/30。"
    )
    lines.extend([
        "- PBO、PSR/DSR、RC/SPA、MCS 只校正冻结 R88 30 候选的条件性比较，不能消除更早因子筛选及研究过程的多重尝试。",
        "- RC/SPA 是相对 Regsim 的家族级 raw-return 检验，不是单候选 alpha 显著性，也不是实盘准入结论。",
        "- MCS 留存表示无法拒绝相同预期 raw-return loss；不等价于 Sharpe 最优、稳定盈利或可上线。",
        "",
    ])
    return "\n".join(lines)


def _original_development_winner(study_root: Path) -> str:
    ranking = pd.read_csv(study_root / "metrics" / "candidate_ranking.csv")
    if len(ranking) != TRIAL_COUNT or "candidate" not in ranking.columns:
        raise Phase2ValidationError("frozen candidate ranking is malformed")
    return str(ranking.iloc[0]["candidate"])


def _build_candidate_diagnostics(
    *,
    risk: pd.DataFrame,
    psr_dsr: pd.DataFrame,
    correlation: pd.DataFrame,
    pbo_candidate: pd.DataFrame,
    relative: pd.DataFrame,
    mcs_with_regsim: pd.DataFrame,
) -> pd.DataFrame:
    """One row per candidate/scope with all candidate-level Phase-2 diagnostics."""

    pbo = pbo_candidate.copy()
    pbo["scope"] = "development_2025"
    merged = risk.merge(psr_dsr, on=["candidate", "scope"], how="inner", validate="one_to_one")
    merged = merged.merge(correlation, on=["candidate", "scope"], how="inner", validate="one_to_one")
    merged = merged.merge(relative, on=["candidate", "scope"], how="left", validate="one_to_one")
    merged = merged.merge(
        mcs_with_regsim,
        on=["candidate", "scope"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_mcs"),
    )
    merged = merged.merge(pbo, on=["candidate", "scope"], how="left", validate="one_to_one")
    return merged.sort_values(["scope", "annualized_sharpe", "candidate"], ascending=[True, False, True]).reset_index(drop=True)


def _compact_candidate_scorecard(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Readable 30-row cross-period summary without creating a new ranking."""

    dev = diagnostics.loc[diagnostics["scope"] == "development_2025"].copy().set_index("candidate")
    report = diagnostics.loc[diagnostics["scope"] == "reporting_2026"].copy().set_index("candidate")
    overall = diagnostics.loc[diagnostics["scope"] == "overall"].copy().set_index("candidate")
    expected = set(dev.index)
    if set(report.index) != expected or set(overall.index) != expected or len(expected) != TRIAL_COUNT:
        raise Phase2ValidationError("candidate compact scorecard requires exactly aligned 30-candidate scopes")
    rows: list[dict[str, object]] = []
    for candidate in sorted(expected):
        d = dev.loc[candidate]
        r = report.loc[candidate]
        o = overall.loc[candidate]
        rows.append(
            {
                "candidate": candidate,
                "dev_sharpe": d["annualized_sharpe"],
                "dev_alpha_hac_t": d["alpha_hac_t"],
                "dev_psr_probability": d["psr_probability"],
                "dev_dsr_probability_conditional_30": d["dsr_probability"],
                "dev_cscv_selected_frequency": d["cscv_selected_frequency"],
                "dev_cscv_selected_oos_median_rank": d["cscv_selected_oos_median_rank"],
                "dev_mcs_r88_plus_regsim_included": d["mcs_included"],
                "report_sharpe": r["annualized_sharpe"],
                "report_alpha_hac_t": r["alpha_hac_t"],
                "report_alpha_hac_p_two_sided": r["alpha_hac_p_two_sided"],
                "report_beta": r["beta"],
                "report_residual_annualized_volatility": r["residual_annualized_volatility"],
                "report_max_drawdown": r["max_drawdown"],
                "report_block_sharpe_min": r["block_sharpe_min"],
                "report_rolling60_beta_std": r["rolling60_beta_std"],
                "report_rolling60_alpha_t_positive_fraction": r["rolling60_alpha_t_positive_fraction"],
                "report_psr_probability": r["psr_probability"],
                "report_dsr_probability_diagnostic_30": r["dsr_probability"],
                "report_regsim_raw_delta_mean_bp_per_day": r["relative_mean_bp_per_day"],
                "report_regsim_delta_hac_t": r["relative_hac_t"],
                "report_mcs_r88_plus_regsim_included": r["mcs_included"],
                "overall_sharpe": o["annualized_sharpe"],
                "overall_regsim_raw_delta_mean_bp_per_day": o["relative_mean_bp_per_day"],
                "overall_mcs_r88_plus_regsim_included": o["mcs_included"],
            }
        )
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def _render_candidate_full_table(compact: pd.DataFrame) -> str:
    lines = [
        "# R88 30 个候选 Phase 2 全量汇总",
        "",
        "该表逐候选列出冻结 2025 选择诊断、2026 reporting 表现、风险及相对 Regsim 原始日收益。",
        "PBO/RC/SPA 是家族级检验，不应把本表中的单候选 HAC 值解释成经多重比较校正后的显著性。",
        "",
        "| candidate | Dev Sharpe | Dev DSR(30) | CSCV selected freq | Report Sharpe | Report α t | Report MDD | Δ Regsim bp/day | Δ HAC t | MCS+Regsim |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in compact.itertuples(index=False):
        selected_freq = "—" if not np.isfinite(row.dev_cscv_selected_frequency) else f"{row.dev_cscv_selected_frequency:.1%}"
        lines.append(
            f"| {row.candidate} | {row.dev_sharpe:.3f} | {row.dev_dsr_probability_conditional_30:.3f} | "
            f"{selected_freq} | {row.report_sharpe:.3f} | {row.report_alpha_hac_t:.3f} | "
            f"{row.report_max_drawdown:.2%} | {row.report_regsim_raw_delta_mean_bp_per_day:.2f} | "
            f"{row.report_regsim_delta_hac_t:.3f} | {'yes' if row.report_mcs_r88_plus_regsim_included else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def run_validation(
    *,
    study_root: Path = DEFAULT_STUDY_ROOT,
    phase1_root: Path = DEFAULT_PHASE1_ROOT,
    regsim_path: Path = DEFAULT_REGSIM_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_OUTPUT_RUN,
    bootstrap_reps: int = BOOTSTRAP_REPS,
    mcs_bootstrap_reps: int = MCS_BOOTSTRAP_REPS,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
) -> Path:
    """Run the full frozen-family Phase-2 analysis in an isolated root."""

    if bootstrap_reps <= 0 or mcs_bootstrap_reps <= 0 or block_length <= 0:
        raise ValueError("bootstrap repetitions and block length must be positive")
    run_root = _make_run_root(output_root, run_name)
    inputs = load_frozen_inputs(
        study_root=study_root,
        phase1_root=phase1_root,
        regsim_path=regsim_path,
    )
    scorecard, phase1_blocks = _read_phase1_scorecard(inputs.phase1_root)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    _write_json(
        run_root / "run_status.json",
        {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "factor_store_writes": False,
            "model_scoring_called": False,
            "model_training_called": False,
        },
    )
    try:
        dev_dates = _scope_dates(inputs.expected_days, "development_2025")
        pbo_summary, pbo_splits, pbo_candidates, pbo_ranks = _pbo_cscv(
            inputs.candidate_returns.loc[dev_dates], inputs.benchmark_returns.loc[dev_dates]
        )
        psr_dsr, correlation, correlation_matrices, correlation_summary = _psr_dsr_and_correlations(inputs)
        risk = _risk_stability_rows(inputs, scorecard=scorecard, block_metrics=phase1_blocks)

        relative_family: dict[str, dict[str, Any]] = {}
        relative_rows: list[pd.DataFrame] = []
        relative_date_contract: dict[str, Any] = {}
        for offset, scope in enumerate(SCOPE_NAMES):
            common_days, deltas = _aligned_relative_returns(inputs, scope)
            family, rows = _family_relative_tests(
                deltas,
                scope=scope,
                reps=bootstrap_reps,
                block_length=block_length,
                seed=RANDOM_SEED + 100 + offset,
            )
            relative_family[scope] = family
            relative_rows.append(rows)
            relative_date_contract[scope] = {
                "common_days": int(len(common_days)),
                "start": str(common_days.min().date()),
                "end": str(common_days.max().date()),
                "date_sha256": hashlib.sha256(
                    ",".join(str(day.date()) for day in common_days).encode("utf-8")
                ).hexdigest(),
            }
        relative = pd.concat(relative_rows, ignore_index=True)

        mcs_internal_summary: dict[str, dict[str, Any]] = {}
        mcs_regsim_summary: dict[str, dict[str, Any]] = {}
        mcs_internal_rows: list[pd.DataFrame] = []
        mcs_regsim_rows: list[pd.DataFrame] = []
        mcs_trace_rows: list[pd.DataFrame] = []
        for offset, scope in enumerate(SCOPE_NAMES):
            _, internal_returns = _mcs_input_returns(inputs, scope, include_regsim=False)
            summary, membership, trace = _mcs_range(
                internal_returns,
                scope=scope,
                reps=mcs_bootstrap_reps,
                block_length=block_length,
                alpha=MCS_ALPHA,
                seed=RANDOM_SEED + 1_000 + offset,
            )
            summary["population"] = "r88_30_only"
            membership["population"] = "r88_30_only"
            trace["population"] = "r88_30_only"
            mcs_internal_summary[scope] = summary
            mcs_internal_rows.append(membership)
            mcs_trace_rows.append(trace)

            _, regsim_returns = _mcs_input_returns(inputs, scope, include_regsim=True)
            summary, membership, trace = _mcs_range(
                regsim_returns,
                scope=scope,
                reps=mcs_bootstrap_reps,
                block_length=block_length,
                alpha=MCS_ALPHA,
                seed=RANDOM_SEED + 2_000 + offset,
            )
            summary["population"] = "r88_30_plus_regsim"
            membership["population"] = "r88_30_plus_regsim"
            trace["population"] = "r88_30_plus_regsim"
            mcs_regsim_summary[scope] = summary
            mcs_regsim_rows.append(membership)
            mcs_trace_rows.append(trace)

        mcs_internal = pd.concat(mcs_internal_rows, ignore_index=True)
        mcs_regsim = pd.concat(mcs_regsim_rows, ignore_index=True)
        mcs_trace = pd.concat(mcs_trace_rows, ignore_index=True)
        candidate_mcs = mcs_regsim.loc[mcs_regsim["candidate"] != "Regsim"].copy()
        diagnostics = _build_candidate_diagnostics(
            risk=risk,
            psr_dsr=psr_dsr,
            correlation=correlation,
            pbo_candidate=pbo_candidates,
            relative=relative,
            mcs_with_regsim=candidate_mcs,
        )
        original_winner = _original_development_winner(inputs.study_root)
        diagnostics["is_frozen_development_winner"] = diagnostics["candidate"] == original_winner
        compact_scorecard = _compact_candidate_scorecard(diagnostics)

        input_evidence = {
            "schema": "r88_single_model_validation_phase2_input_evidence/v1",
            "study_id": STUDY_ID,
            "research_only": True,
            "promotion_allowed": False,
            "study_root": str(inputs.study_root),
            "phase1_root": str(inputs.phase1_root),
            "study_plan_sha256": _sha256_file(inputs.study_root / "study_plan.json"),
            "study_integrity_sha256": _sha256_file(inputs.study_root / "study_integrity.json"),
            "study_integrity_verification": inputs.integrity_verification,
            "candidate_count": len(inputs.sources),
            "candidate_calendar": {
                "days": int(len(inputs.expected_days)),
                "start": str(inputs.expected_days.min().date()),
                "end": str(inputs.expected_days.max().date()),
                "date_sha256": hashlib.sha256(
                    ",".join(str(day.date()) for day in inputs.expected_days).encode("utf-8")
                ).hexdigest(),
            },
            "regsim": {
                "path": str(inputs.regsim_path),
                "sha256": _sha256_file(inputs.regsim_path),
                "common_date_contract": relative_date_contract,
            },
            "phase1_input_evidence_sha256": _sha256_file(inputs.phase1_root / "input_evidence.json"),
            "phase1_scorecard_sha256": _sha256_file(inputs.phase1_root / "single_model_scorecard.csv"),
            "factor_generation": inputs.phase1_evidence.get("factor_generation"),
            "reused": [
                "frozen R88 daily_returns",
                "frozen R88 coverage and warm_start coverage",
                "Phase-1 r3 scorecard and input evidence",
                "frozen Regsim return history",
            ],
            "missing": [],
            "invalid": [],
        }
        method_contract = {
            "schema": "r88_single_model_validation_phase2_method_contract/v1",
            "trial_family": {
                "candidate_count": TRIAL_COUNT,
                "membership": [source.candidate for source in inputs.sources],
                "frozen_development_winner": original_winner,
                "selection_objective": "0.6 * minmax(annualized Sharpe) + 0.4 * minmax(HAC alpha t)",
            },
            "scopes": {
                "development_2025": "2025 only; PBO/CSCV and development DSR interpretation",
                "reporting_2026": "reporting-only; primary relative family tests and MCS result reporting",
                "overall": "descriptive only; never used to choose a replacement",
            },
            "pbo_cscv": {
                **pbo_summary,
                "out_of_sample_scaling": "apply the split's in-sample min/max scales; no OOS clipping",
            },
            "psr_dsr": {
                "psr": "daily Sharpe, target SR=0, skew/kurtosis finite-sample approximation",
                "dsr": "nominal 30-trial deflation using frozen family cross-candidate daily-Sharpe variance; formal interpretation only for frozen 2025 winner",
                "not_corrected": "factor-screen and researcher-wide trial history outside the frozen 30 candidates",
            },
            "relative_family_tests": {
                "return_definition": "raw R88 day_return - raw Regsim day_return on exact common dates",
                "white_reality_check": "centered synchronous circular moving-block maximum mean",
                "hansen_spa": "Bartlett-HAC studentized maximum with lower/consistent/upper recentering; consistent p is primary",
            },
            "mcs": {
                "loss": "-raw day_return",
                "method": "block-bootstrap variance range-statistic MCS approximation",
                "populations": ["R88 30 only", "R88 30 plus Regsim on common dates"],
                "alpha": MCS_ALPHA,
            },
            "bootstrap": {
                "method": "synchronous circular moving-block bootstrap",
                "block_length_days": block_length,
                "family_test_repetitions": bootstrap_reps,
                "mcs_repetitions": mcs_bootstrap_reps,
                "seed": RANDOM_SEED,
            },
            "exclusions": [
                "factor rebuild",
                "model training",
                "model scoring",
                "live runtime",
                "database writes",
                "scheduler operations",
                "live admission",
            ],
        }

        pbo_splits.to_csv(run_root / "pbo_cscv_splits.csv", index=False)
        pbo_ranks.to_csv(run_root / "pbo_cscv_candidate_ranks.csv", index=False)
        pbo_candidates.to_csv(run_root / "pbo_cscv_candidate_summary.csv", index=False)
        psr_dsr.to_csv(run_root / "psr_dsr_scorecard.csv", index=False)
        correlation.to_csv(run_root / "candidate_return_correlation_summary.csv", index=False)
        for scope, matrix in correlation_matrices.items():
            matrix.to_csv(run_root / f"candidate_return_correlation_{scope}.csv")
        risk.to_csv(run_root / "beta_alpha_risk_stability.csv", index=False)
        relative.to_csv(run_root / "relative_regsim_candidate_diagnostics.csv", index=False)
        mcs_internal.to_csv(run_root / "mcs_membership_r88_30.csv", index=False)
        mcs_regsim.to_csv(run_root / "mcs_membership_r88_plus_regsim.csv", index=False)
        mcs_trace.to_csv(run_root / "mcs_elimination_trace.csv", index=False)
        diagnostics.to_csv(run_root / "candidate_phase2_diagnostics.csv", index=False)
        compact_scorecard.to_csv(run_root / "candidate_phase2_compact_scorecard.csv", index=False)
        _write_json(run_root / "pbo_cscv_family_summary.json", pbo_summary)
        _write_json(run_root / "relative_regsim_family_tests.json", relative_family)
        _write_json(run_root / "mcs_summary_r88_30.json", mcs_internal_summary)
        _write_json(run_root / "mcs_summary_r88_plus_regsim.json", mcs_regsim_summary)
        _write_json(run_root / "candidate_correlation_family_summary.json", correlation_summary)
        _write_json(run_root / "input_evidence.json", input_evidence)
        _write_json(run_root / "method_contract.json", method_contract)
        (run_root / "RESULTS.md").write_text(
            _render_results(
                diagnostics,
                pbo=pbo_summary,
                relative_family=relative_family,
                mcs_with_regsim=mcs_regsim_summary,
                family_correlation=correlation_summary,
                frozen_development_winner=original_winner,
            ),
            encoding=RESULTS_ENCODING,
        )
        (run_root / "CANDIDATE_RESULTS.md").write_text(
            _render_candidate_full_table(compact_scorecard),
            encoding=RESULTS_ENCODING,
        )
        manifest = {
            "schema": "r88_single_model_validation_phase2_run/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "promotion_allowed": False,
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "factor_store_writes": False,
            "model_scoring_called": False,
            "model_training_called": False,
            "scope": "frozen daily OOS return statistical validation only",
            "input_evidence": input_evidence,
            "method_contract": method_contract,
            "git": {
                "head": _safe_git(["rev-parse", "HEAD"]),
                "status_porcelain": _safe_git(["status", "--short"]),
            },
        }
        _write_json(run_root / "run_manifest.json", manifest)
        _write_json(
            run_root / "run_status.json",
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_root": str(run_root),
                "candidate_count": TRIAL_COUNT,
                "research_only": True,
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
                "factor_store_writes": False,
                "model_scoring_called": False,
                "model_training_called": False,
            },
        )
    except Exception as exc:
        _write_json(
            run_root / "run_status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "research_only": True,
                "database_writes": False,
                "live_runtime_called": False,
                "scheduler_called": False,
            },
        )
        raise
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--regsim-path", type=Path, default=DEFAULT_REGSIM_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--mcs-bootstrap-reps", type=int, default=MCS_BOOTSTRAP_REPS)
    parser.add_argument("--block-length", type=int, default=BOOTSTRAP_BLOCK_LENGTH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = run_validation(
        study_root=args.study_root,
        phase1_root=args.phase1_root,
        regsim_path=args.regsim_path,
        output_root=args.output_root,
        run_name=args.run_name,
        bootstrap_reps=args.bootstrap_reps,
        mcs_bootstrap_reps=args.mcs_bootstrap_reps,
        block_length=args.block_length,
    )
    print(f"R88 Phase-2 validation completed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
