"""Research-only phase-1 single-model validation for the completed R88 study.

The tool deliberately *does not* score, train, backfill factors, invoke the
live runtime, or modify the R88 study.  It consumes the immutable daily OOS
return artifacts of all completed R88 candidates and creates an isolated
scorecard below ``D:/cbond_on/research_scratch``.

Phase 1 covers strategy-level stability only:

* full/development/reporting return, risk, beta and HAC-alpha metrics;
* rolling 20/60/120-day diagnostics and five chronological blocks;
* moving-block bootstrap confidence intervals for each fixed candidate;
* paired raw-return diagnostics relative to the frozen static Regsim history.

It explicitly does not implement PBO/CSCV, DSR/PSR, SPA/Reality Check/MCS,
or parameter/factor perturbations.  Those are a separately scoped phase.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm


STUDY_ID = "r88_joint_factor_lgbm_20260828_r1"
DEFAULT_STUDY_ROOT = Path(
    r"D:/cbond_on/research_scratch/r88_joint_factor_lgbm_20260828_r1/"
    r"runs/rolling_20250102_20260827"
)
DEFAULT_REGSIM_PATH = Path(
    r"D:/cbond_on/results/analysis/model_switch_scoreopt_live_20260805_50f/"
    r"return_history/Challenger_Regsim.csv"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1")
DEV_START = pd.Timestamp("2025-01-02")
DEV_END = pd.Timestamp("2025-12-31")
REPORT_START = pd.Timestamp("2026-01-01")
ANNUALIZATION = 252.0
ROLLING_WINDOWS = (20, 60, 120)
BLOCK_COUNT = 5
BOOTSTRAP_BLOCK_LENGTH = 10
BOOTSTRAP_REPS = 2_000
RANDOM_SEED = 20_260_901
BENCHMARK_METHOD = "strict_official_prev_close_split"
EXPECTED_IMMUTABLE_HASH_COUNT = 129
RESULTS_ENCODING = "utf-8-sig"


class ValidationError(RuntimeError):
    """Raised when a frozen input contract is incomplete or inconsistent."""


@dataclass(frozen=True)
class CandidateSource:
    candidate: str
    factor_mode: str
    factor_set_id: str
    hyperparameter_id: str
    metrics_path: Path
    daily_returns_path: Path
    coverage_path: Path
    backtest_coverage_path: Path
    warm_start_coverage_path: Path


def _json_default(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValidationError(f"{label} missing required columns: {missing}")


def _resolve_inside(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationError(f"{label} must be under {root}: {resolved}") from exc
    return resolved


def _make_run_root(output_root: Path, run_name: str) -> Path:
    output_root = _resolve_inside(output_root, DEFAULT_OUTPUT_ROOT, label="output root")
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValidationError(f"run name must be a single path component: {run_name!r}")
    run_root = _resolve_inside(output_root / run_name, DEFAULT_OUTPUT_ROOT, label="run root")
    if run_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise ValidationError("a fresh child run directory is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing research run: {run_root}")
    return run_root


def _normalise_daily(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = ("trade_date", "day_return", "benchmark_return", "benchmark_method")
    _require_columns(frame, required, label=label)
    out = frame.loc[:, required].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    for column in ("day_return", "benchmark_return"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if out["trade_date"].isna().any():
        raise ValidationError(f"{label} has invalid trade_date")
    if out["trade_date"].duplicated().any():
        raise ValidationError(f"{label} has duplicate trade_date")
    if not np.isfinite(out[["day_return", "benchmark_return"]].to_numpy(dtype=float)).all():
        raise ValidationError(f"{label} has non-finite return values")
    methods = sorted(set(out["benchmark_method"].astype(str)))
    if methods != [BENCHMARK_METHOD]:
        raise ValidationError(f"{label} benchmark methods differ from contract: {methods}")
    return out.sort_values("trade_date").reset_index(drop=True)


def _semantic_daily_hash(daily: pd.DataFrame) -> str:
    work = daily[["trade_date", "day_return", "benchmark_return"]].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"]).dt.date
    return hashlib.sha256(work.to_csv(index=False).encode("utf-8")).hexdigest()


def _verify_frozen_study_integrity(
    study_root: Path,
    plan: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-verify the immutable R88 study files through its canonical verifier.

    The phase-1 validator is intentionally a consumer of an already-frozen
    study.  Recomputing the declared file hashes here closes the gap between
    merely recording ``study_integrity.json`` and proving that the 129 files it
    names are still unchanged.  The R88 runner is imported lazily so this
    read-only audit does not become a normal model/factor runtime path.
    """

    if str(integrity.get("study_id", "")) != STUDY_ID:
        raise ValidationError("R88 integrity manifest has an unexpected study identity")

    try:
        from harness.tools import run_r88_joint_factor_lgbm_study as r88_runner

        # The frozen R88 run predates the runner's current default study id.
        # Bind only for the duration of the verifier call, then restore every
        # mutated module global so importing this audit tool has no process-wide
        # configuration side effect.
        global_names = (
            "STUDY_ID",
            "DEFAULT_STUDY_ROOT",
            "FACTOR_ROOT",
            "BACKFILL_MANIFEST",
            "R88_INPUT_MODE",
            "HOLDOUT_END",
            "DEFAULT_END",
            "_R88_BINDING",
        )
        previous = {name: getattr(r88_runner, name) for name in global_names}
        try:
            r88_runner._configure_runtime(
                factor_root_text=None,
                manifest_text=None,
                study_id_text=STUDY_ID,
            )
            verified = r88_runner._verify_study_integrity(Path(study_root), dict(plan))
        finally:
            for name, value in previous.items():
                setattr(r88_runner, name, value)
    except (OSError, RuntimeError, KeyError, ValueError, TypeError) as exc:
        raise ValidationError(f"R88 immutable-input integrity failed: {exc}") from exc

    hashes = verified.get("files_sha256") if isinstance(verified, Mapping) else None
    if not isinstance(hashes, Mapping) or len(hashes) != EXPECTED_IMMUTABLE_HASH_COUNT:
        actual = len(hashes) if isinstance(hashes, Mapping) else 0
        raise ValidationError(
            "unexpected immutable R88 hash count: "
            f"expected {EXPECTED_IMMUTABLE_HASH_COUNT}, got {actual}"
        )
    return {
        "status": "passed",
        "verifier": "harness.tools.run_r88_joint_factor_lgbm_study._verify_study_integrity",
        "hash_count": len(hashes),
    }


def _load_candidate_sources(
    study_root: Path,
) -> tuple[list[CandidateSource], dict[str, Any], dict[str, Any], dict[str, Any]]:
    study_root = study_root.resolve()
    plan = _read_json(study_root / "study_plan.json")
    integrity = _read_json(study_root / "study_integrity.json")
    if plan.get("study_id") != STUDY_ID:
        raise ValidationError(f"unexpected study id: {plan.get('study_id')!r}")
    integrity_verification = _verify_frozen_study_integrity(study_root, plan, integrity)
    if not bool(plan.get("research_only")) or bool(plan.get("evidence", {}).get("promotion_allowed", True)):
        raise ValidationError("source study must remain research-only and promotion-disabled")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 30:
        raise ValidationError("expected the complete fixed 30-candidate R88 study")
    sources: list[CandidateSource] = []
    for spec in candidates:
        if not isinstance(spec, dict):
            raise ValidationError("malformed candidate spec")
        name = str(spec.get("candidate", "")).strip()
        if not name:
            raise ValidationError("candidate name is blank")
        metrics_path = study_root / "metrics" / f"{name}.json"
        metrics = _read_json(metrics_path)
        if metrics.get("candidate") != name:
            raise ValidationError(f"metrics candidate mismatch: {name}")
        if str(metrics.get("factor_mode", "")) != str(spec.get("factor_mode", "")):
            raise ValidationError(f"metrics factor_mode mismatch: {name}")
        manifest_path = study_root / "candidate_manifests" / f"{name}.json"
        candidate_manifest = _read_json(manifest_path)
        if str(candidate_manifest.get("factor_set_id", "")) != str(spec.get("factor_set_id", "")):
            raise ValidationError(f"candidate manifest factor_set_id mismatch: {name}")
        if str(candidate_manifest.get("hyperparameter_id", "")) != str(spec.get("hyperparameter_id", "")):
            raise ValidationError(f"candidate manifest hyperparameter_id mismatch: {name}")
        daily_path = Path(str(metrics.get("daily_returns_path", "")))
        _resolve_inside(daily_path, study_root, label=f"{name} daily returns")
        if not daily_path.is_file():
            raise FileNotFoundError(f"missing daily returns for {name}: {daily_path}")
        coverage = study_root / "coverage" / f"{name}.json"
        backtest_coverage = study_root / "backtest_coverage" / f"{name}.json"
        warm_start = study_root / "warm_start_coverage" / f"{name}.json"
        for path in (coverage, backtest_coverage, warm_start):
            if not path.is_file():
                raise FileNotFoundError(path)
        coverage_doc = _read_json(coverage)
        backtest_doc = _read_json(backtest_coverage)
        warm_doc = _read_json(warm_start)
        if coverage_doc.get("missing_expected_score_days") or coverage_doc.get("unexpected_score_days") or coverage_doc.get("invalid_score_days"):
            raise ValidationError(f"score coverage failed for {name}")
        if backtest_doc.get("missing_execution_days") or backtest_doc.get("unexpected_return_days") or backtest_doc.get("duplicate_return_days") or int(backtest_doc.get("nonfinite_return_rows", 0)) or backtest_doc.get("non_ok_diagnostics"):
            raise ValidationError(f"backtest coverage failed for {name}")
        if warm_doc.get("errors") or str(warm_doc.get("status", "")).lower() not in {"ok", "completed"}:
            raise ValidationError(f"warm-start coverage failed for {name}")
        sources.append(
            CandidateSource(
                candidate=name,
                factor_mode=str(spec.get("factor_mode", "")),
                factor_set_id=str(spec.get("factor_set_id", "")),
                hyperparameter_id=str(spec.get("hyperparameter_id", "")),
                metrics_path=metrics_path,
                daily_returns_path=daily_path,
                coverage_path=coverage,
                backtest_coverage_path=backtest_coverage,
                warm_start_coverage_path=warm_start,
            )
        )
    if len({source.candidate for source in sources}) != len(sources):
        raise ValidationError("candidate names are not unique")
    return sources, plan, integrity, integrity_verification


def _load_r88_returns(source: CandidateSource, expected_days: pd.DatetimeIndex) -> pd.DataFrame:
    daily = _normalise_daily(pd.read_csv(source.daily_returns_path), label=source.candidate)
    actual_days = pd.DatetimeIndex(daily["trade_date"])
    if not actual_days.equals(expected_days):
        raise ValidationError(f"{source.candidate} daily return dates do not equal the frozen score calendar")
    metrics = _read_json(source.metrics_path)
    expected_hash = str(metrics.get("overall", {}).get("daily_returns_sha256", ""))
    actual_hash = _semantic_daily_hash(daily)
    if expected_hash != actual_hash:
        raise ValidationError(f"{source.candidate} semantic daily hash mismatch")
    daily.insert(0, "candidate", source.candidate)
    return daily


def _load_regsim(path: Path) -> pd.DataFrame:
    return _normalise_daily(pd.read_csv(path), label="Regsim")


def _scope(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "development_2025":
        return frame.loc[(frame["trade_date"] >= DEV_START) & (frame["trade_date"] <= DEV_END)].copy()
    if name == "reporting_2026":
        return frame.loc[frame["trade_date"] >= REPORT_START].copy()
    if name == "overall":
        return frame.copy()
    raise ValueError(name)


def _max_drawdown(returns: np.ndarray) -> float:
    nav = np.cumprod(1.0 + returns)
    return float(np.min(nav / np.maximum.accumulate(nav) - 1.0))


def _ols_hac(strategy: np.ndarray, benchmark: np.ndarray) -> dict[str, float]:
    n_obs = len(strategy)
    x = np.column_stack([np.ones(n_obs, dtype=float), benchmark])
    coefficients = np.linalg.pinv(x.T @ x) @ (x.T @ strategy)
    residual = strategy - x @ coefficients
    if n_obs < 30:
        alpha_se = float("nan")
        alpha_t = float("nan")
        alpha_p = float("nan")
        lag = 0
    else:
        lag = max(0, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
        score = x * residual[:, None]
        omega = score.T @ score
        for offset in range(1, lag + 1):
            weight = 1.0 - offset / (lag + 1.0)
            cross = score[offset:].T @ score[:-offset]
            omega += weight * (cross + cross.T)
        bread = np.linalg.pinv(x.T @ x)
        covariance = bread @ omega @ bread
        alpha_se = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
        alpha_t = float(coefficients[0] / alpha_se) if alpha_se > 0.0 else float("nan")
        alpha_p = float(2.0 * norm.sf(abs(alpha_t))) if np.isfinite(alpha_t) else float("nan")
    total = float(np.sum((strategy - np.mean(strategy)) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0.0 else float("nan")
    return {
        "alpha_daily": float(coefficients[0]),
        "beta": float(coefficients[1]),
        "residual_daily_volatility": float(np.std(residual, ddof=1)) if n_obs > 1 else float("nan"),
        "r_squared": r_squared,
        "hac_lag": float(lag),
        "alpha_hac_se": alpha_se,
        "alpha_hac_t": alpha_t,
        "alpha_hac_p_two_sided": alpha_p,
    }


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if len(frame) < 2:
        raise ValidationError("need at least two daily return observations")
    strategy = frame["day_return"].to_numpy(dtype=float)
    benchmark = frame["benchmark_return"].to_numpy(dtype=float)
    mean = float(np.mean(strategy))
    std = float(np.std(strategy, ddof=1))
    nav = np.cumprod(1.0 + strategy)
    tail_count = max(1, int(math.ceil(len(strategy) * 0.05)))
    tail = np.sort(strategy)[:tail_count]
    week = frame.assign(period=frame["trade_date"].dt.to_period("W-FRI")).groupby("period")["day_return"].apply(lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0))
    month = frame.assign(period=frame["trade_date"].dt.to_period("M")).groupby("period")["day_return"].apply(lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0))
    result: dict[str, float] = {
        "n_obs": float(len(frame)),
        "cumulative_return": float(nav[-1] - 1.0),
        "annualized_return_simple": float(mean * ANNUALIZATION),
        "annualized_return_geometric": float(nav[-1] ** (ANNUALIZATION / len(frame)) - 1.0),
        "annualized_volatility": float(std * math.sqrt(ANNUALIZATION)),
        "sharpe": float(mean / std * math.sqrt(ANNUALIZATION)) if std > 0.0 else float("nan"),
        "max_drawdown": _max_drawdown(strategy),
        "daily_var_95": float(np.quantile(strategy, 0.05)),
        "daily_cvar_95": float(np.mean(tail)),
        "daily_win_fraction": float(np.mean(strategy > 0.0)),
        "weekly_positive_fraction": float(np.mean(week > 0.0)),
        "monthly_positive_fraction": float(np.mean(month > 0.0)),
    }
    result.update(_ols_hac(strategy, benchmark))
    return result


def _moving_block_indices(n_obs: int, *, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n_obs < block_length:
        return rng.integers(0, n_obs, size=n_obs)
    block_count = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs - block_length + 1, size=block_count)
    return np.concatenate([np.arange(start, start + block_length) for start in starts])[:n_obs]


def _moving_block_index_matrix(
    n_obs: int,
    *,
    block_length: int,
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a batch of moving-block bootstrap index draws."""

    if n_obs < block_length:
        return rng.integers(0, n_obs, size=(rows, n_obs))
    count = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs - block_length + 1, size=(rows, count))
    offsets = np.arange(block_length, dtype=int)
    return (starts[:, :, None] + offsets[None, None, :]).reshape(rows, -1)[:, :n_obs]


def moving_block_bootstrap(frame: pd.DataFrame, *, reps: int, block_length: int, seed: int) -> dict[str, float]:
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")
    strategy = frame["day_return"].to_numpy(dtype=float)
    benchmark = frame["benchmark_return"].to_numpy(dtype=float)
    n_obs = len(frame)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(reps, dtype=float)
    annual_returns = np.empty(reps, dtype=float)
    alpha_bp = np.empty(reps, dtype=float)
    cursor = 0
    while cursor < reps:
        size = min(500, reps - cursor)
        sampled = _moving_block_index_matrix(n_obs, block_length=block_length, rows=size, rng=rng)
        y = strategy[sampled]
        x = benchmark[sampled]
        mean_y = y.mean(axis=1)
        mean_x = x.mean(axis=1)
        centered_x = x - mean_x[:, None]
        centered_y = y - mean_y[:, None]
        denominator = np.sum(centered_x**2, axis=1)
        numerator = np.sum(centered_x * centered_y, axis=1)
        beta = np.divide(numerator, denominator, out=np.zeros(size, dtype=float), where=denominator > 0.0)
        std = y.std(axis=1, ddof=1)
        sharpes[cursor : cursor + size] = np.divide(
            mean_y * math.sqrt(ANNUALIZATION),
            std,
            out=np.full(size, np.nan, dtype=float),
            where=std > 0.0,
        )
        annual_returns[cursor : cursor + size] = mean_y * ANNUALIZATION
        alpha_bp[cursor : cursor + size] = (mean_y - beta * mean_x) * 10_000.0
        cursor += size
    def quantile(values: np.ndarray, level: float) -> float:
        valid = values[np.isfinite(values)]
        return float(np.quantile(valid, level)) if len(valid) else float("nan")
    return {
        "bootstrap_reps": float(reps),
        "bootstrap_block_length": float(block_length),
        "bootstrap_sharpe_q05": quantile(sharpes, 0.05),
        "bootstrap_sharpe_q50": quantile(sharpes, 0.50),
        "bootstrap_sharpe_q95": quantile(sharpes, 0.95),
        "bootstrap_sharpe_probability_positive": float(np.mean(sharpes[np.isfinite(sharpes)] > 0.0)),
        "bootstrap_annual_return_q05": quantile(annual_returns, 0.05),
        "bootstrap_annual_return_q50": quantile(annual_returns, 0.50),
        "bootstrap_annual_return_q95": quantile(annual_returns, 0.95),
        "bootstrap_alpha_bp_q05": quantile(alpha_bp, 0.05),
        "bootstrap_alpha_bp_q50": quantile(alpha_bp, 0.50),
        "bootstrap_alpha_bp_q95": quantile(alpha_bp, 0.95),
    }


def _rolling_metrics(candidate: str, frame: pd.DataFrame, window: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for end in range(window - 1, len(frame)):
        sub = frame.iloc[end - window + 1 : end + 1]
        metrics = calculate_metrics(sub)
        rows.append({
            "candidate": candidate,
            "scope": "development_2025" if sub["trade_date"].iloc[-1] <= DEV_END else "reporting_2026" if sub["trade_date"].iloc[0] >= REPORT_START else "cross_boundary",
            "window_days": window,
            "end_trade_date": sub["trade_date"].iloc[-1],
            "start_trade_date": sub["trade_date"].iloc[0],
            **metrics,
        })
    return rows


def _chronological_blocks(frame: pd.DataFrame, count: int = BLOCK_COUNT) -> list[pd.DataFrame]:
    return [frame.iloc[indexes].copy() for indexes in np.array_split(np.arange(len(frame)), count) if len(indexes)]


def _raw_delta_hac(candidate: pd.DataFrame, regsim: pd.DataFrame) -> dict[str, float]:
    paired = candidate[["trade_date", "day_return"]].merge(regsim[["trade_date", "day_return"]], on="trade_date", suffixes=("_candidate", "_regsim"), validate="one_to_one")
    delta = paired["day_return_candidate"].to_numpy(dtype=float) - paired["day_return_regsim"].to_numpy(dtype=float)
    n_obs = len(delta)
    if n_obs < 30:
        return {"paired_n_obs": float(n_obs), "paired_delta_mean_bp": float(np.mean(delta) * 10_000.0), "paired_delta_hac_t": float("nan"), "paired_delta_hac_p": float("nan")}
    lag = max(0, int(round(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))))
    centered = delta - np.mean(delta)
    omega = float(centered @ centered)
    for offset in range(1, lag + 1):
        weight = 1.0 - offset / (lag + 1.0)
        omega += weight * 2.0 * float(centered[offset:] @ centered[:-offset])
    standard_error = math.sqrt(max(omega / (n_obs**2), 0.0))
    t_value = float(np.mean(delta) / standard_error) if standard_error > 0.0 else float("nan")
    return {
        "paired_n_obs": float(n_obs),
        "paired_delta_mean_bp": float(np.mean(delta) * 10_000.0),
        "paired_delta_hac_lag": float(lag),
        "paired_delta_hac_t": t_value,
        "paired_delta_hac_p": float(2.0 * norm.sf(abs(t_value))) if np.isfinite(t_value) else float("nan"),
    }


def _paired_delta_bootstrap(
    candidate: pd.DataFrame,
    regsim: pd.DataFrame,
    *,
    reps: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    """Moving-block percentile intervals for the paired raw-return delta.

    The same sampled indices are applied to both series, preserving their
    contemporaneous dependence.  These are descriptive confidence intervals,
    not family-wise selection p-values; RC/SPA is deliberately phase 2.
    """

    paired = candidate[["trade_date", "day_return"]].merge(
        regsim[["trade_date", "day_return"]],
        on="trade_date",
        suffixes=("_candidate", "_regsim"),
        validate="one_to_one",
    )
    candidate_values = paired["day_return_candidate"].to_numpy(dtype=float)
    regsim_values = paired["day_return_regsim"].to_numpy(dtype=float)
    n_obs = len(paired)
    if n_obs < 2:
        return {
            "paired_bootstrap_reps": float(reps),
            "paired_bootstrap_block_length": float(block_length),
            "paired_delta_bootstrap_q05_bp": float("nan"),
            "paired_delta_bootstrap_q50_bp": float("nan"),
            "paired_delta_bootstrap_q95_bp": float("nan"),
            "paired_delta_bootstrap_probability_positive": float("nan"),
        }
    rng = np.random.default_rng(seed)
    delta_means = np.empty(reps, dtype=float)
    sharpe_differences = np.empty(reps, dtype=float)
    cursor = 0
    while cursor < reps:
        size = min(500, reps - cursor)
        sampled = _moving_block_index_matrix(n_obs, block_length=block_length, rows=size, rng=rng)
        first = candidate_values[sampled]
        second = regsim_values[sampled]
        delta_means[cursor : cursor + size] = (first - second).mean(axis=1) * 10_000.0
        first_std = first.std(axis=1, ddof=1)
        second_std = second.std(axis=1, ddof=1)
        first_sharpe = np.divide(
            first.mean(axis=1) * math.sqrt(ANNUALIZATION),
            first_std,
            out=np.full(size, np.nan, dtype=float),
            where=first_std > 0.0,
        )
        second_sharpe = np.divide(
            second.mean(axis=1) * math.sqrt(ANNUALIZATION),
            second_std,
            out=np.full(size, np.nan, dtype=float),
            where=second_std > 0.0,
        )
        sharpe_differences[cursor : cursor + size] = first_sharpe - second_sharpe
        cursor += size

    def _quantile(values: np.ndarray, level: float) -> float:
        finite = values[np.isfinite(values)]
        return float(np.quantile(finite, level)) if len(finite) else float("nan")

    return {
        "paired_bootstrap_reps": float(reps),
        "paired_bootstrap_block_length": float(block_length),
        "paired_delta_bootstrap_q05_bp": _quantile(delta_means, 0.05),
        "paired_delta_bootstrap_q50_bp": _quantile(delta_means, 0.50),
        "paired_delta_bootstrap_q95_bp": _quantile(delta_means, 0.95),
        "paired_delta_bootstrap_probability_positive": float(np.mean(delta_means > 0.0)),
        "paired_sharpe_difference_bootstrap_q05": _quantile(sharpe_differences, 0.05),
        "paired_sharpe_difference_bootstrap_q50": _quantile(sharpe_differences, 0.50),
        "paired_sharpe_difference_bootstrap_q95": _quantile(sharpe_differences, 0.95),
    }


def _render_results(scorecard: pd.DataFrame, paired: pd.DataFrame, *, evidence: Mapping[str, Any]) -> str:
    top = scorecard.loc[scorecard["scope"] == "reporting_2026"].sort_values("sharpe", ascending=False).head(8)
    lines = [
        "# R88 单模型验证 Phase 1",
        "",
        "## 状态",
        "",
        "- 研究专用；未调用 live runtime、未写 DB、未修改模型或因子。",
        "- 使用完整 R88 30 候选既有 rolling OOS 日收益；本阶段不重跑模型。",
        "- 2025 是 development/selection，2026 是 reporting-only；本报告不据 2026 重新选模型。",
        "- Regsim 配对统计只使用 399 个共同日期，且只检验原始日收益差；各模型 beta/alpha 保留自身冻结 benchmark。",
        "",
        "## 2026 reporting Sharpe 前八",
        "",
        "| candidate | Sharpe | HAC alpha t | MDD | annualized simple return | rolling60 Sharpe median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.sharpe:.3f} | {row.alpha_hac_t:.3f} | {row.max_drawdown:.2%} | {row.annualized_return_simple:.2%} | {row.rolling60_sharpe_median:.3f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 本阶段是单模型收益稳定性 scorecard，不是 PBO/DSR/SPA/MCS，也不是参数或因子扰动实验。",
        "- 高 Sharpe、HAC alpha t 或 bootstrap 区间不校正研究者多重试验；后续阶段必须将完整 trial family 纳入统计校正。",
        "- 不能由本报告推导任何实盘准入、模型切换或组合权重结论。",
        "",
        "## 关键输入",
        "",
        f"- R88 study: `{evidence['study_root']}`",
        f"- R88 candidate count: `{evidence['candidate_count']}`",
        f"- Regsim common days: `{evidence['regsim_common_days']}`",
        f"- Experiment factor generation: `{evidence['factor_generation']['path']}`",
    ])
    return "\n".join(lines) + "\n"


def run_validation(
    *,
    study_root: Path,
    regsim_path: Path,
    output_root: Path,
    run_name: str,
    bootstrap_reps: int,
    block_length: int,
) -> Path:
    run_root = _make_run_root(output_root, run_name)
    sources, plan, integrity, integrity_verification = _load_candidate_sources(study_root)
    admission = _read_json(study_root / "input_admission.json")
    factor_input = admission.get("factor_input", {})
    generation_path = Path(str(factor_input.get("table_root", "")))
    generation_manifest = generation_path / "generation_manifest.json"
    generation_done = generation_path / "generation.done"
    generation = _read_json(generation_manifest)
    done = _read_json(generation_done)
    if not bool(done.get("ready")) or not bool(generation.get("migration", {}).get("ready_for_consumer")):
        raise ValidationError("R88 experiment factor generation is not ready for consumer")
    expected_days = pd.DatetimeIndex(pd.to_datetime(plan.get("exact_calendar", {}).get("score_days", [])))
    if len(expected_days) != 401 or expected_days.has_duplicates:
        raise ValidationError("unexpected frozen R88 score calendar")
    returns_by_candidate = {source.candidate: _load_r88_returns(source, expected_days) for source in sources}
    benchmark_reference = returns_by_candidate[sources[0].candidate][["trade_date", "benchmark_return", "benchmark_method"]]
    for source in sources[1:]:
        current = returns_by_candidate[source.candidate][["trade_date", "benchmark_return", "benchmark_method"]]
        if not benchmark_reference.equals(current):
            raise ValidationError(f"R88 benchmark differs across candidates: {source.candidate}")
    regsim = _load_regsim(regsim_path)
    score_rows: list[dict[str, object]] = []
    rolling_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    for position, source in enumerate(sources):
        daily = returns_by_candidate[source.candidate]
        for scope_name in ("development_2025", "reporting_2026", "overall"):
            scoped = _scope(daily, scope_name)
            metrics = calculate_metrics(scoped)
            rolling_summary: dict[str, float] = {}
            for window in ROLLING_WINDOWS:
                values = []
                if len(scoped) >= window:
                    for end in range(window - 1, len(scoped)):
                        values.append(calculate_metrics(scoped.iloc[end - window + 1 : end + 1])["sharpe"])
                finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
                rolling_summary[f"rolling{window}_sharpe_median"] = float(np.median(finite)) if len(finite) else float("nan")
                rolling_summary[f"rolling{window}_sharpe_min"] = float(np.min(finite)) if len(finite) else float("nan")
                rolling_summary[f"rolling{window}_sharpe_positive_fraction"] = float(np.mean(finite > 0.0)) if len(finite) else float("nan")
            score_rows.append({
                "candidate": source.candidate,
                "factor_mode": source.factor_mode,
                "factor_set_id": source.factor_set_id,
                "hyperparameter_id": source.hyperparameter_id,
                "scope": scope_name,
                "start": scoped["trade_date"].iloc[0],
                "end": scoped["trade_date"].iloc[-1],
                **metrics,
                **rolling_summary,
            })
            bootstrap_rows.append({
                "candidate": source.candidate,
                "scope": scope_name,
                **moving_block_bootstrap(scoped, reps=bootstrap_reps, block_length=block_length, seed=RANDOM_SEED + position * 10 + len(bootstrap_rows)),
            })
        for window in ROLLING_WINDOWS:
            rolling_rows.extend(_rolling_metrics(source.candidate, daily, window))
        for scope_name in ("development_2025", "reporting_2026", "overall"):
            scoped = _scope(daily, scope_name)
            for index, block in enumerate(_chronological_blocks(scoped), start=1):
                block_rows.append({
                    "candidate": source.candidate,
                    "scope": scope_name,
                    "block": index,
                    "start": block["trade_date"].iloc[0],
                    "end": block["trade_date"].iloc[-1],
                    **calculate_metrics(block),
                })
        for scope_name in ("development_2025", "reporting_2026", "overall"):
            candidate_scope = _scope(daily, scope_name)
            regsim_scope = _scope(regsim, scope_name)
            paired_rows.append({
                "candidate": source.candidate,
                "scope": scope_name,
                **_raw_delta_hac(candidate_scope, regsim_scope),
                **_paired_delta_bootstrap(
                    candidate_scope,
                    regsim_scope,
                    reps=bootstrap_reps,
                    block_length=block_length,
                    seed=RANDOM_SEED + position * 100 + len(paired_rows),
                ),
            })
    scorecard = pd.DataFrame(score_rows).sort_values(["scope", "sharpe"], ascending=[True, False])
    rolling = pd.DataFrame(rolling_rows).sort_values(["candidate", "window_days", "end_trade_date"])
    blocks = pd.DataFrame(block_rows).sort_values(["candidate", "block"])
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(["scope", "candidate"])
    paired = pd.DataFrame(paired_rows).sort_values(["scope", "candidate"])
    common_days = sorted(set(regsim["trade_date"]).intersection(set(expected_days)))
    ranking_path = study_root / "metrics" / "candidate_ranking.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(ranking_path)
    status_docs = [_read_json(path) for path in (study_root / "status").glob("*.json")]
    status_counts = pd.Series([str(doc.get("status", "missing")) for doc in status_docs]).value_counts().to_dict()
    if status_counts.get("completed", 0) != 30:
        raise ValidationError(f"R88 status is not 30/30 completed: {status_counts}")
    r88_days = set(pd.Timestamp(value).date() for value in expected_days)
    regsim_days = set(regsim["trade_date"].dt.date)
    common_benchmark = benchmark_reference.merge(
        regsim[["trade_date", "benchmark_return"]], on="trade_date", how="inner", suffixes=("_r88", "_regsim")
    )
    benchmark_difference = pd.to_numeric(common_benchmark["benchmark_return_r88"], errors="coerce") - pd.to_numeric(common_benchmark["benchmark_return_regsim"], errors="coerce")
    evidence = {
        "schema": "r88_single_model_validation_phase1_input_evidence/v1",
        "study_id": STUDY_ID,
        "research_only": True,
        "promotion_allowed": False,
        "study_root": str(study_root.resolve()),
        "study_plan_sha256": _sha256_file(study_root / "study_plan.json"),
        "study_integrity_sha256": _sha256_file(study_root / "study_integrity.json"),
        "study_integrity_declared_sha256": integrity.get("study_integrity_sha256"),
        "study_integrity_verification": integrity_verification,
        "candidate_count": len(sources),
        "candidate_calendar_days": len(expected_days),
        "candidate_calendar_start": str(expected_days.min().date()),
        "candidate_calendar_end": str(expected_days.max().date()),
        "factor_generation": {
            "path": str(generation_path),
            "generation_manifest_sha256": _sha256_file(generation_manifest),
            "generation_ready": bool(done.get("ready")),
            "generation_completed_day_count": done.get("completed_day_count"),
            "factor_count": factor_input.get("factor_count"),
            "day_count": factor_input.get("day_count"),
        },
        "regsim": {
            "path": str(regsim_path.resolve()),
            "sha256": _sha256_file(regsim_path),
            "days": len(regsim),
            "start": str(regsim["trade_date"].min().date()),
            "end": str(regsim["trade_date"].max().date()),
        },
        "regsim_common_days": len(common_days),
        "regsim_missing_from_r88_calendar": [str(day.date()) for day in expected_days if day.date() not in regsim_days],
        "known_comparator_coverage_gap": [str(day.date()) for day in expected_days if day.date() not in regsim_days],
        "regsim_benchmark_mismatch": {
            "common_days_checked": int(len(common_benchmark)),
            "mismatch_days_gt_1e-12": int(np.sum(np.abs(benchmark_difference.to_numpy(dtype=float)) > 1e-12)),
            "max_abs_difference": float(np.max(np.abs(benchmark_difference.to_numpy(dtype=float)))) if len(benchmark_difference) else float("nan"),
        },
        "status_counts": status_counts,
        "candidate_ranking_sha256": _sha256_file(ranking_path),
        "benchmark_alignment_policy": {
            "single_model_alpha": "use each candidate's own frozen benchmark_return",
            "candidate_vs_regsim": "raw day_return difference on trade_date inner join; no cross-source alpha comparison",
        },
        "reused": ["R88 daily_returns", "R88 coverage", "R88 warm_start_coverage", "Regsim return history"],
        "missing": [],
        "invalid": [],
    }
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    scorecard.to_csv(run_root / "single_model_scorecard.csv", index=False)
    rolling.to_csv(run_root / "rolling_metrics.csv", index=False)
    blocks.to_csv(run_root / "time_block_metrics.csv", index=False)
    bootstrap.to_csv(run_root / "bootstrap_summary.csv", index=False)
    paired.to_csv(run_root / "paired_regsim_raw_return_delta.csv", index=False)
    _write_json(run_root / "input_evidence.json", evidence)
    manifest = {
        "schema": "r88_single_model_validation_phase1_run/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "factor_store_writes": False,
        "model_scoring_called": False,
        "model_training_called": False,
        "scope": "existing daily OOS return validation only",
        "excluded_phase2": ["PBO/CSCV", "PSR/DSR", "White Reality Check", "Hansen SPA", "MCS", "parameter perturbation", "factor perturbation"],
        "bootstrap": {"reps": bootstrap_reps, "block_length": block_length, "seed": RANDOM_SEED},
        "inputs": evidence,
        "git": {"head": _safe_git(["rev-parse", "HEAD"]), "status_porcelain": _safe_git(["status", "--short"])},
    }
    _write_json(run_root / "run_manifest.json", manifest)
    # Keep JSON/CSV machine artifacts as UTF-8 without a BOM; use a BOM only
    # for the human-facing Markdown so Windows PowerShell 5.1 detects UTF-8.
    (run_root / "RESULTS.md").write_text(
        _render_results(scorecard, paired, evidence=evidence),
        encoding=RESULTS_ENCODING,
    )
    _write_json(run_root / "run_status.json", {"status": "completed", "run_root": str(run_root), "created_at_utc": datetime.now(timezone.utc).isoformat()})
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--regsim-path", type=Path, default=DEFAULT_REGSIM_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--block-length", type=int, default=BOOTSTRAP_BLOCK_LENGTH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_reps <= 0 or args.block_length <= 0:
        raise ValueError("bootstrap reps and block length must be positive")
    run_root = run_validation(
        study_root=args.study_root,
        regsim_path=args.regsim_path,
        output_root=args.output_root,
        run_name=args.run_name,
        bootstrap_reps=args.bootstrap_reps,
        block_length=args.block_length,
    )
    print(f"R88 single-model validation completed: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
