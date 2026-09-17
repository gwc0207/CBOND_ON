"""Research-only fixed-strategy health and degradation panel for R88.

The primary question here is *within one fixed strategy*: does its score keep
ordering next-day labels, and does its realized alpha/risk remain stable as
time advances?  This tool intentionally does not rank candidates or choose a
replacement model.  It consumes existing R88 score CSVs, existing label
partitions and existing strategy daily-return CSVs, then writes only an
isolated research report below ``D:/cbond_on/research_scratch``.

The score/label contract is the same as the governed R88 chain: score at T
14:30 is joined with the label at T 14:42 (the strict next-cycle outcome).
The strategy return is the already materialized Top20 strategy daily return;
it is reported separately because it includes execution/cost effects and is
not identical to the raw cross-sectional label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

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
    _load_r88_returns,
    _ols_hac,
    _sha256_file,
)
from harness.tools.validate_r88_single_model_phase2 import (  # noqa: E402
    DEFAULT_STUDY_ROOT,
    EXPECTED_IMMUTABLE_HASH_COUNT,
    FrozenInputs,
    load_frozen_inputs,
)


STUDY_ID = "r88_joint_factor_lgbm_20260828_r1"
DEFAULT_LABEL_ROOT = Path(r"D:/cbond_on/label_data")
DEFAULT_HEALTH_OUTPUT_ROOT = Path(r"D:/cbond_on/research_scratch/r88_single_model_health_20260902_r1")
DEFAULT_RUN_NAME = "health_20260902_r1"
SCORE_TIME = time(14, 30)
LABEL_TIME = time(14, 42)
ANNUALIZATION = 252.0
TOP_K = 20
ROLLING_WINDOWS = (20, 60, 120)
BASELINE_WINDOW = 60
SHIFT_RECENT_WINDOW = 20
SHIFT_PRIOR_WINDOW = 60
PERSISTENCE_DAYS = 5
ALPHA_POSITIVE_EVIDENCE_T = 1.645
ALPHA_NEGATIVE_EVIDENCE_T = -1.645
SCORE_DISPERSION_COMPRESSION_RATIO = 0.50
CUSUM_DRIFT = 0.50
CUSUM_BLOCK_LENGTH = 10
CUSUM_BOOTSTRAP_REPS = 2_000
CUSUM_CONTROL_QUANTILE = 0.95
RESULTS_ENCODING = "utf-8-sig"
SCOPE_NAMES = ("development_2025", "reporting_2026", "overall")


class HealthValidationError(RuntimeError):
    """Raised when health-panel inputs or contracts are incomplete."""


@dataclass(frozen=True)
class ScoreLabelAudit:
    candidate: str
    score_root: Path
    score_files: tuple[Path, ...]
    daily: pd.DataFrame
    coverage: dict[str, Any]


def _json_default(value: object) -> object:
    if isinstance(value, (Path, pd.Timestamp, datetime, date, time)):
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
        raise HealthValidationError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = parent.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HealthValidationError(f"{label} must stay below {root}: {resolved}") from exc
    return resolved


def _make_run_root(output_root: Path, run_name: str) -> Path:
    root = _resolve_inside(output_root, DEFAULT_HEALTH_OUTPUT_ROOT, label="health output root")
    if Path(run_name).name != run_name or run_name in {"", ".", ".."}:
        raise HealthValidationError(f"run name must be one path component: {run_name!r}")
    run_root = _resolve_inside(root / run_name, DEFAULT_HEALTH_OUTPUT_ROOT, label="health run root")
    if run_root == DEFAULT_HEALTH_OUTPUT_ROOT.resolve():
        raise HealthValidationError("a fresh child run directory is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite existing health run: {run_root}")
    return run_root


def _require_score_frame(frame: pd.DataFrame, *, path: Path, expected_day: pd.Timestamp) -> pd.DataFrame:
    required = {"trade_date", "code", "score"}
    if not required.issubset(frame.columns):
        raise HealthValidationError(f"score file missing columns {sorted(required - set(frame.columns))}: {path}")
    out = frame.loc[:, ["trade_date", "code", "score"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out["code"] = out["code"].astype(str).str.strip()
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    if out["trade_date"].isna().any() or set(out["trade_date"].dt.normalize()) != {expected_day.normalize()}:
        raise HealthValidationError(f"score file trade_date mismatch: {path}")
    if out["code"].eq("").any() or out["code"].duplicated().any():
        raise HealthValidationError(f"score file has blank or duplicate code: {path}")
    if not np.isfinite(out["score"].to_numpy(dtype=float)).all():
        raise HealthValidationError(f"score file has non-finite score: {path}")
    return out


def _load_label_day(label_root: Path, day: pd.Timestamp) -> pd.DataFrame:
    path = label_root / f"{day.year:04d}-{day.month:02d}" / f"{day:%Y%m%d}.parquet"
    if not path.is_file():
        raise HealthValidationError(f"missing label partition: {path}")
    try:
        frame = pd.read_parquet(path, columns=["code", "trade_time", "y"])
    except Exception as exc:
        raise HealthValidationError(f"cannot read label partition {path}: {exc}") from exc
    if frame.empty:
        raise HealthValidationError(f"empty label partition: {path}")
    frame = frame.copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.loc[frame["trade_time"].dt.time == LABEL_TIME, ["code", "trade_time", "y"]].copy()
    frame["code"] = frame["code"].astype(str).str.strip()
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    if frame.empty or frame["code"].eq("").any() or frame["code"].duplicated().any():
        raise HealthValidationError(f"label partition has no valid unique {LABEL_TIME} rows: {path}")
    if not np.isfinite(frame["y"].to_numpy(dtype=float)).all():
        raise HealthValidationError(f"label partition has non-finite y: {path}")
    return frame[["code", "y"]]


def _load_label_cache(label_root: Path, expected_days: pd.DatetimeIndex) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[str, Any]]:
    cache: dict[pd.Timestamp, pd.DataFrame] = {}
    hashes: list[tuple[str, str]] = []
    for day in expected_days:
        path = label_root / f"{day.year:04d}-{day.month:02d}" / f"{day:%Y%m%d}.parquet"
        cache[day.normalize()] = _load_label_day(label_root, day)
        hashes.append((str(path.relative_to(label_root)).replace("\\", "/"), _sha256_file(path)))
    payload = "\n".join(f"{name}:{digest}" for name, digest in hashes)
    return cache, {
        "root": str(label_root),
        "partition_count": len(hashes),
        "combined_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "first_partition": hashes[0][0] if hashes else None,
        "last_partition": hashes[-1][0] if hashes else None,
    }


def _score_file_map(score_root: Path, expected_days: pd.DatetimeIndex) -> tuple[Path, ...]:
    files = sorted(score_root.rglob("*.csv"))
    by_day: dict[pd.Timestamp, Path] = {}
    for path in files:
        try:
            day = pd.Timestamp(path.stem)
        except Exception:
            continue
        day = day.normalize()
        if day in set(expected_days.normalize()):
            if day in by_day:
                raise HealthValidationError(f"duplicate score partition for {day.date()} under {score_root}")
            by_day[day] = path
    missing = [day for day in expected_days.normalize() if day not in by_day]
    if missing:
        raise HealthValidationError(f"score root {score_root} missing {len(missing)} expected days; first={missing[0].date()}")
    return tuple(by_day[day] for day in expected_days.normalize())


def _corr(x: np.ndarray, y: np.ndarray, *, rank: bool) -> float:
    if len(x) < 3 or len(y) != len(x):
        return float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return float("nan")
    if rank:
        x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
        y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return float(np.corrcoef(x, y)[0, 1])


def _daily_score_quality(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    day: pd.Timestamp,
    previous_scores: pd.DataFrame | None,
) -> dict[str, Any]:
    merged = scores[["code", "score"]].merge(labels, on="code", how="inner", validate="one_to_one")
    if merged.empty:
        raise HealthValidationError(f"score/label join empty on {day.date()}")
    merged = merged.dropna(subset=["score", "y"]).copy()
    all_score_values = scores["score"].to_numpy(dtype=float)
    score_values = merged["score"].to_numpy(dtype=float)
    label_values = merged["y"].to_numpy(dtype=float)
    order = np.argsort(-score_values, kind="mergesort")
    top_n = min(TOP_K, len(merged))
    top = merged.iloc[order[:top_n]]
    bottom = merged.iloc[order[-top_n:]]
    full_top = scores.nlargest(min(TOP_K, len(scores)), "score")
    full_bottom = scores.nsmallest(min(TOP_K, len(scores)), "score")
    score_mean = float(np.mean(all_score_values))
    score_std = float(np.std(all_score_values, ddof=1)) if len(all_score_values) > 1 else float("nan")
    top_score_mean = float(full_top["score"].mean())
    bottom_score_mean = float(full_bottom["score"].mean())
    row: dict[str, Any] = {
        "trade_date": day,
        "score_count": int(len(scores)),
        "label_count": int(len(labels)),
        "joined_count": int(len(merged)),
        "raw_score_label_join_fraction": float(len(merged) / max(len(scores), 1)),
        "raw_score_label_complete": bool(len(merged) == len(scores)),
        "partial_label_coverage": bool(len(merged) < len(scores)),
        "score_mean": score_mean,
        "score_std": score_std,
        "score_min": float(np.min(all_score_values)),
        "score_max": float(np.max(all_score_values)),
        "score_range": float(np.max(all_score_values) - np.min(all_score_values)),
        "score_q05": float(np.quantile(all_score_values, 0.05)),
        "score_q25": float(np.quantile(all_score_values, 0.25)),
        "score_median": float(np.median(all_score_values)),
        "score_q75": float(np.quantile(all_score_values, 0.75)),
        "score_q95": float(np.quantile(all_score_values, 0.95)),
        "score_top20_mean": top_score_mean,
        "score_bottom20_mean": bottom_score_mean,
        "score_top20_bottom20_gap": top_score_mean - bottom_score_mean,
        "score_top20_mean_z": (top_score_mean - score_mean) / score_std if score_std > 0 else float("nan"),
        "label_mean": float(np.mean(label_values)),
        "label_std": float(np.std(label_values, ddof=1)) if len(label_values) > 1 else float("nan"),
        "raw_score_label_ic": _corr(score_values, label_values, rank=False),
        "raw_score_label_rank_ic": _corr(score_values, label_values, rank=True),
        "raw_score_top20_label_mean": float(top["y"].mean()),
        "raw_score_bottom20_label_mean": float(bottom["y"].mean()),
        "raw_score_top20_bottom20_label_gap": float(top["y"].mean() - bottom["y"].mean()),
        "raw_score_top20_label_positive_fraction": float(np.mean(top["y"].to_numpy(dtype=float) > 0.0)),
        "label_time_contract": LABEL_TIME.strftime("%H:%M"),
        "score_time_contract": SCORE_TIME.strftime("%H:%M"),
    }
    if previous_scores is None:
        row["score_rank_corr_previous_day"] = float("nan")
        row["top20_overlap_previous_day"] = float("nan")
    else:
        previous = previous_scores.set_index("code")["score"]
        current = scores.set_index("code")["score"]
        common = current.index.intersection(previous.index)
        row["score_rank_corr_previous_day"] = _corr(
            current.loc[common].to_numpy(dtype=float), previous.loc[common].to_numpy(dtype=float), rank=True
        ) if len(common) >= 3 else float("nan")
        current_top = set(scores.nlargest(top_n, "score")["code"])
        previous_top = set(previous_scores.nlargest(top_n, "score")["code"])
        row["top20_overlap_previous_day"] = float(len(current_top.intersection(previous_top)) / max(top_n, 1))
    if row["partial_label_coverage"]:
        # Current label partitions are a source-quality audit only.  Do not
        # treat a partial join as a valid raw-label prediction observation.
        for name in (
            "raw_score_label_ic",
            "raw_score_label_rank_ic",
            "raw_score_top20_label_mean",
            "raw_score_bottom20_label_mean",
            "raw_score_top20_bottom20_label_gap",
            "raw_score_top20_label_positive_fraction",
        ):
            row[name] = float("nan")
    return row


def _load_candidate_backtest_ic(source: CandidateSource, expected_days: pd.DatetimeIndex) -> pd.DataFrame:
    metrics = _read_json(source.metrics_path)
    daily_returns_path = Path(str(metrics.get("daily_returns_path", "")))
    ic_path = daily_returns_path.parent / "ic.csv"
    if not ic_path.is_file():
        raise HealthValidationError(f"missing frozen strategy ic.csv for {source.candidate}: {ic_path}")
    frame = pd.read_csv(ic_path)
    required = {"trade_date", "score_day", "execution_lag_trading_days", "ic", "rank_ic", "count"}
    if not required.issubset(frame.columns):
        raise HealthValidationError(f"strategy ic.csv missing columns {sorted(required - set(frame.columns))}: {ic_path}")
    out = frame.loc[:, sorted(required)].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["score_day"] = pd.to_datetime(out["score_day"], errors="coerce").dt.normalize()
    for column in ("ic", "rank_ic"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["execution_lag_trading_days"] = pd.to_numeric(out["execution_lag_trading_days"], errors="coerce")
    if out["trade_date"].isna().any() or out["trade_date"].duplicated().any() or not pd.DatetimeIndex(out["trade_date"]).equals(expected_days):
        raise HealthValidationError(f"strategy ic date coverage differs from frozen R88 calendar: {ic_path}")
    if not np.isfinite(out[["ic", "rank_ic", "count", "execution_lag_trading_days"]].to_numpy(dtype=float)).all():
        raise HealthValidationError(f"strategy ic.csv contains non-finite values: {ic_path}")
    if not (out["score_day"] == out["trade_date"]).all() or not (out["execution_lag_trading_days"] == 0).all():
        raise HealthValidationError(f"strategy ic.csv does not match same-day frozen R88 execution: {ic_path}")
    if not (out["count"] >= TOP_K).all():
        raise HealthValidationError(f"strategy ic.csv has insufficient tradable observations: {ic_path}")
    return out.rename(columns={"ic": "strategy_universe_ic", "rank_ic": "strategy_universe_rank_ic", "count": "strategy_universe_ic_count"})


def _load_candidate_score_quality(
    source: CandidateSource,
    *,
    study_root: Path,
    expected_days: pd.DatetimeIndex,
    label_root: Path,
    label_cache: Mapping[pd.Timestamp, pd.DataFrame] | None = None,
) -> ScoreLabelAudit:
    score_root = study_root / "runtime" / "results" / "scores" / source.candidate
    if not score_root.is_dir():
        raise HealthValidationError(f"missing score root for {source.candidate}: {score_root}")
    score_files = _score_file_map(score_root, expected_days)
    primary_ic = _load_candidate_backtest_ic(source, expected_days)
    metrics_doc = _read_json(source.metrics_path)
    strategy_returns_path = Path(str(metrics_doc.get("daily_returns_path", "")))
    strategy_ic_path = strategy_returns_path.parent / "ic.csv"
    rows: list[dict[str, Any]] = []
    previous: pd.DataFrame | None = None
    joined_total = 0
    score_total = 0
    label_total = 0
    score_hashes: list[tuple[str, str]] = []
    for day, path in zip(expected_days, score_files, strict=True):
        scores = _require_score_frame(pd.read_csv(path), path=path, expected_day=day)
        labels = label_cache.get(day.normalize()) if label_cache is not None else None
        if labels is None:
            labels = _load_label_day(label_root, day)
        row = _daily_score_quality(scores, labels, day=day, previous_scores=previous)
        rows.append(row)
        previous = scores
        score_total += len(scores)
        label_total += len(labels)
        joined_total += int(row["joined_count"])
        score_hashes.append((str(path.relative_to(score_root)).replace("\\", "/"), _sha256_file(path)))
    daily = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    daily = daily.merge(primary_ic, on="trade_date", how="inner", validate="one_to_one")
    if len(daily) != len(expected_days):
        raise HealthValidationError(f"strategy IC merge lost R88 score days for {source.candidate}")
    # The primary predictive-health fields must obey the actual candidate
    # strategy universe and strict cycle execution.  Raw score-label values
    # remain only as coverage/dispersion diagnostics.
    daily["ic"] = daily["strategy_universe_ic"]
    daily["rank_ic"] = daily["strategy_universe_rank_ic"]
    daily["ic_contract"] = "frozen_strategy_universe_strict_cycle_net"
    coverage = {
        "expected_days": int(len(expected_days)),
        "score_days": int(len(score_files)),
        "label_days": int(len(rows)),
        "score_rows": score_total,
        "label_rows": label_total,
        "joined_rows": joined_total,
        "strategy_ic_rows": int(len(primary_ic)),
        "start": str(expected_days.min().date()),
        "end": str(expected_days.max().date()),
        "missing_score_days": [],
        "missing_label_days": [],
        "score_files_combined_sha256": hashlib.sha256(
            "\n".join(f"{name}:{digest}" for name, digest in score_hashes).encode("utf-8")
        ).hexdigest(),
        "strategy_ic_path": str(strategy_ic_path),
        "strategy_ic_sha256": _sha256_file(strategy_ic_path),
        "strategy_returns_path": str(strategy_returns_path),
        "strategy_returns_sha256": _sha256_file(strategy_returns_path),
    }
    return ScoreLabelAudit(source.candidate, score_root, score_files, daily, coverage)


def _rolling_series_mean_std(values: np.ndarray, *, window: int, index: int) -> tuple[float, float]:
    if index + 1 < window:
        return float("nan"), float("nan")
    segment = values[index - window + 1 : index + 1]
    finite = segment[np.isfinite(segment)]
    if len(finite) < window:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")


def _rolling_cvar(values: np.ndarray, *, window: int, index: int, probability: float = 0.05) -> float:
    if index + 1 < window:
        return float("nan")
    segment = values[index - window + 1 : index + 1]
    if not np.isfinite(segment).all():
        return float("nan")
    cutoff = float(np.quantile(segment, probability))
    tail = segment[segment <= cutoff]
    return float(np.mean(tail)) if len(tail) else cutoff


def _rolling_max_drawdown(values: np.ndarray, *, window: int, index: int) -> float:
    if index + 1 < window:
        return float("nan")
    segment = values[index - window + 1 : index + 1]
    if not np.isfinite(segment).all():
        return float("nan")
    nav = np.cumprod(1.0 + segment)
    return float(np.min(nav / np.maximum.accumulate(nav) - 1.0))


def _add_rolling_metrics(daily: pd.DataFrame, *, strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> pd.DataFrame:
    """Add causal rolling prediction, alpha and risk fields to one candidate."""

    if len(daily) != len(strategy_returns) or len(daily) != len(benchmark_returns):
        raise ValueError("daily panel and strategy return lengths differ")
    out = daily.copy().sort_values("trade_date").reset_index(drop=True)
    strategy_returns = np.asarray(strategy_returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    rank_ic = pd.to_numeric(out["rank_ic"], errors="coerce").to_numpy(dtype=float)
    ic_count = pd.to_numeric(out["strategy_universe_ic_count"], errors="coerce").to_numpy(dtype=float)
    ic_coverage_threshold: list[float] = []
    ic_coverage_insufficient: list[bool] = []
    for index, count in enumerate(ic_count):
        prior = ic_count[max(0, index - BASELINE_WINDOW) : index]
        prior = prior[np.isfinite(prior)]
        threshold = max(100.0, 0.5 * float(np.median(prior))) if len(prior) else 100.0
        ic_coverage_threshold.append(threshold)
        ic_coverage_insufficient.append(bool(not np.isfinite(count) or count < threshold))
    out["strategy_ic_coverage_threshold"] = ic_coverage_threshold
    out["strategy_ic_coverage_insufficient"] = ic_coverage_insufficient
    # The frozen strategy IC remains the primary observation even on a small
    # cross-section.  Coverage flags qualify interpretation; they do not
    # silently delete an otherwise valid frozen execution result.
    ic_valid_for_health = ~out["strategy_ic_coverage_insufficient"].fillna(True).to_numpy(dtype=bool)
    rank_ic_health = rank_ic.copy()
    ic = pd.to_numeric(out["ic"], errors="coerce").to_numpy(dtype=float)
    score_std = pd.to_numeric(out["score_std"], errors="coerce").to_numpy(dtype=float)
    score_range = pd.to_numeric(out["score_range"], errors="coerce").to_numpy(dtype=float)
    excess_returns = strategy_returns - benchmark_returns
    for window in ROLLING_WINDOWS:
        fields: dict[str, list[float]] = {
            f"rolling{window}_strategy_mean_return_bp": [],
            f"rolling{window}_strategy_sharpe": [],
            f"rolling{window}_strategy_win_fraction": [],
            f"rolling{window}_rank_ic_mean": [],
            f"rolling{window}_rank_ic_std": [],
            f"rolling{window}_rank_icir": [],
            f"rolling{window}_rank_ic_valid_day_count": [],
            f"rolling{window}_rank_ic_data_inconclusive": [],
            f"rolling{window}_ic_mean": [],
            f"rolling{window}_score_std_median": [],
            f"rolling{window}_score_range_median": [],
            f"rolling{window}_raw_score_top20_label_gap_mean": [],
            f"rolling{window}_strategy_cvar95": [],
            f"rolling{window}_strategy_max_drawdown": [],
            f"rolling{window}_alpha_daily_bp": [],
            f"rolling{window}_alpha_hac_t": [],
            f"rolling{window}_alpha_hac_p": [],
            f"rolling{window}_beta": [],
            f"rolling{window}_residual_annualized_vol": [],
        }
        for index in range(len(out)):
            ret_mean, ret_std = _rolling_series_mean_std(strategy_returns, window=window, index=index)
            rank_mean, rank_std = _rolling_series_mean_std(rank_ic_health, window=window, index=index)
            _, _ = _rolling_series_mean_std(ic, window=window, index=index)
            if index + 1 < window:
                alpha = {"alpha_daily": float("nan"), "alpha_hac_t": float("nan"), "alpha_hac_p_two_sided": float("nan"), "beta": float("nan"), "residual_daily_volatility": float("nan")}
            else:
                y = strategy_returns[index - window + 1 : index + 1]
                b = benchmark_returns[index - window + 1 : index + 1]
                alpha = _ols_hac(y, b) if len(y) >= 2 else {"alpha_daily": float("nan"), "alpha_hac_t": float("nan"), "alpha_hac_p_two_sided": float("nan"), "beta": float("nan"), "residual_daily_volatility": float("nan")}
            def median_window(array: np.ndarray) -> float:
                if index + 1 < window:
                    return float("nan")
                segment = array[index - window + 1 : index + 1]
                return float(np.median(segment)) if np.isfinite(segment).all() else float("nan")
            fields[f"rolling{window}_strategy_mean_return_bp"].append(ret_mean * 10_000.0 if np.isfinite(ret_mean) else float("nan"))
            fields[f"rolling{window}_strategy_sharpe"].append(ret_mean / ret_std * math.sqrt(ANNUALIZATION) if np.isfinite(ret_mean) and np.isfinite(ret_std) and ret_std > 0 else float("nan"))
            fields[f"rolling{window}_strategy_win_fraction"].append(float(np.mean(strategy_returns[index - window + 1 : index + 1] > 0.0)) if index + 1 >= window else float("nan"))
            fields[f"rolling{window}_rank_ic_mean"].append(rank_mean)
            fields[f"rolling{window}_rank_ic_std"].append(rank_std)
            fields[f"rolling{window}_rank_icir"].append(rank_mean / rank_std if np.isfinite(rank_mean) and np.isfinite(rank_std) and rank_std > 0 else float("nan"))
            valid_count = int(np.sum(ic_valid_for_health[index - window + 1 : index + 1])) if index + 1 >= window else 0
            fields[f"rolling{window}_rank_ic_valid_day_count"].append(float(valid_count))
            fields[f"rolling{window}_rank_ic_data_inconclusive"].append(bool(index + 1 >= window and valid_count < window))
            fields[f"rolling{window}_ic_mean"].append(float(np.mean(ic[index - window + 1 : index + 1])) if index + 1 >= window and np.isfinite(ic[index - window + 1 : index + 1]).all() else float("nan"))
            fields[f"rolling{window}_score_std_median"].append(median_window(score_std))
            fields[f"rolling{window}_score_range_median"].append(median_window(score_range))
            gap = pd.to_numeric(out["raw_score_top20_bottom20_label_gap"], errors="coerce").to_numpy(dtype=float, copy=True)
            gap[out["partial_label_coverage"].to_numpy(dtype=bool)] = np.nan
            fields[f"rolling{window}_raw_score_top20_label_gap_mean"].append(float(np.mean(gap[index - window + 1 : index + 1])) if index + 1 >= window and np.isfinite(gap[index - window + 1 : index + 1]).all() else float("nan"))
            fields[f"rolling{window}_strategy_cvar95"].append(_rolling_cvar(strategy_returns, window=window, index=index))
            fields[f"rolling{window}_strategy_max_drawdown"].append(_rolling_max_drawdown(strategy_returns, window=window, index=index))
            fields[f"rolling{window}_alpha_daily_bp"].append(float(alpha["alpha_daily"] * 10_000.0))
            fields[f"rolling{window}_alpha_hac_t"].append(float(alpha["alpha_hac_t"]))
            fields[f"rolling{window}_alpha_hac_p"].append(float(alpha["alpha_hac_p_two_sided"]))
            fields[f"rolling{window}_beta"].append(float(alpha["beta"]))
            fields[f"rolling{window}_residual_annualized_vol"].append(float(alpha["residual_daily_volatility"] * math.sqrt(ANNUALIZATION)))
        for name, values in fields.items():
            out[name] = values
    # Causal comparisons: recent 20 observations versus the immediately prior
    # 60 observations; no future values or full-period statistics are used.
    rank_ic_values = rank_ic_health
    recent_excess_shift: list[float] = []
    recent_rank_shift: list[float] = []
    dispersion_ratio: list[float] = []
    for index in range(len(out)):
        if index + 1 < SHIFT_RECENT_WINDOW + SHIFT_PRIOR_WINDOW:
            recent_excess_shift.append(float("nan")); recent_rank_shift.append(float("nan")); dispersion_ratio.append(float("nan")); continue
        recent = excess_returns[index - SHIFT_RECENT_WINDOW + 1 : index + 1]
        prior = excess_returns[index - SHIFT_RECENT_WINDOW - SHIFT_PRIOR_WINDOW + 1 : index - SHIFT_RECENT_WINDOW + 1]
        pooled = np.concatenate([recent, prior])
        std = float(np.std(pooled, ddof=1))
        recent_excess_shift.append(float((np.mean(recent) - np.mean(prior)) / std if std > 0 else float("nan")))
        recent_ic = rank_ic_values[index - SHIFT_RECENT_WINDOW + 1 : index + 1]
        prior_ic = rank_ic_values[index - SHIFT_RECENT_WINDOW - SHIFT_PRIOR_WINDOW + 1 : index - SHIFT_RECENT_WINDOW + 1]
        if np.isfinite(recent_ic).all() and np.isfinite(prior_ic).all():
            ic_std = float(np.std(np.concatenate([recent_ic, prior_ic]), ddof=1))
            recent_rank_shift.append(float((np.mean(recent_ic) - np.mean(prior_ic)) / ic_std if ic_std > 0 else float("nan")))
        else:
            recent_rank_shift.append(float("nan"))
        recent_score_std = score_std[index - SHIFT_RECENT_WINDOW + 1 : index + 1]
        prior_score_std = score_std[index - SHIFT_RECENT_WINDOW - SHIFT_PRIOR_WINDOW + 1 : index - SHIFT_RECENT_WINDOW + 1]
        dispersion_ratio.append(float(np.median(recent_score_std) / np.median(prior_score_std)) if np.isfinite(recent_score_std).all() and np.isfinite(prior_score_std).all() and np.median(prior_score_std) > 0 else float("nan"))
    out["excess_return_shift_standardized_recent20_vs_prior60"] = recent_excess_shift
    out["rank_ic_shift_standardized_recent20_vs_prior60"] = recent_rank_shift
    out["score_std_ratio_recent20_vs_prior60"] = dispersion_ratio
    out["score_dispersion_compressed"] = [bool(value < SCORE_DISPERSION_COMPRESSION_RATIO) if np.isfinite(value) else None for value in dispersion_ratio]
    return out


def _consecutive_true(values: np.ndarray) -> np.ndarray:
    """Length of the current trailing True run at every position."""

    output = np.zeros(len(values), dtype=int)
    current = 0
    for index, value in enumerate(np.asarray(values, dtype=bool)):
        current = current + 1 if value else 0
        output[index] = current
    return output


def _cusum_negative(values: np.ndarray, *, baseline_mean: float, baseline_std: float) -> np.ndarray:
    """One-sided negative CUSUM on a baseline-standardized quality series."""

    if not np.isfinite(baseline_std) or baseline_std <= 1e-15:
        return np.full(len(values), np.nan, dtype=float)
    output = np.zeros(len(values), dtype=float)
    current = 0.0
    for index, value in enumerate(np.asarray(values, dtype=float)):
        if not np.isfinite(value):
            output[index] = np.nan
            continue
        z_value = (value - baseline_mean) / baseline_std
        current = min(0.0, current + z_value + CUSUM_DRIFT)
        output[index] = current
    return output


def _circular_block_indices(n_obs: int, *, rows: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    count = int(math.ceil(n_obs / block_length))
    starts = rng.integers(0, n_obs, size=(rows, count), dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    return ((starts[:, :, None] + offsets[None, None, :]) % n_obs).reshape(rows, -1)[:, :n_obs]


def _calibrate_cusum_limit(values: np.ndarray, *, seed: int) -> dict[str, float]:
    """Calibrate a baseline-only diagnostic CUSUM limit via circular blocks."""

    baseline = np.asarray(values, dtype=float)
    baseline = baseline[np.isfinite(baseline)]
    if len(baseline) < 30:
        return {"baseline_n": float(len(baseline)), "baseline_mean": float("nan"), "baseline_std": float("nan"), "control_limit": float("nan")}
    mean = float(np.mean(baseline))
    std = float(np.std(baseline, ddof=1))
    if not np.isfinite(std) or std <= 1e-15:
        return {"baseline_n": float(len(baseline)), "baseline_mean": mean, "baseline_std": std, "control_limit": float("nan")}
    z = (baseline - mean) / std
    rng = np.random.default_rng(seed)
    indices = _circular_block_indices(len(z), rows=CUSUM_BOOTSTRAP_REPS, block_length=CUSUM_BLOCK_LENGTH, rng=rng)
    sampled = z[indices]
    current = np.zeros(CUSUM_BOOTSTRAP_REPS, dtype=float)
    minimum = np.zeros(CUSUM_BOOTSTRAP_REPS, dtype=float)
    for index in range(sampled.shape[1]):
        current = np.minimum(0.0, current + sampled[:, index] + CUSUM_DRIFT)
        minimum = np.minimum(minimum, current)
    return {
        "baseline_n": float(len(baseline)),
        "baseline_mean": mean,
        "baseline_std": std,
        "control_limit": float(np.quantile(-minimum, CUSUM_CONTROL_QUANTILE)),
    }


def _pettitt(values: np.ndarray, dates: pd.Series) -> dict[str, Any]:
    """Retrospective Pettitt change-point localization (diagnostic only)."""

    array = np.asarray(values, dtype=float)
    valid = np.isfinite(array)
    array = array[valid]
    date_values = pd.to_datetime(dates, errors="coerce").to_numpy()[valid]
    n_obs = len(array)
    if n_obs < 20:
        return {
            "n_obs": int(n_obs),
            "change_index": None,
            "change_date": None,
            "p_value": float("nan"),
            "pre_mean": float("nan"),
            "post_mean": float("nan"),
            "mean_shift": float("nan"),
        }
    ranks = pd.Series(array).rank(method="average").to_numpy(dtype=float)
    u_values = 2.0 * np.cumsum(ranks) - np.arange(1, n_obs + 1, dtype=float) * (n_obs + 1.0)
    index = int(np.argmax(np.abs(u_values)))
    statistic = float(abs(u_values[index]))
    p_value = min(1.0, float(2.0 * math.exp((-6.0 * statistic**2) / (n_obs**3 + n_obs**2))))
    pre = array[: index + 1]
    post = array[index + 1 :]
    return {
        "n_obs": int(n_obs),
        "change_index": int(index),
        "change_date": str(pd.Timestamp(date_values[index]).date()),
        "p_value": p_value,
        "pre_mean": float(np.mean(pre)),
        "post_mean": float(np.mean(post)) if len(post) else float("nan"),
        "mean_shift": float(np.mean(post) - np.mean(pre)) if len(post) else float("nan"),
    }


def _add_degradation_diagnostics(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add causal degradation flags and retrospective change-point summaries."""

    out = daily.copy().sort_values("trade_date").reset_index(drop=True)
    dates = pd.to_datetime(out["trade_date"], errors="coerce")
    baseline = out.loc[(dates >= DEV_START) & (dates <= DEV_END)]
    rank_values_raw = pd.to_numeric(out["rank_ic"], errors="coerce").to_numpy(dtype=float)
    rank_quality_ok = ~out["strategy_ic_coverage_insufficient"].fillna(True).to_numpy(dtype=bool)
    rank_values = np.where(rank_quality_ok, rank_values_raw, np.nan)
    alpha_values = pd.to_numeric(out["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float)
    baseline_quality_ok = ~baseline["strategy_ic_coverage_insufficient"].fillna(True).to_numpy(dtype=bool)
    base_rank = pd.to_numeric(baseline["rank_ic"], errors="coerce").to_numpy(dtype=float)[baseline_quality_ok]
    base_alpha = pd.to_numeric(baseline.get("rolling60_alpha_hac_t", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    rank_mean = float(np.mean(base_rank)) if len(base_rank) else float("nan")
    rank_std = float(np.std(base_rank, ddof=1)) if len(base_rank) > 1 else float("nan")
    alpha_mean = float(np.mean(base_alpha)) if len(base_alpha) else float("nan")
    alpha_std = float(np.std(base_alpha, ddof=1)) if len(base_alpha) > 1 else float("nan")
    # The 2025 period establishes the reference distribution; the monitoring
    # accumulator is reset at the first reporting day so baseline noise cannot
    # manufacture a future alarm.
    rank_cusum = np.full(len(out), np.nan, dtype=float)
    alpha_cusum = np.full(len(out), np.nan, dtype=float)
    candidate_name = str(out["candidate"].iloc[0]) if "candidate" in out.columns and len(out) else ""
    candidate_seed = int(hashlib.sha256(candidate_name.encode("utf-8")).hexdigest()[:8], 16)
    rank_limit = _calibrate_cusum_limit(base_rank, seed=20_260_902 + candidate_seed % 10_000)
    alpha_limit = _calibrate_cusum_limit(base_alpha, seed=20_261_902 + candidate_seed % 10_000)
    reporting_mask = dates >= REPORT_START
    if reporting_mask.any():
        rank_cusum[reporting_mask.to_numpy()] = _cusum_negative(
            rank_values[reporting_mask.to_numpy()], baseline_mean=rank_mean, baseline_std=rank_std
        )
        alpha_cusum[reporting_mask.to_numpy()] = _cusum_negative(
            alpha_values[reporting_mask.to_numpy()], baseline_mean=alpha_mean, baseline_std=alpha_std
        )
    out["rank_ic_negative_cusum"] = rank_cusum
    out["alpha60_negative_cusum"] = alpha_cusum
    out["rank_ic60_nonpositive"] = pd.to_numeric(out["rolling60_rank_ic_mean"], errors="coerce").to_numpy(dtype=float) <= 0.0
    out["alpha60_non_significant"] = pd.to_numeric(out["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float) < ALPHA_POSITIVE_EVIDENCE_T
    out["alpha60_nonpositive"] = pd.to_numeric(out["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float) <= 0.0
    out["alpha60_negative_evidence"] = pd.to_numeric(out["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float) <= ALPHA_NEGATIVE_EVIDENCE_T
    out["rank_ic_cusum_control_limit"] = rank_limit["control_limit"]
    out["alpha60_cusum_control_limit"] = alpha_limit["control_limit"]
    out["rank_ic_cusum_alarm"] = (
        (rank_cusum <= -rank_limit["control_limit"]) & reporting_mask.to_numpy()
        if np.isfinite(rank_limit["control_limit"])
        else False
    )
    out["alpha60_cusum_alarm"] = (
        (alpha_cusum <= -alpha_limit["control_limit"]) & reporting_mask.to_numpy()
        if np.isfinite(alpha_limit["control_limit"])
        else False
    )
    out["score_dispersion_compression_alarm"] = out["score_dispersion_compressed"].fillna(False).astype(bool)
    out["rank_ic60_nonpositive_persistent5"] = _consecutive_true(out["rank_ic60_nonpositive"].to_numpy(dtype=bool)) >= PERSISTENCE_DAYS
    out["alpha60_non_significant_persistent5"] = _consecutive_true(out["alpha60_non_significant"].to_numpy(dtype=bool)) >= PERSISTENCE_DAYS
    out["alpha60_nonpositive_persistent5"] = _consecutive_true(out["alpha60_nonpositive"].to_numpy(dtype=bool)) >= PERSISTENCE_DAYS
    out["alpha60_negative_evidence_persistent5"] = _consecutive_true(out["alpha60_negative_evidence"].to_numpy(dtype=bool)) >= PERSISTENCE_DAYS
    out["prediction_health_watch"] = (
        out["rank_ic60_nonpositive_persistent5"] | out["rank_ic_cusum_alarm"]
    )
    out["alpha_health_watch"] = out["alpha60_non_significant_persistent5"]
    out["alpha_health_degradation"] = (
        out["alpha60_negative_evidence_persistent5"]
    )
    out["alpha_decay_cusum_watch"] = out["alpha60_cusum_alarm"]
    out["health_watch_diagnostic"] = (
        out["prediction_health_watch"]
        | out["alpha_health_watch"]
        | out["alpha_decay_cusum_watch"]
        | out["score_dispersion_compression_alarm"]
    )
    # A fixed strategy is marked as a degradation diagnostic only when the
    # prediction layer and the realized-alpha layer deteriorate together.
    out["health_degradation_diagnostic"] = (
        out["prediction_health_watch"] & out["alpha_health_degradation"]
    )
    alpha_t_values = pd.to_numeric(out["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float)
    alpha_status: list[str] = []
    health_state: list[str] = []
    for index, alpha_t in enumerate(alpha_t_values):
        if not np.isfinite(alpha_t):
            alpha_status.append("DATA_INSUFFICIENT")
        elif alpha_t <= ALPHA_NEGATIVE_EVIDENCE_T:
            alpha_status.append("NEGATIVE_ALPHA_EVIDENCE")
        elif alpha_t >= ALPHA_POSITIVE_EVIDENCE_T:
            alpha_status.append("POSITIVE_ALPHA_EVIDENCE")
        else:
            alpha_status.append("ALPHA_NOT_ESTABLISHED")
        if bool(out.loc[index, "strategy_ic_coverage_insufficient"]):
            health_state.append("DATA_INCONCLUSIVE")
        elif not np.isfinite(alpha_t) or not np.isfinite(pd.to_numeric(pd.Series([out.loc[index, "rolling60_rank_ic_mean"]]), errors="coerce").iloc[0]):
            health_state.append("WARMUP_DATA_INSUFFICIENT")
        elif bool(out.loc[index, "alpha_health_degradation"]):
            health_state.append("NEGATIVE_ALPHA_EVIDENCE")
        elif bool(out.loc[index, "alpha_decay_cusum_watch"]):
            health_state.append("ALPHA_DECAY_WATCH")
        elif bool(out.loc[index, "alpha_health_watch"]):
            health_state.append("ALPHA_NOT_ESTABLISHED")
        elif bool(out.loc[index, "prediction_health_watch"]):
            health_state.append("WATCH_PREDICTION")
        elif bool(out.loc[index, "score_dispersion_compression_alarm"]):
            health_state.append("SCORE_DISPERSION_WATCH")
        else:
            health_state.append("NORMAL")
    out["alpha60_status"] = alpha_status
    out["health_state"] = health_state
    change_rows: list[dict[str, Any]] = []
    candidate = candidate_name
    for scope, mask in (
        ("development_2025", (dates >= DEV_START) & (dates <= DEV_END)),
        ("reporting_2026", dates >= REPORT_START),
        ("overall", pd.Series(True, index=out.index)),
    ):
        scoped = out.loc[mask].copy()
        if scoped.empty:
            continue
        for metric, values in (
            ("rank_ic", pd.to_numeric(scoped["rank_ic"], errors="coerce").to_numpy(dtype=float)),
            ("rolling60_alpha_hac_t", pd.to_numeric(scoped["rolling60_alpha_hac_t"], errors="coerce").to_numpy(dtype=float)),
            ("strategy_return", pd.to_numeric(scoped["strategy_return"], errors="coerce").to_numpy(dtype=float)),
        ):
            result = _pettitt(values, scoped["trade_date"])
            change_rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "metric": metric,
                    **result,
                    "interpretation": "retrospective localization only; not a real-time trigger",
                }
            )
    return out, pd.DataFrame(change_rows)


def _health_summary_rows(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize one fixed candidate without comparing it with peers."""

    dates = pd.to_datetime(daily["trade_date"], errors="coerce")
    candidate = str(daily["candidate"].iloc[0])
    rows: list[dict[str, Any]] = []
    for scope, mask in (
        ("development_2025", (dates >= DEV_START) & (dates <= DEV_END)),
        ("reporting_2026", dates >= REPORT_START),
        ("overall", pd.Series(True, index=daily.index)),
    ):
        frame = daily.loc[mask].copy()
        if frame.empty:
            continue
        last = frame.iloc[-1]

        def number(name: str) -> float:
            value = pd.to_numeric(pd.Series([last.get(name)]), errors="coerce").iloc[0]
            return float(value) if np.isfinite(value) else float("nan")

        combined_degradation = frame["health_degradation_diagnostic"].fillna(False).astype(bool)
        partial = frame["partial_label_coverage"].fillna(False).astype(bool)
        coverage_low = frame["strategy_ic_coverage_insufficient"].fillna(False).astype(bool)
        rows.append(
            {
                "candidate": candidate,
                "scope": scope,
                "start": str(frame["trade_date"].iloc[0].date()),
                "end": str(frame["trade_date"].iloc[-1].date()),
                "n_days": int(len(frame)),
                "rank_ic_mean": float(pd.to_numeric(frame["rank_ic"], errors="coerce").mean()),
                "rank_ic_rolling60_latest": number("rolling60_rank_ic_mean"),
                "rank_ic_rolling60_min": float(pd.to_numeric(frame["rolling60_rank_ic_mean"], errors="coerce").min()),
                "rank_icir_rolling60_latest": number("rolling60_rank_icir"),
                "alpha60_hac_t_latest": number("rolling60_alpha_hac_t"),
                "alpha60_hac_t_min": float(pd.to_numeric(frame["rolling60_alpha_hac_t"], errors="coerce").min()),
                "strategy_sharpe_rolling60_latest": number("rolling60_strategy_sharpe"),
                "strategy_return_rolling60_latest_bp": number("rolling60_strategy_mean_return_bp"),
                "beta_rolling60_latest": number("rolling60_beta"),
                "residual_vol_rolling60_latest": number("rolling60_residual_annualized_vol"),
                "max_drawdown_rolling60_latest": number("rolling60_strategy_max_drawdown"),
                "score_std_ratio_latest": number("score_std_ratio_recent20_vs_prior60"),
                "excess_return_shift_standardized_latest": number("excess_return_shift_standardized_recent20_vs_prior60"),
                "rank_ic_shift_standardized_latest": number("rank_ic_shift_standardized_recent20_vs_prior60"),
                "combined_degradation_diagnostic_day_count": int(combined_degradation.sum()),
                "combined_degradation_diagnostic_fraction": float(combined_degradation.mean()),
                "prediction_watch_day_count": int(frame["prediction_health_watch"].fillna(False).astype(bool).sum()),
                "alpha_watch_day_count": int(frame["alpha_health_watch"].fillna(False).astype(bool).sum()),
                "alpha_degradation_day_count": int(frame["alpha_health_degradation"].fillna(False).astype(bool).sum()),
                "alpha_decay_cusum_watch_day_count": int(frame["alpha_decay_cusum_watch"].fillna(False).astype(bool).sum()),
                "cusum_rank_control_limit": number("rank_ic_cusum_control_limit"),
                "cusum_alpha_control_limit": number("alpha60_cusum_control_limit"),
                "partial_label_coverage_day_count": int(partial.sum()),
                "strategy_ic_coverage_insufficient_day_count": int(coverage_low.sum()),
                "latest_health_degradation_diagnostic": bool(last["health_degradation_diagnostic"]),
                "latest_health_state": str(last["health_state"]),
                "latest_alpha60_status": str(last["alpha60_status"]),
                "latest_partial_label_coverage": bool(last["partial_label_coverage"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "candidate"]).reset_index(drop=True)


def _load_strategy_returns(source: CandidateSource, expected_days: pd.DatetimeIndex) -> pd.DataFrame:
    metrics = _read_json(source.metrics_path)
    path = Path(str(metrics.get("daily_returns_path", "")))
    if not path.is_file():
        raise HealthValidationError(f"missing strategy daily returns for {source.candidate}: {path}")
    frame = pd.read_csv(path)
    required = {"trade_date", "day_return", "benchmark_return", "benchmark_method", "count", "fill_rate", "cash_weight"}
    if not required.issubset(frame.columns):
        raise HealthValidationError(f"strategy daily returns missing columns {sorted(required - set(frame.columns))}: {path}")
    out = frame.loc[:, sorted(required)].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for column in ("day_return", "benchmark_return", "count", "fill_rate", "cash_weight"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if not pd.DatetimeIndex(out["trade_date"]).equals(expected_days) or out["trade_date"].duplicated().any():
        raise HealthValidationError(f"strategy daily returns dates differ from frozen R88 calendar: {path}")
    if not np.isfinite(out[["day_return", "benchmark_return", "count", "fill_rate", "cash_weight"]].to_numpy(dtype=float)).all():
        raise HealthValidationError(f"strategy daily returns contain non-finite values: {path}")
    methods = sorted(set(out["benchmark_method"].astype(str)))
    if methods != [BENCHMARK_METHOD]:
        raise HealthValidationError(f"strategy daily returns benchmark method differs: {methods}")
    if not (out["count"] == TOP_K).all() or not np.allclose(out["fill_rate"], 1.0) or not np.allclose(out["cash_weight"], 0.0):
        raise HealthValidationError(f"strategy daily returns do not match fixed Top20/full-fill contract: {path}")
    return out


def _render_health_report(summary: pd.DataFrame, changes: pd.DataFrame, *, evidence: Mapping[str, Any]) -> str:
    lines = [
        "# R88 固定策略健康度与性能衰减面板",
        "",
        "## 研究范围",
        "",
        "- 每个候选独立评估，不根据候选间排名选模。",
        "- 主预测质量来自冻结策略 `ic.csv`（严格可交易 universe 与 cycle net return）；当前 label parquet 仅用于原始标签覆盖审计。",
        "- 20 日指标用于早期观察，60 日指标作为主要健康窗口，120 日作为背景；所有告警都是研究诊断，不会自动停用或切换实盘。",
        "",
        "## 代表性固定策略状态（2026 reporting）",
        "",
        "| candidate | 60d RankIC | 60d ICIR | 60d Alpha HAC t | 60d Sharpe | 60d Beta | 60d 回归残差波动 | 60d MDD | 最近20/前60 RankIC 标准化变化 | 预测观察天数 | Alpha观察天数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    report = summary.loc[summary["scope"] == "reporting_2026"].copy().sort_values("candidate")
    for row in report.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.rank_ic_rolling60_latest:.4f} | {row.rank_icir_rolling60_latest:.3f} | "
            f"{row.alpha60_hac_t_latest:.3f} | {row.strategy_sharpe_rolling60_latest:.3f} | "
            f"{row.beta_rolling60_latest:.3f} | {row.residual_vol_rolling60_latest:.2%} | "
            f"{row.max_drawdown_rolling60_latest:.2%} | {row.rank_ic_shift_standardized_latest:.2f} | {row.prediction_watch_day_count} | {row.alpha_watch_day_count} |"
        )
    lines.extend(["", "## 告警定义", "", "- `rank_ic60_nonpositive_persistent5`：滚动 60 日 RankIC 非正连续至少 5 日。", "- `alpha60_non_significant_persistent5`：滚动 60 日 Alpha HAC t 低于 1.645 连续至少 5 日，表示正 Alpha 证据不足，不等同于负 Alpha。", "- `alpha_decay_cusum_watch`：Alpha t 相对固定 2025 基线持续下移的 CUSUM 观察，不等同于当前 Alpha 显著为负。", "- `NEGATIVE_ALPHA_EVIDENCE`：只由滚动 60 日 Alpha HAC t≤-1.645 连续至少 5 日触发。", "- `rank_ic_cusum_alarm`：预测 RankIC 相对固定 2025 基线的负向 CUSUM 观察。", "- `score_dispersion_compression_alarm`：最近 20 日分数截面波动低于此前 60 日的一半。", "- `2026-08-11/12` 的当前 label 覆盖异常只标为 `partial_label_coverage`，不当作模型衰减。", "", "## 变点结果", ""])
    for row in changes.itertuples(index=False):
        if row.scope == "reporting_2026":
            lines.append(f"- {row.candidate} / {row.metric}：Pettitt 日期={row.change_date}，p={row.p_value:.4f}，均值变化={row.mean_shift:.6g}（仅事后定位）。")
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- 原始 score 的绝对数值跨 refit 不可直接比较；应优先看 RankIC、ICIR、Top20 稳定性和滚动 Alpha。",
        "- Alpha t 的重复滚动观察会产生多重检验问题；本面板用于发现衰减候选和日期，不是正式停用阈值。",
        "- 当前面板不改变任何 live config、FactorStore、DB、scheduler 或交易规则。",
        "",
        f"- 输入 score 候选数：{evidence['candidate_count']}；每候选 score 日：{evidence['candidate_days']}；label 覆盖异常日：{evidence['partial_label_days']}。",
        f"- 输出 run：`{evidence['output_root']}`。",
    ])
    return "\n".join(lines) + "\n"


def _render_candidate_health_table(summary: pd.DataFrame) -> str:
    """Render one independent latest-state row for all 30 candidates."""

    report = summary.loc[summary["scope"] == "reporting_2026"].copy().sort_values("candidate")
    lines = [
        "# R88 30 个固定策略健康状态",
        "",
        "每行是一个固定候选自身的 2026 reporting 健康摘要；没有候选间排名或选优操作。",
        "",
        "| candidate | 60d RankIC | 60d ICIR | 60d Alpha HAC t | 60d Sharpe | 60d Beta | RankIC观察天数 | Alpha观察天数 | Alpha恶化天数 | 最新状态 | 标签覆盖异常日 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in report.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.rank_ic_rolling60_latest:.4f} | {row.rank_icir_rolling60_latest:.3f} | "
            f"{row.alpha60_hac_t_latest:.3f} | {row.strategy_sharpe_rolling60_latest:.3f} | "
            f"{row.beta_rolling60_latest:.3f} | {row.prediction_watch_day_count} | "
            f"{row.alpha_watch_day_count} | {row.alpha_degradation_day_count} | "
            f"{row.latest_health_state} | {row.partial_label_coverage_day_count} |"
        )
    return "\n".join(lines) + "\n"


def _build_health_inputs(
    *,
    study_root: Path,
    phase1_root: Path,
    label_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load all fixed candidates and produce daily/summary health frames."""

    frozen: FrozenInputs = load_frozen_inputs(study_root=study_root, phase1_root=phase1_root, regsim_path=DEFAULT_REGSIM_PATH)
    label_cache, label_evidence = _load_label_cache(label_root, frozen.expected_days)
    all_daily: list[pd.DataFrame] = []
    all_changes: list[pd.DataFrame] = []
    all_summary: list[pd.DataFrame] = []
    score_evidence: list[dict[str, Any]] = []
    for source in frozen.sources:
        quality = _load_candidate_score_quality(
            source,
            study_root=frozen.study_root,
            expected_days=frozen.expected_days,
            label_root=label_root,
            label_cache=label_cache,
        )
        returns = _load_strategy_returns(source, frozen.expected_days)
        daily = quality.daily.merge(
            returns[["trade_date", "day_return", "benchmark_return", "count", "fill_rate", "cash_weight"]],
            on="trade_date",
            how="inner",
            validate="one_to_one",
        )
        if len(daily) != len(frozen.expected_days):
            raise HealthValidationError(f"health merge lost days for {source.candidate}")
        daily = daily.rename(columns={"day_return": "strategy_return", "benchmark_return": "strategy_benchmark_return"})
        daily.insert(0, "candidate", source.candidate)
        daily["partial_label_coverage"] = daily["partial_label_coverage"].astype(bool)
        daily = _add_rolling_metrics(
            daily,
            strategy_returns=daily["strategy_return"].to_numpy(dtype=float),
            benchmark_returns=daily["strategy_benchmark_return"].to_numpy(dtype=float),
        )
        daily, changes = _add_degradation_diagnostics(daily)
        summary = _health_summary_rows(daily)
        all_daily.append(daily)
        all_changes.append(changes)
        all_summary.append(summary)
        score_evidence.append({"candidate": source.candidate, **quality.coverage, "score_root": str(quality.score_root)})
    daily_frame = pd.concat(all_daily, ignore_index=True).sort_values(["candidate", "trade_date"]).reset_index(drop=True)
    changes_frame = pd.concat(all_changes, ignore_index=True).sort_values(["candidate", "scope", "metric"]).reset_index(drop=True)
    summary_frame = pd.concat(all_summary, ignore_index=True).sort_values(["scope", "candidate"]).reset_index(drop=True)
    evidence = {
        "schema": "r88_single_model_health_input_evidence/v1",
        "study_id": STUDY_ID,
        "research_only": True,
        "promotion_allowed": False,
        "study_root": str(frozen.study_root),
        "phase1_root": str(frozen.phase1_root),
        "candidate_count": len(frozen.sources),
        "candidate_days": len(frozen.expected_days),
        "candidate_start": str(frozen.expected_days.min().date()),
        "candidate_end": str(frozen.expected_days.max().date()),
        "study_plan_sha256": _sha256_file(frozen.study_root / "study_plan.json"),
        "study_integrity_sha256": _sha256_file(frozen.study_root / "study_integrity.json"),
        "study_integrity_verification": frozen.integrity_verification,
        "phase1_input_evidence_sha256": _sha256_file(frozen.phase1_root / "input_evidence.json"),
        "factor_generation": frozen.phase1_evidence.get("factor_generation"),
        "score_candidates": score_evidence,
        "label_evidence": label_evidence,
        "partial_label_days": sorted({pd.Timestamp(row["trade_date"]).date().isoformat() for frame in all_daily for row in frame.to_dict("records") if bool(row.get("partial_label_coverage"))}),
        "primary_prediction_source": "frozen strategy ic.csv (strict strategy universe and strict cycle net return)",
        "secondary_label_source": "current label_data y for raw score/label coverage and dispersion audit only",
        "strategy_return_source": "frozen daily_returns.csv (Top20 strict cycle net return)",
        "reused": ["R88 candidate score CSVs", "R88 frozen strategy ic.csv", "R88 frozen daily_returns.csv", "current label parquet for audit", "Phase1 input evidence"],
        "missing": [],
        "invalid": [],
    }
    return evidence, {"candidate_count": len(frozen.sources)}, daily_frame, changes_frame, summary_frame


def run_health_panel(
    *,
    study_root: Path = DEFAULT_STUDY_ROOT,
    phase1_root: Path = Path(r"D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1/phase1_20260901_r3"),
    label_root: Path = DEFAULT_LABEL_ROOT,
    output_root: Path = DEFAULT_HEALTH_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
) -> Path:
    run_root = _make_run_root(output_root, run_name)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(exist_ok=False)
    _write_json(run_root / "run_status.json", {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(), "research_only": True, "database_writes": False, "live_runtime_called": False, "scheduler_called": False, "factor_store_writes": False, "model_scoring_called": False, "model_training_called": False})
    try:
        evidence, _, daily, changes, summary = _build_health_inputs(study_root=study_root.resolve(), phase1_root=phase1_root.resolve(), label_root=label_root.resolve())
        daily.to_csv(run_root / "daily_health_panel.csv", index=False)
        changes.to_csv(run_root / "change_point_summary.csv", index=False)
        summary.to_csv(run_root / "health_summary.csv", index=False)
        partial = daily.loc[daily["partial_label_coverage"]].copy()
        partial.to_csv(run_root / "partial_label_coverage_audit.csv", index=False)
        _write_json(run_root / "input_evidence.json", evidence)
        method_contract = {
            "schema": "r88_single_model_health_method_contract/v1",
            "unit_of_analysis": "each fixed candidate independently; no candidate selection or cross-candidate ranking",
            "primary_prediction_metrics": "frozen strategy ic.csv ic/rank_ic/count",
            "secondary_raw_label_metrics": "score CSV joined to current 14:42 label y; partial joins are audit-only and excluded from rolling raw-label summaries",
            "strategy_metrics": "frozen daily_returns.csv day_return/benchmark_return; rolling OLS with Bartlett HAC",
            "windows": {"early_warning": 20, "primary_health": 60, "background": 120, "recent_vs_prior": "recent20 vs immediately preceding60"},
            "states": {
                "prediction_watch": "rolling60 RankIC nonpositive for 5 consecutive days or baseline CUSUM alarm",
                "alpha_watch": "rolling60 HAC alpha t < 1.645 for 5 consecutive days",
                "alpha_decay_cusum_watch": "Alpha CUSUM exceeds its 2025-baseline block-bootstrap control limit; relative decay watch only",
                "negative_alpha_evidence": "rolling60 alpha HAC t <= -1.645 for 5 consecutive days",
                "data_inconclusive": "strategy IC count below max(100, 0.5*prior60 median count)",
                "dispersion_watch": "recent20 score std / prior60 score std < 0.5",
                "combined_degradation_diagnostic": "prediction watch and strict negative-alpha evidence simultaneously",
            },
            "cusum": {"baseline": "2025 development", "drift_k": CUSUM_DRIFT, "block_length": CUSUM_BLOCK_LENGTH, "bootstrap_reps": CUSUM_BOOTSTRAP_REPS, "control_quantile": CUSUM_CONTROL_QUANTILE, "purpose": "diagnostic control limit, not a 5% hypothesis test"},
            "change_point": {"method": "Pettitt retrospective localization", "metrics": ["rank_ic", "rolling60_alpha_hac_t", "strategy_return"], "realtime_use": False},
            "live_effect": False,
        }
        _write_json(run_root / "method_contract.json", method_contract)
        (run_root / "RESULTS.md").write_text(_render_health_report(summary, changes, evidence={**evidence, "output_root": str(run_root), "partial_label_days": len(evidence["partial_label_days"])}), encoding=RESULTS_ENCODING)
        (run_root / "CANDIDATE_HEALTH_RESULTS.md").write_text(
            _render_candidate_health_table(summary), encoding=RESULTS_ENCODING
        )
        manifest = {
            "schema": "r88_single_model_health_run/v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "promotion_allowed": False,
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "factor_store_writes": False,
            "model_scoring_called": False,
            "model_training_called": False,
            "candidate_selection_performed": False,
            "inputs": evidence,
            "method_contract": method_contract,
            "git": {"head": _safe_git(["rev-parse", "HEAD"]), "status_porcelain": _safe_git(["status", "--short"])},
        }
        _write_json(run_root / "run_manifest.json", manifest)
        _write_json(run_root / "run_status.json", {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_root": str(run_root), "candidate_count": len(evidence["score_candidates"]), "research_only": True, "database_writes": False, "live_runtime_called": False, "scheduler_called": False, "factor_store_writes": False, "model_scoring_called": False, "model_training_called": False})
    except Exception as exc:
        _write_json(run_root / "run_status.json", {"status": "failed", "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(exc).__name__, "error": str(exc), "research_only": True, "database_writes": False, "live_runtime_called": False, "scheduler_called": False})
        raise
    return run_root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--phase1-root", type=Path, default=Path(r"D:/cbond_on/research_scratch/r88_single_model_validation_20260901_r1/phase1_20260901_r3"))
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_HEALTH_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = run_health_panel(study_root=args.study_root, phase1_root=args.phase1_root, label_root=args.label_root, output_root=args.output_root, run_name=args.run_name)
    print(f"R88 strategy health panel completed: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
