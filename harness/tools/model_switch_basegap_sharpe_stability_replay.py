"""Research-only robust-Sharpe scorecard for the frozen BaseGap grid.

This is a *new research stage*.  It never calls the live runtime and never
changes the current selector.  The input is the immutable daily replay from
the completed 51-point BaseGap grid.  It ranks the existing grid on a common
active-date design cohort using robust cross-block Sharpe stability and
relative performance against fixed Regsim.

The stage is deliberately retrospective calibration only: validation and final
OOS are reported after the one-time design choice and cannot select a different
parameter.  A later forward-shadow window is required for any live decision.
"""

from __future__ import annotations

import argparse
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


DEFAULT_SOURCE_RUN = Path(
    r"D:\cbond_on\research_scratch\model_switch_basegap_tuning_20260811"
    r"\run_stage_a_20260811_r3"
)
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\research_scratch\model_switch_basegap_sharpe_stability_20260825")

MODELS = ("Regsim", "Ensemble", "HL20")
DESIGN = ("2024-05-08", "2025-06-30")
VALIDATION = ("2025-07-01", "2025-12-31")
FINAL_OOS = ("2026-01-01", "2026-07-30")
BLOCK_COUNT = 5
ANNUALIZATION = 252.0
MAD_SCALE = 1.4826
ROLLING_SHARPE_WINDOW = 40
ROLLING_SHARPE_MIN_PERIODS = 20
MIN_ABSOLUTE_Q_ADVANTAGE = 0.10
MIN_POSITIVE_RELATIVE_BLOCKS = 4
MIN_RELATIVE_BOOTSTRAP_PROBABILITY = 0.75
BLOCK_BOOTSTRAP_LENGTH = 5
BLOCK_BOOTSTRAP_REPS = 20_000
RANDOM_SEED = 20_260_825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _safe_git(args: Sequence[str]) -> str | None:
    completed = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _resolve_inside(path: Path, parent: Path, *, description: str) -> Path:
    resolved = path.resolve()
    root = parent.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description} must be under {root}: {resolved}") from exc
    return resolved


def _assert_source_run(source_run: Path) -> Path:
    resolved = source_run.resolve()
    expected = DEFAULT_SOURCE_RUN.resolve()
    if resolved != expected:
        raise ValueError(f"source run is locked to the verified BaseGap r3 replay: {expected}")
    required = ("daily_selector_replay.csv", "summary_metrics.csv", "run_manifest.json", "input_verification.json")
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete source replay {resolved}: {missing}")
    return resolved


def _assert_output_root(path: Path) -> Path:
    return _resolve_inside(path, DEFAULT_OUTPUT_ROOT, description="research output")


def _make_run_root(output_root: Path, run_name: str | None) -> Path:
    root = _assert_output_root(output_root)
    name = run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"run name must be one plain path component, got {name!r}")
    run_root = _assert_output_root(root / name)
    if run_root == DEFAULT_OUTPUT_ROOT.resolve():
        raise ValueError("a new run leaf below the research output root is required")
    if run_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing research run: {run_root}")
    return run_root


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")


def load_frozen_daily_replay(source_run: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only the r3 frozen replay and verify its recorded input evidence."""

    source_run = _assert_source_run(source_run)
    manifest = json.loads((source_run / "run_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((source_run / "input_verification.json").read_text(encoding="utf-8"))
    if not bool(manifest.get("source_input_verification", {}).get("verified_file_count")):
        raise ValueError("source replay does not record frozen-input verification")
    daily_path = source_run / "daily_selector_replay.csv"
    daily = pd.read_csv(daily_path)
    _require_columns(
        daily,
        (
            "score_day",
            "variant",
            "selected_return",
            "realized_return_Regsim",
            "selection_active",
            "execution_metadata_complete",
        ),
        label="daily selector replay",
    )
    daily["score_day"] = pd.to_datetime(daily["score_day"], errors="coerce")
    if daily["score_day"].isna().any():
        raise ValueError("daily selector replay has invalid score_day")
    for column in ("selected_return", "realized_return_Regsim"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily["selection_active"] = daily["selection_active"].astype(bool)
    daily["execution_metadata_complete"] = daily["execution_metadata_complete"].astype(bool)
    daily = daily.sort_values(["variant", "score_day"]).reset_index(drop=True)
    variants = daily["variant"].drop_duplicates().tolist()
    if len(variants) != 51:
        raise ValueError(f"expected the fixed 51-point grid, got {len(variants)} variants")
    if daily.duplicated(["variant", "score_day"]).any():
        raise ValueError("daily selector replay contains duplicate variant/date rows")
    evidence = {
        "source_run": str(source_run),
        "daily_replay": str(daily_path),
        "daily_replay_sha256": sha256(daily_path),
        "source_manifest_sha256": sha256(source_run / "run_manifest.json"),
        "source_input_verification_sha256": sha256(source_run / "input_verification.json"),
        "verified_frozen_files": int(verification.get("verified_file_count", 0)),
        "variants": variants,
    }
    return daily, evidence


def _scope(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return frame.loc[(frame["score_day"] >= pd.Timestamp(start)) & (frame["score_day"] <= pd.Timestamp(end))].copy()


def common_active_design_days(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Intersection avoids a lookback warm-up date advantage in selector ranking."""

    design = _scope(daily, *DESIGN)
    variants = design["variant"].drop_duplicates().tolist()
    if len(variants) != 51:
        raise ValueError("design slice no longer contains the fixed grid")
    common: set[pd.Timestamp] | None = None
    for variant in variants:
        dates = set(
            design.loc[
                (design["variant"] == variant)
                & design["selection_active"]
                & design["execution_metadata_complete"],
                "score_day",
            ].tolist()
        )
        common = dates if common is None else common & dates
    result = pd.DatetimeIndex(sorted(common or []))
    if len(result) < BLOCK_COUNT * ROLLING_SHARPE_MIN_PERIODS:
        raise ValueError(
            f"common active design sample too short for {BLOCK_COUNT} blocks and rolling Sharpe: {len(result)} days"
        )
    return result


def _sharpe(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(series) < 2:
        return float("nan")
    std = float(series.std(ddof=1))
    # A near-constant series has no meaningful risk-adjusted return.  Treat it
    # as ineligible instead of letting floating-point noise create a huge SR.
    return float(series.mean() / std * math.sqrt(ANNUALIZATION)) if math.isfinite(std) and std > 1e-12 else float("nan")


def _mad_score(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        return {"median": float("nan"), "mad": float("nan"), "q": float("nan"), "min": float("nan"), "std": float("nan")}
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return {
        "median": median,
        "mad": mad,
        "q": float(median - MAD_SCALE * mad),
        "min": float(np.min(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def _rolling_sharpe_positive_ratio(values: pd.Series) -> dict[str, float | int]:
    series = pd.to_numeric(values, errors="coerce").astype(float)
    rolling = series.rolling(ROLLING_SHARPE_WINDOW, min_periods=ROLLING_SHARPE_MIN_PERIODS)
    std = rolling.std(ddof=1)
    sharpe = rolling.mean().div(std).mul(math.sqrt(ANNUALIZATION))
    valid = sharpe.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return {"mean": float("nan"), "std": float("nan"), "positive_ratio": float("nan"), "n": 0}
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
        "positive_ratio": float((valid > 0.0).mean()),
        "n": int(len(valid)),
    }


def _block_labels(days: pd.DatetimeIndex) -> list[np.ndarray]:
    blocks = [np.asarray(block, dtype="datetime64[ns]") for block in np.array_split(days.to_numpy(), BLOCK_COUNT)]
    if any(len(block) == 0 for block in blocks):
        raise ValueError("a design stability block is empty")
    return blocks


def _variant_common_frame(daily: pd.DataFrame, variant: str, common_days: pd.DatetimeIndex) -> pd.DataFrame:
    frame = daily.loc[daily["variant"] == variant].copy().set_index("score_day")
    if not set(common_days).issubset(set(frame.index)):
        raise ValueError(f"variant {variant} lacks a common active design day")
    frame = frame.loc[common_days].copy()
    if not (frame["selection_active"] & frame["execution_metadata_complete"]).all():
        raise ValueError(f"variant {variant} violates common active design contract")
    if frame[["selected_return", "realized_return_Regsim"]].isna().any().any():
        raise ValueError(f"variant {variant} has missing common design returns")
    frame["delta_return"] = frame["selected_return"] - frame["realized_return_Regsim"]
    return frame


def build_sharpe_stability_scorecard(
    daily: pd.DataFrame,
    *,
    bootstrap_reps: int = BLOCK_BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Return a one-row scorecard and long block evidence for the fixed grid."""

    common_days = common_active_design_days(daily)
    blocks = _block_labels(common_days)
    score_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for variant in daily["variant"].drop_duplicates().tolist():
        frame = _variant_common_frame(daily, variant, common_days)
        selected_blocks: list[float] = []
        regsim_blocks: list[float] = []
        delta_blocks: list[float] = []
        delta_means_bp: list[float] = []
        positive_blocks = 0
        for index, block in enumerate(blocks, start=1):
            block_frame = frame.loc[pd.DatetimeIndex(block)]
            selected_sharpe = _sharpe(block_frame["selected_return"])
            regsim_sharpe = _sharpe(block_frame["realized_return_Regsim"])
            delta_sharpe = _sharpe(block_frame["delta_return"])
            delta_mean_bp = float(block_frame["delta_return"].mean() * 10_000.0)
            selected_blocks.append(selected_sharpe)
            regsim_blocks.append(regsim_sharpe)
            delta_blocks.append(delta_sharpe)
            delta_means_bp.append(delta_mean_bp)
            positive_blocks += int(delta_mean_bp > 0.0)
            block_rows.append(
                {
                    "variant": variant,
                    "block": index,
                    "start": str(block_frame.index.min().date()),
                    "end": str(block_frame.index.max().date()),
                    "n_days": int(len(block_frame)),
                    "selected_sharpe": selected_sharpe,
                    "regsim_sharpe": regsim_sharpe,
                    "relative_sharpe": delta_sharpe,
                    "relative_mean_bp": delta_mean_bp,
                }
            )
        selected_summary = _mad_score(selected_blocks)
        regsim_summary = _mad_score(regsim_blocks)
        relative_sharpe_summary = _mad_score(delta_blocks)
        relative_mean_summary = _mad_score(delta_means_bp)
        rolling = _rolling_sharpe_positive_ratio(frame["selected_return"])
        score_rows.append(
            {
                "variant": variant,
                "lookback_days": int(frame["lookback_days"].iloc[0]),
                "nearest_k": int(frame["nearest_k"].iloc[0]),
                "min_periods": int(frame["min_periods"].iloc[0]),
                "metric": str(frame["metric"].iloc[0]),
                "common_active_days": int(len(frame)),
                "common_active_start": str(frame.index.min().date()),
                "common_active_end": str(frame.index.max().date()),
                "selected_sharpe": _sharpe(frame["selected_return"]),
                "regsim_sharpe": _sharpe(frame["realized_return_Regsim"]),
                "relative_sharpe": _sharpe(frame["delta_return"]),
                "relative_mean_bp": float(frame["delta_return"].mean() * 10_000.0),
                "selected_sharpe_median": selected_summary["median"],
                "selected_sharpe_mad": selected_summary["mad"],
                "selected_sharpe_stability_q": selected_summary["q"],
                "selected_sharpe_min": selected_summary["min"],
                "selected_sharpe_std": selected_summary["std"],
                "regsim_sharpe_stability_q": regsim_summary["q"],
                "relative_sharpe_stability_q": relative_sharpe_summary["q"],
                "relative_mean_stability_bp": relative_mean_summary["q"],
                "relative_mean_median_bp": relative_mean_summary["median"],
                "relative_mean_mad_bp": relative_mean_summary["mad"],
                "positive_relative_blocks": int(positive_blocks),
                "rolling_sharpe_mean": rolling["mean"],
                "rolling_sharpe_std": rolling["std"],
                "rolling_sharpe_positive_ratio": rolling["positive_ratio"],
                "rolling_sharpe_n": rolling["n"],
            }
        )
    scorecard = pd.DataFrame(score_rows)
    scorecard["absolute_q_advantage_vs_regsim"] = (
        scorecard["selected_sharpe_stability_q"] - scorecard["regsim_sharpe_stability_q"]
    )
    bootstrap = relative_sharpe_bootstrap(daily, common_days, reps=bootstrap_reps)
    scorecard = scorecard.merge(bootstrap, on="variant", how="left", validate="one_to_one")
    scorecard["passes_absolute_stability"] = scorecard["absolute_q_advantage_vs_regsim"] >= MIN_ABSOLUTE_Q_ADVANTAGE
    scorecard["passes_relative_stability"] = (
        scorecard["relative_mean_stability_bp"] > 0.0
    ) & (scorecard["positive_relative_blocks"] >= MIN_POSITIVE_RELATIVE_BLOCKS)
    scorecard["passes_relative_bootstrap"] = (
        scorecard["relative_sharpe_bootstrap_probability_positive"] >= MIN_RELATIVE_BOOTSTRAP_PROBABILITY
    )
    scorecard["passes_design_gates"] = (
        scorecard["passes_absolute_stability"]
        & scorecard["passes_relative_stability"]
        & scorecard["passes_relative_bootstrap"]
    )
    scorecard = scorecard.sort_values(
        [
            "passes_design_gates",
            "selected_sharpe_stability_q",
            "relative_mean_stability_bp",
            "rolling_sharpe_positive_ratio",
            "relative_sharpe_stability_q",
            "lookback_days",
            "nearest_k",
            "metric",
        ],
        ascending=[False, False, False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    scorecard.insert(0, "design_rank_sharpe_stability", np.arange(1, len(scorecard) + 1))
    return scorecard, pd.DataFrame(block_rows), common_days


def _moving_block_indices(n_days: int, *, rng: np.random.Generator) -> np.ndarray:
    block_count = math.ceil(n_days / BLOCK_BOOTSTRAP_LENGTH)
    starts = rng.integers(0, n_days - BLOCK_BOOTSTRAP_LENGTH + 1, size=block_count)
    blocks = [np.arange(start, start + BLOCK_BOOTSTRAP_LENGTH) for start in starts]
    return np.concatenate(blocks)[:n_days]


def relative_sharpe_bootstrap(
    daily: pd.DataFrame,
    common_days: pd.DatetimeIndex,
    *,
    reps: int,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Moving-block probability that each fixed-grid delta Sharpe is positive."""

    if reps <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    variants = daily["variant"].drop_duplicates().tolist()
    deltas = np.column_stack(
        [
            _variant_common_frame(daily, variant, common_days)["delta_return"].to_numpy(dtype=float)
            for variant in variants
        ]
    )
    sampled_sharpes = np.empty((reps, len(variants)), dtype=float)
    rng = np.random.default_rng(seed)
    cursor = 0
    batch_size = min(500, reps)
    while cursor < reps:
        size = min(batch_size, reps - cursor)
        indices = np.vstack([_moving_block_indices(len(common_days), rng=rng) for _ in range(size)])
        sample = deltas[indices, :]
        means = sample.mean(axis=1)
        stds = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sampled_sharpes[cursor : cursor + size] = means / stds * math.sqrt(ANNUALIZATION)
        cursor += size
    rows: list[dict[str, object]] = []
    for index, variant in enumerate(variants):
        values = sampled_sharpes[:, index]
        values = values[np.isfinite(values)]
        rows.append(
            {
                "variant": variant,
                "relative_sharpe_bootstrap_reps": int(reps),
                "relative_sharpe_bootstrap_n": int(len(values)),
                "relative_sharpe_bootstrap_probability_positive": float((values > 0.0).mean()) if len(values) else float("nan"),
                "relative_sharpe_bootstrap_q05": float(np.quantile(values, 0.05)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def reality_check(daily: pd.DataFrame, common_days: pd.DatetimeIndex, *, seed: int = RANDOM_SEED) -> dict[str, object]:
    """Stationary-in-spirit moving-block max-mean diagnostic for the fixed family."""

    variants = daily["variant"].drop_duplicates().tolist()
    delta_matrix = np.column_stack(
        [
            _variant_common_frame(daily, variant, common_days)["delta_return"].to_numpy(dtype=float)
            for variant in variants
        ]
    )
    observed_means_bp = delta_matrix.mean(axis=0) * 10_000.0
    observed_max_bp = float(np.max(observed_means_bp))
    centered = delta_matrix - delta_matrix.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(BLOCK_BOOTSTRAP_REPS):
        indices = _moving_block_indices(len(common_days), rng=rng)
        max_bp = float(np.max(centered[indices].mean(axis=0) * 10_000.0))
        exceed += int(max_bp >= observed_max_bp)
    return {
        "method": "moving_block_max_mean_reality_check_diagnostic",
        "block_length_days": BLOCK_BOOTSTRAP_LENGTH,
        "repetitions": BLOCK_BOOTSTRAP_REPS,
        "seed": seed,
        "common_active_days": int(len(common_days)),
        "observed_max_relative_mean_bp": observed_max_bp,
        "familywise_empirical_p": float((exceed + 1) / (BLOCK_BOOTSTRAP_REPS + 1)),
        "note": "Diagnostic for the preregistered 51-point family; not a promotion test or a replacement for future validation.",
    }


def _evaluation_metrics(daily: pd.DataFrame, variant: str, *, start: str, end: str, scope: str) -> dict[str, object]:
    frame = _scope(daily.loc[daily["variant"] == variant].copy(), start, end)
    frame = frame.loc[frame["execution_metadata_complete"]].copy()
    if frame.empty:
        return {"scope": scope, "n_days": 0}
    delta = frame["selected_return"] - frame["realized_return_Regsim"]
    return {
        "scope": scope,
        "n_days": int(len(frame)),
        "start": str(frame["score_day"].min().date()),
        "end": str(frame["score_day"].max().date()),
        "selected_sharpe": _sharpe(frame["selected_return"]),
        "regsim_sharpe": _sharpe(frame["realized_return_Regsim"]),
        "relative_sharpe": _sharpe(delta),
        "relative_mean_bp": float(delta.mean() * 10_000.0),
        "selected_return": float((1.0 + frame["selected_return"]).prod() - 1.0),
        "regsim_return": float((1.0 + frame["realized_return_Regsim"]).prod() - 1.0),
        "relative_return_pp": float(((1.0 + frame["selected_return"]).prod() - (1.0 + frame["realized_return_Regsim"]).prod()) * 100.0),
    }


def _write_results(
    run_root: Path,
    *,
    scorecard: pd.DataFrame,
    evaluation: pd.DataFrame,
    common_days: pd.DatetimeIndex,
    reality: Mapping[str, object],
) -> tuple[str | None, str]:
    passing = scorecard.loc[scorecard["passes_design_gates"]].copy()
    chosen = None if passing.empty else str(passing.iloc[0]["variant"])
    recommendation = "no_selector_passed_design_gates_keep_fixed_regsim" if chosen is None else f"design_candidate={chosen}_requires_unseen_forward_validation"
    lines = [
        "# BaseGap robust-Sharpe stability scorecard v1",
        "",
        "## Scope",
        "",
        "- Research-only retrospective calibration over the frozen r3 51-point BaseGap replay.",
        "- No live config, scheduler, database, model state, score root, or live result is changed.",
        f"- Common active design cohort: {common_days.min():%Y-%m-%d} through {common_days.max():%Y-%m-%d}, {len(common_days)} days.",
        "- Design ranking uses only that common active cohort; validation and final OOS are reported after ranking and never choose another point.",
        "",
        "## Fixed objective",
        "",
        "- Five chronological design blocks; absolute quality Q = median(block selected Sharpe) - 1.4826 * MAD(block selected Sharpe).",
        "- Relative quality versus fixed Regsim A = median(block daily delta in bp) - 1.4826 * MAD(block daily delta in bp).",
        f"- Design gates: Q advantage versus Regsim >= {MIN_ABSOLUTE_Q_ADVANTAGE:.2f}; A > 0; at least {MIN_POSITIVE_RELATIVE_BLOCKS}/{BLOCK_COUNT} blocks with positive daily delta; 5-day block-bootstrap P(relative Sharpe > 0) >= {MIN_RELATIVE_BOOTSTRAP_PROBABILITY:.2f}.",
        f"- Rolling Sharpe diagnostic: {ROLLING_SHARPE_WINDOW}-day window, at least {ROLLING_SHARPE_MIN_PERIODS} observations, annualization {int(ANNUALIZATION)}.",
        "",
        "## Result",
        "",
        f"- Recommendation: `{recommendation}`.",
        f"- Family-wise moving-block max-mean diagnostic p={float(reality['familywise_empirical_p']):.4f} for the fixed 51-point family.",
        "",
        "## Top design scorecard rows",
        "",
        "| rank | variant | Q selected | Q Regsim | A relative bp/day | positive blocks | bootstrap P(SRdelta>0) | passes |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in scorecard.head(10).iterrows():
        lines.append(
            "| {rank} | {variant} | {q:.3f} | {qr:.3f} | {a:.3f} | {blocks} | {ratio:.2%} | {passed} |".format(
                rank=int(row["design_rank_sharpe_stability"]),
                variant=row["variant"],
                q=float(row["selected_sharpe_stability_q"]),
                qr=float(row["regsim_sharpe_stability_q"]),
                a=float(row["relative_mean_stability_bp"]),
                blocks=int(row["positive_relative_blocks"]),
                ratio=float(row["relative_sharpe_bootstrap_probability_positive"]),
                passed=bool(row["passes_design_gates"]),
            )
        )
    lines.extend(
        [
            "",
            "## Holdout reporting only",
            "",
            "| scope | n | selected Sharpe | Regsim Sharpe | relative Sharpe | relative mean bp/day | relative return pp |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if chosen is not None:
        selected_evaluation = evaluation.loc[evaluation["variant"] == chosen]
        for _, row in selected_evaluation.iterrows():
            lines.append(
                "| {scope} | {n} | {selected:.3f} | {regsim:.3f} | {relative:.3f} | {mean:.3f} | {ret:.3f} |".format(
                    scope=row["scope"],
                    n=int(row["n_days"]),
                    selected=float(row["selected_sharpe"]),
                    regsim=float(row["regsim_sharpe"]),
                    relative=float(row["relative_sharpe"]),
                    mean=float(row["relative_mean_bp"]),
                    ret=float(row["relative_return_pp"]),
                )
            )
    else:
        lines.append("| none | 0 |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This re-ranks an already observed historical family and is not independent proof of a new live rule.",
            "- The source replay retains its path_full_t1430 PIT caveat and frozen-price snapshot limitation.",
            "- Any future promotion requires a strict-14:29 immutable-input forward-shadow test against fixed Regsim.",
            "",
        ]
    )
    (run_root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return chosen, recommendation


def run_scorecard(*, source_run: Path, run_root: Path) -> dict[str, object]:
    source_run = _assert_source_run(source_run)
    run_root = _make_run_root(run_root.parent, run_root.name)
    run_root.mkdir(parents=True)
    _write_json(
        run_root / "run_status.json",
        {
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc),
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "research_output_only": True,
        },
    )
    daily, evidence = load_frozen_daily_replay(source_run)
    scorecard, blocks, common_days = build_sharpe_stability_scorecard(daily)
    reality = reality_check(daily, common_days)
    scorecard.to_csv(run_root / "design_sharpe_stability_scorecard.csv", index=False)
    blocks.to_csv(run_root / "design_sharpe_stability_blocks.csv", index=False)
    _write_json(run_root / "reality_check_diagnostic.json", reality)
    evaluations: list[dict[str, object]] = []
    for variant in scorecard["variant"].tolist():
        for name, start, end in (("validation", *VALIDATION), ("final_oos", *FINAL_OOS)):
            evaluations.append({"variant": variant, **_evaluation_metrics(daily, variant, start=start, end=end, scope=name)})
    evaluation = pd.DataFrame(evaluations)
    evaluation.to_csv(run_root / "holdout_evaluation.csv", index=False)
    chosen, recommendation = _write_results(
        run_root,
        scorecard=scorecard,
        evaluation=evaluation,
        common_days=common_days,
        reality=reality,
    )
    manifest = {
        "run_class": "research_only_basegap_sharpe_stability_scorecard_v1",
        "database_writes": False,
        "live_runtime_called": False,
        "scheduler_called": False,
        "live_config_written": False,
        "live_result_written": False,
        "model_state_written": False,
        "source_replay_evidence": evidence,
        "comparison_contract": {
            "selector": "frozen pure symmetric direct BaseGap argmax",
            "candidate_returns": "frozen aligned standalone full-cycle day_return",
            "baseline": "fixed Regsim on identical common active days",
            "parameter_family": "existing fixed 51-point grid only",
            "design_common_active_cohort": {
                "start": str(common_days.min().date()),
                "end": str(common_days.max().date()),
                "days": int(len(common_days)),
            },
            "stability": {
                "blocks": BLOCK_COUNT,
                "sharpe": "median(block Sharpe) - 1.4826*MAD(block Sharpe)",
                "relative": "median(block mean selected-Regsim bp/day) - 1.4826*MAD",
                "rolling_window": ROLLING_SHARPE_WINDOW,
                "rolling_min_periods": ROLLING_SHARPE_MIN_PERIODS,
                "annualization": ANNUALIZATION,
            },
            "design_gates": {
                "min_absolute_q_advantage_vs_regsim": MIN_ABSOLUTE_Q_ADVANTAGE,
                "positive_relative_blocks": f">={MIN_POSITIVE_RELATIVE_BLOCKS}/{BLOCK_COUNT}",
                "relative_stability_bp": ">0",
                "relative_sharpe_bootstrap_probability": f">={MIN_RELATIVE_BOOTSTRAP_PROBABILITY}",
            },
            "holdout_policy": "validation/final_oos are reporting-only and cannot select a replacement",
        },
        "recommendation": recommendation,
        "design_selected_variant": chosen,
        "source_code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "git": {
            "head": _safe_git(["rev-parse", "HEAD"]),
            "short_head": _safe_git(["rev-parse", "--short", "HEAD"]),
            "status_porcelain": _safe_git(["status", "--porcelain"]),
        },
        "completed_at_utc": datetime.now(timezone.utc),
    }
    _write_json(run_root / "run_manifest.json", manifest)
    (run_root / "task_state.md").write_text(
        "# Task State: BaseGap robust-Sharpe stability scorecard v1\n\n"
        "## Objective\n\n- Rank the existing frozen 51-point BaseGap family by robust cross-block Sharpe and stable relative value versus fixed Regsim.\n\n"
        "## Current Verified Facts\n\n"
        f"- Common active design days: {len(common_days)} from {common_days.min():%Y-%m-%d} to {common_days.max():%Y-%m-%d}.\n"
        f"- Recommendation: `{recommendation}`.\n\n"
        "## Safety\n\n- Research-only; no live config, DB, scheduler, model state, or production output was written.\n",
        encoding="utf-8",
    )
    _write_json(
        run_root / "run_status.json",
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc),
            "database_writes": False,
            "live_runtime_called": False,
            "scheduler_called": False,
            "research_output_only": True,
            "design_selected_variant": chosen,
            "recommendation": recommendation,
        },
    )
    return {
        "run_root": str(run_root),
        "design_selected_variant": chosen,
        "recommendation": recommendation,
        "common_active_design_days": int(len(common_days)),
        "familywise_reality_check_p": float(reality["familywise_empirical_p"]),
        "verified_frozen_files": int(evidence["verified_frozen_files"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN, help="locked BaseGap r3 replay")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="isolated research root")
    parser.add_argument("--run-name", default=None, help="new plain-name leaf below --output-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _assert_output_root(args.output_root)
    name = args.run_name or f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_root = _make_run_root(output_root, name)
    result = run_scorecard(source_run=args.source_run, run_root=run_root)
    print(json.dumps(result, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
