"""Leakage-controlled OOF score-stacking research for CBOND_ON.

This is deliberately a *research-only* harness tool.  It never imports a live
workflow, model state, database adapter, scheduler, or strategy output.  It
reads already-produced score CSVs and same-score-day 14:42 labels, then learns
only from strictly earlier labelled days for every stacked prediction.

The tool is intended to answer a narrow first question before new model or
factor work: can the existing, genuinely different score streams add
cross-sectional predictive information to Regsim under a chronological OOF
stack?  Equal-weight rank averaging is not used as an optimisation proxy.

All writes are refused outside a caller-supplied experiment root.  The output
root must not already exist, preventing accidental overwrite of past research.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


DEFAULT_SCORE_ROOT = Path(r"D:\cbond_on\results\scores\live")
DEFAULT_LABEL_ROOT = Path(r"D:\cbond_on\label_data")
DEFAULT_OUTPUT_ROOT = Path(r"D:\cbond_on\results\experiments\ic_uplift_oos_20260731")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "regsim",
        "lgbm_screened_no_winsor_neutral_tminus1_regsim_w10_s15_d05_20260708",
    ),
    ModelSpec(
        "baseline",
        "lgbm_screened_no_winsor_neutral_tminus1_refit1_202401_rerun_20260623",
    ),
    ModelSpec(
        "hl20",
        "lgbm_screened_no_winsor_neutral_tminus1_weight_recent_hl20_20260625",
    ),
    ModelSpec(
        "ensemble_rankavg",
        "ensemble_rankavg_baseline_hl20_labeltop20_20260626",
    ),
    ModelSpec(
        "labeltop20",
        "lgbm_screened_no_winsor_neutral_tminus1_weight_labeltop20_20260625",
    ),
)

RAW_FEATURES = tuple(spec.key for spec in MODEL_SPECS)
RANK_FEATURES = tuple(f"{name}__rank_z" for name in RAW_FEATURES)
TARGET_COLUMN = "y__z"

# This is a fixed, deliberately small candidate plan.  It is not a same-sample
# hyperparameter sweep.  Candidate choice belongs to a validation segment; the
# final reporting segment must not drive any new candidate design.
STACK_PLANS: tuple[tuple[str, tuple[str, ...], int, float, bool], ...] = (
    ("stack_raw_ridge_w60_a1", RAW_FEATURES, 60, 1.0, False),
    ("stack_raw_ridge_w120_a1", RAW_FEATURES, 120, 1.0, False),
    ("stack_raw_ridge_w250_a1", RAW_FEATURES, 250, 1.0, False),
    ("stack_rank_ridge_w120_a1", RANK_FEATURES, 120, 1.0, False),
    ("stack_raw_positive_ridge_w120_a1", RAW_FEATURES, 120, 1.0, True),
)


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _daily_zscore(values: pd.Series) -> pd.Series:
    """Cross-sectionally standardise a field without using label information."""

    numeric = pd.to_numeric(values, errors="coerce")
    mean = numeric.mean()
    std = numeric.std(ddof=0)
    if not math.isfinite(float(std)) or float(std) <= 0.0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - mean) / std


def _daily_rank_zscore(values: pd.Series) -> pd.Series:
    """Turn same-day ranks into a unit-variance feature with average-tie ranks."""

    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(method="average", pct=True)
    return _daily_zscore(ranks)


def _corr(frame: pd.DataFrame, prediction: str, method: str) -> float | None:
    subset = frame[[prediction, "y"]].dropna()
    if len(subset) < 3 or subset[prediction].nunique() < 2 or subset["y"].nunique() < 2:
        return None
    return _finite(subset[prediction].corr(subset["y"], method=method))


def _date_partition(day: str, validation_start: str, final_start: str) -> str:
    if day < validation_start:
        return "development"
    if day < final_start:
        return "validation"
    return "final_reporting"


def _score_file_map(root: Path, model_id: str) -> dict[str, Path]:
    model_root = root / model_id
    if not model_root.is_dir():
        raise FileNotFoundError(f"score root missing: {model_root}")
    out: dict[str, Path] = {}
    for path in sorted(model_root.glob("*/*.csv")):
        day = path.stem
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"unexpected score filename: {path}") from exc
        if day in out:
            raise ValueError(f"duplicate score day for {model_id}: {day}")
        out[day] = path
    if not out:
        raise ValueError(f"no score CSVs under {model_root}")
    return out


def _load_label(label_root: Path, day: str) -> pd.DataFrame | None:
    compact = day.replace("-", "")
    path = label_root / day[:7] / f"{compact}.parquet"
    if not path.is_file():
        return None
    label = pd.read_parquet(path, columns=["code", "trade_time", "y"])
    trade_time = pd.to_datetime(label["trade_time"], errors="coerce")
    label = label.loc[
        (trade_time.dt.hour == 14) & (trade_time.dt.minute == 42), ["code", "y"]
    ].copy()
    label["code"] = label["code"].astype(str)
    label["y"] = pd.to_numeric(label["y"], errors="coerce")
    label = label.dropna().drop_duplicates("code", keep=False)
    if label.empty:
        return None
    return label


def _load_score(path: Path, key: str, expected_day: str) -> pd.DataFrame:
    score = pd.read_csv(path, usecols=["trade_date", "code", "score"])
    score["trade_date"] = score["trade_date"].astype(str)
    actual_days = set(score["trade_date"].dropna().unique())
    if actual_days != {expected_day}:
        raise ValueError(
            f"score file date mismatch for {key}: file={expected_day}, values={sorted(actual_days)}"
        )
    score["code"] = score["code"].astype(str)
    score[key] = pd.to_numeric(score["score"], errors="coerce")
    score = score[["code", key]].dropna().drop_duplicates("code", keep=False)
    if score.empty:
        raise ValueError(f"empty usable score file: {path}")
    return score


def load_cross_sections(
    score_root: Path,
    label_root: Path,
    start: str,
    end: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Load only date-aligned score/label cross-sections.

    The label day is deliberately the same named score day.  The economic label
    construction remains owned by the existing label pipeline; this tool only
    consumes its frozen ``y`` output and verifies the 14:42 marker.
    """

    paths = {spec.key: _score_file_map(score_root, spec.model_id) for spec in MODEL_SPECS}
    candidate_days = set.intersection(*(set(mapping) for mapping in paths.values()))
    selected_days = [day for day in sorted(candidate_days) if start <= day <= end]
    sections: dict[str, pd.DataFrame] = {}
    skipped_no_label: list[str] = []
    skipped_invalid: dict[str, str] = {}
    for day in selected_days:
        label = _load_label(label_root, day)
        if label is None:
            skipped_no_label.append(day)
            continue
        merged = label
        try:
            for spec in MODEL_SPECS:
                merged = merged.merge(
                    _load_score(paths[spec.key][day], spec.key, day),
                    on="code",
                    how="inner",
                    validate="one_to_one",
                )
        except (ValueError, KeyError) as exc:
            skipped_invalid[day] = str(exc)
            continue
        if len(merged) < 30:
            skipped_invalid[day] = f"only {len(merged)} matched observations"
            continue
        # Feature transforms use score data only; y is standardised separately
        # solely for chronological stack fitting.
        for name in RAW_FEATURES:
            merged[f"{name}__z"] = _daily_zscore(merged[name])
            merged[f"{name}__rank_z"] = _daily_rank_zscore(merged[name])
        merged[TARGET_COLUMN] = _daily_zscore(merged["y"])
        needed = [*RAW_FEATURES, *[f"{name}__z" for name in RAW_FEATURES], *RANK_FEATURES, TARGET_COLUMN]
        merged = merged.dropna(subset=needed).copy()
        if len(merged) < 30:
            skipped_invalid[day] = "insufficient rows after cross-sectional transforms"
            continue
        merged.insert(0, "score_day", day)
        sections[day] = merged
    metadata: dict[str, object] = {
        "score_days_shared": len(selected_days),
        "sections_loaded": len(sections),
        "skipped_no_label": skipped_no_label,
        "skipped_invalid": skipped_invalid,
        "model_ids": {spec.key: spec.model_id for spec in MODEL_SPECS},
    }
    return sections, metadata


def _fit_predict_stack(
    sections: dict[str, pd.DataFrame],
    features: Sequence[str],
    lookback_days: int,
    alpha: float,
    positive: bool,
    min_history_days: int = 60,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Make walk-forward predictions; each score-day sees strictly prior labels."""

    predictions: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, object]] = []
    days = list(sections)
    for idx, day in enumerate(days):
        start_idx = max(0, idx - lookback_days)
        train_days = days[start_idx:idx]
        if len(train_days) < min_history_days:
            continue
        train = pd.concat([sections[item] for item in train_days], ignore_index=True)
        # Every day receives equal aggregate mass despite a changing universe.
        day_sizes = train.groupby("score_day")["code"].transform("size")
        weights = 1.0 / day_sizes.to_numpy(dtype=float)
        model = Ridge(alpha=alpha, fit_intercept=True, positive=positive)
        model.fit(train.loc[:, features], train[TARGET_COLUMN], sample_weight=weights)
        current = sections[day]
        predictions[day] = pd.Series(
            model.predict(current.loc[:, features]), index=current.index, dtype=float
        )
        audit: dict[str, object] = {
            "score_day": day,
            "train_days": len(train_days),
            "train_start": train_days[0],
            "train_end": train_days[-1],
            "train_end_is_strictly_before_score_day": train_days[-1] < day,
            "rows": len(train),
            "positive": positive,
            "alpha": alpha,
        }
        for name, coef in zip(features, model.coef_, strict=True):
            audit[f"coef__{name}"] = float(coef)
        audit_rows.append(audit)
    return predictions, pd.DataFrame(audit_rows)


def _append_prediction(
    daily: dict[str, pd.DataFrame],
    name: str,
    predictions: dict[str, pd.Series],
) -> None:
    for day, series in predictions.items():
        daily[day][name] = series


def _metric_rows(
    daily: dict[str, pd.DataFrame],
    prediction_names: Iterable[str],
    validation_start: str,
    final_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, object]] = []
    for day, frame in daily.items():
        partition = _date_partition(day, validation_start, final_start)
        for prediction in prediction_names:
            if prediction not in frame:
                continue
            subset = frame[["code", "y", prediction]].dropna()
            if len(subset) < 30:
                continue
            top = subset.nlargest(20, prediction)["y"]
            bottom = subset.nsmallest(20, prediction)["y"]
            daily_rows.append(
                {
                    "score_day": day,
                    "partition": partition,
                    "prediction": prediction,
                    "n": len(subset),
                    "pearson_ic": _corr(subset, prediction, "pearson"),
                    "rank_ic": _corr(subset, prediction, "spearman"),
                    "top20_mean_y": _finite(top.mean()),
                    "bottom20_mean_y": _finite(bottom.mean()),
                    "top20_minus_bottom20": _finite(top.mean() - bottom.mean()),
                }
            )
    daily_metrics = pd.DataFrame(daily_rows)
    if daily_metrics.empty:
        raise RuntimeError("no daily metrics were produced")
    summary_rows: list[dict[str, object]] = []
    for (prediction, partition), group in daily_metrics.groupby(["prediction", "partition"], sort=True):
        row: dict[str, object] = {
            "prediction": prediction,
            "partition": partition,
            "valid_days": int(len(group)),
            "first_score_day": str(group["score_day"].min()),
            "last_score_day": str(group["score_day"].max()),
            "avg_cross_section_n": float(group["n"].mean()),
        }
        for field in ("pearson_ic", "rank_ic", "top20_mean_y", "top20_minus_bottom20"):
            values = group[field].dropna()
            row[f"mean_{field}"] = _finite(values.mean()) if len(values) else None
            row[f"positive_ratio_{field}"] = _finite((values > 0).mean()) if len(values) else None
            row[f"t_{field}"] = (
                _finite(values.mean() / values.std(ddof=1) * math.sqrt(len(values)))
                if len(values) > 1 and values.std(ddof=1) > 0
                else None
            )
        summary_rows.append(row)
    return daily_metrics, pd.DataFrame(summary_rows).sort_values(["partition", "prediction"])


def _paired_comparisons(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare every stacked candidate with Regsim on exactly common days."""

    baseline = daily_metrics.loc[daily_metrics["prediction"] == "regsim"].set_index(
        ["score_day", "partition"]
    )
    rows: list[dict[str, object]] = []
    for prediction in sorted(set(daily_metrics["prediction"]) - {"regsim"}):
        candidate = daily_metrics.loc[daily_metrics["prediction"] == prediction].set_index(
            ["score_day", "partition"]
        )
        shared = candidate.join(
            baseline[["pearson_ic", "rank_ic", "top20_mean_y"]],
            how="inner",
            lsuffix="__candidate",
            rsuffix="__regsim",
        )
        for partition, group in shared.groupby(level="partition", sort=True):
            for metric in ("pearson_ic", "rank_ic", "top20_mean_y"):
                delta = group[f"{metric}__candidate"] - group[f"{metric}__regsim"]
                rows.append(
                    {
                        "prediction": prediction,
                        "partition": partition,
                        "metric": metric,
                        "shared_days": int(len(delta)),
                        "mean_delta": _finite(delta.mean()),
                        "t_delta": (
                            _finite(delta.mean() / delta.std(ddof=1) * math.sqrt(len(delta)))
                            if len(delta) > 1 and delta.std(ddof=1) > 0
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["partition", "prediction", "metric"])


def run_experiment(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing experiment root: {output_root}; choose a new root"
        )
    score_root = Path(args.score_root)
    label_root = Path(args.label_root)
    sections, input_meta = load_cross_sections(score_root, label_root, args.start, args.end)
    if not sections:
        raise RuntimeError("no valid aligned score/label cross-sections in requested window")
    ordered_days = list(sections)
    if args.validation_start not in ordered_days and args.validation_start < ordered_days[0]:
        raise ValueError("validation_start precedes available data")
    if not (args.start < args.validation_start < args.final_start <= args.end):
        raise ValueError("require start < validation_start < final_start <= end")

    daily = {day: frame.copy() for day, frame in sections.items()}
    # Native score streams are fixed controls.  This component rank average is
    # diagnostic only, separating the three-source ensemble construction from
    # its stored implementation.
    for day, frame in daily.items():
        frame["component_rankavg"] = frame.loc[:, ["baseline__rank_z", "hl20__rank_z", "labeltop20__rank_z"]].mean(axis=1)

    audit_frames: list[pd.DataFrame] = []
    prediction_names = [*RAW_FEATURES, "component_rankavg"]
    for name, features, lookback, alpha, positive in STACK_PLANS:
        # Raw plan uses score z-scores, not raw score magnitudes.  This avoids
        # accidental dominance from unrelated model score scales.
        resolved_features = tuple(f"{item}__z" for item in features) if features == RAW_FEATURES else features
        prediction, audit = _fit_predict_stack(
            daily, resolved_features, lookback_days=lookback, alpha=alpha, positive=positive
        )
        _append_prediction(daily, name, prediction)
        prediction_names.append(name)
        if not audit.empty:
            audit.insert(1, "stack", name)
            audit_frames.append(audit)

    daily_metrics, summary = _metric_rows(
        daily, prediction_names, args.validation_start, args.final_start
    )
    paired = _paired_comparisons(daily_metrics)

    output_root.mkdir(parents=True, exist_ok=False)
    daily_metrics.to_csv(output_root / "daily_metrics.csv", index=False, encoding="utf-8")
    summary.to_csv(output_root / "summary_metrics.csv", index=False, encoding="utf-8")
    paired.to_csv(output_root / "paired_vs_regsim.csv", index=False, encoding="utf-8")
    coefficient_audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    coefficient_audit.to_csv(output_root / "walk_forward_coefficients.csv", index=False, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "research-only chronological OOF stacking of existing score streams",
        "write_boundary": str(output_root),
        "inputs": {
            "score_root": str(score_root),
            "label_root": str(label_root),
            "label_contract": "same score-date label parquet, trade_time exactly 14:42",
            "score_contract": "CSV trade_date must equal filename date; merged one-to-one by code",
            **input_meta,
        },
        "time_splits": {
            "start": args.start,
            "validation_start": args.validation_start,
            "final_reporting_start": args.final_start,
            "end": args.end,
            "warning": "final_reporting is chronological but not a virgin holdout if prior research has inspected it",
        },
        "leakage_controls": [
            "stack score day T is fitted only from labelled score days strictly before T",
            "features are same-day cross-sectional transforms of scores only; labels never enter feature transforms",
            "each training day is equal-weighted regardless of its matched cross-sectional row count",
            "no database, scheduler, live strategy, o_0005 pool, model state, or live output is read or written",
            "all candidate variants are fixed in source before execution",
        ],
        "stack_plans": [
            {
                "name": name,
                "features": list(features),
                "lookback_days": lookback,
                "alpha": alpha,
                "positive": positive,
            }
            for name, features, lookback, alpha, positive in STACK_PLANS
        ],
        "outputs": [
            "daily_metrics.csv",
            "summary_metrics.csv",
            "paired_vs_regsim.csv",
            "walk_forward_coefficients.csv",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", default=str(DEFAULT_SCORE_ROOT))
    parser.add_argument("--label-root", default=str(DEFAULT_LABEL_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start", default="2024-05-08")
    parser.add_argument("--end", default="2026-07-30")
    parser.add_argument("--validation-start", default="2026-01-05")
    parser.add_argument("--final-start", default="2026-04-30")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = run_experiment(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
