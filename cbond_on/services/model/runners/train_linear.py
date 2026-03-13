from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.factors.storage import FactorStore
from cbond_on.models.impl.lgbm.trainer import (
    _iter_existing_label_days,
    _read_label_day,
    evaluate_metrics,
)
from cbond_on.models.impl.linear.linear_score import run_linear_score, write_linear_outputs


def _select_factor_cols(sample: pd.DataFrame, cfg: dict) -> list[str]:
    cols = cfg.get("factors")
    if cols:
        return [str(c) for c in cols]
    exclude = {"dt", "code"}
    return [c for c in sample.columns if c not in exclude]


def _format_bins(bin_dir: list[tuple[int, float, int]]) -> str:
    if not bin_dir:
        return "n/a"
    return ",".join([f"{b}:{acc:.3f}({n})" for b, acc, n in bin_dir])


def _load_model_config(path: Path | None) -> dict:
    if path is None:
        return load_config_file("models/linear/model")
    import json5
    with path.open("r", encoding="utf-8") as handle:
        return json5.load(handle) or {}


def main(*, start: str | None = None, end: str | None = None) -> None:
    paths_cfg = load_config_file("paths")
    cfg_path = None
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    cfg = _load_model_config(cfg_path)

    cfg_start = parse_date(cfg.get("start"))
    cfg_end = parse_date(cfg.get("end"))
    start = parse_date(start) if start else cfg_start
    end = parse_date(end) if end else cfg_end
    if start > end:
        raise ValueError("start date must be <= end date")

    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])

    panel_name = cfg.get("panel_name")
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:45"))

    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)
    # pick factor columns from first available label day with factors
    sample = pd.DataFrame()
    for day in _iter_existing_label_days(label_root, start, end):
        sample = store.read_day(day)
        if not sample.empty:
            break
    if sample.empty:
        raise RuntimeError("no factor data found")
    if isinstance(sample.index, pd.MultiIndex):
        sample = sample.reset_index()
    factor_cols = _select_factor_cols(sample, cfg)

    winsor = cfg.get("winsor", {})
    winsor_lower = float(winsor.get("lower", 0.01))
    winsor_upper = float(winsor.get("upper", 0.99))
    zscore = bool(cfg.get("zscore", True))
    min_count = int(cfg.get("min_count", 30))
    bins = int(cfg.get("bins", 5))

    linear_cfg = cfg.get("linear", {})
    lookback_days = int(linear_cfg.get("lookback_days", 60))
    refit_freq = int(linear_cfg.get("refit_freq", 1))
    regression_alpha = float(linear_cfg.get("regression_alpha", 1.0))
    weight_source = str(linear_cfg.get("weight_source", "regression"))
    fallback = str(linear_cfg.get("fallback", "manual"))
    max_weight = float(linear_cfg.get("max_weight", 3.0))
    normalize_weights = str(linear_cfg.get("normalize", "l1"))

    manual_weights = []
    for f in factor_cols:
        manual_weights.append(float(linear_cfg.get("manual_weights", {}).get(f, 0.0)))
    manual_weights = pd.Series(manual_weights, index=factor_cols, dtype=float)

    result = run_linear_score(
        factor_root=factor_root,
        label_root=label_root,
        start=start,
        end=end,
        factor_cols=factor_cols,
        panel_name=panel_name,
        window_minutes=window_minutes,
        factor_time=factor_time,
        label_time=label_time,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        lookback_days=lookback_days,
        refit_freq=refit_freq,
        regression_alpha=regression_alpha,
        weight_source=weight_source,
        fallback=fallback,
        max_weight=max_weight,
        normalize_weights=normalize_weights,
        manual_weights=manual_weights,
    )

    if result.scores.empty:
        raise RuntimeError("no scores generated")

    # evaluate metrics on full sample (cbond_day style)
    scores_df = result.scores.copy()
    scores_df["trade_date"] = pd.to_datetime(scores_df["trade_date"]).dt.date

    def _merge_labels(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for day in sorted(df["trade_date"].unique()):
            label_df = _read_label_day(label_root, day, factor_time=factor_time, label_time=label_time)
            if label_df.empty:
                continue
            if "dt" not in label_df.columns:
                continue
            label_df = label_df[["dt", "code", "y"]].dropna()
            if label_df.empty:
                continue
            day_scores = df[df["trade_date"] == day].copy()
            day_scores["dt"] = pd.to_datetime(day_scores["trade_date"]) + pd.to_timedelta(factor_time + ":00")
            merged = day_scores.merge(label_df, on=["dt", "code"], how="inner")
            if not merged.empty:
                rows.append(merged)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    full_eval = _merge_labels(scores_df)

    def _eval(df: pd.DataFrame) -> dict:
        if df.empty:
            return {
                "mse": float("nan"),
                "r2": float("nan"),
                "dir": float("nan"),
                "ic_mean": float("nan"),
                "ic_ir": float("nan"),
                "rank_ic_mean": float("nan"),
                "rank_ic_ir": float("nan"),
                "bin_dir": [],
            }
        return evaluate_metrics(
            x=df[["score"]],
            y=df["y"],
            dt=df["dt"],
            pred=df["score"].to_numpy(),
            bins=bins,
        )

    full_metrics = _eval(full_eval)

    print(
        f"all mse={full_metrics['mse']:.6f} r2={full_metrics['r2']:.4f} dir={full_metrics['dir']:.4f} "
        f"ic={full_metrics['ic_mean']:.4f} ir={full_metrics['ic_ir']:.4f} "
        f"rank_ic={full_metrics['rank_ic_mean']:.4f} rank_ir={full_metrics['rank_ic_ir']:.4f}"
    )
    print(f"all bins: {_format_bins(full_metrics['bin_dir'])}")

    results_root = Path(paths_cfg["results_root"])
    model_name = cfg.get("model_name", "linear_factor")
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    write_linear_outputs(
        result=result,
        score_path=out_dir / "scores.csv",
        weights_path=out_dir / "weights.csv",
        meta_path=out_dir / "meta.json",
        meta_payload={"config": cfg},
        overwrite=True,
    )

    # also write to configured score outputs (used by backtest/live)
    score_output = cfg.get("score_output")
    if score_output:
        write_linear_outputs(
            result=result,
            score_path=Path(score_output),
            weights_path=Path(cfg.get("weights_output")) if cfg.get("weights_output") else None,
            meta_path=Path(cfg.get("meta_output")) if cfg.get("meta_output") else None,
            meta_payload={"config": cfg},
            overwrite=True,
        )

    metrics_df = pd.DataFrame(
        [{"split": "all", **{k: v for k, v in full_metrics.items() if k != "bin_dir"}}]
    )
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    def _bin_df(split, metrics):
        return pd.DataFrame([
            {"split": split, "bin": b, "dir_acc": acc, "count": n} for b, acc, n in metrics["bin_dir"]
        ])
    bin_df = _bin_df("all", full_metrics)
    bin_df.to_csv(out_dir / "bin_dir.csv", index=False)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
