from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date, resolve_output_path
from cbond_on.core.trading_days import list_trading_days_from_raw
from cbond_on.infra.factors.factor_table_resolution import build_factor_reader
from cbond_on.infra.model.wandb_utils import init_wandb_logger
from cbond_on.infra.model.preprocess_config import parse_winsor_bounds
from cbond_on.infra.model.neutralization import build_neutralizer
from cbond_on.infra.model.impl.lgbm.trainer import (
    _read_label_day,
    evaluate_metrics,
)
from cbond_on.infra.model.impl.linear.linear_score import (
    linear_contract_fingerprint,
    run_linear_score,
    write_linear_outputs,
)


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
        return load_config_file("models/linear/linear_factor_default")
    return load_config_file(str(path))


def _is_relative_to(path: Path, root: Path) -> bool:
    """Compatibility helper for the strict research state-root guard."""

    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _resolve_research_elasticnet_incremental(
    *,
    cfg: dict,
    paths_cfg: dict,
    model_name: str,
    factor_cols: list[str],
    regression_kind: str,
    regression_alpha: float,
    elasticnet_l1_ratio: float,
    max_iter: int,
    lookback_days: int,
    refit_freq: int,
    factor_time: str,
    label_time: str,
    panel_name: str | None,
    winsor_lower: float | None,
    winsor_upper: float | None,
    zscore: bool,
    min_count: int,
    neutralizer,
) -> tuple[bool, bool, Path | None, str | None]:
    """Resolve opt-in research-only ElasticNet continuation.

    Default linear behaviour remains cold-refit.  Warm-start is deliberately
    unavailable unless the config explicitly declares both an incremental
    request and ``experiment.research_only=true``.  The state directory must
    be explicit and contained by this process's configured results root.
    """

    incremental_cfg = dict(cfg.get("incremental", {}))
    experiment_cfg = dict(cfg.get("experiment", {}))
    enabled = bool(incremental_cfg.get("enabled", False))
    warm_start = bool(incremental_cfg.get("warm_start", False))
    save_state = bool(incremental_cfg.get("save_state", False))
    research_only = bool(experiment_cfg.get("research_only", False))
    kind = str(regression_kind).strip().lower().replace("_", "")
    requested = enabled and warm_start
    if not requested:
        return False, False, None, None
    if not research_only:
        raise ValueError("linear ElasticNet warm_start is research-only; set experiment.research_only=true")
    if kind not in {"elasticnet", "enet"}:
        raise ValueError("linear warm_start is supported only for regression_kind=elasticnet")
    state_dir_raw = incremental_cfg.get("state_dir")
    if state_dir_raw in (None, ""):
        raise ValueError("research ElasticNet warm_start requires explicit incremental.state_dir")
    results_root = Path(paths_cfg["results_root"])
    state_dir = resolve_output_path(
        state_dir_raw,
        default_path=results_root / "model_state" / model_name,
        results_root=results_root,
    )
    if not _is_relative_to(state_dir, results_root):
        raise ValueError("research ElasticNet state_dir must be inside paths.results_root")
    fingerprint = linear_contract_fingerprint(
        {
            "format_version": 1,
            "model_name": model_name,
            "regression_kind": "elasticnet",
            "factor_cols": list(factor_cols),
            "regression_alpha": float(regression_alpha),
            "elasticnet_l1_ratio": float(elasticnet_l1_ratio),
            "max_iter": int(max_iter),
            "lookback_days": int(lookback_days),
            "refit_freq": int(refit_freq),
            "factor_time": str(factor_time),
            "label_time": str(label_time),
            "panel_name": str(panel_name or ""),
            "winsor": {"lower": winsor_lower, "upper": winsor_upper},
            "zscore": bool(zscore),
            "min_count": int(min_count),
            "neutralization": neutralizer.summary() if neutralizer is not None else {"enabled": False},
        }
    )
    return True, save_state, state_dir, fingerprint


def main(
    *,
    config_path: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    label_cutoff: str | None = None,
    execution: dict | None = None,
) -> None:
    paths_cfg = load_config_file("paths")
    cfg_path = Path(config_path) if config_path else None
    if cfg_path is None and len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.exists():
            cfg_path = candidate
    cfg = _load_model_config(cfg_path)

    cfg_start = parse_date(cfg.get("start"))
    cfg_end = parse_date(cfg.get("end"))
    start = parse_date(start) if start else cfg_start
    end = parse_date(end) if end else cfg_end
    cutoff_value = label_cutoff if label_cutoff is not None else cfg.get("label_cutoff")
    cutoff_day = parse_date(cutoff_value) if cutoff_value else None
    if start > end:
        raise ValueError("start date must be <= end date")

    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])
    raw_root = Path(paths_cfg["raw_data_root"])
    panel_root = Path(paths_cfg["panel_data_root"])

    panel_name = cfg.get("panel_name")
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:42"))

    store = build_factor_reader(paths_cfg, panel_name=panel_name, window_minutes=window_minutes)
    # Pick factor columns from the target factor days.  A live target does not
    # have a realised label yet, so label availability cannot define the input
    # universe here.
    candidate_factor_days = list_trading_days_from_raw(
        raw_root,
        start,
        end,
        kind="snapshot",
        asset="cbond",
    )
    target_days = [day for day in candidate_factor_days if store.has_day(day)]
    sample = pd.DataFrame()
    for day in target_days:
        sample = store.read_day(day)
        if not sample.empty:
            break
    if sample.empty:
        raise RuntimeError("no factor data found")
    if isinstance(sample.index, pd.MultiIndex):
        sample = sample.reset_index()
    factor_cols = _select_factor_cols(sample, cfg)

    winsor_lower, winsor_upper = parse_winsor_bounds(cfg.get("winsor", {}))
    zscore = bool(cfg.get("zscore", True))
    min_count = int(cfg.get("min_count", 30))
    bins = int(cfg.get("bins", 5))
    # Keep legacy cache placement unless a research config explicitly gives a
    # derived-cache root.  The latter lets the live50 panel input remain
    # read-only throughout this experiment.
    neutralization_cache_raw = cfg.get("neutralization_cache_root")
    neutralization_cache_root: Path | None = None
    if neutralization_cache_raw not in (None, ""):
        neutralization_cache_root = resolve_output_path(
            neutralization_cache_raw,
            default_path=Path(paths_cfg["results_root"]) / "neutralization_cache" / str(cfg.get("model_name", "linear_factor")),
            results_root=paths_cfg["results_root"],
        )
    neutralizer = build_neutralizer(
        cfg.get("neutralization"),
        raw_data_root=raw_root,
        panel_data_root=panel_root,
        neutralization_cache_root=neutralization_cache_root,
    )

    linear_cfg = cfg.get("linear", {})
    lookback_days = int(linear_cfg.get("lookback_days", 60))
    refit_freq = int(linear_cfg.get("refit_freq", 1))
    execution_cfg = dict(execution or {})
    exec_refit = execution_cfg.get("refit_every_n_days")
    if exec_refit is not None:
        refit_freq = max(1, int(exec_refit))
    regression_alpha = float(linear_cfg.get("regression_alpha", 1.0))
    regression_kind = str(linear_cfg.get("regression_kind", linear_cfg.get("regression_model", "ridge")))
    elasticnet_l1_ratio = float(linear_cfg.get("elasticnet_l1_ratio", linear_cfg.get("l1_ratio", 0.5)))
    huber_epsilon = float(linear_cfg.get("huber_epsilon", linear_cfg.get("epsilon", 1.35)))
    max_iter = int(linear_cfg.get("max_iter", 1_000))
    weight_source = str(linear_cfg.get("weight_source", "regression"))
    fallback = str(linear_cfg.get("fallback", "manual"))
    max_weight = float(linear_cfg.get("max_weight", 3.0))
    normalize_weights = str(linear_cfg.get("normalize", "l1"))
    device = str(linear_cfg.get("device", "cpu"))
    gpu_fallback_to_cpu = bool(linear_cfg.get("gpu_fallback_to_cpu", True))

    manual_weights = []
    for f in factor_cols:
        manual_weights.append(float(linear_cfg.get("manual_weights", {}).get(f, 0.0)))
    manual_weights = pd.Series(manual_weights, index=factor_cols, dtype=float)
    model_name = str(cfg.get("model_name", "linear_factor"))
    (
        incremental_warm_start,
        incremental_save_state,
        incremental_state_dir,
        warm_start_fingerprint,
    ) = _resolve_research_elasticnet_incremental(
        cfg=cfg,
        paths_cfg=paths_cfg,
        model_name=model_name,
        factor_cols=factor_cols,
        regression_kind=regression_kind,
        regression_alpha=regression_alpha,
        elasticnet_l1_ratio=elasticnet_l1_ratio,
        max_iter=max_iter,
        lookback_days=lookback_days,
        refit_freq=refit_freq,
        factor_time=factor_time,
        label_time=label_time,
        panel_name=panel_name,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        min_count=min_count,
        neutralizer=neutralizer,
    )
    wandb_logger = init_wandb_logger(
        execution_cfg=execution_cfg,
        model_cfg=cfg,
        model_name=model_name,
        model_type="linear",
        start=start,
        end=end,
        extra_config={
            "lookback_days": int(lookback_days),
            "refit_freq": int(refit_freq),
            "weight_source": str(weight_source),
            "device": str(device),
        },
    )
    wandb_logger.log(
        {
            "factor_count": int(len(factor_cols)),
            "winsor_lower": winsor_lower,
            "winsor_upper": winsor_upper,
            "zscore": bool(zscore),
            "min_count": int(min_count),
            "bins": int(bins),
            "lookback_days": int(lookback_days),
            "refit_freq": int(refit_freq),
            "regression_alpha": float(regression_alpha),
            "regression_kind": str(regression_kind),
            "elasticnet_l1_ratio": float(elasticnet_l1_ratio),
            "huber_epsilon": float(huber_epsilon),
            "max_iter": int(max_iter),
            "weight_source": str(weight_source),
            "fallback": str(fallback),
            "max_weight": float(max_weight),
            "normalize_weights": str(normalize_weights),
            "device": str(device),
            "neutralization_enabled": bool(neutralizer is not None and neutralizer.enabled),
            "neutralization_cache_root": str(neutralization_cache_root) if neutralization_cache_root else "legacy_panel_default",
            "research_elasticnet_warm_start": bool(incremental_warm_start),
            "research_elasticnet_save_state": bool(incremental_save_state),
            "research_elasticnet_state_dir": str(incremental_state_dir) if incremental_state_dir else None,
        },
        prefix="run",
    )
    if neutralizer is not None and neutralizer.enabled:
        wandb_logger.log(neutralizer.summary(), prefix="neutralization")

    result = run_linear_score(
        factor_root=factor_root,
        factor_store=store,
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
        regression_kind=regression_kind,
        elasticnet_l1_ratio=elasticnet_l1_ratio,
        huber_epsilon=huber_epsilon,
        max_iter=max_iter,
        weight_source=weight_source,
        fallback=fallback,
        max_weight=max_weight,
        normalize_weights=normalize_weights,
        manual_weights=manual_weights,
        device=device,
        gpu_fallback_to_cpu=gpu_fallback_to_cpu,
        neutralizer=neutralizer,
        label_cutoff=cutoff_day,
        incremental_enabled=bool(incremental_warm_start),
        incremental_warm_start=bool(incremental_warm_start),
        incremental_save_state=bool(incremental_save_state),
        state_dir=incremental_state_dir,
        warm_start_fingerprint=warm_start_fingerprint,
        target_days=target_days,
        audit_only=bool(
            isinstance(paths_cfg.get("lifecycle"), dict)
            and paths_cfg["lifecycle"].get("status") == "audit_only"
            and paths_cfg["lifecycle"].get("reason") == "no_db_ephemeral_factor_stage"
        ),
    )

    if result.scores.empty:
        wandb_logger.finish({"status": "no_scores_generated"})
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

    # A live call supplies the latest realised-label day as label_cutoff.  Do
    # not reopen later target labels merely to print retrospective metrics: it
    # obscures the same point-in-time boundary enforced by the scorer itself.
    eval_scores_df = scores_df
    if cutoff_day is not None:
        eval_scores_df = scores_df[scores_df["trade_date"] <= cutoff_day].copy()
    full_eval = _merge_labels(eval_scores_df)

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
    artifact_root = resolve_output_path(
        cfg.get("results_root"),
        default_path=results_root,
        results_root=results_root,
    )
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = artifact_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    write_linear_outputs(
        result=result,
        score_path=out_dir / "scores.csv",
        weights_path=out_dir / "weights.csv",
        meta_path=out_dir / "meta.json",
        meta_payload={"config": cfg, "label_cutoff": str(cutoff_day) if cutoff_day else None},
        overwrite=True,
    )

    # also write to configured score outputs (used by backtest/live)
    score_output = cfg.get("score_output")
    if score_output:
        score_path = resolve_output_path(
            score_output,
            default_path=results_root / "scores" / model_name / "scores.csv",
            results_root=results_root,
        )
        weights_path = (
            resolve_output_path(
                cfg.get("weights_output"),
                default_path=results_root / "scores" / model_name / "weights.csv",
                results_root=results_root,
            )
            if cfg.get("weights_output")
            else None
        )
        meta_path = (
            resolve_output_path(
                cfg.get("meta_output"),
                default_path=results_root / "scores" / model_name / "meta.json",
                results_root=results_root,
            )
            if cfg.get("meta_output")
            else None
        )
        score_overwrite = bool(cfg.get("score_overwrite", True))
        score_dedupe = bool(cfg.get("score_dedupe", True))
        write_linear_outputs(
            result=result,
            score_path=score_path,
            weights_path=weights_path,
            meta_path=meta_path,
            meta_payload={"config": cfg, "label_cutoff": str(cutoff_day) if cutoff_day else None},
            overwrite=score_overwrite,
            dedupe=score_dedupe,
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

    wandb_logger.log(
        {
            "rows_scored": int(len(result.scores)),
            "weight_rows": int(len(result.weights_history)),
            "eval_rows": int(len(full_eval)),
            "mse": float(full_metrics.get("mse", float("nan"))),
            "r2": float(full_metrics.get("r2", float("nan"))),
            "dir": float(full_metrics.get("dir", float("nan"))),
            "ic_mean": float(full_metrics.get("ic_mean", float("nan"))),
            "ic_ir": float(full_metrics.get("ic_ir", float("nan"))),
            "rank_ic_mean": float(full_metrics.get("rank_ic_mean", float("nan"))),
            "rank_ic_ir": float(full_metrics.get("rank_ic_ir", float("nan"))),
        },
        prefix="final",
    )
    wandb_logger.finish({"status": "ok", "out_dir": str(out_dir)})
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()


