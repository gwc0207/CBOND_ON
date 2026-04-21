from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from cbond_on.common.config_utils import load_json_like, resolve_config_path
from cbond_on.config.loader import load_config_file, parse_date
from cbond_on.app.usecases.model_score_runtime import run as run_model_score
from cbond_on.infra.factors.quality import expected_factor_columns_from_cfg
from cbond_on.infra.model.eval.evaluator import (
    EvaluationResult,
    evaluate_merged_scores,
    load_scores_frame,
    merge_score_with_label,
)


@dataclass
class _EvalTrial:
    trial_name: str
    trial_dir: Path
    score_output: Path
    state_dir: Path
    factors: list[str]
    summary: dict[str, Any]
    daily: pd.DataFrame


def _safe_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "empty"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _load_factor_list(path_like: str) -> list[str]:
    path = resolve_config_path(path_like)
    payload = load_json_like(path)
    raw: Any
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("factors", [])
    else:
        raw = []
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            name = str(item).strip()
            if name:
                out.append(name)
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        v = str(value).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _build_trial_model_cfg(
    *,
    base_model_cfg: dict[str, Any],
    start_day: date,
    end_day: date,
    factors: list[str],
    model_name: str,
    score_output: Path,
    save_state: bool,
    state_dir: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_model_cfg)
    cfg["start"] = str(start_day)
    cfg["end"] = str(end_day)
    cfg["model_name"] = model_name
    cfg["factors"] = list(factors)
    cfg["score_output"] = str(score_output.as_posix())
    cfg["score_overwrite"] = True
    cfg["score_dedupe"] = True
    inc = dict(cfg.get("incremental", {}))
    inc["enabled"] = True
    inc["skip_existing_scores"] = False
    inc["warm_start"] = False
    inc["save_state"] = bool(save_state)
    inc["state_dir"] = str(state_dir.as_posix())
    cfg["incremental"] = inc
    return cfg


def _evaluate_score_output(
    *,
    score_output: Path,
    label_root: Path,
    factor_time: str,
    label_time: str,
    start_day: date,
    end_day: date,
    bins: int,
) -> EvaluationResult:
    scores = load_scores_frame(score_output)
    merged = merge_score_with_label(
        scores=scores,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start=start_day,
        end=end_day,
    )
    return evaluate_merged_scores(merged, bins=bins)


def _run_model_trial(
    *,
    trial_name: str,
    trial_dir: Path,
    factors: list[str],
    model_id: str,
    model_type: str,
    base_model_cfg: dict[str, Any],
    score_cfg: dict[str, Any],
    execution_override: dict[str, Any],
    start_day: date,
    end_day: date,
    label_root: Path,
    factor_time: str,
    label_time: str,
    bins: int,
    save_state: bool,
) -> _EvalTrial:
    trial_dir.mkdir(parents=True, exist_ok=True)
    score_output = trial_dir / "scores"
    state_dir = trial_dir / "state"
    model_name = f"{base_model_cfg.get('model_name', model_id)}__factor_select__{_safe_name(trial_name)}"
    trial_cfg = _build_trial_model_cfg(
        base_model_cfg=base_model_cfg,
        start_day=start_day,
        end_day=end_day,
        factors=factors,
        model_name=model_name,
        score_output=score_output,
        save_state=save_state,
        state_dir=state_dir,
    )
    trial_cfg_path = trial_dir / "model_config.json"
    trial_cfg_path.write_text(json.dumps(trial_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    trial_score_cfg = copy.deepcopy(score_cfg)
    trial_score_cfg["model_id"] = "factor_select_trial"
    trial_score_cfg["default_model_id"] = "factor_select_trial"
    trial_score_cfg["models"] = {
        "factor_select_trial": {
            "model_type": model_type,
            "model_config": str(trial_cfg_path.as_posix()),
        }
    }
    trial_score_cfg["execution"] = execution_override

    print(f"[factor_select] trial={trial_name} start factors={len(factors)}")
    run_model_score(
        model_id="factor_select_trial",
        start=start_day,
        end=end_day,
        cfg=trial_score_cfg,
    )

    eval_res = _evaluate_score_output(
        score_output=score_output,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        start_day=start_day,
        end_day=end_day,
        bins=bins,
    )

    payload = {
        "trial_name": trial_name,
        "factor_count": len(factors),
        "factors": factors,
        "summary": eval_res.summary,
    }
    (trial_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not eval_res.daily.empty:
        eval_res.daily.to_csv(trial_dir / "evaluation_daily.csv", index=False)

    print(
        f"[factor_select] trial={trial_name} done "
        f"rank_ic_mean={eval_res.summary.get('rank_ic_mean')} "
        f"samples={eval_res.summary.get('samples')}"
    )

    return _EvalTrial(
        trial_name=trial_name,
        trial_dir=trial_dir,
        score_output=score_output,
        state_dir=state_dir,
        factors=factors,
        summary=dict(eval_res.summary),
        daily=eval_res.daily,
    )


def _load_importance_from_state_dir(
    *,
    state_dir: Path,
    importance_type: str,
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"lightgbm is required for factor importance export: {exc}") from exc

    if not state_dir.exists():
        return pd.DataFrame(columns=["factor", "importance_mean", "importance_std", "importance_nonzero_ratio", "checkpoints"])

    records: list[dict[str, Any]] = []
    checkpoints = sorted(state_dir.glob("*.txt"))
    for ckpt in checkpoints:
        try:
            booster = lgb.Booster(model_file=str(ckpt))
            names = list(booster.feature_name())
            importances = list(booster.feature_importance(importance_type=importance_type))
            for factor, imp in zip(names, importances):
                records.append(
                    {
                        "checkpoint": ckpt.stem,
                        "factor": str(factor),
                        "importance": float(imp),
                    }
                )
        except Exception as exc:
            print(f"[factor_select] skip checkpoint importance parse failed: {ckpt.name} ({type(exc).__name__}: {exc})")

    if not records:
        return pd.DataFrame(columns=["factor", "importance_mean", "importance_std", "importance_nonzero_ratio", "checkpoints"])

    df = pd.DataFrame(records)
    agg = (
        df.groupby("factor", as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
            importance_nonzero_ratio=("importance", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            checkpoints=("checkpoint", "nunique"),
        )
        .sort_values(["importance_mean", "importance_nonzero_ratio"], ascending=[False, False], kind="mergesort")
        .reset_index(drop=True)
    )
    agg["importance_std"] = pd.to_numeric(agg["importance_std"], errors="coerce").fillna(0.0)
    return agg


def _plot_importance_topk(
    *,
    importance_df: pd.DataFrame,
    out_path: Path,
    top_k: int,
) -> None:
    if importance_df.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    top_k = max(1, int(top_k))
    top = importance_df.head(top_k).copy()
    top = top.iloc[::-1]

    fig_h = max(6, int(len(top) * 0.28) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(top["factor"], top["importance_mean"], color="#4C78A8", alpha=0.9)
    ax.set_xlabel("importance_mean")
    ax.set_title(f"Top {len(top)} Factor Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _numeric_delta_map(full_summary: dict[str, Any], topn_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted(set(full_summary.keys()) | set(topn_summary.keys()))
    for key in keys:
        fv = full_summary.get(key)
        tv = topn_summary.get(key)
        if isinstance(fv, (int, float)) and isinstance(tv, (int, float)):
            out[key] = float(tv) - float(fv)
    return out


def run(
    *,
    cfg: dict | None = None,
    config_name: str | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> dict[str, Any]:
    selector_cfg = dict(cfg or load_config_file(config_name or "score/factor_select"))
    paths_cfg = load_config_file("paths")
    score_cfg = load_config_file(str(selector_cfg.get("model_score_config", "score/model_score")))
    factor_cfg = load_config_file(str(selector_cfg.get("factor_config", "factor")))

    model_id = str(
        selector_cfg.get("base_model_id")
        or score_cfg.get("model_id")
        or score_cfg.get("default_model_id", "")
    ).strip()
    if not model_id:
        raise ValueError("factor_select missing base_model_id")

    models = dict(score_cfg.get("models", {}))
    if model_id not in models:
        raise KeyError(f"base_model_id not found in model_score config: {model_id}")

    model_entry = dict(models[model_id])
    model_type = str(model_entry.get("model_type", "")).strip().lower()
    model_cfg_key = str(model_entry.get("model_config", "")).strip()
    if not model_type or not model_cfg_key:
        raise ValueError(f"invalid model entry for base model: {model_id}")
    if model_type != "lgbm":
        raise ValueError(f"factor_select importance_topn currently supports lgbm only, got: {model_type}")

    base_model_cfg_path = resolve_config_path(model_cfg_key)
    base_model_cfg = load_json_like(base_model_cfg_path)

    start_day = parse_date(start or selector_cfg.get("start") or base_model_cfg.get("start"))
    end_day = parse_date(end or selector_cfg.get("end") or base_model_cfg.get("end"))
    if start_day > end_day:
        raise ValueError("start must be <= end")

    label_root = Path(paths_cfg["label_data_root"])
    factor_time = str(base_model_cfg.get("factor_time", "14:30"))
    label_time = str(base_model_cfg.get("label_time", "14:42"))
    bins = int(selector_cfg.get("bins", base_model_cfg.get("bins", 5)))
    execution_override = dict(selector_cfg.get("execution", {}))

    baseline_factors = _load_factor_list(str(selector_cfg.get("baseline_factors_file", "score/factor_baseline_factors")))
    if not baseline_factors:
        raise ValueError("baseline factor list is empty")

    blacklist = set(_load_factor_list(str(selector_cfg.get("blacklist_file", "score/factor_blacklist"))) )

    candidate_all = expected_factor_columns_from_cfg(factor_cfg)
    baseline_set = set(baseline_factors)
    candidate_extra = [x for x in candidate_all if x not in baseline_set and x not in blacklist]
    max_candidates = int(selector_cfg.get("max_candidates", 0) or 0)
    if max_candidates > 0:
        candidate_extra = candidate_extra[:max_candidates]

    selection_cfg = dict(selector_cfg.get("selection", {}))
    pool_source = str(selection_cfg.get("pool_source", "baseline_plus_candidates")).strip().lower()
    if pool_source == "baseline_only":
        pool_factors = list(baseline_factors)
    elif pool_source == "candidates_only":
        pool_factors = [x for x in candidate_all if x not in blacklist]
    else:
        pool_factors = baseline_factors + candidate_extra
    pool_factors = _dedupe_keep_order(pool_factors)
    if not pool_factors:
        raise ValueError("factor_select pool factors is empty")

    mode = str(selection_cfg.get("mode", "importance_topn")).strip().lower()
    if mode != "importance_topn":
        raise ValueError(f"unsupported factor_select.selection.mode: {mode}")

    importance_type = str(selection_cfg.get("importance_type", "gain")).strip().lower()
    if importance_type not in {"gain", "split"}:
        raise ValueError("selection.importance_type must be one of: gain, split")

    top_n = int(selection_cfg.get("top_n", 10) or 10)
    min_importance = float(selection_cfg.get("min_importance", 0.0))
    plot_top_k = int(selection_cfg.get("plot_top_k", 30) or 30)

    results_root = Path(paths_cfg["results_root"])
    date_label = f"{start_day:%Y-%m-%d}_{end_day:%Y-%m-%d}"
    out_root = results_root / date_label / "Factor_Select" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    # stage 1: train with full pool
    full_dir = out_root / "stage1_full_pool"
    full_trial = _run_model_trial(
        trial_name="full_pool",
        trial_dir=full_dir,
        factors=pool_factors,
        model_id=model_id,
        model_type=model_type,
        base_model_cfg=base_model_cfg,
        score_cfg=score_cfg,
        execution_override=execution_override,
        start_day=start_day,
        end_day=end_day,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        bins=bins,
        save_state=True,
    )

    importance_df = _load_importance_from_state_dir(
        state_dir=full_trial.state_dir,
        importance_type=importance_type,
    )
    if importance_df.empty:
        raise RuntimeError(
            "failed to extract feature importance from state checkpoints; "
            "ensure rolling refit happened and incremental.save_state is enabled"
        )

    importance_df.to_csv(full_dir / "feature_importance.csv", index=False)
    _plot_importance_topk(
        importance_df=importance_df,
        out_path=full_dir / "feature_importance_topk.png",
        top_k=plot_top_k,
    )

    selected_df = importance_df[pd.to_numeric(importance_df["importance_mean"], errors="coerce") >= float(min_importance)].copy()
    if selected_df.empty:
        raise RuntimeError("no factor survives min_importance threshold")
    if top_n > 0:
        selected_df = selected_df.head(top_n)
    selected_factors = selected_df["factor"].astype(str).tolist()
    selected_factors = _dedupe_keep_order(selected_factors)
    if not selected_factors:
        raise RuntimeError("selected topN factors is empty")

    selected_payload = {
        "selection_mode": mode,
        "importance_type": importance_type,
        "min_importance": min_importance,
        "top_n": top_n,
        "selected_factor_count": len(selected_factors),
        "selected_factors": selected_factors,
    }
    (out_root / "selected_factors_topn.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # stage 2: retrain with topN factors
    topn_dir = out_root / "stage2_topn_retrain"
    topn_trial = _run_model_trial(
        trial_name="topn_retrain",
        trial_dir=topn_dir,
        factors=selected_factors,
        model_id=model_id,
        model_type=model_type,
        base_model_cfg=base_model_cfg,
        score_cfg=score_cfg,
        execution_override=execution_override,
        start_day=start_day,
        end_day=end_day,
        label_root=label_root,
        factor_time=factor_time,
        label_time=label_time,
        bins=bins,
        save_state=False,
    )

    compare_payload = {
        "model_id": model_id,
        "start": str(start_day),
        "end": str(end_day),
        "pool_factor_count": len(pool_factors),
        "selected_factor_count": len(selected_factors),
        "full_pool_summary": full_trial.summary,
        "topn_summary": topn_trial.summary,
        "delta_topn_minus_full": _numeric_delta_map(full_trial.summary, topn_trial.summary),
    }
    (out_root / "compare_summary.json").write_text(
        json.dumps(compare_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    snapshot_payload = {
        "selector_config": selector_cfg,
        "model_score_config_key": str(selector_cfg.get("model_score_config", "score/model_score")),
        "base_model_config_path": str(base_model_cfg_path.as_posix()),
        "baseline_factors_file": str(selector_cfg.get("baseline_factors_file")),
        "blacklist_file": str(selector_cfg.get("blacklist_file")),
        "baseline_factors": baseline_factors,
        "candidate_extra": candidate_extra,
        "pool_factors": pool_factors,
        "blacklist": sorted(list(blacklist)),
    }
    (out_root / "run_config_snapshot.json").write_text(
        json.dumps(snapshot_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "out_root": str(out_root.as_posix()),
        "model_id": model_id,
        "pool_factors": len(pool_factors),
        "selected_factors": len(selected_factors),
        "stage1_rank_ic_mean": full_trial.summary.get("rank_ic_mean"),
        "stage2_rank_ic_mean": topn_trial.summary.get("rank_ic_mean"),
    }
