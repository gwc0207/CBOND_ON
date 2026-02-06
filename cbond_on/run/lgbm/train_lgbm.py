
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cbond_on.core.config import load_config_file, parse_date
from cbond_on.factors.storage import FactorStore
from cbond_on.models.impl.lgbm.trainer import (
    build_dataset,
    evaluate_metrics,
    train_lgbm,
    _iter_existing_label_days,
    _split_days,
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


def main() -> None:
    paths_cfg = load_config_file("paths")
    cfg = load_config_file("models/lgbm/model")

    start = parse_date(cfg.get("start"))
    end = parse_date(cfg.get("end"))

    factor_root = Path(paths_cfg["factor_data_root"])
    label_root = Path(paths_cfg["label_data_root"])

    panel_name = cfg.get("panel_name")
    window_minutes = int(cfg.get("window_minutes", 15))
    factor_time = str(cfg.get("factor_time", "14:30"))
    label_time = str(cfg.get("label_time", "14:45"))

    days = list(_iter_existing_label_days(label_root, start, end))
    if not days:
        raise RuntimeError("no label days found for range")

    train_cfg = cfg.get("train", {})
    train_ratio = float(train_cfg.get("train_ratio", 0.7))
    val_ratio = float(train_cfg.get("val_ratio", 0.15))
    test_ratio = float(train_cfg.get("test_ratio", 0.15))
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_days, val_days, test_days = _split_days(days, train_ratio, val_ratio)
    print(f"train days: {len(train_days)}, val days: {len(val_days)}, test days: {len(test_days)}")

    store = FactorStore(factor_root, panel_name=panel_name, window_minutes=window_minutes)
    # pick factor columns from first available day
    sample = pd.DataFrame()
    for day in train_days:
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

    train_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=train_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )
    val_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=val_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )
    test_data = build_dataset(
        factor_store=store,
        label_root=label_root,
        days=test_days,
        factor_cols=factor_cols,
        min_count=min_count,
        winsor_lower=winsor_lower,
        winsor_upper=winsor_upper,
        zscore=zscore,
        factor_time=factor_time,
        label_time=label_time,
    )

    lgbm_params = cfg.get("lgbm_params", {})
    grid_cfg = cfg.get("grid_search", {})
    early_rounds = cfg.get("early_stopping_rounds")

    # results output dir
    results_root = Path(paths_cfg["results_root"])
    model_name = cfg.get("model_name", "lgbm_factor")
    date_label = f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_root / "models" / model_name / date_label / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    def _best_val_rank_ic(hist: list[dict]) -> float:
        if not hist:
            return float("nan")
        vals = [h.get("val_rank_ic") for h in hist if h.get("val_rank_ic") is not None]
        if not vals:
            return float("nan")
        return float(np.nanmax(vals))

    def _best_iter_from_hist(hist: list[dict]) -> int | None:
        best_val = float("-inf")
        best_it = None
        for h in hist:
            v = h.get("val_rank_ic")
            it = h.get("iteration")
            if v is None or it is None or np.isnan(v):
                continue
            if float(v) > best_val:
                best_val = float(v)
                best_it = int(it)
        return best_it

    model = None
    params = {}
    history = []

    grid_enabled = bool(grid_cfg.get("enable", False))
    if grid_enabled:
        alpha_list = grid_cfg.get("reg_alpha", [])
        lambda_list = grid_cfg.get("reg_lambda", [])
        leaves_list = grid_cfg.get("num_leaves", [])
        depth_list = grid_cfg.get("max_depth", [])
        if not alpha_list:
            alpha_list = [lgbm_params.get("reg_alpha", 0.0)]
        if not lambda_list:
            lambda_list = [lgbm_params.get("reg_lambda", 0.0)]
        if not leaves_list:
            leaves_list = [lgbm_params.get("num_leaves", 31)]
        if not depth_list:
            depth_list = [lgbm_params.get("max_depth", -1)]
        grid_rows = []
        best_score = float("-inf")
        best_params = None
        best_model = None
        best_history = None
        best_iter = None
        for alpha in alpha_list:
            for lam in lambda_list:
                for leaves in leaves_list:
                    for depth in depth_list:
                        trial_params = dict(lgbm_params)
                        trial_params["reg_alpha"] = float(alpha)
                        trial_params["reg_lambda"] = float(lam)
                        trial_params["num_leaves"] = int(leaves)
                        trial_params["max_depth"] = int(depth)
                        trial_model, trial_meta = train_lgbm(
                            train=train_data,
                            val=val_data,
                            lgbm_params=trial_params,
                            early_stopping_rounds=int(early_rounds) if early_rounds else None,
                        )
                        trial_hist = trial_meta.get("history", [])
                        trial_best = _best_val_rank_ic(trial_hist)
                        trial_best_iter = _best_iter_from_hist(trial_hist)
                        grid_rows.append(
                            {
                                "reg_alpha": float(alpha),
                                "reg_lambda": float(lam),
                                "num_leaves": int(leaves),
                                "max_depth": int(depth),
                                "best_val_rank_ic": trial_best,
                                "best_iteration": trial_best_iter,
                            }
                        )
                        if not np.isnan(trial_best) and trial_best > best_score:
                            best_score = trial_best
                            best_params = trial_params
                            best_model = trial_model
                            best_history = trial_hist
                            best_iter = trial_best_iter
        pd.DataFrame(grid_rows).to_csv(out_dir / "grid_search.csv", index=False)
        if best_params is None:
            raise RuntimeError("grid search failed to produce valid params")
        (out_dir / "best_params.json").write_text(
            json.dumps(
                {
                    "best_val_rank_ic": best_score,
                    "best_iteration": best_iter,
                    "params": best_params,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        model = best_model
        params = best_params | {"history": best_history or []}
        history = best_history or []
    else:
        model, params = train_lgbm(
            train=train_data,
            val=val_data,
            lgbm_params=lgbm_params,
            early_stopping_rounds=int(early_rounds) if early_rounds else None,
        )
        history = params.get("history", [])

    train_pred = model.predict(train_data.x)
    val_pred = model.predict(val_data.x)
    test_pred = model.predict(test_data.x)

    train_metrics = evaluate_metrics(train_data.x, train_data.y, train_data.dt, train_pred, bins=bins)
    val_metrics = evaluate_metrics(val_data.x, val_data.y, val_data.dt, val_pred, bins=bins)
    test_metrics = evaluate_metrics(test_data.x, test_data.y, test_data.dt, test_pred, bins=bins)

    print(
        f"train mse={train_metrics['mse']:.6f} r2={train_metrics['r2']:.4f} dir={train_metrics['dir']:.4f} "
        f"ic={train_metrics['ic_mean']:.4f} ir={train_metrics['ic_ir']:.4f} "
        f"rank_ic={train_metrics['rank_ic_mean']:.4f} rank_ir={train_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"val mse={val_metrics['mse']:.6f} r2={val_metrics['r2']:.4f} dir={val_metrics['dir']:.4f} "
        f"ic={val_metrics['ic_mean']:.4f} ir={val_metrics['ic_ir']:.4f} "
        f"rank_ic={val_metrics['rank_ic_mean']:.4f} rank_ir={val_metrics['rank_ic_ir']:.4f}"
    )
    print(
        f"test mse={test_metrics['mse']:.6f} r2={test_metrics['r2']:.4f} dir={test_metrics['dir']:.4f} "
        f"ic={test_metrics['ic_mean']:.4f} ir={test_metrics['ic_ir']:.4f} "
        f"rank_ic={test_metrics['rank_ic_mean']:.4f} rank_ir={test_metrics['rank_ic_ir']:.4f}"
    )

    print(f"train bins: {_format_bins(train_metrics['bin_dir'])}")
    print(f"val bins: {_format_bins(val_metrics['bin_dir'])}")
    print(f"test bins: {_format_bins(test_metrics['bin_dir'])}")

    # save model (best iteration if available)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter and isinstance(best_iter, int):
        model.booster_.save_model(str(out_dir / "model.txt"), num_iteration=best_iter)
    else:
        model.booster_.save_model(str(out_dir / "model.txt"))
    # save config and metrics
    (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "features.json").write_text(json.dumps(factor_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame([
        {"split": "train", **{k: v for k, v in train_metrics.items() if k != "bin_dir"}},
        {"split": "val", **{k: v for k, v in val_metrics.items() if k != "bin_dir"}},
        {"split": "test", **{k: v for k, v in test_metrics.items() if k != "bin_dir"}},
    ])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    if history:
        pd.DataFrame(history).to_csv(out_dir / "metrics_iter.csv", index=False)

    # save bin dir
    def _bin_df(split, metrics):
        return pd.DataFrame([
            {"split": split, "bin": b, "dir_acc": acc, "count": n} for b, acc, n in metrics["bin_dir"]
        ])
    bin_df = pd.concat([
        _bin_df("train", train_metrics),
        _bin_df("val", val_metrics),
        _bin_df("test", test_metrics),
    ], ignore_index=True)
    bin_df.to_csv(out_dir / "bin_dir.csv", index=False)

    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
